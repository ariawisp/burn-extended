use burn_core as burn;

use burn::module::Module;
use burn::nn::{Initializer, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::tensor::{backend::Backend, Tensor};

use crate::activation::swiglu_clamp;

#[derive(Debug, Clone)]
pub struct MoeConfig {
    pub d_model: usize,
    pub ffn_hidden: usize,
    pub num_experts: usize,
    pub experts_per_token: usize,
    pub swiglu_alpha: f32,
    pub swiglu_limit: f32,
    pub initializer: Initializer,
}

#[derive(Module, Debug)]
pub struct MoeGatedSwiGLU<B: Backend> {
    pub norm: RmsNorm<B>,
    pub gate: Linear<B>, // [d_model -> num_experts]
    // Expert parameters as raw tensors to match GPT‑OSS layout and loader mapping.
    pub mlp1_weight: burn::module::Param<Tensor<B, 3>>, // [E, 2*ffn, d_model]
    pub mlp1_bias: burn::module::Param<Tensor<B, 2>>,   // [E, 2*ffn]
    pub mlp2_weight: burn::module::Param<Tensor<B, 3>>, // [E, d_model, ffn]
    pub mlp2_bias: burn::module::Param<Tensor<B, 2>>,   // [E, d_model]

    pub num_experts: usize,
    pub experts_per_token: usize,
    pub swiglu_alpha: f32,
    pub swiglu_limit: f32,
    pub d_model: usize,
    pub ffn_hidden: usize,
}

impl MoeConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MoeGatedSwiGLU<B> {
        let norm = RmsNormConfig::new(self.d_model).init::<B>(device);
        let gate = LinearConfig::new(self.d_model, self.num_experts)
            .with_initializer(self.initializer.clone())
            .init(device);
        let mlp1_w = Tensor::<B, 3>::zeros([self.num_experts, self.ffn_hidden * 2, self.d_model], device);
        let mlp1_b = Tensor::<B, 2>::zeros([self.num_experts, self.ffn_hidden * 2], device);
        let mlp2_w = Tensor::<B, 3>::zeros([self.num_experts, self.d_model, self.ffn_hidden], device);
        let mlp2_b = Tensor::<B, 2>::zeros([self.num_experts, self.d_model], device);
        MoeGatedSwiGLU {
            norm,
            gate,
            mlp1_weight: burn::module::Param::from_tensor(mlp1_w),
            mlp1_bias: burn::module::Param::from_tensor(mlp1_b),
            mlp2_weight: burn::module::Param::from_tensor(mlp2_w),
            mlp2_bias: burn::module::Param::from_tensor(mlp2_b),
            num_experts: self.num_experts,
            experts_per_token: self.experts_per_token,
            swiglu_alpha: self.swiglu_alpha,
            swiglu_limit: self.swiglu_limit,
            d_model: self.d_model,
            ffn_hidden: self.ffn_hidden,
        }
    }
}

impl<B: Backend> MoeGatedSwiGLU<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Shapes
        let [b, t, d] = x.dims();
        let n = b * t;

        // 1) Norm + gating
        let normed = self.norm.forward(x.clone()); // [B,T,d]
        let gates = self.gate.forward(normed.clone()); // [B,T,E]

        // Flatten for compute
        let norm_flat = normed.clone().reshape([n, d]);

        // Pull gates to CPU to compute top-k per token
        let e = self.num_experts;
        let k = self.experts_per_token.min(e).max(1);
        let gates_host = gates
            .clone()
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("gates to f32");

        // weights_per_expert[e][n]
        let mut weights_per_expert: Vec<Vec<f32>> = vec![vec![0.0f32; n]; e];
        for i in 0..n {
            let row = &gates_host[i * e..(i + 1) * e];
            // top-k indices by value
            let mut idx: Vec<usize> = (0..e).collect();
            idx.sort_unstable_by(|&a, &bidx| {
                row[bidx]
                    .partial_cmp(&row[a])
                    .unwrap_or(core::cmp::Ordering::Equal)
            });
            let top = &idx[..k];
            // softmax over selected
            let mut max_v = f32::NEG_INFINITY;
            for &j in top.iter() {
                if row[j] > max_v {
                    max_v = row[j];
                }
            }
            let mut sum = 0.0f32;
            let mut tmp = vec![0.0f32; k];
            for (p, &j) in top.iter().enumerate() {
                let v = (row[j] - max_v).exp();
                tmp[p] = v;
                sum += v;
            }
            if sum > 0.0 {
                for (p, &j) in top.iter().enumerate() {
                    weights_per_expert[j][i] = tmp[p] / sum;
                }
            }
        }

        // Accumulator for outputs
        let mut acc = Tensor::<B, 2>::zeros([n, d], &norm_flat.device());

        for expert in 0..e {
            // Skip if no weight mass routed to this expert
            let w_host = &weights_per_expert[expert];
            let mut has_nonzero = false;
            for &v in w_host.iter() {
                if v != 0.0 { has_nonzero = true; break; }
            }
            if !has_nonzero { continue; }

            // Convert weights to device tensor [n,1]
            let w_ex = Tensor::<B, 1>::from_floats(w_host.as_slice(), &normed.device())
                .reshape([n, 1]);

            // Load expert weights
            let w1 = self
                .mlp1_weight
                .val()
                .clone()
                .slice([expert..expert + 1, 0..self.ffn_hidden * 2, 0..self.d_model])
                .reshape([self.ffn_hidden * 2, self.d_model]);
            let b1 = self
                .mlp1_bias
                .val()
                .clone()
                .slice([expert..expert + 1, 0..self.ffn_hidden * 2])
                .reshape([self.ffn_hidden * 2]);
            let w2 = self
                .mlp2_weight
                .val()
                .clone()
                .slice([expert..expert + 1, 0..self.d_model, 0..self.ffn_hidden])
                .reshape([self.d_model, self.ffn_hidden]);
            let b2 = self
                .mlp2_bias
                .val()
                .clone()
                .slice([expert..expert + 1, 0..self.d_model])
                .reshape([self.d_model]);

            // MLP1: [n,d] x [d,2f] + b1 -> [n,2f]
            let up = norm_flat.clone().matmul(w1.transpose()) + b1.unsqueeze();
            let act = crate::activation::swiglu_clamp(up, self.swiglu_alpha, Some(self.swiglu_limit));
            // Act now [n,f]
            let act_f = act.reshape([n, self.ffn_hidden]);
            // MLP2: [n,f] x [f,d] + b2 -> [n,d]
            let out = act_f.matmul(w2.transpose()) + b2.unsqueeze();
            let out = out * w_ex; // [n,d]
            acc = acc + out;
        }

        let y = acc.reshape([b, t, d]);
        x + y
    }
}
