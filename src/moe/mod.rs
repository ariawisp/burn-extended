use burn_core as burn;

use burn::module::{Ignored, Module};
use burn::nn::{Initializer, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::tensor::{backend::Backend, Tensor};
use alloc::sync::Arc;

// Streaming context for MoE: holds file mapping and per-expert offsets.
#[derive(Debug, Clone)]
pub struct MoeExpertOffsets {
    pub mlp1_blocks_off: u64,
    pub mlp1_scales_off: u64,
    pub mlp2_blocks_off: u64,
    pub mlp2_scales_off: u64,
}

#[derive(Debug)]
pub struct MoeStreamingContext {
    pub mmap: Arc<memmap2::Mmap>, // mapped model.bin (read-only)
    pub experts: alloc::vec::Vec<MoeExpertOffsets>,
    pub rows_mlp1: usize,
    pub cols_mlp1: usize,
    pub rows_mlp2: usize,
    pub cols_mlp2: usize,
    pub ue8_offset: i32,
}


#[derive(Debug, Clone)]
pub struct MoeConfig {
    pub d_model: usize,
    pub ffn_hidden: usize,
    pub num_experts: usize,
    pub experts_per_token: usize,
    pub swiglu_alpha: f32,
    pub swiglu_limit: f32,
    pub initializer: Initializer,
    pub disabled: bool,
}

#[derive(Module, Debug)]
pub struct MoeGatedSwiGLU<B: Backend> {
    pub norm: RmsNorm<B>,
    pub gate: Linear<B>, // [d_model -> num_experts]
    // Expert parameters (quantized MXFP4 blocks/scales + BF16 biases) to match GPT‑OSS layout and reader mapping.
    pub mlp1_blocks: burn::module::Param<Tensor<B, 3>>, // U8: [E, 2*ffn, d_model/2]
    pub mlp1_scales: burn::module::Param<Tensor<B, 2>>, // U8: [E, 2*ffn]
    pub mlp1_bias: burn::module::Param<Tensor<B, 2>>,   // BF16: [E, 2*ffn]
    pub mlp2_blocks: burn::module::Param<Tensor<B, 3>>, // U8: [E, d_model, ffn/2]
    pub mlp2_scales: burn::module::Param<Tensor<B, 2>>, // U8: [E, d_model]
    pub mlp2_bias: burn::module::Param<Tensor<B, 2>>,   // BF16: [E, d_model]

    pub num_experts: usize,
    pub experts_per_token: usize,
    pub swiglu_alpha: f32,
    pub swiglu_limit: f32,
    pub d_model: usize,
    pub ffn_hidden: usize,
    pub disabled: bool,
    // Optional streaming context (preferred runtime path when present)
    pub streaming: Ignored<Option<Arc<MoeStreamingContext>>>,
}

impl MoeConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MoeGatedSwiGLU<B> {
        let norm = RmsNormConfig::new(self.d_model).init::<B>(device);
        let gate = LinearConfig::new(self.d_model, self.num_experts)
            .with_initializer(self.initializer.clone())
            .init(device);
        // Placeholder tensors; loader will overwrite with correct dtype and data.
        let mlp1_blocks = Tensor::<B, 3>::zeros([self.num_experts, self.ffn_hidden * 2, (self.d_model + 1) / 2], device);
        let mlp1_scales = Tensor::<B, 2>::zeros([self.num_experts, self.ffn_hidden * 2], device);
        let mlp1_b = Tensor::<B, 2>::zeros([self.num_experts, self.ffn_hidden * 2], device);
        let mlp2_blocks = Tensor::<B, 3>::zeros([self.num_experts, self.d_model, (self.ffn_hidden + 1) / 2], device);
        let mlp2_scales = Tensor::<B, 2>::zeros([self.num_experts, self.d_model], device);
        let mlp2_b = Tensor::<B, 2>::zeros([self.num_experts, self.d_model], device);
        MoeGatedSwiGLU {
            norm,
            gate,
            mlp1_blocks: burn::module::Param::from_tensor(mlp1_blocks),
            mlp1_scales: burn::module::Param::from_tensor(mlp1_scales),
            mlp1_bias: burn::module::Param::from_tensor(mlp1_b),
            mlp2_blocks: burn::module::Param::from_tensor(mlp2_blocks),
            mlp2_scales: burn::module::Param::from_tensor(mlp2_scales),
            mlp2_bias: burn::module::Param::from_tensor(mlp2_b),
            num_experts: self.num_experts,
            experts_per_token: self.experts_per_token,
            swiglu_alpha: self.swiglu_alpha,
            swiglu_limit: self.swiglu_limit,
            d_model: self.d_model,
            ffn_hidden: self.ffn_hidden,
            disabled: self.disabled,
            streaming: Ignored(None),
        }
    }
}

impl<B: Backend> MoeGatedSwiGLU<B> {
    pub fn set_streaming(&mut self, ctx: Arc<MoeStreamingContext>) {
        self.streaming = Ignored(Some(ctx));
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        if self.disabled {
            // Skip MoE block entirely, act as identity residual.
            return x;
        }
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

        // If mmap streaming is available, use it instead of resident quantized params.
        if let Some(ctx) = self.streaming.0.as_ref() {
            return self.forward_streaming(x, ctx);
        }

        // Dequant helpers (resident-quantized fallback)
        const UE8_OFFSET: i32 = 14;
        const FP4_VALUES: [f32; 16] = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        let dequant_rows = |blocks: &[u8], scales: &[u8], rows: usize, cols: usize| -> Vec<f32> {
            let bpr = (cols + 1) / 2;
            let mut out = vec![0f32; rows * cols];
            for r in 0..rows {
                let si = (scales[r] as i32) - UE8_OFFSET - 127;
                let scale = (2.0f32).powi(si);
                for i in 0..bpr {
                    let byte = blocks[r * bpr + i];
                    let lo = (byte & 0x0F) as usize;
                    let hi = (byte >> 4) as usize;
                    let c = i * 2;
                    if c < cols { out[r * cols + c] = FP4_VALUES[lo] * scale; }
                    if c + 1 < cols { out[r * cols + c + 1] = FP4_VALUES[hi] * scale; }
                }
            }
            out
        };

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

            // Load expert biases
            let b1 = self
                .mlp1_bias
                .val()
                .clone()
                .slice([expert..expert + 1, 0..self.ffn_hidden * 2])
                .reshape([self.ffn_hidden * 2]);
            let b2 = self
                .mlp2_bias
                .val()
                .clone()
                .slice([expert..expert + 1, 0..self.d_model])
                .reshape([self.d_model]);

            // Dequant expert weights for mlp1 and mlp2
            let w1_blocks = self
                .mlp1_blocks
                .val()
                .clone()
                .slice([expert..expert + 1, 0..self.ffn_hidden * 2, 0..(self.d_model + 1) / 2])
                .reshape([self.ffn_hidden * 2, (self.d_model + 1) / 2])
                .into_data()
                .convert::<u8>()
                .into_vec::<u8>()
                .expect("mlp1 blocks u8 vec");
            let w1_scales = self
                .mlp1_scales
                .val()
                .clone()
                .slice([expert..expert + 1, 0..self.ffn_hidden * 2])
                .reshape([self.ffn_hidden * 2])
                .into_data()
                .convert::<u8>()
                .into_vec::<u8>()
                .expect("mlp1 scales u8 vec");
            let w1_deq = dequant_rows(&w1_blocks, &w1_scales, self.ffn_hidden * 2, self.d_model);
            let w1 = Tensor::<B, 2>::from_floats(burn_tensor::TensorData::new(w1_deq, [self.ffn_hidden * 2, self.d_model]), &norm_flat.device())
                ;

            // MLP1: [n,d] x [d,2f] + b1 -> [n,2f]
            let up = norm_flat.clone().matmul(w1.transpose()) + b1.unsqueeze();
            let act = crate::activation::swiglu_clamp(up, self.swiglu_alpha, Some(self.swiglu_limit));
            // Act now [n,f]
            let act_f = act.reshape([n, self.ffn_hidden]);

            let w2_blocks = self
                .mlp2_blocks
                .val()
                .clone()
                .slice([expert..expert + 1, 0..self.d_model, 0..(self.ffn_hidden + 1) / 2])
                .reshape([self.d_model, (self.ffn_hidden + 1) / 2])
                .into_data()
                .convert::<u8>()
                .into_vec::<u8>()
                .expect("mlp2 blocks u8 vec");
            let w2_scales = self
                .mlp2_scales
                .val()
                .clone()
                .slice([expert..expert + 1, 0..self.d_model])
                .reshape([self.d_model])
                .into_data()
                .convert::<u8>()
                .into_vec::<u8>()
                .expect("mlp2 scales u8 vec");
            let w2_deq = dequant_rows(&w2_blocks, &w2_scales, self.d_model, self.ffn_hidden);
            let w2 = Tensor::<B, 2>::from_floats(burn_tensor::TensorData::new(w2_deq, [self.d_model, self.ffn_hidden]), &norm_flat.device());

            // MLP2: [n,f] x [f,d] + b2 -> [n,d]
            let out = act_f.matmul(w2.transpose()) + b2.unsqueeze();
            let out = out * w_ex; // [n,d]
            acc = acc + out;
        }

        let y = acc.reshape([b, t, d]);
        x + y
    }

    fn forward_streaming(&self, x: Tensor<B, 3>, ctx: &MoeStreamingContext) -> Tensor<B, 3> {
        // Shapes
        let [b, t, d] = x.dims();
        let n = b * t;

        // 1) Norm + gating
        let normed = self.norm.forward(x.clone()); // [B,T,d]
        let gates = self.gate.forward(normed.clone()); // [B,T,E]
        let norm_flat = normed.clone().reshape([n, d]);

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
            // top-k indices by value (descending)
            let mut idx: Vec<usize> = (0..e).collect();
            idx.sort_unstable_by(|&a, &bidx| row[bidx].partial_cmp(&row[a]).unwrap_or(core::cmp::Ordering::Equal));
            let top = &idx[..k];
            // softmax over selected
            let mut max_v = f32::NEG_INFINITY;
            for &j in top.iter() { max_v = max_v.max(row[j]); }
            let mut sum = 0.0f32;
            let mut tmp = vec![0.0f32; k];
            for (p, &j) in top.iter().enumerate() { let v = (row[j] - max_v).exp(); tmp[p] = v; sum += v; }
            if sum > 0.0 { for (p, &j) in top.iter().enumerate() { weights_per_expert[j][i] = tmp[p] / sum; } }
        }

        let mut acc = Tensor::<B, 2>::zeros([n, d], &norm_flat.device());

        // MXFP4 constants
        const FP4_VALUES: [f32; 16] = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        let ue8 = ctx.ue8_offset;
        let rows1 = ctx.rows_mlp1;
        let cols1 = ctx.cols_mlp1;
        let rows2 = ctx.rows_mlp2;
        let cols2 = ctx.cols_mlp2;
        let bpr1 = (cols1 + 1) / 2;
        let bpr2 = (cols2 + 1) / 2;

        for expert in 0..e {
            // Skip if no weight mass routed to this expert
            let w_host = &weights_per_expert[expert];
            if !w_host.iter().any(|&v| v != 0.0) { continue; }

            // Expert indices
            let off = &ctx.experts[expert];
            let s1 = off.mlp1_scales_off as usize;
            let b1_off = off.mlp1_blocks_off as usize;
            let s2 = off.mlp2_scales_off as usize;
            let b2_off = off.mlp2_blocks_off as usize;

            // Borrow slices directly from mmap
            let blocks1 = &ctx.mmap[b1_off..b1_off + rows1 * bpr1];
            let scales1 = &ctx.mmap[s1..s1 + rows1];
            let blocks2 = &ctx.mmap[b2_off..b2_off + rows2 * bpr2];
            let scales2 = &ctx.mmap[s2..s2 + rows2];

            // Dequant functions
            let dequant_rows = |blocks: &[u8], scales: &[u8], rows: usize, cols: usize| -> Vec<f32> {
                let bpr = (cols + 1) / 2;
                let mut out = vec![0f32; rows * cols];
                for r in 0..rows {
                    let si = (scales[r] as i32) - ue8 - 127;
                    let scale = (2.0f32).powi(si);
                    for i in 0..bpr {
                        let byte = blocks[r * bpr + i];
                        let lo = (byte & 0x0F) as usize;
                        let hi = (byte >> 4) as usize;
                        let c = i * 2;
                        if c < cols { out[r * cols + c] = FP4_VALUES[lo] * scale; }
                        if c + 1 < cols { out[r * cols + c + 1] = FP4_VALUES[hi] * scale; }
                    }
                }
                out
            };

            // Bias tensors (resident BF16 in module)
            let b1 = self
                .mlp1_bias
                .val()
                .clone()
                .slice([expert..expert + 1, 0..self.ffn_hidden * 2])
                .reshape([self.ffn_hidden * 2]);
            let b2 = self
                .mlp2_bias
                .val()
                .clone()
                .slice([expert..expert + 1, 0..self.d_model])
                .reshape([self.d_model]);

            // Dequantize
            let w1_deq = dequant_rows(blocks1, scales1, rows1, cols1);
            let w2_deq = dequant_rows(blocks2, scales2, rows2, cols2);
            let w1 = Tensor::<B, 2>::from_floats(burn_tensor::TensorData::new(w1_deq, [rows1, cols1]), &norm_flat.device());
            let w2 = Tensor::<B, 2>::from_floats(burn_tensor::TensorData::new(w2_deq, [rows2, cols2]), &norm_flat.device());

            // Compute
            let w_ex = Tensor::<B, 1>::from_floats(w_host.as_slice(), &normed.device()).reshape([n, 1]);
            let up = norm_flat.clone().matmul(w1.transpose()) + b1.unsqueeze();
            let act = crate::activation::swiglu_clamp(up, self.swiglu_alpha, Some(self.swiglu_limit));
            let act_f = act.reshape([n, self.ffn_hidden]);
            let out = act_f.matmul(w2.transpose()) + b2.unsqueeze();
            let out = out * w_ex;
            acc = acc + out;
        }

        let y = acc.reshape([b, t, d]);
        x + y
    }
}
