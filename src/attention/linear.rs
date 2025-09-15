use burn_core as burn;
use burn::serde::{Deserialize, Serialize};

use burn::module::{Content, DisplaySettings, Ignored, Module, ModuleDisplay};
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::{
    config::Config,
    tensor::{Bool, Tensor, activation, backend::Backend},
};

#[derive(Config, Debug)]
pub struct LinearAttentionConfig {
    pub d_model: usize,
    pub n_heads: usize,
    #[config(default = "KernelType::Relu")]
    pub kernel: KernelType,
    #[config(default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}")]
    pub initializer: Initializer,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum KernelType { Relu }

#[derive(Debug, Clone)]
pub struct LinearAttnInput<B: Backend> {
    pub query: Tensor<B, 3>,
    pub key: Tensor<B, 3>,
    pub value: Tensor<B, 3>,
    pub mask_pad: Option<Tensor<B, 2, Bool>>,
}

impl<B: Backend> LinearAttnInput<B> {
    pub fn self_attn(t: Tensor<B, 3>) -> Self { Self { query: t.clone(), key: t.clone(), value: t, mask_pad: None } }
    pub fn mask_pad(mut self, mask: Tensor<B, 2, Bool>) -> Self { self.mask_pad = Some(mask); self }
}

#[derive(Module, Debug)]
#[module(custom_display)]
pub struct LinearAttention<B: Backend> {
    pub query: Linear<B>,
    pub key: Linear<B>,
    pub value: Linear<B>,
    pub output: Linear<B>,
    pub d_model: usize,
    pub n_heads: usize,
    pub d_k: usize,
    pub kernel: Ignored<KernelType>,
}

impl<B: Backend> ModuleDisplay for LinearAttention<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> { DisplaySettings::new().with_new_line_after_attribute(false).optional() }
    fn custom_content(&self, content: Content) -> Option<Content> { content.add("d_model", &self.d_model).add("n_heads", &self.n_heads).add("d_k", &self.d_k).optional() }
}

impl LinearAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LinearAttention<B> {
        let linear = |in_features, out_features| {
            LinearConfig::new(in_features, out_features).with_initializer(self.initializer.clone()).init(device)
        };
        assert!(self.d_model % self.n_heads == 0);
        let d_k = self.d_model / self.n_heads;
        LinearAttention { query: linear(self.d_model, self.d_model), key: linear(self.d_model, self.d_model), value: linear(self.d_model, self.d_model), output: linear(self.d_model, self.d_model), d_model: self.d_model, n_heads: self.n_heads, d_k, kernel: Ignored(self.kernel) }
    }
}

#[derive(Debug, Clone)]
pub struct LinearAttnOutput<B: Backend> { pub context: Tensor<B, 3> }

impl<B: Backend> LinearAttention<B> {
    pub fn forward(&self, input: LinearAttnInput<B>) -> LinearAttnOutput<B> {
        let [batch_size, seq_q, _] = input.query.dims();
        let q = self.attn_linear(input.query, &self.query);
        let mut k = self.attn_linear(input.key, &self.key);
        let mut v = self.attn_linear(input.value, &self.value);
        if let Some(mask) = input.mask_pad.clone() {
            let mask = mask.reshape([batch_size, 1, mask.dims()[1], 1]);
            k = k.mask_fill(mask.clone(), 0.0);
            v = v.mask_fill(mask, 0.0);
        }
        let q_phi = match self.kernel.0 { KernelType::Relu => activation::relu(q) };
        let k_phi = match self.kernel.0 { KernelType::Relu => activation::relu(k) };
        let kv = k_phi.clone().swap_dims(2, 3).matmul(v);
        let k_sum = k_phi.sum_dim(2).swap_dims(2, 3);
        let denom = q_phi.clone().matmul(k_sum).add_scalar(1e-6);
        let context = q_phi.matmul(kv) / denom;
        let context = context.swap_dims(1, 2).reshape([batch_size, seq_q, self.d_model]);
        let context = self.output.forward(context);
        LinearAttnOutput { context }
    }
    fn attn_linear(&self, x: Tensor<B, 3>, linear: &Linear<B>) -> Tensor<B, 4> {
        let [b, s, _] = x.dims();
        linear.forward(x).reshape([b, s, self.n_heads, self.d_k]).swap_dims(1, 2)
    }
}

