use burn_core as burn;

use burn::module::{Content, DisplaySettings, Module, ModuleDisplay};
use burn::nn::{Dropout, DropoutConfig, Initializer, Linear, LinearConfig};
use burn::{
    config::Config,
    tensor::{Tensor, backend::Backend},
};

use super::{AttnWindow, StreamingMhaCache};
use super::streaming::update_cache_window;

/// Configuration for the extended streaming multi-head attention module.
#[derive(Config, Debug)]
pub struct ExtStreamingMultiHeadAttentionConfig {
    pub d_model: usize,
    pub n_heads: usize,
    #[config(default = 0.0)]
    pub dropout: f64,
    #[config(default = false)]
    pub quiet_softmax: bool,
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

/// Extended streaming multi-head attention with optional additive attn_bias.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct ExtStreamingMultiHeadAttention<B: Backend> {
    pub query: Linear<B>,
    pub key: Linear<B>,
    pub value: Linear<B>,
    pub output: Linear<B>,
    pub dropout: Dropout,
    pub d_model: usize,
    pub n_heads: usize,
    pub d_k: usize,
    pub quiet_softmax: bool,
}

impl<B: Backend> ModuleDisplay for ExtStreamingMultiHeadAttention<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("d_model", &self.d_model)
            .add("n_heads", &self.n_heads)
            .add("d_k", &self.d_k)
            .add("dropout", &self.dropout.prob)
            .add("quiet_softmax", &self.quiet_softmax)
            .optional()
    }
}

impl ExtStreamingMultiHeadAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ExtStreamingMultiHeadAttention<B> {
        let linear = |in_features, out_features| {
            LinearConfig::new(in_features, out_features)
                .with_initializer(self.initializer.clone())
                .init(device)
        };

        assert!(
            self.d_model % self.n_heads == 0,
            "d_model must be divisible by n_heads"
        );
        let d_k = self.d_model / self.n_heads;

        ExtStreamingMultiHeadAttention {
            query: linear(self.d_model, self.d_model),
            key: linear(self.d_model, self.d_model),
            value: linear(self.d_model, self.d_model),
            output: linear(self.d_model, self.d_model),
            dropout: DropoutConfig::new(self.dropout).init(),
            d_model: self.d_model,
            n_heads: self.n_heads,
            d_k,
            quiet_softmax: self.quiet_softmax,
        }
    }
}

/// Parameters for extended streaming attention forward.
pub struct ExtStreamingParams<'a, B: Backend> {
    pub rope: Option<&'a burn::nn::rope_encoding::RotaryEncoding<B>>,
    pub start_pos: usize,
    pub window: AttnWindow,
    /// Optional additive bias on attention logits `[B, n_heads, q_len, k_len]` matching window.
    pub attn_bias: Option<&'a Tensor<B, 4>>,
}

impl<B: Backend> ExtStreamingMultiHeadAttention<B> {
    pub fn forward_streaming(
        &self,
        x: Tensor<B, 3>,
        cache: &mut StreamingMhaCache<B>,
        params: ExtStreamingParams<B>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = x.dims();

        let q = self.attention_linear(x.clone(), &self.query);
        let k = self.attention_linear(x.clone(), &self.key);
        let v = self.attention_linear(x, &self.value);

        let (q, k) = if let Some(rope) = params.rope {
            let q_rs = q.swap_dims(1, 2);
            let k_rs = k.swap_dims(1, 2);
            let q_ro = rope.apply(q_rs, params.start_pos);
            let k_ro = rope.apply(k_rs, params.start_pos);
            (q_ro.swap_dims(1, 2), k_ro.swap_dims(1, 2))
        } else {
            (q, k)
        };

        let cache_view = update_cache_window(cache, k, v, params.start_pos, params.window);
        let k_win = cache_view.k;
        let v_win = cache_view.v;

        let attn_scores = crate::attention::compute_scores(q, k_win, self.d_k, &self.dropout);
        let attn_bias = params.attn_bias.cloned();
        let weights = crate::attention::apply_bias_and_softmax(attn_scores, attn_bias, self.quiet_softmax);

        let context = weights.matmul(v_win)
            .swap_dims(1, 2)
            .reshape([batch_size, seq_len, self.d_model]);
        self.output.forward(context)
    }

    fn attention_linear(&self, x: Tensor<B, 3>, linear: &Linear<B>) -> Tensor<B, 4> {
        let [batch_size, seq_length, _d_model] = x.dims();
        linear
            .forward(x)
            .reshape([batch_size, seq_length, self.n_heads, self.d_k])
            .swap_dims(1, 2)
    }
}
