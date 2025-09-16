use burn_core as burn;

use burn::module::{Ignored, Module};
use burn::nn::{
    Embedding, EmbeddingConfig, Initializer, Linear, LinearConfig, RmsNorm, RmsNormConfig,
};
use burn::tensor::{backend::Backend, Int, Tensor};

use crate::activation::swiglu_clamp;
use crate::attention::{
    AttnWindow, StreamingMqaParams, StreamingMultiQueryAttention,
    StreamingMultiQueryAttentionConfig,
};
use crate::cache::{MqaCacheManager, WindowPolicy};
use crate::generate::AutoregressiveModel;

#[derive(Clone, Debug)]
pub struct GptOssConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub kv_heads: usize,
    pub ffn_hidden: usize,
    pub dropout: f64,
    pub swiglu_alpha: f32,
    pub swiglu_limit: f32,
    pub initializer: Initializer,
    pub cache_len: usize,
    pub sink_tokens: usize,
    pub window_policy: WindowPolicy,
    pub max_seq_len: usize,
}

impl Default for GptOssConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            d_model: 1024,
            n_layers: 24,
            n_heads: 16,
            kv_heads: 4,
            ffn_hidden: 4096,
            dropout: 0.0,
            swiglu_alpha: 1.0,
            swiglu_limit: 7.0,
            initializer: Initializer::KaimingUniform {
                gain: 1.0 / num_traits::Float::sqrt(3.0),
                fan_out_only: false,
            },
            cache_len: 8192,
            sink_tokens: 0,
            window_policy: WindowPolicy::EveryOther {
                window: 4096,
                full_on_even: true,
            },
            max_seq_len: 8192,
        }
    }
}

#[derive(Module, Debug)]
pub struct GptOssLayer<B: Backend> {
    attn: StreamingMultiQueryAttention<B>,
    norm_attn: RmsNorm<B>,
    norm_ffn: RmsNorm<B>,
    ffn_up: Linear<B>,
    ffn_down: Linear<B>,
    swiglu_alpha: f32,
    swiglu_limit: f32,
}

impl<B: Backend> GptOssLayer<B> {
    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        cache: &mut crate::attention::StreamingMqaCache<B>,
        params: StreamingMqaParams<B>,
    ) -> Tensor<B, 3> {
        let norm_attn = self.norm_attn.forward(hidden.clone());
        let context = self.attn.forward_streaming(norm_attn, cache, params);
        let hidden = hidden + context;

        let norm_ffn = self.norm_ffn.forward(hidden.clone());
        let up = self.ffn_up.forward(norm_ffn);
        let up = swiglu_clamp(up, self.swiglu_alpha, Some(self.swiglu_limit));
        let down = self.ffn_down.forward(up);
        hidden + down
    }
}

#[derive(Module, Debug)]
pub struct GptOssModel<B: Backend> {
    tok_emb: Embedding<B>,
    layers: alloc::vec::Vec<GptOssLayer<B>>,
    norm_final: RmsNorm<B>,
    lm_head: Linear<B>,
    #[allow(dead_code)]
    rope: burn::nn::RotaryEncoding<B>,
    cfg: Ignored<GptOssConfig>,
}

impl GptOssConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GptOssModel<B> {
        let head_dim = self.d_model / self.n_heads;
        let tok_emb = EmbeddingConfig::new(self.vocab_size, self.d_model).init::<B>(device);

        let mut layers = alloc::vec::Vec::with_capacity(self.n_layers);
        for _ in 0..self.n_layers {
            let attn =
                StreamingMultiQueryAttentionConfig::new(self.d_model, self.n_heads, self.kv_heads)
                    .with_dropout(self.dropout)
                    .with_quiet_softmax(false)
                    .with_initializer(self.initializer.clone())
                    .init::<B>(device);
            let norm_attn = RmsNormConfig::new(self.d_model).init::<B>(device);
            let norm_ffn = RmsNormConfig::new(self.d_model).init::<B>(device);
            let ffn_up = LinearConfig::new(self.d_model, self.ffn_hidden * 2)
                .with_initializer(self.initializer.clone())
                .init(device);
            let ffn_down = LinearConfig::new(self.ffn_hidden, self.d_model)
                .with_initializer(self.initializer.clone())
                .init(device);
            layers.push(GptOssLayer {
                attn,
                norm_attn,
                norm_ffn,
                ffn_up,
                ffn_down,
                swiglu_alpha: self.swiglu_alpha,
                swiglu_limit: self.swiglu_limit,
            });
        }

        let norm_final = RmsNormConfig::new(self.d_model).init::<B>(device);
        let lm_head = LinearConfig::new(self.d_model, self.vocab_size)
            .with_initializer(self.initializer.clone())
            .init(device);
        let rope =
            burn::nn::RotaryEncodingConfig::new(self.max_seq_len, head_dim).init::<B>(device);

        GptOssModel {
            tok_emb,
            layers,
            norm_final,
            lm_head,
            rope,
            cfg: Ignored(self.clone()),
        }
    }
}

impl<B: Backend> AutoregressiveModel<B> for GptOssModel<B> {
    type Cache = MqaCacheManager<B>;

    fn init_cache(&self, batch: usize, device: &B::Device) -> Self::Cache {
        let head_dim = self.cfg.0.d_model / self.cfg.0.n_heads;
        MqaCacheManager::new(
            device,
            self.cfg.0.n_layers,
            self.cfg.0.kv_heads,
            head_dim,
            self.cfg.0.cache_len,
            self.cfg.0.sink_tokens,
            batch,
        )
    }

    fn forward_logits(
        &self,
        tokens: Tensor<B, 2, Int>, // [B, T]
        cache: &mut Self::Cache,
        start_pos: usize,
        window: AttnWindow,
    ) -> Tensor<B, 2> {
        let [b, t] = tokens.dims();
        let mut hidden = self.tok_emb.forward(tokens);
        // Forward through layers with streaming MQA
        for (l, layer) in self.layers.iter().enumerate() {
            let policy_win = self.cfg.0.window_policy.window_for(l);
            let layer_window = crate::cache::combine_windows(window, policy_win);
            let state = crate::generate::runner::StreamState {
                start_pos,
                window: layer_window,
            };
            let params = StreamingMqaParams {
                rope: Some(&self.rope),
                start_pos: state.start_pos,
                window: state.window,
                sinks: None,
                attn_bias: None,
            };
            let cache_l = cache.cache_mut(l);
            hidden = layer.forward(hidden, cache_l, params);
        }
        let hidden = self.norm_final.forward(hidden);
        let logits = self.lm_head.forward(hidden);
        // Return last-position logits: [B, vocab]
        let last = logits
            .clone()
            .slice([0..b, t - 1..t, 0..self.cfg.0.vocab_size]);
        last.squeeze::<2>(1)
    }
}
