use burn_core as burn;

use burn::module::{Ignored, Module};
use burn::nn::{
    Embedding, EmbeddingConfig, Initializer, Linear, LinearConfig, RmsNorm, RmsNormConfig,
};
use burn::tensor::{backend::Backend, Int, Tensor};

use crate::attention::{
    AttnWindow, StreamingMqaParams, StreamingMultiQueryAttention,
    StreamingMultiQueryAttentionConfig,
};
use crate::cache::{MqaCacheManager, WindowPolicy};
use crate::generate::AutoregressiveModel;
use crate::moe::{MoeConfig, MoeGatedSwiGLU};

#[derive(Clone, Debug)]
pub struct GptOssConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub kv_heads: usize,
    pub ffn_hidden: usize,
    pub num_experts: usize,
    pub experts_per_token: usize,
    pub dropout: f64,
    pub swiglu_alpha: f32,
    pub swiglu_limit: f32,
    pub initializer: Initializer,
    pub cache_len: usize,
    pub sink_tokens: usize,
    pub window_policy: WindowPolicy,
    pub max_seq_len: usize,
    pub learned_sinks: bool,
    // RoPE options
    pub use_ntk_yarn: bool,
    pub rope_scaling_factor: f32,
    pub rope_initial_context: f32,
    pub rope_ntk_alpha: f32,
    pub rope_ntk_beta: f32,
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
            num_experts: 64,
            experts_per_token: 4,
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
            learned_sinks: true,
            use_ntk_yarn: false,
            rope_scaling_factor: 32.0,
            rope_initial_context: 4096.0,
            rope_ntk_alpha: 1.0,
            rope_ntk_beta: 32.0,
        }
    }
}

#[derive(Module, Debug)]
pub struct GptOssLayer<B: Backend> {
    attn: StreamingMultiQueryAttention<B>,
    norm_attn: RmsNorm<B>,
    mlp: MoeGatedSwiGLU<B>,
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

        // MoE block handles its own norm and residual add internally.
        self.mlp.forward(hidden)
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
                    .with_learned_sinks(self.learned_sinks)
                    .with_initializer(self.initializer.clone())
                    .init::<B>(device);
            let norm_attn = RmsNormConfig::new(self.d_model).init::<B>(device);
            let mlp = MoeConfig {
                d_model: self.d_model,
                ffn_hidden: self.ffn_hidden,
                num_experts: self.num_experts,
                experts_per_token: self.experts_per_token,
                swiglu_alpha: self.swiglu_alpha,
                swiglu_limit: self.swiglu_limit,
                initializer: self.initializer.clone(),
            }
            .init::<B>(device);
            layers.push(GptOssLayer {
                attn,
                norm_attn,
                mlp,
            });
        }

        let norm_final = RmsNormConfig::new(self.d_model).init::<B>(device);
        let lm_head = LinearConfig::new(self.d_model, self.vocab_size)
            .with_initializer(self.initializer.clone())
            .init(device);
        let rope = if self.use_ntk_yarn {
            crate::rope::init_ntk_yarn::<B>(
                self.max_seq_len,
                head_dim,
                device,
                self.rope_scaling_factor,
                self.rope_initial_context,
                self.rope_ntk_alpha,
                self.rope_ntk_beta,
            )
        } else {
            burn::nn::RotaryEncodingConfig::new(self.max_seq_len, head_dim).init::<B>(device)
        };

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
