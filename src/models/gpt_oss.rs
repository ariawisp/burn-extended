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
use alloc::sync::Arc;

#[derive(Clone, Debug)]
pub struct GptOssConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub kv_heads: usize,
    pub ffn_hidden: usize,
    pub num_experts: usize,
    pub experts_per_token: usize,
    pub disable_moe: bool,
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
    // Verbose logging for debugging
    pub verbose: bool,
}

impl Default for GptOssConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            d_model: 1024,
            n_layers: 24,
            n_heads: 16,
            head_dim: 64,
            kv_heads: 4,
            ffn_hidden: 4096,
            num_experts: 64,
            experts_per_token: 4,
        disable_moe: false,
        dropout: 0.0,
        swiglu_alpha: 1.702,
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
            verbose: false,
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
        let rope_dim = if self.head_dim % 2 == 0 { self.head_dim } else { self.head_dim.saturating_sub(1) };
        let tok_emb = EmbeddingConfig::new(self.vocab_size, self.d_model).init::<B>(device);

        let mut layers = alloc::vec::Vec::with_capacity(self.n_layers);
        for _ in 0..self.n_layers {
            let attn =
                StreamingMultiQueryAttentionConfig::new(self.d_model, self.n_heads, self.kv_heads, self.head_dim)
                    .with_dropout(self.dropout)
                    .with_quiet_softmax(false)
                    .with_learned_sinks(self.learned_sinks)
                    .with_pre_scaled_qk(true)
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
                disabled: self.disable_moe,
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
            .with_bias(false)
            .with_initializer(self.initializer.clone())
            .init(device);
        let rope = if self.use_ntk_yarn {
            crate::rope::init_ntk_yarn::<B>(
                self.max_seq_len,
                rope_dim,
                device,
                self.rope_scaling_factor,
                self.rope_initial_context,
                self.rope_ntk_alpha,
                self.rope_ntk_beta,
            )
        } else {
            burn::nn::RotaryEncodingConfig::new(self.max_seq_len, rope_dim).init::<B>(device)
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
        MqaCacheManager::new(
            device,
            self.cfg.0.n_layers,
            self.cfg.0.kv_heads,
            self.cfg.0.head_dim,
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
        use std::time::Instant;
        let [b, t] = tokens.dims();
        // Process full prefill in one tile to reduce kernel launch/setup overhead.
        let tile_q = t;
        let decode_only_last = cache.caches.first().map(|c| c.len() > 0).unwrap_or(false);

        if decode_only_last {
            if self.cfg.0.verbose { eprintln!("forward_logits: decode path b={} t={} start_pos={}", b, t, start_pos); }
            // Decode step: process only the last token through all layers
            let toks_last = tokens.clone().slice([0..b, t - 1..t]); // [B,1]
            let mut hidden = self.tok_emb.forward(toks_last); // [B,1,d]
            for (l, layer) in self.layers.iter().enumerate() {
                let layer_t0 = Instant::now();
                let policy_win = self.cfg.0.window_policy.window_for(l);
                let layer_window = crate::cache::combine_windows(window, policy_win);
                let cache_l = cache.cache_mut(l);
                let params = StreamingMqaParams {
                    rope: Some(&self.rope),
                    start_pos: start_pos + (t - 1),
                    window: layer_window,
                    sinks: None,
                    attn_bias: None,
                };
                if self.cfg.0.verbose { eprintln!("layer {}: attn start (decode)", l); }
                hidden = layer.forward(hidden, cache_l, params); // [B,1,d]
                if self.cfg.0.verbose { eprintln!("layer {}: attn+moe done in {:.2?}", l, layer_t0.elapsed()); }
            }
            let hidden = self.norm_final.forward(hidden); // [B,1,d]
            let logits = self.lm_head.forward(hidden); // [B,1,V]
            logits.squeeze::<2>(1)
        } else {
            if self.cfg.0.verbose { eprintln!("forward_logits: prefill path b={} t={} layers={}", b, t, self.cfg.0.n_layers); }
            // Prefill: process the full sequence in small tiles per layer
            let mut hidden_full = self.tok_emb.forward(tokens); // [B, T, d]
            for (l, layer) in self.layers.iter().enumerate() {
                let layer_t0 = Instant::now();
                let policy_win = self.cfg.0.window_policy.window_for(l);
                let layer_window = crate::cache::combine_windows(window, policy_win);
                let mut out_layer = burn::tensor::Tensor::<B, 3>::zeros([b, t, self.cfg.0.d_model], &hidden_full.device());
                let cache_l = cache.cache_mut(l);
                let mut s = 0usize;
                if self.cfg.0.verbose { eprintln!("layer {}: start", l); }
                while s < t {
                    let e = core::cmp::min(s + tile_q, t);
                    if self.cfg.0.verbose { eprintln!("layer {}: tile {}..{}", l, s, e); }
                    let chunk = hidden_full.clone().slice([0..b, s..e, 0..self.cfg.0.d_model]);
                    let params = StreamingMqaParams {
                        rope: Some(&self.rope),
                        start_pos: start_pos + s,
                        window: layer_window,
                        sinks: None,
                        attn_bias: None,
                    };
                    let out_chunk = layer.forward(chunk, cache_l, params); // [B, e-s, d]
                    out_layer.inplace(|tens| tens.slice_assign([0..b, s..e, 0..self.cfg.0.d_model], out_chunk.clone()));
                    s = e;
                }
                if self.cfg.0.verbose { eprintln!("layer {}: done in {:.2?}", l, layer_t0.elapsed()); }
                hidden_full = out_layer;
            }
            let hidden = self.norm_final.forward(hidden_full);
            let logits = self.lm_head.forward(hidden);
            let last = logits
                .clone()
                .slice([0..b, t - 1..t, 0..self.cfg.0.vocab_size]);
            last.squeeze::<2>(1)
        }
    }
}

impl<B: Backend> GptOssModel<B> {
    /// Attach per-layer MoE streaming contexts built from a model.bin mmap index.
    pub fn set_moe_streaming_contexts(&mut self, contexts: Vec<Arc<crate::moe::MoeStreamingContext>>) {
        assert_eq!(self.layers.len(), contexts.len(), "contexts length must match layers");
        for (l, ctx) in contexts.into_iter().enumerate() {
            self.layers[l].mlp.set_streaming(ctx);
        }
    }

    /// Set whether attention expects pre-scaled Q/K (skip 1/sqrt(d_k)).
    pub fn set_pre_scaled_qk(&mut self, flag: bool) {
        for layer in &mut self.layers {
            layer.attn.pre_scaled_qk = flag;
        }
    }

    // No device-quant residency stored in the module; we decode tiles on device from bytes.

    pub fn debug_layer_attn_qk(
        &self,
        tokens: Tensor<B, 2, Int>,
        layer_index: usize,
        start_pos: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [b, t] = tokens.dims();
        let hidden = self.tok_emb.forward(tokens);
        let layer = &self.layers[layer_index];
        let normed = layer.norm_attn.forward(hidden);
        // Projections
        let q_lin = layer.attn.query.forward(normed.clone());
        let k_lin = layer.attn.key.forward(normed.clone());
        let q = q_lin
            .clone()
            .reshape([b, t, layer.attn.n_heads, layer.attn.d_k])
            .swap_dims(1, 2); // [B, nH, T, d_k]
        let k0 = k_lin
            .clone()
            .reshape([b, t, layer.attn.kv_heads, layer.attn.d_k])
            .swap_dims(1, 2); // [B, kvH, T, d_k]
        // Apply RoPE with even-width handling
        let rope_width = layer.attn.d_k & !1usize;
        let apply_rope = |xs: Tensor<B, 4>| {
            if rope_width == 0 {
                xs
            } else if rope_width == layer.attn.d_k {
                self.rope.apply(xs.swap_dims(1, 2), start_pos).swap_dims(1, 2)
            } else {
                let dims = xs.dims(); // [B, H, T, d]
                let head = xs
                    .clone()
                    .slice([0..dims[0], 0..dims[1], 0..dims[2], 0..rope_width])
                    .swap_dims(1, 2);
                let tail = xs
                    .clone()
                    .slice([0..dims[0], 0..dims[1], 0..dims[2], rope_width..layer.attn.d_k])
                    .swap_dims(1, 2);
                let head_ro = self.rope.apply(head, start_pos).swap_dims(1, 2);
                Tensor::cat(vec![head_ro, tail], 3)
            }
        };
        let q_ro = apply_rope(q);
        let k_ro = apply_rope(k0);
        (q_ro, k_ro)
    }

    pub fn debug_layer_attn_qkv(
        &self,
        tokens: Tensor<B, 2, Int>,
        layer_index: usize,
        start_pos: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let (q_ro, k_ro) = self.debug_layer_attn_qk(tokens.clone(), layer_index, start_pos);
        let [b, t] = tokens.dims();
        let hidden = self.tok_emb.forward(tokens);
        let layer = &self.layers[layer_index];
        let normed = layer.norm_attn.forward(hidden);
        let v_lin = layer.attn.value.forward(normed);
        let v = v_lin
            .reshape([b, t, layer.attn.kv_heads, layer.attn.d_k])
            .swap_dims(1, 2);
        (q_ro, k_ro, v)
    }

    pub fn debug_get_attn_out_weight(&self, layer_index: usize) -> Tensor<B, 2> {
        self.layers[layer_index].attn.output.weight.val().clone()
    }
    pub fn debug_get_attn_out_bias(&self, layer_index: usize) -> Tensor<B, 1> {
        self.layers[layer_index].attn.output.bias.as_ref().unwrap().val().clone()
    }
    pub fn debug_get_tok_emb(&self) -> Tensor<B, 2> {
        self.tok_emb.weight.val().clone()
    }
    pub fn debug_get_lm_head(&self) -> Tensor<B, 2> {
        self.lm_head.weight.val().clone()
    }
    pub fn debug_get_norm_final(&self) -> Tensor<B, 1> {
        self.norm_final.gamma.val().clone()
    }

    pub fn debug_set_lm_head(&mut self, w: Tensor<B, 2>) {
        self.lm_head.weight = burn::module::Param::from_tensor(w);
    }

    pub fn debug_get_sinks(&self, layer_index: usize) -> Option<Tensor<B, 2>> {
        self.layers[layer_index].attn.debug_get_sinks()
    }

    pub fn debug_attn_weights(
        &self,
        tokens: Tensor<B, 2, Int>,
        layer_index: usize,
        start_pos: usize,
    ) -> Tensor<B, 4> {
        let [b, _t] = tokens.dims();
        let hidden = self.tok_emb.forward(tokens);
        let layer = &self.layers[layer_index];
        let normed = layer.norm_attn.forward(hidden);
        let mut cache = self.init_cache(b, &normed.device());
        layer
            .attn
            .debug_attn_weights(
                normed,
                cache.cache_mut(layer_index),
                crate::attention::StreamingMqaParams { rope: Some(&self.rope), start_pos, window: crate::attention::AttnWindow::Full, sinks: None, attn_bias: None },
            )
    }

    pub fn debug_attn_scores(
        &self,
        tokens: Tensor<B, 2, Int>,
        layer_index: usize,
        start_pos: usize,
    ) -> Tensor<B, 4> {
        let [b, _t] = tokens.dims();
        let hidden = self.tok_emb.forward(tokens);
        let layer = &self.layers[layer_index];
        let normed = layer.norm_attn.forward(hidden);
        let mut cache = self.init_cache(b, &normed.device());
        layer
            .attn
            .debug_attn_scores(
                normed,
                cache.cache_mut(layer_index),
                crate::attention::StreamingMqaParams { rope: Some(&self.rope), start_pos, window: crate::attention::AttnWindow::Full, sinks: None, attn_bias: None },
            )
    }

    pub fn debug_hidden_after_final_norm(
        &self,
        tokens: Tensor<B, 2, Int>,
        start_pos: usize,
    ) -> Tensor<B, 3> {
        let [b, _t] = tokens.dims();
        let mut hidden = self.tok_emb.forward(tokens);
        let mut cache = self.init_cache(b, &hidden.device());
        for (l, layer) in self.layers.iter().enumerate() {
            let state = crate::generate::runner::StreamState { start_pos, window: crate::attention::AttnWindow::Full };
            let params = crate::attention::StreamingMqaParams { rope: Some(&self.rope), start_pos: state.start_pos, window: state.window, sinks: None, attn_bias: None };
            let cache_l = cache.cache_mut(l);
            hidden = layer.forward(hidden, cache_l, params);
        }
        self.norm_final.forward(hidden)
    }
}
