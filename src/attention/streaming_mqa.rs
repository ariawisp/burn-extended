use burn_core as burn;

use burn::module::{Content, DisplaySettings, Module, ModuleDisplay};
use burn::nn::{Dropout, DropoutConfig, Initializer, Linear, LinearConfig};
use burn::{
    config::Config,
    tensor::{backend::Backend, Tensor},
};

use super::AttnWindow;

/// Configuration for streaming multi-query attention (MQA/GQA).
#[derive(Config, Debug)]
pub struct StreamingMultiQueryAttentionConfig {
    /// Model dimension per token (input/output features).
    pub d_model: usize,
    /// Total number of attention heads for Q/O.
    pub n_heads: usize,
    /// Number of shared key/value heads.
    pub num_key_value_heads: usize,
    /// Per-head dimension for attention (decoupled from d_model).
    pub head_dim: usize,
    /// Dropout probability on attention logits.
    #[config(default = 0.0)]
    pub dropout: f64,
    /// Use quiet softmax instead of regular softmax.
    #[config(default = false)]
    pub quiet_softmax: bool,
    /// If true, initialize a learnable sinks parameter stored inside the module
    /// with shape `[kv_heads, groups]` where `groups = n_heads / kv_heads`.
    #[config(default = false)]
    pub learned_sinks: bool,
    /// If true, assume Q and K projections are pre-scaled; skip the 1/sqrt(d_k) factor.
    #[config(default = false)]
    pub pre_scaled_qk: bool,
    /// Parameter initializer for linear layers.
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

/// Streaming multi-query attention with KV cache and sliding window.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct StreamingMultiQueryAttention<B: Backend> {
    /// Query projection: `[d_model -> d_model]`.
    pub query: Linear<B>,
    /// Key projection: `[d_model -> num_key_value_heads * head_dim]`.
    pub key: Linear<B>,
    /// Value projection: `[d_model -> num_key_value_heads * head_dim]`.
    pub value: Linear<B>,
    /// Output projection: `[d_model -> d_model]`.
    pub output: Linear<B>,
    /// Dropout on attention logits.
    pub dropout: Dropout,
    /// Model dimension per token (input/output features).
    pub d_model: usize,
    /// Total number of Q heads.
    pub n_heads: usize,
    /// Number of K/V heads.
    pub kv_heads: usize,
    /// Head dimension (d_model / n_heads).
    pub d_k: usize,
    /// Use quiet softmax.
    pub quiet_softmax: bool,
    /// Optional learnable sinks logits per (kv_head, group) pair: `[kv_heads, groups]`.
    /// When present and no runtime sinks are provided, this parameter is used.
    pub sinks: Option<burn::module::Param<Tensor<B, 2>>>,
    /// Skip 1/sqrt(d_k) scaling in attention scores because Q,K are pre-scaled.
    pub pre_scaled_qk: bool,
}

impl<B: Backend> ModuleDisplay for StreamingMultiQueryAttention<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("d_model", &self.d_model)
            .add("n_heads", &self.n_heads)
            .add("kv_heads", &self.kv_heads)
            .add("d_k", &self.d_k)
            .add("dropout", &self.dropout.prob)
            .add("quiet_softmax", &self.quiet_softmax)
            .optional()
    }
}

impl StreamingMultiQueryAttentionConfig {
    /// Initialize a new streaming MQA module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> StreamingMultiQueryAttention<B> {
        assert!(
            self.n_heads % self.num_key_value_heads == 0,
            "n_heads must be divisible by num_key_value_heads"
        );
        assert!(self.head_dim > 0, "head_dim must be > 0");
        let d_k = self.head_dim;

        let linear = |in_features, out_features| {
            LinearConfig::new(in_features, out_features)
                .with_initializer(self.initializer.clone())
                .init(device)
        };

        StreamingMultiQueryAttention {
            query: linear(self.d_model, self.n_heads * d_k),
            key: linear(self.d_model, self.num_key_value_heads * d_k),
            value: linear(self.d_model, self.num_key_value_heads * d_k),
            output: linear(self.n_heads * d_k, self.d_model),
            dropout: DropoutConfig::new(self.dropout).init(),
            d_model: self.d_model,
            n_heads: self.n_heads,
            kv_heads: self.num_key_value_heads,
            d_k,
            quiet_softmax: self.quiet_softmax,
            sinks: if self.learned_sinks {
                // Initialize to zeros; loader can overwrite with checkpoint values.
                let groups = self.n_heads / self.num_key_value_heads;
                Some(Initializer::Zeros.init::<B, 2, _>([self.num_key_value_heads, groups], device))
            } else {
                None
            },
            pre_scaled_qk: self.pre_scaled_qk,
        }
    }
}

/// Streaming KV cache for MQA.
pub struct StreamingMqaCache<B: Backend> {
    /// Key buffer: `[batch, cache_len, kv_heads, d_k]`.
    pub k: Tensor<B, 4>,
    /// Value buffer: `[batch, cache_len, kv_heads, d_k]`.
    pub v: Tensor<B, 4>,
    /// Absolute position after the latest write (exclusive).
    pub global_end_index: usize,
    /// Local buffer end index (exclusive).
    pub local_end_index: usize,
    /// Number of sink tokens preserved at the beginning of the buffer.
    pub sink_tokens: usize,
    /// Maximum capacity of the rolling buffer.
    pub cache_len: usize,
}

impl<B: Backend> StreamingMqaCache<B> {
    /// Create an empty cache with given capacity.
    pub fn new(
        device: &B::Device,
        batch: usize,
        cache_len: usize,
        kv_heads: usize,
        head_dim: usize,
        sink_tokens: usize,
    ) -> Self {
        let zeros_k = Tensor::<B, 4>::zeros([batch, cache_len, kv_heads, head_dim], device);
        let zeros_v = Tensor::<B, 4>::zeros([batch, cache_len, kv_heads, head_dim], device);

        Self {
            k: zeros_k,
            v: zeros_v,
            global_end_index: 0,
            local_end_index: 0,
            sink_tokens,
            cache_len,
        }
    }

    /// Resets indices while keeping allocated buffers.
    pub fn reset(&mut self) {
        self.global_end_index = 0;
        self.local_end_index = 0;
    }

    /// Current number of valid tokens stored in the cache.
    pub fn len(&self) -> usize {
        self.local_end_index
    }

    /// Whether the cache currently holds no tokens.
    pub fn is_empty(&self) -> bool {
        self.local_end_index == 0
    }

    /// Cache capacity (in tokens).
    pub fn capacity(&self) -> usize {
        self.cache_len
    }

    /// Whether the cache is full.
    pub fn is_full(&self) -> bool {
        self.local_end_index >= self.cache_len
    }

    /// Zero-out K and V buffers in place and reset indices.
    pub fn clear(&mut self) {
        let device = self.k.device();
        let [b, cap, h, d] = self.k.dims();
        self.k = Tensor::<B, 4>::zeros([b, cap, h, d], &device);
        self.v = Tensor::<B, 4>::zeros([b, cap, h, d], &device);
        self.reset();
    }
}

/// Parameters for streaming MQA forward.
pub struct StreamingMqaParams<'a, B: Backend> {
    /// Optional rotary encoding to apply to Q and K with an absolute start offset.
    pub rope: Option<&'a burn::nn::RotaryEncoding<B>>,
    /// Absolute position of the first token in the current chunk.
    pub start_pos: usize,
    /// Window selection policy.
    pub window: AttnWindow,
    /// Optional sinks logits per (kv_head, group) pair, shape `[kv_heads, groups]`.
    pub sinks: Option<&'a Tensor<B, 2>>,
    /// Optional additive attention bias with shape `[B, n_heads, q_len, k_len]`,
    /// where `k_len` must match the active window length.
    pub attn_bias: Option<&'a Tensor<B, 4>>,
}

// Ergonomics: allow using base StreamingParams with MQA by defaulting optional fields.
impl<'a, B: Backend> From<crate::attention::StreamingParams<'a, B>> for StreamingMqaParams<'a, B> {
    fn from(p: crate::attention::StreamingParams<'a, B>) -> Self {
        Self {
            rope: p.rope,
            start_pos: p.start_pos,
            window: p.window,
            sinks: None,
            attn_bias: None,
        }
    }
}

impl<B: Backend> StreamingMultiQueryAttention<B> {
    /// Forward with streaming KV cache and optional windowing and sinks bias.
    ///
    /// Inputs are self-attention by default (query=key=value).
    pub fn forward_streaming(
        &self,
        x: Tensor<B, 3>,
        cache: &mut StreamingMqaCache<B>,
        params: StreamingMqaParams<B>,
    ) -> Tensor<B, 3> {
        #[cfg(feature = "attn_debug")]
        {
            let [b, t, _] = x.dims();
            eprintln!("attn: b={} t={} start_pos={} pre_scaled_qk={} heads={} kv={} d_k={}", b, t, params.start_pos, self.pre_scaled_qk, self.n_heads, self.kv_heads, self.d_k);
        }
        debug_assert!(
            cache.sink_tokens <= cache.cache_len,
            "sink tokens exceed cache capacity"
        );
        let [batch_size, seq_len, _] = x.dims();
        let groups = self.n_heads / self.kv_heads;

        // Projections
        // Q -> [B, nH, T, d_k]
        let q = self.attention_linear_q(x.clone(), &self.query);
        // K,V -> [B, kvH, T, d_k] (after swap)
        let k = self.attention_linear_kv(x.clone(), &self.key);
        let v = self.attention_linear_kv(x, &self.value);

        // Optionally apply RoPE with offset to Q and K (last two dims are [heads, d_k]).
        let (q, k) = if let Some(rope) = params.rope {
            // reshape to [B, T, heads, d_k] for rope apply along sequence
            let q_rs = q.swap_dims(1, 2); // [B, T, nH, d_k]
            let k_rs = k.swap_dims(1, 2); // [B, T, kvH, d_k]

            // If d_k is odd, rotate only the largest even slice of the last dim and keep the tail unchanged.
            let rope_width = self.d_k & !1usize; // round down to even
            let apply_rope = |xs: Tensor<B, 4>| {
                if rope_width == 0 {
                    xs
                } else if rope_width == self.d_k {
                    rope.apply(xs, params.start_pos)
                } else {
                    let dims = xs.dims();
                    let head = xs
                        .clone()
                        .slice([0..dims[0], 0..dims[1], 0..dims[2], 0..rope_width]);
                    let tail = xs
                        .clone()
                        .slice([0..dims[0], 0..dims[1], 0..dims[2], rope_width..self.d_k]);
                    let head_ro = rope.apply(head, params.start_pos);
                    Tensor::cat(vec![head_ro, tail], 3)
                }
            };

            let q_ro = apply_rope(q_rs);
            let k_ro = apply_rope(k_rs);
            (q_ro.swap_dims(1, 2), k_ro.swap_dims(1, 2))
        } else {
            (q, k)
        };

        // Update rolling cache with new K/V tokens
        let num_new = seq_len;
        let current_end = params.start_pos + num_new;
        debug_assert!(
            current_end >= cache.global_end_index,
            "start_pos must be non-decreasing"
        );
        let sink = cache.sink_tokens;
        let cap = cache.cache_len;
        let delta = current_end.saturating_sub(cache.global_end_index);
        let need = cache.local_end_index + delta;

        if need > cap {
            let num_evicted = need - cap;
            crate::attention::evict_and_roll_mqa(
                cache,
                batch_size,
                self.kv_heads,
                self.d_k,
                sink,
                num_evicted,
            );
            cache.local_end_index += delta;
        } else {
            cache.local_end_index += delta;
        }

        // Write new K,V at the end
        let local_end = cache.local_end_index;
        let local_start = local_end - num_new;
        let k_rs = k.swap_dims(1, 2); // [B, T, kvH, d_k]
        let v_rs = v.swap_dims(1, 2);
        cache.k.inplace(|t| {
            t.slice_assign(
                [
                    0..batch_size,
                    local_start..local_end,
                    0..self.kv_heads,
                    0..self.d_k,
                ],
                k_rs,
            )
        });
        cache.v.inplace(|t| {
            t.slice_assign(
                [
                    0..batch_size,
                    local_start..local_end,
                    0..self.kv_heads,
                    0..self.d_k,
                ],
                v_rs,
            )
        });
        cache.global_end_index = current_end;

        // Determine the active window
        let active_len = match params.window {
            AttnWindow::Full => local_end,
            AttnWindow::Window(w) => {
                debug_assert!(
                    cache.sink_tokens + w <= cache.cache_len,
                    "window+sink exceeds cache capacity"
                );
                sink + w.min(local_end.saturating_sub(sink))
            }
        };
        let start = local_end.saturating_sub(active_len);

        // Gather window K,V from cache: [B, kvH, Tk, d_k]
        let k_win = cache
            .k
            .clone()
            .slice([
                0..batch_size,
                start..local_end,
                0..self.kv_heads,
                0..self.d_k,
            ])
            .swap_dims(1, 2);
        let v_win = cache
            .v
            .clone()
            .slice([
                0..batch_size,
                start..local_end,
                0..self.kv_heads,
                0..self.d_k,
            ])
            .swap_dims(1, 2);

        // Expand KV across groups to match Q heads: [B, nH, Tk, d_k]
        let k_exp = k_win
            .unsqueeze_dim::<5>(2) // [B, kvH, 1, Tk, d_k]
            .repeat_dim(2, groups) // [B, kvH, groups, Tk, d_k]
            .reshape([batch_size, self.n_heads, active_len, self.d_k]);
        let v_exp = v_win.unsqueeze_dim::<5>(2).repeat_dim(2, groups).reshape([
            batch_size,
            self.n_heads,
            active_len,
            self.d_k,
        ]);

        // Attention — compute in small q-tiles to bound memory.
        let scale_dk = if self.pre_scaled_qk { 1 } else { self.d_k };
        let mut out = Tensor::<B, 3>::zeros([batch_size, seq_len, self.n_heads * self.d_k], &q.device());
        let tile_q = 16usize;
        let sinks_vec = if let Some(s) = params.sinks {
            Some(s.clone().reshape([self.kv_heads * groups]))
        } else if let Some(param) = self.sinks.as_ref() {
            Some(param.val().clone().reshape([self.kv_heads * groups]))
        } else {
            None
        };
        let mut start_q = 0usize;
        while start_q < seq_len {
            let take_q = core::cmp::min(tile_q, seq_len - start_q);
            #[cfg(feature = "attn_debug")]
            eprintln!("attn tile: start_q={} take_q={} active_len={}", start_q, take_q, active_len);
            let q_tile = q.clone().slice([0..batch_size, 0..self.n_heads, start_q..start_q + take_q, 0..self.d_k]);
            let mut scores = crate::attention::compute_scores(q_tile, k_exp.clone(), scale_dk, &self.dropout);
            if let Some(bias_ref) = params.attn_bias.as_ref() {
                let bias_full = (*bias_ref).clone();
                let bias_tile = bias_full.slice([0..batch_size, 0..self.n_heads, start_q..start_q + take_q, 0..active_len]);
                scores = scores + bias_tile;
            }
            let weights_tile = if let Some(sinks) = sinks_vec.as_ref() {
                crate::attention::apply_sinks_then_softmax(
                    scores,
                    sinks.clone(),
                    batch_size,
                    self.n_heads,
                    take_q,
                    active_len,
                    self.quiet_softmax,
                )
            } else {
                crate::attention::apply_bias_and_softmax(scores, None, self.quiet_softmax)
            };
            let ctx_tile = weights_tile.matmul(v_exp.clone()); // [B, nH, tq, d_k]
            let ctx_tile = ctx_tile.swap_dims(1, 2).reshape([batch_size, take_q, self.n_heads * self.d_k]);
            out.inplace(|t| t.slice_assign([0..batch_size, start_q..start_q + take_q, 0..self.n_heads * self.d_k], ctx_tile.clone()));
            start_q += take_q;
        }
        self.output.forward(out)
    }

    fn attention_linear_q(&self, x: Tensor<B, 3>, linear: &Linear<B>) -> Tensor<B, 4> {
        let [batch_size, seq_length, _d_model] = x.dims();
        linear
            .forward(x)
            .reshape([batch_size, seq_length, self.n_heads, self.d_k])
            .swap_dims(1, 2)
    }

    fn attention_linear_kv(&self, x: Tensor<B, 3>, linear: &Linear<B>) -> Tensor<B, 4> {
        let [batch_size, seq_length, _d_model] = x.dims();
        linear
            .forward(x)
            .reshape([batch_size, seq_length, self.kv_heads, self.d_k])
            .swap_dims(1, 2)
    }

    pub fn debug_get_sinks(&self) -> Option<Tensor<B, 2>> {
        self.sinks.as_ref().map(|p| p.val().clone())
    }

    #[allow(dead_code)]
    pub fn debug_attn_weights(
        &self,
        x: Tensor<B, 3>,
        cache: &mut StreamingMqaCache<B>,
        params: StreamingMqaParams<B>,
    ) -> Tensor<B, 4> {
        let [batch_size, seq_len, _] = x.dims();
        let groups = self.n_heads / self.kv_heads;
        let q = self.attention_linear_q(x.clone(), &self.query);
        let k = self.attention_linear_kv(x.clone(), &self.key);
        let v = self.attention_linear_kv(x, &self.value);

        let (q, k) = if let Some(rope) = params.rope {
            let q_rs = q.swap_dims(1, 2);
            let k_rs = k.swap_dims(1, 2);
            let rope_width = self.d_k & !1usize;
            let apply_rope = |xs: Tensor<B, 4>| {
                if rope_width == 0 {
                    xs
                } else if rope_width == self.d_k {
                    rope.apply(xs, params.start_pos)
                } else {
                    let dims = xs.dims();
                    let head = xs
                        .clone()
                        .slice([0..dims[0], 0..dims[1], 0..dims[2], 0..rope_width]);
                    let tail = xs
                        .clone()
                        .slice([0..dims[0], 0..dims[1], 0..dims[2], rope_width..self.d_k]);
                    let head_ro = rope.apply(head, params.start_pos);
                    Tensor::cat(vec![head_ro, tail], 3)
                }
            };
            let q_ro = apply_rope(q_rs);
            let k_ro = apply_rope(k_rs);
            (q_ro.swap_dims(1, 2), k_ro.swap_dims(1, 2))
        } else {
            (q, k)
        };

        // Update cache
        let num_new = seq_len;
        let current_end = params.start_pos + num_new;
        let sink = cache.sink_tokens;
        let cap = cache.cache_len;
        let delta = current_end.saturating_sub(cache.global_end_index);
        let need = cache.local_end_index + delta;
        if need > cap {
            let num_evicted = need - cap;
            crate::attention::evict_and_roll_mqa(cache, batch_size, self.kv_heads, self.d_k, sink, num_evicted);
            cache.local_end_index += delta;
        } else {
            cache.local_end_index += delta;
        }
        let local_end = cache.local_end_index;
        let local_start = local_end - num_new;
        let k_rs = k.swap_dims(1, 2);
        let v_rs = v.swap_dims(1, 2);
        cache.k.inplace(|t| {
            t.slice_assign([0..batch_size, local_start..local_end, 0..self.kv_heads, 0..self.d_k], k_rs)
        });
        cache.v.inplace(|t| {
            t.slice_assign([0..batch_size, local_start..local_end, 0..self.kv_heads, 0..self.d_k], v_rs)
        });
        cache.global_end_index = current_end;

        // Window
        let active_len = match params.window {
            AttnWindow::Full => local_end,
            AttnWindow::Window(w) => sink + w.min(local_end.saturating_sub(sink)),
        };
        let start = local_end.saturating_sub(active_len);
        let k_win = cache
            .k
            .clone()
            .slice([0..batch_size, start..local_end, 0..self.kv_heads, 0..self.d_k])
            .swap_dims(1, 2); // [B, kvH, Tk, d]
        // v_win not required for weights-only path

        // Compute attention weights in small q-tiles per group (avoid head expansion).
        let scale_dk = if self.pre_scaled_qk { 1 } else { self.d_k };
        let tile_q = 16usize;
        let mut weights_out = Tensor::<B, 4>::zeros(
            [batch_size, self.n_heads, seq_len, active_len],
            &q.device(),
        );
        let mut start_q = 0usize;
        while start_q < seq_len {
            let take_q = core::cmp::min(tile_q, seq_len - start_q);
            let mut wts_tile_all = Tensor::<B, 4>::zeros(
                [batch_size, self.n_heads, take_q, active_len],
                &q.device(),
            );
            for g in 0..groups {
                let h0 = g * self.kv_heads;
                let h1 = h0 + self.kv_heads;
                let q_g = q
                    .clone()
                    .slice([0..batch_size, h0..h1, start_q..start_q + take_q, 0..self.d_k]);
                let scores_g = crate::attention::compute_scores(q_g, k_win.clone(), scale_dk, &self.dropout);
                let weights_g = crate::attention::apply_bias_and_softmax(scores_g, None, self.quiet_softmax);
                wts_tile_all.inplace(|t| {
                    t.slice_assign(
                        [0..batch_size, h0..h1, 0..take_q, 0..active_len],
                        weights_g.clone(),
                    )
                });
            }
            weights_out.inplace(|t| {
                t.slice_assign(
                    [0..batch_size, 0..self.n_heads, start_q..start_q + take_q, 0..active_len],
                    wts_tile_all.clone(),
                )
            });
            start_q += take_q;
        }
        weights_out
    }

    #[allow(dead_code)]
    pub fn debug_attn_scores(
        &self,
        x: Tensor<B, 3>,
        cache: &mut StreamingMqaCache<B>,
        params: StreamingMqaParams<B>,
    ) -> Tensor<B, 4> {
        let [batch_size, seq_len, _] = x.dims();
        let groups = self.n_heads / self.kv_heads;
        let q = self.attention_linear_q(x.clone(), &self.query);
        let k = self.attention_linear_kv(x.clone(), &self.key);
        let (q, k) = if let Some(rope) = params.rope {
            let q_rs = q.swap_dims(1, 2);
            let k_rs = k.swap_dims(1, 2);
            let rope_width = self.d_k & !1usize;
            let apply_rope = |xs: Tensor<B, 4>| {
                if rope_width == 0 { xs } else if rope_width == self.d_k { rope.apply(xs, params.start_pos) } else {
                    let dims = xs.dims();
                    let head = xs.clone().slice([0..dims[0], 0..dims[1], 0..dims[2], 0..rope_width]);
                    let tail = xs.clone().slice([0..dims[0], 0..dims[1], 0..dims[2], rope_width..self.d_k]);
                    let head_ro = rope.apply(head, params.start_pos);
                    Tensor::cat(vec![head_ro, tail], 3)
                }
            };
            let q_ro = apply_rope(q_rs);
            let k_ro = apply_rope(k_rs);
            (q_ro.swap_dims(1, 2), k_ro.swap_dims(1, 2))
        } else { (q, k) };
        // Update cache and expand K
        let num_new = seq_len;
        let current_end = params.start_pos + num_new;
        let sink = cache.sink_tokens;
        let cap = cache.cache_len;
        let delta = current_end.saturating_sub(cache.global_end_index);
        let need = cache.local_end_index + delta;
        if need > cap { let num_evicted = need - cap; crate::attention::evict_and_roll_mqa(cache, batch_size, self.kv_heads, self.d_k, sink, num_evicted); cache.local_end_index += delta; } else { cache.local_end_index += delta; }
        let local_end = cache.local_end_index; let local_start = local_end - num_new;
        let k_rs = k.swap_dims(1, 2);
        cache.k.inplace(|t| { t.slice_assign([0..batch_size, local_start..local_end, 0..self.kv_heads, 0..self.d_k], k_rs) });
        cache.global_end_index = current_end;
        let active_len = match params.window { AttnWindow::Full => local_end, AttnWindow::Window(w) => sink + w.min(local_end.saturating_sub(sink)) };
        let start = local_end.saturating_sub(active_len);
        let k_win = cache.k.clone().slice([0..batch_size, start..local_end, 0..self.kv_heads, 0..self.d_k]).swap_dims(1, 2);
        let k_exp = k_win.unsqueeze_dim::<5>(2).repeat_dim(2, groups).reshape([batch_size, self.n_heads, active_len, self.d_k]);
        let scale_dk = if self.pre_scaled_qk { 1 } else { self.d_k };
        crate::attention::compute_scores(q, k_exp, scale_dk, &self.dropout)
    }
}
