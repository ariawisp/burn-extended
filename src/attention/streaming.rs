use burn_core as burn;

use burn::module::{Content, DisplaySettings, Module, ModuleDisplay};
use burn::nn::{Dropout, DropoutConfig, Initializer, Linear, LinearConfig};
use burn::{
    config::Config,
    tensor::{Tensor, backend::Backend},
};

use burn_tensor::activation::{quiet_softmax, softmax};

/// Window selection policy for streaming attention.
#[derive(Debug, Clone, Copy)]
pub enum AttnWindow {
    /// Attend to all cached tokens (full causal attention over cache).
    Full,
    /// Attend to at most `window_len` most recent tokens plus `sink_tokens`.
    Window(usize),
}

/// Configuration for the streaming multi-head attention module.
#[derive(Config, Debug)]
pub struct StreamingMultiHeadAttentionConfig {
    /// Size of the input/output features (d_model).
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Dropout probability on attention logits.
    #[config(default = 0.0)]
    pub dropout: f64,
    /// Use quiet softmax instead of regular softmax.
    #[config(default = false)]
    pub quiet_softmax: bool,
    /// Parameter initializer for the linear layers.
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

/// Streaming multi-head attention with a rolling KV cache and optional window.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct StreamingMultiHeadAttention<B: Backend> {
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

impl<B: Backend> ModuleDisplay for StreamingMultiHeadAttention<B> {
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

impl StreamingMultiHeadAttentionConfig {
    /// Initialize a new streaming MHA module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> StreamingMultiHeadAttention<B> {
        let linear = |in_features, out_features| {
            LinearConfig::new(in_features, out_features)
                .with_initializer(self.initializer.clone())
                .init(device)
        };

        assert!(
            self.d_model % self.n_heads == 0,
            "d_model must be divisible by n_heads",
        );
        let d_k = self.d_model / self.n_heads;

        StreamingMultiHeadAttention {
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

/// Streaming KV cache for sliding-window attention.
pub struct StreamingMhaCache<B: Backend> {
    pub k: Tensor<B, 4>,
    pub v: Tensor<B, 4>,
    pub global_end_index: usize,
    pub local_end_index: usize,
    pub sink_tokens: usize,
    pub cache_len: usize,
}

impl<B: Backend> StreamingMhaCache<B> {
    pub fn new(
        device: &B::Device,
        batch: usize,
        cache_len: usize,
        n_heads: usize,
        head_dim: usize,
        sink_tokens: usize,
    ) -> Self {
        let zeros_k = Tensor::<B, 4>::zeros([batch, cache_len, n_heads, head_dim], device);
        let zeros_v = Tensor::<B, 4>::zeros([batch, cache_len, n_heads, head_dim], device);

        Self {
            k: zeros_k,
            v: zeros_v,
            global_end_index: 0,
            local_end_index: 0,
            sink_tokens,
            cache_len,
        }
    }

    pub fn reset(&mut self) {
        self.global_end_index = 0;
        self.local_end_index = 0;
    }

    pub fn len(&self) -> usize {
        self.local_end_index
    }

    pub fn is_empty(&self) -> bool {
        self.local_end_index == 0
    }

    pub fn capacity(&self) -> usize {
        self.cache_len
    }

    pub fn is_full(&self) -> bool {
        self.local_end_index >= self.cache_len
    }

    pub fn clone_shallow(&self) -> Self {
        Self {
            k: self.k.clone(),
            v: self.v.clone(),
            global_end_index: self.global_end_index,
            local_end_index: self.local_end_index,
            sink_tokens: self.sink_tokens,
            cache_len: self.cache_len,
        }
    }

    pub fn clear(&mut self) {
        let device = self.k.device();
        let [b, cap, h, d] = self.k.dims();
        self.k = Tensor::<B, 4>::zeros([b, cap, h, d], &device);
        self.v = Tensor::<B, 4>::zeros([b, cap, h, d], &device);
        self.reset();
    }
}

/// Parameters for streaming attention forward passes.
pub struct StreamingParams<'a, B: Backend> {
    pub rope: Option<&'a burn::nn::rope_encoding::RotaryEncoding<B>>,
    pub start_pos: usize,
    pub window: AttnWindow,
}

/// Windowed slice of the streaming cache returned after writing new keys and values.
pub(crate) struct CacheView<B: Backend> {
    pub k: Tensor<B, 4>,
    pub v: Tensor<B, 4>,
    pub active_len: usize,
    pub local_end: usize,
}

/// Update the rolling cache with the newly produced keys/values and return the active
/// window (respecting sink tokens and per-layer window configuration).
///
/// The input `k`/`v` tensors are `[B, n_heads, seq_len, d_k]` matching the current chunk.
/// The returned tensors are `[B, n_heads, active_len, d_k]` ready for attention matmul.
pub(crate) fn update_cache_window<B: Backend>(
    cache: &mut StreamingMhaCache<B>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    start_pos: usize,
    window: AttnWindow,
) -> CacheView<B> {
    let [batch_size, n_heads, seq_len, d_k] = k.dims();
    let num_new = seq_len;
    let current_end = start_pos + num_new;
    debug_assert!(
        current_end >= cache.global_end_index,
        "start_pos must be non-decreasing",
    );

    let sink = cache.sink_tokens;
    let cap = cache.cache_len;
    let delta = current_end.saturating_sub(cache.global_end_index);
    let need = cache.local_end_index + delta;

    if need > cap {
        let num_evicted = need - cap;
        let num_rolled = cache
            .local_end_index
            .saturating_sub(num_evicted + sink);
        if num_rolled > 0 {
            let src_start = sink + num_evicted;
            let src_end = sink + num_evicted + num_rolled;

            let rolled_k = cache.k.clone().slice([
                0..batch_size,
                src_start..src_end,
                0..n_heads,
                0..d_k,
            ]);
            cache.k.inplace(|t| {
                t.slice_assign(
                    [
                        0..batch_size,
                        sink..sink + num_rolled,
                        0..n_heads,
                        0..d_k,
                    ],
                    rolled_k,
                )
            });

            let rolled_v = cache.v.clone().slice([
                0..batch_size,
                src_start..src_end,
                0..n_heads,
                0..d_k,
            ]);
            cache.v.inplace(|t| {
                t.slice_assign(
                    [
                        0..batch_size,
                        sink..sink + num_rolled,
                        0..n_heads,
                        0..d_k,
                    ],
                    rolled_v,
                )
            });
        }

        cache.local_end_index = cache.local_end_index + delta - num_evicted;
    } else {
        cache.local_end_index += delta;
    }

    let local_end = cache.local_end_index;
    let local_start = local_end - num_new;
    let k_rs = k.swap_dims(1, 2);
    let v_rs = v.swap_dims(1, 2);

    cache.k.inplace(|t| {
        t.slice_assign(
            [
                0..batch_size,
                local_start..local_end,
                0..n_heads,
                0..d_k,
            ],
            k_rs,
        )
    });
    cache.v.inplace(|t| {
        t.slice_assign(
            [
                0..batch_size,
                local_start..local_end,
                0..n_heads,
                0..d_k,
            ],
            v_rs,
        )
    });
    cache.global_end_index = current_end;

    let active_len = match window {
        AttnWindow::Full => local_end,
        AttnWindow::Window(w) => {
            debug_assert!(
                cache.sink_tokens + w <= cache.cache_len,
                "window+sink exceeds cache capacity",
            );
            sink + w.min(local_end.saturating_sub(sink))
        }
    };
    let start = local_end.saturating_sub(active_len);

    let k_win = cache
        .k
        .clone()
        .slice([
            0..batch_size,
            start..local_end,
            0..n_heads,
            0..d_k,
        ])
        .swap_dims(1, 2);
    let v_win = cache
        .v
        .clone()
        .slice([
            0..batch_size,
            start..local_end,
            0..n_heads,
            0..d_k,
        ])
        .swap_dims(1, 2);

    CacheView {
        k: k_win,
        v: v_win,
        active_len,
        local_end,
    }
}

impl<B: Backend> StreamingMultiHeadAttention<B> {
    /// Forward pass using the streaming cache with optional attention window.
    pub fn forward_streaming(
        &self,
        x: Tensor<B, 3>,
        cache: &mut StreamingMhaCache<B>,
        params: StreamingParams<B>,
    ) -> Tensor<B, 3> {
        debug_assert!(
            cache.sink_tokens <= cache.cache_len,
            "sink tokens exceed cache capacity",
        );
        let [batch_size, seq_len, _] = x.dims();

        let q = self.attention_linear(x.clone(), &self.query);
        let k = self.attention_linear(x.clone(), &self.key);
        let v = self.attention_linear(x, &self.value);

        // Optionally apply RoPE using the provided absolute offset.
        let (q, k) = if let Some(rope) = params.rope {
            debug_assert!(
                self.d_k % 2 == 0,
                "RotaryEncoding requires even head_dim",
            );
            let q_rs = q.swap_dims(1, 2);
            let k_rs = k.swap_dims(1, 2);
            let q_ro = rope.apply(q_rs, params.start_pos);
            let k_ro = rope.apply(k_rs, params.start_pos);
            (q_ro.swap_dims(1, 2), k_ro.swap_dims(1, 2))
        } else {
            (q, k)
        };

        // Update rolling cache with new tokens.
        let cache_view = update_cache_window(cache, k, v, params.start_pos, params.window);
        let k_win = cache_view.k;
        let v_win = cache_view.v;

        let mut attn_scores = q
            .matmul(k_win.transpose())
            .div_scalar((self.d_k as f32).sqrt());
        attn_scores = self.dropout.forward(attn_scores);

        let weights = if self.quiet_softmax {
            quiet_softmax(attn_scores, 3)
        } else {
            softmax(attn_scores, 3)
        };

        let context = weights
            .matmul(v_win)
            .swap_dims(1, 2)
            .reshape([batch_size, seq_len, self.d_model]);
        self.output.forward(context)
    }

    fn attention_linear(&self, x: Tensor<B, 3>, linear: &Linear<B>) -> Tensor<B, 4> {
        let [batch_size, seq_length, _] = x.dims();
        linear
            .forward(x)
            .reshape([batch_size, seq_length, self.n_heads, self.d_k])
            .swap_dims(1, 2)
    }
}
