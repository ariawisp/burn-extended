use burn_core as burn;

use burn::tensor::{Tensor, backend::Backend};

/// Window selection policy for streaming attention (baseline variant).
#[derive(Debug, Clone, Copy)]
pub enum AttnWindow {
    Full,
    Window(usize),
}

/// Streaming KV cache for baseline MHA (owned by burn-extended).
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
}

