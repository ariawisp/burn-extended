use burn_core as burn;

use burn::tensor::{backend::Backend, Tensor};

/// Fixed K/V cache for cross-attention with an encoder context.
/// Shapes: k, v are `[B, key_len, heads, head_dim]`.
#[derive(Debug, Clone)]
pub struct CrossAttnCache<B: Backend> {
    pub k: Tensor<B, 4>,
    pub v: Tensor<B, 4>,
    pub is_init: bool,
}

impl<B: Backend> CrossAttnCache<B> {
    pub fn new(
        device: &B::Device,
        batch: usize,
        key_len: usize,
        heads: usize,
        head_dim: usize,
    ) -> Self {
        let k = Tensor::<B, 4>::zeros([batch, key_len, heads, head_dim], device);
        let v = Tensor::<B, 4>::zeros([batch, key_len, heads, head_dim], device);
        Self {
            k,
            v,
            is_init: false,
        }
    }

    /// Replace the K/V context; marks the cache initialized.
    pub fn set(&mut self, k: Tensor<B, 4>, v: Tensor<B, 4>) {
        self.k = k;
        self.v = v;
        self.is_init = true;
    }

    /// Clears the K/V context and marks uninitialized.
    pub fn clear(&mut self) {
        let device = self.k.device();
        let [b, t, h, d] = self.k.dims();
        self.k = Tensor::<B, 4>::zeros([b, t, h, d], &device);
        self.v = Tensor::<B, 4>::zeros([b, t, h, d], &device);
        self.is_init = false;
    }
}
