use burn_core as burn;

use burn::tensor::{Tensor, backend::Backend};

/// Simple Euler step for flow-matching/diffusion.
pub fn euler_step<B: Backend, const D: usize>(x: Tensor<B, D>, dx: Tensor<B, D>, dt: f32) -> Tensor<B, D> {
    x + dx.mul_scalar(dt)
}

