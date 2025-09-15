use burn_core as burn;

use burn::tensor::{Tensor, backend::Backend};

/// Heun's method (improved Euler) for diffusion/flow-matching.
pub fn heun_step<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    dx: Tensor<B, D>,
    dx_pred: Tensor<B, D>,
    dt: f32,
) -> Tensor<B, D> {
    x + (dx + dx_pred).mul_scalar(0.5 * dt)
}

