use burn_core as burn;

use burn::tensor::{backend::Backend, Tensor};
use burn_tensor::activation::sigmoid;

/// GPT‑OSS SwiGLU (interleaved) with optional clamp and alpha=1.702 by default.
///
/// Input last dimension must be even and is interpreted as interleaved pairs:
/// [x_glu0, x_lin0, x_glu1, x_lin1, ...]. The output reduces the last dim by 2.
/// Formula (matching Triton/Metal reference):
///   x_glu' = clamp(x_glu,       [-inf, limit])
///   x_lin' = clamp(x_linear,    [-limit, +limit])
///   out    = (x_glu' * sigmoid(alpha * x_glu')) * (x_lin' + 1)
pub fn swiglu_clamp<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    alpha: f32,
    clamp_limit: Option<f32>,
) -> Tensor<B, D> {
    let mut dims = tensor.dims();
    let last_index = dims.len() - 1;
    let last = dims[last_index];
    assert!(last % 2 == 0, "SwiGLU input last dimension must be even");
    let half = last / 2;

    let numel = tensor.shape().num_elements();
    let leading = numel / last;
    let resh2 = tensor.reshape([leading, last]);
    let resh3 = resh2.reshape([leading, half, 2]);
    let x_glu = resh3.clone().slice([0..leading, 0..half, 0..1]).reshape([leading, half]);
    let x_lin = resh3.clone().slice([0..leading, 0..half, 1..2]).reshape([leading, half]);

    let (x_glu_c, x_lin_c) = if let Some(limit) = clamp_limit {
        (x_glu.clone().clamp(f32::NEG_INFINITY, limit), x_lin.clone().clamp(-limit, limit))
    } else {
        (x_glu.clone(), x_lin.clone())
    };

    let out_glu = x_glu_c.clone() * sigmoid(x_glu_c.clone().mul_scalar(alpha));
    let ones = Tensor::<B, 2>::ones([leading, half], &out_glu.device());
    let out = out_glu * (x_lin_c + ones);

    dims[last_index] = half;
    let shape: [usize; D] = dims;
    out.reshape(shape)
}
