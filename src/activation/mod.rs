use burn::tensor::{Tensor, backend::Backend};
use burn_tensor::activation::sigmoid;

/// Applies SwiGLU activation with an optional clamp applied to the result.
///
/// The input tensor must have an even-sized last dimension. The first half of the
/// last dimension is treated as the value branch and the second half as the gate.
pub fn swiglu_clamp<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    alpha: f32,
    clamp_limit: Option<f32>,
) -> Tensor<B, D> {
    let dims = tensor.dims();
    let last = dims[dims.len() - 1];
    assert!(last % 2 == 0, "SwiGLU input last dimension must be even");
    let half = last / 2;

    let numel = tensor.shape().num_elements();
    let leading = numel / last;
    let reshaped = tensor.reshape([leading, last]);

    let value = reshaped.clone().slice([0..leading, 0..half]);
    let gate = reshaped.slice([0..leading, half..last]);

    let swish = sigmoid(gate.clone().mul_scalar(alpha)) * gate;
    let mut activated = swish * value;

    if let Some(limit) = clamp_limit {
        activated = activated.clamp_scalar(-limit, limit);
    }

    let mut final_shape = dims.to_vec();
    final_shape[dims.len() - 1] = half;
    activated.reshape(final_shape)
}
