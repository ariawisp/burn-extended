use burn_core as burn;

use burn::tensor::{backend::Backend, Int, Tensor};

/// Generate ALiBi slopes for `n_heads` following the common recipe used in literature.
/// See: https://github.com/ofirpress/attention_with_linear_biases
pub fn alibi_slopes(n_heads: usize) -> alloc::vec::Vec<f32> {
    use alloc::vec::Vec;
    // Helper: closest power of two <= n
    fn closest_power_of_2(n: usize) -> usize {
        let mut k = 1usize;
        while k * 2 <= n {
            k *= 2;
        }
        k
    }

    let mut slopes = Vec::with_capacity(n_heads);
    let m = closest_power_of_2(n_heads);
    // Base slopes for power-of-two heads
    for i in 0..m {
        let exp = -(2f32.powf(i as f32 + 3.0) / 2f32.powf(m as f32));
        let slope = 2f32.powf(exp);
        slopes.push(slope);
    }
    // If not power-of-two, generate remaining by interleaving
    if m < n_heads {
        let extra = n_heads - m;
        let last = slopes[slopes.len() - 1];
        let ratio = last * 0.5;
        for i in 0..extra {
            slopes.push(ratio * 2f32.powf(-(i as f32)));
        }
    }
    slopes
}

/// Convenience: return slopes as a tensor `[n_heads]` on device.
pub fn alibi_slopes_tensor<B: Backend>(n_heads: usize, device: &B::Device) -> Tensor<B, 1> {
    let s = alibi_slopes(n_heads);
    Tensor::from_floats(s.as_slice(), device)
}

/// Build an ALiBi bias tensor shaped `[B, H, Tq, Tk]` aligned to absolute positions.
///
/// - `q_start`/`k_start` are absolute positions for the first query/key indices.
/// - ALiBi bias per head `h` is: `slopes[h] * (k_pos - q_pos)`.
pub fn alibi_bias_for_positions<B: Backend>(
    batch: usize,
    slopes: &Tensor<B, 1>, // [H]
    t_query: usize,
    t_key: usize,
    q_start: usize,
    k_start: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let n_heads = slopes.dims()[0];
    let q_idx = Tensor::<B, 1, Int>::arange(q_start as i64..(q_start + t_query) as i64, device)
        .float()
        .reshape([t_query, 1]);
    let k_idx = Tensor::<B, 1, Int>::arange(k_start as i64..(k_start + t_key) as i64, device)
        .float()
        .reshape([1, t_key]);
    let diff = k_idx - q_idx; // [Tq, Tk]
    let s = slopes.clone().reshape([n_heads, 1, 1]); // [H,1,1]
    let bias_h = s * diff.unsqueeze(); // [H, Tq, Tk]
    bias_h
        .unsqueeze_dim::<4>(0)
        .repeat_dim(0, batch) // [B, H, Tq, Tk]
}

