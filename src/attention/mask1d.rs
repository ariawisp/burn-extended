use burn_core as burn;

use burn::tensor::{Bool, Int, Shape, Tensor, backend::Backend};

/// Generate a 1D padding mask from sequence lengths: [B, max_len], true marks padding.
pub fn lengths_to_mask<B: Backend>(lengths: &[usize], max_len: usize, device: &B::Device) -> Tensor<B, 2, Bool> {
    let batch = lengths.len();
    let mut mask = Tensor::<B, 2, Bool>::full([batch, max_len], true, device);
    for (i, &len_i) in lengths.iter().enumerate() {
        let keep = len_i.min(max_len);
        if keep > 0 {
            let unmask = Tensor::<B, 2, Bool>::full([1, keep], false, device);
            mask = mask.slice_assign([i..i + 1, 0..keep], unmask);
        }
    }
    mask
}

/// Generate a causal mask [L, L] where true masks future tokens.
pub fn generate_causal_mask_1d<B: Backend>(seq_len: usize, device: &B::Device) -> Tensor<B, 2, Bool> {
    Tensor::<B, 2, Bool>::tril_mask([seq_len, seq_len], 0, device)
}

/// Generate a windowed causal mask with optional sink tokens.
/// Returns a `[batch, seq_len, seq_len]` tensor matching `AttnWindow` behaviour.
pub fn generate_windowed_causal_mask<B: Backend>(
    batch_size: usize,
    seq_length: usize,
    window_len: Option<usize>,
    sink_tokens: usize,
    device: &B::Device,
) -> Tensor<B, 3, Bool> {
    let base = Tensor::<B, 2, Bool>::tril_mask([seq_length, seq_length], 0, device);

    if let Some(w) = window_len {
        let mut mask = base;
        for i in 0..seq_length {
            let start = if i < sink_tokens {
                0
            } else {
                let window_start = i.saturating_sub(w);
                window_start.max(sink_tokens)
            };

            if start > 0 {
                let fill = Tensor::<B, 2, Bool>::full([1, start], true, device);
                mask = mask.slice_assign([i..i + 1, 0..start], fill);
            }
        }
        mask.reshape([1, seq_length, seq_length]).repeat_dim(0, batch_size)
    } else {
        base.reshape([1, seq_length, seq_length]).repeat_dim(0, batch_size)
    }
}

/// Generate a chunked causal mask [L, L] with Conformer-style left chunks.
pub fn generate_chunked_causal_mask_1d<B: Backend>(seq_len: usize, chunk_size: usize, num_left_chunks: isize, device: &B::Device) -> Tensor<B, 2, Bool> {
    let mut mask = Tensor::<B, 2, Bool>::full([seq_len, seq_len], true, device);
    if chunk_size == 0 { return mask; }
    for i in 0..seq_len {
        let chunk_idx = i / chunk_size;
        let start = if num_left_chunks < 0 { 0 } else { ((chunk_idx as isize - num_left_chunks).max(0) as usize) * chunk_size };
        let mut end = ((chunk_idx + 1) * chunk_size).min(seq_len);
        if end > i + 1 { end = i + 1; }
        if end > start {
            let unmask = Tensor::<B, 2, Bool>::full([1, end - start], false, device);
            mask = mask.slice_assign([i..i + 1, start..end], unmask);
        }
    }
    mask
}
