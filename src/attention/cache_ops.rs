use burn_core as burn;

use burn::tensor::backend::Backend;

/// Evict and roll the streaming MHA cache window left by `num_evicted` tokens on dim=1.
/// Preserves the first `sink_tokens` entries.
pub fn evict_and_roll_mha<B: Backend>(
    cache: &mut super::StreamingMhaCache<B>,
    batch_size: usize,
    n_heads: usize,
    d_k: usize,
    sink_tokens: usize,
    num_evicted: usize,
) {
    let avail = cache.local_end_index.saturating_sub(sink_tokens);
    if avail == 0 || num_evicted >= avail {
        cache.local_end_index = cache.local_end_index.saturating_sub(num_evicted);
        return;
    }

    let k_sub = cache.k.clone().slice([
        0..batch_size,
        sink_tokens..cache.local_end_index,
        0..n_heads,
        0..d_k,
    ]);
    let k_rolled = k_sub.roll(&[-(num_evicted as i64)], &[1]);
    cache.k.inplace(|t| {
        t.slice_assign(
            [
                0..batch_size,
                sink_tokens..sink_tokens + avail,
                0..n_heads,
                0..d_k,
            ],
            k_rolled,
        )
    });

    let v_sub = cache.v.clone().slice([
        0..batch_size,
        sink_tokens..cache.local_end_index,
        0..n_heads,
        0..d_k,
    ]);
    let v_rolled = v_sub.roll(&[-(num_evicted as i64)], &[1]);
    cache.v.inplace(|t| {
        t.slice_assign(
            [
                0..batch_size,
                sink_tokens..sink_tokens + avail,
                0..n_heads,
                0..d_k,
            ],
            v_rolled,
        )
    });

    cache.local_end_index = cache.local_end_index.saturating_sub(num_evicted);
}

/// Evict and roll the streaming MQA cache window left by `num_evicted` tokens on dim=1.
/// Preserves the first `sink_tokens` entries.
pub fn evict_and_roll_mqa<B: Backend>(
    cache: &mut crate::attention::StreamingMqaCache<B>,
    batch_size: usize,
    kv_heads: usize,
    d_k: usize,
    sink_tokens: usize,
    num_evicted: usize,
) {
    let avail = cache.local_end_index.saturating_sub(sink_tokens);
    if avail == 0 || num_evicted >= avail {
        cache.local_end_index = cache.local_end_index.saturating_sub(num_evicted);
        return;
    }

    let k_sub = cache.k.clone().slice([
        0..batch_size,
        sink_tokens..cache.local_end_index,
        0..kv_heads,
        0..d_k,
    ]);
    let k_rolled = k_sub.roll(&[-(num_evicted as i64)], &[1]);
    cache.k.inplace(|t| {
        t.slice_assign(
            [
                0..batch_size,
                sink_tokens..sink_tokens + avail,
                0..kv_heads,
                0..d_k,
            ],
            k_rolled,
        )
    });

    let v_sub = cache.v.clone().slice([
        0..batch_size,
        sink_tokens..cache.local_end_index,
        0..kv_heads,
        0..d_k,
    ]);
    let v_rolled = v_sub.roll(&[-(num_evicted as i64)], &[1]);
    cache.v.inplace(|t| {
        t.slice_assign(
            [
                0..batch_size,
                sink_tokens..sink_tokens + avail,
                0..kv_heads,
                0..d_k,
            ],
            v_rolled,
        )
    });

    cache.local_end_index = cache.local_end_index.saturating_sub(num_evicted);
}
