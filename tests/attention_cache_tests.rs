use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn_extended::attention::{
    evict_and_roll_mha, evict_and_roll_mqa, StreamingMhaCache, StreamingMqaCache,
};
use burn_ndarray::NdArray;

type TB = NdArray<f32>;

fn device() -> <TB as Backend>::Device {
    Default::default()
}

#[test]
fn evict_and_roll_mha_basic() {
    let device = device();
    let batch = 1;
    let cap = 6;
    let n_heads = 1;
    let d_k = 1;
    let sink_tokens = 2;

    let mut cache = StreamingMhaCache::new(&device, batch, cap, n_heads, d_k, sink_tokens);

    // Fill K/V with identifiable values along the time dimension for batch=1, head=1, dim=1.
    // We set K[b=0, t, h=0, d=0] = t as f32
    let k_vals: Vec<f32> = (0..cap).map(|i| i as f32).collect();
    let v_vals: Vec<f32> = (0..cap).map(|i| (100 + i) as f32).collect();
    let k_tensor = Tensor::<TB, 1>::from_floats(k_vals.as_slice(), &device).reshape([1, cap, 1, 1]);
    let v_tensor = Tensor::<TB, 1>::from_floats(v_vals.as_slice(), &device).reshape([1, cap, 1, 1]);
    cache.k = k_tensor;
    cache.v = v_tensor;
    cache.local_end_index = cap; // buffer full

    // Evict 2 tokens; expected avail = 4, after rolling left by 2 with wrap: [0,1,4,5,2,3]
    evict_and_roll_mha(&mut cache, batch, n_heads, d_k, sink_tokens, 2);

    assert_eq!(cache.local_end_index, 4);
    let k_after = cache.k.clone().into_data();
    // Only first local_end_index entries are considered active; check they are [0,1,4,5]
    let flat = k_after.to_vec::<f32>().unwrap();
    // Layout is [B, T, H, D] with H=D=1. Step by 1 over T.
    assert_eq!(flat[0], 0.0);
    assert_eq!(flat[1], 1.0);
    assert_eq!(flat[2], 4.0);
    assert_eq!(flat[3], 5.0);
}

#[test]
fn evict_and_roll_mqa_basic() {
    let device = device();
    let batch = 1;
    let cap = 6;
    let kv_heads = 2;
    let d_k = 1;
    let sink_tokens = 1;

    let mut cache = StreamingMqaCache::new(&device, batch, cap, kv_heads, d_k, sink_tokens);

    // For kv_heads=2, assign time index as floats duplicated across heads for clarity
    let k_vals: Vec<f32> = (0..cap * kv_heads).map(|i| (i / kv_heads) as f32).collect();
    let v_vals: Vec<f32> = (0..cap * kv_heads)
        .map(|i| (200 + (i / kv_heads)) as f32)
        .collect();
    let k_tensor =
        Tensor::<TB, 1>::from_floats(k_vals.as_slice(), &device).reshape([1, cap, kv_heads, 1]);
    let v_tensor =
        Tensor::<TB, 1>::from_floats(v_vals.as_slice(), &device).reshape([1, cap, kv_heads, 1]);
    cache.k = k_tensor;
    cache.v = v_tensor;
    cache.local_end_index = cap;

    // Evict 3 tokens; avail = 5, roll left by 3.
    evict_and_roll_mqa(&mut cache, batch, kv_heads, d_k, sink_tokens, 3);
    assert_eq!(cache.local_end_index, cap.saturating_sub(3));

    // Validate the first sink token is preserved at t=0
    let k_after = cache.k.clone().into_data();
    // entry at t=0 for any head should remain 0.0
    let flat = k_after.to_vec::<f32>().unwrap();
    assert_eq!(flat[0], 0.0);
    assert_eq!(flat[1], 0.0);
}
