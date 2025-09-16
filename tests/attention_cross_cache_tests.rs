use burn::tensor::backend::Backend;
use burn_ndarray::NdArray;

use burn_extended::attention::CrossAttnCache;

type TB = NdArray<f32>;

fn device() -> <TB as Backend>::Device {
    Default::default()
}

#[test]
fn cross_attn_cache_init_and_clear() {
    let device = device();
    let mut cache = CrossAttnCache::<TB>::new(&device, 2, 5, 3, 4);
    assert!(!cache.is_init);
    let k = cache.k.clone();
    let v = cache.v.clone();
    cache.set(k, v);
    assert!(cache.is_init);
    cache.clear();
    assert!(!cache.is_init);
}
