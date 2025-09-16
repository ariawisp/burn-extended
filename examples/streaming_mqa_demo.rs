use burn_ndarray::NdArray as B;
use burn_tensor::backend::Backend;
use burn_tensor::{Distribution, Tensor};

use burn_extended::attention::{StreamingMultiQueryAttentionConfig, StreamingParams};
use burn_extended::cache::{MqaCacheManager, WindowPolicy};

fn main() {
    let device = <B as Backend>::Device::default();

    let b = 1usize;
    let t = 32usize;
    let d_model = 64usize;
    let n_heads = 4usize;
    let kv_heads = 2usize;
    let head_dim = d_model / n_heads;
    let num_layers = 2usize;
    let chunk = 8usize;
    let cache_len = 64usize;
    let sink_tokens = 2usize;

    let layer_cfg =
        StreamingMultiQueryAttentionConfig::new(d_model, n_heads, kv_heads).with_dropout(0.0);
    let layers: Vec<_> = (0..num_layers)
        .map(|_| layer_cfg.clone().init::<B>(&device))
        .collect();

    let mut caches = MqaCacheManager::new(
        &device,
        num_layers,
        kv_heads,
        head_dim,
        cache_len,
        sink_tokens,
        b,
    );
    let policy = WindowPolicy::EveryOther {
        window: 16,
        full_on_even: true,
    };

    let x = Tensor::<B, 3>::random([b, t, d_model], Distribution::Default, &device);

    let mut outputs = Vec::new();
    for i in 0..(t / chunk) {
        let start = i * chunk;
        let mut h = x.clone().slice([0..b, start..start + chunk, 0..d_model]);
        for l in 0..num_layers {
            let window = policy.window_for(l);
            let params = StreamingParams {
                rope: None,
                start_pos: start,
                window,
            };
            let y = layers[l].forward_streaming(h, caches.cache_mut(l), params.into());
            h = y;
        }
        outputs.push(h);
    }

    let y = Tensor::cat(outputs, 1);
    println!("streaming_mqa_demo output dims: {:?}", y.dims());
}
