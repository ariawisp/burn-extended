use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor};
use burn_ndarray::NdArray;

use burn_extended::attention::{
    AttnWindow, ExtStreamingMultiHeadAttentionConfig, StreamingMhaCache, StreamingMqaCache,
    StreamingMultiQueryAttentionConfig, StreamingParams,
};

type TB = NdArray<f32>;

fn device() -> <TB as Backend>::Device {
    Default::default()
}

#[test]
fn params_into_mqa_ext_compiles_and_runs() {
    let device = device();
    let b = 1;
    let t = 8;
    let d_model = 16;
    let n_heads = 4;
    let kv_heads = 2;
    let head_dim = d_model / n_heads;

    let x = Tensor::<TB, 3>::random([b, t, d_model], Distribution::Default, &device);

    // MQA path using StreamingParams -> StreamingMqaParams via Into
    let mqa = StreamingMultiQueryAttentionConfig::new(d_model, n_heads, kv_heads)
        .with_dropout(0.0)
        .init::<TB>(&device);
    let mut mqa_cache = StreamingMqaCache::new(&device, b, 32, kv_heads, head_dim, 0);
    let base = StreamingParams {
        rope: None,
        start_pos: 0,
        window: AttnWindow::Window(t),
    };
    let _ = mqa.forward_streaming(x.clone(), &mut mqa_cache, base.into());

    // Ext MHA path using StreamingParams -> ExtStreamingParams via Into
    let ext = ExtStreamingMultiHeadAttentionConfig::new(d_model, n_heads)
        .with_dropout(0.0)
        .init::<TB>(&device);
    let mut mha_cache = StreamingMhaCache::new(&device, b, 32, n_heads, head_dim, 0);
    let base = StreamingParams {
        rope: None,
        start_pos: 0,
        window: AttnWindow::Window(t),
    };
    let _ = ext.forward_streaming(x, &mut mha_cache, base.into());
}
