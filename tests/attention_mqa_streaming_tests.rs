use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Shape, Tensor};
use burn_extended::attention::{
    AttnWindow, StreamingMqaCache, StreamingMqaParams, StreamingMultiQueryAttentionConfig,
};
use burn_ndarray::NdArray;
use burn_tensor::{ops::FloatElem, Tolerance};

type TB = NdArray<f32>;

fn device() -> <TB as Backend>::Device {
    Default::default()
}

#[test]
fn streaming_mqa_window_matches_full() {
    let device = device();
    let b = 2;
    let t = 12;
    let d_model = 32;
    let n_heads = 4;
    let kv_heads = 2;

    let x = Tensor::<TB, 3>::random([b, t, d_model], Distribution::Default, &device);

    let attn = StreamingMultiQueryAttentionConfig::new(d_model, n_heads, kv_heads)
        .with_dropout(0.0)
        .init::<TB>(&device);

    let mut cache_full = StreamingMqaCache::new(&device, b, 64, kv_heads, d_model / n_heads, 0);
    let mut cache_window = StreamingMqaCache::new(&device, b, 64, kv_heads, d_model / n_heads, 0);

    let out_full = attn.forward_streaming(
        x.clone(),
        &mut cache_full,
        StreamingMqaParams {
            rope: None,
            start_pos: 0,
            window: AttnWindow::Full,
            sinks: None,
            attn_bias: None,
        },
    );

    let out_window = attn.forward_streaming(
        x,
        &mut cache_window,
        StreamingMqaParams {
            rope: None,
            start_pos: 0,
            window: AttnWindow::Window(t),
            sinks: None,
            attn_bias: None,
        },
    );

    assert_eq!(out_full.shape(), Shape::new([b, t, d_model]));
    out_full
        .into_data()
        .assert_approx_eq::<FloatElem<TB>>(&out_window.into_data(), Tolerance::default());
}

#[test]
fn streaming_mqa_with_chunks_consistent() {
    let device = device();
    let b = 1;
    let t = 16;
    let d_model = 32;
    let n_heads = 4;
    let kv_heads = 2;

    let x = Tensor::<TB, 3>::random([b, t, d_model], Distribution::Default, &device);

    let attn = StreamingMultiQueryAttentionConfig::new(d_model, n_heads, kv_heads)
        .with_dropout(0.0)
        .init::<TB>(&device);

    let mut cache_full = StreamingMqaCache::new(&device, b, 64, kv_heads, d_model / n_heads, 0);
    let full = attn.forward_streaming(
        x.clone(),
        &mut cache_full,
        StreamingMqaParams {
            rope: None,
            start_pos: 0,
            window: AttnWindow::Window(t),
            sinks: None,
            attn_bias: None,
        },
    );

    let mut cache_chunked = StreamingMqaCache::new(&device, b, 64, kv_heads, d_model / n_heads, 0);
    let chunk = 4;
    let mut outputs = Vec::new();
    for i in 0..(t / chunk) {
        let start = i * chunk;
        let slice = x.clone().slice([0..b, start..start + chunk, 0..d_model]);
        let out = attn.forward_streaming(
            slice,
            &mut cache_chunked,
            StreamingMqaParams {
                rope: None,
                start_pos: start,
                window: AttnWindow::Window(t),
                sinks: None,
                attn_bias: None,
            },
        );
        outputs.push(out);
    }
    let chunked = Tensor::cat(outputs, 1);

    full.into_data()
        .assert_approx_eq::<FloatElem<TB>>(&chunked.into_data(), Tolerance::rel_abs(0.5, 0.2));
}

#[test]
fn streaming_mqa_quiet_softmax_executes() {
    // This test just exercises the quiet_softmax branch to ensure it runs.
    let device = device();
    let b = 1;
    let t = 8;
    let d_model = 16;
    let n_heads = 4;
    let kv_heads = 2;

    let x = Tensor::<TB, 3>::random([b, t, d_model], Distribution::Default, &device);
    let attn = StreamingMultiQueryAttentionConfig::new(d_model, n_heads, kv_heads)
        .with_dropout(0.0)
        .with_quiet_softmax(true)
        .init::<TB>(&device);
    let mut cache = StreamingMqaCache::new(&device, b, 32, kv_heads, d_model / n_heads, 0);
    let out = attn.forward_streaming(
        x,
        &mut cache,
        StreamingMqaParams {
            rope: None,
            start_pos: 0,
            window: AttnWindow::Window(t),
            sinks: None,
            attn_bias: None,
        },
    );
    assert_eq!(out.dims(), [b, t, d_model]);
}

#[test]
fn streaming_mqa_uses_module_sinks_when_set() {
    let device = device();
    let b = 1;
    let t = 8;
    let d_model = 16;
    let n_heads = 4;
    let kv_heads = 2;
    let groups = n_heads / kv_heads;

    let mut attn = StreamingMultiQueryAttentionConfig::new(d_model, n_heads, kv_heads)
        .with_dropout(0.0)
        .init::<TB>(&device);
    // Set a simple sinks vector [n_heads]
    let sinks = Tensor::<TB, 1>::from_floats([0.0; 4], &device);
    attn.sinks_weight = Some(sinks);

    let mut cache = StreamingMqaCache::new(&device, b, 32, kv_heads, d_model / n_heads, 0);
    let x = Tensor::<TB, 3>::random([b, t, d_model], Distribution::Default, &device);
    let out = attn.forward_streaming(
        x,
        &mut cache,
        StreamingMqaParams {
            rope: None,
            start_pos: 0,
            window: AttnWindow::Window(t),
            sinks: None,
            attn_bias: None,
        },
    );
    assert_eq!(out.dims(), [b, t, d_model]);
}
