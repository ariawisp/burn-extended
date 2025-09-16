use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Shape, Tensor};
use burn_extended::attention::{
    AttnWindow, ExtStreamingMultiHeadAttentionConfig, ExtStreamingParams, StreamingMhaCache,
};
use burn_ndarray::NdArray;

type TB = NdArray<f32>;

fn device() -> <TB as Backend>::Device {
    Default::default()
}

#[test]
fn ext_streaming_window_matches_full() {
    let device = device();
    let b = 2;
    let t = 12;
    let d_model = 32;
    let n_heads = 4;

    let x = Tensor::<TB, 3>::random([b, t, d_model], Distribution::Default, &device);

    let attn = ExtStreamingMultiHeadAttentionConfig::new(d_model, n_heads)
        .with_dropout(0.0)
        .init::<TB>(&device);

    let mut cache_full = StreamingMhaCache::new(&device, b, 64, n_heads, d_model / n_heads, 0);
    let mut cache_window = StreamingMhaCache::new(&device, b, 64, n_heads, d_model / n_heads, 0);

    let out_full = attn.forward_streaming(
        x.clone(),
        &mut cache_full,
        ExtStreamingParams {
            rope: None,
            start_pos: 0,
            window: AttnWindow::Full,
            attn_bias: None,
        },
    );

    let out_window = attn.forward_streaming(
        x,
        &mut cache_window,
        ExtStreamingParams {
            rope: None,
            start_pos: 0,
            window: AttnWindow::Window(t),
            attn_bias: None,
        },
    );

    assert_eq!(out_full.shape(), Shape::new([b, t, d_model]));
    // For identical inputs and no bias, window=t matches full.
    let diff = (out_full - out_window)
        .abs()
        .sum()
        .into_data()
        .to_vec::<f32>()
        .unwrap()[0];
    assert!(diff < 1e-3);
}

#[test]
fn ext_streaming_attn_bias_affects_output() {
    let device = device();
    let b = 1;
    let t = 8;
    let d_model = 16;
    let n_heads = 4;

    let x = Tensor::<TB, 3>::random([b, t, d_model], Distribution::Default, &device);

    let attn = ExtStreamingMultiHeadAttentionConfig::new(d_model, n_heads)
        .with_dropout(0.0)
        .init::<TB>(&device);

    let mut cache_a = StreamingMhaCache::new(&device, b, 32, n_heads, d_model / n_heads, 0);
    let mut cache_b = StreamingMhaCache::new(&device, b, 32, n_heads, d_model / n_heads, 0);

    // Zero bias baseline
    let out_a = attn.forward_streaming(
        x.clone(),
        &mut cache_a,
        ExtStreamingParams {
            rope: None,
            start_pos: 0,
            window: AttnWindow::Window(t),
            attn_bias: None,
        },
    );

    // Positive bias on last key position for all heads: shape [B, H, Tq, Tk]
    let mut bias = Tensor::<TB, 4>::zeros([b, n_heads, t, t], &device);
    let large = Tensor::<TB, 4>::full([b, n_heads, t, 1], 5.0, &device);
    bias = bias.slice_assign([0..b, 0..n_heads, 0..t, t - 1..t], large);

    let out_b = attn.forward_streaming(
        x,
        &mut cache_b,
        ExtStreamingParams {
            rope: None,
            start_pos: 0,
            window: AttnWindow::Window(t),
            attn_bias: Some(&bias),
        },
    );

    // With a strong bias, outputs should differ
    let diff = (out_a - out_b)
        .abs()
        .sum()
        .into_data()
        .to_vec::<f32>()
        .unwrap()[0];
    assert!(diff > 1e-4);
}

#[test]
fn ext_streaming_quiet_softmax_executes() {
    let device = device();
    let b = 1;
    let t = 8;
    let d_model = 16;
    let n_heads = 4;

    let x = Tensor::<TB, 3>::random([b, t, d_model], Distribution::Default, &device);
    let attn = ExtStreamingMultiHeadAttentionConfig::new(d_model, n_heads)
        .with_dropout(0.0)
        .with_quiet_softmax(true)
        .init::<TB>(&device);
    let mut cache = StreamingMhaCache::new(&device, b, 32, n_heads, d_model / n_heads, 0);
    let out = attn.forward_streaming(
        x,
        &mut cache,
        ExtStreamingParams {
            rope: None,
            start_pos: 0,
            window: AttnWindow::Window(t),
            attn_bias: None,
        },
    );
    assert_eq!(out.dims(), [b, t, d_model]);
}
