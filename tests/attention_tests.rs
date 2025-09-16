use burn::backend::Backend;
use burn_extended::attention::{
    generate_chunked_causal_mask_1d,
    generate_causal_mask_1d,
    generate_windowed_causal_mask,
    lengths_to_mask,
    AttnWindow,
    StreamingMhaCache,
    StreamingMultiHeadAttentionConfig,
    StreamingParams,
};
use burn::nn::rope_encoding::RotaryEncodingConfig;
use burn::tensor::{Distribution, Shape, Tensor};
use burn_ndarray::NdArray;
use burn_tensor::{ops::FloatElem, Tolerance};

type TB = NdArray<f32>;

fn device() -> <TB as burn::backend::Backend>::Device {
    Default::default()
}

#[test]
fn streaming_window_matches_full() {
    let device = device();
    let b = 2;
    let t = 12;
    let d_model = 32;
    let n_heads = 4;

    let x = Tensor::<TB, 3>::random([b, t, d_model], Distribution::Default, &device);

    let attn = StreamingMultiHeadAttentionConfig::new(d_model, n_heads)
        .with_dropout(0.0)
        .init::<TB>(&device);

    let mut cache_full = StreamingMhaCache::new(&device, b, 64, n_heads, d_model / n_heads, 0);
    let mut cache_window = StreamingMhaCache::new(&device, b, 64, n_heads, d_model / n_heads, 0);

    let out_full = attn.forward_streaming(
        x.clone(),
        &mut cache_full,
        StreamingParams {
            rope: None,
            start_pos: 0,
            window: AttnWindow::Full,
        },
    );

    let out_window = attn.forward_streaming(
        x,
        &mut cache_window,
        StreamingParams {
            rope: None,
            start_pos: 0,
            window: AttnWindow::Window(t),
        },
    );

    assert_eq!(out_full.shape(), Shape::new([b, t, d_model]));
    out_full.into_data().assert_approx_eq::<FloatElem<TB>>(
        &out_window.into_data(),
        Tolerance::default(),
    );
}

#[test]
fn streaming_with_rope_is_consistent_across_chunks() {
    let device = device();
    let b = 1;
    let t = 16;
    let d_model = 32;
    let n_heads = 4;
    let head_dim = d_model / n_heads;

    let x = Tensor::<TB, 3>::random([b, t, d_model], Distribution::Default, &device);
    let rope = RotaryEncodingConfig::new(512, head_dim).init::<TB>(&device);

    let attn = StreamingMultiHeadAttentionConfig::new(d_model, n_heads)
        .with_dropout(0.0)
        .init::<TB>(&device);

    let mut cache_full = StreamingMhaCache::new(&device, b, 64, n_heads, head_dim, 0);
    let full = attn.forward_streaming(
        x.clone(),
        &mut cache_full,
        StreamingParams {
            rope: Some(&rope),
            start_pos: 0,
            window: AttnWindow::Window(t),
        },
    );

    let mut cache_chunked = StreamingMhaCache::new(&device, b, 64, n_heads, head_dim, 0);
    let chunk = 4;
    let mut outputs = Vec::new();
    for i in 0..(t / chunk) {
        let start = i * chunk;
        let slice = x.clone().slice([0..b, start..start + chunk, 0..d_model]);
        let out = attn.forward_streaming(
            slice,
            &mut cache_chunked,
            StreamingParams {
                rope: Some(&rope),
                start_pos: start,
                window: AttnWindow::Window(t),
            },
        );
        outputs.push(out);
    }
    let chunked = Tensor::cat(outputs, 1);

    full.into_data().assert_approx_eq::<FloatElem<TB>>(
        &chunked.into_data(),
        Tolerance::rel_abs(0.5, 0.2),
    );
}

#[test]
fn mask_helpers_behave_expected() {
    let device = device();
    let mask_lengths = lengths_to_mask::<TB>(&[3, 1], 5, &device).into_data();
    mask_lengths.assert_eq(
        &burn::tensor::TensorData::from([
            [false, false, false, true, true],
            [false, true, true, true, true],
        ]),
        false,
    );

    let causal = generate_causal_mask_1d::<TB>(4, &device).into_data();
    causal.assert_eq(
        &burn::tensor::TensorData::from([
            [false, true, true, true],
            [false, false, true, true],
            [false, false, false, true],
            [false, false, false, false],
        ]),
        false,
    );

    let windowed = generate_windowed_causal_mask::<TB>(1, 6, Some(2), 1, &device)
        .squeeze(0)
        .into_data();
    windowed.assert_eq(
        &burn::tensor::TensorData::from([
            [false, true, true, true, true, true],
            [false, false, true, true, true, true],
            [false, false, false, true, true, true],
            [false, false, false, false, true, true],
            [false, false, false, false, false, true],
            [false, false, false, false, false, false],
        ]),
        false,
    );

    let chunked = generate_chunked_causal_mask_1d::<TB>(8, 2, 1, &device).into_data();
    let row3: Vec<bool> = chunked.convert().value[3].clone();
    assert!(row3[..4].iter().all(|v| !*v));
    assert!(row3[4..].iter().all(|v| *v));
}
