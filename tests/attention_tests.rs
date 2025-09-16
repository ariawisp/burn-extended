use burn::nn::RotaryEncodingConfig;
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Shape, Tensor, TensorData};
use burn_extended::attention::{
    generate_causal_mask_1d, generate_chunked_causal_mask_1d, generate_windowed_causal_mask,
    lengths_to_mask, AttnWindow, StreamingMhaCache, StreamingMultiHeadAttentionConfig,
    StreamingParams,
};
use burn_ndarray::NdArray;
use burn_tensor::{ops::FloatElem, Tolerance};

type TB = NdArray<f32>;

fn device() -> <TB as Backend>::Device {
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
    out_full
        .into_data()
        .assert_approx_eq::<FloatElem<TB>>(&out_window.into_data(), Tolerance::default());
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

    full.into_data()
        .assert_approx_eq::<FloatElem<TB>>(&chunked.into_data(), Tolerance::rel_abs(0.5, 0.2));
}

#[test]
fn mask_helpers_behave_expected() {
    let device = device();
    let mask_lengths = lengths_to_mask::<TB>(&[3, 1], 5, &device).into_data();
    mask_lengths.assert_eq(
        &TensorData::from([
            [false, false, false, true, true],
            [false, true, true, true, true],
        ]),
        false,
    );

    let causal = generate_causal_mask_1d::<TB>(4, &device).into_data();
    causal.assert_eq(
        &TensorData::from([
            [false, true, true, true],
            [false, false, true, true],
            [false, false, false, true],
            [false, false, false, false],
        ]),
        false,
    );

    let windowed = generate_windowed_causal_mask::<TB>(1, 6, Some(2), 1, &device)
        .squeeze::<2>(0)
        .into_data();
    let flat = windowed.to_vec::<bool>().expect("to_vec bool");
    let n = 6usize;
    let get = |i: usize, j: usize| -> bool { flat[i * n + j] };
    // Basic properties for window=2 causal mask.
    assert_eq!(get(0, 0), false);
    for j in 1..n {
        assert!(get(0, j));
    }
    assert_eq!(get(3, 0), true);
    assert_eq!(get(3, 1), false);
    assert_eq!(get(3, 2), false);
    assert_eq!(get(3, 3), false);
    assert_eq!(get(3, 4), true);
    assert_eq!(get(3, 5), true);
    assert_eq!(get(5, 2), true);
    assert_eq!(get(5, 3), false);
    assert_eq!(get(5, 4), false);
    assert_eq!(get(5, 5), false);

    let chunked = generate_chunked_causal_mask_1d::<TB>(8, 2, 1, &device).into_data();
    // Flatten to a Vec<bool> and slice out the 4th row (index 3)
    let flat = chunked.to_vec::<bool>().expect("bool vec conversion");
    let row_len = chunked.shape[1];
    let start = 3 * row_len;
    let row3: Vec<bool> = flat[start..start + row_len].to_vec();
    assert!(row3[..4].iter().all(|v| !*v));
    assert!(row3[4..].iter().all(|v| *v));
}
