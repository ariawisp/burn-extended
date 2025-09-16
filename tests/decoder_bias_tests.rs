use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor};
use burn_ndarray::NdArray;

use burn_extended::decoder::{DecoderBlockConfig, DecoderBlockInput};

type TB = NdArray<f32>;

fn device() -> <TB as Backend>::Device {
    Default::default()
}

#[test]
fn decoder_attn_bias_changes_output() {
    let device = device();
    let d_model = 16;
    let n_heads = 2;
    let t = 6;
    let b = 1;

    let config = DecoderBlockConfig::new(d_model, n_heads, n_heads, 32)
        .with_dropout(0.0)
        .with_swiglu_limit(5.0);
    let block = config.init::<TB>(&device);

    let hidden = Tensor::<TB, 3>::random([b, t, d_model], Distribution::Default, &device);

    let out_a = block.forward(DecoderBlockInput::new(hidden.clone())).hidden;

    // Positive bias on the last key position for all heads
    let mut bias = Tensor::<TB, 4>::zeros([b, n_heads, t, t], &device);
    let large = Tensor::<TB, 4>::full([b, n_heads, t, 1], 4.0, &device);
    bias = bias.slice_assign([0..b, 0..n_heads, 0..t, t - 1..t], large);

    let out_b = block
        .forward(DecoderBlockInput::new(hidden).attn_bias(bias))
        .hidden;

    // Outputs should differ when bias is applied
    let diff = (out_a - out_b)
        .abs()
        .sum()
        .into_data()
        .to_vec::<f32>()
        .unwrap()[0];
    assert!(diff > 1e-4);
}
