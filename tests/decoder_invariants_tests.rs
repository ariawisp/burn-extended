use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn_extended::decoder::{DecoderBlockConfig, DecoderBlockInput};
use burn_ndarray::NdArray;

type TB = NdArray<f32>;

fn device() -> <TB as Backend>::Device {
    Default::default()
}

#[test]
fn decoder_zero_input_is_finite_and_shaped() {
    let device = device();
    let config = DecoderBlockConfig::new(32, 4, 4, 64).with_dropout(0.0).with_swiglu_limit(5.0);
    let block = config.init::<TB>(&device);

    let hidden = Tensor::<TB, 3>::zeros([2, 8, 32], &device);
    let output = block.forward(DecoderBlockInput::new(hidden)).hidden;
    assert_eq!(output.dims(), [2, 8, 32]);
    let data = output.into_data().to_vec::<f32>().unwrap();
    assert!(data.iter().all(|x| x.is_finite()));
}

#[test]
fn decoder_accepts_masks_and_bias() {
    let device = device();
    let config = DecoderBlockConfig::new(16, 2, 2, 32).with_dropout(0.0).with_swiglu_limit(5.0);
    let block = config.init::<TB>(&device);

    let b = 2;
    let t = 6;
    let hidden = Tensor::<TB, 3>::random([b, t, 16], burn::tensor::Distribution::Default, &device);
    // Pad mask [B, T]
    let mask_pad = burn_extended::attention::lengths_to_mask::<TB>(&[t, t / 2], t, &device);
    // Attn mask [B, T, T]
    let mask_attn = burn_extended::attention::generate_windowed_causal_mask::<TB>(b, t, Some(3), 1, &device);
    // Bias [B, nH=2, T, Tk] -- use Tk=T here
    let attn_bias = Tensor::<TB, 4>::zeros([b, 2, t, t], &device);

    let output = block
        .forward(
            DecoderBlockInput::new(hidden)
                .mask_pad(mask_pad)
                .mask_attn(mask_attn)
                .attn_bias(attn_bias),
        )
        .hidden;
    assert_eq!(output.dims(), [b, t, 16]);
}
