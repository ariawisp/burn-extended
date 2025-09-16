use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor};
use burn_ndarray::NdArray;

use burn_extended::attention::{lengths_to_mask, LinearAttentionConfig, LinearAttnInput};

type TB = NdArray<f32>;

fn device() -> <TB as Backend>::Device {
    Default::default()
}

#[test]
fn linear_attention_forward_shapes() {
    let device = device();
    let d_model = 32;
    let n_heads = 4;
    let attn = LinearAttentionConfig::new(d_model, n_heads).init::<TB>(&device);
    let x = Tensor::<TB, 3>::random([2, 8, d_model], Distribution::Default, &device);
    let out = attn.forward(LinearAttnInput::self_attn(x)).context;
    assert_eq!(out.dims(), [2, 8, d_model]);
}

#[test]
fn linear_attention_respects_pad_mask_shape() {
    let device = device();
    let d_model = 16;
    let n_heads = 2;
    let attn = LinearAttentionConfig::new(d_model, n_heads).init::<TB>(&device);
    let x = Tensor::<TB, 3>::random([1, 6, d_model], Distribution::Default, &device);
    let mask = lengths_to_mask::<TB>(&[3], 6, &device);
    let out = attn
        .forward(LinearAttnInput::self_attn(x).mask_pad(mask))
        .context;
    assert_eq!(out.dims(), [1, 6, d_model]);
}
