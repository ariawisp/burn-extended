use burn::backend::Backend;
use burn::tensor::{Distribution, Tensor};
use burn_ndarray::NdArray;

use burn_extended::decoder::{DecoderBlockConfig, DecoderBlockInput};

type TB = NdArray<f32>;

fn device() -> <TB as Backend>::Device {
    Default::default()
}

#[test]
fn decoder_block_forward_preserves_shape() {
    let device = device();
    let config = DecoderBlockConfig::new(32, 4, 4, 64)
        .with_dropout(0.0)
        .with_swiglu_limit(5.0);
    let block = config.init::<TB>(&device);

    let hidden = Tensor::<TB, 3>::random([2, 8, 32], Distribution::Default, &device);
    let output = block.forward(DecoderBlockInput::new(hidden)).hidden;
    assert_eq!(output.dims(), [2, 8, 32]);
}
