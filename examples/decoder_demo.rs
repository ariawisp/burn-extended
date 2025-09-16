use burn_ndarray::NdArray as B;
use burn_tensor::backend::Backend;
use burn_tensor::{Distribution, Tensor};

use burn_extended::attention::generate_windowed_causal_mask;
use burn_extended::decoder::{DecoderBlockConfig, DecoderBlockInput};

fn main() {
    let device = <B as Backend>::Device::default();

    let d_model = 64usize;
    let n_heads = 4usize;
    let kv_heads = 4usize;
    let ffn_hidden = 128usize;
    let b = 2usize;
    let t = 12usize;

    let block = DecoderBlockConfig::new(d_model, n_heads, kv_heads, ffn_hidden)
        .with_dropout(0.0)
        .with_swiglu_limit(5.0)
        .init::<B>(&device);

    let hidden = Tensor::<B, 3>::random([b, t, d_model], Distribution::Default, &device);
    let mask_attn = generate_windowed_causal_mask::<B>(b, t, Some(6), 1, &device);

    // Optional: additive attention bias (zeros here for demo)
    let attn_bias = Tensor::<B, 4>::zeros([b, n_heads, t, t], &device);

    let output = block
        .forward(
            DecoderBlockInput::new(hidden)
                .mask_attn(mask_attn)
                .attn_bias(attn_bias),
        )
        .hidden;

    println!("decoder_demo output dims: {:?}", output.dims());
}
