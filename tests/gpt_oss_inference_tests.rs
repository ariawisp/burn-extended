use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use burn_ndarray::NdArray;

use burn_extended::attention::AttnWindow;
use burn_extended::generate::{generate, AutoregressiveModel, GenerationConfig};
use burn_extended::models::gpt_oss::GptOssConfig;

type TB = NdArray<f32>;

fn device() -> <TB as Backend>::Device {
    Default::default()
}

#[test]
fn gpt_oss_ar_model_basic_generation() {
    let device = device();
    let cfg = GptOssConfig {
        vocab_size: 100,
        d_model: 64,
        n_layers: 2,
        n_heads: 4,
        kv_heads: 2,
        ffn_hidden: 128,
        dropout: 0.0,
        swiglu_alpha: 1.0,
        swiglu_limit: 7.0,
        initializer: burn::nn::Initializer::KaimingUniform {
            gain: 1.0 / num_traits::Float::sqrt(3.0),
            fan_out_only: false,
        },
        cache_len: 128,
        sink_tokens: 0,
        window_policy: burn_extended::cache::WindowPolicy::Fixed(64),
        max_seq_len: 128,
    };
    let model = cfg.init::<TB>(&device);

    // Use generate on a tiny prompt
    let prompts = vec![vec![1usize, 2, 3]];
    let gen_cfg = GenerationConfig {
        max_new_tokens: 4,
        eos_token: None,
        sampler: Default::default(),
        window: AttnWindow::Window(64),
    };
    let out = generate(&model, &device, &prompts, gen_cfg);
    assert_eq!(out.len(), 1);
    assert!(out[0].len() >= 3);
}
