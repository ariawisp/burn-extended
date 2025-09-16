#![recursion_limit = "256"]

use std::path::PathBuf;

use burn::backend::wgpu::{self, Wgpu as B, WgpuDevice};
use burn::tensor::Tensor;
use burn_extended::generate::{generate, AutoregressiveModel, GenerationConfig};
use burn_extended::loader::{build_gptoss_qkv_splits, load_gptoss_sinks};
use burn_extended::models::gpt_oss::{GptOssConfig};

fn main() {
    let device = WgpuDevice::default();
    wgpu::init_setup::<wgpu::graphics::Metal>(&device, Default::default());

    // Tiny demo config so it runs on CPU/GPU easily; replace with real sizes.
    let cfg = GptOssConfig {
        vocab_size: 32000,
        d_model: 256,
        n_layers: 4,
        n_heads: 8,
        kv_heads: 2,
        ffn_hidden: 1024,
        dropout: 0.0,
        swiglu_alpha: 1.0,
        swiglu_limit: 7.0,
        initializer: burn::nn::Initializer::KaimingUniform {
            gain: 1.0 / num_traits::Float::sqrt(3.0),
            fan_out_only: false,
        },
        cache_len: 1024,
        sink_tokens: 0,
        window_policy: burn_extended::cache::WindowPolicy::EveryOther { window: 256, full_on_even: true },
        max_seq_len: 2048,
        learned_sinks: true,
        use_ntk_yarn: false,
        rope_scaling_factor: 32.0,
        rope_initial_context: 4096.0,
        rope_ntk_alpha: 1.0,
        rope_ntk_beta: 32.0,
    };
    let mut model = cfg.init::<B>(&device);

    // Optionally load a checkpoint if provided via env var `GPT_OSS_CKPT`.
    if let Ok(p) = std::env::var("GPT_OSS_CKPT") {
        let path = PathBuf::from(p);
        if path.exists() {
            let head_dim = cfg.d_model / cfg.n_heads;
            // Load fused QKV
            let splits = build_gptoss_qkv_splits(cfg.n_layers, cfg.n_heads, cfg.kv_heads, head_dim);
            let _ = burn_extended::loader::load_safetensors_qkv_split::<B, _>(
                &mut model,
                &path,
                &splits,
                /*from_pytorch*/ true,
                /*allow_partial*/ true,
                /*validate*/ false,
            ).expect("qkv load");
            // Load sinks (reshape [n_heads] -> [kv_heads, groups])
            let _ = load_gptoss_sinks::<B, _>(
                &mut model,
                &path,
                cfg.n_layers,
                cfg.n_heads,
                cfg.kv_heads,
                /*from_pytorch*/ true,
                /*allow_partial*/ true,
                /*validate*/ false,
            ).expect("sinks load");
        } else {
            eprintln!("GPT_OSS_CKPT not found at {:?}; running with random weights", path);
        }
    } else {
        eprintln!("GPT_OSS_CKPT env not set; running with random weights");
    }

    // Dummy prompt (token IDs). In real use, tokenize with Harmony-compatible tokenizer.
    let prompts = vec![vec![1usize, 2, 3, 4, 5]];
    let gen_cfg = GenerationConfig {
        max_new_tokens: 16,
        eos_token: None,
        sampler: burn_extended::sampling::SamplerConfig { temperature: 0.8, top_k: Some(50), ..Default::default() },
        window: burn_extended::attention::AttnWindow::Window(256),
    };
    let outputs = generate::<B, _>(&model, &device, &prompts, gen_cfg);
    println!("gpt-oss infer (demo) output lens: {:?}", outputs.iter().map(|t| t.len()).collect::<Vec<_>>());
}

