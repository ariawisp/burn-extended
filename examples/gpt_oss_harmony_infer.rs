#![recursion_limit = "256"]

use std::path::PathBuf;

use anyhow::Context;
use burn::backend::wgpu::{self, Wgpu as B, WgpuDevice};
use burn_extended::attention::AttnWindow;
use burn_extended::generate::{generate, GenerationConfig};
use burn_extended::loader::{
    build_gptoss_qkv_splits, load_gptoss_moe, load_safetensors_qkv_split, load_gptoss_sinks, load_gptoss_lm_head,
};
use burn_extended::models::gpt_oss::GptOssModel;
use burn_extended::models::gpt_oss::GptOssConfig;
use openai_harmony::{load_harmony_encoding, HarmonyEncodingName};
use openai_harmony::chat::{Message, Role, SystemContent};

fn main() -> anyhow::Result<()> {
    let device = WgpuDevice::default();
    wgpu::init_setup::<wgpu::graphics::Metal>(&device, Default::default());

    // Discover checkpoint directory
    let dir = std::env::var("GPT_OSS_DIR")
        .context("Set GPT_OSS_DIR to the directory with config.json and model.safetensors")?;
    let dir = PathBuf::from(dir);
    let cfg_path = dir.join("config.json");
    let ckpt_path = dir.join("model.safetensors");
    anyhow::ensure!(cfg_path.exists(), "config.json not found at {:?}", cfg_path);
    anyhow::ensure!(ckpt_path.exists(), "model.safetensors not found at {:?}", ckpt_path);

    // Load config and init model
    let cfg = GptOssConfig::from_config_json(&cfg_path)?;
    let mut model: GptOssModel<B> = cfg.init::<B>(&device);

    // Load weights: QKV + sinks
    let head_dim = cfg.head_dim;
    let splits = build_gptoss_qkv_splits(cfg.n_layers, cfg.n_heads, cfg.kv_heads, head_dim);
    let _ = load_safetensors_qkv_split::<B, _>(
        &mut model,
        &ckpt_path,
        &splits,
        /*from_pytorch*/ true,
        /*allow_partial*/ true,
        /*validate*/ false,
    )?;
    let _ = load_gptoss_sinks::<B, _>(
        &mut model,
        &ckpt_path,
        cfg.n_layers,
        cfg.n_heads,
        cfg.kv_heads,
        true,
        true,
        false,
    )?;

    // Load MoE MXFP4 weights + biases/gate/norm
    let _ = load_gptoss_moe::<B, _>(
        &mut model,
        &ckpt_path,
        cfg.n_layers,
        true,
        true,
        false,
    )?;
    // Map lm_head without PyTorch adapter to match model.bin orientation
    let _ = load_gptoss_lm_head::<B, _>(&mut model, &ckpt_path, true, false)?;

    // Harmony prompt
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)?;
    let sys = Message::from_role_and_content(Role::System, SystemContent::new());
    let user = Message::from_role_and_content(Role::User, "Explain quantum entanglement in simple terms.");
    let prefill = encoding.render_conversation_for_completion([&sys, &user], Role::Assistant, None)?;
    let stop_set = encoding.stop_tokens_for_assistant_actions()?;
    // Choose one EOS token (our generator supports a single EOS). Prefer the minimal id.
    let eos = stop_set.iter().copied().min();

    // Run generation
    let prompts = vec![prefill.into_iter().map(|r| r as usize).collect::<Vec<_>>()];
    let gen_cfg = GenerationConfig {
        max_new_tokens: 64,
        eos_token: eos.map(|v| v as usize),
        sampler: burn_extended::sampling::SamplerConfig { temperature: 0.8, top_k: Some(50), ..Default::default() },
        window: AttnWindow::Full,
    };
    let outputs = generate::<B, _>(&model, &device, &prompts, gen_cfg);
    println!("gpt-oss harmony infer output lens: {:?}", outputs.iter().map(|t| t.len()).collect::<Vec<_>>());
    Ok(())
}
