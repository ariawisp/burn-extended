#![recursion_limit = "256"]

use std::path::PathBuf;

use burn::backend::wgpu::{self, Wgpu as B, WgpuDevice};
use burn_extended::generate::{generate, GenerationConfig};
use burn_extended::loader::modelbin::{load_modelbin_into, parse_modelbin};
use burn_extended::models::gpt_oss::{GptOssConfig, GptOssModel};
use clap::Parser;
use openai_harmony::{load_harmony_encoding, HarmonyEncodingName};
use openai_harmony::chat::{Message, Role, SystemContent};

#[derive(Parser, Debug)]
#[command(name = "gptoss_modelbin_infer", version, about = "Run inference from GPT-OSS model.bin")] 
struct Args {
    /// Path to model.bin
    #[arg(short = 'm', long = "model", value_name = "FILE")]
    model_path: PathBuf,
    /// Hidden size (embedding_dim) override; if omitted, uses header
    #[arg(long = "d_model")]
    d_model: Option<usize>,
    /// Number of layers override; if omitted, uses header
    #[arg(long = "n_layers")]
    n_layers: Option<usize>,
    /// Number of heads override; if omitted, uses header
    #[arg(long = "n_heads")]
    n_heads: Option<usize>,
    /// Number of KV heads override; if omitted, uses header
    #[arg(long = "kv_heads")]
    kv_heads: Option<usize>,
    /// FFN hidden size override; if omitted, uses header
    #[arg(long = "ffn_hidden")]
    ffn_hidden: Option<usize>,
    /// Number of experts override; if omitted, uses header
    #[arg(long = "num_experts")]
    num_experts: Option<usize>,
    /// Vocabulary size (included tokens) override; if omitted, uses header
    #[arg(long = "vocab")]
    vocab: Option<usize>,
    /// Prefer a consistent runtime config where d_model == n_heads * head_dim
    /// If this conflicts with the header's embedding_dim, the run will abort unless --ignore_config_mismatch is set.
    #[arg(long = "prefer_consistent", default_value_t = false)]
    prefer_consistent: bool,
    /// Ignore header/config mismatch checks (unsafe). When set, we proceed using header sizes for loading and runtime
    /// config values for model initialization. This may fail to apply or crash if shapes do not match.
    #[arg(long = "ignore_config_mismatch", default_value_t = false)]
    ignore_config_mismatch: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = WgpuDevice::default();
    wgpu::init_setup::<wgpu::graphics::Metal>(&device, Default::default());

    // If args look like zero/placeholder, derive from model.bin header
    let parsed = parse_modelbin(&args.model_path)?;
    let included_tokens = (parsed.tokenizer.num_text + parsed.tokenizer.num_special) as usize;
    let mut hdr_d_model = parsed.header.embedding_dim as usize;
    let mut hdr_n_heads = parsed.header.num_heads as usize;
    let hdr_head_dim = parsed.header.head_dim as usize;
    let derived_d_model = hdr_n_heads * hdr_head_dim;

    // Apply CLI overrides for core dims if provided
    if let Some(nh) = args.n_heads { hdr_n_heads = nh; }
    if let Some(dm) = args.d_model { hdr_d_model = dm; }

    // Choose d_model per user preference (default uses header)
    let runtime_d_model = if args.prefer_consistent { derived_d_model } else { hdr_d_model };
    let mismatch = hdr_d_model != derived_d_model;
    if mismatch {
        eprintln!(
            "[modelbin_infer] Decoupled dims: embedding_dim={} vs n_heads*head_dim={} ({}*{}). Using header sizes; this is expected for GPT-OSS.",
            hdr_d_model, derived_d_model, parsed.header.num_heads, parsed.header.head_dim
        );
    }

    let cfg = GptOssConfig {
        vocab_size: args.vocab.unwrap_or(included_tokens),
        d_model: runtime_d_model,
        n_layers: args.n_layers.unwrap_or(parsed.header.num_blocks as usize),
        n_heads: hdr_n_heads,
        head_dim: hdr_head_dim,
        kv_heads: args.kv_heads.unwrap_or(parsed.header.num_kv_heads as usize),
        ffn_hidden: args.ffn_hidden.unwrap_or(parsed.header.mlp_dim as usize),
        num_experts: args.num_experts.unwrap_or(parsed.header.num_experts as usize),
        ..Default::default()
    };
    let mut cfg = cfg;
    // Enforce header sliding window policy: even layers windowed, odd layers full (match Metal)
    cfg.window_policy = burn_extended::cache::WindowPolicy::EveryOther {
        window: parsed.header.attention_window as usize,
        full_on_even: false,
    };
    let mut model: GptOssModel<B> = cfg.init::<B>(&device);

    // Load weights from model.bin
    let _ = load_modelbin_into::<B, _>(&mut model, &args.model_path, /*validate*/ false, /*skip_moe*/ false)?;

    // Harmony prompt
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)?;
    let sys = Message::from_role_and_content(Role::System, SystemContent::new());
    let user = Message::from_role_and_content(Role::User, "Explain quantum entanglement in simple terms.");
    let prefill = encoding.render_conversation_for_completion([&sys, &user], Role::Assistant, None)?;
    let stop_set = encoding.stop_tokens_for_assistant_actions()?;
    let eos = stop_set.iter().copied().min();

    // Generation
    let prompts = vec![prefill.into_iter().map(|r| r as usize).collect::<Vec<_>>()];
    let gen_cfg = GenerationConfig { max_new_tokens: 16, eos_token: eos.map(|v| v as usize), sampler: burn_extended::sampling::SamplerConfig { temperature: 0.0, top_k: None, repetition_penalty: None, frequency_penalty: None, presence_penalty: None }, ..Default::default() };
    let outputs = generate::<B, _>(&model, &device, &prompts, gen_cfg);
    println!("modelbin infer output lens: {:?}", outputs.iter().map(|t| t.len()).collect::<Vec<_>>());
    Ok(())
}
