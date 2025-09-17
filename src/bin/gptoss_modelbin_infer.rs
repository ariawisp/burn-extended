#![recursion_limit = "256"]

use std::path::PathBuf;

use anyhow::Context;
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
    /// Hidden size (embedding_dim)
    #[arg(long = "d_model")]
    d_model: usize,
    /// Number of layers
    #[arg(long = "n_layers")]
    n_layers: usize,
    /// Number of heads
    #[arg(long = "n_heads")]
    n_heads: usize,
    /// Number of KV heads
    #[arg(long = "kv_heads")]
    kv_heads: usize,
    /// FFN hidden size
    #[arg(long = "ffn_hidden")]
    ffn_hidden: usize,
    /// Number of experts
    #[arg(long = "num_experts")]
    num_experts: usize,
    /// Vocabulary size (included tokens)
    #[arg(long = "vocab")]
    vocab: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = WgpuDevice::default();
    wgpu::init_setup::<wgpu::graphics::Metal>(&device, Default::default());

    // If args look like zero/placeholder, derive from model.bin header
    let parsed = parse_modelbin(&args.model_path)?;
    let included_tokens = (parsed.tokenizer.num_text + parsed.tokenizer.num_special) as usize;
    let cfg = GptOssConfig {
        vocab_size: if args.vocab > 0 { args.vocab } else { included_tokens },
        d_model: if args.d_model > 0 { args.d_model } else { parsed.header.embedding_dim as usize },
        n_layers: if args.n_layers > 0 { args.n_layers } else { parsed.header.num_blocks as usize },
        n_heads: if args.n_heads > 0 { args.n_heads } else { parsed.header.num_heads as usize },
        kv_heads: if args.kv_heads > 0 { args.kv_heads } else { parsed.header.num_kv_heads as usize },
        ffn_hidden: if args.ffn_hidden > 0 { args.ffn_hidden } else { parsed.header.mlp_dim as usize },
        num_experts: if args.num_experts > 0 { args.num_experts } else { parsed.header.num_experts as usize },
        ..Default::default()
    };
    let mut model: GptOssModel<B> = cfg.init::<B>(&device);

    // Load weights from model.bin
    let _ = load_modelbin_into::<B, _>(&mut model, &args.model_path, /*validate*/ false)?;

    // Harmony prompt
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)?;
    let sys = Message::from_role_and_content(Role::System, SystemContent::new());
    let user = Message::from_role_and_content(Role::User, "Explain quantum entanglement in simple terms.");
    let prefill = encoding.render_conversation_for_completion([&sys, &user], Role::Assistant, None)?;
    let stop_set = encoding.stop_tokens_for_assistant_actions()?;
    let eos = stop_set.iter().copied().min();

    // Generation
    let prompts = vec![prefill.into_iter().map(|r| r as usize).collect::<Vec<_>>()];
    let gen_cfg = GenerationConfig { max_new_tokens: 16, eos_token: eos.map(|v| v as usize), ..Default::default() };
    let outputs = generate::<B, _>(&model, &device, &prompts, gen_cfg);
    println!("modelbin infer output lens: {:?}", outputs.iter().map(|t| t.len()).collect::<Vec<_>>());
    Ok(())
}
