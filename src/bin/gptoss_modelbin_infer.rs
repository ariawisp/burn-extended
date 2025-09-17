#![recursion_limit = "256"]

use std::path::PathBuf;

use burn::backend::wgpu::{self, Wgpu as B, WgpuDevice};
use burn_extended::generate::AutoregressiveModel;
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
    /// Prompt text to complete (Harmony format will be applied)
    #[arg(short = 'p', long = "prompt", value_name = "TEXT")]
    prompt: Option<String>,
    /// Max new tokens to generate (default: 16)
    #[arg(long = "max_new_tokens")]
    max_new_tokens: Option<usize>,
    /// Sampling temperature (default: 0.0 = greedy)
    #[arg(long = "temperature", default_value_t = 0.0)]
    temperature: f32,
    /// top-k sampling (optional)
    #[arg(long = "top_k")]
    top_k: Option<usize>,
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
    /// Verbose debug logging
    #[arg(long = "debug", default_value_t = false)]
    debug: bool,
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
    let user_text = args
        .prompt
        .unwrap_or_else(|| "Explain quantum entanglement in simple terms.".to_string());
    let user = Message::from_role_and_content(Role::User, user_text);
    let prefill = encoding.render_conversation_for_completion([&sys, &user], Role::Assistant, None)?;

    // Generation
    // Single-sample incremental generation with Harmony stop tokens
    let mut prefill_ids: Vec<usize> = prefill.into_iter().map(|r| r as usize).collect();
    if args.debug {
        eprintln!("prefill_len={}", prefill_ids.len());
    }
    let stop_set: std::collections::HashSet<usize> = match encoding.stop_tokens_for_assistant_actions() {
        Ok(v) => v.into_iter().map(|u| u as usize).collect(),
        Err(_) => std::collections::HashSet::new(),
    };
    // Mirror python gpt_oss.generate semantics: default unlimited tokens (limit=0)
    let max_new = args.max_new_tokens.unwrap_or(0);
    let sampler = burn_extended::sampling::SamplerConfig {
        temperature: args.temperature,
        top_k: args.top_k,
        repetition_penalty: None,
        frequency_penalty: None,
        presence_penalty: None,
    };
    let mut cache = model.init_cache(1, &device);
    // Prefill cache using the full prefill sequence and sample the first token from its logits
    let mut completion: Vec<usize> = Vec::new();
    let mut start_pos = prefill_ids.len();
    let mut last_token: usize;
    {
        let input = burn::tensor::Tensor::<B, 2, burn::tensor::Int>::from_ints(
            burn_tensor::TensorData::new(
                prefill_ids.iter().map(|&t| t as i64).collect::<Vec<_>>(),
                [1, prefill_ids.len()],
            ),
            &device,
        );
        let t0 = std::time::Instant::now();
        let logits_prefill = model.forward_logits(input, &mut cache, 0, burn_extended::attention::AttnWindow::Full);
        if args.debug {
            eprintln!("prefill forward done in {:.2?}", t0.elapsed());
        }
        // Sample first token from prefill logits (next position)
        let next = burn_extended::sampling::process_and_sample::<B>(
            logits_prefill,
            None,
            sampler,
            true,
        );
        let next_id = next
            .into_data()
            .convert::<i64>()
            .into_vec::<i64>()
            .expect("token")[0] as usize;
        if args.debug { eprintln!("first_token={}", next_id); }
        if stop_set.contains(&next_id) {
            // No completion content; print empty assistant text
            println!("");
            return Ok(());
        }
        completion.push(next_id);
        prefill_ids.push(next_id);
        last_token = next_id;
        // Do not advance cache here; the next decode step will append this token at start_pos
    }
    // Step-wise decode for remaining tokens (if any)
    // Unlimited when max_new == 0; guard with a large safety ceiling
    let safety_cap = if max_new == 0 { 8192 } else { max_new };
    let mut produced = 0usize;
    while produced < safety_cap {
        // Provide the last token as a 1-length input; forward_logits uses the cache and processes only this position.
        let t1 = std::time::Instant::now();
        let input = burn::tensor::Tensor::<B, 2, burn::tensor::Int>::from_ints(
            burn_tensor::TensorData::new(vec![last_token as i64], [1, 1]),
            &device,
        );
        let logits = model.forward_logits(input, &mut cache, start_pos, burn_extended::attention::AttnWindow::Full);
        if args.debug { eprintln!("decode step {}: forward {:.2?} start_pos={} last_token={}", produced, t1.elapsed(), start_pos, last_token); }
        // Sample next token
        let next = burn_extended::sampling::process_and_sample::<B>(
            logits,
            None,
            sampler,
            true,
        );
        let next_id = next
            .into_data()
            .convert::<i64>()
            .into_vec::<i64>()
            .expect("token")[0] as usize;
        if args.debug { eprintln!("sampled next_id={}", next_id); }
        // Stop on Harmony stop tokens
        if stop_set.contains(&next_id) {
            if args.debug { eprintln!("hit stop token"); }
            break;
        }
        completion.push(next_id);
        prefill_ids.push(next_id);
        last_token = next_id;
        start_pos += 1;
        produced += 1;
    }

    // Decode completion back into Harmony messages and print assistant text.
    let tokens_u32: Vec<u32> = completion.iter().map(|&u| u as u32).collect();
    match encoding.parse_messages_from_completion_tokens(tokens_u32, Some(Role::Assistant)) {
        Ok(messages) => {
            let mut text = String::new();
            for msg in messages {
                if let Role::Assistant = msg.author.role {
                    for c in msg.content {
                        if let openai_harmony::chat::Content::Text(t) = c {
                            if !text.is_empty() { text.push('\n'); }
                            text.push_str(&t.text);
                        }
                    }
                }
            }
            println!("{}", text);
        }
        Err(e) => {
            eprintln!("[warn] Failed to parse completion tokens: {}", e);
            println!("modelbin infer output lens: {}", completion.len());
        }
    }
    Ok(())
}
