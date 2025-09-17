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
    let hdr_d_model = parsed.header.embedding_dim as usize;
    let hdr_n_heads = parsed.header.num_heads as usize;
    let hdr_head_dim = parsed.header.head_dim as usize;
    let derived_d_model = hdr_n_heads * hdr_head_dim;

    // Use header sizes always (GPT-OSS supports decoupled dims; we follow header).
    let runtime_d_model = hdr_d_model;
    if hdr_d_model != derived_d_model {
        eprintln!(
            "[modelbin_infer] Decoupled dims: embedding_dim={} vs n_heads*head_dim={} ({}*{}). Using header sizes; this is expected for GPT-OSS.",
            hdr_d_model, derived_d_model, parsed.header.num_heads, parsed.header.head_dim
        );
    }

    let cfg = GptOssConfig {
        vocab_size: included_tokens,
        d_model: runtime_d_model,
        n_layers: parsed.header.num_blocks as usize,
        n_heads: hdr_n_heads,
        head_dim: hdr_head_dim,
        kv_heads: parsed.header.num_kv_heads as usize,
        ffn_hidden: parsed.header.mlp_dim as usize,
        num_experts: parsed.header.num_experts as usize,
        experts_per_token: parsed.header.num_active_experts as usize,
        verbose: args.debug,
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
            // print top-5 logits indices
            let vec = logits_prefill.clone().into_data().convert::<f32>().into_vec::<f32>().expect("logits vec");
            let mut idx: Vec<usize> = (0..vec.len()).collect();
            idx.sort_unstable_by(|&a,&b| vec[b].partial_cmp(&vec[a]).unwrap_or(core::cmp::Ordering::Equal));
            let topn = idx.iter().take(5).map(|&i| (i, vec[i])).collect::<Vec<_>>();
            eprintln!("prefill top5: {:?}", topn);
        }
        // Sample first token from prefill logits (next position)
        let next_id = if args.temperature <= 0.0 && args.top_k.is_none() {
            // Greedy on CPU
            let v = logits_prefill
                .clone()
                .into_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("logits vec");
            let mut best = 0usize; let mut bestv = f32::NEG_INFINITY;
            for (i,&x) in v.iter().enumerate() { let xv = if x.is_finite() { x } else { f32::NEG_INFINITY }; if xv > bestv { bestv = xv; best = i; } }
            best
        } else {
            let next = burn_extended::sampling::process_and_sample::<B>(
                logits_prefill,
                None,
                sampler,
                true,
            );
            next.into_data().convert::<i64>().into_vec::<i64>().expect("token")[0] as usize
        };
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
    // Helper: robust greedy sampling on CPU for temperature<=0
    let vocab_size = included_tokens;
    let greedy_cpu = |logits: burn::tensor::Tensor<B, 2>| -> usize {
        let data = logits
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("logits f32");
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in data.iter().enumerate() {
            let val = if v.is_finite() { v } else { f32::NEG_INFINITY };
            if val > best_v { best_v = val; best = i; }
        }
        best
    };
    // Step-wise decode for remaining tokens (if any)
    // Unlimited when max_new == 0; rely on stop tokens with a very large ceiling
    let safety_cap = if max_new == 0 { 1_000_000 } else { max_new };
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
        let next_id = if args.temperature <= 0.0 && args.top_k.is_none() {
            greedy_cpu(logits)
        } else {
            let next = burn_extended::sampling::process_and_sample::<B>(
                logits,
                None,
                sampler,
                true,
            );
            next
                .into_data()
                .convert::<i64>()
                .into_vec::<i64>()
                .expect("token")[0] as usize
        };
        if next_id >= vocab_size {
            eprintln!("[warn] sampled invalid token id {} (vocab={}), stopping.", next_id, vocab_size);
            break;
        }
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
