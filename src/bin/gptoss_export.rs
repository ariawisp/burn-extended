use std::fs::{self, File};
use std::io::{BufWriter, Write, Seek, SeekFrom};
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde::Deserialize;
use openai_harmony::{load_harmony_encoding, HarmonyEncodingName, HarmonyEncoding};

#[derive(Parser, Debug)]
#[command(name = "gptoss_export", version, about = "SafeTensors -> model.bin exporter for GPT-OSS (scaffold)")]
struct Args {
    /// Directory containing config.json and model.safetensors
    #[arg(short = 's', long = "src")]
    src: PathBuf,
    /// Output model.bin file
    #[arg(short = 'd', long = "dst")]
    dst: PathBuf,
}

#[derive(Debug, Deserialize)]
struct GptOssJsonConfig {
    context_length: Option<usize>,
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_experts: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    swiglu_limit: Option<f32>,
    rope_theta: f32,
    sliding_window: usize,
    initial_context_length: f32,
    rope_scaling_factor: f32,
    rope_ntk_alpha: f32,
    rope_ntk_beta: f32,
}

fn write_padding<W: Write + Seek>(mut w: W, align: usize) -> Result<()> {
    let pos = w.seek(SeekFrom::Current(0))? as usize;
    let pad = (align - (pos % align)) % align;
    if pad > 0 {
        let buf = vec![0u8; pad];
        w.write_all(&buf)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let src = args.src;
    let dst = args.dst;

    let cfg_path = src.join("config.json");
    let st_path = src.join("model.safetensors");
    if !cfg_path.exists() {
        bail!("config.json not found at {:?}", cfg_path);
    }
    if !st_path.exists() {
        bail!("model.safetensors not found at {:?}", st_path);
    }

    let cfg_text = fs::read_to_string(&cfg_path).context("read config.json")?;
    let cfg: GptOssJsonConfig = serde_json::from_str(&cfg_text).context("parse config.json")?;

    // Derive YaRN parameters identical to Python script
    let head_dim = cfg.head_dim as f32;
    let yarn_low = head_dim / 2.0
        * (cfg.initial_context_length / (cfg.rope_ntk_beta * std::f32::consts::PI * 2.0))
            .ln()
        / cfg.rope_theta.ln();
    let yarn_high = head_dim / 2.0
        * (cfg.initial_context_length / (cfg.rope_ntk_alpha * std::f32::consts::PI * 2.0))
            .ln()
        / cfg.rope_theta.ln();
    let interpolation_scale = 1.0 / cfg.rope_scaling_factor;
    let yarn_offset = -yarn_low / (yarn_high - yarn_low);
    let yarn_scale = 1.0 / (yarn_high - yarn_low);
    let yarn_multiplier = 0.1 * cfg.rope_scaling_factor.ln() + 1.0;

    // Build tokenizer data via Harmony (o200k_harmony)
    let encoding: HarmonyEncoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)?;
    // GPT‑OSS uses 200000 text tokens (0..199999) and special tokens up to id 200013 inclusive.
    let num_text_tokens: u32 = 200000;
    let num_included_tokens: u32 = 200013 + 1; // inclusive upper bound + 1

    // Precompute tokenizer sizes
    const O200K_HARMONY_PATTERN: &str = concat!(
        "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        "|",
        "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        "|",
        "\\p{N}{1,3}",
        "|",
        " ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*",
        "|",
        "\\s*[\\r\\n]+",
        "|",
        "\\s+(?!\\S)",
        "|",
        "\\s+",
    );
    let regex = O200K_HARMONY_PATTERN.to_string();
    let regex_size: u32 = (regex.as_bytes().len() + 1) as u32; // include trailing NUL
    let mut tokens_size: u32 = 0;
    for t in 0..num_text_tokens {
        // Decode single token bytes
        let bytes = encoding.tokenizer().decode_bytes([t])
            .map_err(|_| anyhow::anyhow!("Failed to decode token id {}", t))?;
        if bytes.is_empty() {
            // Some tokens may be zero-length; still write len=0
            tokens_size += 2; // u16 length
        } else {
            tokens_size += 2 + bytes.len() as u32;
        }
    }

    // Build map of included special token ids and their UUIDs
    let specials: [(&str, &str); 10] = [
        ("<|start|>", "55a77c2f-8a01-4c54-8ac2-313bfc7e208d"),
        ("<|message|>", "16e40431-f47f-4b22-b59b-8b278fc30a54"),
        ("<|end|>", "fcac2f6d-4705-4f6b-b228-642accac7238"),
        ("<|return|>", "f799ff69-1992-43c4-a3d8-d831f475dc75"),
        ("<|refusal|>", "e15ba702-28c4-4292-ab8f-ffa434709128"),
        ("<|constrain|>", "c0bb14c7-6022-49da-ad08-792d67e8b470"),
        ("<|channel|>", "fd3dda11-c8ab-4033-876e-d93deb172c93"),
        ("<|call|>", "1220f796-e388-4de5-b487-fe2eb5fe03c0"),
        ("<|untrusted|>", "07d7da55-b346-4cff-8b37-7cefacf8a3e8"),
        ("<|end_untrusted|>", "f265bd9c-c717-469e-a447-920687d65d90"),
    ];
    // Map from token id in [num_text_tokens..num_included_tokens) to UUID bytes
    let mut special_id_to_uuid: std::collections::HashMap<u32, [u8; 16]> = std::collections::HashMap::new();
    for (tok_str, uuid_s) in specials.iter() {
        use std::collections::HashSet;
        let mut allowed: HashSet<&str> = HashSet::new();
        allowed.insert(tok_str);
        let (ids, _last_len) = encoding.tokenizer().encode(tok_str, &allowed);
        if ids.len() == 1 {
            special_id_to_uuid.insert(ids[0], uuid_bytes(uuid_s)?);
        }
    }

    // Open output
    let f = File::create(&dst).context("create dst file")?;
    let mut w = BufWriter::new(f);

    // FILE_MAGIC: b"GPT-OSS v1.0\0"
    let mut magic = Vec::new();
    magic.extend_from_slice(b"GPT-OSS v1.0");
    magic.extend_from_slice(&0u32.to_le_bytes());
    w.write_all(&magic)?;

    // UUIDs copied from the Python script
    let gptoss_model_uuid = uuid_bytes("df52dc86-1789-4ed0-a295-66f10508145b")?;
    let apple_gpu_layout_uuid = uuid_bytes("229177a8-5775-4268-bfd8-d588b351c56d")?;

    // Model header
    w.write_all(&gptoss_model_uuid)?;
    write_u32(&mut w, (cfg.initial_context_length * cfg.rope_scaling_factor) as u32)?; // context_length
    write_u32(&mut w, cfg.num_hidden_layers as u32)?; // num_blocks
    write_u32(&mut w, cfg.num_experts as u32)?; // num_experts
    write_u32(&mut w, 4)?; // num_active_experts (constant)
    write_u32(&mut w, cfg.hidden_size as u32)?; // embedding_dim
    write_u32(&mut w, cfg.intermediate_size as u32)?; // mlp_dim
    write_f32(&mut w, cfg.swiglu_limit.unwrap_or(7.0))?; // swiglu_limit
    write_u32(&mut w, cfg.head_dim as u32)?; // head_dim
    write_u32(&mut w, cfg.num_attention_heads as u32)?; // num_heads
    write_u32(&mut w, cfg.num_key_value_heads as u32)?; // num_kv_heads
    write_u32(&mut w, cfg.sliding_window as u32)?; // attention_window
    write_f32(&mut w, cfg.rope_theta)?; // rope_theta
    write_f32(&mut w, interpolation_scale)?; // interpolation_scale
    write_f32(&mut w, yarn_offset)?; // yarn_offset
    write_f32(&mut w, yarn_scale)?; // yarn_scale
    write_f32(&mut w, yarn_multiplier)?; // yarn_multiplier
    write_f32(&mut w, 1.0e-5)?; // rmsnorm_epsilon
    w.write_all(&apple_gpu_layout_uuid)?;

    // Tokenizer UUID (tiktoken)
    let tiktoken_uuid = uuid_bytes("7401aded-2a95-40cb-b782-9ccebaafe72b")?;
    w.write_all(&tiktoken_uuid)?;
    write_u32(&mut w, num_included_tokens - num_text_tokens)?; // num_special_tokens
    write_u32(&mut w, num_text_tokens)?; // num_text_tokens
    write_u32(&mut w, regex_size)?; // regex_size
    write_u32(&mut w, tokens_size)?; // tokens_size

    // Special tokens UUID table for ids in [num_text_tokens..num_included_tokens)
    for tid in num_text_tokens..num_included_tokens {
        if let Some(uuid) = special_id_to_uuid.get(&tid) {
            w.write_all(uuid)?;
        } else {
            w.write_all(&[0u8; 16])?;
        }
    }
    // Regex as ASCII + NUL
    w.write_all(regex.as_bytes())?;
    w.write_all(&[0u8])?;
    // Text tokens: for t in 0..num_text_tokens write u16 length + bytes
    for t in 0..num_text_tokens {
        let bytes = encoding.tokenizer().decode_bytes([t])
            .map_err(|_| anyhow::anyhow!("Failed to decode token id {}", t))?;
        let len = bytes.len() as u16;
        w.write_all(&len.to_le_bytes())?;
        if len > 0 { w.write_all(&bytes)?; }
    }
    write_padding(&mut w, 16)?;

    w.flush()?;
    eprintln!("[gptoss_export] Wrote header + tokenizer payload to {:?}", dst);
    eprintln!("[gptoss_export] Next: write attention QKV/out, sinks, and MoE MXFP4 blocks.");
    Ok(())
}

fn write_u32<W: Write>(w: &mut W, v: u32) -> Result<()> {
    w.write_all(&v.to_le_bytes())?;
    Ok(())
}
fn write_f32<W: Write>(w: &mut W, v: f32) -> Result<()> {
    w.write_all(&v.to_le_bytes())?;
    Ok(())
}

fn uuid_bytes(s: &str) -> Result<[u8; 16]> {
    let u = uuid::Uuid::parse_str(s)?;
    Ok(*u.as_bytes())
}
