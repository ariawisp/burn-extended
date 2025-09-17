use std::fs::{self, File};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde::Deserialize;
use openai_harmony::{load_harmony_encoding, HarmonyEncodingName, HarmonyEncoding};
use safetensors::{tensor::TensorView, SafeTensors};
use half::bf16;

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

fn write_padding<W: Write + Seek>(w: &mut W, align: usize) -> Result<()> {
    let pos = w.seek(SeekFrom::Current(0))? as usize;
    let pad = (align - (pos % align)) % align;
    if pad > 0 {
        let buf = vec![0u8; pad];
        w.write_all(&buf)?;
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct StView<'a> {
    name: String,
    shape: Vec<usize>,
    bytes: &'a [u8],
}

fn st_get<'a>(st: &'a SafeTensors<'a>, name: &str) -> Result<StView<'a>> {
    let view: TensorView = st
        .tensor(name)
        .with_context(|| format!("tensor not found: {}", name))?;
    Ok(StView {
        name: name.to_string(),
        shape: view.shape().to_vec(),
        bytes: view.data(),
    })
}

fn rows_cols(shape: &[usize]) -> Result<(usize, usize)> {
    anyhow::ensure!(shape.len() >= 2, "expected matrix with >=2 dims, got {:?}", shape);
    let rows = shape[0];
    let cols: usize = shape[1..].iter().product();
    Ok((rows, cols))
}

fn slice_rows(bytes: &[u8], rows: usize, cols: usize, n_keep_rows: usize, elsize: usize) -> Result<Vec<u8>> {
    anyhow::ensure!(n_keep_rows <= rows, "slice_rows: n_keep_rows > rows");
    let row_bytes = cols * elsize;
    let n = n_keep_rows * row_bytes;
    anyhow::ensure!(bytes.len() >= n, "slice_rows: insufficient bytes");
    Ok(bytes[..n].to_vec())
}

fn write_rmsnorm_gain<W: Write + Seek>(w: &mut W, view: &StView) -> Result<()> {
    write_padding(w, 16)?;
    w.write_all(view.bytes)?;
    Ok(())
}

fn write_attn_sink<W: Write + Seek>(w: &mut W, view: &StView) -> Result<()> {
    write_padding(w, 16)?;
    w.write_all(view.bytes)?;
    Ok(())
}

fn write_linear_weights<W: Write + Seek>(w: &mut W, tensors: &[&[u8]]) -> Result<()> {
    write_padding(w, 16)?;
    for t in tensors {
        w.write_all(t)?;
    }
    Ok(())
}

fn bf16_from_le_bytes(buf: &[u8]) -> Vec<bf16> {
    let mut out = Vec::with_capacity(buf.len() / 2);
    let mut i = 0;
    while i < buf.len() {
        let lo = buf[i] as u16;
        let hi = (buf[i + 1] as u16) << 8;
        out.push(bf16::from_bits(hi | lo));
        i += 2;
    }
    out
}

fn bf16_to_le_bytes(vals: &[bf16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for v in vals {
        let bits = v.to_bits();
        out.push((bits & 0x00FF) as u8);
        out.push((bits >> 8) as u8);
    }
    out
}

// Interleave halves of QK rows: mapping from [H, 2, Hd/2, C] -> [H, Hd, C]
fn interleave_qk(qk_in: &[bf16], heads: usize, head_dim: usize, cols: usize) -> Vec<bf16> {
    let half = head_dim / 2;
    let mut out = vec![bf16::ZERO; heads * head_dim * cols];
    for h in 0..heads {
        for i in 0..half {
            for c in 0..cols {
                // original row indices
                let row0 = h * head_dim + 0 * half + i; // half_idx=0
                let row1 = h * head_dim + 1 * half + i; // half_idx=1
                // interleaved rows
                let out_row0 = h * head_dim + (i * 2 + 0);
                let out_row1 = h * head_dim + (i * 2 + 1);
                let src0 = row0 * cols + c;
                let src1 = row1 * cols + c;
                let dst0 = out_row0 * cols + c;
                let dst1 = out_row1 * cols + c;
                out[dst0] = qk_in[src0];
                out[dst1] = qk_in[src1];
            }
        }
    }
    out
}

// Transform fused QKV buffer (BF16) by interleaving Q/K halves and scaling Q,K rows.
fn transform_qkv_bf16(
    bytes: &[u8],
    rows: usize,
    cols: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Result<Vec<u8>> {
    anyhow::ensure!(head_dim % 2 == 0, "head_dim must be even");
    let qkv_vals = bf16_from_le_bytes(bytes);
    anyhow::ensure!(qkv_vals.len() == rows * cols, "unexpected qkv size");
    let qk_rows = head_dim * (n_heads + n_kv_heads);
    let v_rows = head_dim * n_kv_heads;
    anyhow::ensure!(rows == qk_rows + v_rows, "rows mismatch for qkv");

    // Interleave QK
    let qk_in = &qkv_vals[0..qk_rows * cols];
    let qk_inter = interleave_qk(qk_in, n_heads + n_kv_heads, head_dim, cols);

    // Split into Q and K heads, apply scaling
    let mut qk_scaled = qk_inter; // length: (n_heads + n_kv_heads) * head_dim * cols
    // Scale Q by 0.5
    for h in 0..n_heads {
        let base = (h * head_dim) * cols;
        for i in 0..(head_dim * cols) {
            qk_scaled[base + i] = bf16::from_f32(qk_scaled[base + i].to_f32() * 0.5);
        }
    }
    // Scale K by 0.25
    for kh in 0..n_kv_heads {
        let h = n_heads + kh;
        let base = (h * head_dim) * cols;
        for i in 0..(head_dim * cols) {
            qk_scaled[base + i] = bf16::from_f32(qk_scaled[base + i].to_f32() * 0.25);
        }
    }

    // V block unchanged
    let v_in = &qkv_vals[qk_rows * cols..];

    // Concatenate Q, K, V in that order
    let mut out_vals = Vec::with_capacity(rows * cols);
    // Q heads [0..n_heads)
    for h in 0..n_heads {
        let base = (h * head_dim) * cols;
        out_vals.extend_from_slice(&qk_scaled[base..base + head_dim * cols]);
    }
    // K heads [n_heads..n_heads+n_kv_heads)
    for kh in 0..n_kv_heads {
        let h = n_heads + kh;
        let base = (h * head_dim) * cols;
        out_vals.extend_from_slice(&qk_scaled[base..base + head_dim * cols]);
    }
    // V rows follow
    out_vals.extend_from_slice(v_in);

    Ok(bf16_to_le_bytes(&out_vals))
}

fn transform_bias_bf16(
    bytes: &[u8],
    rows: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Result<Vec<u8>> {
    // Bias is 1D of length rows = head_dim * (n_heads + 2*kv_heads)
    let cols = 1usize;
    transform_qkv_bf16(bytes, rows, cols, n_heads, n_kv_heads, head_dim)
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

    // Derive YaRN parameters identical to Python script (compute in f64 then cast)
    let head_dim_f64 = cfg.head_dim as f64;
    let init_ctx_f64 = cfg.initial_context_length as f64;
    let rope_theta_f64 = cfg.rope_theta as f64;
    let alpha_f64 = cfg.rope_ntk_alpha as f64;
    let beta_f64 = cfg.rope_ntk_beta as f64;
    let scale_f64 = cfg.rope_scaling_factor as f64;
    let two_pi = std::f64::consts::PI * 2.0;
    let yarn_low_f64 = head_dim_f64 / 2.0
        * (init_ctx_f64 / (beta_f64 * two_pi)).ln()
        / rope_theta_f64.ln();
    let yarn_high_f64 = head_dim_f64 / 2.0
        * (init_ctx_f64 / (alpha_f64 * two_pi)).ln()
        / rope_theta_f64.ln();
    let interpolation_scale = (1.0f64 / scale_f64) as f32;
    let yarn_offset = (-(yarn_low_f64) / (yarn_high_f64 - yarn_low_f64)) as f32;
    let yarn_scale = (1.0f64 / (yarn_high_f64 - yarn_low_f64)) as f32;
    let yarn_multiplier = (0.1f64 * scale_f64.ln() + 1.0) as f32;

    // Build tokenizer data via Harmony using o200k_base pattern and id layout compatible with Python.
    let encoding: HarmonyEncoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)?;
    // Vocab upper bound: o200k variants top out at 201088 (inclusive)
    let n_vocab: u32 = 201_088 + 1;
    // Classify token ids into text vs special using the tokenizer's special set.
    let mut num_text_tokens: u32 = 0;
    for t in 0..n_vocab {
        if !encoding
            .tokenizer()
            .is_special_token(u32::try_from(t).unwrap())
        {
            num_text_tokens += 1;
        }
    }
    // Python exporter fixes included token upper bound to 200013 inclusive.
    let num_included_tokens: u32 = 200_013 + 1;

    // Regex pattern exactly as o200k_base.
    const O200K_BASE_PATTERN: &str = concat!(
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
    let regex = O200K_BASE_PATTERN.to_string();
    let regex_size: u32 = (regex.as_bytes().len() + 1) as u32; // include trailing NUL
    let mut tokens_size: u32 = 0;
    for t in 0..num_text_tokens {
        let tid = u32::try_from(t).unwrap();
        let bytes = encoding
            .tokenizer()
            .decode_bytes([tid])
            .map_err(|_| anyhow::anyhow!("Failed to decode token id {}", tid))?;
        tokens_size += 2 + bytes.len() as u32; // even if empty, add 2 for len
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
    // Fallback: ensure GPT-OSS numeric ids map as in Python exporter.
    let numeric_overrides: [(u32, &str); 9] = [
        (200006, "55a77c2f-8a01-4c54-8ac2-313bfc7e208d"), // <|start|>
        (200008, "16e40431-f47f-4b22-b59b-8b278fc30a54"), // <|message|>
        (200007, "fcac2f6d-4705-4f6b-b228-642accac7238"), // <|end|>
        (200002, "f799ff69-1992-43c4-a3d8-d831f475dc75"), // <|return|>
        (200013, "e15ba702-28c4-4292-ab8f-ffa434709128"), // <|refusal|>
        (200003, "c0bb14c7-6022-49da-ad08-792d67e8b470"), // <|constrain|>
        (200005, "fd3dda11-c8ab-4033-876e-d93deb172c93"), // <|channel|>
        (200012, "1220f796-e388-4de5-b487-fe2eb5fe03c0"), // <|call|>
        (200000, "07d7da55-b346-4cff-8b37-7cefacf8a3e8"), // <|untrusted|>
    ];
    for (tid, uuid_s) in numeric_overrides.iter() {
        special_id_to_uuid.entry(*tid).or_insert(uuid_bytes(uuid_s)?);
    }

    // Open output
    let f = File::create(&dst).context("create dst file")?;
    let mut w = BufWriter::new(f);
    let mut pos = |w: &mut BufWriter<File>| -> Result<u64> {
        Ok(w.seek(SeekFrom::Current(0))?)
    };

    // FILE_MAGIC: b"GPT-OSS v1.0\0"
    let mut magic = Vec::new();
    magic.extend_from_slice(b"GPT-OSS v1.0");
    magic.extend_from_slice(&0u32.to_le_bytes());
    w.write_all(&magic)?;
    let mut off = pos(&mut w)?;
    eprintln!("[gptoss_export] Wrote file magic, offset={off}");

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
    off = pos(&mut w)?;
    eprintln!("[gptoss_export] Wrote model header, offset={off}");

    // Tokenizer UUID (tiktoken)
    let tiktoken_uuid = uuid_bytes("7401aded-2a95-40cb-b782-9ccebaafe72b")?;
    w.write_all(&tiktoken_uuid)?;
    write_u32(&mut w, num_included_tokens - num_text_tokens)?; // num_special_tokens
    write_u32(&mut w, num_text_tokens)?; // num_text_tokens
    write_u32(&mut w, regex_size)?; // regex_size
    write_u32(&mut w, tokens_size)?; // tokens_size
    off = pos(&mut w)?;
    eprintln!("[gptoss_export] Tokenizer header: num_special={}, num_text={}, regex_size={}, tokens_size={}, offset={off}", num_included_tokens - num_text_tokens, num_text_tokens, regex_size, tokens_size);

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
        let tid = u32::try_from(t).unwrap();
        let bytes = encoding
            .tokenizer()
            .decode_bytes([tid])
            .map_err(|_| anyhow::anyhow!("Failed to decode token id {}", tid))?;
        let len = bytes.len() as u16;
        w.write_all(&len.to_le_bytes())?;
        if len > 0 { w.write_all(&bytes)?; }
    }
    // Align to 16KB as Python exporter does after tokenizer
    write_padding(&mut w, 16384)?;
    off = pos(&mut w)?;
    eprintln!("[gptoss_export] Tokenizer payload written, aligned to 16KB, offset={off}");

    // Load SafeTensors once
    let st_bytes = fs::read(&st_path).context("read model.safetensors")?;
    let st = SafeTensors::deserialize(&st_bytes).context("parse model.safetensors")?;

    eprintln!("[gptoss_export] Writing embeddings...");
    // Embedding: filter to included tokens
    {
        let emb = st_get(&st, "embedding.weight")?;
        let (rows, cols) = rows_cols(&emb.shape)?;
        let elsize = emb.bytes.len() / (rows * cols);
        let keep = (num_included_tokens) as usize;
        let sliced = slice_rows(emb.bytes, rows, cols, keep, elsize)?;
        let start = pos(&mut w)?;
        write_linear_weights(&mut w, &[&sliced])?;
        let end = pos(&mut w)?;
        eprintln!("[gptoss_export] Embedding weight bytes={} (aligned section size={})", sliced.len(), end - start);
    }

    // Per-layer attention + gate
    for l in 0..cfg.num_hidden_layers {
        eprintln!("[gptoss_export] Layer {l}: attention + gate...");
        // attn.norm.scale
        let start_layer = pos(&mut w)?;
        let norm = st_get(&st, &format!("block.{l}.attn.norm.scale"))?;
        write_rmsnorm_gain(&mut w, &norm)?;

        // attn.qkv.{weight,bias} with transform
        let qkv_w = st_get(&st, &format!("block.{l}.attn.qkv.weight"))?;
        let (rows_w, cols_w) = rows_cols(&qkv_w.shape)?;
        anyhow::ensure!(rows_w == cfg.head_dim * (cfg.num_attention_heads + 2 * cfg.num_key_value_heads),
            "unexpected qkv.weight rows: {}", rows_w);
        let qkv_w_t = transform_qkv_bf16(
            qkv_w.bytes,
            rows_w,
            cols_w,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        )?;
        let qkv_b = st_get(&st, &format!("block.{l}.attn.qkv.bias"))?;
        let (rows_b, cols_b) = if qkv_b.shape.len() == 1 { (qkv_b.shape[0], 1usize) } else { rows_cols(&qkv_b.shape)? };
        anyhow::ensure!(cols_b == 1, "qkv.bias must be 1D");
        anyhow::ensure!(rows_b == cfg.head_dim * (cfg.num_attention_heads + 2 * cfg.num_key_value_heads),
            "unexpected qkv.bias rows: {}", rows_b);
        let qkv_b_t = transform_bias_bf16(
            qkv_b.bytes,
            rows_b,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        )?;
        let qkv_bytes = qkv_w_t.len() + qkv_b_t.len();
        write_linear_weights(&mut w, &[&qkv_w_t, &qkv_b_t])?;

        // attn.sinks (BF16)
        let sinks = st_get(&st, &format!("block.{l}.attn.sinks"))?;
        let sinks_bytes = sinks.bytes.len();
        write_attn_sink(&mut w, &sinks)?;

        // attn.out.{weight,bias}
        let out_w = st_get(&st, &format!("block.{l}.attn.out.weight"))?;
        let out_b = st_get(&st, &format!("block.{l}.attn.out.bias"))?;
        let out_bytes = out_w.bytes.len() + out_b.bytes.len();
        write_linear_weights(&mut w, &[out_w.bytes, out_b.bytes])?;

        // mlp.norm.scale
        let mlp_norm = st_get(&st, &format!("block.{l}.mlp.norm.scale"))?;
        write_rmsnorm_gain(&mut w, &mlp_norm)?;

        // mlp.gate.{weight,bias}
        let gate_w = st_get(&st, &format!("block.{l}.mlp.gate.weight"))?;
        let gate_b = st_get(&st, &format!("block.{l}.mlp.gate.bias"))?;
        let gate_bytes = gate_w.bytes.len() + gate_b.bytes.len();
        write_linear_weights(&mut w, &[gate_w.bytes, gate_b.bytes])?;

        let end_layer = pos(&mut w)?;
        eprintln!(
            "[gptoss_export] Layer {l}: attn.qkv(w+b)={}B, sinks={}B, attn.out(w+b)={}B, gate(w+b)={}B, layer_section={}B",
            qkv_bytes, sinks_bytes, out_bytes, gate_bytes, end_layer - start_layer
        );
    }

    // final norm.scale
    eprintln!("[gptoss_export] Final norm + unembedding...");
    let final_norm = st_get(&st, "norm.scale")?;
    let start_final = pos(&mut w)?;
    write_rmsnorm_gain(&mut w, &final_norm)?;

    // unembedding.weight filtered
    {
        let unemb = st_get(&st, "unembedding.weight")?;
        let (rows, cols) = rows_cols(&unemb.shape)?;
        let elsize = unemb.bytes.len() / (rows * cols);
        let keep = (num_included_tokens) as usize;
        let sliced = slice_rows(unemb.bytes, rows, cols, keep, elsize)?;
        write_linear_weights(&mut w, &[&sliced])?;
    }
    let end_final = pos(&mut w)?;
    eprintln!("[gptoss_export] Final norm + unembedding section={}B", end_final - start_final);

    // MoE sections per layer
    const UE8_OFFSET: u8 = 14;
    for l in 0..cfg.num_hidden_layers {
        eprintln!("[gptoss_export] Layer {l}: MoE blocks/scales/biases...");
        // align to 16KB before MoE layer group
        write_padding(&mut w, 16384)?;
        let mlp1_blocks = st_get(&st, &format!("block.{l}.mlp.mlp1_weight.blocks"))?;
        let mlp1_scales = st_get(&st, &format!("block.{l}.mlp.mlp1_weight.scales"))?;
        let mlp1_bias = st_get(&st, &format!("block.{l}.mlp.mlp1_bias"))?;
        let mlp2_blocks = st_get(&st, &format!("block.{l}.mlp.mlp2_weight.blocks"))?;
        let mlp2_scales = st_get(&st, &format!("block.{l}.mlp.mlp2_weight.scales"))?;
        let mlp2_bias = st_get(&st, &format!("block.{l}.mlp.mlp2_bias"))?;

        // First dim is experts
        let experts = cfg.num_experts as usize;
        let per_e_mlp1_blocks = mlp1_blocks.bytes.len() / experts;
        let per_e_mlp1_scales = mlp1_scales.bytes.len() / experts;
        let per_e_mlp1_bias = mlp1_bias.bytes.len() / experts;
        let per_e_mlp2_blocks = mlp2_blocks.bytes.len() / experts;
        let per_e_mlp2_scales = mlp2_scales.bytes.len() / experts;
        let per_e_mlp2_bias = mlp2_bias.bytes.len() / experts;

        let mut layer_bytes: u64 = 0;
        for e in 0..experts {
            // mlp1 blocks
            write_padding(&mut w, 16)?;
            let off = e * per_e_mlp1_blocks;
            w.write_all(&mlp1_blocks.bytes[off..off + per_e_mlp1_blocks])?;
            layer_bytes += (16 + per_e_mlp1_blocks) as u64; // padding is upper bound; exact padding bytes may be <16

            // mlp1 scales (+offset)
            write_padding(&mut w, 16)?;
            let off_s = e * per_e_mlp1_scales;
            let mut tmp = mlp1_scales.bytes[off_s..off_s + per_e_mlp1_scales].to_vec();
            for b in &mut tmp { *b = b.wrapping_add(UE8_OFFSET); }
            w.write_all(&tmp)?;
            layer_bytes += (16 + per_e_mlp1_scales) as u64;

            // mlp1 bias
            write_padding(&mut w, 16)?;
            let off_b = e * per_e_mlp1_bias;
            w.write_all(&mlp1_bias.bytes[off_b..off_b + per_e_mlp1_bias])?;
            layer_bytes += (16 + per_e_mlp1_bias) as u64;

            // mlp2 blocks
            write_padding(&mut w, 16)?;
            let off2 = e * per_e_mlp2_blocks;
            w.write_all(&mlp2_blocks.bytes[off2..off2 + per_e_mlp2_blocks])?;
            layer_bytes += (16 + per_e_mlp2_blocks) as u64;

            // mlp2 scales (+offset)
            write_padding(&mut w, 16)?;
            let off2s = e * per_e_mlp2_scales;
            let mut tmp2 = mlp2_scales.bytes[off2s..off2s + per_e_mlp2_scales].to_vec();
            for b in &mut tmp2 { *b = b.wrapping_add(UE8_OFFSET); }
            w.write_all(&tmp2)?;
            layer_bytes += (16 + per_e_mlp2_scales) as u64;

            // mlp2 bias
            write_padding(&mut w, 16)?;
            let off2b = e * per_e_mlp2_bias;
            w.write_all(&mlp2_bias.bytes[off2b..off2b + per_e_mlp2_bias])?;
            layer_bytes += (16 + per_e_mlp2_bias) as u64;
        }
        eprintln!("[gptoss_export] Layer {l}: MoE expert payload ~{}B (incl. padding)", layer_bytes);
    }

    w.flush()?;
    let total = pos(&mut w)?;
    eprintln!("[gptoss_export] Export completed: {:?} ({} bytes)", dst, total);
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
