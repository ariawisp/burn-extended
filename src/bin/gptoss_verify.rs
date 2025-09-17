use anyhow::{anyhow, bail, Context, Result};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

use clap::Parser;
use safetensors::SafeTensors;

#[derive(Parser, Debug)]
#[command(name = "gptoss_verify", version, about = "Structural sanity checker for GPT-OSS model.bin")] 
struct Args {
    /// Path to model.bin
    #[arg(value_name = "FILE")]
    file: PathBuf,
    /// Optional path to the original SafeTensors directory (with model.safetensors) for MoE size validation
    #[arg(short = 's', long = "safetensors", value_name = "DIR")]
    safetensors_dir: Option<PathBuf>,
}

fn read_exact<R: Read>(mut r: R, buf: &mut [u8]) -> Result<()> {
    r.read_exact(buf).map_err(|e| anyhow!(e))
}

fn read_u32<R: Read>(mut r: R) -> Result<u32> {
    let mut b = [0u8; 4];
    read_exact(&mut r, &mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_f32<R: Read>(mut r: R) -> Result<f32> {
    let mut b = [0u8; 4];
    read_exact(&mut r, &mut b)?;
    Ok(f32::from_le_bytes(b))
}

fn align_up(off: u64, align: u64) -> u64 {
    if align == 0 { return off; }
    let rem = off % align;
    if rem == 0 { off } else { off + (align - rem) }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut f = File::open(&args.file).with_context(|| format!("open {:?}", args.file))?;
    let mut cur = 0u64;

    // Magic: "GPT-OSS v1.0\0" encoded as bytes with trailing u32 0 in Rust exporter
    let mut magic = [0u8; 16];
    read_exact(&mut f, &mut magic)?; cur += 16;
    if &magic[..12] != b"GPT-OSS v1.0" {
        bail!("invalid magic header");
    }

    // Model UUID (16), then model header fields
    let mut model_uuid = [0u8; 16];
    read_exact(&mut f, &mut model_uuid)?; cur += 16;
    // Expected df52dc86-1789-4ed0-a295-66f10508145b
    let expected_model_uuid = uuid::Uuid::parse_str("df52dc86-1789-4ed0-a295-66f10508145b")?.as_bytes().clone();
    if model_uuid != expected_model_uuid {
        eprintln!("[verify] Warning: unexpected model UUID: {:x?}", model_uuid);
    }

    let _context_length = read_u32(&mut f)?; cur += 4;
    let num_blocks = read_u32(&mut f)?; cur += 4;
    let num_experts = read_u32(&mut f)?; cur += 4;
    let num_active_experts = read_u32(&mut f)?; cur += 4;
    let embedding_dim = read_u32(&mut f)?; cur += 4;
    let mlp_dim = read_u32(&mut f)?; cur += 4;
    let swiglu_limit = read_f32(&mut f)?; cur += 4;
    let head_dim = read_u32(&mut f)?; cur += 4;
    let num_heads = read_u32(&mut f)?; cur += 4;
    let num_kv_heads = read_u32(&mut f)?; cur += 4;
    let attention_window = read_u32(&mut f)?; cur += 4;
    let rope_theta = read_f32(&mut f)?; cur += 4;
    let _interpolation_scale = read_f32(&mut f)?; cur += 4;
    let yarn_offset = read_f32(&mut f)?; cur += 4;
    let yarn_scale = read_f32(&mut f)?; cur += 4;
    let yarn_multiplier = read_f32(&mut f)?; cur += 4;
    let rmsnorm_epsilon = read_f32(&mut f)?; cur += 4;

    let mut layout_uuid = [0u8; 16];
    read_exact(&mut f, &mut layout_uuid)?; cur += 16;
    let expected_layout_uuid = uuid::Uuid::parse_str("229177a8-5775-4268-bfd8-d588b351c56d")?.as_bytes().clone();
    if layout_uuid != expected_layout_uuid {
        eprintln!("[verify] Warning: unexpected layout UUID");
    }

    println!(
        "[verify] Model header: blocks={}, experts={} (active={}), d_model={}, mlp_dim={}, head_dim={}, heads={}, kv_heads={}, window={}, swiglu_limit={:.3}, rope_theta={:.1}, yarn(mult={:.4},off={:.4},scale={:.6}), eps={:.1e}",
        num_blocks, num_experts, num_active_experts, embedding_dim, mlp_dim, head_dim, num_heads, num_kv_heads, attention_window, swiglu_limit, rope_theta, yarn_multiplier, yarn_offset, yarn_scale, rmsnorm_epsilon
    );

    // Tokenizer header
    let mut tok_uuid = [0u8; 16];
    read_exact(&mut f, &mut tok_uuid)?; cur += 16;
    let tok_special = read_u32(&mut f)?; cur += 4;
    let tok_text = read_u32(&mut f)?; cur += 4;
    let regex_size = read_u32(&mut f)?; cur += 4;
    let tokens_size = read_u32(&mut f)?; cur += 4;
    println!(
        "[verify] Tokenizer header: text={}, special={}, regex_size={}, tokens_size={}, offset={}",
        tok_text, tok_special, regex_size, tokens_size, cur
    );

    // Special UUID table
    let mut trash = vec![0u8; tok_special as usize * 16];
    read_exact(&mut f, &mut trash)?; cur += trash.len() as u64;

    // Regex ASCII + NUL (regex_size includes the nul)
    let mut regex_bytes = vec![0u8; regex_size as usize];
    read_exact(&mut f, &mut regex_bytes)?; cur += regex_bytes.len() as u64;
    if *regex_bytes.last().unwrap_or(&0) != 0 {
        bail!("regex not NUL terminated");
    }

    // Text tokens blob
    let mut tok_blob = vec![0u8; tokens_size as usize];
    read_exact(&mut f, &mut tok_blob)?; cur += tokens_size as u64;

    // Align to 16KB
    let aligned = align_up(cur, 16384);
    if aligned != cur {
        // Check the padding bytes are zero
        let pad = (aligned - cur) as usize;
        let mut pad_buf = vec![0u8; pad];
        read_exact(&mut f, &mut pad_buf)?; cur = aligned;
        if pad_buf.iter().any(|&b| b != 0) {
            eprintln!("[verify] Warning: 16KB padding contains non-zero bytes");
        }
    }
    println!("[verify] Tokenizer payload ends at {}, aligned to 16KB boundary {}", cur, cur);

    // Embedding section (unknown dtype; BF16=2 bytes, E4M3FN=1 byte)
    if cur % 16 != 0 { eprintln!("[verify] Warning: embedding section not 16-byte aligned: {}", cur); }
    let included_tokens = tok_text as u64 + tok_special as u64;
    let emb_dim = embedding_dim as u64;
    let emb_bf16 = included_tokens * emb_dim * 2;
    let emb_fp8 = included_tokens * emb_dim * 1;
    println!(
        "[verify] Embedding offset={}, included_tokens={}, embedding_dim={}, bf16_size={}B, fp8_size={}B",
        cur, included_tokens, emb_dim, emb_bf16, emb_fp8
    );

    // Assume BF16 embedding, step through attention/gate sections per layer
    let mut off = cur + emb_bf16;
    for l in 0..(num_blocks as u64) {
        // attn.norm.scale
        let need = align_up(off, 16);
        if need != off {
            let mut pad = vec![0u8; (need - off) as usize];
            read_exact(&mut f, &mut pad)?;
            off = need;
        }
        let attn_norm = emb_dim * 2;
        f.seek(SeekFrom::Current(attn_norm as i64))?;
        println!("[verify] L{l}: attn.norm.scale offset={}, size={}B", off, attn_norm);
        off += attn_norm;

        // qkv (weight+bias), align 16
        let need = align_up(off, 16);
        if need != off {
            let mut pad = vec![0u8; (need - off) as usize];
            read_exact(&mut f, &mut pad)?;
            off = need;
        }
        let rows = (head_dim as u64) * ((num_heads + 2 * num_kv_heads) as u64);
        let qkv_w = rows * emb_dim * 2;
        let qkv_b = rows * 2;
        f.seek(SeekFrom::Current((qkv_w + qkv_b) as i64))?;
        println!("[verify] L{l}: attn.qkv (w+b) offset={}, size={}B", off, qkv_w + qkv_b);
        off += qkv_w + qkv_b;

        // sinks, align 16
        let need = align_up(off, 16);
        if need != off {
            let mut pad = vec![0u8; (need - off) as usize];
            read_exact(&mut f, &mut pad)?;
            off = need;
        }
        let sinks = (num_heads as u64) * 2;
        f.seek(SeekFrom::Current(sinks as i64))?;
        println!("[verify] L{l}: attn.sinks offset={}, size={}B", off, sinks);
        off += sinks;

        // attn.out (w+b), align 16
        let need = align_up(off, 16);
        if need != off {
            let mut pad = vec![0u8; (need - off) as usize];
            read_exact(&mut f, &mut pad)?;
            off = need;
        }
        let out_w = emb_dim * emb_dim * 2;
        let out_b = emb_dim * 2;
        f.seek(SeekFrom::Current((out_w + out_b) as i64))?;
        println!("[verify] L{l}: attn.out (w+b) offset={}, size={}B", off, out_w + out_b);
        off += out_w + out_b;

        // mlp.norm.scale, align 16
        let need = align_up(off, 16);
        if need != off {
            let mut pad = vec![0u8; (need - off) as usize];
            read_exact(&mut f, &mut pad)?;
            off = need;
        }
        let mlp_norm = emb_dim * 2;
        f.seek(SeekFrom::Current(mlp_norm as i64))?;
        println!("[verify] L{l}: mlp.norm.scale offset={}, size={}B", off, mlp_norm);
        off += mlp_norm;

        // mlp.gate (w+b), align 16
        let need = align_up(off, 16);
        if need != off {
            let mut pad = vec![0u8; (need - off) as usize];
            read_exact(&mut f, &mut pad)?;
            off = need;
        }
        let gate_w = (num_experts as u64) * emb_dim * 2;
        let gate_b = (num_experts as u64) * 2;
        f.seek(SeekFrom::Current((gate_w + gate_b) as i64))?;
        println!("[verify] L{l}: mlp.gate (w+b) offset={}, size={}B", off, gate_w + gate_b);
        off += gate_w + gate_b;
    }

    // Final norm.scale, align 16
    let need = align_up(off, 16);
    if need != off {
        let mut pad = vec![0u8; (need - off) as usize];
        read_exact(&mut f, &mut pad)?;
        off = need;
    }
    let fin_norm = emb_dim * 2;
    f.seek(SeekFrom::Current(fin_norm as i64))?;
    println!("[verify] final norm.scale offset={}, size={}B", off, fin_norm);
    off += fin_norm;

    // Unembedding, align 16
    let need = align_up(off, 16);
    if need != off {
        let mut pad = vec![0u8; (need - off) as usize];
        read_exact(&mut f, &mut pad)?;
        off = need;
    }
    let unemb = included_tokens * emb_dim * 2;
    f.seek(SeekFrom::Current(unemb as i64))?;
    println!("[verify] unembedding weight offset={}, size={}B", off, unemb);
    off += unemb;

    println!("[verify] Reached pre-MoE offset {} (next alignment to 16KB expected per layer)", off);

    // Optional: validate MoE expert payload sizes using SafeTensors shapes
    if let Some(dir) = args.safetensors_dir {
        let st_path = dir.join("model.safetensors");
        let data = std::fs::read(&st_path).with_context(|| format!("read {:?}", st_path))?;
        let st = SafeTensors::deserialize(&data).context("parse safetensors")?;
        let experts = num_experts as usize;
        // Use layer 0 shapes to derive per-expert byte sizes
        let v = st.tensor("block.0.mlp.mlp1_weight.blocks").context("mlp1 blocks")?;
        let bytes_mlp1_blocks_total = v.data().len();
        let per_e_mlp1_blocks = bytes_mlp1_blocks_total / experts;
        let v = st.tensor("block.0.mlp.mlp1_weight.scales").context("mlp1 scales")?;
        let bytes_mlp1_scales_total = v.data().len();
        let per_e_mlp1_scales = bytes_mlp1_scales_total / experts;
        let v = st.tensor("block.0.mlp.mlp1_bias").context("mlp1 bias")?;
        let bytes_mlp1_bias_total = v.data().len();
        let per_e_mlp1_bias = bytes_mlp1_bias_total / experts;
        let v = st.tensor("block.0.mlp.mlp2_weight.blocks").context("mlp2 blocks")?;
        let bytes_mlp2_blocks_total = v.data().len();
        let per_e_mlp2_blocks = bytes_mlp2_blocks_total / experts;
        let v = st.tensor("block.0.mlp.mlp2_weight.scales").context("mlp2 scales")?;
        let bytes_mlp2_scales_total = v.data().len();
        let per_e_mlp2_scales = bytes_mlp2_scales_total / experts;
        let v = st.tensor("block.0.mlp.mlp2_bias").context("mlp2 bias")?;
        let bytes_mlp2_bias_total = v.data().len();
        let per_e_mlp2_bias = bytes_mlp2_bias_total / experts;

        println!(
            "[verify] MoE per-expert sizes (layer0): mlp1 blocks={}B, scales={}B, bias={}B; mlp2 blocks={}B, scales={}B, bias={}B",
            per_e_mlp1_blocks, per_e_mlp1_scales, per_e_mlp1_bias, per_e_mlp2_blocks, per_e_mlp2_scales, per_e_mlp2_bias
        );

        // Now iterate layers and experts to step MoE payloads
        for l in 0..(num_blocks as u64) {
            // 16KB alignment before MoE layer section
            let need = align_up(off, 16384);
            if need != off {
                let mut pad = vec![0u8; (need - off) as usize];
                read_exact(&mut f, &mut pad)?;
                off = need;
            }
            let mut layer_bytes: u64 = 0;
            for _e in 0..experts {
                // mlp1 blocks
                let need = align_up(off, 16);
                if need != off { let mut pad = vec![0u8; (need - off) as usize]; read_exact(&mut f, &mut pad)?; off = need; }
                f.seek(SeekFrom::Current(per_e_mlp1_blocks as i64))?; off += per_e_mlp1_blocks as u64; layer_bytes += (16 + per_e_mlp1_blocks) as u64;
                // mlp1 scales
                let need = align_up(off, 16);
                if need != off { let mut pad = vec![0u8; (need - off) as usize]; read_exact(&mut f, &mut pad)?; off = need; }
                f.seek(SeekFrom::Current(per_e_mlp1_scales as i64))?; off += per_e_mlp1_scales as u64; layer_bytes += (16 + per_e_mlp1_scales) as u64;
                // mlp1 bias
                let need = align_up(off, 16);
                if need != off { let mut pad = vec![0u8; (need - off) as usize]; read_exact(&mut f, &mut pad)?; off = need; }
                f.seek(SeekFrom::Current(per_e_mlp1_bias as i64))?; off += per_e_mlp1_bias as u64; layer_bytes += (16 + per_e_mlp1_bias) as u64;
                // mlp2 blocks
                let need = align_up(off, 16);
                if need != off { let mut pad = vec![0u8; (need - off) as usize]; read_exact(&mut f, &mut pad)?; off = need; }
                f.seek(SeekFrom::Current(per_e_mlp2_blocks as i64))?; off += per_e_mlp2_blocks as u64; layer_bytes += (16 + per_e_mlp2_blocks) as u64;
                // mlp2 scales
                let need = align_up(off, 16);
                if need != off { let mut pad = vec![0u8; (need - off) as usize]; read_exact(&mut f, &mut pad)?; off = need; }
                f.seek(SeekFrom::Current(per_e_mlp2_scales as i64))?; off += per_e_mlp2_scales as u64; layer_bytes += (16 + per_e_mlp2_scales) as u64;
                // mlp2 bias
                let need = align_up(off, 16);
                if need != off { let mut pad = vec![0u8; (need - off) as usize]; read_exact(&mut f, &mut pad)?; off = need; }
                f.seek(SeekFrom::Current(per_e_mlp2_bias as i64))?; off += per_e_mlp2_bias as u64; layer_bytes += (16 + per_e_mlp2_bias) as u64;
            }
            println!("[verify] L{l}: MoE layer payload ~{}B (incl. padding)", layer_bytes);
        }
    }

    // File size summary
    let total = f.seek(SeekFrom::End(0))?;
    println!("[verify] File size: {} bytes", total);
    Ok(())
}
