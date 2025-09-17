use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use half::bf16;
use rand::{rngs::StdRng, RngCore, SeedableRng};
use safetensors::tensor::{Dtype, TensorView};

#[derive(Parser, Debug)]
#[command(name = "gptoss_fixture", version, about = "Generate a tiny GPT-OSS-like SafeTensors fixture + config.json")]
struct Args {
    /// Output directory to create (will be created if missing)
    #[arg(short = 'o', long = "out")] 
    out: PathBuf,
}

fn bf16_bytes(len: usize, seed: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(len * 2);
    let mut rng = StdRng::seed_from_u64(seed);
    for _ in 0..len {
        // Small values around zero
        let v = ((rng.next_u32() % 1000) as f32) / 1000.0 - 0.5;
        let b = bf16::from_f32(v).to_bits();
        out.push((b & 0x00FF) as u8);
        out.push((b >> 8) as u8);
    }
    out
}

fn u8_bytes(len: usize, seed: u64) -> Vec<u8> {
    let mut out = vec![0u8; len];
    let mut rng = StdRng::seed_from_u64(seed);
    for i in 0..len {
        out[i] = (rng.next_u32() % 8) as u8; // small values
    }
    out
}

fn main() -> Result<()> {
    let args = Args::parse();
    fs::create_dir_all(&args.out).context("create output dir")?;

    // Minimal, but consistent with Python exporter expectations
    let num_hidden_layers = 1usize;
    let num_experts = 2usize;
    let num_attention_heads = 1usize;
    let num_key_value_heads = 1usize;
    let head_dim = 64usize; // important for QK scaling parity
    let hidden_size = head_dim * num_attention_heads; // 64
    let intermediate_size = 64usize; // keep small
    let sliding_window = 16usize;
    let initial_context_length = 64.0f32;
    let rope_theta = 10000.0f32;
    let rope_scaling_factor = 1.0f32;
    let rope_ntk_alpha = 1.0f32;
    let rope_ntk_beta = 32.0f32;
    let swiglu_limit = 7.0f32;
    // The Python exporter will slice embeddings to 200014 rows
    let vocab_size = 200_014usize;

    // Write config.json
    let config = serde_json::json!({
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "num_experts": num_experts,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "intermediate_size": intermediate_size,
        "sliding_window": sliding_window,
        "initial_context_length": initial_context_length,
        "rope_theta": rope_theta,
        "rope_scaling_factor": rope_scaling_factor,
        "rope_ntk_alpha": rope_ntk_alpha,
        "rope_ntk_beta": rope_ntk_beta,
        "swiglu_limit": swiglu_limit,
    });
    fs::write(args.out.join("config.json"), serde_json::to_vec_pretty(&config)?)?;

    // Build tensors (own buffers and names until serialization)
    #[derive(Debug)]
    struct Entry { name: String, dtype: Dtype, shape: Vec<usize>, data: Vec<u8> }
    let mut owned: Vec<Entry> = Vec::new();

    // Embedding / Unembedding
    let emb = bf16_bytes(vocab_size * hidden_size, 1);
    owned.push(Entry { name: "embedding.weight".to_string(), dtype: Dtype::BF16, shape: vec![vocab_size, hidden_size], data: emb });
    let unemb = bf16_bytes(vocab_size * hidden_size, 2);
    owned.push(Entry { name: "unembedding.weight".to_string(), dtype: Dtype::BF16, shape: vec![vocab_size, hidden_size], data: unemb });

    for l in 0..num_hidden_layers {
        // attn.norm.scale
        let scale = bf16_bytes(hidden_size, 100 + l as u64);
        owned.push(Entry { name: format!("block.{l}.attn.norm.scale"), dtype: Dtype::BF16, shape: vec![hidden_size], data: scale });

        // fused qkv: rows = Hd * (n_heads + 2*kv), cols = hidden_size
        let qkv_rows = head_dim * (num_attention_heads + 2 * num_key_value_heads);
        let qkv_w = bf16_bytes(qkv_rows * hidden_size, 200 + l as u64);
        owned.push(Entry { name: format!("block.{l}.attn.qkv.weight"), dtype: Dtype::BF16, shape: vec![qkv_rows, hidden_size], data: qkv_w });
        let qkv_b = bf16_bytes(qkv_rows, 201 + l as u64);
        owned.push(Entry { name: format!("block.{l}.attn.qkv.bias"), dtype: Dtype::BF16, shape: vec![qkv_rows], data: qkv_b });

        // sinks
        let sinks = bf16_bytes(num_attention_heads, 210 + l as u64);
        owned.push(Entry { name: format!("block.{l}.attn.sinks"), dtype: Dtype::BF16, shape: vec![num_attention_heads], data: sinks });

        // attn.out
        let out_w = bf16_bytes(hidden_size * hidden_size, 220 + l as u64);
        owned.push(Entry { name: format!("block.{l}.attn.out.weight"), dtype: Dtype::BF16, shape: vec![hidden_size, hidden_size], data: out_w });
        let out_b = bf16_bytes(hidden_size, 221 + l as u64);
        owned.push(Entry { name: format!("block.{l}.attn.out.bias"), dtype: Dtype::BF16, shape: vec![hidden_size], data: out_b });

        // mlp.norm.scale
        let mlp_scale = bf16_bytes(hidden_size, 300 + l as u64);
        owned.push(Entry { name: format!("block.{l}.mlp.norm.scale"), dtype: Dtype::BF16, shape: vec![hidden_size], data: mlp_scale });

        // mlp.gate
        let gate_w = bf16_bytes(num_experts * hidden_size, 310 + l as u64);
        owned.push(Entry { name: format!("block.{l}.mlp.gate.weight"), dtype: Dtype::BF16, shape: vec![num_experts, hidden_size], data: gate_w });
        let gate_b = bf16_bytes(num_experts, 311 + l as u64);
        owned.push(Entry { name: format!("block.{l}.mlp.gate.bias"), dtype: Dtype::BF16, shape: vec![num_experts], data: gate_b });
    }

    // final norm.scale
    let final_scale = bf16_bytes(hidden_size, 400);
    owned.push(Entry { name: "norm.scale".to_string(), dtype: Dtype::BF16, shape: vec![hidden_size], data: final_scale });

    // MoE weights (quantized blocks + biased scales) and biases
    let per_e_blocks = 64usize;
    let per_e_scales = 32usize;
    for l in 0..num_hidden_layers {
        let mlp1_blocks = u8_bytes(per_e_blocks * num_experts, 500 + l as u64);
        owned.push(Entry { name: format!("block.{l}.mlp.mlp1_weight.blocks"), dtype: Dtype::U8, shape: vec![num_experts, per_e_blocks], data: mlp1_blocks });
        let mlp1_scales = u8_bytes(per_e_scales * num_experts, 501 + l as u64); // zeros/small
        owned.push(Entry { name: format!("block.{l}.mlp.mlp1_weight.scales"), dtype: Dtype::U8, shape: vec![num_experts, per_e_scales], data: mlp1_scales });
        let mlp1_bias = bf16_bytes(num_experts * (2 * intermediate_size), 502 + l as u64);
        owned.push(Entry { name: format!("block.{l}.mlp.mlp1_bias"), dtype: Dtype::BF16, shape: vec![num_experts, 2 * intermediate_size], data: mlp1_bias });

        let mlp2_blocks = u8_bytes(per_e_blocks * num_experts, 510 + l as u64);
        owned.push(Entry { name: format!("block.{l}.mlp.mlp2_weight.blocks"), dtype: Dtype::U8, shape: vec![num_experts, per_e_blocks], data: mlp2_blocks });
        let mlp2_scales = u8_bytes(per_e_scales * num_experts, 511 + l as u64);
        owned.push(Entry { name: format!("block.{l}.mlp.mlp2_weight.scales"), dtype: Dtype::U8, shape: vec![num_experts, per_e_scales], data: mlp2_scales });
        let mlp2_bias = bf16_bytes(num_experts * hidden_size, 512 + l as u64);
        owned.push(Entry { name: format!("block.{l}.mlp.mlp2_bias"), dtype: Dtype::BF16, shape: vec![num_experts, hidden_size], data: mlp2_bias });
    }

    // Serialize
    let mut views: Vec<(&str, TensorView)> = Vec::with_capacity(owned.len());
    for e in &owned {
        views.push((
            e.name.as_str(),
            TensorView::new(e.dtype, e.shape.clone(), &e.data)?,
        ));
    }
    let st_path = args.out.join("model.safetensors");
    safetensors::tensor::serialize_to_file(views, None, &st_path)
        .with_context(|| format!("serialize to {:?}", st_path))?;

    eprintln!("[gptoss_fixture] Wrote fixture to {:?}", args.out);
    Ok(())
}
