use std::path::PathBuf;
use std::process::Command;

use burn_core as burn;
use burn_ndarray::NdArray as B;
use burn_tensor::{Int, Tensor};
use burn_extended::generate::AutoregressiveModel;

use burn_extended::loader::{
    build_gptoss_qkv_splits, load_gptoss_sinks, load_safetensors_map, load_gptoss_qkv_rows, load_gptoss_lm_head,
};
use burn_extended::loader::modelbin::load_modelbin_into;
use burn_extended::models::gpt_oss::{GptOssConfig, GptOssModel};
use burn_store::{TensorSnapshot, ModuleSnapshot};
use burn_tensor::{DType, TensorData};
use safetensors::SafeTensors;

fn bin(name: &str) -> Option<PathBuf> {
    let key = format!("CARGO_BIN_EXE_{}", name);
    std::env::var_os(&key).map(PathBuf::from)
}

fn tmp_dir(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("{}_{}", name, nanos));
    p
}

fn run_fixture() -> (PathBuf, PathBuf) {
    let out = tmp_dir("gptoss_fixture_infer");
    std::fs::create_dir_all(&out).unwrap();
    let rust_bin = out.join("model_rust.bin");

    // Generate fixture (config.json + model.safetensors)
    let status = if let Some(fixture) = bin("gptoss_fixture") {
        Command::new(&fixture)
            .args(["-o", out.to_str().unwrap()])
            .status()
            .expect("spawn fixture")
    } else {
        Command::new("cargo")
            .args([
                "run",
                "-q",
                "-p",
                "burn-extended",
                "--bin",
                "gptoss_fixture",
                "--",
                "-o",
                out.to_str().unwrap(),
            ])
            .status()
            .expect("cargo run fixture")
    };
    assert!(status.success(), "fixture generation failed");
    assert!(out.join("config.json").exists());
    assert!(out.join("model.safetensors").exists());

    // Export model.bin via Rust exporter
    let status = if let Some(exporter) = bin("gptoss_export") {
        Command::new(&exporter)
            .args(["-s", out.to_str().unwrap(), "-d", rust_bin.to_str().unwrap()])
            .status()
            .expect("spawn exporter")
    } else {
        Command::new("cargo")
            .args([
                "run",
                "-q",
                "-p",
                "burn-extended",
                "--bin",
                "gptoss_export",
                "--",
                "-s",
                out.to_str().unwrap(),
                "-d",
                rust_bin.to_str().unwrap(),
            ])
            .status()
            .expect("cargo run exporter")
    };
    assert!(status.success(), "rust exporter failed");
    (out, rust_bin)
}

#[test]
fn attention_only_inference_parity_fixture() {
    let (dir, modelbin) = run_fixture();
    let cfg_path = dir.join("config.json");
    let ckpt_path = dir.join("model.safetensors");

    // Load config and init two identical models (disable MoE)
    let mut cfg = GptOssConfig::from_config_json(&cfg_path).unwrap();
    cfg.disable_moe = true;
    let device = &burn_ndarray::NdArrayDevice::Cpu;
    let mut model_safe: GptOssModel<B> = cfg.clone().init::<B>(device);
    let mut model_bin: GptOssModel<B> = cfg.clone().init::<B>(device);

    // Load SafeTensors QKV from row-fused format with exporter transform
    let head_dim = cfg.head_dim;
    let _ = load_gptoss_qkv_rows::<B, _>(
        &mut model_safe,
        &ckpt_path,
        cfg.n_layers,
        cfg.n_heads,
        cfg.kv_heads,
        head_dim,
        /*allow_partial*/ true,
        /*validate*/ false,
    )
    .unwrap();
    let _ = load_gptoss_sinks::<B, _>(
        &mut model_safe,
        &ckpt_path,
        cfg.n_layers,
        cfg.n_heads,
        cfg.kv_heads,
        true,
        true,
        false,
    )
    .unwrap();

    // Map embeddings, attn.out, norms, unembedding
    let mut maps: Vec<(String, String)> = Vec::new();
    maps.push(("embedding.weight".into(), "tok_emb.weight".into()));
    for l in 0..cfg.n_layers {
        maps.push((
            format!("block.{l}.attn.norm.scale"),
            format!("layers.{l}.norm_attn.scale"),
        ));
        maps.push((
            format!("block.{l}.attn.out.weight"),
            format!("layers.{l}.attn.output.weight"),
        ));
        maps.push((
            format!("block.{l}.attn.out.bias"),
            format!("layers.{l}.attn.output.bias"),
        ));
    }
    maps.push(("norm.scale".into(), "norm_final.scale".into()));
    let _ = load_safetensors_map::<B, _>(
        &mut model_safe,
        &ckpt_path,
        &maps,
        /*from_pytorch*/ true,
        /*allow_partial*/ true,
        /*validate*/ false,
    )
    .unwrap();
    // Load lm_head without PyTorch adapter to match model.bin orientation
    let lm_map = vec![("unembedding.weight".to_string(), "lm_head.weight".to_string())];
    let _ = load_safetensors_map::<B, _>(
        &mut model_safe,
        &ckpt_path,
        &lm_map,
        /*from_pytorch*/ false,
        /*allow_partial*/ true,
        /*validate*/ false,
    )
    .unwrap();

    // Load model.bin weights into the other model (skip MoE)
    let _ = load_modelbin_into::<B, _>(&mut model_bin, &modelbin, /*validate*/ false, /*skip_moe*/ true).unwrap();
    // For this loader, we pre-scaled Q/K during transform; use pre_scaled_qk=true
    model_safe.set_pre_scaled_qk(true);

    // Build a tiny prompt and compare logits
    let vocab = cfg.vocab_size;
    let b = 1;
    let t = 4;
    let tokens: Vec<usize> = (0..(b * t)).map(|i| i % vocab).collect();
    let toks = Tensor::<B, 1, Int>::from_ints(tokens.as_slice(), device).reshape([b, t]);

    let mut cache_safe = model_safe.init_cache(b, device);
    let mut cache_bin = model_bin.init_cache(b, device);
    // Also compare Q/K after projection+RoPE to localize mismatches
    let (q_safe, k_safe, v_safe) = model_safe.debug_layer_attn_qkv(toks.clone(), 0, 0);
    let (q_bin, k_bin, v_bin) = model_bin.debug_layer_attn_qkv(toks.clone(), 0, 0);
    let q_s = q_safe.into_data().convert::<f32>().into_vec::<f32>().unwrap();
    let q_b = q_bin.into_data().convert::<f32>().into_vec::<f32>().unwrap();
    let mut max_abs_q = 0f32; let mut sum_sq_q = 0f32; for i in 0..q_s.len() { let d = (q_s[i]-q_b[i]).abs(); max_abs_q = max_abs_q.max(d); sum_sq_q += d*d; }
    let rmse_q = (sum_sq_q / (q_s.len() as f32)).sqrt();
    assert!(rmse_q < 1e-3, "Q RMSE too high: {} (max_abs={})", rmse_q, max_abs_q);
    let k_s = k_safe.into_data().convert::<f32>().into_vec::<f32>().unwrap();
    let k_bv = k_bin.into_data().convert::<f32>().into_vec::<f32>().unwrap();
    let mut max_abs_k = 0f32; let mut sum_sq_k = 0f32; for i in 0..k_s.len() { let d = (k_s[i]-k_bv[i]).abs(); max_abs_k = max_abs_k.max(d); sum_sq_k += d*d; }
    let rmse_k = (sum_sq_k / (k_s.len() as f32)).sqrt();
    assert!(rmse_k < 1e-3, "K RMSE too high: {} (max_abs={})", rmse_k, max_abs_k);
    let v_s = v_safe.into_data().convert::<f32>().into_vec::<f32>().unwrap();
    let v_b = v_bin.into_data().convert::<f32>().into_vec::<f32>().unwrap();
    let mut max_abs_v = 0f32; let mut sum_sq_v = 0f32; for i in 0..v_s.len() { let d = (v_s[i]-v_b[i]).abs(); max_abs_v = max_abs_v.max(d); sum_sq_v += d*d; }
    let rmse_v = (sum_sq_v / (v_s.len() as f32)).sqrt();
    assert!(rmse_v < 1e-3, "V RMSE too high: {} (max_abs={})", rmse_v, max_abs_v);

    // Compare learned sinks and attn.out weights too
    let sinks_s = model_safe.debug_get_sinks(0).unwrap().into_data().convert::<f32>().into_vec::<f32>().unwrap();
    let sinks_b = model_bin.debug_get_sinks(0).unwrap().into_data().convert::<f32>().into_vec::<f32>().unwrap();
    let mut max_abs_s = 0f32; let mut sum_sq_s = 0f32; for i in 0..sinks_s.len() { let d = (sinks_s[i]-sinks_b[i]).abs(); max_abs_s = max_abs_s.max(d); sum_sq_s += d*d; }
    let rmse_s = (sum_sq_s / (sinks_s.len() as f32)).sqrt();
    assert!(rmse_s < 1e-6, "sinks mismatch rmse={} max_abs={}", rmse_s, max_abs_s);
    let w_safe = model_safe.debug_get_attn_out_weight(0).into_data().convert::<f32>().into_vec::<f32>().unwrap();
    let w_bin = model_bin.debug_get_attn_out_weight(0).into_data().convert::<f32>().into_vec::<f32>().unwrap();
    let mut max_abs_w = 0f32; let mut sum_sq_w = 0f32; for i in 0..w_safe.len() { let d = (w_safe[i]-w_bin[i]).abs(); max_abs_w = max_abs_w.max(d); sum_sq_w += d*d; }
    let rmse_w = (sum_sq_w / (w_safe.len() as f32)).sqrt();
    assert!(rmse_w < 1e-6, "attn.out.weight mismatch rmse={} max_abs={}", rmse_w, max_abs_w);

    // Compare embedding, norm_final, and lm_head weights
    let emb_s = model_safe.debug_get_tok_emb().into_data().convert::<f32>().into_vec::<f32>().unwrap();
    let emb_b = model_bin.debug_get_tok_emb().into_data().convert::<f32>().into_vec::<f32>().unwrap();
    let mut max_abs = 0f32; let mut sum_sq = 0f32; for i in 0..emb_s.len() { let d = (emb_s[i]-emb_b[i]).abs(); max_abs = max_abs.max(d); sum_sq += d*d; }
    let rmse = (sum_sq / (emb_s.len() as f32)).sqrt();
    assert!(rmse < 1e-6, "embedding mismatch rmse={} max_abs={}", rmse, max_abs);
    let nf_s = model_safe.debug_get_norm_final().into_data().convert::<f32>().into_vec::<f32>().unwrap();
    let nf_b = model_bin.debug_get_norm_final().into_data().convert::<f32>().into_vec::<f32>().unwrap();
    let mut max_abs = 0f32; let mut sum_sq = 0f32; for i in 0..nf_s.len() { let d = (nf_s[i]-nf_b[i]).abs(); max_abs = max_abs.max(d); sum_sq += d*d; }
    let rmse = (sum_sq / (nf_s.len() as f32)).sqrt();
    assert!(rmse < 1e-6, "norm_final mismatch rmse={} max_abs={}", rmse, max_abs);
    // Load lm_head with GPT-OSS-specific loader (no PyTorch adapter) to match model.bin
    let _ = load_gptoss_lm_head::<B, _>(&mut model_safe, &ckpt_path, true, false).unwrap();

    // Compare attention weights and raw scores for layer 0
    let wts_s = model_safe.debug_attn_weights(toks.clone(), 0, 0).into_data().convert::<f32>().into_vec::<f32>().unwrap();
    let wts_b = model_bin.debug_attn_weights(toks.clone(), 0, 0).into_data().convert::<f32>().into_vec::<f32>().unwrap();
    let mut max_abs_wt = 0f32; let mut sum_sq_wt = 0f32; for i in 0..wts_s.len() { let d = (wts_s[i]-wts_b[i]).abs(); max_abs_wt = max_abs_wt.max(d); sum_sq_wt += d*d; }
    let rmse_wt = (sum_sq_wt / (wts_s.len() as f32)).sqrt();
    assert!(rmse_wt < 1e-6, "attn weights mismatch rmse={} max_abs={}", rmse_wt, max_abs_wt);
    let scr_s = model_safe.debug_attn_scores(toks.clone(), 0, 0).into_data().convert::<f32>().into_vec::<f32>().unwrap();
    let scr_b = model_bin.debug_attn_scores(toks.clone(), 0, 0).into_data().convert::<f32>().into_vec::<f32>().unwrap();
    let mut max_abs_sc = 0f32; let mut sum_sq_sc = 0f32; for i in 0..scr_s.len() { let d = (scr_s[i]-scr_b[i]).abs(); max_abs_sc = max_abs_sc.max(d); sum_sq_sc += d*d; }
    let rmse_sc = (sum_sq_sc / (scr_s.len() as f32)).sqrt();
    assert!(rmse_sc < 1e-6, "attn scores mismatch rmse={} max_abs={}", rmse_sc, max_abs_sc);

    // Force lm_head to be identical and compare hidden after final norm + logits
    let w_bin_t = model_bin.debug_get_lm_head();
    model_safe.debug_set_lm_head(w_bin_t);
    let hidden_s = model_safe.debug_hidden_after_final_norm(toks.clone(), 0).into_data().convert::<f32>().into_vec::<f32>().unwrap();
    let hidden_b = model_bin.debug_hidden_after_final_norm(toks.clone(), 0).into_data().convert::<f32>().into_vec::<f32>().unwrap();
    let mut max_abs_h = 0f32; let mut sum_sq_h = 0f32; for i in 0..hidden_s.len() { let d = (hidden_s[i]-hidden_b[i]).abs(); max_abs_h = max_abs_h.max(d); sum_sq_h += d*d; }
    let rmse_h = (sum_sq_h / (hidden_s.len() as f32)).sqrt();
    assert!(rmse_h < 1e-3, "hidden post-norm mismatch rmse={} max_abs={}", rmse_h, max_abs_h);
    let logits_safe = model_safe.forward_logits(toks.clone(), &mut cache_safe, 0, burn_extended::attention::AttnWindow::Full);
    let logits_bin = model_bin.forward_logits(toks, &mut cache_bin, 0, burn_extended::attention::AttnWindow::Full);

    let a = logits_safe
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("to vec");
    let bdat = logits_bin
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("to vec");
    assert_eq!(a.len(), bdat.len());
    let mut max_abs = 0f32;
    let mut sum_sq = 0f32;
    for i in 0..a.len() {
        let d = (a[i] - bdat[i]).abs();
        max_abs = max_abs.max(d);
        sum_sq += d * d;
    }
    let rmse = (sum_sq / (a.len() as f32)).sqrt();
    assert!(rmse < 5e-2, "RMSE too high: {} (max_abs={})", rmse, max_abs);
}
