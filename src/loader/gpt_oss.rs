use burn_core as burn;

use burn::module::Module;
use burn_store::safetensors::SafetensorsError;
use burn_store::{ApplyResult, ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter, TensorSnapshot};
use burn_tensor::backend::Backend;
use burn_tensor::TensorData;
use safetensors::{tensor::TensorView, SafeTensors};
use std::path::Path;

use super::common::burn_dtype_from_safetensors;
use super::qkv::{QkvSplitSpec, QkvSplitStrategy, load_safetensors_qkv_split};
use super::mxfp4::{load_safetensors_mxfp4_apply, Mxfp4Spec};
use super::qkv::load_safetensors_map;
use half::bf16;
use burn_tensor::DType;

/// Build QKV split specifications for GPT-OSS fused attention weights.
/// Source names follow `block.{L}.attn.qkv.(weight|bias)` in the checkpoint.
/// Targets map into `layers.{L}.attn.(query|key|value).(weight|bias)`.
pub fn build_gptoss_qkv_splits(
    n_layers: usize,
    n_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Vec<QkvSplitSpec> {
    let mut specs = Vec::with_capacity(n_layers);
    for l in 0..n_layers {
        let fused_w = format!("block.{l}.attn.qkv.weight");
        let fused_b = format!("block.{l}.attn.qkv.bias");
        let q_w = format!("layers.{l}.attn.query.weight");
        let k_w = format!("layers.{l}.attn.key.weight");
        let v_w = format!("layers.{l}.attn.value.weight");
        let q_b = format!("layers.{l}.attn.query.bias");
        let k_b = format!("layers.{l}.attn.key.bias");
        let v_b = format!("layers.{l}.attn.value.bias");
        specs.push(QkvSplitSpec {
            fused_weight: fused_w,
            fused_bias: Some(fused_b),
            q_weight: q_w,
            k_weight: k_w,
            v_weight: v_w,
            q_bias: Some(q_b),
            k_bias: Some(k_b),
            v_bias: Some(v_b),
            strategy: QkvSplitStrategy::Heads {
                n_heads,
                kv_heads,
                head_dim,
            },
        });
    }
    specs
}

fn view_to_snapshot_with_path(
    view: &TensorView,
    target_path: &str,
    shape_override: Option<Vec<usize>>,
) -> Result<TensorSnapshot, SafetensorsError> {
    let dtype = burn_dtype_from_safetensors(view.dtype())?;
    let shape = shape_override.unwrap_or_else(|| view.shape().to_vec());
    let bytes = view.data().to_vec();
    Ok(TensorSnapshot::from_data(
        TensorData::from_bytes_vec(bytes, shape, dtype),
        target_path.split('.').map(|s| s.to_string()).collect(),
        vec!["SafeTensor".to_string()],
        burn_core::module::ParamId::new(),
    ))
}

/// Load GPT-OSS learned sinks parameters, reshaping from `[n_heads]` (checkpoint)
/// to `[kv_heads, groups]` expected by `StreamingMultiQueryAttention` when `learned_sinks=true`.
///
/// Source:  `block.{L}.attn.sinks`
/// Target:  `layers.{L}.attn.sinks`
pub fn load_gptoss_sinks<B: Backend, M>(
    model: &mut M,
    path: &Path,
    n_layers: usize,
    n_heads: usize,
    kv_heads: usize,
    from_pytorch: bool,
    allow_partial: bool,
    validate: bool,
) -> Result<ApplyResult, SafetensorsError>
where
    M: Module<B> + Clone,
{
    use std::fs;
    let data = fs::read(path).map_err(|err| SafetensorsError::Other(err.to_string()))?;
    let st =
        SafeTensors::deserialize(&data).map_err(|err| SafetensorsError::Other(err.to_string()))?;

    let groups = n_heads / kv_heads;
    let mut snaps: Vec<TensorSnapshot> = Vec::new();
    for l in 0..n_layers {
        let name = format!("block.{l}.attn.sinks");
        let view = st
            .tensor(&name)
            .map_err(|_| SafetensorsError::TensorNotFound(name.clone()))?;
        let target = format!("layers.{l}.attn.sinks");
        let reshaped = view_to_snapshot_with_path(&view, &target, Some(vec![kv_heads, groups]))?;
        if from_pytorch {
            if let Some(adapted) = PyTorchToBurnAdapter.adapt_tensor(&reshaped) {
                snaps.push(adapted);
            }
        } else {
            snaps.push(reshaped);
        }
    }

    let result = model.apply(snaps);
    if validate && !result.errors.is_empty() {
        return Err(SafetensorsError::ValidationFailed(format!(
            "Import errors: {:?}",
            result.errors
        )));
    }
    if !allow_partial && !result.missing.is_empty() {
        return Err(SafetensorsError::TensorNotFound(format!(
            "Missing tensors: {:?}",
            result.missing
        )));
    }
    Ok(result)
}

/// Convenience loader: apply fused QKV split and sinks mapping.
pub fn load_gptoss_weights<B: Backend, M>(
    model: &mut M,
    path: &Path,
    n_layers: usize,
    n_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    from_pytorch: bool,
    allow_partial: bool,
    validate: bool,
) -> Result<(ApplyResult, ApplyResult), SafetensorsError>
where
    M: Module<B> + Clone,
{
    let splits = build_gptoss_qkv_splits(n_layers, n_heads, kv_heads, head_dim);
    let res_qkv = load_safetensors_qkv_split::<B, M>(
        model,
        path,
        &splits,
        from_pytorch,
        allow_partial,
        validate,
    )?;
    let res_sinks = load_gptoss_sinks::<B, M>(
        model,
        path,
        n_layers,
        n_heads,
        kv_heads,
        from_pytorch,
        allow_partial,
        validate,
    )?;
    Ok((res_qkv, res_sinks))
}

/// Load lm_head directly from SafeTensors without PyTorch adapter (to match model.bin orientation).
/// Maps: `unembedding.weight` -> `lm_head.weight`.
pub fn load_gptoss_lm_head<B: Backend, M>(
    model: &mut M,
    path: &std::path::Path,
    allow_partial: bool,
    validate: bool,
) -> Result<ApplyResult, SafetensorsError>
where
    M: Module<B> + Clone,
{
    let maps = vec![("unembedding.weight".to_string(), "lm_head.weight".to_string())];
    super::qkv::load_safetensors_map::<B, _>(
        model,
        path,
        &maps,
        /*from_pytorch*/ false,
        allow_partial,
        validate,
    )
}

/// Transform fused QKV laid out by rows (rows = head_dim*(n_heads+2*kv_heads)) to interleave Q/K halves and
/// pre-scale Q by 0.5 and K by 0.25, then split by rows and apply to target module params.
pub fn load_gptoss_qkv_rows<B: Backend, M>(
    model: &mut M,
    path: &std::path::Path,
    n_layers: usize,
    n_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    allow_partial: bool,
    validate: bool,
) -> Result<ApplyResult, SafetensorsError>
where
    M: Module<B> + Clone,
{
    use std::fs;
    let data = fs::read(path).map_err(|err| SafetensorsError::Other(err.to_string()))?;
    let st = SafeTensors::deserialize(&data).map_err(|e| SafetensorsError::Other(e.to_string()))?;

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
    fn interleave_qk(qk_in: &[bf16], heads: usize, head_dim: usize, cols: usize) -> Vec<bf16> {
        let half = head_dim / 2;
        let mut out = vec![bf16::ZERO; heads * head_dim * cols];
        for h in 0..heads {
            for i in 0..half {
                for c in 0..cols {
                    let row0 = h * head_dim + 0 * half + i;
                    let row1 = h * head_dim + 1 * half + i;
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
    fn transform_qkv_bf16(
        bytes: &[u8],
        rows: usize,
        cols: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Vec<u8> {
        let qkv_vals = bf16_from_le_bytes(bytes);
        let qk_rows = head_dim * (n_heads + n_kv_heads);
        let qk_in = &qkv_vals[0..qk_rows * cols];
        let mut qk_scaled = interleave_qk(qk_in, n_heads + n_kv_heads, head_dim, cols);
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
        let v_in = &qkv_vals[qk_rows * cols..];
        let mut out_vals = Vec::with_capacity(rows * cols);
        for h in 0..n_heads {
            let base = (h * head_dim) * cols;
            out_vals.extend_from_slice(&qk_scaled[base..base + head_dim * cols]);
        }
        for kh in 0..n_kv_heads {
            let h = n_heads + kh;
            let base = (h * head_dim) * cols;
            out_vals.extend_from_slice(&qk_scaled[base..base + head_dim * cols]);
        }
        out_vals.extend_from_slice(v_in);
        bf16_to_le_bytes(&out_vals)
    }

    let mut snaps: Vec<TensorSnapshot> = Vec::new();
    for l in 0..n_layers {
        let wname = format!("block.{l}.attn.qkv.weight");
        let bname = format!("block.{l}.attn.qkv.bias");
        let w = st.tensor(&wname).map_err(|_| SafetensorsError::TensorNotFound(wname.clone()))?;
        let b = st.tensor(&bname).map_err(|_| SafetensorsError::TensorNotFound(bname.clone()))?;
        let rows = w.shape()[0];
        let cols = w.shape()[1];
        let expected_rows = head_dim * (n_heads + 2 * kv_heads);
        if rows != expected_rows {
            return Err(SafetensorsError::ValidationFailed(format!(
                "unexpected qkv rows: got {}, expected {}",
                rows, expected_rows
            )));
        }
        let w_t = transform_qkv_bf16(w.data(), rows, cols, n_heads, kv_heads, head_dim);
        let b_t = transform_qkv_bf16(b.data(), rows, 1, n_heads, kv_heads, head_dim);

        let q_rows = n_heads * head_dim;
        let kv_rows = kv_heads * head_dim;
        let row_bytes = cols * 2;
        let qw = w_t[0..q_rows * row_bytes].to_vec();
        let kw = w_t[q_rows * row_bytes..(q_rows + kv_rows) * row_bytes].to_vec();
        let vw = w_t[(q_rows + kv_rows) * row_bytes..].to_vec();
        let qb = &b_t[0..q_rows * 2];
        let kb = &b_t[q_rows * 2..(q_rows + kv_rows) * 2];
        let vb = &b_t[(q_rows + kv_rows) * 2..];

        let mut push = |name: String, bytes: Vec<u8>, shape: Vec<usize>| {
            snaps.push(TensorSnapshot::from_data(
                TensorData::from_bytes_vec(bytes, shape, DType::BF16),
                name.split('.').map(|s| s.to_string()).collect(),
                vec!["SafeTensor".to_string()],
                burn_core::module::ParamId::new(),
            ));
        };
        push(format!("layers.{l}.attn.query.weight"), qw, vec![q_rows, cols]);
        push(format!("layers.{l}.attn.key.weight"), kw, vec![kv_rows, cols]);
        push(format!("layers.{l}.attn.value.weight"), vw, vec![kv_rows, cols]);
        push(format!("layers.{l}.attn.query.bias"), qb.to_vec(), vec![q_rows]);
        push(format!("layers.{l}.attn.key.bias"), kb.to_vec(), vec![kv_rows]);
        push(format!("layers.{l}.attn.value.bias"), vb.to_vec(), vec![kv_rows]);
    }

    let result = model.apply(snaps);
    if validate && !result.errors.is_empty() {
        return Err(SafetensorsError::ValidationFailed(format!(
            "Import errors: {:?}",
            result.errors
        )));
    }
    if !allow_partial && !result.missing.is_empty() {
        return Err(SafetensorsError::TensorNotFound(format!(
            "Missing tensors: {:?}",
            result.missing
        )));
    }
    Ok(result)
}

/// Build MXFP4 specs for GPT‑OSS MoE weights.
pub fn build_gptoss_moe_mxfp4_specs(n_layers: usize) -> Vec<Mxfp4Spec> {
    let mut specs = Vec::with_capacity(n_layers * 2);
    for l in 0..n_layers {
        specs.push(Mxfp4Spec {
            blocks: format!("block.{l}.mlp.mlp1_weight.blocks"),
            scales: format!("block.{l}.mlp.mlp1_weight.scales"),
            target: format!("layers.{l}.mlp.mlp1_weight"),
        });
        specs.push(Mxfp4Spec {
            blocks: format!("block.{l}.mlp.mlp2_weight.blocks"),
            scales: format!("block.{l}.mlp.mlp2_weight.scales"),
            target: format!("layers.{l}.mlp.mlp2_weight"),
        });
    }
    specs
}

/// Load MoE MXFP4 weights and BF16 biases/gate/norm for GPT‑OSS.
pub fn load_gptoss_moe<B: Backend, M>(
    model: &mut M,
    path: &Path,
    n_layers: usize,
    from_pytorch: bool,
    allow_partial: bool,
    validate: bool,
) -> Result<ApplyResult, SafetensorsError>
where
    M: Module<B> + Clone,
{
    let specs = build_gptoss_moe_mxfp4_specs(n_layers);
    let _ = load_safetensors_mxfp4_apply::<B, _>(
        model,
        path,
        &specs,
        from_pytorch,
        allow_partial,
        validate,
    )
    .map_err(|e| SafetensorsError::Other(e.to_string()))?;

    // Map BF16 biases and gate & norm parameters.
    let mut maps = Vec::new();
    for l in 0..n_layers {
        maps.push((
            format!("block.{l}.mlp.mlp1_bias"),
            format!("layers.{l}.mlp.mlp1_bias"),
        ));
        maps.push((
            format!("block.{l}.mlp.mlp2_bias"),
            format!("layers.{l}.mlp.mlp2_bias"),
        ));
        maps.push((
            format!("block.{l}.mlp.gate.weight"),
            format!("layers.{l}.mlp.gate.weight"),
        ));
        maps.push((
            format!("block.{l}.mlp.gate.bias"),
            format!("layers.{l}.mlp.gate.bias"),
        ));
        maps.push((
            format!("block.{l}.mlp.norm.scale"),
            format!("layers.{l}.mlp.norm.scale"),
        ));
    }
    load_safetensors_map::<B, _>(
        model,
        path,
        &maps,
        from_pytorch,
        allow_partial,
        validate,
    )
}
