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
