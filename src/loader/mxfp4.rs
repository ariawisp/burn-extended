use burn_core as burn;

use burn_store::{ApplyResult, ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter, TensorSnapshot};
use burn_tensor::{backend::Backend, DType, TensorData};
use safetensors::{tensor::TensorView, SafeTensors};
use std::path::Path;

use super::common::burn_dtype_from_safetensors;

/// FP4 lookup table used by GPT‑OSS MXFP4 blocks.
const FP4_VALUES: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// A mapping from (blocks, scales) to a target parameter path for dequantized tensor.
#[derive(Clone, Debug)]
pub struct Mxfp4Spec {
    pub blocks: String,
    pub scales: String,
    pub target: String,
}

fn dequant_mxfp4(blocks_view: &TensorView, scales_view: &TensorView) -> TensorData {
    let dtype = burn_dtype_from_safetensors(blocks_view.dtype()).expect("blocks dtype");
    assert!(matches!(dtype, DType::U8));
    let blocks_shape = blocks_view.shape();
    let scales_shape = scales_view.shape();
    // Expect same prefix shape except that blocks last dim B will expand to 2*B
    assert_eq!(&blocks_shape[..blocks_shape.len() - 1], scales_shape);
    let rows_total: usize = scales_shape.iter().product();
    let b = blocks_shape[blocks_shape.len() - 1];

    let blocks_bytes = blocks_view.data();
    let scales_bytes = scales_view.data();
    // Scales are int8 stored in u8 buffer; convert to i32 then subtract bias 127
    let mut out = vec![0f32; rows_total * b * 2];
    for r in 0..rows_total {
        let scale_i = scales_bytes[r] as i32 - 127; // cast as unsigned -> int, then bias
        for i in 0..b {
            let byte = blocks_bytes[r * b + i];
            let lo = (byte & 0x0F) as usize;
            let hi = (byte >> 4) as usize;
            let v_lo = FP4_VALUES[lo];
            let v_hi = FP4_VALUES[hi];
            let idx = r * (b * 2) + (i * 2);
            let scale = (2.0f32).powi(scale_i);
            out[idx] = v_lo * scale;
            out[idx + 1] = v_hi * scale;
        }
    }
    let mut shape: Vec<usize> = blocks_shape.to_vec();
    *shape.last_mut().unwrap() = b * 2;
    TensorData::new(out, shape)
}

/// Test helper: dequantize MXFP4 from raw byte slices with given shapes.
pub fn dequant_mxfp4_bytes(
    blocks: &[u8],
    blocks_shape: &[usize],
    scales: &[u8],
    scales_shape: &[usize],
) -> Vec<f32> {
    let rows_total: usize = scales_shape.iter().product();
    let b = *blocks_shape.last().unwrap();
    let mut out = vec![0f32; rows_total * b * 2];
    for r in 0..rows_total {
        let scale_i = scales[r] as i32 - 127;
        let scale = (2.0f32).powi(scale_i);
        for i in 0..b {
            let byte = blocks[r * b + i];
            let lo = (byte & 0x0F) as usize;
            let hi = (byte >> 4) as usize;
            let idx = r * (b * 2) + (i * 2);
            out[idx] = FP4_VALUES[lo] * scale;
            out[idx + 1] = FP4_VALUES[hi] * scale;
        }
    }
    out
}

/// Load MXFP4 tensors and apply to a model after dequantization to f32/bf16.
pub fn load_safetensors_mxfp4_apply<B: Backend, M>(
    model: &mut M,
    path: &Path,
    specs: &[Mxfp4Spec],
    from_pytorch: bool,
    allow_partial: bool,
    validate: bool,
) -> anyhow::Result<ApplyResult>
where
    M: burn::module::Module<B> + Clone,
{
    use std::fs;
    let data = fs::read(path)?;
    let st = SafeTensors::deserialize(&data)?;

    let mut snaps: Vec<TensorSnapshot> = Vec::new();
    for spec in specs {
        let blocks = st.tensor(&spec.blocks)?;
        let scales = st.tensor(&spec.scales)?;
        let deq = dequant_mxfp4(&blocks, &scales);
        let snap = TensorSnapshot::from_data(
            deq,
            spec.target.split('.').map(|s| s.to_string()).collect(),
            vec!["SafeTensor".to_string()],
            burn_core::module::ParamId::new(),
        );
        if from_pytorch {
            if let Some(adapted) = PyTorchToBurnAdapter.adapt_tensor(&snap) {
                snaps.push(adapted);
            }
        } else {
            snaps.push(snap);
        }
    }

    let result = model.apply(snaps);
    if validate && !result.errors.is_empty() {
        anyhow::bail!("Import errors: {:?}", result.errors);
    }
    if !allow_partial && !result.missing.is_empty() {
        anyhow::bail!("Missing tensors: {:?}", result.missing);
    }
    Ok(result)
}
