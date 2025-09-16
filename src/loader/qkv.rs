use burn_core as burn;

use burn::module::Module;
use burn_store::safetensors::SafetensorsError;
use burn_store::{
    ApplyResult, ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter, TensorSnapshot,
};
use burn_tensor::{backend::Backend, DType, TensorData};

use std::path::Path;

use safetensors::{tensor::TensorView, SafeTensors};

use super::common::{burn_dtype_from_safetensors, elem_size};

/// Strategy to split fused QKV along the last dimension.
#[derive(Clone, Debug)]
pub enum QkvSplitStrategy {
    Sizes {
        q: usize,
        k: usize,
        v: usize,
    },
    Heads {
        n_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    },
}

/// A single mapping from a fused QKV weight/bias to separate Q, K, V targets.
#[derive(Clone, Debug)]
pub struct QkvSplitSpec {
    pub fused_weight: String,
    pub fused_bias: Option<String>,
    pub q_weight: String,
    pub k_weight: String,
    pub v_weight: String,
    pub q_bias: Option<String>,
    pub k_bias: Option<String>,
    pub v_bias: Option<String>,
    pub strategy: QkvSplitStrategy,
}

fn split_along_last_dim(
    bytes: &[u8],
    shape: &[usize],
    dtype: DType,
    sizes: (usize, usize, usize),
) -> (TensorData, TensorData, TensorData) {
    let total_last = shape.last().copied().unwrap_or(0);
    let (q, k, v) = sizes;
    debug_assert_eq!(q + k + v, total_last);
    let item = elem_size(dtype);
    let outer: usize = shape[..shape.len() - 1].iter().product();
    let stride = total_last * item;
    let q_stride = q * item;
    let k_stride = k * item;
    let v_stride = v * item;

    let mut qb = Vec::with_capacity(bytes.len() * q / total_last);
    let mut kb = Vec::with_capacity(bytes.len() * k / total_last);
    let mut vb = Vec::with_capacity(bytes.len() * v / total_last);

    for i in 0..outer {
        let base = i * stride;
        qb.extend_from_slice(&bytes[base..base + q_stride]);
        kb.extend_from_slice(&bytes[base + q_stride..base + q_stride + k_stride]);
        vb.extend_from_slice(
            &bytes[base + q_stride + k_stride..base + q_stride + k_stride + v_stride],
        );
    }

    let mut q_shape = shape.to_vec();
    let mut k_shape = shape.to_vec();
    let mut v_shape = shape.to_vec();
    *q_shape.last_mut().unwrap() = q;
    *k_shape.last_mut().unwrap() = k;
    *v_shape.last_mut().unwrap() = v;

    (
        TensorData::from_bytes(qb, q_shape, dtype),
        TensorData::from_bytes(kb, k_shape, dtype),
        TensorData::from_bytes(vb, v_shape, dtype),
    )
}

fn view_to_snapshot(name: &str, view: &TensorView) -> Result<TensorSnapshot, SafetensorsError> {
    let dtype = burn_dtype_from_safetensors(view.dtype())?;
    let shape = view.shape().to_vec();
    let path_parts: Vec<String> = name.split('.').map(|s| s.to_string()).collect();
    let bytes = view.data().to_vec();
    let shape_for_fn = shape.clone();
    let data_fn = alloc::rc::Rc::new(move || {
        TensorData::from_bytes(bytes.clone(), shape_for_fn.clone(), dtype)
    });
    Ok(TensorSnapshot::from_closure(
        data_fn,
        dtype,
        shape,
        path_parts,
        vec!["SafeTensor".to_string()],
        burn_core::module::ParamId::new(),
    ))
}

/// Load a SafeTensors file, splitting fused QKV weights/biases into separate Q, K, V snapshots,
/// and apply to the model. For all non-fused entries, loads them as-is.
pub fn load_safetensors_qkv_split<B: Backend, M>(
    model: &mut M,
    path: &Path,
    splits: &[QkvSplitSpec],
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

    use hashbrown::HashSet;
    let mut fused_names: HashSet<&str> = HashSet::new();
    for s in splits {
        fused_names.insert(s.fused_weight.as_str());
        if let Some(b) = &s.fused_bias {
            fused_names.insert(b.as_str());
        }
    }

    let mut snapshots: Vec<TensorSnapshot> = Vec::new();

    for (name, view) in st.tensors() {
        if fused_names.contains(name.as_str()) {
            continue;
        }
        let snap = view_to_snapshot(&name, &view)?;
        let snap = if from_pytorch {
            if let Some(adapted) = PyTorchToBurnAdapter.adapt_tensor(&snap) {
                adapted
            } else {
                continue;
            }
        } else {
            snap
        };
        snapshots.push(snap);
    }

    for spec in splits {
        let wview = st
            .tensor(&spec.fused_weight)
            .map_err(|_| SafetensorsError::TensorNotFound(spec.fused_weight.clone()))?;
        let dtype = burn_dtype_from_safetensors(wview.dtype())?;
        let shape = wview.shape().to_vec();
        let bytes = wview.data().to_vec();
        let (q_size, k_size, v_size) = match spec.strategy {
            QkvSplitStrategy::Sizes { q, k, v } => (q, k, v),
            QkvSplitStrategy::Heads {
                n_heads,
                kv_heads,
                head_dim,
            } => (n_heads * head_dim, kv_heads * head_dim, kv_heads * head_dim),
        };
        let (qd, kd, vd) = split_along_last_dim(&bytes, &shape, dtype, (q_size, k_size, v_size));
        let qw = TensorSnapshot::from_data(
            qd,
            spec.q_weight.split('.').map(|s| s.to_string()).collect(),
            vec!["SafeTensor".to_string()],
            burn_core::module::ParamId::new(),
        );
        let kw = TensorSnapshot::from_data(
            kd,
            spec.k_weight.split('.').map(|s| s.to_string()).collect(),
            vec!["SafeTensor".to_string()],
            burn_core::module::ParamId::new(),
        );
        let vw = TensorSnapshot::from_data(
            vd,
            spec.v_weight.split('.').map(|s| s.to_string()).collect(),
            vec!["SafeTensor".to_string()],
            burn_core::module::ParamId::new(),
        );

        let mut split_snaps = vec![qw, kw, vw];

        if let Some(fb) = &spec.fused_bias {
            let bview = st
                .tensor(fb)
                .map_err(|_| SafetensorsError::TensorNotFound(fb.clone()))?;
            let bdtype = burn_dtype_from_safetensors(bview.dtype())?;
            let bshape = bview.shape().to_vec();
            let bbytes = bview.data().to_vec();
            let (qb, kb, vb) =
                split_along_last_dim(&bbytes, &bshape, bdtype, (q_size, k_size, v_size));
            if let Some(qb_name) = &spec.q_bias {
                split_snaps.push(TensorSnapshot::from_data(
                    qb,
                    qb_name.split('.').map(|s| s.to_string()).collect(),
                    vec!["SafeTensor".to_string()],
                    burn_core::module::ParamId::new(),
                ));
            }
            if let Some(kb_name) = &spec.k_bias {
                split_snaps.push(TensorSnapshot::from_data(
                    kb,
                    kb_name.split('.').map(|s| s.to_string()).collect(),
                    vec!["SafeTensor".to_string()],
                    burn_core::module::ParamId::new(),
                ));
            }
            if let Some(vb_name) = &spec.v_bias {
                split_snaps.push(TensorSnapshot::from_data(
                    vb,
                    vb_name.split('.').map(|s| s.to_string()).collect(),
                    vec!["SafeTensor".to_string()],
                    burn_core::module::ParamId::new(),
                ));
            }
        }

        if from_pytorch {
            for snap in split_snaps.into_iter() {
                if let Some(adapted) = PyTorchToBurnAdapter.adapt_tensor(&snap) {
                    snapshots.push(adapted);
                }
            }
        } else {
            snapshots.extend(split_snaps);
        }
    }

    let result = model.apply(snapshots);
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
