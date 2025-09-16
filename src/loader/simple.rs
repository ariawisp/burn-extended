use burn_core as burn;

use burn::module::Module;
use burn_store::safetensors::SafetensorsError;
use burn_store::{ApplyResult, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use burn_tensor::backend::Backend;

use std::path::Path;

/// Simple loader configuration for 1:1 mapping cases.
#[derive(Clone, Debug, Default)]
pub struct SimpleLoadConfig {
    pub allow_partial: bool,
    pub validate: bool,
    pub from_pytorch: bool,
}

/// Load a model from a SafeTensors file with optional PyTorch→Burn adapter and options.
pub fn load_apply_file<B: Backend, M>(
    model: &mut M,
    path: &Path,
    cfg: &SimpleLoadConfig,
) -> Result<ApplyResult, SafetensorsError>
where
    M: Module<B> + Clone,
{
    let bytes = std::fs::read(path).map_err(|err| SafetensorsError::Other(err.to_string()))?;
    let mut store = SafetensorsStore::from_bytes(Some(bytes));
    if cfg.from_pytorch {
        store = store.with_from_adapter(PyTorchToBurnAdapter);
    }
    if cfg.allow_partial {
        store = store.allow_partial(true);
    }
    store = store.validate(cfg.validate);
    model.apply_from(&mut store)
}
