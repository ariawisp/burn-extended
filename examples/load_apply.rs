//! Load a SafeTensors file and apply weights to a model using burn-store.

use std::path::PathBuf;

use burn_extended::loader::{load_apply_file, SimpleLoadConfig};
use clap::Parser;

#[derive(Parser, Debug)]
struct Args {
    /// Path to the .safetensors file
    #[arg(long)]
    path: PathBuf,
    /// Treat source as PyTorch (apply PyTorch->Burn adapter)
    #[arg(long, default_value_t = true)]
    from_pytorch: bool,
    /// Allow partial loading (skip missing tensors)
    #[arg(long, default_value_t = true)]
    allow_partial: bool,
    /// Validate after apply (fail on errors)
    #[arg(long, default_value_t = false)]
    validate: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    // Placeholder: replace with your actual model type and device
    // struct MyModel<B: Backend>(/* ... */);
    // let device = <B as Backend>::Device::default();
    // let mut model = MyModel::new(&device);

    // For demonstration, we only show how to set up the loader config.
    let cfg = SimpleLoadConfig {
        allow_partial: args.allow_partial,
        validate: args.validate,
        from_pytorch: args.from_pytorch,
    };

    // let result = load_apply_file(&mut model, &args.path, &cfg)?;
    // println!("{}", result);
    println!("Configured loader: {:?} -> {:?}", args, cfg);
    Ok(())
}
