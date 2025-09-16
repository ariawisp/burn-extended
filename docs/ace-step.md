# ACE‑Step — Inference Notes

Repository: https://github.com/ace-step/ACE-Step

This document captures ACE‑Step’s attention requirements and how `burn‑extended` supports them in a reusable way.

Key requirements
- Streaming MHA with a learned, data‑dependent attention bias.
- Sliding‑window attention and long‑sequence caching.
- RoPE position encoding.
- Autoregressive generation loop and sampling.

What burn-extended provides
- Streaming attention: `attention::ExtStreamingMultiHeadAttention{Config}` wraps the baseline streaming MHA and adds an `attn_bias: Option<&Tensor<_,4>>` term pre-softmax.
- Bias utilities: `bias::alibi_bias(...)` example and space for custom bias builders.
- Diffusion schedulers: `diffusion::{FlowMatchEuler, FlowMatchHeun, FlowMatchPingPong}` implement the shared `DiffusionScheduler` trait with `retrieve_timesteps` for schedule resampling.
- Guidance helpers: `diffusion::guidance::{cfg, cfg_double, apg, cfg_zero_star}` plus `MomentumBuffer` for APG-style conditioning.
- Cache/window policies: `cache::MhaCacheManager` + `WindowPolicy`.
- Generation tools: `sampling` processors + `generate` harness.

Additive attention bias
- Shape: `[B, n_heads, q_len, k_len]`, aligned to the active window.
- Construct per chunk using your policy network or a handcrafted heuristic, then pass via `ExtStreamingParams { attn_bias: Some(&bias) }`.

Sliding window
- Use `WindowPolicy` to decide per‑layer `AttnWindow` (e.g., fixed window for all layers).

RoPE
- If standard RoPE is needed, use Burn’s `RotaryEncodingConfig::init` and pass into streaming params.
- If extended context is desired later, use `rope::init_ntk_yarn(...)` from this repo.

Diffusion schedulers and guidance
- Choose a scheduler implementing `DiffusionScheduler`, call `set_timesteps(num_steps)`, then iterate `step(model_out, timestep, sample, omega)` using the sigmas/timesteps tensors.
- `retrieve_timesteps` lets you resample Diffusers-style schedules when mixing training/inference step counts.
- Pair the scheduler output with guidance helpers: `cfg`/`cfg_double` for classic classifier-free guidance, `apg` with a `MomentumBuffer` for ACE-Step's momentum projection, and `cfg_zero_star` when projecting onto high-rank negative prompts.

Backend init and generation harness (sketch)
```rust
use burn_wgpu::{Wgpu as B, WgpuDevice};
use burn_extended::{attention::*, cache::*, generate::*, sampling::*};

let device = WgpuDevice::default();
burn_wgpu::init_setup::<burn_wgpu::graphics::Metal>(&device, Default::default());

struct AceStepArModel<B: Backend> { /* ... */ }
impl<B: Backend> AutoregressiveModel<B> for AceStepArModel<B> {
    type Cache = MhaCacheManager<B>;
    fn init_cache(&self, batch: usize, device: &B::Device) -> Self::Cache { /* per-layer MHA caches */ }
    fn forward_logits(&self, tokens: Tensor<B,2,Int>, cache: &mut Self::Cache, start_pos: usize, window: AttnWindow) -> Tensor<B,2> {
        // embed -> per-layer { ExtStreamingMHA with attn_bias } -> norm -> head
        unimplemented!()
    }
}
```

Checkpoint loading (1:1 mapping) with store helper
```rust
use burn_extended::loader::{load_apply_file, SimpleLoadConfig};

let cfg = SimpleLoadConfig { allow_partial: true, validate: false, from_pytorch: true };
let result = load_apply_file(&mut model, std::path::Path::new("ace_step.safetensors"), &cfg)?;
assert!(result.is_success());
```

Open items
- Implement the precise bias‑generating module used by ACE‑Step and map its parameters.
- Provide a small example to visualize how the bias changes with context.
