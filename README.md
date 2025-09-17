# burn-extended

Reusable building blocks that sit above `burn-core` so long-context transformers, video stacks, and diffusion pipelines can share the same primitives before they land upstream. The crate keeps implementations backend-agnostic and mirrors the APIs proposed in our Burn PRs.

## Capabilities

- Attention
  - Streaming multi-head attention with rolling K/V cache, sink token preservation, and `AttnWindow` control (`attention::StreamingMultiHeadAttention`).
  - Streaming MQA/GQA with optional additive bias and sinks helpers (`attention::StreamingMultiQueryAttention`).
  - Linear attention (kernelized positive feature maps) with padding mask support (`attention::LinearAttention`).
  - 1D mask helpers for padding, causal, windowed, and chunked scenarios (`attention::mask1d`).
- Positional encoding
  - NTK/YaRN rotary scaling for 1D RoPE (`rope::init_ntk_yarn`).
  - 3D rotary encoding over (frames, height, width) with streaming offsets (`rope::Rope3dEncoding`).
- Tokenization helpers
  - Conv2d/ConvTranspose2d patch embedding for images (`image::ImagePatchEmbedding`, `ImageUnpatchify`).
  - Conv3d/ConvTranspose3d patch embedding for video grids (`video::VideoPatchEmbedding`, `VideoUnpatchify`).
- Diffusion / flow-matching utilities
  - `DiffusionScheduler` trait plus Euler, Heun, and PingPong schedulers.
  - `retrieve_timesteps` resampling utility compatible with diffusers-style schedules.
  - Guidance helpers: CFG (single/double), APG with momentum, and zero-star projection (`diffusion::guidance`).
- Generation + caches
  - Streaming cache managers for MHA/MQA layers, window policies, samplers, and the autoregressive runner in `generate`.

Each module keeps comments close to the code so upstreaming stays mechanical. When Burn absorbs a feature you can flip consumers over to the upstream module and drop the shim here.

## Model Notes

The `docs/` folder captures how these pieces map onto reference projects:

- [GPT-OSS](docs/gpt-oss.md) — streaming GQA/MQA, sinks bias, NTK/YaRN, mask helpers, linear attention.
- [ACE-Step](docs/ace-step.md) — streaming MHA with additive bias, diffusion schedulers, guidance utilities.
- [Matrix-Game-2](docs/matrix-game-2.md) — streaming attention with sink tokens, video patching, 3D RoPE.

Those notes stay model-specific so this README can highlight the primitives at a glance.

## Roadmap

- Add SwiGLU-with-clamp helper and merge it into GPT-OSS blocks.
- Expose ready-to-wire decoder blocks for GPT-OSS and ACE-Step once upstream APIs stabilize.
- Provide toy diffusion examples that exercise the schedulers and guidance helpers.

## Usage

`burn-extended` targets inference on WGPU backends (Metal on macOS). Example entrypoints:

```bash
cargo run -p burn-extended --example gpt_oss
cargo run -p burn-extended --example ace_step
cargo run -p burn-extended --example matrix_game_2
```

Loader utilities expect checkpoints in `burn-store`/`safetensors` format and align with the split helpers documented in the model notes.

## Exporter Parity Test (tiny fixture)

- Generate a small GPT‑OSS‑like SafeTensors fixture (1 layer, 2 experts, head_dim=64):
  - `cargo run -p burn-extended --bin gptoss_fixture -- -o /tmp/gptoss_fixture`
- Export with the Rust exporter:
  - `cargo run -p burn-extended --bin gptoss_export -- -s /tmp/gptoss_fixture -d /tmp/model_rust.bin`
- Export with the Python exporter (requires deps):
  - `pip install -e ../harmony && pip install -e ../gpt-oss[metal]`
  - `python ../gpt-oss/gpt_oss/metal/scripts/create-local-model.py -s /tmp/gptoss_fixture -d /tmp/model_python.bin`
- Compare artifacts:
  - `cmp -l /tmp/model_rust.bin /tmp/model_python.bin || echo "differs"`

Notes
- The fixture embeds 200,014 tokens to match GPT‑OSS tokenizer filtering.
- On macOS, Metal kernels are built by `pip install -e ../gpt-oss[metal]`.

## Export GPT‑OSS 20B (original SafeTensors)

- Download the original checkpoint from HF:
  - `hf download openai/gpt-oss-20b --include "original/*" --local-dir gpt-oss-20b`
- Export to model.bin with Rust exporter:
  - `cargo run -p burn-extended --bin gptoss_export -- -s gpt-oss-20b/original -d gpt-oss-20b/metal/model.bin`
- Optional: run the Python Metal exporter for comparison:
  - `python ../gpt-oss/gpt_oss/metal/scripts/create-local-model.py -s gpt-oss-20b/original -d gpt-oss-20b/metal/model_python.bin`
- Quick diff:
  - `cmp -l gpt-oss-20b/metal/model.bin gpt-oss-20b/metal/model_python.bin || echo "differs"`

Tip: Set `RUST_LOG=warn` to print exporter section sizes and offsets.
