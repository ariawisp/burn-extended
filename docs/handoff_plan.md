# burn-extended — Handoff Plan (GPT‑OSS model.bin + MoE)

This doc records current status, what’s been completed, and a concrete plan for the next engineer to take over. We ship the “one correct path” (no feature flags, no CLI overrides for shapes) and are now focused on performance parity for MoE on GPU.

## Scope Summary
- Primary: Correct GPT‑OSS inference in Burn with model.bin reader and MoE (MXFP4). SafeTensors is canonical; exporter/reader align with GPT‑OSS Metal.
- Performance goal: MoE execution fully on device (no CPU dequant, minimal H2D), competitive with GPT‑OSS Metal/Triton.

---

## What’s Done (Current State)

Numerics & IO
- Exporter parity with Python (headers, tokenizer regex/UUIDs, embeddings, QKV transform with pre‑scaled Q/K, sinks, attn out, gate/norm, final norm, MoE blocks/scales/biases with proper alignment & UE8 offset).
- Structural verifier and tiny fixture tests; parity validated for attention and final logits.
- Reader parses model.bin headers/tokenizer and loads non‑MoE weights; decoupled attention supported (d_model ≠ n_heads*head_dim) with pre‑scaled Q/K honored.
- Learned sinks reshaped to [kv_heads, groups] and applied as bias.

Model/runtime
- Single correct CLI: all shapes from header, MoE always on, unlimited tokens by default (stop tokens terminate).
- SwiGLU corrected to GPT‑OSS semantics (alpha=1.702, interleaved, clamp, out = swish(x_glu) · (x_lin + 1)).
- Prefill processed as a single tile per layer to reduce dispatch overhead.
- Added robust debug logging (no flags): layer/tile timing and approximate H2D per MoE tile.

MoE execution (in progress)
- Routing (top‑k) runs on GPU.
- FP4 decode performed on GPU using per‑tile byte slices (nibble decode with f32 math, per‑group exp2 scales), no CPU dequant.

---

## Known Performance Bottleneck (Root Cause)

Even after routing + FP4 decode moved to GPU, the path still incurs too many GPU dispatches and transient H2D for index/gather flows. The correct solution is the same as GPT‑OSS Metal/Triton: keep quantized blocks/scales resident on device (u8), and use fused kernels that read raw u8 buffers directly (no int‑index inflation), performing W1 + fused SwiGLU and W2 with compact gather/scatter on device.

The recent regression shown in logs (approx_H2D ~3.2 GiB/tile) came from uploading full expert matrices per layer; this has been removed. We’re back to per‑tile byte slices (orders of magnitude smaller), but fused kernels + residency are required to reach expected speed.

---

## Next Engineer: Plan of Record (Single Path, No Flags)

1) Persistent Device Buffers (u8)
- Upload per‑layer raw u8 buffers once:
  - W1 blocks [E, 2f, d/2 bytes], W1 scales [E, 2f, gpr]
  - W2 blocks [E, d, f/2 bytes], W2 scales [E, d, gpr]
  - BF16 biases
- Expose a small descriptor (QuantLinearMxFp4) with layout metadata (rows, cols, bpr, gpr, strides).

2) Compact Routing on GPU
- Kernel: top‑k per token row + softmax; output compact indices [n,k] (u32) + weights [n,k] (f32).
- Keep `gather/scatter` indices on device.

3) Fused WGSL Kernels
- Kernel A (W1 + SwiGLU):
  - Read u8 blocks + per‑group scales directly; FP4 decode (nibble: & 0xF, >> 4) + exp2(group_exp − 14 − 127); matmul to [m, 2f]; fused SwiGLU + clamp → [m, f].
- Kernel B (W2 + Scatter):
  - Read u8 blocks/scales; matmul [m, f] × [f, d] → [m, d]; multiply by gate weight; scatter‑add back to [n, d] using compact indices.

4) Integrate into MoE forward
- Replace the current decompose/gather/reshape graph with: route (kernel) → compact gather (device) → fused W1 (kernel) → fused W2 (kernel) → scatter.
- Attention path remains unchanged (already correct & reasonably fast).

5) Instrumentation & Validation
- Keep existing per‑layer/tile timing; add per‑kernel timings behind `--debug`.
- Validate numerics against CPU path on small configs (layer outputs and final logits).

Acceptance Criteria
- 20B model.bin runs at expected GPU speed for 64‑token prefills and steady decode (MoE is not the bottleneck).
- No CPU dequant; no large H2D spikes; memory stable.
- Fixture parity & structural verifier remain green.

---

## How to Run (Current)
- Export: `cargo run -p burn-extended --bin gptoss_export -- -s ~/gpt-oss-20b/original -d ~/gpt-oss-20b/metal/model.bin`
- Verify: `cargo run -p burn-extended --bin gptoss_verify -- -s ~/gpt-oss-20b/original ~/gpt-oss-20b/metal/model.bin`
- Inference (model.bin):
  - `cargo run -p burn-extended --bin gptoss_modelbin_infer -- --model ~/gpt-oss-20b/metal/model.bin`
  - With logs: add `--debug` (prefill tiles, per‑layer times, approx H2D)

Expected behavior (until fused kernels land)
- Correct outputs; prefill+decode will be slower than desired, but should not show multi‑GiB H2D spikes.

---

## Code Map
- Exporter / Verifier: `src/bin/gptoss_export.rs`, `src/bin/gptoss_verify.rs`
- Reader: `src/loader/modelbin.rs`
- Inference: `src/bin/gptoss_modelbin_infer.rs`
- Model / Config: `src/models/gpt_oss.rs`, `src/models/gpt_oss_config.rs`
- MoE module (routing + current GPU FP4 decode path): `src/moe/mod.rs`
- Attention: `src/attention/*`
- Tests / parity: `tests/inference_parity_fixture.rs`

---

## Risks & Notes
- WGSL validation: avoid mixing int/float in the same expression; do nibble math in f32 then cast.
- Persistent buffers: keep as raw u8; kernels should read them directly (no i64 index inflation).
- Memory: persistent u8 buffers per layer are large; ensure unified memory budget is respected (Apple Silicon is fine for 20B).
- Dispatch count: fused kernels should dramatically reduce the number of launches per layer vs. generic ops.

---

## Ask / Next Owner
- Implement persistent u8 device buffers for MoE, routing kernel, and fused W1/W2 kernels as the only path (no flags).
- Keep existing logs; add kernel timings under `--debug`.
- Validate numerics and measure perf; iterate tile shapes and workgroup sizes as needed.

Once fused kernels are in, we’ll be at the “one correct way” with performance on par with GPT‑OSS Metal/Triton.
