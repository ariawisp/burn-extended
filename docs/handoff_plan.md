# burn-extended — Handoff Plan and Phased Roadmap

This document outlines the remaining work to achieve long‑term, correct GPT‑OSS parity in Burn, plus follow‑on deliverables for ACE‑Step and Matrix‑Game integration. It is structured for a clean handoff to another engineer, with phases, concrete tasks, acceptance criteria, and notes on risks and dependencies.

## Scope Summary
- Core goal: fully correct GPT‑OSS inference in Burn with MoE and MXFP4 semantics, SafeTensors as the canonical format, and a Rust exporter/reader for Metal `model.bin` parity.
- Secondary: ACE‑Step diffusion schedulers/guidance already landed; extend examples and bias providers. Matrix‑Game streaming attention/3D RoPE/video patching landed; add minimal environment/head wiring.

---

## Phase 2 — Exporter (SafeTensors → model.bin) Parity
Owner: next engineer
Status: Header + tokenizer payload implemented. Weight sections TBD.

Tasks
- Attention sections
  - Read fused QKV (weight+bias): `block.{L}.attn.qkv.(weight|bias)` from SafeTensors.
  - Apply GPT‑OSS Metal transform (head_dim=64):
    - Split QK and V blocks along rows.
    - Interleave QK halves (pairwise) to restore `[head_dim]` ordering as in Metal.
    - Scale Q rows by 0.5 and K rows by 0.25.
    - Keep V unchanged.
  - Write transformed QKV (weight and bias), with 16‑byte alignment.
  - Write `block.{L}.attn.out.(weight|bias)` raw BF16.
  - Write `block.{L}.attn.sinks` raw BF16.
- MoE sections
  - Write `block.{L}.mlp.mlp1_weight.blocks/scales`, `mlp2_weight.blocks/scales` MXFP4 with UE8 offset as per Python.
  - Write `block.{L}.mlp.mlp1_bias`, `mlp2_bias` BF16.
- Alignment
  - Mirror Python script alignment (16/16384 bytes) before each section.
- Differential test (parity)
  - Create a tiny synthetic SafeTensors (1–2 layers) fixture.
  - Export with Python `create-local-model.py` and Rust `gptoss_export`, assert byte‑for‑byte equality.
- CLI polish and docs
  - Add progress logging per section and size summary.
  - Document usage in `README.md` with example commands.

Acceptance Criteria
- `gptoss_export` produces a `model.bin` that matches Python exporter bytes for a small fixture and passes a structural sanity check for a real checkpoint.
- QKV transform verified by reading back and comparing against Python’s intermediate tensors for one layer.

Risks/Dependencies
- Exact ordering and alignment must match the Python implementation; rely on code comments and incremental parity tests.
- Time boxed testing on macOS Metal runner is desirable (if available) to confirm runtime acceptance.

Estimate
- 5–7 days including parity test and docs.

---

## Phase 3 — Native model.bin Reader (Optional but Valuable)
Owner: next engineer
Status: Not started

Tasks
- Parser
  - Read file magic, model header, tokenizer section, and section boundaries.
  - Load attention and MoE sections.
- Mapping to Burn model
  - Dequant MXFP4 to BF16/FP32 on load; copy into `MoeGatedSwiGLU` and attention params.
  - Respect QKV transforms (reverse if necessary or accept Metal ordering if we directly use the same param layout).
- Validation
  - Round‑trip: SafeTensors → model.bin (Rust) → reader → Burn model produces identical params (for FP fields) and equal decoded MXFP4 tensors (within tolerance).
  - Run Harmony example against model.bin via reader and confirm generation runs.

Acceptance Criteria
- Reader successfully loads real 20B weights into the Burn model and basic generation works.
- Round‑trip test passes on small fixture; parity within 1e‑3 BF16 tolerance for dequantized tensors.

Risks/Dependencies
- Must stay in lockstep with exporter layout.
- Memory usage acceptance for dequant‑on‑load.

Estimate
- 1–2 weeks.

---

## Phase 4 — Quantized Runtime and Performance
Owner: next engineer (with performance focus)
Status: Not started

Tasks
- MXFP4 kernels (WGPU/CubeCL)
  - Implement matmul for MXFP4 blocks+scales (row‑wise exponent) for MoE weights.
  - Integrate into MoE forward path with minimal overhead.
- Performance in attention
  - Investigate kernel fusion and cache layout to reduce host copies; validate `quiet_softmax` branch.
  - Assess sliding‑window path costs and sink preservation overhead.
- Sampling/logits processors
  - Optional: move top‑k and penalties to GPU; preserve CPU fallback.
- Benchmarks
  - Add microbenchmarks for MoE matmul and streaming attention compared to BF16 baselines.

Acceptance Criteria
- MXFP4 MoE runs with measurable speedup vs. dequant‑on‑load baselines for medium batch/sequence.
- No numerical drift beyond expected BF16 quantization effects.

Risks/Dependencies
- Kernel complexity and portability across WGPU backends (Metal, Vulkan, etc.).

Estimate
- 2–4+ weeks depending on kernel scope.

---

## Phase 5 — Documentation, CI, and Upstreaming Candidates
Owner: next engineer
Status: Partially covered

Tasks
- Docs
  - Update top‑level README with GPT‑OSS usage, Harmony integration, exporter/reader usage, and example commands.
  - Document MoE and MXFP4 internals with diagrams and references; cross‑link to `docs/gpt-oss-architecture.md`.
- CI
  - Add test matrix for macOS + Linux; ensure `cargo test` and examples compile.
  - Add parity test job that runs Python exporter and Rust exporter on a tiny fixture and diffs artifacts.
- Upstreaming
  - Identify reusable modules to propose to burn‑core (e.g., streaming attention kernels, cache policies, mask helpers).
  - Prepare PRs with isolated commits and unit tests.

Acceptance Criteria
- CI is green; docs cover end‑to‑end flows; at least one upstream PR draft prepared.

Estimate
- 1–2 weeks.

---

## GPT‑OSS Validation & Hardening (ongoing)
Owner: next engineer

Tasks
- End‑to‑end numerics
  - Small config parity vs. PyTorch for layer outputs (tolerance 1e‑3 bf16).
  - Harmony prompt/stop token path verified with round‑trip decode.
- Config loader robustness
  - Handle missing fields, defaults, and mismatched sizes with clear errors.
- ALiBi option (optional)
  - Keep bias hooks as optional, default off to match reference.

Acceptance Criteria
- Multiple sanity conversations generate successfully with real 20B weights.

---

## ACE‑Step — Follow‑On (not blocking GPT‑OSS)
Owner: next engineer

Tasks
- Bias provider module
  - Define a trait and a minimal implementation to supply additive bias per chunk.
- Diffusion demos
  - Small end‑to‑end generation that uses `FlowMatchEuler`/`Heun` and guidance helpers.
- Checkpoint mapping
  - Confirm 1:1 tensor names for ACE‑Step safetensors; add preset loader if needed.

Acceptance Criteria
- Bias provider example compiles and runs; diffusion sample produces output tensors with expected shapes.

Estimate
- 3–5 days for MVP.

---

## Matrix‑Game‑2 — Follow‑On (not blocking GPT‑OSS)
Owner: next engineer

Tasks
- Minimal action head and codec
  - Implement a thin projection and simple action codec.
- Control loop demo
  - Small winit or terminal loop using streaming attention + sinks, with 3D RoPE and video patching.
- Optional: loader for any public checkpoints.

Acceptance Criteria
- Example runs a short interactive loop; tokens/actions processed correctly.

Estimate
- 3–5 days for MVP.

---

## Reference: Current Files to Know
- Exporter CLI: `src/bin/gptoss_export.rs`
- MoE module: `src/moe/mod.rs`
- GPT‑OSS model: `src/models/gpt_oss.rs`, config loader: `src/models/gpt_oss_config.rs`
- Loaders: `src/loader/gpt_oss.rs`, `src/loader/mxfp4.rs`, `src/loader/qkv.rs`
- Attention: `src/attention/*` (streaming MHA/MQA, sinks, masks, linear, ALiBi)
- Harmony example: `examples/gpt_oss_harmony_infer.rs`
- Spec: `docs/gpt-oss-architecture.md`

---

## Handoff Notes
- Branch/build
  - Current branch compiles on macOS with WGPU; `cargo test` passes.
- Running examples
  - `cargo run -p burn-extended --example gpt_oss_harmony_infer` (set `GPT_OSS_DIR` to original checkpoint dir)
- Quick checkpoints
  - For exporter testing, start with a tiny synthetic SafeTensors to avoid long I/O.
- Coordination
  - If Harmony crate needs small API openings (regex string or special token mapping), prefer PR to Harmony; otherwise keep local inlined constants and `encode` with allowed specials as currently done.

---

## Contact & Questions
- If questions arise around the Python exporter semantics, compare against `gpt_oss/metal/scripts/create-local-model.py` and add incremental parity asserts while implementing sections.
