# burn-extended — Handoff Plan and Roadmap (Reader + MoE streaming)

This doc summarizes current status, what’s been completed, and a concrete plan for the next engineer to take over. Phase 2 (exporter) is complete. Phase 3 (native model.bin reader) is implemented with attention dimension decoupling and mmap-based MoE streaming. We added fixture-level parity tests that validate numerics between the SafeTensors path and the model.bin path. The remaining critical work is to bound MoE memory usage on long prefills via chunked (time-sliced) execution and to add a small amount of runtime instrumentation.

## Scope Summary
- Core goal: Correct GPT‑OSS inference in Burn with MoE and MXFP4 semantics; SafeTensors as canonical format; Rust exporter/reader aligned with Metal `model.bin`.
- Secondary: ACE‑Step utilities and Matrix‑Game helpers are present and can proceed once GPT‑OSS is solid.

---

## Phase 2 — Exporter (SafeTensors → model.bin) Parity
Owner: previous engineer
Status: COMPLETE

What’s done
- Full exporter parity with Python Metal exporter: header, tokenizer payload (regex and special UUIDs), embeddings, per‑layer attention (QKV transform: interleave halves + pre‑scale Q/K for head_dim=64), sinks, attn.out, mlp.norm, gate, final norm, and MoE blocks/scales/biases with correct 16/16384 alignment and UE8 offset.
- Tiny fixture parity (byte‑for‑byte) and structural verifier against 20B.

Artifacts
- Exporter: `src/bin/gptoss_export.rs`
- Fixture: `src/bin/gptoss_fixture.rs`
- Structural verifier: `src/bin/gptoss_verify.rs`
- Tests: `tests/exporter_parity_tests.rs`

Acceptance (met)
- Tiny fixture parity MATCH vs Python exporter.
- Real 20B checkpoint exported and structurally verified.

---

## Phase 3 — Native model.bin Reader and Decoupled Attention
Owner: current + next engineer
Status: Implemented; needs MoE time-slice execution to bound RAM on long prefills

What’s done
- Reader parses headers/tokenizer; loads non‑MoE weights (embedding, per‑layer attn + gate/norm, final norm, unembedding) per Metal layout (16/16KB alignment).
- MoE: reader stores quantized MXFP4 blocks/scales (U8) and biases (BF16); we use mmap at runtime to pull expert weights lazily.
- Attention decoupling: d_model != n_heads*head_dim supported.
  - query: [d_model → n_heads*head_dim]
  - key/value: [d_model → kv_heads*head_dim]
  - output: [n_heads*head_dim → d_model]
  - pre‑scaled Q/K from exporter honored (skip extra 1/sqrt(d_k))
- Sinks mapping: learned sinks reshaped from [n_heads] to [kv_heads, groups].
- Fixture inference parity: SafeTensors vs model.bin
  - Q/K/V after projection+RoPE — MATCH
  - Attention scores and softmax weights — MATCH
  - attn.out weight — MATCH
  - Embedding, final norm (gamma) — MATCH
  - Final logits — MATCH (after ensuring lm_head has no bias; GPT‑OSS head is bias=false)

Files
- Reader and verifier: `src/loader/modelbin.rs`, `src/bin/gptoss_verify.rs`
- Model and attention: `src/models/gpt_oss.rs`, `src/attention/streaming_mqa.rs`
- SafeTensors GPT‑OSS helpers: `src/loader/gpt_oss.rs` (row‑fused QKV transform; lm_head loader without PyTorch adapter)
- Parity test (fixture): `tests/inference_parity_fixture.rs`

Open issue observed on 20B run
- Memory spikes with MoE on long prefills (e.g., >10k tokens or heavy Harmony prefill) causing host/device RAM to balloon (e.g., 55 GB). Root cause: processing the entire prefill as one chunk increases the union of routed experts per layer, causing many expert dequants per layer in one pass.

Next tasks (handoff priorities)
1) Streamed MoE execution (automatic) — CRITICAL
   - Dequantize MoE weights in tiles during matmul (no full expert materialization) so memory stays bounded regardless of prefill length.
   - Always-on: no runtime knobs required. Maintain exact MXFP4 math (UE8 offset 14, exponent bias −127, FP4 LUT).
   - Optional: tiny in-call LRU for dequant scratch within a tile window to avoid repeated dequants if the same expert recurs.

2) Minimal runtime instrumentation (debug‑only)
   - Log per‑slice: routed experts (set size), bytes dequantized, and peak scratch.
   - Optionally export simple prometheus‑style counters behind a feature flag.

3) End‑to‑end validation (20B)
   - With streamed MoE enabled by default: load 20B model.bin, run Harmony prompt(s), confirm bounded memory (<2× model.bin size total host+device, excluding KV cache) and stable generation.
   - Keep fixture parity green (unit tests).

Acceptance Criteria
- 20B model.bin runs with MoE enabled and memory bounded (target <2× model.bin size total RAM excluding KV cache).
- Fixture parity tests all pass.
- Structural verifier remains green.

Risks/Dependencies
- Metal/WGPU allocator can retain buffers between slices; ensure we reuse scratch buffers and avoid per‑token allocations.
- Large prompts must be chunked consistently across layers to keep memory stable.

---

## Phase 4 — Quantized Runtime and Performance
Owner: performance‑focused engineer
Status: Not started (unblocked after time‑sliced MoE)

Tasks
- MXFP4 device kernels (WGPU/CubeCL): matmul on FP4 blocks with UE8 scales to avoid host dequant.
- Integrate into MoE forward path; minimize host copies and CPU work.
- Attention perf: iterate on quiet_softmax and cache layout.
- Benchmarks vs BF16.

Acceptance
- MXFP4 MoE yields measurable speedups at medium batch/sequence sizes.

Estimate
- 2–4+ weeks.

---

## Phase 5 — Documentation, CI, and Upstreaming
Owner: next engineer
Status: Partially covered

Tasks
- Docs: expand README with model.bin usage, verifier steps, fixture parity, and a note that MoE memory bounding is automatic (no tuning required).
- CI: macOS + Linux; run `cargo test`, compile examples, fixture parity job (export both ways, diff), structural job (verify fixture and sample of real checkpoint headers).
- Upstreaming: propose reusable parts to burn‑core (streaming caches, sinks bias, masks, decoupled MQA).

Acceptance
- CI green; at least one upstream PR draft.

Estimate
- 1–2 weeks.

---

## GPT‑OSS Validation & Hardening (ongoing)
Owner: next engineer

Tasks
- Numerics: confirm small config parity vs PyTorch on representative blocks (attention, MoE expert forward in isolation).
- Config robustness: clear errors on mismatches; safe defaults.
- Optional ALiBi: keep hooks off by default; verify bias path correctness.

Acceptance
- End‑to‑end generation succeeds for multiple prompts on 20B (bounded memory, stable latency).

---

## ACE‑Step — Follow‑On (not blocking GPT‑OSS)
Owner: next engineer

Tasks
- Bias provider trait + minimal impl (additive bias per chunk).
- Diffusion demos using FlowMatch Euler/Heun + guidance helpers.
- SafeTensors mapping for ACE‑Step checkpoints.

Acceptance
- Bias provider example runs; diffusion demo yields expected shapes.

Estimate
- 3–5 days for MVP.

---

## Matrix‑Game‑2 — Follow‑On (not blocking GPT‑OSS)
Owner: next engineer

Tasks
- Minimal action head + codec.
- Control loop demo using streaming attention + sinks; 3D RoPE/video patches.
- Optional checkpoint loader.

Acceptance
- Interactive loop runs; tokens/actions handled correctly.

Estimate
- 3–5 days for MVP.

---

## Quick Start / Commands
- Export 20B model.bin
  - `cargo run -p burn-extended --bin gptoss_export -- -s ~/gpt-oss-20b/original -d ~/gpt-oss-20b/metal/model.bin`
- Verify structure
  - `cargo run -p burn-extended --bin gptoss_verify -- -s ~/gpt-oss-20b/original ~/gpt-oss-20b/metal/model.bin`
- Tiny fixture parity
  - `cargo run -p burn-extended --bin gptoss_fixture -- -o /tmp/gptoss_fixture`
  - `cargo test -p burn-extended --test inference_parity_fixture`
- Inference (SafeTensors)
  - `export GPT_OSS_DIR=~/gpt-oss-20b/original && cargo run -p burn-extended --example gpt_oss_harmony_infer`
- Inference (model.bin)
  - `cargo run -p burn-extended --bin gptoss_modelbin_infer -- --model ~/gpt-oss-20b/metal/model.bin`
  - MoE memory bounding is automatic (streamed dequant + tiled matmuls)
  - Debug: `--no_moe` to validate non‑MoE path

---

## Reference: Files to Know
- Exporter: `src/bin/gptoss_export.rs`
- Verifier: `src/bin/gptoss_verify.rs`
- Reader: `src/loader/modelbin.rs`
- Inference (model.bin): `src/bin/gptoss_modelbin_infer.rs`
- MoE module (routing + mmap streaming): `src/moe/mod.rs`
- GPT‑OSS model/config: `src/models/gpt_oss.rs`, `src/models/gpt_oss_config.rs`
- SafeTensors loaders: `src/loader/gpt_oss.rs`, `src/loader/mxfp4.rs`, `src/loader/qkv.rs`
- Attention: `src/attention/*`
- Harmony example: `examples/gpt_oss_harmony_infer.rs`
- Fixture parity: `tests/inference_parity_fixture.rs`
- Architecture notes: `docs/gpt-oss-architecture.md`

---

## Handoff Notes
- Branch/build
  - Repo builds on macOS (WGPU). Tests pass; parity test requires Python deps.
- Coordination
  - Harmony regex/specials now match exporter; any changes should be done via Harmony PRs to avoid divergence.
- Performance & memory
  - Reader now stores MoE quantized form and biases; on‑the‑fly dequant in MoE forward is the next step to keep memory bounded.
  - Attention pre‑scaling honored; RoPE width robust for odd head_dim.

## Contact & Questions
- For exporter semantics, compare against `gpt_oss/metal/scripts/create-local-model.py`. The verifier and parity test are the fastest way to catch regressions.
