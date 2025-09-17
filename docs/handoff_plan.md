# burn-extended — Handoff Plan and Roadmap (Reader + MoE streaming)

This doc captures current status, what’s been completed, and a concrete plan for the next engineer to take over. The primary goal (Phase 2) remains complete: Rust exporter bit‑parity with the Python Metal exporter plus structural verification on a real 20B checkpoint. Phase 3 (native reader) is implemented with correct layout parsing and inference stubs; the remaining critical work is to stream MoE at runtime to keep memory bounded and to finalize attention dimension decoupling for GPT‑OSS.

## Scope Summary
- Core goal: Correct GPT‑OSS inference in Burn with MoE and MXFP4 semantics; SafeTensors as canonical format; Rust exporter/reader aligned with Metal `model.bin` layout.
- Secondary: ACE‑Step schedulers/guidance and Matrix‑Game utilities are present; examples and integrations can proceed once GPT‑OSS is solid.

---

## Phase 2 — Exporter (SafeTensors → model.bin) Parity
Owner: previous engineer (completed)
Status: COMPLETE

What’s done
- Full exporter: header (YaRN params in f64 for parity), tokenizer payload (text/special counts, regex bytes, special UUID table), embeddings, per‑layer attention (QKV transform: interleave + Q/K scaling for head_dim=64), sinks, attn.out, mlp.norm, gate, final norm, and MoE blocks/scales/biases with UE8 offset and correct 16/16384 alignment.
- Tiny fixture + parity test: byte‑for‑byte equal to Python exporter on a synthetic 1‑layer/2‑expert fixture. Script + cargo test provided.
- Structural verifier: checks tokenizer, embeddings, per‑layer sections, and (optionally) MoE sizes using original SafeTensors.
- Docs: README includes parity steps, HF download, export commands.

Artifacts
- Exporter: `src/bin/gptoss_export.rs`
- Fixture generator: `src/bin/gptoss_fixture.rs`
- Parity script: `scripts/parity_test.sh`
- Test: `tests/exporter_parity_tests.rs`
- Verifier: `src/bin/gptoss_verify.rs`

Acceptance (met)
- Tiny fixture parity MATCH vs Python.
- Real 20B checkpoint exported and structurally verified with expected offsets and sizes.

---

## Phase 3 — Native model.bin Reader
Owner: current + next engineer
Status: Implemented (layout complete); streaming MoE pending

What’s done
- Reader parses headers/tokenizer and loads all non‑MoE sections into the Burn model (embedding, per‑layer attn + gate/norm, final norm, unembedding) exactly per the Metal layout (alignments 16/16384).
- MoE payloads: reader now parses per‑layer expert blocks/scales (U8) and biases (BF16). We proved dequant correctness and then pivoted to storing the quantized form for a streaming design.
- Inference stub from model.bin derives config from header by default; optional overrides exist but are not required anymore.
- Structural verifier runs on the 20B checkpoint and matches sizes/offsets; exporter parity on fixture remains byte‑for‑byte (vs Python) when WITH_PY=1.
- Attention fixes for GPT‑OSS:
  - pre‑scaled Q/K gating applied in exporter is honored at runtime (skip extra 1/sqrt(d_k)).
  - RoPE application tolerates odd head_dim by rotating the largest even slice.

Files
- Reader: `src/loader/modelbin.rs` (`parse_modelbin`, `load_modelbin_into`)
- Inference: `src/bin/gptoss_modelbin_infer.rs` (auto‑header config, `--no_moe` debug flag)
- MoE module: `src/moe/mod.rs` (prepped for streaming dequant: quantized storage + bias; top‑k routing already implemented)
- Verifier: `src/bin/gptoss_verify.rs`

Open issues discovered
- GPT‑OSS 20B decouples embedding_dim (2880) from attention head dims (64 heads × 64 = 4096). Our current module still assumes `d_model == n_heads * head_dim` internally. We added guardrails to keep things running but a clean decoupling is required for perfect shape semantics.
- MoE memory: dequantizing all experts to float tensors or storing quantized as resident tensors explodes memory. We need lazy, per‑token dequant of only the routed experts from the model.bin mapping.

Next tasks (proposed for handoff)
1) Streaming MoE (critical)
   - Replace resident MoE tensors with per‑layer/expert offsets into model.bin (blocks/scales). Keep biases resident (BF16).
   - Add a small IO layer (mmap or pread) to fetch only the top‑k experts’ blocks/scales per token chunk; dequant on the fly to device tensors and compute MLP1/2; free buffers immediately.
   - Keep exact MXFP4 math (UE8 offset 14, exponent bias −127, FP4 LUT).

2) Attention dimension decoupling
   - Update `StreamingMultiQueryAttention` to use:
     - query: `[embedding_dim -> n_heads * head_dim]`
     - key/value: `[embedding_dim -> kv_heads * head_dim]`
     - output: `[n_heads * head_dim -> embedding_dim]`
   - Adjust config to carry `embedding_dim` and `head_dim` independently and derive only where appropriate. Ensure RoPE uses `head_dim` (even part) and projections use `embedding_dim`.

3) End‑to‑end validation (20B)
   - With streaming MoE enabled: load 20B model.bin, run Harmony prompt(s), ensure stable memory (no multi‑x inflate) and reasonable latency.
   - Small prompt regression: confirm decoded text is sane and consistent between SafeTensors path and model.bin reader within expected numeric tolerance.

Acceptance Criteria
- 20B model.bin loads and generates with MoE enabled without exceeding a small factor over file size in RAM (target <2× for host + device combined, excluding KV cache).
- Structural verifier remains green; fixture parity test remains green.
- Attention outputs numerically stable after decoupling.

Risks/Dependencies
- Careful with file IO strategy (mmap vs pread) on macOS/Linux and WGPU sync points.
- Ensure top‑k routing doesn’t thrash allocations; consider scratch buffers with reuse.

---

## Phase 4 — Quantized Runtime and Performance
Owner: performance‑focused engineer
Status: Not started (unblocked after streaming MoE)

Tasks
- MXFP4 kernels (WGPU/CubeCL): matmul for FP4 blocks + biased UE8 scales to avoid host dequant.
- Integrate into MoE forward path; reduce host copies.
- Attention perf: revisit quiet_softmax and cache layout.
- Benchmarks: microbenchmarks vs BF16 baseline.

Acceptance
- MXFP4 MoE runs with measurable speedup at medium batch/sequence.

Estimate
- 2–4+ weeks.

---

## Phase 5 — Documentation, CI, and Upstreaming
Owner: next engineer
Status: Partially covered

Tasks
- Docs
  - Expand README with reader usage, verifier, and modelbin inference stub.
  - Strengthen MXFP4 doc (internals + references); link to `docs/gpt-oss-architecture.md`.
- CI
  - macOS + Linux matrix; run `cargo test` and ensure examples compile.
  - Parity job: run Python exporter and Rust exporter on tiny fixture and diff artifacts.
  - Structural job: run model.bin verifier against the fixture and (optionally) a small sample of a real checkpoint.
- Upstreaming
  - Propose reusable pieces (streaming attention, cache, masks) to burn‑core.

Acceptance
- CI green; at least one upstream PR draft opened with isolated commits and tests.

Estimate
- 1–2 weeks.

---

## GPT‑OSS Validation & Hardening (ongoing)
Owner: next engineer

Tasks
- Numerics: small config parity vs PyTorch for representative blocks.
- Config robustness: clear errors on mismatches; provide safe defaults.
- Header/config reconciliation: support `embedding_dim != n_heads * head_dim` cleanly; fail with actionable messages when not supported.
- Optional ALiBi: keep hooks off by default; verify mask/bias paths.

Acceptance
- End‑to‑end generation succeeds for multiple prompts on 20B.

---

## ACE‑Step — Follow‑On (not blocking GPT‑OSS)
Owner: next engineer

Tasks
- Bias provider trait + minimal impl for additive bias per chunk.
- Diffusion demos using FlowMatch Euler/Heun + guidance helpers.
- Checkpoint mapping for ACE‑Step safetensors.

Acceptance
- Bias provider example runs; diffusion demo yields tensors with expected shapes.

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
- Interactive loop runs; tokens/actions are handled correctly.

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
  - `WITH_PY=1 cargo test -p burn-extended --test exporter_parity_tests`
- Inference (SafeTensors)
  - `export GPT_OSS_DIR=~/gpt-oss-20b/original && cargo run -p burn-extended --example gpt_oss_harmony_infer`
- Inference (model.bin)
  - `cargo run -p burn-extended --bin gptoss_modelbin_infer -- --model ~/gpt-oss-20b/metal/model.bin`
  - Optional: `--no_moe` runs without MoE (for debugging memory/perf while streaming MoE lands).

---

## Reference: Files to Know
- Exporter: `src/bin/gptoss_export.rs`
- Verifier: `src/bin/gptoss_verify.rs`
- Reader (pre‑MoE): `src/loader/modelbin.rs`
- Inference from model.bin (stub): `src/bin/gptoss_modelbin_infer.rs`
- MoE module (routing + streaming dequant scaffolding): `src/moe/mod.rs`
- GPT‑OSS model/config: `src/models/gpt_oss.rs`, `src/models/gpt_oss_config.rs`
- SafeTensors loaders: `src/loader/gpt_oss.rs`, `src/loader/mxfp4.rs`, `src/loader/qkv.rs`
- Attention: `src/attention/*`
- Harmony example: `examples/gpt_oss_harmony_infer.rs`
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
