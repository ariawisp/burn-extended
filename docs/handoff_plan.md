# burn-extended — Updated Handoff Plan and Roadmap (Post-Exporter Parity)

This doc captures current status, what’s been completed, and a concrete plan for the next engineer to take over. The primary goal (Phase 2) is complete: Rust exporter bit‑parity with the Python Metal exporter plus structural verification on a real 20B checkpoint. Phase 3 (native reader) is partially implemented and queued next.

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
Owner: next engineer
Status: IN PROGRESS (pre‑MoE loader implemented)

What’s done
- Reader scaffold: parses headers/tokenizer and loads all non‑MoE sections into the Burn model (embedding, per‑layer attn + gate/norm, final norm, unembedding).
- Inference stub from model.bin: derives config from header and runs a small Harmony prompt (blocked currently by header/config mismatch on the 20B sample).

Files
- Reader: `src/loader/modelbin.rs` (functions: `parse_modelbin`, `load_modelbin_into`)
- Inference from model.bin: `src/bin/gptoss_modelbin_infer.rs`

Next tasks
1) MoE load + dequant
   - Parse per‑expert MoE weights (mlp1/2 blocks/scales, biases) from model.bin.
   - Dequant MXFP4: subtract UE8 offset (14), apply exponent bias (−127) using FP4 LUT, materialize BF16 or FP32.
   - Map to `MoeGatedSwiGLU` tensors (shapes match SafeTensors mapping already used elsewhere).

2) Header→config reconciliation
   - Ensure `d_model == n_heads * head_dim` in the runtime config; if the header uses a different `embedding_dim`, prefer consistent model dimensions while reading bytes using header sizes.
   - Add a CLI flag to override config or to disable RoPE checks for debugging.

3) Validation
   - Round‑trip: SafeTensors → model.bin → reader → in‑memory tensors equal (BF16 fields) and dequant within tolerance for MXFP4.
   - Run Harmony example via model.bin reader (20B): confirm generation produces sane output.

Acceptance Criteria
- Reader loads a real 20B model.bin and generation runs.
- Round‑trip parity on tiny fixture within 1e‑3 BF16 tolerance; MoE dequant within expected FP4 error.

Risks/Dependencies
- Keep exactly in sync with exporter layout and alignments.
- Memory overhead during dequant; consider streaming decode if RAM is tight.

Estimate
- 1 week (MoE load + config reconciliation + validation).

---

## Phase 4 — Quantized Runtime and Performance
Owner: performance‑focused engineer
Status: Not started

Tasks
- MXFP4 kernels (WGPU/CubeCL): matmul for FP4 blocks + biased UE8 scales.
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
- Inference (model.bin — WIP)
  - `cargo run -p burn-extended --bin gptoss_modelbin_infer -- --model ~/gpt-oss-20b/metal/model.bin --d_model 0 --n_layers 0 --n_heads 0 --kv_heads 0 --ffn_hidden 0 --num_experts 0 --vocab 0`
  - Note: This stub derives config from header; MoE loading and config reconciliation still pending.

---

## Reference: Files to Know
- Exporter: `src/bin/gptoss_export.rs`
- Verifier: `src/bin/gptoss_verify.rs`
- Reader (pre‑MoE): `src/loader/modelbin.rs`
- Inference from model.bin (stub): `src/bin/gptoss_modelbin_infer.rs`
- MoE module: `src/moe/mod.rs`
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
- Performance
  - Reader currently dequants on load (for MoE once implemented). Plan kernels for runtime MXFP4 in Phase 4.

## Contact & Questions
- For exporter semantics, compare against `gpt_oss/metal/scripts/create-local-model.py`. The verifier and parity test are the fastest way to catch regressions.
