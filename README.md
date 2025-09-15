# burn-extended

High‑level, reusable building blocks for fast, long‑context inference on top of Burn.

This package is model‑agnostic and focuses on the most demanded primitives for modern transformer‑style models: attention variants (streaming MQA/MHA), RoPE helpers, sampling/generation utilities, multi‑layer cache management, and attention bias helpers. It complements Burn core and can be maintained as a separate repository.

What this repo provides
- Attention primitives
  - Streaming Multi‑Query/Grouped‑Query Attention (MQA/GQA) with sinks bias, sliding window, and additive logits bias
  - Non‑streaming MQA with masks + additive logits bias
  - Extended Streaming MHA with additive logits bias (drop‑in when MQA isn’t required)
- RoPE helpers
  - NTK/YaRN scaling + concentration wrapper over Burn’s Rotary frequency‑scaling
- Sampling and generation
  - Logits processors (temperature, top‑k, repetition/frequency/presence penalties)
  - Greedy/multinomial samplers and a minimal generation harness for chunked decoding
- Caches and windows
  - Multi‑layer cache managers (MHA/MQA) and per‑layer window policies (e.g., full vs. sliding by layer)
- Attention bias utilities
  - Sinks helpers and ALiBi bias generator (additive, window‑shaped)

Model coverage (current targets)

| Model           | Key requirements                                                                 | Provided by burn‑extended                                           | Still model‑specific                                      |
|-----------------|-----------------------------------------------------------------------------------|----------------------------------------------------------------------|------------------------------------------------------------|
| GPT‑OSS         | GQA/MQA; learned sinks bias; sliding window (alternate layers); NTK/YaRN RoPE     | Streaming MQA with sinks + window; NTK/YaRN RoPE; cache/window tools; generation + samplers | Decoder block wiring (RMSNorm, residuals, SwiGLU clamp); per‑layer sinks params; weight loader |
| ACE‑Step        | Streaming MHA with custom learned attention bias; sliding window; RoPE; generation| Extended Streaming MHA with additive `attn_bias`; bias utils; cache/window tools; generation | Exact block + bias policy function; weights and task heads  |
| Matrix‑Game‑2   | Streaming MHA with sink tokens; simple interaction loop                           | Streaming cache with sink preservation; window policy; generation utilities                   | Minimal head + environment glue (state↔tokens↔actions)     |

Examples
- `examples/gpt_oss.rs` — Streaming MQA + sinks + NTK/YaRN RoPE (WGPU/Metal)
- `examples/ace_step.rs` — Streaming MHA + additive attention bias
- `examples/matrix_game_2.rs` — Streaming MHA with sink tokens and sliding window

Roadmap
- Add optional SwiGLU‑with‑clamp utility
- Provide ready‑to‑use GPT‑OSS decoder block and a minimal weight loader
- Add a concrete bias‑policy example for ACE‑Step
- Expose simple CLIs for examples (window/chunk sizes, sinks on/off, sampler config)

Notes
- Expects a sibling checkout of the Burn repo; uses path dependencies to `burn-core` and `burn-tensor`.
- Backend‑agnostic; examples are configured for WGPU/Metal by default.
