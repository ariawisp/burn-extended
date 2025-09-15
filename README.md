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

| Model | Key requirements | Provided by burn‑extended | Still model‑specific |
|---|---|---|---|
| [GPT‑OSS](https://github.com/openai/gpt-oss) | <ul><li>GQA/MQA</li><li>Learned sinks bias</li><li>Sliding window (alternate layers)</li><li>NTK/YaRN RoPE</li></ul> | <ul><li>Streaming MQA with sinks + window</li><li>NTK/YaRN RoPE helper</li><li>Cache/window tools</li><li>Generation + samplers</li></ul> | <ul><li>Decoder block wiring (RMSNorm, residuals, SwiGLU clamp)</li><li>Per‑layer sinks params</li><li>Weight loader</li></ul> |
| [ACE‑Step](https://github.com/ace-step/ACE-Step) | <ul><li>Streaming MHA with learned attention bias</li><li>Sliding window</li><li>RoPE</li><li>Generation</li></ul> | <ul><li>Extended Streaming MHA with additive <code>attn_bias</code></li><li>Bias utilities</li><li>Cache/window tools</li><li>Generation harness</li></ul> | <ul><li>Exact block + bias policy function</li><li>Weights and task heads</li></ul> |
| [Matrix‑Game‑2](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2) | <ul><li>Streaming MHA with sink tokens</li><li>Simple interaction loop</li></ul> | <ul><li>Streaming cache with sink preservation</li><li>Window policy helpers</li><li>Generation utilities</li></ul> | <ul><li>Minimal head</li><li>Environment glue (state↔tokens↔actions)</li></ul> |

Roadmap
- Add optional SwiGLU‑with‑clamp utility
- Provide ready‑to‑use GPT‑OSS decoder block and a minimal weight loader
- Add a concrete bias‑policy example for ACE‑Step
- Expose simple CLIs for examples (window/chunk sizes, sinks on/off, sampler config)

Notes
- Expects a sibling checkout of the Burn repo; uses path dependencies to `burn-core` and `burn-tensor`.
- Backend‑agnostic; examples are configured for WGPU/Metal by default.
