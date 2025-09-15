# burn-extended

High-level, reusable building blocks for fast, long-context inference on top of Burn.

This package collects attention variants, RoPE helpers, sampling and generation utilities that are commonly needed across modern LLM-style and sequence models. It is designed to live outside the core Burn repo and remain model-agnostic, while making it straightforward to implement model-specific logic.

What this repo provides
- Attention primitives
  - Streaming Multi‑Query / Grouped‑Query Attention (MQA/GQA) with:
    - Fewer K/V heads than Q heads (num_key_value_heads < num_attention_heads)
    - Sliding‑window attention over a rolling KV cache
    - Optional learned “sinks” logit column per head‑group (GPT‑OSS style)
    - Optional additive attention bias hook (for ALiBi or custom biases)
  - Non‑streaming MQA (training or non‑cached inference) with masks + additive bias
  - Extended Streaming MHA that accepts an additive bias term (drop‑in when MQA isn’t required)
- RoPE (Rotary) helpers
  - NTK/YaRN scaling convenience that wraps Burn’s frequency‑scaling entrypoint and applies concentration
- Sampling and generation
  - Logits processors: temperature, top‑k, repetition/frequency/presence penalties
  - Simple samplers (greedy, multinomial)
  - A small generation harness (AutoregressiveModel trait + generator) for chunked decoding with caches
- Caching and window policies
  - Multi‑layer cache managers for MHA/MQA
  - Per‑layer window policies (e.g., full on even layers, windowed on odd)
- Attention bias utilities
  - Sinks helpers (reshape per‑head sinks into [kv_heads, groups])
  - ALiBi bias generator (additive, shaped to the active window)

How this maps to the three models

GPT‑OSS
- Requirements
  - GQA/MQA: `num_attention_heads` with fewer `num_key_value_heads`
  - Learned sinks bias (append a sentinel logit per head‑group)
  - Sliding window (often on alternate layers)
  - NTK/YaRN RoPE scaling and concentration
  - Standard Transformer blocks (RMSNorm, SwiGLU with clamp), sampler, and weight loading
- What burn‑extended provides
  - Streaming MQA with windowed KV cache and sinks bias
  - Non‑streaming MQA for training/validation
  - RoPE NTK/YaRN helper
  - Generation + sampling utilities
  - Cache manager + every‑other‑layer window policy
- Still model‑specific
  - Exact GPT‑OSS decoder block (RMSNorm, residual wiring, SwiGLU clamp)
  - Layer‑local sinks parameters and a weight loader from GPT‑OSS checkpoints

ACE‑Step
- Requirements
  - Streaming MHA with a custom, learned attention‑biasing policy
  - Sliding window and cache management
  - RoPE (standard) and a generation loop
- What burn‑extended provides
  - Extended Streaming MHA with additive `attn_bias`
  - Bias utilities (ALiBi as an example; pluggable custom bias tensors)
  - Cache manager + window policies and generation harness
- Still model‑specific
  - ACE‑Step block definition and the function that generates its attention bias
  - Weight import and any task‑specific heads

Matrix‑Game‑2
- Requirements
  - Streaming MHA with “sink tokens” preserved across windows
  - Simple sampler and a loop that interacts with the environment
- What burn‑extended provides
  - Streaming cache with sink‑token preservation (via Burn’s cache; extended MHA works with it)
  - Window policies and generation utilities to step through sequences
- Still model‑specific
  - Minimal model head and the environment glue (state→tokens, tokens→actions)

Examples
- `examples/gpt_oss.rs` — Streaming MQA + sinks + NTK/YaRN RoPE (runs with WGPU/Metal)
- `examples/ace_step.rs` — Streaming MHA + additive attention bias
- `examples/matrix_game_2.rs` — Streaming MHA with sink tokens and a sliding window

Repository layout
- `src/attention/`
  - `streaming_mqa.rs` — Streaming MQA (GQA) with sinks and `attn_bias`
  - `mqa.rs` — Non‑streaming MQA with masks + `attn_bias`
  - `streaming_ext.rs` — Extended Streaming MHA with `attn_bias`
- `src/rope/` — RoPE NTK/YaRN helper
- `src/sampling/` — logits processors and basic samplers
- `src/generate/` — autoregressive generation harness
- `src/cache/` — multi‑layer cache managers and window policies
- `src/bias/` — sinks helpers and ALiBi builder
- `examples/` — runnable demos for the three target models

Roadmap (next steps)
- Add optional SwiGLU‑with‑clamp activation utility
- Provide ready‑to‑use GPT‑OSS decoder block and a minimal weight loader
- Add a bias‑policy example for ACE‑Step
- Package examples with simple CLIs (window/chunk size, sinks on/off, sampler config)

Notes
- This repo expects a sibling checkout of the Burn repo and uses path dependencies to `burn-core` and `burn-tensor`.
- The code is backend agnostic, with examples configured for WGPU/Metal.
