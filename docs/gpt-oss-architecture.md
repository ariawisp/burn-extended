# GPT‑OSS Architecture Spec for burn-extended

This document captures the model details needed to implement GPT‑OSS accurately in burn‑extended.

Scope
- Decoder‑only transformer with MoE MLPs and streaming attention.
- Inference features: streaming MQA/GQA with learned sinks, sliding window policy, NTK/YaRN RoPE, optional ALiBi.
- Weight formats: SafeTensors (canonical), Metal `model.bin` (export target).

Attention
- Heads: `n_heads` for Q/O, `kv_heads` for shared K/V; `groups = n_heads / kv_heads`.
- Projections: Q: `[B, T, n_heads, d_k]`, K/V: `[B, T, kv_heads, d_k]` then expand KV across `groups`.
- Sinks: learned per (kv_head, group) or per head; appended as a sentinel logit before softmax.
- Sliding window: alternate layers switch between Full vs Window(W); sink tokens are always preserved.
- RoPE: NTK/YaRN scaling with parameters from config (`scaling_factor`, `initial_context_length`, `ntk_alpha`, `ntk_beta`).
- Optional ALiBi: per‑head linear bias aligned to absolute positions and active windows.

MoE MLP
- Routing: linear gate → top‑k experts per token (k=4) → softmax over selected experts.
- Expert 1: `mlp1_weight: [E, 2*ffn, d_model]`, `mlp1_bias: [E, 2*ffn]` → SwiGLU(+clamp) → `[ffn]`.
- Expert 2: `mlp2_weight: [E, d_model, ffn]`, `mlp2_bias: [E, d_model]` → residual add.
- Shapes match GPT‑OSS reference; world_size sharding is out of scope here (single‑rank inference).

Weights & loaders
- Fused QKV: `block.L.attn.qkv.(weight|bias)` → split into Q/K/V using head counts and `head_dim`.
- Sinks: `block.L.attn.sinks` shaped `[n_heads]` → reshape to `[kv_heads, groups]`.
- MoE MXFP4: `mlp1_weight.blocks/scales`, `mlp2_weight.blocks/scales` dequantized using FP4 LUT
  `[+0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]`
  and exponent bias `-127` (see reference `gpt_oss/torch/weights.py`).

Metal export (later phase)
- Single binary with headers, tokenizer payload (o200k_gptoss + Harmony), aligned sections.
- Q/K scaling baked for `head_dim=64` and alternating layer window policy encoded in header.
- MoE weights kept quantized as MXFP4 (blocks + biased scales).

Validation
- Unit tests for QKV split, sinks reshape, MXFP4 dequant.
- Numeric checks vs. reference for small configs at layer/output level.

