# GPT‑OSS — Inference Notes

Repository: https://github.com/openai/gpt-oss

This document captures the model‑specific requirements of GPT‑OSS and how `burn‑extended` satisfies them with reusable, model‑agnostic components. It also outlines the remaining wiring needed to reach parity.

Key requirements
- Multi‑Query/Grouped‑Query Attention (MQA/GQA): fewer K/V heads than Q heads.
- Learned “sinks” attention bias: append a per‑head‑group sentinel logit to QK before softmax.
- Sliding‑window attention for long sequences (often on alternate layers).
- RoPE with NTK/YaRN context scaling and concentration.
- Standard transformer components (RMSNorm, SwiGLU with clamp) and a sampler.
- Weight import from GPT‑OSS checkpoints (fused QKV → separate Q/K/V).

What burn‑extended provides
- Streaming MQA/GQA: `attention::StreamingMultiQueryAttention{Config}`
  - `n_heads` and `num_key_value_heads` allow GQA head sharing.
  - Rolling KV cache, sink token preservation, and sliding `AttnWindow`.
  - Optional sinks column via `StreamingMqaParams { sinks: Option<&Tensor<_,2>> }`.
  - Optional additive attention bias via `attn_bias: Option<&Tensor<_,4>>` (for ALiBi or custom shaping).
- Non‑streaming MQA: `attention::MultiQueryAttention{Config}` for training/eval without cache.
- RoPE NTK/YaRN: `rope::init_ntk_yarn(...)` convenience over Burn’s frequency‑scaling hooks.
- Generation tools: `sampling` (temperature/top‑k/penalties) and `generate` harness for chunked decoding.
- Cache/window policies: `cache::{MqaCacheManager, WindowPolicy}` to manage per‑layer caches and windows.

Shapes and head grouping
- Let `d_model`, `n_heads`, `kv_heads`, and `head_dim = d_model / n_heads`.
- Query projection yields `[B, n_heads, T, head_dim]`.
- Key/Value projections yield `[B, kv_heads, T, head_dim]` and are expanded over `groups = n_heads / kv_heads` to `[B, n_heads, T, head_dim]`.

Sinks bias
- GPT‑OSS appends a learned sinks column (one extra logit) per head‑group.
- In `StreamingMultiQueryAttention`, pass a `sinks` tensor with shape `[kv_heads, groups]`, which is flattened to `[n_heads]` internally, broadcast to `[B, n_heads, Tq, 1]`, concatenated to QK, softmaxed, then dropped before V matmul.
- To disable sinks (but keep the branch wired), pass large negative logits (e.g., −1e9).

Sliding window policy (alternate layers)
- Use `cache::WindowPolicy::EveryOther { window, full_on_even }` and apply `policy.window_for(layer_idx)` for each layer’s `AttnWindow`.

RoPE NTK/YaRN
- Call `rope::init_ntk_yarn(max_seq_len, head_dim, device, scaling_factor, init_ctx_len, alpha, beta)`.
- Applies by‑parts inverse‑frequency scaling and concentration `0.1 * ln(scale) + 1.0`.

Weight import mapping (outline)
- GPT‑OSS packs QKV in a single linear: `hidden_size -> head_dim * (n_heads + 2*kv_heads)`.
- Split the fused weight/bias into:
  - `q_weight: [hidden_size, n_heads * head_dim]`
  - `k_weight: [hidden_size, kv_heads * head_dim]`
  - `v_weight: [hidden_size, kv_heads * head_dim]`
- Load into `StreamingMultiQueryAttention.query/key/value` in that order.
- For sinks parameters, export the learned vector per layer to shape `[kv_heads, groups]`.

SwiGLU clamp
- GPT‑OSS uses a clamped SwiGLU. Options:
  - Clamp the activations around the SwiGLU call in your block module.
  - Provide a small `SwiGluWithClamp` helper (planned in the roadmap).

Generation harness (sketch)
```rust
use burn_extended::{attention::*, cache::*, generate::*, sampling::*, rope};

struct GptOssArModel<B: Backend> { /* ... */ }
impl<B: Backend> AutoregressiveModel<B> for GptOssArModel<B> {
    type Cache = MqaCacheManager<B>;
    fn init_cache(&self, batch: usize, device: &B::Device) -> Self::Cache { /* allocate per-layer caches */ }
    fn forward_logits(&self, tokens: Tensor<B,2,Int>, cache: &mut Self::Cache, start_pos: usize, window: AttnWindow) -> Tensor<B,2> {
        // embed -> per-layer { streaming MQA w/ sinks + window } -> norm -> lm_head
        unimplemented!()
    }
}

let cfg = GenerationConfig { max_new_tokens: 256, eos_token: None, sampler: SamplerConfig { temperature: 0.8, top_k: Some(50), ..Default::default() }, window: AttnWindow::Window(4096) };
let outputs = generate(&model, &device, &[prompt_tokens], cfg);
```

Open items
- Implement the exact decoder block (RMSNorm + SwiGLU clamp + residuals) and wire per‑layer sinks.
- Provide a weight loader utility tailored to GPT‑OSS checkpoints.

