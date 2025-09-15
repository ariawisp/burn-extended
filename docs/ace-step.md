# ACE‑Step — Inference Notes

Repository: https://github.com/ace-step/ACE-Step

This document captures ACE‑Step’s attention requirements and how `burn‑extended` supports them in a reusable way.

Key requirements
- Streaming MHA with a learned, data‑dependent attention bias.
- Sliding‑window attention and long‑sequence caching.
- RoPE position encoding.
- Autoregressive generation loop and sampling.

What burn‑extended provides
- Extended Streaming MHA: `attention::ExtStreamingMultiHeadAttention{Config}`
  - Same projections and cache semantics as Burn’s streaming MHA.
  - Additional `attn_bias: Option<&Tensor<_,4>>` term added to logits pre‑softmax.
- Bias utilities: `bias::alibi_bias(...)` example and space for custom bias builders.
- Cache/window policies: `cache::MhaCacheManager` + `WindowPolicy`.
- Generation tools: `sampling` processors + `generate` harness.

Additive attention bias
- Shape: `[B, n_heads, q_len, k_len]`, aligned to the active window.
- Construct per chunk using your policy network or a handcrafted heuristic, then pass via `ExtStreamingParams { attn_bias: Some(&bias) }`.

Sliding window
- Use `WindowPolicy` to decide per‑layer `AttnWindow` (e.g., fixed window for all layers).

RoPE
- If standard RoPE is needed, use Burn’s `RotaryEncodingConfig::init` and pass into streaming params.
- If extended context is desired later, use `rope::init_ntk_yarn(...)` from this repo.

Generation harness (sketch)
```rust
use burn_extended::{attention::*, cache::*, generate::*, sampling::*};

struct AceStepArModel<B: Backend> { /* ... */ }
impl<B: Backend> AutoregressiveModel<B> for AceStepArModel<B> {
    type Cache = MhaCacheManager<B>;
    fn init_cache(&self, batch: usize, device: &B::Device) -> Self::Cache { /* per-layer MHA caches */ }
    fn forward_logits(&self, tokens: Tensor<B,2,Int>, cache: &mut Self::Cache, start_pos: usize, window: AttnWindow) -> Tensor<B,2> {
        // embed -> per-layer { ExtStreamingMHA with attn_bias } -> norm -> head
        unimplemented!()
    }
}
```

Open items
- Implement the precise bias‑generating module used by ACE‑Step and map its parameters.
- Provide a small example to visualize how the bias changes with context.

