# Matrix‑Game‑2 — Inference Notes

Repository: https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2

This document captures the minimal pieces required to run streaming inference with sink tokens in a Matrix‑Game‑2‑style setup.

Key requirements
- Streaming MHA over long sequences with sink tokens preserved across windows.
- Simple token generation loop to interact with the environment.

What burn‑extended provides
- Extended streaming MHA: `attention::ExtStreamingMultiHeadAttention{Config}` works with Burn’s `StreamingMhaCache`, which supports `sink_tokens`.
- Window policy helpers and a lightweight generation harness.

Sink tokens
- Initialize the `StreamingMhaCache` with `sink_tokens > 0` to keep the first `sink_tokens` positions always attendable, regardless of the sliding window span.
- Choose a window length `W` such that `sink_tokens + W <= cache_len`.

Interaction loop (sketch)
```rust
// Pseudocode for environment interaction
loop {
    // 1) Encode current environment state into tokens
    let tokens = encode_state_to_tokens(env_state);
    // 2) Run one decoding step (or a small chunk) with streaming cache
    let logits = model.forward_logits(tokens, &mut cache, start_pos, window);
    // 3) Sample next action token
    let next = process_and_sample(logits, Some(&history), sampler_cfg, true);
    // 4) Apply action to environment and update state/history
    env_state = step_env(env_state, next);
    start_pos += 1;
}
```

Open items
- Define the minimal head to project hidden states to action space.
- Provide a compact state↔tokens↔actions encoding/decoding format.

