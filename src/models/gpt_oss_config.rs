use burn_core as burn;

use serde::Deserialize;

use super::gpt_oss::GptOssConfig;

#[derive(Debug, Deserialize)]
struct GptOssJsonConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub num_experts: usize,
    #[serde(default)]
    pub experts_per_token: Option<usize>,
    pub sliding_window: usize,
    pub initial_context_length: f32,
    #[serde(rename = "rope_theta")]
    pub _rope_theta: f32,
    pub rope_scaling_factor: f32,
    pub rope_ntk_alpha: f32,
    pub rope_ntk_beta: f32,
    #[serde(default)]
    pub swiglu_limit: Option<f32>,
}

impl GptOssConfig {
    pub fn from_config_json(path: &std::path::Path) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let cfg: GptOssJsonConfig = serde_json::from_str(&data)?;
        let d_model = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        anyhow::ensure!(d_model % n_heads == 0, "d_model must divide n_heads");
        let head_dim = d_model / n_heads;
        anyhow::ensure!(head_dim % 2 == 0, "head_dim must be even for RoPE (got {head_dim})");
        Ok(GptOssConfig {
            vocab_size: cfg.vocab_size,
            d_model,
            n_layers: cfg.num_hidden_layers,
            n_heads,
            kv_heads: cfg.num_key_value_heads,
            ffn_hidden: cfg.intermediate_size,
            num_experts: cfg.num_experts,
            experts_per_token: cfg.experts_per_token.unwrap_or(4),
            dropout: 0.0,
            swiglu_alpha: 1.0,
            swiglu_limit: cfg.swiglu_limit.unwrap_or(7.0),
            initializer: burn::nn::Initializer::KaimingUniform {
                gain: 1.0 / num_traits::Float::sqrt(3.0),
                fan_out_only: false,
            },
            cache_len: (cfg.initial_context_length * cfg.rope_scaling_factor) as usize,
            sink_tokens: 0,
            window_policy: crate::cache::WindowPolicy::EveryOther { window: cfg.sliding_window, full_on_even: true },
            max_seq_len: (cfg.initial_context_length * cfg.rope_scaling_factor) as usize,
            learned_sinks: true,
            use_ntk_yarn: true,
            rope_scaling_factor: cfg.rope_scaling_factor,
            rope_initial_context: cfg.initial_context_length,
            rope_ntk_alpha: cfg.rope_ntk_alpha,
            rope_ntk_beta: cfg.rope_ntk_beta,
        })
    }
}
