use burn_core as burn;

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn_tensor::activation::{quiet_softmax, softmax};

/// Compute attention scores = (Q x K^T) / sqrt(d_k) and apply dropout.
pub fn compute_scores<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    d_k: usize,
    dropout: &burn::nn::Dropout,
) -> Tensor<B, 4> {
    let attn_scores = q.matmul(k.transpose()).div_scalar((d_k as f32).sqrt());
    dropout.forward(attn_scores)
}

/// Apply additive attention bias and softmax.
pub fn apply_bias_and_softmax<B: Backend>(
    mut attn_scores: Tensor<B, 4>,
    attn_bias: Option<Tensor<B, 4>>,
    quiet: bool,
) -> Tensor<B, 4> {
    if let Some(bias) = attn_bias {
        attn_scores = attn_scores + bias;
    }
    if quiet {
        quiet_softmax(attn_scores, 3)
    } else {
        softmax(attn_scores, 3)
    }
}

/// Append sinks sentinel column and softmax across [Tk+1]; then discard sinks column.
/// sinks_per_head: [n_heads]
pub fn apply_sinks_then_softmax<B: Backend>(
    attn_scores: Tensor<B, 4>,
    sinks_per_head: Tensor<B, 1>,
    batch: usize,
    n_heads: usize,
    t_query: usize,
    t_key: usize,
    quiet: bool,
) -> Tensor<B, 4> {
    let mut s = sinks_per_head.reshape([1, n_heads, 1, 1]);
    s = s.repeat_dim(0, batch);
    s = s.repeat_dim(2, t_query);
    let attn_scores_cat = Tensor::cat(vec![attn_scores, s], 3);
    let w_all = if quiet {
        quiet_softmax(attn_scores_cat, 3)
    } else {
        softmax(attn_scores_cat, 3)
    };
    w_all.slice([0..batch, 0..n_heads, 0..t_query, 0..t_key])
}
