use burn_core as burn;

use burn::module::{Ignored, Module};
use burn::nn::{Initializer, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::tensor::{backend::Backend, Tensor};
use alloc::sync::Arc;

// Debug macro (enable with `--features moe_debug`)
#[cfg(feature = "moe_debug")]
macro_rules! moe_dbg { ($($t:tt)*) => { eprintln!($($t)*); } }
#[cfg(not(feature = "moe_debug"))]
macro_rules! moe_dbg { ($($t:tt)*) => {}; }

// Streaming context for MoE: holds file mapping and per-expert offsets.
#[derive(Debug, Clone)]
pub struct MoeExpertOffsets {
    pub mlp1_blocks_off: u64,
    pub mlp1_scales_off: u64,
    pub mlp2_blocks_off: u64,
    pub mlp2_scales_off: u64,
}

#[derive(Debug)]
pub struct MoeStreamingContext {
    pub mmap: Arc<memmap2::Mmap>, // mapped model.bin (read-only)
    pub experts: alloc::vec::Vec<MoeExpertOffsets>,
    pub rows_mlp1: usize,
    pub cols_mlp1: usize,
    pub rows_mlp2: usize,
    pub cols_mlp2: usize,
    pub ue8_offset: i32,
}


#[derive(Debug, Clone)]
pub struct MoeConfig {
    pub d_model: usize,
    pub ffn_hidden: usize,
    pub num_experts: usize,
    pub experts_per_token: usize,
    pub swiglu_alpha: f32,
    pub swiglu_limit: f32,
    pub initializer: Initializer,
    pub disabled: bool,
}

#[derive(Module, Debug)]
pub struct MoeGatedSwiGLU<B: Backend> {
    pub norm: RmsNorm<B>,
    pub gate: Linear<B>, // [d_model -> num_experts]
    // Expert parameters (quantized MXFP4 blocks/scales + BF16 biases) to match GPT‑OSS layout and reader mapping.
    pub mlp1_blocks: burn::module::Param<Tensor<B, 3>>, // U8: [E, 2*ffn, d_model/2]
    pub mlp1_scales: burn::module::Param<Tensor<B, 2>>, // U8: [E, 2*ffn]
    pub mlp1_bias: burn::module::Param<Tensor<B, 2>>,   // BF16: [E, 2*ffn]
    pub mlp2_blocks: burn::module::Param<Tensor<B, 3>>, // U8: [E, d_model, ffn/2]
    pub mlp2_scales: burn::module::Param<Tensor<B, 2>>, // U8: [E, d_model]
    pub mlp2_bias: burn::module::Param<Tensor<B, 2>>,   // BF16: [E, d_model]

    pub num_experts: usize,
    pub experts_per_token: usize,
    pub swiglu_alpha: f32,
    pub swiglu_limit: f32,
    pub d_model: usize,
    pub ffn_hidden: usize,
    pub disabled: bool,
    // Optional streaming context (preferred runtime path when present)
    pub streaming: Ignored<Option<Arc<MoeStreamingContext>>>,
}

impl MoeConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MoeGatedSwiGLU<B> {
        let norm = RmsNormConfig::new(self.d_model).init::<B>(device);
        let gate = LinearConfig::new(self.d_model, self.num_experts)
            .with_initializer(self.initializer.clone())
            .init(device);
        // Placeholder tensors; loader will overwrite with correct dtype and data.
        let mlp1_blocks = Tensor::<B, 3>::zeros([self.num_experts, self.ffn_hidden * 2, (self.d_model + 1) / 2], device);
        let mlp1_scales = Tensor::<B, 2>::zeros([self.num_experts, self.ffn_hidden * 2], device);
        let mlp1_b = Tensor::<B, 2>::zeros([self.num_experts, self.ffn_hidden * 2], device);
        let mlp2_blocks = Tensor::<B, 3>::zeros([self.num_experts, self.d_model, (self.ffn_hidden + 1) / 2], device);
        let mlp2_scales = Tensor::<B, 2>::zeros([self.num_experts, self.d_model], device);
        let mlp2_b = Tensor::<B, 2>::zeros([self.num_experts, self.d_model], device);
        MoeGatedSwiGLU {
            norm,
            gate,
            mlp1_blocks: burn::module::Param::from_tensor(mlp1_blocks),
            mlp1_scales: burn::module::Param::from_tensor(mlp1_scales),
            mlp1_bias: burn::module::Param::from_tensor(mlp1_b),
            mlp2_blocks: burn::module::Param::from_tensor(mlp2_blocks),
            mlp2_scales: burn::module::Param::from_tensor(mlp2_scales),
            mlp2_bias: burn::module::Param::from_tensor(mlp2_b),
            num_experts: self.num_experts,
            experts_per_token: self.experts_per_token,
            swiglu_alpha: self.swiglu_alpha,
            swiglu_limit: self.swiglu_limit,
            d_model: self.d_model,
            ffn_hidden: self.ffn_hidden,
            disabled: self.disabled,
            streaming: Ignored(None),
        }
    }
}

impl<B: Backend> MoeGatedSwiGLU<B> {
    pub fn set_streaming(&mut self, ctx: Arc<MoeStreamingContext>) {
        self.streaming = Ignored(Some(ctx));
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        if self.disabled {
            // Skip MoE block entirely, act as identity residual.
            return x;
        }
        // Shapes
        let [b, t, d] = x.dims();
        let n = b * t;

        // 1) Norm + gating
        let normed = self.norm.forward(x.clone()); // [B,T,d]
        let gates = self.gate.forward(normed.clone()); // [B,T,E]

        // Flatten for compute
        let norm_flat = normed.clone().reshape([n, d]);

        // Pull gates to CPU to compute top-k per token
        let e = self.num_experts;
        let k = self.experts_per_token.min(e).max(1);
        let gates_host = gates
            .clone()
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("gates to f32");

        // weights_per_expert[e][n]
        let mut weights_per_expert: Vec<Vec<f32>> = vec![vec![0.0f32; n]; e];
        for i in 0..n {
            let row = &gates_host[i * e..(i + 1) * e];
            // top-k indices by value
            let mut idx: Vec<usize> = (0..e).collect();
            idx.sort_unstable_by(|&a, &bidx| {
                row[bidx]
                    .partial_cmp(&row[a])
                    .unwrap_or(core::cmp::Ordering::Equal)
            });
            let top = &idx[..k];
            // softmax over selected
            let mut max_v = f32::NEG_INFINITY;
            for &j in top.iter() {
                if row[j] > max_v {
                    max_v = row[j];
                }
            }
            let mut sum = 0.0f32;
            let mut tmp = vec![0.0f32; k];
            for (p, &j) in top.iter().enumerate() {
                let v = (row[j] - max_v).exp();
                tmp[p] = v;
                sum += v;
            }
            if sum > 0.0 {
                for (p, &j) in top.iter().enumerate() {
                    weights_per_expert[j][i] = tmp[p] / sum;
                }
            }
        }

        // Accumulator for outputs
        let mut acc = Tensor::<B, 2>::zeros([n, d], &norm_flat.device());

        // If mmap streaming is available, use it instead of resident quantized params.
        if let Some(ctx) = self.streaming.0.as_ref() {
            return self.forward_streaming(x, ctx);
        }

        // Dequant helpers (resident-quantized fallback)
        const UE8_OFFSET: i32 = 14;
        const FP4_VALUES: [f32; 16] = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        let dequant_rows = |blocks: &[u8], scales: &[u8], rows: usize, cols: usize| -> Vec<f32> {
            let bpr = (cols + 1) / 2;
            let mut out = vec![0f32; rows * cols];
            for r in 0..rows {
                let si = (scales[r] as i32) - UE8_OFFSET;
                let scale = (2.0f32).powi(si);
                for i in 0..bpr {
                    let byte = blocks[r * bpr + i];
                    let lo = (byte & 0x0F) as usize;
                    let hi = (byte >> 4) as usize;
                    let c = i * 2;
                    if c < cols { out[r * cols + c] = FP4_VALUES[lo] * scale; }
                    if c + 1 < cols { out[r * cols + c + 1] = FP4_VALUES[hi] * scale; }
                }
            }
            out
        };

        for expert in 0..e {
            // Skip if no weight mass routed to this expert
            let w_host = &weights_per_expert[expert];
            let mut has_nonzero = false;
            for &v in w_host.iter() {
                if v != 0.0 { has_nonzero = true; break; }
            }
            if !has_nonzero { continue; }

            // Convert weights to device tensor [n,1]
            let w_ex = Tensor::<B, 1>::from_floats(w_host.as_slice(), &normed.device())
                .reshape([n, 1]);

            // Load expert biases
            let b1 = self
                .mlp1_bias
                .val()
                .clone()
                .slice([expert..expert + 1, 0..self.ffn_hidden * 2])
                .reshape([self.ffn_hidden * 2]);
            let b2 = self
                .mlp2_bias
                .val()
                .clone()
                .slice([expert..expert + 1, 0..self.d_model])
                .reshape([self.d_model]);

            // Dequant expert weights for mlp1 and mlp2
            let w1_blocks = self
                .mlp1_blocks
                .val()
                .clone()
                .slice([expert..expert + 1, 0..self.ffn_hidden * 2, 0..(self.d_model + 1) / 2])
                .reshape([self.ffn_hidden * 2, (self.d_model + 1) / 2])
                .into_data()
                .convert::<u8>()
                .into_vec::<u8>()
                .expect("mlp1 blocks u8 vec");
            let w1_scales = self
                .mlp1_scales
                .val()
                .clone()
                .slice([expert..expert + 1, 0..self.ffn_hidden * 2])
                .reshape([self.ffn_hidden * 2])
                .into_data()
                .convert::<u8>()
                .into_vec::<u8>()
                .expect("mlp1 scales u8 vec");
            let w1_deq = dequant_rows(&w1_blocks, &w1_scales, self.ffn_hidden * 2, self.d_model);
            let w1 = Tensor::<B, 2>::from_floats(burn_tensor::TensorData::new(w1_deq, [self.ffn_hidden * 2, self.d_model]), &norm_flat.device())
                ;

            // MLP1: [n,d] x [d,2f] + b1 -> [n,2f]
            let up = norm_flat.clone().matmul(w1.transpose()) + b1.unsqueeze();
            let act = crate::activation::swiglu_clamp(up, self.swiglu_alpha, Some(self.swiglu_limit));
            // Act now [n,f]
            let act_f = act.reshape([n, self.ffn_hidden]);

            let w2_blocks = self
                .mlp2_blocks
                .val()
                .clone()
                .slice([expert..expert + 1, 0..self.d_model, 0..(self.ffn_hidden + 1) / 2])
                .reshape([self.d_model, (self.ffn_hidden + 1) / 2])
                .into_data()
                .convert::<u8>()
                .into_vec::<u8>()
                .expect("mlp2 blocks u8 vec");
            let w2_scales = self
                .mlp2_scales
                .val()
                .clone()
                .slice([expert..expert + 1, 0..self.d_model])
                .reshape([self.d_model])
                .into_data()
                .convert::<u8>()
                .into_vec::<u8>()
                .expect("mlp2 scales u8 vec");
            let w2_deq = dequant_rows(&w2_blocks, &w2_scales, self.d_model, self.ffn_hidden);
            let w2 = Tensor::<B, 2>::from_floats(burn_tensor::TensorData::new(w2_deq, [self.d_model, self.ffn_hidden]), &norm_flat.device());

            // MLP2: [n,f] x [f,d] + b2 -> [n,d]
            let out = act_f.matmul(w2.transpose()) + b2.unsqueeze();
            let out = out * w_ex; // [n,d]
            acc = acc + out;
        }

        let y = acc.reshape([b, t, d]);
        x + y
    }

    fn forward_streaming(&self, x: Tensor<B, 3>, ctx: &MoeStreamingContext) -> Tensor<B, 3> {
        // Shapes
        let [b, t, d] = x.dims();
        let n = b * t;

        // 1) Norm + gating
        let normed = self.norm.forward(x.clone()); // [B,T,d]
        let gates = self.gate.forward(normed.clone()); // [B,T,E]
        let norm_flat = normed.clone().reshape([n, d]);

        let e = self.num_experts;
        let k = self.experts_per_token.min(e).max(1);
        let gates_host = gates
            .clone()
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("gates to f32");

        // Output accumulator: built by tiles over tokens, then written back.
        let mut acc = Tensor::<B, 2>::zeros([n, d], &norm_flat.device());

        // MXFP4 constants and layout helpers
        const FP4_VALUES: [f32; 16] = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        let ue8 = ctx.ue8_offset;
        let rows1 = ctx.rows_mlp1; // 2f
        let cols1 = ctx.cols_mlp1; // d_model
        let rows2 = ctx.rows_mlp2; // d_model
        let cols2 = ctx.cols_mlp2; // f
        let bpr1 = (cols1 + 1) / 2;
        let bpr2 = (cols2 + 1) / 2;

        // Dequant a contiguous range of rows for W1 (shape [rows, cols1])
        let dequant_rows_range = |blocks: &[u8], scales: &[u8], row_start: usize, row_count: usize, cols: usize| -> Vec<f32> {
            let mut out = vec![0f32; row_count * cols];
            let bpr = (cols + 1) / 2;
            for r in 0..row_count {
                let row_idx = row_start + r;
                let si = (scales[row_idx] as i32) - ue8;
                let scale = (2.0f32).powi(si);
                for i in 0..bpr {
                    let byte = blocks[row_idx * bpr + i];
                    let lo = (byte & 0x0F) as usize;
                    let hi = (byte >> 4) as usize;
                    let c = i * 2;
                    if c < cols { out[r * cols + c] = FP4_VALUES[lo] * scale; }
                    if c + 1 < cols { out[r * cols + c + 1] = FP4_VALUES[hi] * scale; }
                }
            }
            out
        };

        // Dequant a slice of columns for W2 (rows=rows2, cols=cols2). Returns [rows2, col_count]
        let dequant_cols_range = |blocks: &[u8], scales: &[u8], col_start: usize, col_count: usize, rows: usize, cols: usize| -> Vec<f32> {
            let mut out = vec![0f32; rows * col_count];
            let bpr = (cols + 1) / 2;
            for r in 0..rows {
                let si = (scales[r] as i32) - ue8;
                let scale = (2.0f32).powi(si);
                for c_off in 0..col_count {
                    let c = col_start + c_off;
                    let byte = blocks[r * bpr + (c / 2)];
                    let nib = if c % 2 == 0 { (byte & 0x0F) as usize } else { (byte >> 4) as usize };
                    out[r * col_count + c_off] = FP4_VALUES[nib] * scale;
                }
            }
            out
        };

        // Token-tiling: limit intermediate [n, *] tensors to a manageable size
        let tile_n = usize::min(256, usize::max(1, n));
        let mut start_n = 0;
        while start_n < n {
            let take_n = core::cmp::min(tile_n, n - start_n);
            let norm_tile = norm_flat.clone().slice([start_n..start_n + take_n, 0..d]); // [take_n, d]

            // Build top-k weights per expert for this tile and track which experts are used
            let mut used = vec![false; e];
            let mut weights_tile_per_expert: Vec<Vec<f32>> = vec![vec![0.0f32; take_n]; e];
            for i_off in 0..take_n {
                let i = start_n + i_off;
                let row = &gates_host[i * e..(i + 1) * e];
                let mut idx: Vec<usize> = (0..e).collect();
                idx.sort_unstable_by(|&a, &bidx| row[bidx].partial_cmp(&row[a]).unwrap_or(core::cmp::Ordering::Equal));
                let top = &idx[..k];
                let mut max_v = f32::NEG_INFINITY;
                for &j in top.iter() { max_v = max_v.max(row[j]); }
                let mut sum = 0.0f32;
                let mut tmp = vec![0.0f32; k];
                for (p, &j) in top.iter().enumerate() { let v = (row[j] - max_v).exp(); tmp[p] = v; sum += v; }
                if sum > 0.0 {
                    for (p, &j) in top.iter().enumerate() {
                        weights_tile_per_expert[j][i_off] = tmp[p] / sum;
                        used[j] = true;
                    }
                }
            }

            // Accumulator for this token tile
            let mut acc_tile = Tensor::<B, 2>::zeros([take_n, d], &norm_tile.device());

            let mut experts_used = 0usize;
            let mut tokens_used_total = 0usize;
            for expert in 0..e {
                if !used[expert] { continue; }
                experts_used += 1;

                // Expert mmap offsets and slices
                let off = &ctx.experts[expert];
                let s1 = off.mlp1_scales_off as usize;
                let b1_off = off.mlp1_blocks_off as usize;
                let s2 = off.mlp2_scales_off as usize;
                let b2_off = off.mlp2_blocks_off as usize;
                let blocks1 = &ctx.mmap[b1_off..b1_off + rows1 * bpr1];
                let scales1 = &ctx.mmap[s1..s1 + rows1];
                let blocks2 = &ctx.mmap[b2_off..b2_off + rows2 * bpr2];
                let scales2 = &ctx.mmap[s2..s2 + rows2];

                // Bias tensors
                let b1_full = self
                    .mlp1_bias
                    .val()
                    .clone()
                    .slice([expert..expert + 1, 0..self.ffn_hidden * 2])
                    .reshape([self.ffn_hidden * 2]);
                let b2 = self
                    .mlp2_bias
                    .val()
                    .clone()
                    .slice([expert..expert + 1, 0..self.d_model])
                    .reshape([self.d_model]);

                // Build compacted index ranges for tokens routed to this expert
                let w_host_tile = &weights_tile_per_expert[expert];
                let mut ranges: Vec<(usize, usize)> = Vec::new();
                let mut i = 0usize;
                while i < take_n {
                    // skip zeros
                    while i < take_n && w_host_tile[i] == 0.0 { i += 1; }
                    if i >= take_n { break; }
                    let start_i = i;
                    while i < take_n && w_host_tile[i] != 0.0 { i += 1; }
                    let end_i = i; // exclusive
                    ranges.push((start_i, end_i));
                }
                if ranges.is_empty() { continue; }
                tokens_used_total += ranges.iter().map(|(a,b)| b - a).sum::<usize>();

                // f-dimension tiling; accumulate expert contribution (pre-bias, pre-weight) for this tile
                let f = self.ffn_hidden;
                let tile_f = usize::min(256, f.max(1));
                let mut start_f = 0;
                let mut ex_tile_sum = Tensor::<B, 2>::zeros([take_n, d], &norm_tile.device());
                while start_f < f {
                    let take_f = core::cmp::min(tile_f, f - start_f);
                    // Dequant W1 rows tile
                    let w1_a = dequant_rows_range(blocks1, scales1, start_f, take_f, cols1);
                    let w1_b = dequant_rows_range(blocks1, scales1, f + start_f, take_f, cols1);
                    let mut w1_tile = Vec::with_capacity((take_f * 2) * cols1);
                    w1_tile.extend_from_slice(&w1_a);
                    w1_tile.extend_from_slice(&w1_b);
                    let w1_t = Tensor::<B, 2>::from_floats(burn_tensor::TensorData::new(w1_tile, [take_f * 2, cols1]), &norm_tile.device());
                    // Bias for this W1 tile
                    let b1_a = b1_full.clone().slice([start_f..start_f + take_f]);
                    let b1_b = b1_full.clone().slice([f + start_f..f + start_f + take_f]);
                    let b1_t = Tensor::cat(vec![b1_a, b1_b], 0);

                    // Dequant W2 columns tile once for this f-slice
                    let w2_cols = dequant_cols_range(blocks2, scales2, start_f, take_f, rows2, cols2);
                    let w2_t = Tensor::<B, 2>::from_floats(burn_tensor::TensorData::new(w2_cols, [rows2, take_f]), &norm_tile.device());

                    // Compact all routed tokens for this expert into a single batch to reduce kernel launches
                    let total_m: usize = ranges.iter().map(|(a,b)| b - a).sum();
                    if total_m > 0 {
                        let mut parts: Vec<Tensor<B, 2>> = Vec::with_capacity(ranges.len());
                        for (rs, re) in ranges.iter().copied() {
                            parts.push(norm_tile.clone().slice([rs..re, 0..d]));
                        }
                        let norm_compact = Tensor::cat(parts, 0); // [m, d]
                        let up_compact = norm_compact.matmul(w1_t.clone().transpose()) + b1_t.clone().unsqueeze();
                        let act_compact = crate::activation::swiglu_clamp(up_compact, self.swiglu_alpha, Some(self.swiglu_limit));
                        let part_compact = act_compact.matmul(w2_t.clone().transpose()); // [m, d]
                        // Scatter-add compact results back into ex_tile_sum using the same ranges
                        let mut offset = 0usize;
                        for (rs, re) in ranges.iter().copied() {
                            let r_len = re - rs;
                            let sub = part_compact.clone().slice([offset..offset + r_len, 0..d]);
                            let ex_slice = ex_tile_sum.clone().slice([rs..re, 0..d]);
                            let ex_new = ex_slice + sub;
                            ex_tile_sum.inplace(|t| t.slice_assign([rs..re, 0..d], ex_new.clone()));
                            offset += r_len;
                        }
                    }

                    start_f += take_f;
                }

                // After accumulating all f-tiles: apply bias and weights per range, then accumulate into acc_tile
                for (rs, re) in ranges.iter().copied() {
                    let r_len = re - rs;
                    let mut out_sub = ex_tile_sum.clone().slice([rs..re, 0..d]);
                    out_sub = out_sub + b2.clone().unsqueeze();
                    let w_ex_sub = Tensor::<B, 1>::from_floats(&w_host_tile[rs..re], &norm_tile.device()).reshape([r_len, 1]);
                    out_sub = out_sub * w_ex_sub; // [r_len, d]
                    let acc_slice = acc_tile.clone().slice([rs..re, 0..d]);
                    let new_slice = acc_slice + out_sub;
                    acc_tile.inplace(|t| t.slice_assign([rs..re, 0..d], new_slice.clone()));
                }
            }
            // Touch counters to avoid unused warnings when debug is off
            let _ = (experts_used, tokens_used_total);
            if tokens_used_total > 0 {
                moe_dbg!(
                    "moe tile: experts_used={} tokens_used_total={} take_n={}",
                    experts_used, tokens_used_total, take_n
                );
            }

            // Write accumulated tile into output accumulator
            acc.inplace(|t| t.slice_assign([start_n..start_n + take_n, 0..d], acc_tile));

            start_n += take_n;
        }

        let y = acc.reshape([b, t, d]);
        x + y
    }
}
