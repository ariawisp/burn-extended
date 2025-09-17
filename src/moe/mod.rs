use burn_core as burn;

use burn::module::{Ignored, Module};
use burn::nn::{Initializer, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::tensor::{backend::Backend, Int, Tensor};
use alloc::sync::Arc;

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
                // SafeTensors scales are stored with a bias of 127 (see loader/mxfp4.rs).
                // model.bin adds UE8_OFFSET on top. Undo both here: (byte - UE8_OFFSET - 127).
                let si = (scales[r] as i32) - UE8_OFFSET - 127;
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

        // MXFP4 layout helpers
        let ue8 = ctx.ue8_offset;
        let rows1 = ctx.rows_mlp1; // 2f
        let cols1 = ctx.cols_mlp1; // d_model
        let rows2 = ctx.rows_mlp2; // d_model
        let cols2 = ctx.cols_mlp2; // f
        let bpr1 = (cols1 + 1) / 2;
        let bpr2 = (cols2 + 1) / 2;
        let gpr1 = (cols1 + 31) / 32; // scale groups per row for W1
        let gpr2 = (cols2 + 31) / 32; // scale groups per row for W2

        // Build FP4 LUT [16] and column group indices [cols1] on device for this call
        const FP4_VALUES: [f32; 16] = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        let lut1d = Tensor::<B, 1>::from_floats(burn_tensor::TensorData::new(FP4_VALUES.to_vec(), [16]), &norm_flat.device());
        let w1_groups = Tensor::<B, 1, Int>::arange(0..(cols1 as i64), &norm_flat.device()).div_scalar(32 as i64);

        // Device dequant helpers (decode only required rows/cols on device using gather)
        let decode_rows_device_from_bytes = |blocks: &[u8], scales: &[u8], row_start: usize, row_count: usize, cols: usize, gpr: usize| -> Tensor<B, 2> {
            // Build blocks tile [rows, bpr]
            let bpr = (cols + 1) / 2;
            let mut blk_i64: Vec<i64> = Vec::with_capacity(row_count * bpr);
            for r in 0..row_count {
                let off = (row_start + r) * bpr;
                for i in 0..bpr { blk_i64.push(blocks[off + i] as i64); }
            }
            let blocks_t = Tensor::<B, 2, Int>::from_ints(
                burn_tensor::TensorData::new(blk_i64, [row_count, bpr]),
                &norm_flat.device(),
            );
            // Compute nibble indices
            let blocks_f = blocks_t.clone().float();
            let hi_f = blocks_f.clone().div_scalar(16.0f32).floor();
            let lo_f = blocks_f - hi_f.clone().mul_scalar(16.0f32);
            let hi_idx = hi_f.int();
            let lo_idx = lo_f.int();
            // Build per-row LUT for gather: [rows, 16]
            let lut_rows = lut1d.clone().unsqueeze_dim::<2>(0).repeat_dim(0, row_count);
            let lo_vals = lut_rows.clone().gather(1, lo_idx); // [rows,bpr]
            let hi_vals = lut_rows.gather(1, hi_idx); // [rows,bpr]
            let lo_u = lo_vals.unsqueeze_dim::<3>(2);
            let hi_u = hi_vals.unsqueeze_dim::<3>(2);
            let decoded = Tensor::cat(vec![lo_u, hi_u], 2).reshape([row_count, bpr * 2]);
            // Build per-element scales matrix [row_count, cols]
            let mut scl_i64: Vec<i64> = Vec::with_capacity(row_count * gpr);
            for r in 0..row_count {
                let off = (row_start + r) * gpr;
                for g in 0..gpr { scl_i64.push(scales[off + g] as i64); }
            }
            let scales_u8 = Tensor::<B, 2, Int>::from_ints(
                burn_tensor::TensorData::new(scl_i64, [row_count, gpr]),
                &norm_flat.device(),
            );
            let scales_f = scales_u8
                .float()
                .sub_scalar((ue8 + 127) as f32)
                .mul_scalar(core::f32::consts::LN_2)
                .exp(); // [row_count, gpr]
            let idx2d = w1_groups.clone().reshape([1, cols]).repeat_dim(0, row_count);
            let scales_full = scales_f.gather(1, idx2d);
            decoded * scales_full
        };

        let decode_cols_device_from_bytes = |blocks: &[u8], scales: &[u8], col_start: usize, col_count: usize, rows: usize, cols: usize, gpr: usize| -> Tensor<B, 2> {
            let bpr = (cols + 1) / 2;
            let start_byte = col_start / 2;
            let last_col = col_start + col_count - 1;
            let end_byte = last_col / 2;
            let bytes_count = end_byte - start_byte + 1;
            let offset_in_pair = (col_start % 2) as usize;
            // Blocks tile [rows, bytes_count]
            let mut blk_i64: Vec<i64> = Vec::with_capacity(rows * bytes_count);
            for r in 0..rows {
                let off = r * bpr + start_byte;
                for i in 0..bytes_count { blk_i64.push(blocks[off + i] as i64); }
            }
            let blocks_t = Tensor::<B, 2, Int>::from_ints(
                burn_tensor::TensorData::new(blk_i64, [rows, bytes_count]),
                &norm_flat.device(),
            );
            let blocks_f = blocks_t.clone().float();
            let hi_f = blocks_f.clone().div_scalar(16.0f32).floor();
            let lo_f = blocks_f - hi_f.clone().mul_scalar(16.0f32);
            let hi_idx = hi_f.int();
            let lo_idx = lo_f.int();
            let lut_rows = lut1d.clone().unsqueeze_dim::<2>(0).repeat_dim(0, rows);
            let lo_vals = lut_rows.clone().gather(1, lo_idx);
            let hi_vals = lut_rows.gather(1, hi_idx);
            let pairs = Tensor::cat(vec![lo_vals.unsqueeze_dim::<3>(2), hi_vals.unsqueeze_dim::<3>(2)], 2)
                .reshape([rows, bytes_count * 2]);
            let decoded = pairs.clone().slice([0..rows, offset_in_pair..offset_in_pair + col_count]);

            // Scales tile for columns
            let group_start = col_start / 32;
            let group_end = (last_col) / 32;
            let group_count = group_end - group_start + 1;
            let mut scl_i64: Vec<i64> = Vec::with_capacity(rows * group_count);
            for r in 0..rows {
                let off = r * gpr + group_start;
                for g in 0..group_count { scl_i64.push(scales[off + g] as i64); }
            }
            let scales_u8 = Tensor::<B, 2, burn::tensor::Int>::from_ints(
                burn_tensor::TensorData::new(scl_i64, [rows, group_count]),
                &norm_flat.device(),
            );
            let scales_f = scales_u8
                .float()
                .sub_scalar((ue8 + 127) as f32)
                .mul_scalar(core::f32::consts::LN_2)
                .exp();
            let rel_idx = Tensor::<B, 1, burn::tensor::Int>::arange(0..(col_count as i64), &norm_flat.device())
                .add_scalar(col_start as i64)
                .div_scalar(32 as i64)
                .sub_scalar(group_start as i64);
            let idx2d = rel_idx.reshape([1, col_count]).repeat_dim(0, rows);
            let scales_full = scales_f.gather(1, idx2d);
            decoded * scales_full
        };


        // Dequant a contiguous range of rows for W1 (shape [rows, cols1])
        // Device path only: no host dequant helpers retained

        // Token-tiling: limit intermediate [n, *] tensors to a manageable size
        let tile_n = usize::min(256, usize::max(1, n));
        let mut start_n = 0;
        while start_n < n {
            let take_n = core::cmp::min(tile_n, n - start_n);
            let norm_tile = norm_flat.clone().slice([start_n..start_n + take_n, 0..d]); // [take_n, d]

            // Build top-k on device and pull only per-expert weights for this token tile
            let gates_tile = gates.clone().reshape([n, e]).slice([start_n..start_n + take_n, 0..e]); // [take_n, e]
            let (topv, topi) = gates_tile.clone().topk_with_indices(k, 1); // [take_n,k] values, indices
            let maxv = topv.clone().max_dim(1).reshape([take_n, 1]);
            let exps = (topv - maxv).exp();
            let sumv = exps.clone().sum_dim(1).reshape([take_n, 1]);
            let soft = exps / sumv; // [take_n,k]
            let mut used = vec![false; e];
            let mut weights_tile_per_expert: Vec<Vec<f32>> = vec![vec![0.0f32; take_n]; e];
            // For each expert, select its weight per row if present among top-k
            for expert in 0..e {
                // mask = (topi == expert)
                let expert_fill = Tensor::<B, 2, Int>::from_ints(
                    burn_tensor::TensorData::new(vec![expert as i64; take_n * k], [take_n, k]),
                    &topi.device(),
                );
                let mask = topi.clone().equal(expert_fill);
                let masked = soft.clone().mask_fill(mask.clone(), 0.0);
                let w = masked.sum_dim(1); // [take_n]
                let w_host = w.into_data().convert::<f32>().into_vec::<f32>().expect("weights vec");
                // track used
                let mut any = false;
                for &v in &w_host { if v != 0.0 { any = true; break; } }
                used[expert] = any;
                weights_tile_per_expert[expert] = w_host;
            }

            // Accumulator for this token tile
            let mut acc_tile = Tensor::<B, 2>::zeros([take_n, d], &norm_tile.device());

            let mut experts_used = 0usize;
            let mut tokens_used_total = 0usize;
            let mut bytes_h2d_total: usize = 0;
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
                let scales1 = &ctx.mmap[s1..s1 + rows1 * gpr1];
                let blocks2 = &ctx.mmap[b2_off..b2_off + rows2 * bpr2];
                let scales2 = &ctx.mmap[s2..s2 + rows2 * gpr2];

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
                    // Estimate H2D bytes for this expert and f-slice
                    let w1_bytes = (take_f * 2) * bpr1 + (take_f * 2) * gpr1; // blocks + scales (u8)
                    let bytes_count = (take_f + 1) / 2; // W2 bytes per row
                    let w2_bytes = rows2 * bytes_count + rows2 * ((take_f + 31) / 32);
                    bytes_h2d_total += w1_bytes + w2_bytes;
                    // Dequant W1 rows tile on device from bytes
                    let w1_a = decode_rows_device_from_bytes(blocks1, scales1, start_f, take_f, cols1, gpr1);
                    let w1_b = decode_rows_device_from_bytes(blocks1, scales1, f + start_f, take_f, cols1, gpr1);
                    let w1_t = Tensor::cat(vec![w1_a, w1_b], 0); // [2*take_f, cols1]
                    // Bias for this W1 tile
                    let b1_a = b1_full.clone().slice([start_f..start_f + take_f]);
                    let b1_b = b1_full.clone().slice([f + start_f..f + start_f + take_f]);
                    let b1_t = Tensor::cat(vec![b1_a, b1_b], 0);

                    // Dequant W2 columns tile on device from bytes
                    let w2_t = decode_cols_device_from_bytes(blocks2, scales2, start_f, take_f, rows2, cols2, gpr2);

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
            // Log approximate H2D volume per token tile (bytes on host -> device as i64 indices)
            let bytes_mb = (bytes_h2d_total as f64) * 8.0 / (1024.0 * 1024.0); // rough upper bound if uploaded as Int
            eprintln!("moe tile: experts_used={} tokens_used_total={} take_n={} approx_H2D~{:.1} MiB", experts_used, tokens_used_total, take_n, bytes_mb);

            // Write accumulated tile into output accumulator
            acc.inplace(|t| t.slice_assign([start_n..start_n + take_n, 0..d], acc_tile));

            start_n += take_n;
        }

        let y = acc.reshape([b, t, d]);
        x + y
    }
}
