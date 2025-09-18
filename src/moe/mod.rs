use burn as burn;

use burn::module::Module;
use burn::nn::{Initializer, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::tensor::{backend::Backend, Tensor, Int};
use burn::tensor::TensorPrimitive;
use cubecl::prelude::{CubeDim, CubeCount};
use bytemuck;
// No CubeCL kernels here; using Burn tensor ops on WGPU
use alloc::sync::Arc;

// Per-expert offsets for resident GPU path.
#[derive(Debug, Clone)]
pub struct MoeExpertOffsets {
    pub mlp1_blocks_off: u64,
    pub mlp1_scales_off: u64,
    pub mlp2_blocks_off: u64,
    pub mlp2_scales_off: u64,
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
    pub verbose: bool,
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
    pub verbose: bool,
    // Resident quantized weights (device). Single canonical path expects this to be present.
    pub resident: burn::module::Ignored<Option<Arc<crate::quant::MoeQuantLayerResident>>>,
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
            verbose: self.verbose,
            resident: burn::module::Ignored(None),
        }
    }
}

impl<B: Backend> MoeGatedSwiGLU<B> {
    pub fn set_resident(&mut self, res: Arc<crate::quant::MoeQuantLayerResident>) {
        self.resident = burn::module::Ignored(Some(res));
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3>
    where
        B: Backend<
            FloatTensorPrimitive = burn_cubecl::tensor::CubeTensor<cubecl::wgpu::WgpuRuntime>,
            IntTensorPrimitive = burn_cubecl::tensor::CubeTensor<cubecl::wgpu::WgpuRuntime>,
        >,
    {
        if self.disabled {
            return x;
        }
        let normed = self.norm.forward(x.clone());
        let gates = self.gate.forward(normed.clone());
        if self.resident.0.is_none() {
            panic!("MoE resident buffers missing; fastest path requires device-resident weights");
        }
        self.forward_resident(x, normed, gates)
    }

    // Streaming path removed (resident-only fast path)
    fn forward_streaming(&self, x: Tensor<B, 3>, _ctx: &()) -> Tensor<B, 3> {
        x
    }

    fn forward_resident(&self, x: Tensor<B, 3>, normed: Tensor<B, 3>, gates: Tensor<B, 3>) -> Tensor<B, 3>
    where
        B: Backend<
            FloatTensorPrimitive = burn_cubecl::tensor::CubeTensor<cubecl::wgpu::WgpuRuntime>,
            IntTensorPrimitive = burn_cubecl::tensor::CubeTensor<cubecl::wgpu::WgpuRuntime>,
        >,
    {
        // Shapes
        let [b, t, d] = x.dims();
        let n = b * t;
        let e = self.num_experts;
        let k = self.experts_per_token.min(e).max(1);
        let norm_flat = normed.clone().reshape([n, d]);
        // Compute top-k on device via Burn op
        let gates2d = gates.clone().reshape([n, e]);
        let (topv, topi) = gates2d.clone().topk_with_indices(k, 1);
        let maxv = topv.clone().max_dim(1).reshape([n, 1]);
        let exps = (topv - maxv).exp();
        let sumv = exps.clone().sum_dim(1).reshape([n, 1]);
        let soft = exps / sumv; // [n,k]

        let res = self.resident.0.as_ref().unwrap();
        let desc1 = res.w1.desc;
        let desc2 = res.w2.desc;
        let cols1 = desc1.cols; // d
        let rows2 = desc2.rows; // d
        let f = self.ffn_hidden;
        let ue8 = desc1.ue8_offset;
        let gpr1 = desc1.gpr; // scale groups per row for W1
        let gpr2 = desc2.gpr; // scale groups per row for W2

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

        let mut acc = Tensor::<B, 2>::zeros([n, d], &norm_flat.device());
        let tile_n = usize::min(256, usize::max(1, n));
        let mut start_n = 0usize;
        while start_n < n {
            let take_n = core::cmp::min(tile_n, n - start_n);
            let norm_tile = norm_flat.clone().slice([start_n..start_n + take_n, 0..d]);
            let mut acc_tile = Tensor::<B, 2>::zeros([take_n, d], &norm_tile.device());

            // Compute top-k for this tile
            let gates_tile = gates.clone().reshape([n, e]).slice([start_n..start_n + take_n, 0..e]);
            let (topv_t, topi_t) = gates_tile.clone().topk_with_indices(k, 1);
            let maxv_t = topv_t.clone().max_dim(1).reshape([take_n, 1]);
            let soft_t = (topv_t - maxv_t).exp();
            let sumv_t = soft_t.clone().sum_dim(1).reshape([take_n, 1]);
            let soft_t = soft_t / sumv_t; // [take_n,k]

            // Compute expert weights per row on device: weights[row] = sum_p soft[row,p] where topi[row,p] == expert
            for expert in 0..e {
                let expert_fill = Tensor::<B, 2, Int>::from_ints(
                    burn_tensor::TensorData::new(vec![expert as i64; take_n * k], [take_n, k]),
                    &topi_t.device(),
                );
                let mask = topi_t.clone().equal(expert_fill);
                let soft_sel = soft_t.clone().mask_where(mask, Tensor::<B, 2>::zeros([take_n, k], &soft_t.device()));
                let weights_t = soft_sel.sum_dim(1).reshape([take_n, 1]);
                let any_used = weights_t.clone().sum().into_data().convert::<f32>().into_vec::<f32>().expect("sum")[0] != 0.0;
                if !any_used { continue; }
                // Bias tensors for this expert
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

                // f-tiling with fused kernels
                let base_block1 = (expert * (desc1.rows * desc1.bpr)) as u32;
                let base_scale1 = (expert * (desc1.rows * desc1.gpr)) as u32;
                let base_block2 = (expert * (desc2.rows * desc2.bpr)) as u32;
                let base_scale2 = (expert * (desc2.rows * desc2.gpr)) as u32;
                let ue8_127 = (desc1.ue8_offset + 127) as u32;
                let client = res.w1.blocks.client.clone();
                let device = res.w1.blocks.device.clone();
                let input_cube = norm_tile.clone().into_primitive().tensor();
                let weights_cube = weights_t.clone().into_primitive().tensor();
                let b2_cube = b2.clone().into_primitive().tensor();
                let mut ex_tile_sum = Tensor::<B, 2>::zeros([take_n, d], &norm_tile.device());
                let mut start_f_loc = 0u32;
                let tile_f = core::cmp::min(256u32, (f as u32).max(1));
                while start_f_loc < (f as u32) {
                    let take_f_loc = core::cmp::min(tile_f, (f as u32) - start_f_loc);
                    // Bias slices
                    let b1_a = b1_full.clone().slice([start_f_loc as usize..(start_f_loc + take_f_loc) as usize]).reshape([take_f_loc as usize]);
                    let b1_b = b1_full.clone().slice([(f as u32 + start_f_loc) as usize..(f as u32 + start_f_loc + take_f_loc) as usize]).reshape([take_f_loc as usize]);
                    let b1_a_cube = b1_a.into_primitive().tensor();
                    let b1_b_cube = b1_b.into_primitive().tensor();
                    // Act tile buffer
                    let act_bytes = take_n * (take_f_loc as usize) * core::mem::size_of::<f32>();
                    let act_handle = client.empty(act_bytes);
                    let act_cube = burn_cubecl::tensor::CubeTensor::new_contiguous(client.clone(), device.clone(), [take_n, take_f_loc as usize].into(), act_handle, burn_tensor::DType::F32);
                    // Build W1 fused cfg buffer [base_block, base_scale, cols, bpr, gpr, ue8_127, row0_start, row1_start]
                    let w1_cfg_host: [u32; 8] = [
                        base_block1,
                        base_scale1,
                        cols1 as u32,
                        desc1.bpr as u32,
                        desc1.gpr as u32,
                        ue8_127,
                        start_f_loc,
                        (f as u32) + start_f_loc,
                    ];
                    let w1_cfg_handle = client.create(bytemuck::cast_slice(&w1_cfg_host));
                    let w1_cfg = burn_cubecl::tensor::CubeTensor::new_contiguous(
                        client.clone(),
                        device.clone(),
                        [w1_cfg_host.len()].into(),
                        w1_cfg_handle,
                        burn_tensor::DType::U32,
                    );
                    // Launch W1 + SwiGLU
                    let cube_dim_a = CubeDim { x: 32, y: 8, z: 1 };
                    let gx_a = ((take_n as u32) + cube_dim_a.x - 1) / cube_dim_a.x;
                    let gy_a = (take_f_loc + cube_dim_a.y - 1) / cube_dim_a.y;
                    let count_a = CubeCount::Static(gx_a, gy_a, 1);
                    crate::quant::w1_swiglu_dense_kernel::launch::<burn::backend::wgpu::WgpuRuntime>(
                        &client,
                        count_a,
                        cube_dim_a,
                        input_cube.as_tensor_arg::<f32>(1),
                        res.w1.blocks.as_array_arg::<u8>(1),
                        res.w1.scales.as_array_arg::<u8>(1),
                        act_cube.as_tensor_arg::<f32>(1),
                        w1_cfg.as_array_arg::<u32>(1),
                        b1_a_cube.as_tensor_arg::<f32>(1),
                        b1_b_cube.as_tensor_arg::<f32>(1),
                    );
                    // Scatter W2
                    let ex_tile_cube = ex_tile_sum.clone().into_primitive().tensor();
                    let cube_dim_b = CubeDim { x: 32, y: 8, z: 1 };
                    let gx_b = ((take_n as u32) + cube_dim_b.x - 1) / cube_dim_b.x;
                    let gy_b = ((rows2 as u32) + cube_dim_b.y - 1) / cube_dim_b.y;
                    let count_b = CubeCount::Static(gx_b, gy_b, 1);
                    // Build W2 fused cfg buffer [base_block, base_scale, rows, cols, bpr, gpr, ue8_127, f_start, take_f]
                    let w2_cfg_host: [u32; 9] = [
                        base_block2,
                        base_scale2,
                        rows2 as u32,
                        f as u32,
                        desc2.bpr as u32,
                        desc2.gpr as u32,
                        ue8_127,
                        start_f_loc,
                        take_f_loc,
                    ];
                    let w2_cfg_handle = client.create(bytemuck::cast_slice(&w2_cfg_host));
                    let w2_cfg = burn_cubecl::tensor::CubeTensor::new_contiguous(
                        client.clone(),
                        device.clone(),
                        [w2_cfg_host.len()].into(),
                        w2_cfg_handle,
                        burn_tensor::DType::U32,
                    );
                    crate::quant::w2_scatter_dense_kernel::launch::<burn::backend::wgpu::WgpuRuntime>(
                        &client,
                        count_b,
                        cube_dim_b,
                        act_cube.as_tensor_arg::<f32>(1),
                        weights_cube.as_tensor_arg::<f32>(1),
                        res.w2.blocks.as_array_arg::<u8>(1),
                        res.w2.scales.as_array_arg::<u8>(1),
                        ex_tile_cube.as_tensor_arg::<f32>(1),
                        b2_cube.as_tensor_arg::<f32>(1),
                        w2_cfg.as_array_arg::<u32>(1),
                    );
                    ex_tile_sum = Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(ex_tile_cube));
                    start_f_loc += take_f_loc;
                }
                acc_tile = acc_tile + ex_tile_sum;
            }
            acc.inplace(|t| t.slice_assign([start_n..start_n + take_n, 0..d], acc_tile));
            start_n += take_n;
        }
        let y = acc.reshape([b, t, d]);
        x + y
    }
}
