use burn as burn;

use alloc::vec::Vec;
use burn::tensor::backend::Backend;
use burn::backend::wgpu::WgpuRuntime;
use burn_cubecl::tensor::CubeTensor;
use cubecl::{cube, prelude::*};
// Using direct cubecl crates for fused kernels
// No custom CubeCL kernels here; resident path uses Burn ops on WGPU
use burn_wgpu::WgpuDevice;

use super::{MoeQuantLayerHost, QuantLinearMxFp4Desc};

#[derive(Debug, Clone)]
pub struct QuantLinearMxFp4Resident {
    pub blocks: CubeTensor<WgpuRuntime>, // raw u8 buffer [E * rows * bpr]
    pub scales: CubeTensor<WgpuRuntime>, // raw u8 buffer [E * rows * gpr]
    pub desc: QuantLinearMxFp4Desc,
    // Host copies retained for device-agnostic decode path
    pub blocks_host: Vec<u8>,
    pub scales_host: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct MoeQuantLayerResident {
    pub w1: QuantLinearMxFp4Resident,
    pub w2: QuantLinearMxFp4Resident,
}

impl MoeQuantLayerResident {
    pub fn new(
        device: &WgpuDevice,
        w1_desc: QuantLinearMxFp4Desc,
        w1_blocks: Vec<u8>,
        w1_scales: Vec<u8>,
        w2_desc: QuantLinearMxFp4Desc,
        w2_blocks: Vec<u8>,
        w2_scales: Vec<u8>,
    ) -> Self {
        let client = WgpuRuntime::client(device);
        let h1b = client.create(&w1_blocks);
        let h1s = client.create(&w1_scales);
        let h2b = client.create(&w2_blocks);
        let h2s = client.create(&w2_scales);
        let blocks1 = CubeTensor::new_contiguous(client.clone(), device.clone(), [w1_blocks.len()].into(), h1b, burn_tensor::DType::U8);
        let scales1 = CubeTensor::new_contiguous(client.clone(), device.clone(), [w1_scales.len()].into(), h1s, burn_tensor::DType::U8);
        let blocks2 = CubeTensor::new_contiguous(client.clone(), device.clone(), [w2_blocks.len()].into(), h2b, burn_tensor::DType::U8);
        let scales2 = CubeTensor::new_contiguous(client.clone(), device.clone(), [w2_scales.len()].into(), h2s, burn_tensor::DType::U8);
        Self {
            w1: QuantLinearMxFp4Resident { blocks: blocks1, scales: scales1, desc: w1_desc, blocks_host: w1_blocks, scales_host: w1_scales },
            w2: QuantLinearMxFp4Resident { blocks: blocks2, scales: scales2, desc: w2_desc, blocks_host: w2_blocks, scales_host: w2_scales },
        }
    }
}

// Helper to attach to the GPT-OSS model layers via downcast.
pub fn attach_quant_to_model_layers<B>(
    model: &mut dyn core::any::Any,
    layers: Vec<MoeQuantLayerResident>,
) where
    B: Backend<
        FloatTensorPrimitive = CubeTensor<WgpuRuntime>,
        IntTensorPrimitive = CubeTensor<WgpuRuntime>,
    >,
{
    if let Some(m_any) = model.downcast_mut::<crate::models::gpt_oss::GptOssModel<B>>() {
        m_any.set_moe_quant_resident(layers);
    }
}

pub fn attach_quant_from_host_to_model<B>(
    model: &mut dyn core::any::Any,
    layers: Vec<MoeQuantLayerHost>,
) where
    B: Backend<
        FloatTensorPrimitive = CubeTensor<WgpuRuntime>,
        IntTensorPrimitive = CubeTensor<WgpuRuntime>,
    >,
{
    if let Some(m_any) = model.downcast_mut::<crate::models::gpt_oss::GptOssModel<B>>() {
        m_any.set_moe_quant_from_host(layers);
    }
}

// =====================
// CubeCL fused kernels
// =====================

#[cube(launch)]
pub fn w1_swiglu_dense_kernel(
    input: &Tensor<f32>,
    blocks: &Array<u8>,
    scales: &Array<u8>,
    out: &mut Tensor<f32>,
    cfg: &Array<u32>, // [base_block, base_scale, cols, bpr, gpr, ue8_127, row0_start, row1_start]
    bias0: &Tensor<f32>,
    bias1: &Tensor<f32>,
) {
    let row = ABSOLUTE_POS_X;
    let f_rel = ABSOLUTE_POS_Y;
    if row >= out.shape(0) || f_rel >= out.shape(1) { terminate!(); }
    let base_block = cfg[0];
    let base_scale = cfg[1];
    let d = cfg[2];
    let bpr = cfg[3];
    let gpr = cfg[4];
    let ue8_127 = cfg[5];
    let row0_start = cfg[6];
    let row1_start = cfg[7];
    let in_s0 = input.stride(0);
    let in_s1 = input.stride(1);
    let row_g = row0_start + f_rel;
    let row_l = row1_start + f_rel;
    let mut sum_g: f32 = 0.0;
    let mut sum_l: f32 = 0.0;
    let lut: Array<f32> = Array::from_data::<f32>([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ]);
    for c in 0..d {
        let inp = input[row * in_s0 + c * in_s1];
        let g = c >> 5;
        // GLU
        let scale_g_u8 = scales[base_scale + row_g * gpr + g];
        let scale_g = Exp::exp(((scale_g_u8 as i32 - ue8_127 as i32) as f32) * core::f32::consts::LN_2);
        let byte_g = blocks[base_block + row_g * bpr + (c >> 1)];
        let nib_g = if (c & 1) == 0 { byte_g & 0x0F } else { (byte_g >> 4) & 0x0F };
        let w_g: f32 = lut[nib_g as u32] * scale_g;
        // LIN
        let scale_l_u8 = scales[base_scale + row_l * gpr + g];
        let scale_l = Exp::exp(((scale_l_u8 as i32 - ue8_127 as i32) as f32) * core::f32::consts::LN_2);
        let byte_l = blocks[base_block + row_l * bpr + (c >> 1)];
        let nib_l = if (c & 1) == 0 { byte_l & 0x0F } else { (byte_l >> 4) & 0x0F };
        let w_l: f32 = lut[nib_l as u32] * scale_l;
        sum_g += inp * w_g;
        sum_l += inp * w_l;
    }
    sum_g += bias0[f_rel * bias0.stride(0)];
    sum_l += bias1[f_rel * bias1.stride(0)];
    let limit = 7.0f32;
    let alpha = 1.702f32;
    let x_glu = Min::min(sum_g, limit);
    let x_lin = Max::max(-limit, Min::min(sum_l, limit));
    let sig = 1.0 / (1.0 + Exp::exp(-alpha * x_glu));
    out[row * out.stride(0) + f_rel * out.stride(1)] = (x_glu * sig) * (x_lin + 1.0);
}

#[cube(launch)]
pub fn w2_scatter_dense_kernel(
    act: &Tensor<f32>,
    weights: &Tensor<f32>,
    blocks: &Array<u8>,
    scales: &Array<u8>,
    out: &mut Tensor<f32>,
    bias: &Tensor<f32>,
    cfg: &Array<u32>, // [base_block, base_scale, rows, cols, bpr, gpr, ue8_127, f_start, take_f]
) {
    let row = ABSOLUTE_POS_X;
    let d_col = ABSOLUTE_POS_Y;
    if row >= act.shape(0) || d_col >= out.shape(1) { terminate!(); }
    let base_block = cfg[0];
    let base_scale = cfg[1];
    let _rows = cfg[2];
    let _cols = cfg[3];
    let bpr = cfg[4];
    let gpr = cfg[5];
    let ue8_127 = cfg[6];
    let f_start = cfg[7];
    let take_f = cfg[8];
    let w = weights[row * weights.stride(0)];
    if w == 0.0 { terminate!(); }
    let act_s0 = act.stride(0);
    let act_s1 = act.stride(1);
    let mut sum: f32 = 0.0;
    let lut: Array<f32> = Array::from_data::<f32>([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ]);
    for fr in 0..take_f {
        let f_global = f_start + fr;
        let a = act[row * act_s0 + fr * act_s1];
        let g = f_global >> 5;
        let scale_u8 = scales[base_scale + d_col * gpr + g];
        let scale = Exp::exp(((scale_u8 as i32 - ue8_127 as i32) as f32) * core::f32::consts::LN_2);
        let byte = blocks[base_block + d_col * bpr + (f_global >> 1)];
        let nib = if (f_global & 1) == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F };
        let wt: f32 = lut[nib as u32] * scale;
        sum += a * wt;
    }
    let out_index = row * out.stride(0) + d_col * out.stride(1);
    out[out_index] = out[out_index] + (sum + bias[d_col * bias.stride(0)]) * w;
}

// Fused kernels are defined with the `cubecl` attribute; keeping them here would
// require additional trait bounds for compilation on all targets. Since this
// crate is WGPU‑focused, we can enable them later if needed.
