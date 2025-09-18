// burn_core not required in CPU-only path

#[derive(Debug, Clone, Copy)]
pub struct QuantLinearMxFp4Desc {
    pub experts: usize,
    pub rows: usize,
    pub cols: usize,
    pub bpr: usize, // bytes per row of blocks
    pub gpr: usize, // groups per row for scales (per 32-col)
    pub ue8_offset: i32,
}

#[derive(Debug, Clone)]
pub struct MoeQuantLayerHost {
    pub w1_desc: QuantLinearMxFp4Desc,
    pub w2_desc: QuantLinearMxFp4Desc,
    pub w1_blocks: alloc::vec::Vec<u8>,
    pub w1_scales: alloc::vec::Vec<u8>,
    pub w2_blocks: alloc::vec::Vec<u8>,
    pub w2_scales: alloc::vec::Vec<u8>,
}

// GPU path (CubeCL + WGPU)
mod gpu;
pub use gpu::*;
