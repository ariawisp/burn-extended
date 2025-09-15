use burn_core as burn;

use burn::config::Config;
use burn::module::Module;
use burn::nn::conv::{Conv3d, Conv3dConfig, ConvTranspose3d, ConvTranspose3dConfig};
use burn::tensor::{Tensor, backend::Backend};

#[derive(Config, Debug)]
pub struct VideoPatchEmbeddingConfig {
    pub in_channels: usize,
    pub embed_dim: usize,
    pub patch_size: [usize; 3],
    #[config(default = "None")]
    pub stride: Option<[usize; 3]>,
    #[config(default = true)]
    pub bias: bool,
}

#[derive(Module, Debug)]
pub struct VideoPatchEmbedding<B: Backend> { conv: Conv3d<B>, patch: [usize; 3], stride: [usize; 3] }

impl VideoPatchEmbeddingConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> VideoPatchEmbedding<B> {
        let stride = self.stride.unwrap_or(self.patch_size);
        let conv = Conv3dConfig::new([self.in_channels, self.embed_dim], self.patch_size)
            .with_stride(stride)
            .with_bias(self.bias)
            .init(device);
        VideoPatchEmbedding { conv, patch: self.patch_size, stride }
    }
}

impl<B: Backend> VideoPatchEmbedding<B> {
    pub fn grid_sizes(&self, depth: usize, height: usize, width: usize) -> [usize; 3] {
        let [st, sh, sw] = self.stride;
        let [kt, kh, kw] = self.patch;
        let fp = depth.saturating_sub(kt) / st + 1;
        let hp = height.saturating_sub(kh) / sh + 1;
        let wp = width.saturating_sub(kw) / sw + 1;
        [fp, hp, wp]
    }
    pub fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 3> {
        let [b, _c, _f, _h, _w] = x.dims();
        let y = self.conv.forward(x); // [B, D, Fp, Hp, Wp]
        let [_, d, fp, hp, wp] = y.dims();
        y.reshape([b, d, fp * hp * wp]).swap_dims(1, 2)
    }
    pub fn forward_5d(&self, x: Tensor<B, 5>) -> Tensor<B, 5> { self.conv.forward(x) }
}

#[derive(Config, Debug)]
pub struct VideoUnpatchifyConfig {
    pub out_channels: usize,
    pub patch_size: [usize; 3],
    #[config(default = "None")]
    pub stride: Option<[usize; 3]>,
    #[config(default = true)]
    pub bias: bool,
}

#[derive(Module, Debug)]
pub struct VideoUnpatchify<B: Backend> { deconv: ConvTranspose3d<B>, stride: [usize; 3] }

impl VideoUnpatchifyConfig {
    pub fn init<B: Backend>(&self, embed_dim: usize, device: &B::Device) -> VideoUnpatchify<B> {
        let stride = self.stride.unwrap_or(self.patch_size);
        let deconv = ConvTranspose3dConfig::new([embed_dim, self.out_channels], self.patch_size)
            .with_stride(stride)
            .with_bias(self.bias)
            .init(device);
        VideoUnpatchify { deconv, stride }
    }
}

impl<B: Backend> VideoUnpatchify<B> {
    pub fn forward(&self, tokens: Tensor<B, 3>, grid_sizes: [usize; 3]) -> Tensor<B, 5> {
        let [b, n, d] = tokens.dims();
        let [fp, hp, wp] = grid_sizes;
        assert_eq!(n, fp * hp * wp);
        let x = tokens.swap_dims(1, 2).reshape([b, d, fp, hp, wp]);
        self.deconv.forward(x)
    }
    pub fn forward_5d(&self, x: Tensor<B, 5>) -> Tensor<B, 5> { self.deconv.forward(x) }
}

