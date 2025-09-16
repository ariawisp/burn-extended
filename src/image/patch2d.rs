use burn_core as burn;

use burn::config::Config;
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig};
use burn::tensor::{backend::Backend, Tensor};

#[derive(Config, Debug)]
pub struct ImagePatchEmbeddingConfig {
    pub in_channels: usize,
    pub embed_dim: usize,
    pub patch_size: [usize; 2],
    #[config(default = "None")]
    pub stride: Option<[usize; 2]>,
    #[config(default = true)]
    pub bias: bool,
}

#[derive(Module, Debug)]
pub struct ImagePatchEmbedding<B: Backend> {
    conv: Conv2d<B>,
    patch: [usize; 2],
    stride: [usize; 2],
}

impl ImagePatchEmbeddingConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ImagePatchEmbedding<B> {
        let stride = self.stride.unwrap_or(self.patch_size);
        let conv = Conv2dConfig::new([self.in_channels, self.embed_dim], self.patch_size)
            .with_stride(stride)
            .with_bias(self.bias)
            .init(device);
        ImagePatchEmbedding {
            conv,
            patch: self.patch_size,
            stride,
        }
    }
}

impl<B: Backend> ImagePatchEmbedding<B> {
    pub fn grid_sizes(&self, height: usize, width: usize) -> [usize; 2] {
        let [sh, sw] = self.stride;
        let [kh, kw] = self.patch;
        let hp = height.saturating_sub(kh) / sh + 1;
        let wp = width.saturating_sub(kw) / sw + 1;
        [hp, wp]
    }
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let [b, _c, _h, _w] = x.dims();
        let y = self.conv.forward(x); // [B, D, Hp, Wp]
        let [_, d, hp, wp] = y.dims();
        y.reshape([b, d, hp * wp]).swap_dims(1, 2)
    }
    pub fn forward_4d(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.conv.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct ImageUnpatchifyConfig {
    pub out_channels: usize,
    pub patch_size: [usize; 2],
    #[config(default = "None")]
    pub stride: Option<[usize; 2]>,
    #[config(default = true)]
    pub bias: bool,
}

#[derive(Module, Debug)]
pub struct ImageUnpatchify<B: Backend> {
    deconv: ConvTranspose2d<B>,
    stride: [usize; 2],
}

impl ImageUnpatchifyConfig {
    pub fn init<B: Backend>(&self, embed_dim: usize, device: &B::Device) -> ImageUnpatchify<B> {
        let stride = self.stride.unwrap_or(self.patch_size);
        let deconv = ConvTranspose2dConfig::new([embed_dim, self.out_channels], self.patch_size)
            .with_stride(stride)
            .with_bias(self.bias)
            .init(device);
        ImageUnpatchify { deconv, stride }
    }
}

impl<B: Backend> ImageUnpatchify<B> {
    pub fn forward(&self, tokens: Tensor<B, 3>, grid_sizes: [usize; 2]) -> Tensor<B, 4> {
        let [b, n, d] = tokens.dims();
        let [hp, wp] = grid_sizes;
        assert_eq!(n, hp * wp);
        let x = tokens.swap_dims(1, 2).reshape([b, d, hp, wp]);
        self.deconv.forward(x)
    }
    pub fn forward_4d(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.deconv.forward(x)
    }
}
