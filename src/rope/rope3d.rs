use burn_core as burn;

use burn::config::Config;
use burn::module::{Content, DisplaySettings, Module, ModuleDisplay};
use burn::tensor::{Int, Tensor, backend::Backend};

#[derive(Config, Debug)]
pub struct Rope3dEncodingConfig {
    pub max_f: usize,
    pub max_h: usize,
    pub max_w: usize,
    pub d_head: usize,
    #[config(default = "None")]
    pub half_dim_split: Option<[usize; 3]>,
    #[config(default = "10000.0")]
    pub theta: f32,
}

impl Rope3dEncodingConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Rope3dEncoding<B> {
        assert!(self.d_head % 2 == 0, "d_head must be even");
        let half = self.d_head / 2;
        let split = if let Some([f, h, w]) = self.half_dim_split {
            assert!(f + h + w == half);
            [f, h, w]
        } else {
            let base = half / 3;
            let rem = half - base * 2;
            [rem, base, base]
        };
        Rope3dEncoding::new(self.max_f, self.max_h, self.max_w, split, self.theta, device)
    }
}

#[derive(Module, Debug)]
#[module(custom_display)]
pub struct Rope3dEncoding<B: Backend> {
    freq_f: Tensor<B, 3>,
    freq_h: Tensor<B, 3>,
    freq_w: Tensor<B, 3>,
    split: [usize; 3],
}

impl<B: Backend> ModuleDisplay for Rope3dEncoding<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new().with_new_line_after_attribute(false).optional()
    }
    fn custom_content(&self, content: Content) -> Option<Content> {
        let [f_pairs, h_pairs, w_pairs] = self.split;
        content.add("f_pairs", &f_pairs).add("h_pairs", &h_pairs).add("w_pairs", &w_pairs).optional()
    }
}

impl<B: Backend> Rope3dEncoding<B> {
    fn new(
        max_f: usize,
        max_h: usize,
        max_w: usize,
        split: [usize; 3],
        theta: f32,
        device: &B::Device,
    ) -> Self {
        let [f_pairs, h_pairs, w_pairs] = split;
        let freq_f = Self::precompute_axis(max_f, f_pairs, theta, device);
        let freq_h = Self::precompute_axis(max_h, h_pairs, theta, device);
        let freq_w = Self::precompute_axis(max_w, w_pairs, theta, device);
        Self { freq_f, freq_h, freq_w, split }
    }

    fn precompute_axis(max_pos: usize, pairs: usize, theta: f32, device: &B::Device) -> Tensor<B, 3> {
        if pairs == 0 { return Tensor::<B, 3>::zeros([max_pos, 0, 2], device); }
        let exponent = Tensor::<B, 1, Int>::arange(0..pairs as i64, device).float().div_scalar(pairs as f32 * 2.0);
        let base = exponent.mul_scalar(theta.ln()).exp().recip();
        let pos = Tensor::<B, 1, Int>::arange(0..max_pos as i64, device).float().unsqueeze();
        let base = base.unsqueeze_dim::<2>(0);
        let freqs = pos.matmul(base);
        let cos = freqs.clone().cos().unsqueeze_dim::<3>(2);
        let sin = freqs.sin().unsqueeze_dim::<3>(2);
        Tensor::cat(vec![cos, sin], 2)
    }

    pub fn apply(&self, x: Tensor<B, 4>, grid_sizes: [usize; 3], start_frame: usize) -> Tensor<B, 4> {
        let [b, s, n_heads, d_head] = x.dims();
        assert_eq!(d_head % 2, 0);
        let half = d_head / 2;
        let [f_pairs, h_pairs, w_pairs] = self.split;
        assert_eq!(f_pairs + h_pairs + w_pairs, half);
        let [f, h, w] = grid_sizes;
        assert_eq!(f * h * w, s);

        // Indices
        let hw = h * w;
        let f_idx = Tensor::<B, 1, Int>::arange(0..s as i64, &x.device())
            .float()
            .div_scalar(hw as f32)
            .floor()
            .int()
            .add_scalar(start_frame as i64)
            .clamp(0, self.freq_f.dims()[0] as i64 - 1);
        let h_idx = Tensor::<B, 1, Int>::arange(0..s as i64, &x.device())
            .div_scalar(w as i64)
            .remainder_scalar(h as i64);
        let w_idx = Tensor::<B, 1, Int>::arange(0..s as i64, &x.device()).remainder_scalar(w as i64);

        let gather_axis = |table: &Tensor<B, 3>, idx: Tensor<B, 1, Int>| -> Tensor<B, 3> {
            let max = table.dims()[0];
            let pairs = table.dims()[1];
            if pairs == 0 { return Tensor::<B, 3>::zeros([s, 0, 2], &x.device()); }
            let idx = idx.clamp(0, max as i64 - 1).reshape([s, 1, 1]).repeat_dim(1, pairs).repeat_dim(2, 2);
            table.clone().gather(0, idx)
        };
        let fc = gather_axis(&self.freq_f, f_idx);
        let hc = gather_axis(&self.freq_h, h_idx);
        let wc = gather_axis(&self.freq_w, w_idx);
        let freq = Tensor::cat(vec![fc, hc, wc], 1).unsqueeze_dim::<4>(2).repeat_dim(2, 2).reshape([s, d_head, 2]);

        let sign = Tensor::<B, 2>::from_floats([[1.0, 0.0, 0.0, 1.0], [0.0, -1.0, 1.0, 0.0]], &x.device());
        let x_rs = x.reshape([b * n_heads, s, d_head / 2, 2]).matmul(sign.unsqueeze()).reshape([b * n_heads, s, d_head, 2]);
        let out = x_rs * freq.clone().unsqueeze_dim::<4>(0).repeat_dim(0, b * n_heads).reshape([b * n_heads, s, d_head, 2]);
        out.sum_dim(3).reshape([b, s, n_heads, d_head])
    }
}

