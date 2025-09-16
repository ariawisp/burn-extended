use burn_core as burn;

use burn::tensor::{backend::Backend, Tensor};

use super::{cfg as cfg_guidance, cfg_double, cfg_zero_star, apg as apg_guidance, DiffusionScheduler};

/// A lightweight, reusable diffusion pipeline that drives a scheduler with optional guidance.
pub struct DiffusionPipeline<S> {
    pub scheduler: S,
}

impl<S> DiffusionPipeline<S> {
    pub fn new(scheduler: S) -> Self {
        Self { scheduler }
    }
}

impl<S> DiffusionPipeline<S> {
    /// Run with a single model prediction function: `predict(sample, timestep)`.
    pub fn run_single<B: Backend, const D: usize, F>(
        &mut self,
        mut sample: Tensor<B, D>,
        num_steps: usize,
        mut predict: F,
        omega: f32,
    ) -> Tensor<B, D>
    where
        S: DiffusionScheduler<B, D>,
        F: FnMut(Tensor<B, D>, f32) -> Tensor<B, D>,
    {
        let device = sample.device();
        self.scheduler.set_timesteps(num_steps);
        self.scheduler.reset();
        let timesteps = self.scheduler.timesteps(&device).into_data().to_vec::<f32>().unwrap();
        for _ in 0..num_steps {
            let idx = self.scheduler.step_index();
            let t = timesteps[idx];
            let pred = predict(sample.clone(), t);
            sample = self.scheduler.step(pred, t, sample, omega);
        }
        sample
    }

    /// Run with classifier-free guidance: provide `predict_uncond` and `predict_cond`.
    pub fn run_cfg<B: Backend, const D: usize, FU, FC>(
        &mut self,
        mut sample: Tensor<B, D>,
        num_steps: usize,
        mut predict_uncond: FU,
        mut predict_cond: FC,
        guidance_scale: f32,
        omega: f32,
    ) -> Tensor<B, D>
    where
        S: DiffusionScheduler<B, D>,
        FU: FnMut(Tensor<B, D>, f32) -> Tensor<B, D>,
        FC: FnMut(Tensor<B, D>, f32) -> Tensor<B, D>,
    {
        let device = sample.device();
        self.scheduler.set_timesteps(num_steps);
        self.scheduler.reset();
        let timesteps = self.scheduler.timesteps(&device).into_data().to_vec::<f32>().unwrap();
        for _ in 0..num_steps {
            let idx = self.scheduler.step_index();
            let t = timesteps[idx];
            let uncond = predict_uncond(sample.clone(), t);
            let cond = predict_cond(sample.clone(), t);
            let guided = cfg_guidance(cond, uncond, guidance_scale);
            sample = self.scheduler.step(guided, t, sample, omega);
        }
        sample
    }

    /// Run with APG-like guidance; provide `predict_uncond` and `predict_cond`.
    pub fn run_apg<B: Backend, const D: usize, FU, FC>(
        &mut self,
        mut sample: Tensor<B, D>,
        num_steps: usize,
        mut predict_uncond: FU,
        mut predict_cond: FC,
        guidance_scale: f32,
        momentum: f32,
        norm_eps: Option<f32>,
        omega: f32,
    ) -> Tensor<B, D>
    where
        S: DiffusionScheduler<B, D>,
        FU: FnMut(Tensor<B, D>, f32) -> Tensor<B, D>,
        FC: FnMut(Tensor<B, D>, f32) -> Tensor<B, D>,
    {
        let device = sample.device();
        self.scheduler.set_timesteps(num_steps);
        self.scheduler.reset();
        let timesteps = self.scheduler.timesteps(&device).into_data().to_vec::<f32>().unwrap();
        let mut buf = super::MomentumBuffer::<B, D>::new(momentum);
        for _ in 0..num_steps {
            let idx = self.scheduler.step_index();
            let t = timesteps[idx];
            let uncond = predict_uncond(sample.clone(), t);
            let cond = predict_cond(sample.clone(), t);
            let guided = apg_guidance(cond, uncond, guidance_scale, Some(&mut buf), norm_eps);
            sample = self.scheduler.step(guided, t, sample, omega);
        }
        sample
    }

    /// Run with double CFG (e.g., text + lyric) guidance.
    pub fn run_cfg_double<B: Backend, const D: usize, FU, FC, FO>(
        &mut self,
        mut sample: Tensor<B, D>,
        num_steps: usize,
        mut predict_uncond: FU,
        mut predict_cond: FC,
        mut predict_only_text: FO,
        guidance_scale_text: f32,
        guidance_scale_lyric: f32,
        omega: f32,
    ) -> Tensor<B, D>
    where
        S: DiffusionScheduler<B, D>,
        FU: FnMut(Tensor<B, D>, f32) -> Tensor<B, D>,
        FC: FnMut(Tensor<B, D>, f32) -> Tensor<B, D>,
        FO: FnMut(Tensor<B, D>, f32) -> Tensor<B, D>,
    {
        let device = sample.device();
        self.scheduler.set_timesteps(num_steps);
        self.scheduler.reset();
        let timesteps = self.scheduler.timesteps(&device).into_data().to_vec::<f32>().unwrap();
        for _ in 0..num_steps {
            let idx = self.scheduler.step_index();
            let t = timesteps[idx];
            let uncond = predict_uncond(sample.clone(), t);
            let cond = predict_cond(sample.clone(), t);
            let only_text = predict_only_text(sample.clone(), t);
            let guided = cfg_double(cond, uncond, only_text, guidance_scale_text, guidance_scale_lyric);
            sample = self.scheduler.step(guided, t, sample, omega);
        }
        sample
    }

    /// Run with zero-star variant guidance.
    pub fn run_zero_star<B: Backend, const D: usize, FU, FC>(
        &mut self,
        mut sample: Tensor<B, D>,
        num_steps: usize,
        mut predict_uncond: FU,
        mut predict_cond: FC,
        guidance_scale: f32,
        omega: f32,
    ) -> Tensor<B, D>
    where
        S: DiffusionScheduler<B, D>,
        FU: FnMut(Tensor<B, D>, f32) -> Tensor<B, D>,
        FC: FnMut(Tensor<B, D>, f32) -> Tensor<B, D>,
    {
        let device = sample.device();
        self.scheduler.set_timesteps(num_steps);
        self.scheduler.reset();
        let timesteps = self.scheduler.timesteps(&device).into_data().to_vec::<f32>().unwrap();
        for _ in 0..num_steps {
            let idx = self.scheduler.step_index();
            let t = timesteps[idx];
            let uncond = predict_uncond(sample.clone(), t);
            let cond = predict_cond(sample.clone(), t);
            let guided = cfg_zero_star(cond, uncond, guidance_scale);
            sample = self.scheduler.step(guided, t, sample, omega);
        }
        sample
    }
}
