use burn_core as burn;

use burn::tensor::{backend::Backend, Tensor};

use super::scheduler::{tensor_from_vec, DiffusionScheduler};
use super::utils::logistic_rescale;

/// Configuration for the Euler flow-matching scheduler.
#[derive(Debug, Clone)]
pub struct FlowMatchEulerConfig {
    pub num_train_timesteps: usize,
    pub shift: f32,
    pub sigma_max: f32,
    /// If true, apply dynamic time shifting using `time_shift_mu` when building timesteps.
    pub use_dynamic_shifting: bool,
    /// Mu parameter for dynamic time shifting; if None and `use_dynamic_shifting` is true,
    /// no shifting is applied until set at runtime.
    pub time_shift_mu: Option<f32>,
    /// Logistic rescale parameters for omega in step().
    pub omega_lower: f32,
    pub omega_upper: f32,
    pub omega_x0: f32,
    pub omega_k: f32,
    /// Whether to mean-center the update like ACE-Step's variant.
    pub mean_center_update: bool,
}

impl Default for FlowMatchEulerConfig {
    fn default() -> Self {
        Self {
            num_train_timesteps: 1000,
            shift: 1.0,
            sigma_max: 1.0,
            use_dynamic_shifting: false,
            time_shift_mu: None,
            omega_lower: 0.9,
            omega_upper: 1.1,
            omega_x0: 0.0,
            omega_k: 0.1,
            mean_center_update: false,
        }
    }
}

/// Euler flow-matching scheduler.
pub struct FlowMatchEuler<B: Backend, const D: usize> {
    cfg: FlowMatchEulerConfig,
    sigmas: alloc::vec::Vec<f32>,
    timesteps: alloc::vec::Vec<f32>,
    step_index: usize,
    _phantom: core::marker::PhantomData<B>,
}

impl<B: Backend, const D: usize> FlowMatchEuler<B, D> {
    pub fn new(cfg: FlowMatchEulerConfig) -> Self {
        let mut sched = Self {
            cfg,
            sigmas: alloc::vec::Vec::new(),
            timesteps: alloc::vec::Vec::new(),
            step_index: 0,
            _phantom: core::marker::PhantomData,
        };
        sched.set_timesteps(50);
        sched
    }

    /// Apply ACE-Step style dynamic time shifting with the given `mu` for subsequent schedules.
    pub fn set_time_shift_mu(&mut self, mu: f32) {
        self.cfg.use_dynamic_shifting = true;
        self.cfg.time_shift_mu = Some(mu);
    }

    /// Disable dynamic time shifting.
    pub fn clear_time_shift(&mut self) {
        self.cfg.use_dynamic_shifting = false;
        self.cfg.time_shift_mu = None;
    }

    fn build_schedule(&mut self, num_inference_steps: usize) {
        let n = self.cfg.num_train_timesteps as f32;
        let sigma_min = 1.0 / n;
        let sigma_max = self.cfg.sigma_max;

        let mut sigmas = alloc::vec::Vec::with_capacity(num_inference_steps + 1);
        for i in 0..num_inference_steps {
            let a = i as f32 / (num_inference_steps.saturating_sub(1).max(1) as f32);
            let mut s = sigma_max * (1.0 - a) + sigma_min * a;
            if self.cfg.use_dynamic_shifting {
                if let Some(mu) = self.cfg.time_shift_mu {
                    // ACE-Step style time shift with sigma=1.0
                    let t = s;
                    let num = mu.exp();
                    let denom = num + (1.0 / t - 1.0).powf(1.0);
                    s = num / denom;
                }
            } else {
                s = self.cfg.shift * s / (1.0 + (self.cfg.shift - 1.0) * s);
            }
            sigmas.push(s);
        }
        sigmas.push(0.0);

        let timesteps = sigmas
            .iter()
            .map(|s| s * self.cfg.num_train_timesteps as f32)
            .collect();

        self.sigmas = sigmas;
        self.timesteps = timesteps;
        self.step_index = 0;
    }
}

impl<B: Backend, const D: usize> DiffusionScheduler<B, D> for FlowMatchEuler<B, D> {
    fn sigmas(&self, device: &B::Device) -> Tensor<B, 1> {
        tensor_from_vec(device, &self.sigmas)
    }

    fn timesteps(&self, device: &B::Device) -> Tensor<B, 1> {
        tensor_from_vec(device, &self.timesteps)
    }

    fn set_timesteps(&mut self, num_inference_steps: usize) {
        self.build_schedule(num_inference_steps);
    }

    fn num_train_timesteps(&self) -> usize {
        self.cfg.num_train_timesteps
    }

    fn reset(&mut self) {
        self.step_index = 0;
    }

    fn step_index(&self) -> usize {
        self.step_index
    }

    fn step(
        &mut self,
        model_output: Tensor<B, D>,
        _timestep: f32,
        sample: Tensor<B, D>,
        omega: f32,
    ) -> Tensor<B, D> {
        let i = self.step_index;
        let sigma = self.sigmas[i];
        let sigma_next = self.sigmas[i + 1];
        let omega = logistic_rescale(
            omega,
            self.cfg.omega_lower,
            self.cfg.omega_upper,
            self.cfg.omega_x0,
            self.cfg.omega_k,
        );
        let delta = sigma_next - sigma;
        let dx = model_output.mul_scalar(delta);
        let prev_sample = if self.cfg.mean_center_update {
            // Center dx mean and scale deviations by omega
            let m_val = dx.clone().mean().into_scalar();
            let m_full = Tensor::<B, D>::full(dx.shape().dims::<D>(), m_val, &dx.device());
            let centered = (dx - m_full.clone()).mul_scalar(omega) + m_full;
            sample + centered
        } else {
            sample + dx.mul_scalar(omega)
        };

        self.step_index += 1;
        prev_sample
    }

    fn scale_noise(
        &mut self,
        sample: Tensor<B, D>,
        timestep: f32,
        noise: Tensor<B, D>,
    ) -> Tensor<B, D> {
        let idx = self
            .timesteps
            .iter()
            .position(|t| (*t - timestep).abs() < 1e-3)
            .unwrap_or(self.step_index);
        let sigma = self.sigmas[idx];
        noise.mul_scalar(sigma).add(sample.mul_scalar(1.0 - sigma))
    }
}
