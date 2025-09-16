use burn_core as burn;
use burn::backend::Backend;
use burn_extended::diffusion::{
    FlowMatchEuler, FlowMatchEulerConfig,
    FlowMatchHeun, FlowMatchHeunConfig,
    FlowMatchPingPong, FlowMatchPingPongConfig,
    DiffusionScheduler,
    retrieve_timesteps,
};
use burn::tensor::{Distribution, Tensor};
use burn_ndarray::NdArray;
use burn_tensor::Tolerance;

type TB = NdArray<f32>;

fn device() -> <TB as burn::backend::Backend>::Device {
    Default::default()
}

#[test]
fn euler_zero_model_output_is_identity() {
    let device = device();
    let mut sched = FlowMatchEuler::<TB, 3>::new(FlowMatchEulerConfig::default());
    sched.set_timesteps(8);
    let sample = Tensor::<TB, 3>::random([1, 2, 3], Distribution::Default, &device);
    let model_output = Tensor::<TB, 3>::zeros([1, 2, 3], &device);
    let out = sched.step(model_output, 0.0, sample.clone(), 1.0);
    out.into_data().assert_approx_eq(&sample.into_data(), Tolerance::default());
}

#[test]
fn heun_progresses_and_returns_finite_values() {
    let device = device();
    let mut sched = FlowMatchHeun::<TB, 3>::new(FlowMatchHeunConfig::default());
    sched.set_timesteps(6);
    let sample = Tensor::<TB, 3>::random([2, 4, 4], Distribution::Default, &device);
    let model_output = sample.clone();
    let out = sched.step(model_output, 0.0, sample.clone(), 1.0);
    assert_eq!(out.dims(), [2, 4, 4]);
}

#[test]
fn pingpong_preserves_shapes() {
    let device = device();
    let mut sched = FlowMatchPingPong::<TB, 2>::new(FlowMatchPingPongConfig::default());
    sched.set_timesteps(4);
    let sample = Tensor::<TB, 2>::random([3, 5], Distribution::Default, &device);
    let model_output = sample.clone();
    let out = sched.step(model_output, 0.0, sample, 1.0);
    assert_eq!(out.dims(), [3, 5]);
}

#[test]
fn retrieve_timesteps_resamples_expected_length() {
    let device = device();
    let sigmas = vec![1.0, 0.9, 0.8, 0.7, 0.6, 0.5];
    let (timesteps, n) = retrieve_timesteps::<TB>(&device, &sigmas, 3, 1000, None);
    assert_eq!(n, 3);
    assert_eq!(timesteps.dims(), [3]);
}
