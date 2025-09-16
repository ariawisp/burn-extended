use burn_ndarray::NdArray as B;
use burn_tensor::backend::Backend;
use burn_tensor::Tensor;

use burn_extended::diffusion::{
    retrieve_timesteps, DiffusionScheduler, FlowMatchEuler, FlowMatchEulerConfig, FlowMatchHeun,
    FlowMatchHeunConfig, FlowMatchPingPong, FlowMatchPingPongConfig,
};

fn main() {
    let device = <B as Backend>::Device::default();

    // Build schedulers
    let mut euler = FlowMatchEuler::<B, 3>::new(FlowMatchEulerConfig::default());
    let mut heun = FlowMatchHeun::<B, 3>::new(FlowMatchHeunConfig::default());
    let mut pingpong = FlowMatchPingPong::<B, 3>::new(FlowMatchPingPongConfig::default());

    // Set timesteps
    euler.set_timesteps(8);
    heun.set_timesteps(8);
    pingpong.set_timesteps(8);

    // Sample input
    let sample = Tensor::<B, 3>::zeros([1, 2, 4], &device);
    let model_out = Tensor::<B, 3>::zeros([1, 2, 4], &device);

    // One step from each scheduler
    let out_e = euler.step(model_out.clone(), 0.0, sample.clone(), 1.0);
    let out_h = heun.step(model_out.clone(), 0.0, sample.clone(), 1.0);
    let out_p = pingpong.step(model_out, 0.0, sample, 1.0);

    println!("Euler out dims: {:?}", out_e.dims());
    println!("Heun out dims:  {:?}", out_h.dims());
    println!("PingPong out dims: {:?}", out_p.dims());

    // Demonstrate retrieve_timesteps utility
    let sigmas = vec![1.0, 0.8, 0.6, 0.4, 0.2, 0.0];
    let (timesteps, n) = retrieve_timesteps::<B>(&device, &sigmas, 4, 1000, None);
    println!("timesteps len: {} dims: {:?}", n, timesteps.dims());
}
