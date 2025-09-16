use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor};
use burn_ndarray::NdArray;

use burn_extended::diffusion::{DiffusionPipeline, FlowMatchEuler, FlowMatchEulerConfig};

type TB = NdArray<f32>;

fn device() -> <TB as Backend>::Device {
    Default::default()
}

#[test]
fn pipeline_run_single_progresses() {
    let device = device();
    let mut pipe = DiffusionPipeline::new(FlowMatchEuler::<TB, 2>::new(FlowMatchEulerConfig::default()));
    let sample = Tensor::<TB, 2>::random([2, 4], Distribution::Default, &device);
    // Dummy predict: identity mapping
    let out = pipe.run_single(sample.clone(), 3, |x, _t| x, 1.0);
    assert_eq!(out.dims(), [2, 4]);
}

#[test]
fn pipeline_run_cfg_executes() {
    let device = device();
    let mut pipe = DiffusionPipeline::new(FlowMatchEuler::<TB, 3>::new(FlowMatchEulerConfig::default()));
    let sample = Tensor::<TB, 3>::random([1, 2, 4], Distribution::Default, &device);
    let out = pipe.run_cfg(
        sample,
        2,
        |x, _t| x.clone(), // uncond
        |x, _t| x,         // cond
        1.5,
        1.0,
    );
    assert_eq!(out.dims(), [1, 2, 4]);
}

