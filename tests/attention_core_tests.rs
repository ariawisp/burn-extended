use burn::nn::DropoutConfig;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn_ndarray::NdArray;

type TB = NdArray<f32>;

fn device() -> <TB as Backend>::Device {
    Default::default()
}

#[test]
fn compute_scores_scales_by_sqrt_dk() {
    let device = device();
    // Shapes: [B=1, H=1, Tq=1, d_k=1]
    let q = Tensor::<TB, 1>::from_floats([2.0], &device).reshape([1, 1, 1, 1]);
    let k = Tensor::<TB, 1>::from_floats([2.0], &device).reshape([1, 1, 1, 1]);
    let dropout = DropoutConfig::new(0.0).init();

    let scores = burn_extended::attention::compute_scores(q, k, 1, &dropout);
    let v = scores.into_data().to_vec::<f32>().unwrap();
    // 2 * 2 / sqrt(1) = 4
    assert!((v[0] - 4.0).abs() < 1e-5);
}

#[test]
fn apply_bias_and_softmax_behaves() {
    let device = device();
    // attn_scores zeros: shape [1,1,1,2]
    let scores = Tensor::<TB, 1>::from_floats([0.0, 0.0], &device).reshape([1, 1, 1, 2]);
    // bias = [0, ln(9)] so softmax -> [1/10, 9/10]
    let bias = Tensor::<TB, 1>::from_floats([0.0, 9.0_f32.ln()], &device).reshape([1, 1, 1, 2]);
    let w = burn_extended::attention::apply_bias_and_softmax(scores, Some(bias), false);
    let v = w.into_data().to_vec::<f32>().unwrap();
    assert!((v[0] - 0.1).abs() < 1e-4);
    assert!((v[1] - 0.9).abs() < 1e-4);
}

#[test]
fn apply_sinks_then_softmax_appends_and_slices() {
    let device = device();
    // attn_scores zeros: [B=1, H=2, Tq=1, Tk=3]
    let scores = Tensor::<TB, 1>::from_floats([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &device)
        .reshape([1, 2, 1, 3]);
    // sinks per head: [ln(1)=0, ln(9)]
    let sinks = Tensor::<TB, 1>::from_floats([0.0, 9.0_f32.ln()], &device);
    let w = burn_extended::attention::apply_sinks_then_softmax(scores, sinks, 1, 2, 1, 3, false);
    let vals = w.into_data().to_vec::<f32>().unwrap();
    // Head 0: each = 1/4
    assert!((vals[0] - 0.25).abs() < 1e-5);
    assert!((vals[1] - 0.25).abs() < 1e-5);
    assert!((vals[2] - 0.25).abs() < 1e-5);
    // Head 1: each = 1/(3+9) = 1/12
    assert!((vals[3] - (1.0 / 12.0)).abs() < 1e-5);
    assert!((vals[4] - (1.0 / 12.0)).abs() < 1e-5);
    assert!((vals[5] - (1.0 / 12.0)).abs() < 1e-5);
}
