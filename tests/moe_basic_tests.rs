use burn::backend::wgpu::{Wgpu as TB, WgpuDevice};
use burn::tensor::{Distribution, Tensor};
use burn_extended::moe::{MoeConfig, MoeGatedSwiGLU};

#[test]
fn moe_forward_shapes_and_residual() {
    let device = WgpuDevice::default();
    burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Metal>(&device, Default::default());

    let d_model = 16usize;
    let ffn = 32usize;
    let e = 4usize;
    // Route to all experts equally by setting k=e (experts_per_token)
    let cfg = MoeConfig {
        d_model,
        ffn_hidden: ffn,
        num_experts: e,
        experts_per_token: e,
        swiglu_alpha: 1.0,
        swiglu_limit: 7.0,
        initializer: burn::nn::Initializer::Zeros,
        disabled: false,
        verbose: false,
    };
    let moe = cfg.init::<TB>(&device);

    let x = Tensor::<TB, 3>::random([2, 3, d_model], Distribution::Default, &device);
    let y = moe.forward(x.clone());
    assert_eq!(y.dims(), [2, 3, d_model]);

    // With all-zero weights and biases, output should be residual (y == x)
    let diff = (y - x).abs().sum().into_scalar();
    assert!(diff < 1e-5, "MoE residual expected with zero-initialized params");
}
