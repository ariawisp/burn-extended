//! Initialize Burn WGPU backend (with Metal on macOS), and create a device.
//! Demonstrates how to set up the backend for use with burn-extended modules.

use burn_wgpu::{Wgpu as B, WgpuDevice};

fn main() {
    // Choose a device; default picks the first adapter.
    let device = WgpuDevice::default();

    // Optionally select the graphics API (Metal recommended on macOS).
    // This is a one-time runtime init for the given device.
    burn_wgpu::init_setup::<burn_wgpu::graphics::Metal>(&device, Default::default());

    // Now B::Device is ready to be used across burn-extended modules.
    // Example: println!("Device ready: {:?}", device);
    println!("WGPU backend initialized (Metal) for device: {:?}", device);
}

