use burn_core as burn;

/// Timestep retrieval utility compatible with common diffusion libraries.
pub fn retrieve_timesteps(num_inference_steps: usize, strength: Option<f32>) -> Vec<f32> {
    let mut steps: Vec<f32> = (0..num_inference_steps).map(|i| i as f32 / num_inference_steps as f32).collect();
    if let Some(s) = strength { let keep = ((num_inference_steps as f32) * s).ceil() as usize; steps = steps.into_iter().take(keep.max(1)).collect(); }
    steps
}

