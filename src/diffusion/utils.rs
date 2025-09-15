/// Simple utility to compute linear timesteps and scales.
pub fn linspace(start: f32, end: f32, steps: usize) -> Vec<f32> {
    if steps <= 1 { return vec![start]; }
    let delta = (end - start) / (steps as f32 - 1.0);
    (0..steps).map(|i| start + delta * i as f32).collect()
}

