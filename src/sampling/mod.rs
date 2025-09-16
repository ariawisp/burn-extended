use burn_core as burn;
extern crate alloc;

use burn::tensor::{Int, Tensor, backend::Backend};

/// Basic logits processing configuration for sampling.
#[derive(Clone, Copy, Debug, Default)]
pub struct SamplerConfig {
    pub temperature: f32,             // <= 0.0 means greedy
    pub top_k: Option<usize>,         // keep top-k logits, mask others
    pub repetition_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
}

/// Apply temperature scaling to logits. No-op if `temperature <= 0` or `temperature == 1`.
pub fn apply_temperature<B: Backend>(logits: Tensor<B, 2>, temperature: f32) -> Tensor<B, 2> {
    if temperature > 0.0 && (temperature - 1.0).abs() > core::f32::EPSILON {
        logits / Tensor::from_floats([temperature], &logits.device()).unsqueeze()
    } else {
        logits
    }
}

/// Mask logits outside the top-k per row by setting them to -inf.
/// Note: This is an O(V*logV) CPU fallback using host sorting for portability.
pub fn apply_top_k_cpu<B: Backend>(logits: Tensor<B, 2>, k: usize) -> Tensor<B, 2> {
    if k == 0 { return logits; }
    let device = logits.device();
    let [b, v] = logits.dims();
    let data = logits.into_data().convert::<f32>();
    let mut out: Vec<f32> = vec![f32::NEG_INFINITY; b * v];
    for bi in 0..b {
        let row = &data.value[bi * v..(bi + 1) * v];
        // Indices sorted by logit descending
        let mut idx: Vec<usize> = (0..v).collect();
        idx.sort_unstable_by(|&i, &j| row[j].partial_cmp(&row[i]).unwrap_or(core::cmp::Ordering::Equal));
        for &col in idx.iter().take(k.min(v)) {
            out[bi * v + col] = row[col];
        }
    }
    Tensor::<B, 2>::from_floats(out, &device).reshape([b, v])
}

/// Apply repetition, frequency and presence penalties using token history.
/// This operates on CPU for simplicity and portability.
pub fn apply_penalties_cpu<B: Backend>(
    logits: Tensor<B, 2>,
    tokens_history: Option<&[Vec<usize>]>,
    repetition_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
) -> Tensor<B, 2> {
    if tokens_history.is_none() {
        return logits;
    }
    let [b, v] = logits.dims();
    let device = logits.device();
    let mut out = logits.into_data().convert::<f32>().value;
    let rep = repetition_penalty.unwrap_or(1.0);
    let freq = frequency_penalty.unwrap_or(0.0);
    let pres = presence_penalty.unwrap_or(0.0);

    for bi in 0..b.min(tokens_history.unwrap().len()) {
        let history = &tokens_history.unwrap()[bi];
        if history.is_empty() { continue; }
        // Count occurrences
        let mut counts = vec![0usize; v];
        for &tok in history.iter() {
            if tok < v { counts[tok] += 1; }
        }
        for tok in 0..v {
            let c = counts[tok] as f32;
            if c > 0.0 {
                // Repetition penalty: decrease positive logits more, increase negative less
                if rep != 1.0 {
                    let idx = bi * v + tok;
                    let l = out[idx];
                    out[idx] = if l > 0.0 { l / rep } else { l * rep };
                }
                // Frequency/presence penalties: subtract proportional biases
                let idx = bi * v + tok;
                out[idx] -= freq * c + pres;
            }
        }
    }
    Tensor::<B, 2>::from_floats(out, &device).reshape([b, v])
}

/// Greedy sampling (argmax).
pub fn sample_greedy<B: Backend>(logits: Tensor<B, 2>) -> Tensor<B, 1, Int> {
    // argmax along vocab dim
    let [b, _v] = logits.dims();
    let indices = logits.argmax(1).reshape([b]);
    indices.int()
}

/// Simple CPU multinomial sampling after softmax. Suitable for demos.
pub fn sample_multinomial_cpu<B: Backend>(logits: Tensor<B, 2>) -> Tensor<B, 1, Int> {
    use rand::distr::{weighted::WeightedIndex, Distribution};
    use rand::thread_rng;
    let [b, v] = logits.dims();
    let device = logits.device();
    let mut out = vec![0i64; b];
    let probs = logits.softmax(1).into_data().convert::<f32>().value;
    let mut rng = thread_rng();
    for bi in 0..b {
        let row = &probs[bi * v..(bi + 1) * v];
        let dist = WeightedIndex::new(row.iter().map(|&p| if p.is_finite() && p > 0.0 { p } else { 0.0 })).unwrap();
        let sample = dist.sample(&mut rng) as i64;
        out[bi] = sample;
    }
    Tensor::<B, 1, Int>::from_ints(out, &device)
}

/// Apply typical processors to logits and sample next tokens.
pub fn process_and_sample<B: Backend>(
    logits: Tensor<B, 2>,
    tokens_history: Option<&[Vec<usize>]>,
    cfg: SamplerConfig,
    greedy_when_temp_zero: bool,
) -> Tensor<B, 1, Int> {
    // Build a default pipeline from config
    let mut list = ProcessorList::new();
    if let Some(k) = cfg.top_k { list.push(TopKProcessor { k }); }
    list.push(PenaltyProcessor { repetition_penalty: cfg.repetition_penalty, frequency_penalty: cfg.frequency_penalty, presence_penalty: cfg.presence_penalty });
    list.push(TemperatureProcessor { temperature: cfg.temperature });
    let l = list.apply(logits, tokens_history);
    if cfg.temperature <= 0.0 && greedy_when_temp_zero { sample_greedy(l) } else { sample_multinomial_cpu(l) }
}

/// A composable logits processor.
pub trait LogitsProcessor<B: Backend> {
    fn process(&self, logits: Tensor<B, 2>, history: Option<&[Vec<usize>]>) -> Tensor<B, 2>;
}

/// A simple processor list that applies processors in order.
pub struct ProcessorList<B: Backend> {
    procs: alloc::vec::Vec<alloc::boxed::Box<dyn LogitsProcessor<B>>>,
}

impl<B: Backend> ProcessorList<B> {
    pub fn new() -> Self { Self { procs: alloc::vec::Vec::new() } }
    pub fn push<P: LogitsProcessor<B> + 'static>(&mut self, p: P) { self.procs.push(alloc::boxed::Box::new(p)); }
    pub fn apply(&self, mut logits: Tensor<B, 2>, history: Option<&[Vec<usize>]>) -> Tensor<B, 2> {
        for p in &self.procs { logits = p.process(logits, history); }
        logits
    }
}

pub struct TemperatureProcessor { pub temperature: f32 }
impl<B: Backend> LogitsProcessor<B> for TemperatureProcessor {
    fn process(&self, logits: Tensor<B, 2>, _history: Option<&[Vec<usize>]>) -> Tensor<B, 2> {
        apply_temperature(logits, self.temperature)
    }
}

pub struct TopKProcessor { pub k: usize }
impl<B: Backend> LogitsProcessor<B> for TopKProcessor {
    fn process(&self, logits: Tensor<B, 2>, _history: Option<&[Vec<usize>]>) -> Tensor<B, 2> {
        apply_top_k_cpu(logits, self.k)
    }
}

pub struct PenaltyProcessor {
    pub repetition_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
}
impl<B: Backend> LogitsProcessor<B> for PenaltyProcessor {
    fn process(&self, logits: Tensor<B, 2>, history: Option<&[Vec<usize>]>) -> Tensor<B, 2> {
        apply_penalties_cpu(
            logits,
            history,
            self.repetition_penalty,
            self.frequency_penalty,
            self.presence_penalty,
        )
    }
}
