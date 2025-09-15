use burn_core as burn;

use burn::tensor::backend::Backend;
use crate::attention::{AttnWindow, StreamingMhaCache, StreamingMqaCache, StreamingMqaParams, StreamingParams};

/// Minimal streaming state for chunked decoding.
#[derive(Clone, Debug)]
pub struct StreamState {
    pub start_pos: usize,
    pub window: AttnWindow,
}

impl StreamState {
    pub fn new(window: AttnWindow) -> Self { Self { start_pos: 0, window } }
    pub fn begin_chunk(&self) -> Self { self.clone() }
    pub fn advance(mut self, by: usize) -> Self { self.start_pos += by; self }
}

impl StreamState {
    pub fn params_mha<B: Backend>(
        &self,
        rope: Option<&burn::nn::rope_encoding::RotaryEncoding<B>>,
    ) -> StreamingParams<B> {
        StreamingParams { rope, start_pos: self.start_pos, window: self.window }
    }

    pub fn params_mqa<B: Backend>(
        &self,
        rope: Option<&burn::nn::rope_encoding::RotaryEncoding<B>>,
        sinks: Option<&burn::tensor::Tensor<B, 2>>,
    ) -> StreamingMqaParams<B> {
        StreamingMqaParams { rope, start_pos: self.start_pos, window: self.window, sinks, attn_bias: None }
    }
}

