use burn_core as burn;

use burn::config::Config;
use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Initializer, Linear, LinearConfig};
use burn::nn::rms_norm::{RmsNorm, RmsNormConfig};
use burn::tensor::{Bool, Tensor, backend::Backend};

use crate::activation::swiglu_clamp;
use crate::attention::{MultiQueryAttention, MultiQueryAttentionConfig, MqaInput};

/// Configuration for a baseline decoder block using multi-query attention and SwiGLU FFN.
#[derive(Config, Debug)]
pub struct DecoderBlockConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub num_key_value_heads: usize,
    pub ffn_hidden: usize,
    #[config(default = 0.0)]
    pub dropout: f64,
    #[config(default = 1.0)]
    pub swiglu_alpha: f32,
    #[config(default = 7.0)]
    pub swiglu_limit: f32,
    #[config(default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}")]
    pub initializer: Initializer,
}

#[derive(Module, Debug)]
pub struct DecoderBlock<B: Backend> {
    attn: MultiQueryAttention<B>,
    norm_attn: RmsNorm<B>,
    norm_ffn: RmsNorm<B>,
    ffn_up: Linear<B>,
    ffn_down: Linear<B>,
    dropout: Dropout,
    swiglu_alpha: f32,
    swiglu_limit: f32,
}

impl DecoderBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DecoderBlock<B> {
        assert!(self.d_model % self.n_heads == 0, "d_model must divide n_heads");
        assert!(self.n_heads % self.num_key_value_heads == 0, "n_heads must divide kv_heads");

        let attn = MultiQueryAttentionConfig::new(
            self.d_model,
            self.n_heads,
            self.num_key_value_heads,
        )
        .with_dropout(self.dropout)
        .with_initializer(self.initializer.clone())
        .init::<B>(device);

        let norm_attn = RmsNormConfig::new(self.d_model).init::<B>(device);
        let norm_ffn = RmsNormConfig::new(self.d_model).init::<B>(device);

        let ffn_up = LinearConfig::new(self.d_model, self.ffn_hidden * 2)
            .with_initializer(self.initializer.clone())
            .init(device);
        let ffn_down = LinearConfig::new(self.ffn_hidden, self.d_model)
            .with_initializer(self.initializer.clone())
            .init(device);

        DecoderBlock {
            attn,
            norm_attn,
            norm_ffn,
            ffn_up,
            ffn_down,
            dropout: DropoutConfig::new(self.dropout).init(),
            swiglu_alpha: self.swiglu_alpha,
            swiglu_limit: self.swiglu_limit,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DecoderBlockInput<B: Backend> {
    pub hidden: Tensor<B, 3>,
    pub mask_pad: Option<Tensor<B, 2, Bool>>,
    pub mask_attn: Option<Tensor<B, 3, Bool>>,
    pub attn_bias: Option<Tensor<B, 4>>,
}

impl<B: Backend> DecoderBlockInput<B> {
    pub fn new(hidden: Tensor<B, 3>) -> Self {
        Self {
            hidden,
            mask_pad: None,
            mask_attn: None,
            attn_bias: None,
        }
    }

    pub fn mask_pad(mut self, mask: Tensor<B, 2, Bool>) -> Self {
        self.mask_pad = Some(mask);
        self
    }

    pub fn mask_attn(mut self, mask: Tensor<B, 3, Bool>) -> Self {
        self.mask_attn = Some(mask);
        self
    }

    pub fn attn_bias(mut self, bias: Tensor<B, 4>) -> Self {
        self.attn_bias = Some(bias);
        self
    }
}

#[derive(Debug, Clone)]
pub struct DecoderBlockOutput<B: Backend> {
    pub hidden: Tensor<B, 3>,
}

impl<B: Backend> DecoderBlock<B> {
    pub fn forward(&self, input: DecoderBlockInput<B>) -> DecoderBlockOutput<B> {
        let DecoderBlockInput {
            hidden,
            mask_pad,
            mask_attn,
            attn_bias,
        } = input;

        let norm_attn = self.norm_attn.forward(hidden.clone());
        let mut attn_input = MqaInput::self_attn(norm_attn);
        if let Some(mask) = mask_pad {
            attn_input = attn_input.mask_pad(mask);
        }
        if let Some(mask) = mask_attn {
            attn_input = attn_input.mask_attn(mask);
        }
        if let Some(bias) = attn_bias {
            attn_input = attn_input.attn_bias(bias);
        }

        let attn_out = self.attn.forward(attn_input);
        let attn_context = self.dropout.forward(attn_out.context);
        let residual_attn = hidden + attn_context;

        let norm_ffn = self.norm_ffn.forward(residual_attn.clone());
        let up = self.ffn_up.forward(norm_ffn);
        let activated = swiglu_clamp(up, self.swiglu_alpha, Some(self.swiglu_limit));
        let down = self.ffn_down.forward(activated);
        let hidden = residual_attn + self.dropout.forward(down);

        DecoderBlockOutput { hidden }
    }
}
