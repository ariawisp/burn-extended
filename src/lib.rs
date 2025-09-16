#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

// Alias burn-core as `burn` so derive macros and internal paths match expectations.
#[allow(unused_imports)]
use burn_core as burn;

pub mod activation;
pub mod attention;
pub mod bias;
pub mod cache;
pub mod decoder;
pub mod diffusion;
pub mod generate;
pub mod image;
pub mod loader;
pub mod models;
pub mod moe;
pub mod rope;
pub mod sampling;
pub mod video;
