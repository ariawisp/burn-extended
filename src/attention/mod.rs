pub mod streaming_mqa;
pub mod mqa;
pub mod streaming_ext;
pub mod cache_ops;

// Re-export common types for convenience
pub use streaming_mqa::*;
pub use mqa::*;
pub use streaming_ext::*;
pub use cache_ops::*;

// Convenience: re-export AttnWindow from burn-core
pub use burn::nn::attention::AttnWindow;
pub use burn::nn::attention::StreamingMhaCache;
