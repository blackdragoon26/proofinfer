//! proofinfer: a dependency-free Llama inference engine.
//!
//! The crate is built on `std` alone. There is no `clap`, no `serde`, no
//! `tokenizers`, no `safetensors`. That is a deliberate constraint rather than
//! a limitation I grew into: for a project about machine-verified inference, the
//! smallest possible dependency surface is the point. Everything below the
//! kernel level is something a reviewer has to take on faith, and every crate
//! in `Cargo.lock` widens that surface by another transitive `unsafe` blob.
//!
//! Module map:
//!
//! * [`ops`]       - numeric kernels (rmsnorm, softmax, silu, matmul, Q8_0, RoPE, GQA)
//! * [`model`]     - config, FP32 checkpoint loading, scratch + KV cache, forward
//! * [`quant`]     - Q8_0 v2 quantized checkpoint loading and inference
//! * [`pool`]      - zero-dependency persistent worker pool for row-partitioned matmul
//! * [`tokenizer`] - the Llama 2 BPE vocabulary reader and encoder

pub mod model;
pub mod ops;
pub mod pool;
pub mod quant;
pub mod tokenizer;
