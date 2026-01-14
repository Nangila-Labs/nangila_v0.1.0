//! Nangila CUDA Kernels
//!
//! This crate provides GPU-accelerated implementations of:
//! - Fused predict-subtract-quantize kernel
//! - Fused dequantize-add-reconstruct kernel
//! - Dynamic gamma computation
//!
//! When CUDA is not available, CPU fallbacks are used automatically.

pub mod bindings;
pub mod kernels;

pub use bindings::{
    compute_gamma_cuda, dequantize_and_reconstruct_cuda, predict_and_quantize_cuda, CudaStream,
    CUDA_STREAM_DEFAULT,
};
pub use kernels::{dequantize_and_add, predict_and_quantize};

/// Check if CUDA support is available
pub fn cuda_available() -> bool {
    cfg!(feature = "cuda")
}
