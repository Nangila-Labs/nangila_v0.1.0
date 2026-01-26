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
pub mod state;

#[cfg(test)]
mod bindings_test;

pub use bindings::{
    dequantize_and_reconstruct_cuda, predict_and_quantize_cuda, CudaStream, SyncMode,
    CUDA_STREAM_DEFAULT,
};

#[cfg(feature = "cuda")]
pub use bindings::{
    compute_crc32_cuda, compute_gamma_cuda, copy_device_to_host_async, synchronize_stream,
};
pub use kernels::{dequantize_and_add, predict_and_quantize};

#[cfg(feature = "cuda")]
pub use state::{GpuBuffer, GpuLayerState, GpuStateManager};

/// Check if CUDA support is available
pub fn cuda_available() -> bool {
    cfg!(feature = "cuda")
}
