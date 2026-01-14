//! Rust FFI bindings for CUDA kernels
//!
//! This module provides safe Rust wrappers around the CUDA kernel launches.
//! When CUDA is not available, it falls back to CPU implementations.

use std::ffi::c_void;

/// CUDA error codes
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaError {
    Success = 0,
    InvalidValue = 1,
    MemoryAllocation = 2,
    InitializationError = 3,
    LaunchFailure = 719,
    Unknown = -1,
}

impl CudaError {
    pub fn from_code(code: i32) -> Self {
        match code {
            0 => CudaError::Success,
            1 => CudaError::InvalidValue,
            2 => CudaError::MemoryAllocation,
            3 => CudaError::InitializationError,
            719 => CudaError::LaunchFailure,
            _ => CudaError::Unknown,
        }
    }

    pub fn is_success(self) -> bool {
        self == CudaError::Success
    }
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaError::Success => write!(f, "CUDA Success"),
            CudaError::InvalidValue => write!(f, "CUDA Invalid Value"),
            CudaError::MemoryAllocation => write!(f, "CUDA Memory Allocation Error"),
            CudaError::InitializationError => write!(f, "CUDA Initialization Error"),
            CudaError::LaunchFailure => write!(f, "CUDA Kernel Launch Failure"),
            CudaError::Unknown => write!(f, "CUDA Unknown Error"),
        }
    }
}

impl std::error::Error for CudaError {}

/// Result type for CUDA operations
pub type CudaResult<T> = Result<T, CudaError>;

/// CUDA stream handle (opaque pointer)
pub type CudaStream = *mut c_void;

/// Null CUDA stream (default stream)
pub const CUDA_STREAM_DEFAULT: CudaStream = std::ptr::null_mut();

// FFI declarations for CUDA kernels
#[cfg(feature = "cuda")]
extern "C" {
    /// Get last CUDA error
    fn cudaGetLastError() -> i32;

    /// Launch fused predict-subtract-quantize kernel
    pub fn launch_predict_subtract_quantize(
        gradient: *const f32,
        g_current: *const f32,
        g_previous: *const f32,
        momentum: f32,
        gamma: f32,
        output: *mut u8,
        n: i32,
        stream: CudaStream,
    );

    /// Launch fused dequantize-add-reconstruct kernel
    pub fn launch_dequantize_add_reconstruct(
        packed: *const u8,
        g_current: *const f32,
        g_previous: *const f32,
        momentum: f32,
        gamma: f32,
        output: *mut f32,
        n: i32,
        stream: CudaStream,
    );

    /// Launch gamma computation kernel
    pub fn launch_compute_gamma(
        residuals: *const f32,
        gamma_out: *mut f32,
        workspace: *mut f32,
        n: i32,
        stream: CudaStream,
    );
}

/// Check for CUDA errors after kernel launch
#[cfg(feature = "cuda")]
fn check_cuda_error() -> CudaResult<()> {
    let code = unsafe { cudaGetLastError() };
    let error = CudaError::from_code(code);
    if error.is_success() {
        Ok(())
    } else {
        Err(error)
    }
}

/// Safe wrapper for predict-subtract-quantize
///
/// # Safety
/// All pointers must be valid CUDA device pointers with sufficient allocation.
#[cfg(feature = "cuda")]
pub unsafe fn predict_and_quantize_cuda(
    gradient: *const f32,
    g_current: *const f32,
    g_previous: *const f32,
    momentum: f32,
    gamma: f32,
    output: *mut u8,
    n: usize,
    stream: CudaStream,
) -> CudaResult<()> {
    launch_predict_subtract_quantize(
        gradient,
        g_current,
        g_previous,
        momentum,
        gamma.max(1e-8), // Guard against zero
        output,
        n as i32,
        stream,
    );
    check_cuda_error()
}

/// Safe wrapper for dequantize-add-reconstruct
///
/// # Safety
/// All pointers must be valid CUDA device pointers with sufficient allocation.
#[cfg(feature = "cuda")]
pub unsafe fn dequantize_and_reconstruct_cuda(
    packed: *const u8,
    g_current: *const f32,
    g_previous: *const f32,
    momentum: f32,
    gamma: f32,
    output: *mut f32,
    n: usize,
    stream: CudaStream,
) -> CudaResult<()> {
    launch_dequantize_add_reconstruct(
        packed,
        g_current,
        g_previous,
        momentum,
        gamma.max(1e-8), // Guard against zero
        output,
        n as i32,
        stream,
    );
    check_cuda_error()
}

/// Safe wrapper for gamma computation
///
/// # Safety
/// All pointers must be valid CUDA device pointers.
/// workspace must have size >= ceil(n / 256) * sizeof(f32)
#[cfg(feature = "cuda")]
pub unsafe fn compute_gamma_cuda(
    residuals: *const f32,
    gamma_out: *mut f32,
    workspace: *mut f32,
    n: usize,
    stream: CudaStream,
) -> CudaResult<()> {
    launch_compute_gamma(residuals, gamma_out, workspace, n as i32, stream);
    check_cuda_error()
}

// CPU fallback stubs when CUDA is not available
#[cfg(not(feature = "cuda"))]
pub unsafe fn predict_and_quantize_cuda(
    _gradient: *const f32,
    _g_current: *const f32,
    _g_previous: *const f32,
    _momentum: f32,
    _gamma: f32,
    _output: *mut u8,
    _n: usize,
    _stream: CudaStream,
) -> CudaResult<()> {
    panic!("CUDA not available. Use CPU fallback in kernels/mod.rs");
}

#[cfg(not(feature = "cuda"))]
pub unsafe fn dequantize_and_reconstruct_cuda(
    _packed: *const u8,
    _g_current: *const f32,
    _g_previous: *const f32,
    _momentum: f32,
    _gamma: f32,
    _output: *mut f32,
    _n: usize,
    _stream: CudaStream,
) -> CudaResult<()> {
    panic!("CUDA not available. Use CPU fallback in kernels/mod.rs");
}

#[cfg(not(feature = "cuda"))]
pub unsafe fn compute_gamma_cuda(
    _residuals: *const f32,
    _gamma_out: *mut f32,
    _workspace: *mut f32,
    _n: usize,
    _stream: CudaStream,
) -> CudaResult<()> {
    panic!("CUDA not available. Use CPU fallback in kernels/mod.rs");
}
