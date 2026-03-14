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

/// Synchronization mode for CUDA kernel error checking
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncMode {
    /// Async: No synchronization, maximum performance (production)
    Async = 0,
    /// Always: Always synchronize, catch all errors immediately (debug)
    Always = 1,
    /// Periodic: Synchronize every 100 calls, balanced approach (default)
    Periodic = 2,
}

impl Default for SyncMode {
    fn default() -> Self {
        SyncMode::Periodic
    }
}

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
        sync_mode: i32,
        step: u64,
        layer_id: u32,
    ) -> i32;

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
        sync_mode: i32,
    ) -> i32;

    /// Launch gamma computation kernel
    pub fn launch_compute_gamma(
        residuals: *const f32,
        gamma_out: *mut f32,
        workspace: *mut f32,
        n: i32,
        stream: CudaStream,
    );

    /// Launch CRC32 kernel
    pub fn launch_compute_crc32(data: *const u8, len: usize, out_crc: *mut u32, stream: CudaStream);

    /// Async copy from device to host (or device to device)
    pub fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: i32,
        stream: CudaStream,
    ) -> i32;
}

#[allow(non_upper_case_globals)]
pub const cudaMemcpyHostToHost: i32 = 0;
#[allow(non_upper_case_globals)]
pub const cudaMemcpyHostToDevice: i32 = 1;
#[allow(non_upper_case_globals)]
pub const cudaMemcpyDeviceToHost: i32 = 2;
#[allow(non_upper_case_globals)]
pub const cudaMemcpyDeviceToDevice: i32 = 3;
#[allow(non_upper_case_globals)]
pub const cudaMemcpyDefault: i32 = 4;

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
///
/// # Arguments
/// * `gradient` - Current gradient to compress (device pointer)
/// * `g_current` - Gradient from step t for prediction (device pointer)
/// * `g_previous` - Gradient from step t-1 for prediction (device pointer)
/// * `momentum` - Momentum coefficient (typically 0.9)
/// * `gamma` - Quantization scale factor
/// * `output` - Output buffer for packed INT4 data (device pointer)
/// * `n` - Number of elements
/// * `stream` - CUDA stream for async execution
/// * `sync_mode` - Synchronization mode for error checking
///
/// # Returns
/// `Ok(())` on success, `Err(CudaError)` with detailed error information on failure
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
    sync_mode: SyncMode,
    step: u64,
    layer_id: u32,
) -> CudaResult<()> {
    // Validate inputs before calling kernel
    if gradient.is_null() || g_current.is_null() || g_previous.is_null() || output.is_null() {
        return Err(CudaError::InvalidValue);
    }

    if n == 0 || n > 1_000_000_000 {
        return Err(CudaError::InvalidValue);
    }

    // Guard against invalid gamma
    let safe_gamma = gamma.max(1e-8).min(1e6);
    if !safe_gamma.is_finite() {
        return Err(CudaError::InvalidValue);
    }

    // Validate momentum
    if momentum < 0.0 || momentum > 1.0 || !momentum.is_finite() {
        return Err(CudaError::InvalidValue);
    }

    let result = launch_predict_subtract_quantize(
        gradient,
        g_current,
        g_previous,
        momentum,
        safe_gamma,
        output,
        n as i32,
        stream,
        sync_mode as i32,
        step,
        layer_id,
    );

    let error = CudaError::from_code(result);
    if error.is_success() {
        Ok(())
    } else {
        Err(error)
    }
}

/// Safe wrapper for dequantize-add-reconstruct
///
/// # Safety
/// All pointers must be valid CUDA device pointers with sufficient allocation.
///
/// # Arguments
/// * `packed` - Packed INT4 compressed data (device pointer)
/// * `g_current` - Gradient from step t for prediction (device pointer)
/// * `g_previous` - Gradient from step t-1 for prediction (device pointer)
/// * `momentum` - Momentum coefficient (typically 0.9)
/// * `gamma` - Quantization scale factor
/// * `output` - Output buffer for reconstructed gradient (device pointer)
/// * `n` - Number of elements
/// * `stream` - CUDA stream for async execution
/// * `sync_mode` - Synchronization mode for error checking
///
/// # Returns
/// `Ok(())` on success, `Err(CudaError)` with detailed error information on failure
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
    sync_mode: SyncMode,
) -> CudaResult<()> {
    // Validate inputs
    if packed.is_null() || g_current.is_null() || g_previous.is_null() || output.is_null() {
        return Err(CudaError::InvalidValue);
    }

    if n == 0 || n > 1_000_000_000 {
        return Err(CudaError::InvalidValue);
    }

    // Guard against invalid gamma
    let safe_gamma = gamma.max(1e-8).min(1e6);
    if !safe_gamma.is_finite() {
        return Err(CudaError::InvalidValue);
    }

    // Validate momentum
    if momentum < 0.0 || momentum > 1.0 || !momentum.is_finite() {
        return Err(CudaError::InvalidValue);
    }

    let result = launch_dequantize_add_reconstruct(
        packed,
        g_current,
        g_previous,
        momentum,
        safe_gamma,
        output,
        n as i32,
        stream,
        sync_mode as i32,
    );

    let error = CudaError::from_code(result);
    if error.is_success() {
        Ok(())
    } else {
        Err(error)
    }
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

/// Safe wrapper for CRC32 computation
///
/// # Safety
/// All pointers must be valid CUDA device pointers.
/// out_crc must be a valid device pointer to a single u32.
#[cfg(feature = "cuda")]
pub unsafe fn compute_crc32_cuda(
    data: *const u8,
    len: usize,
    out_crc: *mut u32,
    stream: CudaStream,
) -> CudaResult<()> {
    launch_compute_crc32(data, len, out_crc, stream);
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
    _sync_mode: SyncMode,
    _step: u64,
    _layer_id: u32,
) -> CudaResult<()> {
    Err(CudaError::InitializationError)
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
    _sync_mode: SyncMode,
) -> CudaResult<()> {
    Err(CudaError::InitializationError)
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

#[cfg(feature = "cuda")]
pub unsafe fn copy_device_to_host_async(
    src: *const u8,
    dst: *mut u8,
    count: usize,
    stream: CudaStream,
) -> CudaResult<()> {
    let result = cudaMemcpyAsync(
        dst as *mut c_void,
        src as *const c_void,
        count,
        cudaMemcpyDeviceToHost,
        stream,
    );
    // cudaMemcpyAsync returns cudaError_t directly
    let error = CudaError::from_code(result);
    if error.is_success() {
        Ok(())
    } else {
        Err(error)
    }
}

#[cfg(feature = "cuda")]
pub unsafe fn synchronize_stream(stream: CudaStream) -> CudaResult<()> {
    extern "C" {
        fn cudaStreamSynchronize(stream: CudaStream) -> i32;
    }
    let code = cudaStreamSynchronize(stream);
    let error = CudaError::from_code(code);
    if error.is_success() {
        Ok(())
    } else {
        Err(error)
    }
}

// Fallbacks
#[cfg(not(feature = "cuda"))]
pub unsafe fn copy_device_to_host_async(
    _src: *const u8,
    _dst: *mut u8,
    _count: usize,
    _stream: CudaStream,
) -> CudaResult<()> {
    panic!("CUDA not available.");
}

#[cfg(not(feature = "cuda"))]
pub unsafe fn synchronize_stream(_stream: CudaStream) -> CudaResult<()> {
    panic!("CUDA not available.");
}
