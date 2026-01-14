//! C FFI exports for NCCL intercept integration
//!
//! These functions are exported with `extern "C"` to be callable from
//! the C++ NCCL intercept shim (libnangila_intercept.so).
//!
//! # Safety
//! All functions in this module use raw pointers and require careful
//! handling of memory ownership and lifetime guarantees.

use crate::NangilaHook;
use nangila_core::{
    bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16,
    DataType, NangilaConfig, Tensor, TopologyMask,
};
use std::ffi::{c_char, CStr};
use std::fs::File;
use std::io::BufReader;
use std::ptr;
use std::slice;

/// Error codes for FFI functions
#[repr(i32)]
pub enum NangilaResult {
    Success = 0,
    InvalidPointer = 1,
    InvalidPath = 2,
    IoError = 3,
    CompressionError = 4,
    DecompressionError = 5,
}

/// Opaque handle to NangilaHook for C code
pub type NangilaHandle = *mut NangilaHook;

/// Initialize Nangila with a mask file
///
/// # Arguments
/// * `mask_path` - Path to .nzmask file (null-terminated C string)
///
/// # Returns
/// Handle to NangilaHook, or null on failure
///
/// # Safety
/// The returned handle must be freed with `nangila_free()`
#[no_mangle]
pub unsafe extern "C" fn nangila_init(mask_path: *const c_char) -> NangilaHandle {
    if mask_path.is_null() {
        return ptr::null_mut();
    }

    let path = match CStr::from_ptr(mask_path).to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            tracing::error!("Failed to open mask file {}: {}", path, e);
            return ptr::null_mut();
        }
    };

    let mut reader = BufReader::new(file);
    let mask = match TopologyMask::load(&mut reader) {
        Ok(m) => m,
        Err(e) => {
            tracing::error!("Failed to load mask: {}", e);
            return ptr::null_mut();
        }
    };

    let hook = NangilaHook::new(NangilaConfig::default(), mask);
    Box::into_raw(Box::new(hook))
}

/// Initialize Nangila with all-driver topology (no Passengers)
///
/// # Arguments
/// * `num_layers` - Number of layers to track
///
/// # Returns
/// Handle to NangilaHook
#[no_mangle]
pub extern "C" fn nangila_init_all_drivers(num_layers: u32) -> NangilaHandle {
    let hook = NangilaHook::all_drivers(num_layers as usize);
    Box::into_raw(Box::new(hook))
}

/// Compress gradient data before sending
///
/// # Arguments
/// * `handle` - Nangila handle from `nangila_init()`
/// * `sendbuff` - Input gradient buffer (FP32)
/// * `count` - Number of elements
/// * `layer_id` - Layer identifier
/// * `out_compressed` - Output buffer for compressed data
/// * `out_compressed_size` - Maximum size of output buffer / actual size written
///
/// # Returns
/// 0 on success, error code on failure
#[no_mangle]
pub unsafe extern "C" fn nangila_compress(
    handle: NangilaHandle,
    sendbuff: *const f32,
    count: usize,
    layer_id: u32,
    out_compressed: *mut u8,
    out_compressed_size: *mut usize,
) -> i32 {
    if handle.is_null() || sendbuff.is_null() || out_compressed.is_null() || out_compressed_size.is_null() {
        return NangilaResult::InvalidPointer as i32;
    }

    let hook = &mut *handle;
    let max_size = *out_compressed_size;

    // Read input gradient
    let data = slice::from_raw_parts(sendbuff, count).to_vec();
    let gradient = Tensor::new(data, vec![count]);

    // Compress via hook
    let packet_bytes = hook.on_send(layer_id, gradient);

    if packet_bytes.len() > max_size {
        tracing::error!(
            "Compressed size {} exceeds buffer size {}",
            packet_bytes.len(),
            max_size
        );
        return NangilaResult::CompressionError as i32;
    }

    // Copy to output buffer
    let out_slice = slice::from_raw_parts_mut(out_compressed, packet_bytes.len());
    out_slice.copy_from_slice(&packet_bytes);
    *out_compressed_size = packet_bytes.len();

    NangilaResult::Success as i32
}

/// Decompress received gradient data
///
/// # Arguments
/// * `handle` - Nangila handle
/// * `compressed` - Compressed data received from network
/// * `compressed_size` - Size of compressed data
/// * `layer_id` - Layer identifier
/// * `recvbuff` - Output buffer for decompressed gradient (FP32)
/// * `count` - Expected number of elements
///
/// # Returns
/// 0 on success, error code on failure
#[no_mangle]
pub unsafe extern "C" fn nangila_decompress(
    handle: NangilaHandle,
    compressed: *const u8,
    compressed_size: usize,
    layer_id: u32,
    recvbuff: *mut f32,
    count: usize,
) -> i32 {
    if handle.is_null() || compressed.is_null() || recvbuff.is_null() {
        return NangilaResult::InvalidPointer as i32;
    }

    let hook = &mut *handle;

    // Read compressed packet
    let packet_bytes = slice::from_raw_parts(compressed, compressed_size);

    // Decompress via hook
    let gradient = hook.on_receive(layer_id, packet_bytes);

    // Copy to output buffer
    let out_slice = slice::from_raw_parts_mut(recvbuff, count);
    let copy_len = gradient.data.len().min(count);
    out_slice[..copy_len].copy_from_slice(&gradient.data[..copy_len]);

    // Zero-fill if gradient is smaller than expected
    if copy_len < count {
        for i in copy_len..count {
            out_slice[i] = 0.0;
        }
    }

    NangilaResult::Success as i32
}

/// Update state after All-Reduce completes
///
/// # Arguments
/// * `handle` - Nangila handle
/// * `layer_id` - Layer identifier
/// * `gradient` - Final gradient after All-Reduce
/// * `count` - Number of elements
#[no_mangle]
pub unsafe extern "C" fn nangila_on_complete(
    handle: NangilaHandle,
    layer_id: u32,
    gradient: *const f32,
    count: usize,
) -> i32 {
    if handle.is_null() || gradient.is_null() {
        return NangilaResult::InvalidPointer as i32;
    }

    let hook = &mut *handle;
    let data = slice::from_raw_parts(gradient, count).to_vec();
    let tensor = Tensor::new(data, vec![count]);

    hook.on_complete(layer_id, tensor);

    NangilaResult::Success as i32
}

/// Advance to the next training step
#[no_mangle]
pub unsafe extern "C" fn nangila_step(handle: NangilaHandle) -> i32 {
    if handle.is_null() {
        return NangilaResult::InvalidPointer as i32;
    }

    let hook = &mut *handle;
    hook.step();

    NangilaResult::Success as i32
}

/// Check if compression is enabled
#[no_mangle]
pub unsafe extern "C" fn nangila_is_enabled(handle: NangilaHandle) -> i32 {
    if handle.is_null() {
        return 0;
    }

    let hook = &*handle;
    if hook.is_compression_enabled() { 1 } else { 0 }
}

/// Get current step number
#[no_mangle]
pub unsafe extern "C" fn nangila_current_step(handle: NangilaHandle) -> u64 {
    if handle.is_null() {
        return 0;
    }

    let hook = &*handle;
    hook.current_step() as u64
}

/// Get predictor state hash for verification
#[no_mangle]
pub unsafe extern "C" fn nangila_predictor_hash(handle: NangilaHandle) -> u64 {
    if handle.is_null() {
        return 0;
    }

    let hook = &*handle;
    hook.predictor_hash()
}

/// Free Nangila handle
///
/// # Safety
/// Must be called exactly once for each successful `nangila_init()` call
#[no_mangle]
pub unsafe extern "C" fn nangila_free(handle: NangilaHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

/// Get compressed buffer size estimate
///
/// Returns maximum size needed for compressed output buffer
#[no_mangle]
pub extern "C" fn nangila_max_compressed_size(count: usize) -> usize {
    // Packet header (16 bytes) + worst case (no compression, FP32 + overhead)
    16 + 12 + count * 4 + 32
}

// =============================================================================
// FFI Extended Functions with Data Type Support (FP16/BF16)
// =============================================================================

/// Compress gradient data with explicit dtype
///
/// # Arguments
/// * `handle` - Nangila handle from `nangila_init()`
/// * `sendbuff` - Input gradient buffer (raw bytes, any dtype)
/// * `count` - Number of elements
/// * `dtype` - Data type: 0=FP32, 1=FP16, 2=BF16
/// * `layer_id` - Layer identifier
/// * `out_compressed` - Output buffer for compressed data
/// * `out_compressed_size` - Maximum size of output buffer / actual size written
///
/// # Returns
/// 0 on success, error code on failure
#[no_mangle]
pub unsafe extern "C" fn nangila_compress_ex(
    handle: NangilaHandle,
    sendbuff: *const u8,
    count: usize,
    dtype: i32,
    layer_id: u32,
    out_compressed: *mut u8,
    out_compressed_size: *mut usize,
) -> i32 {
    if handle.is_null() || sendbuff.is_null() || out_compressed.is_null() || out_compressed_size.is_null() {
        return NangilaResult::InvalidPointer as i32;
    }

    let dtype_enum = match DataType::from_i32(dtype) {
        Some(d) => d,
        None => return NangilaResult::CompressionError as i32,
    };

    let hook = &mut *handle;
    let max_size = *out_compressed_size;

    // Convert input bytes to FP32
    let data: Vec<f32> = match dtype_enum {
        DataType::Float32 => {
            let ptr = sendbuff as *const f32;
            slice::from_raw_parts(ptr, count).to_vec()
        }
        DataType::Float16 => {
            let bytes = slice::from_raw_parts(sendbuff, count * 2);
            f16_to_f32(bytes)
        }
        DataType::BFloat16 => {
            let bytes = slice::from_raw_parts(sendbuff, count * 2);
            bf16_to_f32(bytes)
        }
    };

    let gradient = Tensor::new_with_dtype(data, vec![count], dtype_enum);

    // Compress via hook
    let packet_bytes = hook.on_send(layer_id, gradient);

    if packet_bytes.len() > max_size {
        tracing::error!(
            "Compressed size {} exceeds buffer size {}",
            packet_bytes.len(),
            max_size
        );
        return NangilaResult::CompressionError as i32;
    }

    // Copy to output buffer
    let out_slice = slice::from_raw_parts_mut(out_compressed, packet_bytes.len());
    out_slice.copy_from_slice(&packet_bytes);
    *out_compressed_size = packet_bytes.len();

    NangilaResult::Success as i32
}

/// Decompress received gradient data with explicit dtype
///
/// # Arguments
/// * `handle` - Nangila handle
/// * `compressed` - Compressed data received from network
/// * `compressed_size` - Size of compressed data
/// * `layer_id` - Layer identifier
/// * `recvbuff` - Output buffer for decompressed gradient (in specified dtype)
/// * `count` - Expected number of elements
/// * `dtype` - Data type: 0=FP32, 1=FP16, 2=BF16
///
/// # Returns
/// 0 on success, error code on failure
#[no_mangle]
pub unsafe extern "C" fn nangila_decompress_ex(
    handle: NangilaHandle,
    compressed: *const u8,
    compressed_size: usize,
    layer_id: u32,
    recvbuff: *mut u8,
    count: usize,
    dtype: i32,
) -> i32 {
    if handle.is_null() || compressed.is_null() || recvbuff.is_null() {
        return NangilaResult::InvalidPointer as i32;
    }

    let dtype_enum = match DataType::from_i32(dtype) {
        Some(d) => d,
        None => return NangilaResult::DecompressionError as i32,
    };

    let hook = &mut *handle;

    // Read compressed packet
    let packet_bytes = slice::from_raw_parts(compressed, compressed_size);

    // Decompress via hook (returns FP32)
    let gradient = hook.on_receive(layer_id, packet_bytes);

    // Convert FP32 output to requested dtype
    match dtype_enum {
        DataType::Float32 => {
            let out_slice = slice::from_raw_parts_mut(recvbuff as *mut f32, count);
            let copy_len = gradient.data.len().min(count);
            out_slice[..copy_len].copy_from_slice(&gradient.data[..copy_len]);
            for i in copy_len..count {
                out_slice[i] = 0.0;
            }
        }
        DataType::Float16 => {
            let f16_bytes = f32_to_f16(&gradient.data);
            let out_slice = slice::from_raw_parts_mut(recvbuff, count * 2);
            let copy_len = f16_bytes.len().min(count * 2);
            out_slice[..copy_len].copy_from_slice(&f16_bytes[..copy_len]);
        }
        DataType::BFloat16 => {
            let bf16_bytes = f32_to_bf16(&gradient.data);
            let out_slice = slice::from_raw_parts_mut(recvbuff, count * 2);
            let copy_len = bf16_bytes.len().min(count * 2);
            out_slice[..copy_len].copy_from_slice(&bf16_bytes[..copy_len]);
        }
    }

    NangilaResult::Success as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_init_free() {
        unsafe {
            let handle = nangila_init_all_drivers(10);
            assert!(!handle.is_null());
            nangila_free(handle);
        }
    }

    #[test]
    fn test_ffi_compress_decompress() {
        unsafe {
            let handle = nangila_init_all_drivers(1);

            let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
            let mut compressed = vec![0u8; nangila_max_compressed_size(4)];
            let mut compressed_size = compressed.len();

            let result = nangila_compress(
                handle,
                input.as_ptr(),
                input.len(),
                0,
                compressed.as_mut_ptr(),
                &mut compressed_size,
            );
            assert_eq!(result, 0);
            assert!(compressed_size > 0);

            let mut output = vec![0.0f32; 4];
            let result = nangila_decompress(
                handle,
                compressed.as_ptr(),
                compressed_size,
                0,
                output.as_mut_ptr(),
                output.len(),
            );
            assert_eq!(result, 0);

            // During warmup, should be exact match
            assert_eq!(input, output);

            nangila_free(handle);
        }
    }
}
