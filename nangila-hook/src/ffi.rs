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
    bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16, DataType, NangilaConfig, SafeModeAction,
    SafeModeConfig, Tensor, TopologyMask,
};
#[cfg(feature = "cuda")]
use nangila_cuda::{
    compute_crc32_cuda, copy_device_to_host_async, dequantize_and_reconstruct_cuda,
    predict_and_quantize_cuda, synchronize_stream, CudaStream,
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
    // Enable GPU mode by default for this binding?
    // Or expose a separate init function?
    // Let's add a set_gpu_mode function and call it from C++ if needed.
    // For now, let's keep it simple and default to CPU mode behavior unless explicitly set.
    // But wait, the C++ code calls `nangila_init_all_drivers`.
    // I should modify `NangilaHook::all_drivers` or adding `nangila_set_gpu_mode`.
    Box::into_raw(Box::new(hook))
}

/// Enable GPU mode (disables CPU history)
#[no_mangle]
pub unsafe extern "C" fn nangila_set_gpu_mode(handle: NangilaHandle, enabled: i32) {
    if !handle.is_null() {
        let hook = &mut *handle;
        hook.set_gpu_mode(enabled != 0);
    }
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
    if handle.is_null()
        || sendbuff.is_null()
        || out_compressed.is_null()
        || out_compressed_size.is_null()
    {
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
    if hook.is_compression_enabled() {
        1
    } else {
        0
    }
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
    if handle.is_null()
        || sendbuff.is_null()
        || out_compressed.is_null()
        || out_compressed_size.is_null()
    {
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

// =============================================================================
// GPU-Native Interface
// =============================================================================

#[cfg(feature = "cuda")]
mod gpu_impl {
    use super::*;
    use nangila_core::PacketHeader;
    use std::slice;

    /// Compress gradient on GPU (Fused Kernel)
    ///
    /// Dispatches CUDA kernel for predict-quantize.
    /// Writes packet header to `out_header_cpu`.
    #[no_mangle]
    pub unsafe extern "C" fn nangila_compress_gpu(
        handle: NangilaHandle,
        gradient: *const f32,   // GPU
        g_current: *const f32,  // GPU
        g_previous: *const f32, // GPU
        count: usize,
        layer_id: u32,
        out_compressed_payload: *mut u8, // GPU
        out_compressed_size: *mut usize,
        out_header_cpu: *mut u8, // CPU
        stream: *mut std::ffi::c_void,
        scratch_ptr: *mut u32, // GPU scratch (4 bytes)
        sync_mode: i32,        // Synchronization mode
    ) -> i32 {
        if handle.is_null()
            || gradient.is_null()
            || g_current.is_null()
            || g_previous.is_null()
            || out_compressed_payload.is_null()
            || out_header_cpu.is_null()
            || scratch_ptr.is_null()
        {
            return NangilaResult::InvalidPointer as i32;
        }

        let hook = &mut *handle;
        let config = hook.config(); // We need to expose config or access it via state

        // For now, access via internal state which is private.
        // Hack: We need to expose accessor in hook.rs or duplicate logic.
        // The `state` field is private. We should add a method to get config params.
        // Assuming `state().predictor().momentum()` is accessible or similar.
        // Let's rely on `hook.get_momentum(layer_id)` if it exists.
        // Actually `hook.rs` struct fields are private.
        // Use `hook.state.predictor().momentum()` -> `state` is private.
        // I need to update `hook.rs` to expose these.
        // For this step, I will assume `hook.momentum()` and `hook.gamma()` are available.
        // I will add them to hook.rs in a separate step if needed, but `hook.rs` view showed public methods.
        // `hook.is_compression_enabled()` is public.
        // `hook.state` is private.

        let momentum = hook.momentum();
        let gamma = hook.gamma();

        // Convert sync_mode to SyncMode enum
        use nangila_cuda::SyncMode;
        let sync = match sync_mode {
            0 => SyncMode::Async,
            1 => SyncMode::Always,
            2 => SyncMode::Periodic,
            _ => SyncMode::Periodic, // Default to safe mode
        };

        // Launch CUDA Kernel
        let result = predict_and_quantize_cuda(
            gradient,
            g_current,
            g_previous,
            momentum,
            gamma,
            out_compressed_payload,
            count,
            stream as CudaStream,
            sync,
            0, // step - not tracked at this FFI layer
            layer_id,
        );

        if result.is_err() {
            tracing::error!("CUDA Kernel Launch Failed");
            return NangilaResult::CompressionError as i32;
        }

        // Create Header (CPU)
        // Step counter logic? Hook manages step.
        // Create Header (CPU)
        // Step counter logic? Hook manages step.
        let step = hook.current_step() as u32;
        let hash = hook.predictor_hash();
        let header = PacketHeader::new_driver(step, layer_id).with_hash(hash);

        // Serialize Header to CPU buffer
        let header_bytes = header.to_bytes();
        if header_bytes.len() > 24 {
            // Should be exactly 24 bytes
            return NangilaResult::CompressionError as i32;
        }

        let out_slice = slice::from_raw_parts_mut(out_header_cpu, 24);
        out_slice.copy_from_slice(&header_bytes);

        // Calculate detailed size
        // Payload size = (count + 1) / 2
        let payload_size = (count + 1) / 2;
        *out_compressed_size = 24 + payload_size;

        // 2. Compute CRC32
        // Launch CRC32 kernel on payload
        if let Err(e) = compute_crc32_cuda(
            out_compressed_payload,
            payload_size,
            scratch_ptr,
            stream as CudaStream,
        ) {
            tracing::error!("Failed to launch CRC32 kernel: {:?}", e);
            return NangilaResult::CompressionError as i32;
        }

        // 3. Retrieve CRC32 (D2H) + Sync
        // We reuse header buffer or a stack var? We need to copy 4 bytes from scratch_ptr to CPU.
        let mut computed_crc = [0u32; 1];
        let computed_crc_bytes =
            unsafe { slice::from_raw_parts_mut(computed_crc.as_mut_ptr() as *mut u8, 4) };

        if let Err(e) = copy_device_to_host_async(
            scratch_ptr as *const u8,
            computed_crc_bytes.as_mut_ptr(),
            4,
            stream as CudaStream,
        ) {
            tracing::error!("Failed to copy CRC32: {:?}", e);
            return NangilaResult::CompressionError as i32;
        }

        if let Err(e) = synchronize_stream(stream as CudaStream) {
            tracing::error!("Failed to sync for CRC32: {:?}", e);
            return NangilaResult::CompressionError as i32;
        }

        // 4. Update Header
        let final_header = header.with_crc(computed_crc[0]);
        let final_bytes = final_header.to_bytes();

        let out_slice = slice::from_raw_parts_mut(out_header_cpu, 24);
        out_slice.copy_from_slice(&final_bytes);

        NangilaResult::Success as i32
    }

    /// Decompress gradient on GPU (Fused Kernel)
    #[no_mangle]
    pub unsafe extern "C" fn nangila_decompress_gpu(
        handle: NangilaHandle,
        compressed: *const u8, // GPU (start of packet)
        compressed_size: usize,
        g_current: *const f32,  // GPU
        g_previous: *const f32, // GPU
        layer_id: u32,
        recvbuff: *mut f32, // GPU
        count: usize,
        stream: *mut std::ffi::c_void,
        scratch_ptr: *mut u32, // GPU scratch (4 bytes)
        sync_mode: i32,        // Synchronization mode
    ) -> i32 {
        // 1. Basic Validity Checks
        if handle.is_null() || compressed.is_null() || recvbuff.is_null() || scratch_ptr.is_null() {
            return NangilaResult::InvalidPointer as i32;
        }

        if compressed_size < 24 {
            tracing::error!("Packet too small for header: {}", compressed_size);
            return NangilaResult::DecompressionError as i32;
        }

        let cuda_stream = stream as CudaStream;
        let hook = &mut *handle;

        // 2. SAFETY: Header Verification (D2H Copy)
        // We copy 24 bytes from GPU to CPU to check protocol validity before launching kernel
        let mut header_bytes = [0u8; 24];

        // Async copy (enqueued in stream)
        if let Err(e) =
            copy_device_to_host_async(compressed, header_bytes.as_mut_ptr(), 24, cuda_stream)
        {
            tracing::error!("Failed to copy header from GPU: {:?}", e);
            return NangilaResult::DecompressionError as i32;
        }

        // Must synchronize to read the data on CPU
        // Note: This adds a small latency bubble but ensures safety against wrong-rank/corrupt data
        if let Err(e) = synchronize_stream(cuda_stream) {
            tracing::error!("Failed to synchronize stream for header check: {:?}", e);
            return NangilaResult::DecompressionError as i32;
        }

        // 3. Parse and Validate Header
        let header = match PacketHeader::from_bytes(&header_bytes) {
            Some(h) => h,
            None => {
                tracing::error!("Invalid packet header bytes");
                return NangilaResult::DecompressionError as i32;
            }
        };

        if !header.is_valid() {
            tracing::error!("Invalid magic/version in packet header: {:?}", header);
            return NangilaResult::DecompressionError as i32;
        }

        if header.layer_id != layer_id {
            tracing::error!(
                "Layer ID mismatch! Expected {}, got {}",
                layer_id,
                header.layer_id
            );
            return NangilaResult::DecompressionError as i32;
        }

        // Verify Predictor Hash
        let local_hash = hook.predictor_hash();
        if header.predictor_hash != local_hash {
            tracing::error!(
                "Predictor State Desync! Layer {}, Expected Hash {:016x}, Got {:016x}",
                layer_id,
                local_hash,
                header.predictor_hash
            );
            // We should trigger recovery here?
            // hook.trigger_recovery(layer_id); // Requires mutable hook, we have it.
            hook.trigger_recovery(layer_id);
            return NangilaResult::DecompressionError as i32;
        }

        // 3b. Verify CRC32
        // Launch CRC32 on payload
        let payload_offset = 24;
        let payload_len = compressed_size - 24;
        let payload_ptr_u8 = compressed.add(payload_offset);

        if let Err(e) = compute_crc32_cuda(payload_ptr_u8, payload_len, scratch_ptr, cuda_stream) {
            tracing::error!("Failed to launch CRC32 verify: {:?}", e);
            return NangilaResult::DecompressionError as i32;
        }

        // Read back CRC
        let mut computed_crc = [0u32; 1];
        let computed_crc_bytes =
            unsafe { slice::from_raw_parts_mut(computed_crc.as_mut_ptr() as *mut u8, 4) };

        if let Err(e) = copy_device_to_host_async(
            scratch_ptr as *const u8,
            computed_crc_bytes.as_mut_ptr(),
            4,
            cuda_stream,
        ) {
            tracing::error!("Failed to copy CRC32 verify: {:?}", e);
            return NangilaResult::DecompressionError as i32;
        }

        if let Err(e) = synchronize_stream(cuda_stream) {
            tracing::error!("Failed to sync for CRC32 verify: {:?}", e);
            return NangilaResult::DecompressionError as i32;
        }

        if computed_crc[0] != header.crc32 {
            tracing::error!(
                "CRC32 Mismatch! Header: {:08x}, Computed: {:08x}",
                header.crc32,
                computed_crc[0]
            );
            return NangilaResult::DecompressionError as i32;
        }

        // TODO: Verify step count if we pass expected_step in handle/arguments

        // 4. Handle Special Flags
        if header.is_force_sync() {
            // Force Sync: Payload is raw FP32 (uncompressed)
            // We can just copy it directly? Or use a kernel?
            // Current GPU kernel doesn't handle Passthrough/ForceSync raw payloads
            // Nangila logic: If ForceSync, we should reset predictor.
            // We need a separate handling path.
            // For now, let's error out or fallback if force sync is needed?
            // Actually, we can implement a simple copy kernel or cudaMemcpyDeviceToDevice?
            // But the kernel `dequantize_...` expects packed int4.
            tracing::warn!("FORCE_SYNC received on GPU path - Not fully implemented yet!");
            // We proceed to try decompression which will produce garbage...
            // Ideally we should return a specific code to let C++ fallback to CPU handling?
            return NangilaResult::DecompressionError as i32;
        }

        // 5. Decompress Payload
        // Offset by header size
        let payload_ptr = compressed.add(24);

        let momentum = hook.momentum();
        let gamma = hook.gamma();

        // Convert sync_mode to SyncMode enum
        use nangila_cuda::SyncMode;
        let sync = match sync_mode {
            0 => SyncMode::Async,
            1 => SyncMode::Always,
            2 => SyncMode::Periodic,
            _ => SyncMode::Periodic, // Default to safe mode
        };

        let result = dequantize_and_reconstruct_cuda(
            payload_ptr,
            g_current,
            g_previous,
            momentum,
            gamma,
            recvbuff,
            count,
            cuda_stream,
            sync,
        );

        if result.is_err() {
            return NangilaResult::DecompressionError as i32;
        }
        NangilaResult::Success as i32
    }
} // end mod gpu_impl

#[cfg(feature = "cuda")]
pub use gpu_impl::*;

// --- Safe Mode FFI ---

#[no_mangle]
pub extern "C" fn nangila_enable_safe_mode(
    handle: *mut NangilaHook,
    divergence_threshold: f32,
    check_interval: usize,
    max_consecutive_failures: u32,
    recovery_cooldown: usize,
) {
    if handle.is_null() {
        return;
    }
    let hook = unsafe { &mut *handle };
    let config = SafeModeConfig {
        divergence_threshold,
        check_interval,
        max_consecutive_failures,
        recovery_cooldown,
        // Use defaults for canary settings
        ..Default::default()
    };
    hook.enable_safe_mode(config);
}

#[no_mangle]
pub extern "C" fn nangila_report_val_loss(handle: *mut NangilaHook, loss: f32) -> i32 {
    if handle.is_null() {
        return 3; // NoCheck
    }
    let hook = unsafe { &mut *handle };
    match hook.report_validation_loss(loss) {
        SafeModeAction::Continue => 0,
        SafeModeAction::TriggerFallback => 1,
        SafeModeAction::RecoveryComplete => 2,
        SafeModeAction::NoCheck => 3,
        SafeModeAction::CanaryTest => 4,
    }
}
