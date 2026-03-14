//! GPU state management for persistent history buffers
//!
//! This module manages GPU-resident history buffers for gradient prediction,
//! eliminating the need for CPU round-trips.

use crate::bindings::{CudaError, CudaResult};
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::ffi::c_void;

#[cfg(feature = "cuda")]
extern "C" {
    fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(ptr: *mut c_void) -> i32;
    fn cudaMemset(ptr: *mut c_void, value: i32, count: usize) -> i32;
    fn cudaGetDevice(device: *mut i32) -> i32;
}

/// GPU memory buffer with RAII semantics
pub struct GpuBuffer {
    ptr: *mut f32,
    size_bytes: usize,
    device_id: i32,
}

impl GpuBuffer {
    /// Allocate a new GPU buffer
    #[cfg(feature = "cuda")]
    pub fn new(num_elements: usize) -> CudaResult<Self> {
        let size_bytes = num_elements * std::mem::size_of::<f32>();
        let mut ptr: *mut c_void = std::ptr::null_mut();

        unsafe {
            let result = cudaMalloc(&mut ptr as *mut *mut c_void, size_bytes);
            let error = CudaError::from_code(result);
            if !error.is_success() {
                return Err(error);
            }

            // Zero-initialize
            let result = cudaMemset(ptr, 0, size_bytes);
            let error = CudaError::from_code(result);
            if !error.is_success() {
                cudaFree(ptr); // Clean up on error
                return Err(error);
            }

            // Get current device
            let mut device_id = 0;
            cudaGetDevice(&mut device_id);

            Ok(Self {
                ptr: ptr as *mut f32,
                size_bytes,
                device_id,
            })
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub fn new(_num_elements: usize) -> CudaResult<Self> {
        Err(CudaError::InitializationError)
    }

    /// Get raw device pointer
    pub fn as_ptr(&self) -> *mut f32 {
        self.ptr
    }

    /// Get buffer size in bytes
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    /// Get device ID
    pub fn device_id(&self) -> i32 {
        self.device_id
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        unsafe {
            if !self.ptr.is_null() {
                cudaFree(self.ptr as *mut c_void);
                self.ptr = std::ptr::null_mut();
            }
        }
    }
}

// Explicit Send/Sync bounds - GPU pointers are safe to send between threads
// as long as they're used with proper CUDA stream synchronization
unsafe impl Send for GpuBuffer {}
unsafe impl Sync for GpuBuffer {}

/// State for a single layer on GPU
pub struct GpuLayerState {
    /// Gradient at step t (for prediction)
    pub g_current: GpuBuffer,
    /// Gradient at step t-1 (for prediction)
    pub g_previous: GpuBuffer,
    /// Number of elements
    pub num_elements: usize,
    /// Current step for this layer
    pub step: u64,
}

impl GpuLayerState {
    /// Create new GPU state for a layer
    pub fn new(num_elements: usize) -> CudaResult<Self> {
        Ok(Self {
            g_current: GpuBuffer::new(num_elements)?,
            g_previous: GpuBuffer::new(num_elements)?,
            num_elements,
            step: 0,
        })
    }

    /// Swap current and previous buffers (at step boundary)
    pub fn advance_step(&mut self) {
        std::mem::swap(&mut self.g_current, &mut self.g_previous);
        self.step += 1;
    }

    /// Get pointers for kernel launch
    pub fn get_pointers(&self) -> (*mut f32, *mut f32) {
        (self.g_current.as_ptr(), self.g_previous.as_ptr())
    }
}

/// Manager for all GPU layer states
pub struct GpuStateManager {
    layers: HashMap<u32, GpuLayerState>,
}

impl GpuStateManager {
    /// Create new empty state manager
    pub fn new() -> Self {
        Self {
            layers: HashMap::new(),
        }
    }

    /// Get or create state for a layer
    pub fn get_or_create(
        &mut self,
        layer_id: u32,
        num_elements: usize,
    ) -> CudaResult<&mut GpuLayerState> {
        if !self.layers.contains_key(&layer_id) {
            let state = GpuLayerState::new(num_elements)?;
            self.layers.insert(layer_id, state);
        }

        Ok(self.layers.get_mut(&layer_id).unwrap())
    }

    /// Get existing state for a layer
    pub fn get(&self, layer_id: u32) -> Option<&GpuLayerState> {
        self.layers.get(&layer_id)
    }

    /// Get mutable state for a layer
    pub fn get_mut(&mut self, layer_id: u32) -> Option<&mut GpuLayerState> {
        self.layers.get_mut(&layer_id)
    }

    /// Advance all layers to next step
    pub fn advance_all(&mut self) {
        for state in self.layers.values_mut() {
            state.advance_step();
        }
    }

    /// Get total GPU memory allocated (bytes)
    pub fn total_memory_bytes(&self) -> usize {
        self.layers
            .values()
            .map(|s| s.g_current.size_bytes() + s.g_previous.size_bytes())
            .sum()
    }
}

impl Default for GpuStateManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "cuda")]
    fn test_gpu_buffer_allocation() {
        let buffer = GpuBuffer::new(1024);
        assert!(buffer.is_ok());
        let buffer = buffer.unwrap();
        assert_eq!(buffer.size_bytes(), 1024 * 4); // f32 = 4 bytes
    }

    #[test]
    fn test_state_manager() {
        #[cfg(feature = "cuda")]
        {
            let mut manager = GpuStateManager::new();
            let result = manager.get_or_create(0, 1024);
            assert!(result.is_ok());
            assert_eq!(manager.total_memory_bytes(), 1024 * 4 * 2); // 2 buffers
        }

        #[cfg(not(feature = "cuda"))]
        {
            let manager = GpuStateManager::new();
            assert_eq!(manager.total_memory_bytes(), 0);
        }
    }
}
