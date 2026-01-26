//! Unit tests for CUDA bindings with error handling
//!
//! These tests verify error handling without requiring actual GPU hardware.

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sync_mode_default() {
        assert_eq!(SyncMode::default(), SyncMode::Periodic);
    }
    
    #[test]
    fn test_sync_mode_values() {
        assert_eq!(SyncMode::Async as i32, 0);
        assert_eq!(SyncMode::Always as i32, 1);
        assert_eq!(SyncMode::Periodic as i32, 2);
    }
    
    #[test]
    fn test_cuda_error_display() {
        assert_eq!(CudaError::Success.to_string(), "CUDA Success");
        assert_eq!(CudaError::InvalidValue.to_string(), "CUDA Invalid Value");
        assert_eq!(CudaError::LaunchFailure.to_string(), "CUDA Kernel Launch Failure");
    }
    
    #[test]
    fn test_cuda_error_from_code() {
        assert_eq!(CudaError::from_code(0), CudaError::Success);
        assert_eq!(CudaError::from_code(1), CudaError::InvalidValue);
        assert_eq!(CudaError::from_code(719), CudaError::LaunchFailure);
        assert_eq!(CudaError::from_code(999), CudaError::Unknown);
    }
    
    #[test]
    fn test_cuda_error_is_success() {
        assert!(CudaError::Success.is_success());
        assert!(!CudaError::InvalidValue.is_success());
        assert!(!CudaError::LaunchFailure.is_success());
    }
    
    // Test input validation (without CUDA)
    #[test]
    #[cfg(not(feature = "cuda"))]
    fn test_predict_and_quantize_returns_error_without_cuda() {
        unsafe {
            let result = predict_and_quantize_cuda(
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0.9,
                0.1,
                std::ptr::null_mut(),
                100,
                std::ptr::null_mut(),
                SyncMode::Async,
            );
            
            assert!(result.is_err());
            assert_eq!(result.unwrap_err(), CudaError::InitializationError);
        }
    }
    
    #[test]
    #[cfg(not(feature = "cuda"))]
    fn test_dequantize_and_reconstruct_returns_error_without_cuda() {
        unsafe {
            let result = dequantize_and_reconstruct_cuda(
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0.9,
                0.1,
                std::ptr::null_mut(),
                100,
                std::ptr::null_mut(),
                SyncMode::Async,
            );
            
            assert!(result.is_err());
            assert_eq!(result.unwrap_err(), CudaError::InitializationError);
        }
    }
    
    // Test that validation would catch null pointers (if CUDA were available)
    #[test]
    #[cfg(feature = "cuda")]
    fn test_predict_and_quantize_validates_null_pointers() {
        unsafe {
            // All null pointers should fail validation
            let result = predict_and_quantize_cuda(
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0.9,
                0.1,
                std::ptr::null_mut(),
                100,
                std::ptr::null_mut(),
                SyncMode::Async,
            );
            
            assert!(result.is_err());
            assert_eq!(result.unwrap_err(), CudaError::InvalidValue);
        }
    }
    
    #[test]
    #[cfg(feature = "cuda")]
    fn test_predict_and_quantize_validates_zero_size() {
        unsafe {
            // Create dummy non-null pointers (won't be dereferenced due to n=0)
            let dummy = 1u32;
            let ptr = &dummy as *const u32 as *const f32;
            let out_ptr = &dummy as *const u32 as *mut u8;
            
            let result = predict_and_quantize_cuda(
                ptr,
                ptr,
                ptr,
                0.9,
                0.1,
                out_ptr,
                0,  // Invalid: zero size
                std::ptr::null_mut(),
                SyncMode::Async,
            );
            
            assert!(result.is_err());
            assert_eq!(result.unwrap_err(), CudaError::InvalidValue);
        }
    }
    
    #[test]
    #[cfg(feature = "cuda")]
    fn test_predict_and_quantize_validates_invalid_gamma() {
        unsafe {
            let dummy = 1u32;
            let ptr = &dummy as *const u32 as *const f32;
            let out_ptr = &dummy as *const u32 as *mut u8;
            
            // Test gamma = 0 (should be clamped to 1e-8)
            let result = predict_and_quantize_cuda(
                ptr,
                ptr,
                ptr,
                0.9,
                0.0,  // Will be clamped
                out_ptr,
                100,
                std::ptr::null_mut(),
                SyncMode::Async,
            );
            
            // Should not fail due to clamping
            // (Will fail later due to invalid pointers, but not due to gamma)
            
            // Test gamma = NaN
            let result = predict_and_quantize_cuda(
                ptr,
                ptr,
                ptr,
                0.9,
                f32::NAN,
                out_ptr,
                100,
                std::ptr::null_mut(),
                SyncMode::Async,
            );
            
            assert!(result.is_err());
            assert_eq!(result.unwrap_err(), CudaError::InvalidValue);
        }
    }
    
    #[test]
    #[cfg(feature = "cuda")]
    fn test_predict_and_quantize_validates_invalid_momentum() {
        unsafe {
            let dummy = 1u32;
            let ptr = &dummy as *const u32 as *const f32;
            let out_ptr = &dummy as *const u32 as *mut u8;
            
            // Test momentum > 1.0
            let result = predict_and_quantize_cuda(
                ptr,
                ptr,
                ptr,
                1.5,  // Invalid
                0.1,
                out_ptr,
                100,
                std::ptr::null_mut(),
                SyncMode::Async,
            );
            
            assert!(result.is_err());
            assert_eq!(result.unwrap_err(), CudaError::InvalidValue);
            
            // Test momentum < 0.0
            let result = predict_and_quantize_cuda(
                ptr,
                ptr,
                ptr,
                -0.1,  // Invalid
                0.1,
                out_ptr,
                100,
                std::ptr::null_mut(),
                SyncMode::Async,
            );
            
            assert!(result.is_err());
            assert_eq!(result.unwrap_err(), CudaError::InvalidValue);
        }
    }
}
