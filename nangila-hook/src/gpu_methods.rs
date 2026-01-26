    // === GPU-Native Methods ===
    
    /// GPU-native compression: gradient stays on GPU throughout
    ///
    /// # Arguments
    /// * `layer_id` - Layer identifier
    /// * `gradient_ptr` - Device pointer to gradient tensor
    /// * `num_elements` - Number of elements in gradient
    /// * `stream` - CUDA stream for async execution
    ///
    /// # Returns
    /// Device pointer to compressed output buffer
    #[cfg(feature = "cuda")]
    pub unsafe fn on_send_gpu(
        &mut self,
        layer_id: LayerId,
        gradient_ptr: *const f32,
        num_elements: usize,
        stream: CudaStream,
    ) -> Result<*const u8, Box<dyn std::error::Error>> {
        // Get or create GPU state for this layer
        let gpu_manager = self.gpu_state_manager.as_mut()
            .ok_or("GPU mode not enabled. Call set_gpu_mode(true) first")?;
        
        let gpu_state = gpu_manager.get_or_create(layer_id, num_elements)?;
        
        // Allocate output buffer (INT4 = numel/2 bytes, plus metadata)
        let output_size = num_elements / 2 + 128; // Extra space for metadata
        let mut output_buffer = nangila_cuda::GpuBuffer::new(output_size)?;
        
        // Launch CUDA kernel: predict, subtract, quantize
        let (g_current_ptr, g_previous_ptr) = gpu_state.get_pointers();
        let momentum = self.momentum();
        let gamma = self.gamma();
        
        predict_and_quantize_cuda(
            gradient_ptr,
            g_current_ptr,
            g_previous_ptr,
            momentum,
            gamma,
            output_buffer.as_ptr() as *mut u8,
            num_elements,
            stream,
            SyncMode::Periodic,
            gpu_state.step,
            layer_id,
        )?;
        
        Ok(output_buffer.as_ptr() as *const u8)
    }
    
    /// GPU-native state update: update history buffers on GPU
    ///
    /// # Arguments
    /// * `layer_id` - Layer identifier
    /// * `gradient_ptr` - Device pointer to gradient tensor
    /// * `num_elements` - Number of elements
    /// * `stream` - CUDA stream for async execution
    #[cfg(feature = "cuda")]
    pub unsafe fn on_complete_gpu(
        &mut self,
        layer_id: LayerId,
        gradient_ptr: *const f32,
        num_elements: usize,
        stream: CudaStream,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Get GPU state
        let gpu_manager = self.gpu_state_manager.as_mut()
            .ok_or("GPU mode not enabled")?;
        
        let gpu_state = gpu_manager.get_or_create(layer_id, num_elements)?;
        
        // Copy gradient to g_current buffer using cudaMemcpyAsync
        extern "C" {
            fn cudaMemcpyAsync(
                dst: *mut std::ffi::c_void,
                src: *const std::ffi::c_void,
                count: usize,
                kind: i32,
                stream: *mut std::ffi::c_void,
            ) -> i32;
        }
        
        const cudaMemcpyDeviceToDevice: i32 = 3;
        
        let result = cudaMemcpyAsync(
            gpu_state.g_current.as_ptr() as *mut std::ffi::c_void,
            gradient_ptr as *const std::ffi::c_void,
            num_elements * std::mem::size_of::<f32>(),
            cudaMemcpyDeviceToDevice,
            stream,
        );
        
        if result != 0 {
            return Err(format!("cudaMemcpyAsync failed with code {}", result).into());
        }
        
        Ok(())
    }
    
    /// Advance all GPU layer states to next step
    #[cfg(feature = "cuda")]
    pub fn step_gpu(&mut self) {
        if let Some(manager) = &mut self.gpu_state_manager {
            manager.advance_all();
        }
        self.step(); // Also advance CPU state
    }
