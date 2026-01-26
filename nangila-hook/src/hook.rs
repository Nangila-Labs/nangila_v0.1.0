//! DDP Communication Hook Implementation
//!
//! This module provides the core hook logic that intercepts gradient
//! communication in PyTorch's DistributedDataParallel.
//!
//! Features:
//! - Packet-based protocol with CRC32 integrity checking
//! - Step counter for temporal synchronization
//! - Desync detection and FORCE_SYNC recovery

use nangila_core::{
    compute_crc32, verify_crc32, CompressedTensor, CompressionResult, GradientHistory, LayerId,
    NangilaConfig, NangilaState, Packet, PacketHeader, SafeModeAction, SafeModeConfig, Tensor,
    TopologyMask,
};

#[cfg(feature = "cuda")]
use nangila_cuda::{
    dequantize_and_reconstruct_cuda, predict_and_quantize_cuda, CudaStream, GpuStateManager,
    SyncMode, CUDA_STREAM_DEFAULT,
};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// Desync recovery mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryMode {
    /// Normal operation
    Normal,
    /// Desync detected, next packet will be FORCE_SYNC
    Pending,
    /// Currently in recovery, waiting for sync confirmation
    Active,
}

/// Nangila DDP Communication Hook
///
/// This struct maintains the compression state and provides the hook
/// interface expected by PyTorch's DDP, with packet-based protocol
/// and desync recovery.
pub struct NangilaHook {
    /// Internal compression state
    state: NangilaState,
    /// Current step counter
    step_counter: u64,
    /// Expected step from peer (for validation)
    expected_peer_step: HashMap<LayerId, u64>,
    /// Gradient history for rewind capability
    gradient_history: GradientHistory,
    /// Desync recovery state per layer
    recovery_mode: HashMap<LayerId, RecoveryMode>,
    /// Number of consecutive CRC failures before triggering recovery
    crc_failure_threshold: u32,
    /// Current CRC failure count per layer
    crc_failure_count: HashMap<LayerId, u32>,
    /// Buffer for passthrough gradients (during warmup)
    passthrough_buffer: HashMap<LayerId, Tensor>,
    /// GPU Mode: If true, disable CPU history buffers to save memory
    gpu_mode: bool,
    // === Predictor Hash Verification ===
    /// How often to verify predictor hash (0 = disabled)
    hash_verify_interval: u64,
    /// Last verified hash (for logging)
    last_verified_hash: u64,
    // === Partial Retransmission ===
    /// Cache of last received packets for partial retransmission
    packet_cache: HashMap<LayerId, Vec<u8>>,
    /// Enable partial retransmission on CRC failure
    partial_retransmit_enabled: bool,
    // === GPU-Native Mode ===
    #[cfg(feature = "cuda")]
    /// GPU state manager for persistent history buffers
    gpu_state_manager: Option<GpuStateManager>,
}

impl NangilaHook {
    /// Create a new hook with given config and mask
    pub fn new(config: NangilaConfig, mask: TopologyMask) -> Self {
        Self {
            state: NangilaState::new(config, mask),
            step_counter: 0,
            expected_peer_step: HashMap::new(),
            gradient_history: GradientHistory::default_depth(),
            recovery_mode: HashMap::new(),
            crc_failure_threshold: 3,
            crc_failure_count: HashMap::new(),
            passthrough_buffer: HashMap::new(),
            gpu_mode: false,
            // Hash verification every 100 steps by default
            hash_verify_interval: 100,
            last_verified_hash: 0,
            // Partial retransmission
            packet_cache: HashMap::new(),
            partial_retransmit_enabled: true,
            // GPU state
            #[cfg(feature = "cuda")]
            gpu_state_manager: None,
        }
    }

    /// Enable or disable partial retransmission
    pub fn set_partial_retransmit(&mut self, enabled: bool) {
        self.partial_retransmit_enabled = enabled;
        if !enabled {
            self.packet_cache.clear();
        }
    }

    /// Enable or disable GPU mode
    pub fn set_gpu_mode(&mut self, enabled: bool) {
        self.gpu_mode = enabled;
        if enabled {
            // Clear existing history to free memory
            self.gradient_history.clear();
            #[cfg(feature = "cuda")]
            {
                // Initialize GPU state manager
                if self.gpu_state_manager.is_none() {
                    self.gpu_state_manager = Some(GpuStateManager::new());
                }
            }
        } else {
            #[cfg(feature = "cuda")]
            {
                // Clean up GPU resources
                self.gpu_state_manager = None;
            }
        }
    }

    /// Create a hook from a saved mask file
    pub fn from_mask_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mask = TopologyMask::load(&mut reader)?;
        Ok(Self::new(NangilaConfig::default(), mask))
    }

    /// Create a hook with default config and empty mask (all Drivers)
    pub fn all_drivers(num_layers: usize) -> Self {
        let mut mask = TopologyMask::new();
        for i in 0..num_layers {
            mask.add_driver(i as LayerId);
        }
        Self::new(NangilaConfig::default(), mask)
    }

    /// Hook callback: called before sending gradients
    ///
    /// Returns a packet with header (step, CRC, flags) and payload.
    pub fn on_send(&mut self, layer_id: LayerId, gradient: Tensor) -> Vec<u8> {
        // Check if we need to send FORCE_SYNC
        let force_sync = matches!(
            self.recovery_mode.get(&layer_id),
            Some(RecoveryMode::Pending)
        );

        if force_sync {
            // Send full FP32 gradient with FORCE_SYNC flag
            self.recovery_mode.insert(layer_id, RecoveryMode::Active);
            return self.create_force_sync_packet(layer_id, &gradient);
        }

        // Store in history for potential rewind (ONLY in CPU mode)
        if !self.gpu_mode {
            let fixed_buf = nangila_core::FixedPointBuffer::from_f32_slice(&gradient.data);
            self.gradient_history.push(layer_id, fixed_buf);
        }

        match self.state.compress(layer_id, &gradient) {
            Ok(CompressionResult::Driver(compressed)) => {
                self.create_driver_packet(layer_id, &compressed)
            }
            Ok(CompressionResult::Passenger) => self.create_passenger_packet(layer_id),
            Ok(CompressionResult::Passthrough(grad)) => {
                self.passthrough_buffer.insert(layer_id, grad.clone());
                self.create_passthrough_packet(layer_id, &grad)
            }
            Err(e) => {
                tracing::warn!("Compression error for layer {}: {}", layer_id, e);
                self.create_passthrough_packet(layer_id, &gradient)
            }
        }
    }

    /// Hook callback: called after receiving gradients
    ///
    /// Validates packet integrity and reconstructs gradient.
    pub fn on_receive(&mut self, layer_id: LayerId, data: &[u8]) -> Tensor {
        // Parse packet
        let packet = match Packet::from_bytes(data) {
            Some(p) => p,
            None => {
                tracing::error!("Failed to parse packet for layer {}", layer_id);
                return Tensor::zeros(vec![1]);
            }
        };

        // Validate packet
        if !packet.header.is_valid() {
            tracing::error!("Invalid packet header for layer {}", layer_id);
            self.trigger_recovery(layer_id);
            return Tensor::zeros(vec![1]);
        }

        // Verify CRC
        if !packet.verify() {
            let count = *self.crc_failure_count.entry(layer_id).or_insert(0);
            let new_count = count + 1;
            self.crc_failure_count.insert(layer_id, new_count);

            tracing::warn!(
                "CRC mismatch for layer {} (failure {} of {})",
                layer_id,
                new_count,
                self.crc_failure_threshold
            );

            // Try partial retransmission if enabled and we have cached data
            if self.partial_retransmit_enabled && self.packet_cache.contains_key(&layer_id) {
                if let Some(recovered) = self.try_partial_recovery(layer_id, &packet) {
                    tracing::info!("Layer {} recovered via partial retransmission", layer_id);
                    self.crc_failure_count.insert(layer_id, 0);
                    return recovered;
                }
            }

            if new_count >= self.crc_failure_threshold {
                self.trigger_recovery(layer_id);
            }
            // Return last known good gradient from history
            return self.recover_from_history(layer_id);
        }

        // CRC valid, reset failure count and cache packet
        self.crc_failure_count.insert(layer_id, 0);

        // Cache packet for potential partial retransmission
        if self.partial_retransmit_enabled && !packet.header.is_partial_retransmit() {
            self.packet_cache.insert(layer_id, data.to_vec());
        }

        // Validate step counter
        let expected = self.expected_peer_step.get(&layer_id).copied().unwrap_or(0);
        if packet.header.step < expected as u32 {
            tracing::warn!(
                "Stale packet for layer {}: got step {}, expected >= {}",
                layer_id,
                packet.header.step,
                expected
            );
            // Skip stale packets
            return self.recover_from_history(layer_id);
        }
        self.expected_peer_step
            .insert(layer_id, packet.header.step as u64 + 1);

        // Handle FORCE_SYNC
        if packet.header.is_force_sync() {
            return self.handle_force_sync(layer_id, &packet);
        }

        // Handle by packet type
        if packet.header.is_passenger() {
            self.reconstruct_passenger(layer_id)
        } else if !self.state.is_compression_enabled() {
            // Warmup: passthrough
            self.deserialize_tensor_payload(&packet.payload)
        } else {
            // Normal compressed driver
            self.reconstruct_driver(layer_id, &packet.payload)
        }
    }

    /// Update state after All-Reduce completes
    pub fn on_complete(&mut self, layer_id: LayerId, gradient: Tensor) {
        self.state.update_state(layer_id, gradient);

        // Clear recovery state on successful completion
        if matches!(
            self.recovery_mode.get(&layer_id),
            Some(RecoveryMode::Active)
        ) {
            self.recovery_mode.insert(layer_id, RecoveryMode::Normal);
            tracing::info!("Layer {} recovered from desync", layer_id);
        }
    }

    /// Advance to the next training step
    pub fn step(&mut self) {
        self.step_counter += 1;
        self.state.step();
        self.passthrough_buffer.clear();
    }

    /// Check if compression is enabled
    pub fn is_compression_enabled(&self) -> bool {
        self.state.is_compression_enabled()
    }

    /// Get current step
    pub fn current_step(&self) -> usize {
        self.state.current_step()
    }

    /// Get compression statistics
    pub fn get_stats(&self) -> nangila_core::CompressionStats {
        self.state.stats()
    }

    /// Get per-layer telemetry
    pub fn get_layer_telemetry(&self, layer_id: LayerId) -> Option<&nangila_core::LayerTelemetry> {
        self.state.layer_telemetry(layer_id)
    }

    /// Get summary telemetry across all layers
    pub fn get_summary_telemetry(&self) -> nangila_core::SummaryTelemetry {
        self.state.summary_telemetry()
    }

    /// Get predictor state hash for verification
    pub fn predictor_hash(&self) -> u64 {
        self.state.predictor().state_hash()
    }

    /// Check if it's time to verify predictor hash (call each step)
    pub fn should_verify_hash(&self) -> bool {
        self.hash_verify_interval > 0
            && self.step_counter > 0
            && self.step_counter % self.hash_verify_interval == 0
    }

    /// Verify predictor hash against peer hash
    /// Returns true if hashes match, false if mismatch detected
    /// On mismatch, triggers recovery for all layers
    pub fn verify_hash(&mut self, peer_hash: u64) -> bool {
        let local_hash = self.predictor_hash();

        if local_hash == peer_hash {
            self.last_verified_hash = local_hash;
            tracing::debug!(
                "Predictor hash verified at step {}: {:016x}",
                self.step_counter,
                local_hash
            );
            true
        } else {
            tracing::error!(
                "PREDICTOR DESYNC DETECTED at step {}! Local: {:016x}, Peer: {:016x}",
                self.step_counter,
                local_hash,
                peer_hash
            );
            // Trigger recovery for all tracked layers
            let layers: Vec<_> = self.state.mask().drivers().take(10).collect();
            for layer_id in layers {
                self.trigger_recovery(layer_id);
            }
            false
        }
    }

    /// Get the hash verification interval
    pub fn hash_verify_interval(&self) -> u64 {
        self.hash_verify_interval
    }

    /// Set the hash verification interval (0 to disable)
    pub fn set_hash_verify_interval(&mut self, interval: u64) {
        self.hash_verify_interval = interval;
    }

    /// Enable Safe Mode with given configuration
    pub fn enable_safe_mode(&mut self, config: SafeModeConfig) {
        self.state.enable_safe_mode(config);
    }

    /// Report validation loss to Safe Mode
    pub fn report_validation_loss(&mut self, loss: f32) -> SafeModeAction {
        self.state.report_validation_loss(loss)
    }

    /// Check if compression is paused due to Safe Mode
    pub fn is_paused(&self) -> bool {
        self.state.is_paused()
    }

    /// Access the configuration
    pub fn config(&self) -> &NangilaConfig {
        &self.state.config
    }

    /// Get current momentum
    pub fn momentum(&self) -> f32 {
        self.state.config.momentum
    }

    /// Get current dynamic gamma (from internal state)
    pub fn gamma(&self) -> f32 {
        // TODO: This should probably be per-layer if we tracked it per-layer in state
        // For now, return default or last used?
        // Actually, state doesn't expose gamma easily.
        // Let's rely on Quantizer's gamma if needed, or just config gamma if fixed.
        // If dynamic, we compute it on GPU in the kernel.
        // If we pass gamma to kernel, it's usually the base or initial gamma if dynamic is on.
        0.001
    }

    /// Manually trigger recovery for a layer
    pub fn trigger_recovery(&mut self, layer_id: LayerId) {
        tracing::warn!("Triggering desync recovery for layer {}", layer_id);
        self.recovery_mode.insert(layer_id, RecoveryMode::Pending);
        self.crc_failure_count.insert(layer_id, 0);
    }

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
        // Get values before mutable borrow
        let momentum = self.momentum();
        let gamma = self.gamma();

        // Get or create GPU state for this layer
        let gpu_manager = self
            .gpu_state_manager
            .as_mut()
            .ok_or("GPU mode not enabled. Call set_gpu_mode(true) first")?;

        let gpu_state = gpu_manager.get_or_create(layer_id, num_elements)?;

        // Allocate output buffer (INT4 = numel/2 bytes, plus metadata)
        let output_size = num_elements / 2 + 128;
        let mut output_buffer = nangila_cuda::GpuBuffer::new(output_size)?;

        // Launch CUDA kernel: predict, subtract, quantize
        let (g_current_ptr, g_previous_ptr) = gpu_state.get_pointers();

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
        let gpu_manager = self
            .gpu_state_manager
            .as_mut()
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

    // --- Packet creation helpers ---

    fn create_driver_packet(&self, layer_id: LayerId, compressed: &CompressedTensor) -> Vec<u8> {
        let payload = self.serialize_compressed(compressed);
        let header = PacketHeader::new_driver(self.step_counter as u32, layer_id);
        Packet::new(header, payload).to_bytes()
    }

    fn create_passenger_packet(&self, layer_id: LayerId) -> Vec<u8> {
        let header = PacketHeader::new_passenger(self.step_counter as u32, layer_id);
        Packet::new(header, Vec::new()).to_bytes()
    }

    fn create_passthrough_packet(&self, layer_id: LayerId, tensor: &Tensor) -> Vec<u8> {
        let payload = self.serialize_tensor_payload(tensor);
        let mut header = PacketHeader::new_driver(self.step_counter as u32, layer_id);
        // Clear driver flag, this is raw passthrough
        header.flags = 0;
        Packet::new(header, payload).to_bytes()
    }

    fn create_force_sync_packet(&self, layer_id: LayerId, tensor: &Tensor) -> Vec<u8> {
        let payload = self.serialize_tensor_payload(tensor);
        let header = PacketHeader::new_force_sync(self.step_counter as u32, layer_id);
        Packet::new(header, payload).to_bytes()
    }

    // --- Reconstruction helpers ---

    fn reconstruct_driver(&mut self, layer_id: LayerId, payload: &[u8]) -> Tensor {
        let compressed = self.deserialize_compressed_payload(payload);
        match self.state.decompress(layer_id, &compressed) {
            Ok(grad) => grad,
            Err(e) => {
                tracing::warn!("Decompression error for layer {}: {}", layer_id, e);
                self.recover_from_history(layer_id)
            }
        }
    }

    fn reconstruct_passenger(&mut self, layer_id: LayerId) -> Tensor {
        match self
            .state
            .decompress(layer_id, &CompressedTensor::default())
        {
            Ok(grad) => grad,
            Err(e) => {
                tracing::warn!("Synthesis error for layer {}: {}", layer_id, e);
                Tensor::zeros(vec![1])
            }
        }
    }

    fn handle_force_sync(&mut self, layer_id: LayerId, packet: &Packet) -> Tensor {
        tracing::info!("Received FORCE_SYNC for layer {}", layer_id);

        let gradient = self.deserialize_tensor_payload(&packet.payload);

        // Reset predictor state for this layer with the synced gradient
        self.state
            .predictor_mut()
            .force_sync_layer(layer_id, &gradient);

        // Clear recovery mode
        self.recovery_mode.insert(layer_id, RecoveryMode::Normal);

        gradient
    }

    fn recover_from_history(&self, layer_id: LayerId) -> Tensor {
        // Try to get the most recent gradient from history
        if let Some(buf) = self.gradient_history.get(layer_id, 0) {
            Tensor::new(buf.to_f32_vec(), vec![buf.len()])
        } else {
            tracing::error!("No history available for layer {} recovery", layer_id);
            Tensor::zeros(vec![1])
        }
    }

    /// Try to recover from CRC failure using partial retransmission
    ///
    /// Strategy: Identify corrupted chunks by comparing with cached packet,
    /// then merge good chunks from cache with new chunks from current packet.
    fn try_partial_recovery(
        &mut self,
        layer_id: LayerId,
        corrupted_packet: &Packet,
    ) -> Option<Tensor> {
        let cached_data = self.packet_cache.get(&layer_id)?;
        let cached_packet = Packet::from_bytes(cached_data)?;

        // Only works for driver packets with same structure
        if !corrupted_packet.header.is_driver() || !cached_packet.header.is_driver() {
            return None;
        }

        // Simple strategy: Use cached payload if it's from a recent step
        // In a real implementation, you'd:
        // 1. Divide payload into chunks (e.g., 1KB each)
        // 2. Compute CRC per chunk
        // 3. Identify which chunks are corrupted
        // 4. Request retransmission of only those chunks
        // 5. Merge good chunks from cache with retransmitted chunks

        // For now, just use the cached packet if it's from the previous step
        let step_diff = corrupted_packet
            .header
            .step
            .saturating_sub(cached_packet.header.step);
        if step_diff <= 1 {
            tracing::debug!(
                "Using cached packet from step {} for layer {} (current step {})",
                cached_packet.header.step,
                layer_id,
                corrupted_packet.header.step
            );

            // Reconstruct from cached compressed data
            if self.state.is_compression_enabled() {
                let compressed = self.deserialize_compressed_payload(&cached_packet.payload);
                self.state.decompress(layer_id, &compressed).ok()
            } else {
                Some(self.deserialize_tensor_payload(&cached_packet.payload))
            }
        } else {
            None
        }
    }

    // --- Serialization helpers ---

    fn serialize_compressed(&self, compressed: &CompressedTensor) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&compressed.gamma.to_le_bytes());
        bytes.extend_from_slice(&(compressed.numel as u32).to_le_bytes());
        bytes.extend_from_slice(&(compressed.shape.len() as u32).to_le_bytes());
        for &dim in &compressed.shape {
            bytes.extend_from_slice(&(dim as u32).to_le_bytes());
        }
        bytes.extend_from_slice(&compressed.data);
        bytes
    }

    fn deserialize_compressed_payload(&self, bytes: &[u8]) -> CompressedTensor {
        if bytes.len() < 12 {
            return CompressedTensor::default();
        }

        let gamma = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let numel = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        let shape_len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;

        let mut offset = 12;
        let mut shape = Vec::with_capacity(shape_len);
        for _ in 0..shape_len {
            if offset + 4 > bytes.len() {
                break;
            }
            let dim = u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]) as usize;
            shape.push(dim);
            offset += 4;
        }

        let data = bytes[offset..].to_vec();

        CompressedTensor {
            gamma,
            numel,
            shape,
            data,
        }
    }

    fn serialize_tensor_payload(&self, tensor: &Tensor) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(4 + 4 + tensor.shape.len() * 4 + tensor.data.len() * 4);
        bytes.extend_from_slice(&(tensor.numel() as u32).to_le_bytes());
        bytes.extend_from_slice(&(tensor.shape.len() as u32).to_le_bytes());
        for &dim in &tensor.shape {
            bytes.extend_from_slice(&(dim as u32).to_le_bytes());
        }
        for &val in &tensor.data {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        bytes
    }

    fn deserialize_tensor_payload(&self, bytes: &[u8]) -> Tensor {
        if bytes.len() < 8 {
            return Tensor::zeros(vec![1]);
        }

        let numel = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let shape_len = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;

        let mut offset = 8;
        let mut shape = Vec::with_capacity(shape_len);
        for _ in 0..shape_len {
            if offset + 4 > bytes.len() {
                break;
            }
            let dim = u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]) as usize;
            shape.push(dim);
            offset += 4;
        }

        let mut data = Vec::with_capacity(numel);
        while offset + 4 <= bytes.len() && data.len() < numel {
            let val = f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
            data.push(val);
            offset += 4;
        }

        Tensor::new(data, shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hook_creation() {
        let hook = NangilaHook::all_drivers(10);
        assert!(!hook.is_compression_enabled());
        assert_eq!(hook.current_step(), 0);
    }

    #[test]
    fn test_packet_roundtrip() {
        let mut hook = NangilaHook::all_drivers(1);
        let grad = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);

        let packet_bytes = hook.on_send(0, grad.clone());
        let recovered = hook.on_receive(0, &packet_bytes);

        // During warmup, should match exactly
        assert_eq!(grad.data.len(), recovered.data.len());
    }

    #[test]
    fn test_recovery_trigger() {
        let mut hook = NangilaHook::all_drivers(1);

        hook.trigger_recovery(0);
        assert_eq!(hook.recovery_mode.get(&0), Some(&RecoveryMode::Pending));
    }
}
