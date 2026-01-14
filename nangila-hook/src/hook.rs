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
    NangilaConfig, NangilaState, Packet, PacketHeader, Tensor, TopologyMask,
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

        // Store in history for potential rewind
        let fixed_buf = nangila_core::FixedPointBuffer::from_f32_slice(&gradient.data);
        self.gradient_history.push(layer_id, fixed_buf);

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
            let count = self.crc_failure_count.entry(layer_id).or_insert(0);
            *count += 1;
            tracing::warn!(
                "CRC mismatch for layer {} (failure {} of {})",
                layer_id,
                count,
                self.crc_failure_threshold
            );

            if *count >= self.crc_failure_threshold {
                self.trigger_recovery(layer_id);
            }
            // Return last known good gradient from history
            return self.recover_from_history(layer_id);
        }

        // CRC valid, reset failure count
        self.crc_failure_count.insert(layer_id, 0);

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

    /// Get predictor state hash for verification
    pub fn predictor_hash(&self) -> u64 {
        self.state.predictor().state_hash()
    }

    /// Manually trigger recovery for a layer
    pub fn trigger_recovery(&mut self, layer_id: LayerId) {
        tracing::warn!("Triggering desync recovery for layer {}", layer_id);
        self.recovery_mode.insert(layer_id, RecoveryMode::Pending);
        self.crc_failure_count.insert(layer_id, 0);
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
