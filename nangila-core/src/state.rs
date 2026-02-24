//! NangilaState: Main state container for compression/decompression
//!
//! This is the primary interface for the Nangila compression system,
//! coordinating the Predictor, Quantizer, Mask, and Reconstructor.

use crate::safe_mode::{SafeMode, SafeModeAction, SafeModeConfig, SafeModeStats};
use crate::{
    compressor::{Compressor, PredictionResidualCompressor},
    config::CompressorType,
    CompressedTensor, LayerId, NangilaConfig, Reconstructor, Result, Tensor,
    TopologyMask,
    // Predictor and Quantizer are now internal to specific compressors
};
use std::collections::HashMap;

/// Per-layer compression telemetry
#[derive(Debug, Clone, Default)]
pub struct LayerTelemetry {
    /// Total bytes before compression
    pub total_original_bytes: u64,
    /// Total bytes after compression
    pub total_compressed_bytes: u64,
    /// Number of compressions
    pub compression_count: u64,
    /// Average compression ratio
    pub avg_compression_ratio: f32,
    /// Min compression ratio seen
    pub min_compression_ratio: f32,
    /// Max compression ratio seen
    pub max_compression_ratio: f32,
    /// Average prediction error (L2 norm of residual / L2 norm of gradient)
    pub avg_prediction_error: f32,
    /// Number of times this layer was a passenger (skipped)
    pub passenger_skip_count: u64,
}

impl LayerTelemetry {
    fn new() -> Self {
        Self {
            total_original_bytes: 0,
            total_compressed_bytes: 0,
            compression_count: 0,
            avg_compression_ratio: 0.0,
            min_compression_ratio: f32::MAX,
            max_compression_ratio: 0.0,
            avg_prediction_error: 0.0,
            passenger_skip_count: 0,
        }
    }

    fn record_compression(
        &mut self,
        original_bytes: usize,
        compressed_bytes: usize,
        prediction_error: f32,
    ) {
        self.total_original_bytes += original_bytes as u64;
        self.total_compressed_bytes += compressed_bytes as u64;
        self.compression_count += 1;

        let ratio = original_bytes as f32 / compressed_bytes.max(1) as f32;
        self.min_compression_ratio = self.min_compression_ratio.min(ratio);
        self.max_compression_ratio = self.max_compression_ratio.max(ratio);

        // Update running average
        let alpha = 0.1; // EMA smoothing
        self.avg_compression_ratio = alpha * ratio + (1.0 - alpha) * self.avg_compression_ratio;
        self.avg_prediction_error =
            alpha * prediction_error + (1.0 - alpha) * self.avg_prediction_error;
    }

    fn record_passenger_skip(&mut self) {
        self.passenger_skip_count += 1;
    }
}

/// Main state container for Nangila compression
#[derive(Debug)]
pub struct NangilaState {
    /// Configuration
    pub config: NangilaConfig,
    /// Topology mask (Driver/Passenger mapping)
    mask: TopologyMask,
    /// Composable compressor engine
    compressor: Box<dyn Compressor>,
    /// Reconstructor for decompression (Passenger synthesis)
    reconstructor: Reconstructor,
    /// Current training step
    step: usize,
    /// Whether compression is currently enabled
    compression_enabled: bool,
    /// Safe Mode convergence monitor (optional)
    safe_mode: Option<SafeMode>,
    /// Whether compression is paused due to divergence
    paused_for_divergence: bool,
    /// Per-layer compression telemetry
    layer_telemetry: HashMap<LayerId, LayerTelemetry>,
}

/// Result of compression for a single layer
#[derive(Debug)]
pub enum CompressionResult {
    /// Layer is a Driver: compressed residual data
    Driver(CompressedTensor),
    /// Layer is a Passenger: no data to send
    Passenger,
    /// Compression disabled (warmup phase): original gradient
    Passthrough(Tensor),
}

impl NangilaState {
    /// Create a new NangilaState with given config and mask
    pub fn new(config: NangilaConfig, mask: TopologyMask) -> Self {
        let compressor: Box<dyn Compressor> = match config.compressor_type {
            CompressorType::PredictionResidual => {
                Box::new(PredictionResidualCompressor::new(config.clone()))
            }
            CompressorType::DGC => {
                Box::new(crate::dgc::DGCCompressor::new(config.clone()))
            }
            CompressorType::PowerSGD => {
                Box::new(crate::power_sgd::PowerSGDCompressor::new(config.clone()))
            }
        };

        // Reconstructor is still needed for Passengers
        let reconstructor =
            Reconstructor::new(crate::Quantizer::new(config.quantize_bits, config.dynamic_gamma));

        Self {
            config,
            mask,
            compressor,
            reconstructor,
            step: 0,
            compression_enabled: false,
            safe_mode: None,
            paused_for_divergence: false,
            layer_telemetry: HashMap::new(),
        }
    }

    /// Create a NangilaState with default config and empty mask
    pub fn default_with_mask(mask: TopologyMask) -> Self {
        Self::new(NangilaConfig::default(), mask)
    }

    /// Compress a gradient for transmission
    ///
    /// Returns the compression result based on layer role and warmup state.
    pub fn compress(&mut self, layer_id: LayerId, gradient: &Tensor) -> Result<CompressionResult> {
        // During warmup, pass through raw gradients
        if !self.is_compression_enabled() {
            // Still update predictor during shadow-run phase if ready
            if self.step >= self.config.warmup_steps {
                // We use update() here to mimic shadow run behavior
                let _ = self.compressor.update(layer_id, gradient);
            }
            return Ok(CompressionResult::Passthrough(gradient.clone()));
        }

        // Check layer role
        let role = self.mask.get_role(layer_id)?;

        if role.is_passenger() {
            // Passengers send nothing
            self.layer_telemetry
                .entry(layer_id)
                .or_insert_with(LayerTelemetry::new)
                .record_passenger_skip();
            return Ok(CompressionResult::Passenger);
        }

        // Driver: delegate to compressor
        let packet = self.compressor.compress(gradient, layer_id)?;
        
        // Deserialize back to CompressedTensor purely for compatibility/telemetry
        // This is inefficient but part of the migration
        let compressed: CompressedTensor = bincode::deserialize(&packet.payload)?;
        
        // Record telemetry (approximate)
        let original_bytes = gradient.numel() * 4;
        let compressed_bytes = packet.payload.len();
        self.layer_telemetry
            .entry(layer_id)
            .or_insert_with(LayerTelemetry::new)
            .record_compression(original_bytes, compressed_bytes, 0.0); // No error metric available from trait

        Ok(CompressionResult::Driver(compressed))
    }

    /// Decompress received data and reconstruct gradients
    ///
    /// For Drivers: dequantize residual and add prediction
    /// For Passengers: synthesize from Driver gradient
    pub fn decompress(
        &mut self,
        layer_id: LayerId,
        compressed: &CompressedTensor,
    ) -> Result<Tensor> {
        let role = self.mask.get_role(layer_id)?;

        let gradient = if role.is_driver() {
            // Re-serialize strictly to interface with compressor trait (inefficient but necessary for migration)
            let payload = bincode::serialize(compressed)?;
            // Use dummy header/step valid for decompress? Or rely on payload
            let packet = crate::Packet::new(crate::PacketHeader::default(), payload); 
            
            let gradient = self.compressor.decompress(&packet, layer_id)?;
            
            // Register with reconstructor for potential passenger synthesis
            self.reconstructor.cache_driver_gradient(layer_id, gradient.clone());
            
            gradient
        } else {
            self.reconstructor
                .synthesize_passenger(layer_id, &self.mask)?
        };

        Ok(gradient)
    }

    /// Decompress a specific shard of the gradient
    ///
    /// For FSDP: Drivers are dequantized from compressed data.
    /// Passengers are synthesized from their cached source driver.
    /// Note: For passengers, the source driver must be reconstructed/cached first.
    pub fn decompress_partial(
        &mut self,
        layer_id: LayerId,
        compressed: &CompressedTensor,
        start_index: usize,
        end_index: usize,
    ) -> Result<Tensor> {
        let role = self.mask.get_role(layer_id)?;

        let gradient = if role.is_driver() {
            // Partial decompression is not yet supported on Compressor trait level!
            // We need to extend trait or fallback.
            // For now, keeping old logic via direct access if possible? No, we lost 'predictor'.
            // Falling back to full decompression (inefficient but correct)
            // Or TODO: Add decompress_partial to trait.
            // For Phase 1 strict adherence, let's do full decompress + slice.
            
            // Re-serialize
            let payload = bincode::serialize(compressed)?;
            let packet = crate::Packet::new(crate::PacketHeader::default(), payload);
            
            let full_grad = self.compressor.decompress(&packet, layer_id)?;
            
            // Slice it 
            // Tensor structure is simple Vec<f32>.
            // Need to map indices
            // This is actually tricky because 'decompress_partial' implies we only transmit/store partial?
            // FSDP uses it to get SHARD.
            //
            // If the buffer is full-sized, we just slice.
            // If we only have partial data?
            // CompressedTensor is the full compressed data usually.
            
            // Let's stub with TODO or full implementation.
            // Slicing:
            let start = start_index.min(full_grad.numel());
            let end = end_index.min(full_grad.numel());
            let data = full_grad.data[start..end].to_vec();
            Tensor::new(data, vec![end - start])
            
        } else {
            // For passengers, we need the source driver to be cached
            // This requires drivers to be reconstructed before passengers in FSDP
            self.reconstructor.synthesize_passenger_partial(
                layer_id,
                &self.mask,
                start_index,
                end_index,
            )?
        };

        Ok(gradient)
    }

    pub fn decompress_all(
        &mut self,
        compressed_drivers: &HashMap<LayerId, CompressedTensor>,
    ) -> Result<HashMap<LayerId, Tensor>> {
        // We can't use Reconstructor::reconstruct_all easily without exposing Predictor from Compressor.
        // Instead, loop and decompress individually using Compressor trait.
        // This loses batch optimization potential but works for abstraction.
        
        let mut results = HashMap::new();
        for (&layer_id, compressed) in compressed_drivers {
             let payload = bincode::serialize(compressed)?;
             let packet = crate::Packet::new(crate::PacketHeader::default(), payload);
             let tensor = self.compressor.decompress(&packet, layer_id)?;
             
             // Register with reconstructor
             self.reconstructor.cache_driver_gradient(layer_id, tensor.clone());
             
             results.insert(layer_id, tensor);
        }
        
        // Passengers? 'reconstruct_all' usually handles passengers too.
        // We need to synthesize passengers efficiently.
        // Reconstructor::synthesize_passengers_bulk?
        // Let's assume we just returned drivers here.
        // Original reconstruct_all did both drivers and passengers.
        
        // Synthesize passengers
        // We need 'reconstructor' to handle this.
        // But 'reconstructor' in NangilaState is intended for Passengers now.
        // Does 'reconstructor' have access to the decompressed drivers?
        // We can pass them.
        
        // For now, let's implement simplified logic.
        // Iterate passengers in mask
        for (passenger_id, _, _) in self.mask.passengers() {
             let tensor = self.reconstructor.synthesize_passenger(passenger_id, &self.mask)?;
             results.insert(passenger_id, tensor);
        }
        
        Ok(results)
    }

    /// Update predictor state with actual gradient (post-All-Reduce)
    ///
    /// IMPORTANT: Call this with the reconstructed gradient to maintain
    /// bit-perfect consensus between sender and receiver predictors.
    /// Update predictor state with actual gradient (post-All-Reduce)
    ///
    /// IMPORTANT: Call this with the reconstructed gradient to maintain
    /// bit-perfect consensus between sender and receiver predictors.
    pub fn update_state(&mut self, layer_id: LayerId, gradient: Tensor) {
        let _ = self.compressor.update(layer_id, &gradient);
    }

    /// Advance to the next training step
    pub fn step(&mut self) {
        self.step += 1;
        self.compressor.step();

        // Enable compression after warmup + shadow-run
        let enable_step = self.config.warmup_steps + self.config.shadow_run_steps;
        if self.step >= enable_step && !self.compression_enabled {
            self.compression_enabled = true;
            tracing::info!(
                "Nangila compression enabled at step {} ({} drivers, {} passengers, {:.1}x mask compression)",
                self.step,
                self.mask.num_drivers(),
                self.mask.num_passengers(),
                self.mask.compression_ratio()
            );
        }

        // Clear reconstructor cache at end of step
        self.reconstructor.clear_cache();
    }

    /// Check if compression is currently enabled
    pub fn is_compression_enabled(&self) -> bool {
        self.compression_enabled && !self.paused_for_divergence
    }

    /// Get the current step
    pub fn current_step(&self) -> usize {
        self.step
    }

    /// Get a reference to the topology mask
    pub fn mask(&self) -> &TopologyMask {
        &self.mask
    }

    /// Get a mutable reference to the topology mask
    pub fn mask_mut(&mut self) -> &mut TopologyMask {
        &mut self.mask
    }

    /// Get predictor state hash for verification
    pub fn predictor_hash(&self) -> u64 {
        self.compressor.state_hash()
    }

    /// Handle force sync (reset state from full gradient)
    pub fn force_sync_layer(&mut self, layer_id: LayerId, gradient: &Tensor) -> Result<()> {
        self.compressor.force_sync_layer(layer_id, gradient)
    }

    // Predictor and Quantizer accessors removed as they are now encapsulated

    /// Get compression statistics
    pub fn stats(&self) -> CompressionStats {
        CompressionStats {
            step: self.step,
            compression_enabled: self.compression_enabled && !self.paused_for_divergence,
            num_drivers: self.mask.num_drivers(),
            num_passengers: self.mask.num_passengers(),
            mask_compression_ratio: self.mask.compression_ratio(),
            quantizer_gamma: 0.0, // Unavailable in generic compressor interface
        }
    }

    /// Get per-layer telemetry
    pub fn layer_telemetry(&self, layer_id: LayerId) -> Option<&LayerTelemetry> {
        self.layer_telemetry.get(&layer_id)
    }

    /// Get all layer telemetry
    pub fn all_layer_telemetry(&self) -> &HashMap<LayerId, LayerTelemetry> {
        &self.layer_telemetry
    }

    /// Get summary telemetry across all layers
    pub fn summary_telemetry(&self) -> SummaryTelemetry {
        let mut total_original = 0u64;
        let mut total_compressed = 0u64;
        let mut total_compressions = 0u64;
        let mut total_passenger_skips = 0u64;
        let mut avg_prediction_error = 0.0f32;
        let mut count = 0;

        for telemetry in self.layer_telemetry.values() {
            total_original += telemetry.total_original_bytes;
            total_compressed += telemetry.total_compressed_bytes;
            total_compressions += telemetry.compression_count;
            total_passenger_skips += telemetry.passenger_skip_count;
            if telemetry.compression_count > 0 {
                avg_prediction_error += telemetry.avg_prediction_error;
                count += 1;
            }
        }

        let overall_ratio = if total_compressed > 0 {
            total_original as f32 / total_compressed as f32
        } else {
            1.0
        };

        let avg_error = if count > 0 {
            avg_prediction_error / count as f32
        } else {
            0.0
        };

        SummaryTelemetry {
            total_original_bytes: total_original,
            total_compressed_bytes: total_compressed,
            overall_compression_ratio: overall_ratio,
            total_compressions,
            total_passenger_skips,
            avg_prediction_error: avg_error,
            num_layers_tracked: self.layer_telemetry.len(),
        }
    }

    // --- Safe Mode methods ---

    /// Enable Safe Mode convergence monitoring
    pub fn enable_safe_mode(&mut self, config: SafeModeConfig) {
        self.safe_mode = Some(SafeMode::new(config));
        tracing::info!(
            "Safe Mode enabled with divergence threshold {:.2}%",
            self.safe_mode
                .as_ref()
                .unwrap()
                .stats()
                .baseline_loss
                .unwrap_or(0.0)
                * 100.0
        );
    }

    /// Report validation loss to Safe Mode
    ///
    /// Call this every N steps with the validation loss.
    /// Returns the action taken (Continue, TriggerFallback, RecoveryComplete).
    pub fn report_validation_loss(&mut self, loss: f32) -> SafeModeAction {
        let Some(ref mut safe_mode) = self.safe_mode else {
            return SafeModeAction::NoCheck;
        };

        if !safe_mode.should_check(self.step) {
            return SafeModeAction::NoCheck;
        }

        let action = safe_mode.check(self.step, loss);

        match action {
            SafeModeAction::TriggerFallback => {
                self.paused_for_divergence = true;
                tracing::warn!("Safe Mode: compression paused due to divergence");
            }
            SafeModeAction::RecoveryComplete => {
                self.paused_for_divergence = false;
                tracing::info!("Safe Mode: compression resumed after recovery");
            }
            _ => {}
        }

        action
    }

    /// Check if compression is paused due to divergence
    pub fn is_paused(&self) -> bool {
        self.paused_for_divergence
    }

    /// Get Safe Mode statistics
    pub fn safe_mode_stats(&self) -> Option<SafeModeStats> {
        self.safe_mode.as_ref().map(|sm| sm.stats())
    }

    /// Check if Safe Mode is enabled
    pub fn has_safe_mode(&self) -> bool {
        self.safe_mode.is_some()
    }
}

/// Compression statistics
#[derive(Debug, Clone)]
pub struct CompressionStats {
    pub step: usize,
    pub compression_enabled: bool,
    pub num_drivers: usize,
    pub num_passengers: usize,
    pub mask_compression_ratio: f32,
    pub quantizer_gamma: f32,
}

/// Summary telemetry across all layers
#[derive(Debug, Clone)]
pub struct SummaryTelemetry {
    pub total_original_bytes: u64,
    pub total_compressed_bytes: u64,
    pub overall_compression_ratio: f32,
    pub total_compressions: u64,
    pub total_passenger_skips: u64,
    pub avg_prediction_error: f32,
    pub num_layers_tracked: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tensor(vals: &[f32]) -> Tensor {
        Tensor::new(vals.to_vec(), vec![vals.len()])
    }

    #[test]
    fn test_warmup_passthrough() {
        let mut mask = TopologyMask::new();
        mask.add_driver(0);

        let config = NangilaConfig {
            warmup_steps: 10,
            shadow_run_steps: 5,
            ..Default::default()
        };

        let mut state = NangilaState::new(config, mask);

        // During warmup, should passthrough
        let grad = make_tensor(&[1.0, 2.0, 3.0]);
        let result = state.compress(0, &grad).unwrap();
        assert!(matches!(result, CompressionResult::Passthrough(_)));
    }

    #[test]
    fn test_compression_after_warmup() {
        let mut mask = TopologyMask::new();
        mask.add_driver(0);

        let config = NangilaConfig {
            warmup_steps: 2,
            shadow_run_steps: 2,
            ..Default::default()
        };

        let mut state = NangilaState::new(config, mask);

        // Build up history during warmup
        for i in 0..5 {
            let grad = make_tensor(&[1.0 + i as f32 * 0.1, 2.0, 3.0]);
            let _ = state.compress(0, &grad);
            state.update_state(0, grad);
            state.step();
        }

        // Should now be compressing
        assert!(state.is_compression_enabled());

        let grad = make_tensor(&[1.5, 2.0, 3.0]);
        let result = state.compress(0, &grad).unwrap();
        assert!(matches!(result, CompressionResult::Driver(_)));
    }

    #[test]
    fn test_passenger_skipped() {
        let mut mask = TopologyMask::new();
        mask.add_driver(0);
        mask.add_passenger(1, 0, 0.5, 0.0);

        let mut config = NangilaConfig::default();
        config.warmup_steps = 0;
        config.shadow_run_steps = 0;

        let mut state = NangilaState::new(config, mask);

        // Build minimal history
        state.update_state(0, make_tensor(&[1.0, 2.0]));
        state.update_state(0, make_tensor(&[1.1, 2.1]));
        state.step();

        let grad = make_tensor(&[0.5, 1.0]);
        let result = state.compress(1, &grad).unwrap();
        assert!(matches!(result, CompressionResult::Passenger));
    }
}
