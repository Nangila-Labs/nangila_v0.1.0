//! NangilaState: Main state container for compression/decompression
//!
//! This is the primary interface for the Nangila compression system,
//! coordinating the Predictor, Quantizer, Mask, and Reconstructor.

use crate::safe_mode::{SafeMode, SafeModeAction, SafeModeConfig, SafeModeStats};
use crate::{
    CompressedTensor, LayerId, NangilaConfig, Predictor, Quantizer, Reconstructor, Result,
    Tensor, TopologyMask,
};
use std::collections::HashMap;

/// Main state container for Nangila compression
#[derive(Debug)]
pub struct NangilaState {
    /// Configuration
    pub config: NangilaConfig,
    /// Topology mask (Driver/Passenger mapping)
    mask: TopologyMask,
    /// Gradient predictor
    predictor: Predictor,
    /// Quantizer for residuals
    quantizer: Quantizer,
    /// Reconstructor for decompression
    reconstructor: Reconstructor,
    /// Current training step
    step: usize,
    /// Whether compression is currently enabled
    compression_enabled: bool,
    /// Safe Mode convergence monitor (optional)
    safe_mode: Option<SafeMode>,
    /// Whether compression is paused due to divergence
    paused_for_divergence: bool,
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
        let quantizer = Quantizer::new(config.quantize_bits, config.dynamic_gamma);
        let predictor = Predictor::new(
            config.momentum,
            config.warmup_steps + config.shadow_run_steps,
        );
        let reconstructor = Reconstructor::new(Quantizer::new(config.quantize_bits, config.dynamic_gamma));

        Self {
            config,
            mask,
            predictor,
            quantizer,
            reconstructor,
            step: 0,
            compression_enabled: false,
            safe_mode: None,
            paused_for_divergence: false,
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
        if !self.compression_enabled {
            // Still update predictor during shadow-run phase
            if self.step >= self.config.warmup_steps {
                self.predictor.update(layer_id, gradient.clone());
            }
            return Ok(CompressionResult::Passthrough(gradient.clone()));
        }

        // Check layer role
        let role = self.mask.get_role(layer_id)?;

        if role.is_passenger() {
            // Passengers send nothing
            return Ok(CompressionResult::Passenger);
        }

        // Driver: compute residual and quantize
        let prediction = self.predictor.predict(layer_id)?;
        let residual = gradient.sub(&prediction);
        let compressed = self.quantizer.quantize(&residual);

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
            self.reconstructor
                .reconstruct_driver(layer_id, compressed, &self.predictor)?
        } else {
            self.reconstructor.synthesize_passenger(layer_id, &self.mask)?
        };

        Ok(gradient)
    }

    /// Decompress all layers from compressed Driver data
    pub fn decompress_all(
        &mut self,
        compressed_drivers: &HashMap<LayerId, CompressedTensor>,
    ) -> Result<HashMap<LayerId, Tensor>> {
        self.reconstructor
            .reconstruct_all(compressed_drivers, &self.mask, &self.predictor)
    }

    /// Update predictor state with actual gradient (post-All-Reduce)
    ///
    /// IMPORTANT: Call this with the reconstructed gradient to maintain
    /// bit-perfect consensus between sender and receiver predictors.
    pub fn update_state(&mut self, layer_id: LayerId, gradient: Tensor) {
        self.predictor.update(layer_id, gradient);
    }

    /// Advance to the next training step
    pub fn step(&mut self) {
        self.step += 1;
        self.predictor.step();

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
        self.compression_enabled
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

    /// Get a reference to the predictor
    pub fn predictor(&self) -> &Predictor {
        &self.predictor
    }

    /// Get a mutable reference to the predictor (for desync recovery)
    pub fn predictor_mut(&mut self) -> &mut Predictor {
        &mut self.predictor
    }

    /// Get compression statistics
    pub fn stats(&self) -> CompressionStats {
        CompressionStats {
            step: self.step,
            compression_enabled: self.compression_enabled && !self.paused_for_divergence,
            num_drivers: self.mask.num_drivers(),
            num_passengers: self.mask.num_passengers(),
            mask_compression_ratio: self.mask.compression_ratio(),
            quantizer_gamma: self.quantizer.gamma(),
        }
    }

    // --- Safe Mode methods ---

    /// Enable Safe Mode convergence monitoring
    pub fn enable_safe_mode(&mut self, config: SafeModeConfig) {
        self.safe_mode = Some(SafeMode::new(config));
        tracing::info!("Safe Mode enabled with divergence threshold {:.2}%",
            self.safe_mode.as_ref().unwrap().stats().baseline_loss.unwrap_or(0.0) * 100.0
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
