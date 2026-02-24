//! Configuration and hyperparameters for Nangila.

use thiserror::Error;

/// Configuration validation errors
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("momentum must be in range [0.0, 1.0], got {0}")]
    InvalidMomentum(f32),

    #[error("sculptor_threshold must be in range [0.0, 1.0], got {0}")]
    InvalidThreshold(f32),

    #[error("quantize_bits must be in range [1, 8], got {0}")]
    InvalidBits(u8),

    #[error("monitor_sample_fraction must be in range [0.0, 1.0], got {0}")]
    InvalidSampleFraction(f32),

    #[error("promotion_threshold must be positive, got {0}")]
    InvalidPromotionThreshold(f32),

    #[error("dgc_sparsity must be in range [0.0, 1.0], got {0}")]
    InvalidSparsity(f32),

    #[error("power_sgd_rank must be positive, got {0}")]
    InvalidRank(usize),
}

/// Type of compressor to use
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompressorType {
    #[default]
    PredictionResidual,
    DGC,
    PowerSGD,
    // Placeholders for future codecs
}

/// Main configuration for Nangila compression
#[derive(Debug, Clone)]
pub struct NangilaConfig {
    /// Momentum coefficient for predictor (typically 0.9)
    pub momentum: f32,

    /// Sculptor correlation threshold τ for Passenger detection
    /// Higher = more conservative (fewer Passengers)
    pub sculptor_threshold: f32,

    /// Number of warmup steps before enabling compression
    /// (LR warmup + predictor warmup)
    pub warmup_steps: usize,

    /// Shadow-run steps after LR warmup (predictor learns dynamics)
    pub shadow_run_steps: usize,

    /// Quantization bit width (typically 4 for INT4)
    pub quantize_bits: u8,

    /// Whether to use dynamic gamma (quantization scale)
    pub dynamic_gamma: bool,

    /// Monitoring interval for drift detection (steps)
    pub monitor_interval: usize,

    /// Fraction of Passengers to sample for drift monitoring
    pub monitor_sample_fraction: f32,

    /// Error threshold for promoting Passenger to Driver
    pub promotion_threshold: f32,

    /// Type of compressor to use
    pub compressor_type: CompressorType,

    /// DGC: Sparsity ratio (e.g. 0.999 means keep top 0.1%)
    pub dgc_sparsity: f32,

    /// PowerSGD: Rank for matrix factorization (e.g. 1 or 2)
    pub power_sgd_rank: usize,
}

impl Default for NangilaConfig {
    fn default() -> Self {
        Self {
            momentum: 0.9,
            sculptor_threshold: 0.95,
            warmup_steps: 1000,
            shadow_run_steps: 100,
            quantize_bits: 4,
            dynamic_gamma: true,
            monitor_interval: 1000,
            monitor_sample_fraction: 0.10,
            promotion_threshold: 0.15,
            compressor_type: CompressorType::default(),
            dgc_sparsity: 0.999,
            power_sgd_rank: 1,
        }
    }
}

impl NangilaConfig {
    /// Create a new validated configuration
    pub fn new(
        momentum: f32,
        sculptor_threshold: f32,
        warmup_steps: usize,
        shadow_run_steps: usize,
        quantize_bits: u8,
    ) -> Result<Self, ConfigError> {
        // Validate inputs
        if !(0.0..=1.0).contains(&momentum) {
            return Err(ConfigError::InvalidMomentum(momentum));
        }
        if !(0.0..=1.0).contains(&sculptor_threshold) {
            return Err(ConfigError::InvalidThreshold(sculptor_threshold));
        }
        if !(1..=8).contains(&quantize_bits) {
            return Err(ConfigError::InvalidBits(quantize_bits));
        }

        Ok(Self {
            momentum,
            sculptor_threshold,
            warmup_steps,
            shadow_run_steps,
            quantize_bits,
            dynamic_gamma: true,
            monitor_interval: 1000,
            monitor_sample_fraction: 0.10,
            promotion_threshold: 0.15,
            compressor_type: CompressorType::default(),
            dgc_sparsity: 0.999,
            power_sgd_rank: 1,
        })
    }

    /// Create a conservative configuration (safer, less compression)
    pub fn conservative() -> Self {
        Self {
            sculptor_threshold: 0.97,
            monitor_interval: 500,
            dgc_sparsity: 0.99,
            power_sgd_rank: 2,
            ..Default::default()
        }
    }

    /// Create an aggressive configuration (more compression, needs monitoring)
    pub fn aggressive() -> Self {
        Self {
            sculptor_threshold: 0.90,
            monitor_interval: 1000,
            dgc_sparsity: 0.999,
            power_sgd_rank: 1,
            ..Default::default()
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), ConfigError> {
        if !(0.0..=1.0).contains(&self.momentum) {
            return Err(ConfigError::InvalidMomentum(self.momentum));
        }
        if !(0.0..=1.0).contains(&self.sculptor_threshold) {
            return Err(ConfigError::InvalidThreshold(self.sculptor_threshold));
        }
        if !(1..=8).contains(&self.quantize_bits) {
            return Err(ConfigError::InvalidBits(self.quantize_bits));
        }
        if !(0.0..=1.0).contains(&self.monitor_sample_fraction) {
            return Err(ConfigError::InvalidSampleFraction(
                self.monitor_sample_fraction,
            ));
        }
        if self.promotion_threshold <= 0.0 {
            return Err(ConfigError::InvalidPromotionThreshold(
                self.promotion_threshold,
            ));
        }
        if !(0.0..=1.0).contains(&self.dgc_sparsity) {
            return Err(ConfigError::InvalidSparsity(self.dgc_sparsity));
        }
        if self.power_sgd_rank == 0 {
            return Err(ConfigError::InvalidRank(self.power_sgd_rank));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_config() {
        let config = NangilaConfig::new(0.9, 0.95, 1000, 100, 4);
        assert!(config.is_ok());
    }

    #[test]
    fn test_invalid_momentum() {
        let config = NangilaConfig::new(1.5, 0.95, 1000, 100, 4);
        assert!(matches!(config, Err(ConfigError::InvalidMomentum(_))));
    }

    #[test]
    fn test_invalid_bits() {
        let config = NangilaConfig::new(0.9, 0.95, 1000, 100, 16);
        assert!(matches!(config, Err(ConfigError::InvalidBits(_))));
    }
}
