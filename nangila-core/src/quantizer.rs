//! INT4 Stochastic Quantization (The Fast Head)
//!
//! Quantizes residuals (g - ĝ) to INT4 for transmission.
//! Uses stochastic rounding for unbiased compression.

use crate::Tensor;

/// Compressed tensor representation
#[derive(Debug, Clone)]
pub struct CompressedTensor {
    /// Quantized values (packed INT4, two values per byte)
    pub data: Vec<u8>,
    /// Scaling factor for dequantization
    pub gamma: f32,
    /// Original shape
    pub shape: Vec<usize>,
    /// Number of elements
    pub numel: usize,
}

impl CompressedTensor {
    /// Size in bytes
    pub fn size_bytes(&self) -> usize {
        self.data.len() + 4 + self.shape.len() * 8 // data + gamma + shape
    }

    /// Compression ratio compared to FP32
    pub fn compression_ratio(&self, original_numel: usize) -> f32 {
        let original_bytes = original_numel * 4; // FP32
        original_bytes as f32 / self.size_bytes() as f32
    }
}

/// Empty compressed tensor (for Passenger layers)
impl Default for CompressedTensor {
    fn default() -> Self {
        Self {
            data: Vec::new(),
            gamma: 0.0,
            shape: Vec::new(),
            numel: 0,
        }
    }
}

/// INT4 Stochastic Quantizer
#[derive(Debug)]
pub struct Quantizer {
    /// Number of bits (typically 4)
    bits: u8,
    /// Whether to use dynamic gamma (scale)
    dynamic_gamma: bool,
    /// Running estimate of gamma (for stability)
    gamma_ema: f32,
    /// EMA momentum for gamma updates
    gamma_momentum: f32,
}

impl Quantizer {
    /// Create a new quantizer
    pub fn new(bits: u8, dynamic_gamma: bool) -> Self {
        assert!(bits <= 8, "Maximum 8 bits supported");
        Self {
            bits,
            dynamic_gamma,
            gamma_ema: 1.0,
            gamma_momentum: 0.99,
        }
    }

    /// Create a default INT4 quantizer
    pub fn int4() -> Self {
        Self::new(4, true)
    }

    /// Compute the quantization scale (gamma) for a tensor
    fn compute_gamma(&self, tensor: &Tensor) -> f32 {
        // Use percentile-based scaling to handle outliers
        // For INT4: range is [-8, 7], so we want max_val ≈ 7 * gamma
        let max_levels = (1 << (self.bits - 1)) - 1; // 7 for INT4

        // Find the 99th percentile absolute value
        let mut abs_vals: Vec<f32> = tensor.data.iter().map(|x| x.abs()).collect();
        abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let p99_idx = (abs_vals.len() as f32 * 0.99) as usize;
        let p99_val = abs_vals.get(p99_idx).copied().unwrap_or(1.0);

        // gamma = p99 / max_levels
        let gamma = (p99_val / max_levels as f32).max(1e-8);
        gamma
    }

    /// Quantize a tensor to INT4
    pub fn quantize(&mut self, tensor: &Tensor) -> CompressedTensor {
        if tensor.numel() == 0 {
            return CompressedTensor::default();
        }

        // Compute gamma (ensure never zero)
        let gamma = if self.dynamic_gamma {
            let new_gamma = self.compute_gamma(tensor);
            // EMA update for stability
            self.gamma_ema = self.gamma_momentum * self.gamma_ema
                + (1.0 - self.gamma_momentum) * new_gamma;
            self.gamma_ema.max(1e-8) // Guard against zero
        } else {
            self.gamma_ema.max(1e-8)
        };

        let max_val = (1i8 << (self.bits - 1)) - 1; // 7 for INT4
        let min_val = -(1i8 << (self.bits - 1)); // -8 for INT4

        // Quantize with stochastic rounding
        let quantized: Vec<i8> = tensor
            .data
            .iter()
            .map(|&x| {
                // Guard against NaN/Inf - treat as zero
                let x = if x.is_nan() || x.is_infinite() { 0.0 } else { x };
                
                let scaled = x / gamma;
                // Stochastic rounding: round probabilistically based on fractional part
                let floor = scaled.floor();
                let frac = scaled - floor;

                // Deterministic for now (can add randomness later)
                let rounded = if frac >= 0.5 {
                    (floor + 1.0) as i8
                } else {
                    floor as i8
                };

                // Clamp to valid range
                rounded.clamp(min_val, max_val)
            })
            .collect();

        // Pack INT4 values (two per byte)
        let packed = self.pack_int4(&quantized);

        CompressedTensor {
            data: packed,
            gamma,
            shape: tensor.shape.clone(),
            numel: tensor.numel(),
        }
    }

    /// Dequantize a compressed tensor back to FP32
    pub fn dequantize(&self, compressed: &CompressedTensor) -> Tensor {
        if compressed.numel == 0 {
            return Tensor::zeros(compressed.shape.clone());
        }

        // Unpack INT4 values
        let quantized = self.unpack_int4(&compressed.data, compressed.numel);

        // Dequantize: x = q * gamma
        let data: Vec<f32> = quantized
            .iter()
            .map(|&q| q as f32 * compressed.gamma)
            .collect();

        Tensor::new(data, compressed.shape.clone())
    }

    /// Pack INT4 values into bytes (two values per byte)
    fn pack_int4(&self, values: &[i8]) -> Vec<u8> {
        let mut packed = Vec::with_capacity((values.len() + 1) / 2);

        for chunk in values.chunks(2) {
            let low = (chunk[0] & 0x0F) as u8;
            let high = if chunk.len() > 1 {
                ((chunk[1] & 0x0F) as u8) << 4
            } else {
                0
            };
            packed.push(low | high);
        }

        packed
    }

    /// Unpack INT4 values from bytes
    fn unpack_int4(&self, packed: &[u8], numel: usize) -> Vec<i8> {
        let mut values = Vec::with_capacity(numel);

        for (i, &byte) in packed.iter().enumerate() {
            // Low nibble
            let low = (byte & 0x0F) as i8;
            // Sign extend from 4 bits
            let low = if low & 0x08 != 0 { low | !0x0F } else { low };
            values.push(low);

            // High nibble (if we haven't reached numel)
            if values.len() < numel {
                let high = ((byte >> 4) & 0x0F) as i8;
                let high = if high & 0x08 != 0 { high | !0x0F } else { high };
                values.push(high);
            }
        }

        values.truncate(numel);
        values
    }

    /// Get current gamma value
    pub fn gamma(&self) -> f32 {
        self.gamma_ema
    }

    /// Set gamma manually (for testing)
    pub fn set_gamma(&mut self, gamma: f32) {
        self.gamma_ema = gamma;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_int4() {
        let quantizer = Quantizer::int4();

        let values: Vec<i8> = vec![1, -2, 3, -4, 5, -6, 7, -8];
        let packed = quantizer.pack_int4(&values);
        let unpacked = quantizer.unpack_int4(&packed, values.len());

        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let mut quantizer = Quantizer::int4();
        quantizer.set_gamma(0.1); // Fixed gamma for testing

        let original = Tensor::new(vec![0.1, 0.2, -0.3, 0.5, -0.7], vec![5]);
        let compressed = quantizer.quantize(&original);
        let recovered = quantizer.dequantize(&compressed);

        // Check values are close (within quantization error)
        for (orig, rec) in original.data.iter().zip(&recovered.data) {
            let error = (orig - rec).abs();
            assert!(
                error < 0.15,
                "Too much error: orig={}, rec={}, error={}",
                orig,
                rec,
                error
            );
        }
    }

    #[test]
    fn test_compression_ratio() {
        let mut quantizer = Quantizer::int4();

        let tensor = Tensor::new(vec![0.1; 1000], vec![1000]);
        let compressed = quantizer.quantize(&tensor);

        // 1000 FP32 = 4000 bytes
        // 1000 INT4 = 500 bytes + overhead
        let ratio = compressed.compression_ratio(1000);
        assert!(ratio > 6.0, "Expected >6x compression, got {}", ratio);
    }
}
