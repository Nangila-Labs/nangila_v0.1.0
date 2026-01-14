//! Data type conversion utilities for FP16/BF16 support
//!
//! Provides functions to convert between FP32, FP16, and BF16 formats.

use half::{bf16, f16};

/// Convert raw FP16 bytes to FP32 vector
pub fn f16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            f16::from_bits(bits).to_f32()
        })
        .collect()
}

/// Convert raw BF16 bytes to FP32 vector
pub fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            bf16::from_bits(bits).to_f32()
        })
        .collect()
}

/// Convert FP32 vector to raw FP16 bytes
pub fn f32_to_f16(data: &[f32]) -> Vec<u8> {
    data.iter()
        .flat_map(|&v| f16::from_f32(v).to_bits().to_le_bytes())
        .collect()
}

/// Convert FP32 vector to raw BF16 bytes
pub fn f32_to_bf16(data: &[f32]) -> Vec<u8> {
    data.iter()
        .flat_map(|&v| bf16::from_f32(v).to_bits().to_le_bytes())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_roundtrip() {
        let original = vec![1.0f32, -2.5, 3.14159, 0.0, 100.0];
        let f16_bytes = f32_to_f16(&original);
        let recovered = f16_to_f32(&f16_bytes);
        
        // FP16 has limited precision, check within tolerance
        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 0.01, "FP16 roundtrip failed: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_bf16_roundtrip() {
        let original = vec![1.0f32, -2.5, 3.14159, 0.0, 100.0];
        let bf16_bytes = f32_to_bf16(&original);
        let recovered = bf16_to_f32(&bf16_bytes);
        
        // BF16 has even more limited precision (7 bits mantissa)
        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 0.1, "BF16 roundtrip failed: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_f16_special_values() {
        let values = vec![f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0];
        let f16_bytes = f32_to_f16(&values);
        let recovered = f16_to_f32(&f16_bytes);
        
        assert!(recovered[0].is_infinite() && recovered[0] > 0.0);
        assert!(recovered[1].is_infinite() && recovered[1] < 0.0);
        assert_eq!(recovered[2], 0.0);
        assert_eq!(recovered[3], 0.0);
    }
}
