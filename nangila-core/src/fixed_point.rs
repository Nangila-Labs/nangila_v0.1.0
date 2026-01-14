//! Q8.23 Fixed-Point Arithmetic for Deterministic Computation
//!
//! This module provides bit-exact arithmetic across different GPU architectures.
//! The Q8.23 format uses:
//! - 1 sign bit
//! - 8 integer bits
//! - 23 fractional bits
//!
//! Range: [-256.0, 255.99999988]
//! Precision: ~1.19e-7 (2^-23)

use std::ops::{Add, Mul, Sub};

/// Scaling factor for Q8.23: 2^23 = 8388608
const SCALE: i32 = 1 << 23;
const SCALE_F32: f32 = 8388608.0;

/// Q8.23 fixed-point number
///
/// Internally stored as i32 where the value represents x * 2^23
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct Q8_23(i32);

impl Q8_23 {
    /// Zero constant
    pub const ZERO: Self = Self(0);

    /// One constant
    pub const ONE: Self = Self(SCALE);

    /// Maximum representable value (~255.999999)
    pub const MAX: Self = Self(i32::MAX);

    /// Minimum representable value (~-256.0)
    pub const MIN: Self = Self(i32::MIN);

    /// Create a Q8.23 from raw bits
    #[inline]
    pub const fn from_bits(bits: i32) -> Self {
        Self(bits)
    }

    /// Get the raw bits
    #[inline]
    pub const fn to_bits(self) -> i32 {
        self.0
    }

    /// Convert from f32 to Q8.23
    ///
    /// Values outside the representable range are clamped.
    #[inline]
    pub fn from_f32(f: f32) -> Self {
        // Clamp to valid range before conversion
        let clamped = f.clamp(-256.0, 255.99999988);
        let scaled = clamped * SCALE_F32;
        Self(scaled.round() as i32)
    }

    /// Convert from Q8.23 to f32
    #[inline]
    pub fn to_f32(self) -> f32 {
        self.0 as f32 / SCALE_F32
    }

    /// Saturating addition (clamps on overflow)
    #[inline]
    pub fn saturating_add(self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0))
    }

    /// Saturating subtraction (clamps on overflow)
    #[inline]
    pub fn saturating_sub(self, other: Self) -> Self {
        Self(self.0.saturating_sub(other.0))
    }

    /// Fixed-point multiplication
    ///
    /// Uses i64 intermediate to prevent overflow, then shifts back.
    #[inline]
    pub fn mul(self, other: Self) -> Self {
        // Use i64 to prevent overflow during multiplication
        let result = (self.0 as i64 * other.0 as i64) >> 23;
        // Saturate to i32 range
        Self(result.clamp(i32::MIN as i64, i32::MAX as i64) as i32)
    }

    /// Multiply by a scalar (f32 coefficient like momentum)
    ///
    /// Converts scalar to Q8.23 first for determinism.
    #[inline]
    pub fn mul_scalar(self, scalar: f32) -> Self {
        self.mul(Self::from_f32(scalar))
    }

    /// Fused multiply-add: self + a * b
    ///
    /// More efficient than separate mul and add.
    #[inline]
    pub fn fma(self, a: Self, b: Self) -> Self {
        let product = (a.0 as i64 * b.0 as i64) >> 23;
        let sum = self.0 as i64 + product;
        Self(sum.clamp(i32::MIN as i64, i32::MAX as i64) as i32)
    }

    /// Absolute value
    #[inline]
    pub fn abs(self) -> Self {
        Self(self.0.saturating_abs())
    }

    /// Check if value is negative
    #[inline]
    pub fn is_negative(self) -> bool {
        self.0 < 0
    }
}

impl Add for Q8_23 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        self.saturating_add(rhs)
    }
}

impl Sub for Q8_23 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        self.saturating_sub(rhs)
    }
}

impl Mul for Q8_23 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Q8_23::mul(self, rhs)
    }
}

/// A vector of Q8.23 values for gradient storage
#[derive(Clone, Debug, Default)]
pub struct FixedPointBuffer {
    data: Vec<Q8_23>,
}

impl FixedPointBuffer {
    /// Create a new buffer with given capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
        }
    }

    /// Create from f32 slice
    pub fn from_f32_slice(slice: &[f32]) -> Self {
        Self {
            data: slice.iter().map(|&f| Q8_23::from_f32(f)).collect(),
        }
    }

    /// Convert to f32 vector
    pub fn to_f32_vec(&self) -> Vec<f32> {
        self.data.iter().map(|q| q.to_f32()).collect()
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get raw data slice
    pub fn as_slice(&self) -> &[Q8_23] {
        &self.data
    }

    /// Get mutable raw data slice
    pub fn as_mut_slice(&mut self) -> &mut [Q8_23] {
        &mut self.data
    }

    /// Element-wise addition
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.len(), other.len());
        Self {
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| *a + *b)
                .collect(),
        }
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.len(), other.len());
        Self {
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| *a - *b)
                .collect(),
        }
    }

    /// Scale by a constant
    pub fn scale(&self, factor: f32) -> Self {
        let factor_q = Q8_23::from_f32(factor);
        Self {
            data: self.data.iter().map(|a| *a * factor_q).collect(),
        }
    }

    /// Fused multiply-add: self + other * scale
    pub fn fma(&self, other: &Self, scale: f32) -> Self {
        assert_eq!(self.len(), other.len());
        let scale_q = Q8_23::from_f32(scale);
        Self {
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| a.fma(*b, scale_q))
                .collect(),
        }
    }

    /// Compute hash for deterministic verification
    pub fn hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for val in &self.data {
            val.0.hash(&mut hasher);
        }
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_roundtrip() {
        let values = [0.0f32, 1.0, -1.0, 0.5, -0.5, 0.9, 127.5, -128.0];

        for &v in &values {
            let q = Q8_23::from_f32(v);
            let recovered = q.to_f32();
            let error = (v - recovered).abs();
            assert!(
                error < 1.2e-7,
                "Roundtrip error too large for {}: got {}, error {}",
                v,
                recovered,
                error
            );
        }
    }

    #[test]
    fn test_addition() {
        let a = Q8_23::from_f32(1.5);
        let b = Q8_23::from_f32(2.5);
        let c = a + b;
        assert!((c.to_f32() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_subtraction() {
        let a = Q8_23::from_f32(5.0);
        let b = Q8_23::from_f32(3.0);
        let c = a - b;
        assert!((c.to_f32() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_multiplication() {
        let a = Q8_23::from_f32(2.0);
        let b = Q8_23::from_f32(3.0);
        let c = a * b;
        assert!((c.to_f32() - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_momentum_calculation() {
        // Simulate: ĝ = g_t + μ * (g_t - g_{t-1})
        let g_t = Q8_23::from_f32(1.0);
        let g_t_minus_1 = Q8_23::from_f32(0.8);
        let mu = Q8_23::from_f32(0.9);

        let delta = g_t - g_t_minus_1; // 0.2
        let momentum_term = delta * mu; // 0.18
        let prediction = g_t + momentum_term; // 1.18

        let expected = 1.0 + 0.9 * (1.0 - 0.8);
        assert!(
            (prediction.to_f32() - expected).abs() < 1e-5,
            "Expected {}, got {}",
            expected,
            prediction.to_f32()
        );
    }

    #[test]
    fn test_determinism() {
        // Same inputs must produce identical outputs
        let inputs: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.001).sin()).collect();

        let buf1 = FixedPointBuffer::from_f32_slice(&inputs);
        let buf2 = FixedPointBuffer::from_f32_slice(&inputs);

        // Perform same operations
        let result1 = buf1.scale(0.9).add(&buf1);
        let result2 = buf2.scale(0.9).add(&buf2);

        assert_eq!(
            result1.hash(),
            result2.hash(),
            "Determinism violated: hashes differ"
        );
    }

    #[test]
    fn test_saturation() {
        // Test overflow handling
        let max = Q8_23::from_f32(255.0);
        let big = Q8_23::from_f32(10.0);
        let result = max + big;
        // Should saturate, not wrap
        assert!(result.to_f32() > 200.0);
    }
}
