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
    /// Large values that require clamping may indicate gradient explosions.
    ///
    /// If gradient clipping is needed, use `from_f32_clipped` instead.
    #[inline]
    pub fn from_f32(f: f32) -> Self {
        // Track saturation for diagnostics (using static atomic counter)
        static SATURATION_COUNT: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(0);
        static LAST_LOG_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

        // Clamp to valid range before conversion
        let clamped = f.clamp(-256.0, 255.99999988);

        // Check for saturation (value was clamped)
        if f != clamped && !f.is_nan() {
            let count = SATURATION_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;

            // Log every 1000 saturations to avoid spam
            let last_log = LAST_LOG_COUNT.load(std::sync::atomic::Ordering::Relaxed);
            if count >= last_log + 1000 {
                LAST_LOG_COUNT.store(count, std::sync::atomic::Ordering::Relaxed);
                tracing::warn!(
                    "Q8.23 saturation: {} values clamped to range [-256, 256) (last value: {})",
                    count,
                    f
                );
            }
        }

        let scaled = clamped * SCALE_F32;
        Self(scaled.round() as i32)
    }

    /// Convert from f32 with gradient clipping
    ///
    /// Applies percentile-based clipping before conversion to prevent
    /// gradient explosions from saturating the fixed-point representation.
    #[inline]
    pub fn from_f32_clipped(f: f32, clip_value: f32) -> Self {
        let clipped = f.clamp(-clip_value, clip_value);
        Self::from_f32(clipped)
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

    /// Create from f32 slice with gradient clipping
    pub fn from_f32_slice_clipped(slice: &[f32], clip_value: f32) -> Self {
        Self {
            data: slice
                .iter()
                .map(|&f| Q8_23::from_f32_clipped(f, clip_value))
                .collect(),
        }
    }

    /// Create from f32 slice with automatic percentile-based clipping
    /// Clips at the 99.9th percentile to handle outliers
    pub fn from_f32_slice_auto_clip(slice: &[f32]) -> Self {
        // Compute 99.9th percentile for clipping
        let mut abs_vals: Vec<f32> = slice.iter().map(|x| x.abs()).collect();
        abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let p999_idx = ((abs_vals.len() as f32) * 0.999) as usize;
        let clip_value = abs_vals.get(p999_idx).copied().unwrap_or(256.0).min(256.0);

        Self::from_f32_slice_clipped(slice, clip_value)
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

    /// Get a slice of the buffer
    pub fn slice(&self, start: usize, end: usize) -> &[Q8_23] {
        let start = start.min(self.data.len());
        let end = end.min(self.data.len());
        if start >= end {
            &[]
        } else {
            &self.data[start..end]
        }
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
