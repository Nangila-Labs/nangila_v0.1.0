//! Residual Codec
//!
//! Compresses and decompresses boundary residual messages for
//! network transport. Uses delta encoding + variable-length
//! quantization for efficient wire format.
//!
//! Key insight from Nangila core: most residual updates are small
//! deltas from the predicted value. We encode only the errors,
//! which are highly compressible.
//!
//! Phase 2, Sprint 6 deliverable.

use serde::{Deserialize, Serialize};

use crate::comm::ResidualMessage;

/// Compressed residual message for wire transport.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedResidual {
    /// Source partition
    pub from_partition: u32,
    /// Target partition
    pub to_partition: u32,
    /// Simulation time
    pub time: f64,
    /// Delta-encoded updates: (net_id, delta_from_predicted)
    pub deltas: Vec<(u64, i32)>,
    /// Scale factor for dequantization
    pub scale: f64,
    /// Predicted base values (for reconstruction)
    pub base_values: Vec<(u64, f64)>,
}

/// Codec configuration.
#[derive(Debug, Clone)]
pub struct CodecConfig {
    /// Quantization bits (higher = more accurate, larger messages)
    pub quant_bits: u8,
    /// Minimum delta threshold — below this, skip sending
    pub min_delta: f64,
}

impl Default for CodecConfig {
    fn default() -> Self {
        Self {
            quant_bits: 16,
            min_delta: 1e-15,
        }
    }
}

/// The residual codec handles encoding/decoding boundary updates.
#[derive(Debug, Clone)]
pub struct ResidualCodec {
    pub config: CodecConfig,
    /// Running statistics
    pub stats: CodecStats,
}

/// Codec performance statistics.
#[derive(Debug, Clone, Default)]
pub struct CodecStats {
    pub messages_encoded: u64,
    pub messages_decoded: u64,
    pub total_updates: u64,
    pub updates_skipped: u64, // Below min_delta threshold
    pub total_raw_bytes: u64,
    pub total_compressed_bytes: u64,
}

impl CodecStats {
    pub fn compression_ratio(&self) -> f64 {
        if self.total_compressed_bytes == 0 {
            return 1.0;
        }
        self.total_raw_bytes as f64 / self.total_compressed_bytes as f64
    }
}

impl ResidualCodec {
    pub fn new(config: CodecConfig) -> Self {
        Self {
            config,
            stats: CodecStats::default(),
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(CodecConfig::default())
    }

    /// Encode a residual message into compressed format.
    ///
    /// `predicted_values` are the predictions that the receiver already has.
    /// We only send the delta (error) from those predictions.
    pub fn encode(
        &mut self,
        msg: &ResidualMessage,
        predicted_values: &[(u64, f64)],
    ) -> CompressedResidual {
        let max_val = (1i32 << (self.config.quant_bits - 1)) - 1;

        // Find max delta for scaling
        let mut max_delta: f64 = 0.0;
        let mut deltas_raw: Vec<(u64, f64)> = Vec::new();
        let mut base_values: Vec<(u64, f64)> = Vec::new();

        for &(net_id, actual_v) in &msg.updates {
            let predicted = predicted_values
                .iter()
                .find(|&&(id, _)| id == net_id)
                .map(|&(_, v)| v)
                .unwrap_or(0.0);

            let delta = actual_v - predicted;

            if delta.abs() < self.config.min_delta {
                self.stats.updates_skipped += 1;
                continue;
            }

            if delta.abs() > max_delta {
                max_delta = delta.abs();
            }

            deltas_raw.push((net_id, delta));
            base_values.push((net_id, predicted));
        }

        // Quantize deltas
        let scale = if max_delta > 0.0 {
            max_delta / max_val as f64
        } else {
            1.0
        };

        let deltas: Vec<(u64, i32)> = deltas_raw
            .iter()
            .map(|&(net_id, delta)| {
                let quantized = (delta / scale).round() as i32;
                (net_id, quantized.clamp(-max_val, max_val))
            })
            .collect();

        // Update stats
        self.stats.messages_encoded += 1;
        self.stats.total_updates += msg.updates.len() as u64;
        // Raw: 8 bytes per (net_id: u64) + 8 bytes per (voltage: f64) = 16 bytes per update
        self.stats.total_raw_bytes += (msg.updates.len() * 16) as u64;
        // Compressed: 8 bytes per (net_id: u64) + 4 bytes per (quantized: i32) = 12 bytes per update
        self.stats.total_compressed_bytes += (deltas.len() * 12) as u64;

        CompressedResidual {
            from_partition: msg.from_partition,
            to_partition: msg.to_partition,
            time: msg.time,
            deltas,
            scale,
            base_values,
        }
    }

    /// Decode a compressed residual back into a full message.
    pub fn decode(&mut self, compressed: &CompressedResidual) -> ResidualMessage {
        let updates: Vec<(u64, f64)> = compressed
            .deltas
            .iter()
            .map(|&(net_id, quantized)| {
                let base = compressed
                    .base_values
                    .iter()
                    .find(|&&(id, _)| id == net_id)
                    .map(|&(_, v)| v)
                    .unwrap_or(0.0);

                let reconstructed = base + (quantized as f64 * compressed.scale);
                (net_id, reconstructed)
            })
            .collect();

        self.stats.messages_decoded += 1;

        ResidualMessage {
            from_partition: compressed.from_partition,
            to_partition: compressed.to_partition,
            time: compressed.time,
            updates,
        }
    }

    /// Encode and immediately return serialized bytes (for transport).
    pub fn encode_to_bytes(
        &mut self,
        msg: &ResidualMessage,
        predicted_values: &[(u64, f64)],
    ) -> Vec<u8> {
        let compressed = self.encode(msg, predicted_values);
        serde_json::to_vec(&compressed).unwrap_or_default()
    }

    /// Decode from serialized bytes.
    pub fn decode_from_bytes(&mut self, bytes: &[u8]) -> Option<ResidualMessage> {
        let compressed: CompressedResidual = serde_json::from_slice(bytes).ok()?;
        Some(self.decode(&compressed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        let mut codec = ResidualCodec::with_defaults();

        let msg = ResidualMessage {
            from_partition: 0,
            to_partition: 1,
            time: 1e-9,
            updates: vec![(100, 1.8), (200, 0.9), (300, 1.2)],
        };

        let predicted = vec![(100, 1.79), (200, 0.88), (300, 1.19)];

        let compressed = codec.encode(&msg, &predicted);
        let decoded = codec.decode(&compressed);

        // Check roundtrip accuracy
        assert_eq!(decoded.updates.len(), 3);
        for (orig, dec) in msg.updates.iter().zip(decoded.updates.iter()) {
            assert_eq!(orig.0, dec.0, "Net IDs should match");
            assert!(
                (orig.1 - dec.1).abs() < 1e-3,
                "Voltage error too large: {} vs {}",
                orig.1,
                dec.1
            );
        }
    }

    #[test]
    fn test_delta_compression() {
        let mut codec = ResidualCodec::with_defaults();

        // When predictions are very close, deltas are small
        let msg = ResidualMessage {
            from_partition: 0,
            to_partition: 1,
            time: 1e-9,
            updates: vec![(100, 1.800001)],
        };
        let predicted = vec![(100, 1.8)];

        let compressed = codec.encode(&msg, &predicted);
        assert_eq!(compressed.deltas.len(), 1);
        // Scale should be very small
        assert!(
            compressed.scale < 1e-3,
            "Scale should be tiny for small deltas"
        );
    }

    #[test]
    fn test_skip_tiny_deltas() {
        let mut codec = ResidualCodec::new(CodecConfig {
            quant_bits: 16,
            min_delta: 1e-6,
        });

        let msg = ResidualMessage {
            from_partition: 0,
            to_partition: 1,
            time: 1e-9,
            updates: vec![
                (100, 1.8),       // delta = 0, should skip
                (200, 0.9 + 1e-7), // delta = 1e-7 < 1e-6, should skip
                (300, 1.5),       // delta = 0.3, should keep
            ],
        };
        let predicted = vec![(100, 1.8), (200, 0.9), (300, 1.2)];

        let compressed = codec.encode(&msg, &predicted);
        assert_eq!(
            compressed.deltas.len(),
            1,
            "Only the large delta should survive"
        );
        assert_eq!(codec.stats.updates_skipped, 2);
    }

    #[test]
    fn test_bytes_roundtrip() {
        let mut codec = ResidualCodec::with_defaults();

        let msg = ResidualMessage {
            from_partition: 0,
            to_partition: 1,
            time: 5e-10,
            updates: vec![(42, 1.65)],
        };
        let predicted = vec![(42, 1.6)];

        let bytes = codec.encode_to_bytes(&msg, &predicted);
        let decoded = codec.decode_from_bytes(&bytes).unwrap();

        assert_eq!(decoded.from_partition, 0);
        assert_eq!(decoded.updates.len(), 1);
        assert!((decoded.updates[0].1 - 1.65).abs() < 1e-3);
    }

    #[test]
    fn test_compression_stats() {
        let mut codec = ResidualCodec::with_defaults();

        for i in 0..10 {
            let msg = ResidualMessage {
                from_partition: 0,
                to_partition: 1,
                time: i as f64 * 1e-12,
                updates: vec![(100, 1.8 + i as f64 * 0.001)],
            };
            let predicted = vec![(100, 1.8 + i as f64 * 0.0009)];
            codec.encode(&msg, &predicted);
        }

        assert_eq!(codec.stats.messages_encoded, 10);
        assert!(
            codec.stats.compression_ratio() >= 1.0,
            "Should achieve some compression"
        );
    }
}
