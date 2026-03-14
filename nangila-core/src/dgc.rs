use crate::{compressor::Compressor, LayerId, NangilaConfig, Packet, PacketHeader, Result, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Sparse tensor representation (Indices + Values)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseTensor {
    /// Flattened indices of non-zero elements
    pub indices: Vec<u32>,
    /// Values at the corresponding indices
    pub values: Vec<f32>,
    /// Original shape of the tensor
    pub shape: Vec<usize>,
    /// Total number of elements in the original tensor
    pub numel: usize,
}

impl SparseTensor {
    /// Calculate compression ratio
    pub fn compression_ratio(&self) -> f32 {
        let original_size = self.numel * 4; // FP32
                                            // Indices (u32 = 4 bytes) + Values (f32 = 4 bytes)
        let compressed_size = self.indices.len() * 4 + self.values.len() * 4;
        if compressed_size > 0 {
            original_size as f32 / compressed_size as f32
        } else {
            0.0
        }
    }
}

/// Deep Gradient Compression (DGC) Compressor
///
/// Implements:
/// 1. Gradient Accumulation: Accumulates gradients over steps until they are sent.
/// 2. Top-k Sparsification: Selects the top-k largest gradients (by magnitude).
/// 3. Momentum Correction (optional, usually handled by optimizer or wrapper).
///    For this implementation, we focus on residual accumulation.
///
/// Ref: "Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training"
#[derive(Debug)]
pub struct DGCCompressor {
    config: NangilaConfig,
    /// Accumulated residuals for each layer (Memory C)
    /// Key: LayerId, Value: Tensor (FP32)
    residuals: HashMap<LayerId, Tensor>,
    /// Current step
    step: usize,
}

impl DGCCompressor {
    pub fn new(config: NangilaConfig) -> Self {
        Self {
            config,
            residuals: HashMap::new(),
            step: 0,
        }
    }

    /// Select top-k elements (simplistic implementation)
    ///
    /// For production, use a min-heap or selection algorithm (e.g. introselect).
    /// Here we sort for simplicity and correctness first.
    fn top_k(tensor: &Tensor, k: usize) -> (Vec<u32>, Vec<f32>) {
        if k >= tensor.numel() {
            // If k is larger than size, return everything
            let indices: Vec<u32> = (0..tensor.numel() as u32).collect();
            let values = tensor.data.clone();
            return (indices, values);
        }

        // Create vector of (index, abs_value)
        let mut with_indices: Vec<(usize, f32)> = tensor
            .data
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v.abs()))
            .collect();

        // Sort by magnitude descending
        // Note: partial_cmp can fail for NaN, unwrap is safe if no NaNs
        with_indices
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top k
        let mut indices = Vec::with_capacity(k);
        let mut values = Vec::with_capacity(k);

        for (idx, _) in with_indices.iter().take(k) {
            indices.push(*idx as u32);
            values.push(tensor.data[*idx]);
        }

        // Sorting indices might help compression/locality later, but not strictly required
        // Let's keep them in magnitude order or input order?
        // Input order is better for cache locality during update, but requires re-sorting indices.
        // For now, magnitude order is fine.

        (indices, values)
    }
}

impl Compressor for DGCCompressor {
    fn compress(&mut self, gradient: &Tensor, layer_id: LayerId) -> Result<Packet> {
        // 1. Accumulate gradient: residual_{t} = residual_{t-1} + gradient_{t}
        let residual = self
            .residuals
            .entry(layer_id)
            .or_insert_with(|| Tensor::zeros(gradient.shape.clone()));

        // Add current gradient to residual
        // Note: Tensor::add returns new tensor. In-place add would be better for perf.
        // For now, using immutable add.
        *residual = residual.add(gradient);

        // 2. Determine k based on sparsity ratio
        // sparsity usually means e.g. 0.999 (99.9% zeros).
        // So we keep (1 - sparsity) * numel
        let keep_ratio = 1.0 - self.config.dgc_sparsity;
        let k = (residual.numel() as f32 * keep_ratio).ceil() as usize;

        // 3. Select indices
        let (indices, values) = Self::top_k(residual, k);

        // 4. Update residual: zero out sent gradients
        // We need to modify 'residual' in place or replace it.
        // Since we have mutable access to 'residual' via the map entry (if we kept it),
        // but we need to iterate indices.

        for &idx in &indices {
            residual.data[idx as usize] = 0.0;
        }

        // 5. Pack into SparseTensor
        let sparse = SparseTensor {
            indices,
            values,
            shape: residual.shape.clone(),
            numel: residual.numel(),
        };

        // 6. Serialize
        let payload = bincode::serialize(&sparse)?;

        // 7. Create Packet
        let header = PacketHeader::new_driver(self.step as u32, layer_id);

        Ok(Packet::new(header, payload))
    }

    fn decompress(&mut self, packet: &Packet, _layer_id: LayerId) -> Result<Tensor> {
        // 1. Deserialize
        let sparse: SparseTensor = bincode::deserialize(&packet.payload)?;

        // 2. Reconstruct dense tensor
        // Initialize zeros
        // How do we know the shape? It's in the SparseTensor
        let mut dense_data = vec![0.0; sparse.numel];

        // Scatter values
        for (&idx, &val) in sparse.indices.iter().zip(&sparse.values) {
            if (idx as usize) < dense_data.len() {
                dense_data[idx as usize] = val;
            } else {
                return Err(crate::NangilaError::InvalidFormat(format!(
                    "Index {} out of bounds for size {}",
                    idx,
                    dense_data.len()
                )));
            }
        }

        Ok(Tensor::new(dense_data, sparse.shape))
    }

    fn update(&mut self, _layer_id: u32, _gradient: &Tensor) -> Result<()> {
        // DGC state update is usually implicit in the accumulation step.
        // The 'update' method in Compressor trait was primarily for Predictor-based logic
        // where we need the *aggregated* gradient to update the predictor.
        //
        // In DGC, the sender maintains the residual. The receiver just sums what it gets.
        // However, if we implement momentum correction on top of DGC, we might use this.
        // For basic DGC, we don't need to do anything with the aggregated gradient here
        // because we've already cleared the sent values from our residual in 'compress'.
        Ok(())
    }

    fn step(&mut self) {
        self.step += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tensor(vals: &[f32]) -> Tensor {
        Tensor::new(vals.to_vec(), vec![vals.len()])
    }

    #[test]
    fn test_dgc_top_k_selection() {
        let mut config = NangilaConfig::default();
        config.dgc_sparsity = 0.6; // Keep top 40%

        let mut compressor = DGCCompressor::new(config);

        // [1.0, 5.0, 2.0, 4.0, 3.0] -> 5 elements
        // Keep 40% = 2 elements -> Should keep 5.0 and 4.0
        let grad = make_tensor(&[1.0, 5.0, 2.0, 4.0, 3.0]);

        let packet = compressor.compress(&grad, 0).unwrap();

        // Decompress
        let decompressed = compressor.decompress(&packet, 0).unwrap();

        // Check kept values
        assert_eq!(decompressed.data[1], 5.0);
        assert_eq!(decompressed.data[3], 4.0);

        // Check zeroed values
        // Note: DGC decompression reconstructs a dense tensor where missing values are 0
        assert_eq!(decompressed.data[0], 0.0);
        assert_eq!(decompressed.data[2], 0.0);
        assert_eq!(decompressed.data[4], 0.0);
    }

    #[test]
    fn test_dgc_accumulation() {
        let mut config = NangilaConfig::default();
        config.dgc_sparsity = 0.99; // Aggressive sparsity, keep ~1% (ceil -> 1 element for size 5)

        let mut compressor = DGCCompressor::new(config);

        // Step 1: Gradient [1.0, 0.0, 0.0, 0.0, 0.0]
        let grad1 = make_tensor(&[1.0, 0.0, 0.0, 0.0, 0.0]);
        let packet1 = compressor.compress(&grad1, 0).unwrap();
        let dec1 = compressor.decompress(&packet1, 0).unwrap();

        // Should send 1.0 (largest)
        assert_eq!(dec1.data[0], 1.0);

        // Step 2: Gradient [0.1, 2.0, 0.0, 0.0, 0.0]
        // Residual for index 0 is now 0.0 (since it was sent).
        // Residual for index 1 is 2.0.
        let grad2 = make_tensor(&[0.1, 2.0, 0.0, 0.0, 0.0]);
        let packet2 = compressor.compress(&grad2, 0).unwrap();
        let dec2 = compressor.decompress(&packet2, 0).unwrap();

        // Should send 2.0
        assert_eq!(dec2.data[1], 2.0);

        // Validate accumulation
        // Step 3: [0.3, 0.0, 0.0, 0.0, 0.0]
        // Index 0 accum: 0.1 (from step 2) + 0.3 = 0.4. Since k=1, and we send largest,
        // if others are 0, it should send 0.4.

        let grad3 = make_tensor(&[0.3, 0.0, 0.0, 0.0, 0.0]);
        let packet3 = compressor.compress(&grad3, 0).unwrap();
        let dec3 = compressor.decompress(&packet3, 0).unwrap();

        assert!((dec3.data[0] - 0.4).abs() < 1e-6);
    }
}
