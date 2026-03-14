use crate::{compressor::Compressor, LayerId, NangilaConfig, Packet, PacketHeader, Result, Tensor};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// PowerSGD Compressed Packet
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerSGDPacket {
    /// Matrix P (M x R) - Flattened
    pub p_data: Vec<f32>,
    pub p_shape: Vec<usize>,
    /// Matrix Q (N x R) - Flattened
    pub q_data: Vec<f32>,
    pub q_shape: Vec<usize>,
    /// Original shape of the gradient (M x N)
    pub original_shape: Vec<usize>,
}

/// PowerSGD Compressor
///
/// Implements low-rank gradient compression using Power Iteration.
/// Decomposes Gradient G (M x N) into P (M x R) and Q (N x R) such that G ~ P @ Q.t()
#[derive(Debug)]
pub struct PowerSGDCompressor {
    config: NangilaConfig,
    /// Memory (Residuals) M_t
    residuals: HashMap<LayerId, Tensor>,
    /// State (Q) for warm-start. Map from LayerId to Matrix Q (flattened or structured)
    /// We verify shape matches on access.
    q_memory: HashMap<LayerId, Array2<f32>>,
    step: usize,
}

impl PowerSGDCompressor {
    pub fn new(config: NangilaConfig) -> Self {
        Self {
            config,
            residuals: HashMap::new(),
            q_memory: HashMap::new(),
            step: 0,
        }
    }

    /// Orthonormalize columns of matrix A using Gram-Schmidt
    /// A is (Rows x Cols), we normalize columns.
    fn orthogonalize(mat: &mut Array2<f32>) {
        let (_, cols) = mat.dim();
        for i in 0..cols {
            // 1. Project against previous columns
            for j in 0..i {
                let col_j = mat.column(j).to_owned();
                let dot = mat.column(i).dot(&col_j);
                // col_i = col_i - dot * col_j
                let mut col_i = mat.column_mut(i);
                col_i.scaled_add(-dot, &col_j);
            }
            // 2. Normalize
            let mut col_i = mat.column_mut(i);
            let norm = col_i.dot(&col_i).sqrt();
            if norm > 1e-10 {
                col_i.mapv_inplace(|x| x / norm);
            }
        }
    }
}

impl Compressor for PowerSGDCompressor {
    fn compress(&mut self, gradient: &Tensor, layer_id: LayerId) -> Result<Packet> {
        // 1. Only apply to 2D tensors. Flatten others?
        // PowerSGD is typically for linear/conv weights which are 2D or 4D.
        // For 1D (bias), we might skip or just use DGC/None.
        // Let's assume we treat everything as Matrix (M, N).
        // If 1D, N=1.

        let shape = &gradient.shape;
        if shape.len() == 1 {
            // Fallback for 1D: Just send as is (or DGC if we had mixed support).
            // Since this is a dedicated compressor, let's just serialize full tensor for 1D.
            // Or actually, PowerSGD usually skips small tensors (bias, LayerNorm).
            // For now, let's just pass 1D through as uncompressed (but wrapped in our packet format? No, that's messy).
            // Let's implement full rank P=G, Q=1 for 1D.
            // Or better: Let's assume we only use this for valid layers.
            // If we get 1D, we can treat as M x 1.
            // Q would be 1 x 1 (scalar 1.0).
        }

        // View gradient as 2D matrix (flattening extra dims if needed, e.g. Conv2D [Out, In, K, K] -> [Out, In*K*K])
        let m = shape[0];
        let n: usize = shape.iter().skip(1).product();
        if n == 0 {
            // empty tensor
            let header = PacketHeader::new_driver(self.step as u32, layer_id);
            return Ok(Packet::new(header, vec![]));
        }

        // 2. Accumulate residual
        let residual = self
            .residuals
            .entry(layer_id)
            .or_insert_with(|| Tensor::zeros(gradient.shape.clone()));
        // Add immutable for now
        *residual = residual.add(gradient);

        // 3. Power Iteration
        // Matrix M (aggregated gradient)
        // We need to convert Tensor to Array2
        let matrix_m = Array2::from_shape_vec((m, n), residual.data.clone())
            .map_err(|e| crate::NangilaError::InvalidFormat(e.to_string()))?;

        // Rank r
        let r = self.config.power_sgd_rank; // We need to add this to config!
                                            // Safety check
        let r = std::cmp::min(r, std::cmp::min(m, n));

        // Initialize Q (N x R) randomly or use memory
        let q = self.q_memory.entry(layer_id).or_insert_with(|| {
            let mut arr = Array2::<f32>::zeros((n, r));
            for i in 0..n {
                for j in 0..r {
                    arr[[i, j]] = ((i + j) % 100) as f32 / 100.0;
                }
            }
            Self::orthogonalize(&mut arr);
            arr
        });

        // Power Iteration Step 1: P = M x Q
        let mut p = matrix_m.dot(&*q);

        // Orthogonalize P
        Self::orthogonalize(&mut p);

        // Power Iteration Step 2: Q = M^T x P
        let new_q = matrix_m.t().dot(&p);

        // Update memory Q
        let mut next_q_state = new_q.clone();
        Self::orthogonalize(&mut next_q_state);
        *q = next_q_state;

        // 4. Update Residual
        // Approx = P x Q^T
        // P (orthogonal) x Q (un-normalized) recovers M
        let approx = p.dot(&new_q.t());

        // Subtract approx from residual
        for (i, v) in residual.data.iter_mut().enumerate() {
            let row = i / n;
            let col = i % n;
            *v -= approx[[row, col]];
        }

        // 5. Serialize P and Q
        let packet_payload = PowerSGDPacket {
            p_data: p.as_standard_layout().to_owned().into_raw_vec(),
            p_shape: vec![m, r],
            q_data: new_q.as_standard_layout().to_owned().into_raw_vec(),
            q_shape: vec![n, r],
            original_shape: gradient.shape.clone(),
        };

        let payload = bincode::serialize(&packet_payload)?;
        let header = PacketHeader::new_driver(self.step as u32, layer_id);

        Ok(Packet::new(header, payload))
    }

    fn decompress(&mut self, packet: &Packet, _layer_id: LayerId) -> Result<Tensor> {
        let data: PowerSGDPacket = bincode::deserialize(&packet.payload)?;

        let m = data.p_shape[0];
        let r_p = data.p_shape[1];
        let n = data.q_shape[0];
        let r_q = data.q_shape[1];

        assert_eq!(r_p, r_q, "Rank mismatch in P and Q");

        // Reconstruct: P x Q^T
        let p = Array2::from_shape_vec((m, r_p), data.p_data)
            .map_err(|e| crate::NangilaError::InvalidFormat(e.to_string()))?;
        let q = Array2::from_shape_vec((n, r_q), data.q_data)
            .map_err(|e| crate::NangilaError::InvalidFormat(e.to_string()))?;

        let reconstruction = p.dot(&q.t());

        Ok(Tensor::new(
            reconstruction
                .as_standard_layout()
                .to_owned()
                .into_raw_vec(),
            data.original_shape,
        ))
    }

    fn update(&mut self, _layer_id: u32, _gradient: &Tensor) -> Result<()> {
        Ok(())
    }

    fn step(&mut self) {
        self.step += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn make_tensor(vals: &[f32], shape: Vec<usize>) -> Tensor {
        Tensor::new(vals.to_vec(), shape)
    }

    #[test]
    fn test_orthogonalization() {
        // Create a matrix with dependent columns
        // Col 0: [1, 0]
        // Col 1: [1, 1]
        // Ortho -> Col 0: [1, 0], Col 1: [0, 1] after projection and normalization
        // Wait, input to orthogonalize is (Rows, Cols).
        // We normalize COLUMNS.

        let mut mat = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 0.0, 1.0]).unwrap();
        // Col 0 is [1, 0], Col 1 is [1, 1] if column-major?
        // ndarray default is RowMajor.
        // from_shape_vec((2,2), ...) -> Row 0: [1, 1], Row 1: [0, 1]
        // So Col 0: [1, 0], Col 1: [1, 1].

        PowerSGDCompressor::orthogonalize(&mut mat);

        let col0 = mat.column(0);
        let col1 = mat.column(1);

        // Dot product should be 0
        let dot = col0.dot(&col1);
        assert!(dot.abs() < 1e-6);

        // Norms should be 1
        assert!((col0.dot(&col0) - 1.0).abs() < 1e-6);
        assert!((col1.dot(&col1) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_power_sgd_compression() {
        let mut config = NangilaConfig::default();
        config.power_sgd_rank = 1;

        let mut compressor = PowerSGDCompressor::new(config);

        // Create a rank-1 matrix: u * v^T
        // u = [1, 0], v = [1, 1]
        // M = [[1, 1], [0, 0]]
        let data = vec![1.0, 1.0, 0.0, 0.0];
        let shape = vec![2, 2];
        let tensor = make_tensor(&data, shape);

        // Compress
        let packet = compressor.compress(&tensor, 0).unwrap();

        // Decompress
        let reconstructed = compressor.decompress(&packet, 0).unwrap();

        // Check reconstruction quality
        // Rank 1 approximation of Rank 1 matrix should be exact (ignoring float precision)
        for (i, (a, b)) in tensor
            .data
            .iter()
            .zip(reconstructed.data.iter())
            .enumerate()
        {
            assert!(
                (a - b).abs() < 1e-5,
                "Mismatch at index {}: {} vs {}",
                i,
                a,
                b
            );
        }
    }
}
