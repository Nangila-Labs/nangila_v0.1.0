//! Reconstructor: Receiver-side Gradient Reconstruction
//!
//! Reconstructs the full gradient vector from:
//! - Driver layers: dequantized residuals + local prediction
//! - Passenger layers: synthesized from Driver using coupling factors

use crate::{
    mask::LayerRole, CompressedTensor, LayerId, NangilaError, Predictor, Quantizer, Result, Tensor,
    TopologyMask,
};
use std::collections::HashMap;

/// Reconstructor for receiver-side gradient recovery
#[derive(Debug)]
pub struct Reconstructor {
    /// Quantizer for dequantization
    quantizer: Quantizer,
    /// Cache of reconstructed Driver gradients (for Passenger synthesis)
    driver_cache: HashMap<LayerId, Tensor>,
    /// Current step counter (for validation)
    current_step: usize,
    /// Last step when a driver was reconstructed (for staleness detection)
    last_reconstruction_step: Option<usize>,
}

impl Reconstructor {
    /// Create a new reconstructor
    pub fn new(quantizer: Quantizer) -> Self {
        Self {
            quantizer,
            driver_cache: HashMap::new(),
            current_step: 0,
            last_reconstruction_step: None,
        }
    }

    /// Reconstruct a Driver layer gradient
    ///
    /// g = ĝ (prediction) + dequantize(q) (residual)
    pub fn reconstruct_driver(
        &mut self,
        layer_id: LayerId,
        compressed: &CompressedTensor,
        predictor: &Predictor,
    ) -> Result<Tensor> {
        // Get prediction from local predictor
        let prediction = predictor.predict(layer_id)?;

        // Dequantize residual
        let residual = self.quantizer.dequantize(compressed);

        // Reconstruct: g = ĝ + r
        let gradient = prediction.add(&residual);

        // Cache for Passenger synthesis with step marker
        self.driver_cache.insert(layer_id, gradient.clone());
        self.last_reconstruction_step = Some(self.current_step);

        Ok(gradient)
    }

    /// Reconstruct a partial Driver layer gradient
    pub fn reconstruct_driver_partial(
        &mut self,
        layer_id: LayerId,
        compressed: &CompressedTensor,
        predictor: &Predictor,
        start_index: usize,
        end_index: usize,
    ) -> Result<Tensor> {
        // Get partial prediction
        let prediction = predictor.predict_partial(layer_id, start_index, end_index)?;

        // Dequantize partial residual
        let residual = self
            .quantizer
            .dequantize_partial(compressed, start_index, end_index);

        // Reconstruct: g = ĝ + r
        let gradient = prediction.add(&residual);

        // Note: we don't cache partial reconstructions as they are usually transient for FSDP

        Ok(gradient)
    }

    /// Synthesize a Passenger layer gradient from its Driver
    ///
    /// g_passenger = α * g_driver + β
    pub fn synthesize_passenger(&self, layer_id: LayerId, mask: &TopologyMask) -> Result<Tensor> {
        let role = mask.get_role(layer_id)?;

        match role {
            LayerRole::Passenger {
                source_id,
                alpha,
                beta,
            } => {
                // Validate driver exists in cache
                let driver_grad = self.driver_cache.get(source_id).ok_or_else(|| {
                    tracing::error!(
                        "Passenger {} depends on driver {} which is not in cache. \
                             This indicates driver was not reconstructed this step.",
                        layer_id,
                        source_id
                    );
                    NangilaError::LayerNotFound(*source_id)
                })?;

                // Validate driver gradient is not stale
                if let Some(last_step) = self.last_reconstruction_step {
                    if last_step != self.current_step {
                        tracing::warn!(
                            "Passenger {} using driver {} from step {} (current: {}). \
                             Driver cache may be stale!",
                            layer_id,
                            source_id,
                            last_step,
                            self.current_step
                        );
                    }
                }

                // Validate driver gradient size is reasonable
                if driver_grad.numel() == 0 {
                    return Err(NangilaError::InvalidFormat(format!(
                        "Driver {} has zero elements, cannot synthesize passenger {}",
                        source_id, layer_id
                    )));
                }

                // g_passenger = α * g_driver + β
                let mut passenger_grad = driver_grad.scale(*alpha);

                // Add bias term
                if beta.abs() > 1e-8 {
                    for val in &mut passenger_grad.data {
                        *val += beta;
                    }
                }

                Ok(passenger_grad)
            }
            LayerRole::Driver => Err(NangilaError::LayerNotFound(layer_id)), // Not a passenger
        }
    }

    /// Reconstruct all layers given compressed Driver data
    pub fn reconstruct_all(
        &mut self,
        compressed_drivers: &HashMap<LayerId, CompressedTensor>,
        mask: &TopologyMask,
        predictor: &Predictor,
    ) -> Result<HashMap<LayerId, Tensor>> {
        let mut gradients = HashMap::new();

        // First pass: reconstruct all Drivers
        for layer_id in mask.drivers() {
            if let Some(compressed) = compressed_drivers.get(&layer_id) {
                let grad = self.reconstruct_driver(layer_id, compressed, predictor)?;
                gradients.insert(layer_id, grad);
            }
        }

        // Second pass: synthesize all Passengers
        for (layer_id, source_id, _alpha) in mask.passengers() {
            // Ensure the source Driver was reconstructed
            if !self.driver_cache.contains_key(&source_id) {
                continue; // Skip if Driver not available
            }
            let grad = self.synthesize_passenger(layer_id, mask)?;
            gradients.insert(layer_id, grad);
        }

        Ok(gradients)
    }

    /// Clear the driver cache (call at end of step)
    pub fn clear_cache(&mut self) {
        self.driver_cache.clear();
        self.current_step += 1;
    }

    /// Synthesize a partial Passenger layer gradient from its Driver
    ///
    /// For FSDP use cases where only a shard of the gradient is needed.
    /// Requires the driver gradient to be cached (either full or covering the requested range).
    ///
    /// g_passenger[start:end] = α * g_driver[start:end] + β
    pub fn synthesize_passenger_partial(
        &self,
        layer_id: LayerId,
        mask: &TopologyMask,
        start_index: usize,
        end_index: usize,
    ) -> Result<Tensor> {
        let role = mask.get_role(layer_id)?;

        match role {
            LayerRole::Passenger {
                source_id,
                alpha,
                beta,
            } => {
                let driver_grad = self
                    .driver_cache
                    .get(source_id)
                    .ok_or(NangilaError::LayerNotFound(*source_id))?;

                // Validate indices
                let driver_len = driver_grad.numel();
                if end_index > driver_len || start_index >= end_index {
                    return Err(NangilaError::InvalidFormat(format!(
                        "Invalid partial range: [{}, {}) for driver len {}",
                        start_index, end_index, driver_len
                    )));
                }

                // Slice driver gradient
                let slice_len = end_index - start_index;
                let driver_slice: Vec<f32> = driver_grad.data[start_index..end_index].to_vec();

                // Scale and add bias: g_passenger = α * g_driver + β
                let passenger_data: Vec<f32> =
                    driver_slice.iter().map(|&val| val * alpha + beta).collect();

                Ok(Tensor::new(passenger_data, vec![slice_len]))
            }
            LayerRole::Driver => Err(NangilaError::LayerNotFound(layer_id)),
        }
    }

    /// Get a reference to the internal quantizer
    pub fn quantizer(&self) -> &Quantizer {
        &self.quantizer
    }

    /// Get a mutable reference to the internal quantizer
    pub fn quantizer_mut(&mut self) -> &mut Quantizer {
        &mut self.quantizer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tensor(vals: &[f32]) -> Tensor {
        Tensor::new(vals.to_vec(), vec![vals.len()])
    }

    #[test]
    fn test_driver_reconstruction() {
        let mut predictor = Predictor::new(0.9, 0);
        let mut quantizer = Quantizer::int4();
        quantizer.set_gamma(0.1);

        // Build up predictor history
        predictor.update(0, make_tensor(&[1.0, 2.0, 3.0]));
        predictor.update(0, make_tensor(&[1.1, 2.1, 3.1]));

        // True gradient at step 2
        let true_grad = make_tensor(&[1.2, 2.2, 3.2]);

        // Prediction: ~[1.19, 2.19, 3.19]
        let prediction = predictor.predict(0).unwrap();

        // Residual
        let residual = true_grad.sub(&prediction);

        // Compress residual
        let compressed = quantizer.quantize(&residual, 0, 0);

        // Reconstruct
        let mut reconstructor = Reconstructor::new(Quantizer::int4());
        reconstructor.quantizer_mut().set_gamma(0.1);
        let reconstructed = reconstructor
            .reconstruct_driver(0, &compressed, &predictor)
            .unwrap();

        // Check reconstruction is close to true gradient
        for (true_val, rec_val) in true_grad.data.iter().zip(&reconstructed.data) {
            let error = (true_val - rec_val).abs();
            assert!(error < 0.2, "Reconstruction error too high: {}", error);
        }
    }

    #[test]
    fn test_passenger_synthesis() {
        let quantizer = Quantizer::int4();
        let mut reconstructor = Reconstructor::new(quantizer);

        // Set up mask with Driver 0 and Passenger 1 (α=0.5, β=0.1)
        let mut mask = TopologyMask::new();
        mask.add_driver(0);
        mask.add_passenger(1, 0, 0.5, 0.1);

        // Manually add Driver gradient to cache
        reconstructor
            .driver_cache
            .insert(0, make_tensor(&[2.0, 4.0, 6.0]));

        // Synthesize Passenger
        let passenger_grad = reconstructor.synthesize_passenger(1, &mask).unwrap();

        // Expected: 0.5 * [2, 4, 6] + 0.1 = [1.1, 2.1, 3.1]
        assert!((passenger_grad.data[0] - 1.1).abs() < 0.01);
        assert!((passenger_grad.data[1] - 2.1).abs() < 0.01);
        assert!((passenger_grad.data[2] - 3.1).abs() < 0.01);
    }

    #[test]
    fn test_passenger_partial_synthesis() {
        let quantizer = Quantizer::int4();
        let mut reconstructor = Reconstructor::new(quantizer);

        // Set up mask with Driver 0 and Passenger 1 (α=0.5, β=0.1)
        let mut mask = TopologyMask::new();
        mask.add_driver(0);
        mask.add_passenger(1, 0, 0.5, 0.1);

        // Manually add full Driver gradient to cache
        reconstructor
            .driver_cache
            .insert(0, make_tensor(&[10.0, 20.0, 30.0, 40.0, 50.0]));

        // Synthesize partial Passenger (indices 1..4)
        let partial = reconstructor
            .synthesize_passenger_partial(1, &mask, 1, 4)
            .unwrap();

        // Expected: 0.5 * [20, 30, 40] + 0.1 = [10.1, 15.1, 20.1]
        assert_eq!(partial.numel(), 3);
        assert!((partial.data[0] - 10.1).abs() < 0.01);
        assert!((partial.data[1] - 15.1).abs() < 0.01);
        assert!((partial.data[2] - 20.1).abs() < 0.01);
    }

    #[test]
    fn test_passenger_partial_bounds() {
        let quantizer = Quantizer::int4();
        let mut reconstructor = Reconstructor::new(quantizer);

        let mut mask = TopologyMask::new();
        mask.add_driver(0);
        mask.add_passenger(1, 0, 0.5, 0.0);

        reconstructor
            .driver_cache
            .insert(0, make_tensor(&[1.0, 2.0, 3.0]));

        // Out of bounds
        assert!(reconstructor
            .synthesize_passenger_partial(1, &mask, 0, 10)
            .is_err());

        // Invalid range (start >= end)
        assert!(reconstructor
            .synthesize_passenger_partial(1, &mask, 2, 2)
            .is_err());
    }
}
