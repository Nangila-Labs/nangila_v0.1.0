//! Momentum-based Gradient Predictor (The Slow Head)
//!
//! Predicts the next gradient based on momentum dynamics:
//! ĝ_{t+1} = g_t + μ * (g_t - g_{t-1})
//!
//! This module uses Q8.23 fixed-point arithmetic internally for
//! deterministic, bit-exact computation across different hardware.

use crate::checkpoint::LayerHistorySnapshot;
use crate::fixed_point::{FixedPointBuffer, Q8_23};
use crate::{LayerId, NangilaError, Result, Tensor};
use std::collections::HashMap;

/// History entry for a single layer (stored in fixed-point)
#[derive(Debug, Clone)]
struct LayerHistory {
    /// Gradient from step t-1 (fixed-point)
    prev: Option<FixedPointBuffer>,
    /// Gradient from step t (current, fixed-point)
    current: Option<FixedPointBuffer>,
}

impl LayerHistory {
    fn new() -> Self {
        Self {
            prev: None,
            current: None,
        }
    }

    /// Update history with a new gradient, shifting the window
    fn update(&mut self, gradient: &Tensor) {
        // Convert to fixed-point for deterministic storage
        let fixed = FixedPointBuffer::from_f32_slice(&gradient.data);
        self.prev = self.current.take();
        self.current = Some(fixed);
    }

    /// Check if we have enough history to predict
    fn can_predict(&self) -> bool {
        self.prev.is_some() && self.current.is_some()
    }

    /// Export to snapshot format for checkpointing
    fn to_snapshot(&self) -> LayerHistorySnapshot {
        LayerHistorySnapshot {
            prev: self
                .prev
                .as_ref()
                .map(|b| b.as_slice().iter().map(|q| q.to_bits()).collect()),
            current: self
                .current
                .as_ref()
                .map(|b| b.as_slice().iter().map(|q| q.to_bits()).collect()),
        }
    }

    /// Restore from snapshot
    fn from_snapshot(snapshot: &LayerHistorySnapshot) -> Self {
        Self {
            prev: snapshot.prev.as_ref().map(|v| {
                let data: Vec<Q8_23> = v.iter().map(|&bits| Q8_23::from_bits(bits)).collect();
                // We need to construct a FixedPointBuffer from Q8_23 values
                let f32_data: Vec<f32> = data.iter().map(|q| q.to_f32()).collect();
                FixedPointBuffer::from_f32_slice(&f32_data)
            }),
            current: snapshot.current.as_ref().map(|v| {
                let data: Vec<Q8_23> = v.iter().map(|&bits| Q8_23::from_bits(bits)).collect();
                let f32_data: Vec<f32> = data.iter().map(|q| q.to_f32()).collect();
                FixedPointBuffer::from_f32_slice(&f32_data)
            }),
        }
    }
}

/// Momentum-based gradient predictor with fixed-point arithmetic
#[derive(Debug)]
pub struct Predictor {
    /// Momentum coefficient (typically 0.9)
    momentum: f32,
    /// Momentum as fixed-point for deterministic computation
    momentum_q: Q8_23,
    /// Per-layer gradient history (stored in fixed-point)
    history: HashMap<LayerId, LayerHistory>,
    /// Current step count
    step: usize,
    /// Whether the predictor is warmed up
    warmed_up: bool,
    /// Required warmup steps
    warmup_steps: usize,
}

impl Predictor {
    /// Create a new predictor with given momentum coefficient
    pub fn new(momentum: f32, warmup_steps: usize) -> Self {
        Self {
            momentum,
            momentum_q: Q8_23::from_f32(momentum),
            history: HashMap::new(),
            step: 0,
            warmed_up: false,
            warmup_steps,
        }
    }

    /// Predict the next gradient for a layer
    ///
    /// Uses momentum extrapolation: ĝ_{t+1} = g_t + μ * (g_t - g_{t-1})
    /// All computation is done in Q8.23 fixed-point for determinism.
    pub fn predict(&self, layer_id: LayerId) -> Result<Tensor> {
        let history = self
            .history
            .get(&layer_id)
            .ok_or(NangilaError::LayerNotFound(layer_id))?;

        if !history.can_predict() {
            return Err(NangilaError::InsufficientHistory(layer_id));
        }

        let g_t = history.current.as_ref().unwrap();
        let g_t_minus_1 = history.prev.as_ref().unwrap();

        // Fixed-point computation:
        // delta = g_t - g_{t-1}
        // momentum_term = delta * μ
        // prediction = g_t + momentum_term
        let delta = g_t.sub(g_t_minus_1);
        let momentum_term = delta.scale(self.momentum);
        let prediction = g_t.add(&momentum_term);

        // Convert back to f32 tensor for output
        let data = prediction.to_f32_vec();
        let shape = vec![data.len()];
        Ok(Tensor::new(data, shape))
    }

    /// Update predictor state with actual gradient (for synchronized state)
    ///
    /// IMPORTANT: Always call this with the reconstructed/actual gradient,
    /// not speculative values, to maintain bit-perfect consensus.
    pub fn update(&mut self, layer_id: LayerId, gradient: Tensor) {
        self.history
            .entry(layer_id)
            .or_insert_with(LayerHistory::new)
            .update(&gradient);
    }

    /// Advance the step counter (call once per training step)
    pub fn step(&mut self) {
        self.step += 1;
        if self.step >= self.warmup_steps && !self.warmed_up {
            self.warmed_up = true;
            tracing::info!(
                "Predictor warmed up after {} steps, compression enabled",
                self.step
            );
        }
    }

    /// Check if the predictor is ready to make predictions
    pub fn is_ready(&self) -> bool {
        self.warmed_up
    }

    /// Check if a specific layer can be predicted
    pub fn can_predict(&self, layer_id: LayerId) -> bool {
        self.history
            .get(&layer_id)
            .map(|h| h.can_predict())
            .unwrap_or(false)
    }

    /// Get the current step count
    pub fn current_step(&self) -> usize {
        self.step
    }

    /// Get the momentum coefficient
    pub fn momentum(&self) -> f32 {
        self.momentum
    }

    /// Reset the predictor state (for testing/debugging)
    pub fn reset(&mut self) {
        self.history.clear();
        self.step = 0;
        self.warmed_up = false;
    }

    /// Get the number of layers being tracked
    pub fn num_tracked_layers(&self) -> usize {
        self.history.len()
    }

    /// Compute a deterministic hash of the predictor state
    pub fn state_hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        self.step.hash(&mut hasher);

        let mut layer_ids: Vec<_> = self.history.keys().collect();
        layer_ids.sort();

        for &layer_id in &layer_ids {
            layer_id.hash(&mut hasher);
            if let Some(hist) = self.history.get(layer_id) {
                if let Some(ref buf) = hist.prev {
                    hasher.write_u64(buf.hash());
                }
                if let Some(ref buf) = hist.current {
                    hasher.write_u64(buf.hash());
                }
            }
        }

        hasher.finish()
    }

    /// Export history snapshots for checkpointing
    pub fn export_snapshots(&self) -> HashMap<LayerId, LayerHistorySnapshot> {
        self.history
            .iter()
            .map(|(&id, hist)| (id, hist.to_snapshot()))
            .collect()
    }

    /// Restore from history snapshots
    pub fn restore_snapshots(&mut self, snapshots: &HashMap<LayerId, LayerHistorySnapshot>) {
        self.history.clear();
        for (&id, snapshot) in snapshots {
            self.history
                .insert(id, LayerHistory::from_snapshot(snapshot));
        }
    }

    /// Reset state for a specific layer (for desync recovery)
    pub fn reset_layer(&mut self, layer_id: LayerId) {
        self.history.remove(&layer_id);
    }

    /// Force set history for a layer (for FORCE_SYNC recovery)
    pub fn force_sync_layer(&mut self, layer_id: LayerId, gradient: &Tensor) {
        let mut hist = LayerHistory::new();
        hist.update(gradient);
        hist.update(gradient); // Set both prev and current to same value
        self.history.insert(layer_id, hist);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tensor(vals: &[f32]) -> Tensor {
        Tensor::new(vals.to_vec(), vec![vals.len()])
    }

    #[test]
    fn test_momentum_prediction() {
        let mut predictor = Predictor::new(0.9, 0);

        // Step 0: g_0 = [1.0, 2.0, 3.0]
        predictor.update(0, make_tensor(&[1.0, 2.0, 3.0]));

        // Step 1: g_1 = [1.1, 2.1, 3.1]
        predictor.update(0, make_tensor(&[1.1, 2.1, 3.1]));

        // Predict g_2:
        // ĝ_2 = g_1 + 0.9 * (g_1 - g_0)
        //     = [1.1, 2.1, 3.1] + 0.9 * [0.1, 0.1, 0.1]
        //     = [1.19, 2.19, 3.19]
        let pred = predictor.predict(0).unwrap();
        assert!((pred.data[0] - 1.19).abs() < 0.001);
        assert!((pred.data[1] - 2.19).abs() < 0.001);
        assert!((pred.data[2] - 3.19).abs() < 0.001);
    }

    #[test]
    fn test_insufficient_history() {
        let mut predictor = Predictor::new(0.9, 0);
        predictor.update(0, make_tensor(&[1.0, 2.0]));

        // Only one gradient, can't predict yet
        assert!(!predictor.can_predict(0));
        assert!(predictor.predict(0).is_err());
    }

    #[test]
    fn test_warmup() {
        let mut predictor = Predictor::new(0.9, 100);
        assert!(!predictor.is_ready());

        for _ in 0..99 {
            predictor.step();
        }
        assert!(!predictor.is_ready());

        predictor.step(); // Step 100
        assert!(predictor.is_ready());
    }

    #[test]
    fn test_deterministic_hash() {
        let mut p1 = Predictor::new(0.9, 0);
        let mut p2 = Predictor::new(0.9, 0);

        // Same updates should produce same hash
        for i in 0..10 {
            let grad = make_tensor(&[i as f32 * 0.1, i as f32 * 0.2]);
            p1.update(0, grad.clone());
            p2.update(0, grad);
        }

        assert_eq!(p1.state_hash(), p2.state_hash());
    }

    #[test]
    fn test_force_sync() {
        let mut predictor = Predictor::new(0.9, 0);

        // Force sync sets both prev and current
        predictor.force_sync_layer(0, &make_tensor(&[1.0, 2.0, 3.0]));

        assert!(predictor.can_predict(0));
        // Prediction should equal current (since prev == current, delta = 0)
        let pred = predictor.predict(0).unwrap();
        assert!((pred.data[0] - 1.0).abs() < 0.001);
    }
}
