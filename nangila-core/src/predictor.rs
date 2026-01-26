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

/// Memory optimization mode for predictor history
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryMode {
    /// Store history for all layers (default for small models)
    Full,
    /// Store history only for driver layers (saves 50-70% RAM)
    DriversOnly,
    /// Automatic: switch to DriversOnly if model exceeds threshold
    Auto,
}

impl Default for MemoryMode {
    fn default() -> Self {
        MemoryMode::Auto
    }
}

/// Momentum-based gradient predictor with fixed-point arithmetic
#[derive(Debug)]
pub struct Predictor {
    /// Base momentum coefficient (typically 0.9)
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
    // === Adaptive momentum fields ===
    /// Per-layer adaptive momentum (adjusted based on prediction error)
    adaptive_momentum: HashMap<LayerId, f32>,
    /// Per-layer prediction error history (for adaptation)
    error_history: HashMap<LayerId, Vec<f32>>,
    /// Error history window size
    error_window: usize,
    /// Minimum momentum (floor for damping)
    min_momentum: f32,
    /// Maximum momentum (ceiling)
    max_momentum: f32,
    /// Adaptation rate
    adaptation_rate: f32,
    // === Memory optimization ===
    /// Memory mode for history storage
    memory_mode: MemoryMode,
    /// Effective mode after auto-detection (resolved from Auto)
    effective_mode: MemoryMode,
    /// Parameter threshold for switching to DriversOnly (default: 10B)
    large_model_threshold: usize,
    /// Total tracked parameters (for auto-detection)
    total_tracked_params: usize,
    /// Set of driver layer IDs (for DriversOnly mode filtering)
    driver_layers: std::collections::HashSet<LayerId>,
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
            // Adaptive momentum defaults (disabled by default)
            adaptive_momentum: HashMap::new(),
            error_history: HashMap::new(),
            error_window: 10,
            min_momentum: 0.5,
            max_momentum: 0.95,
            adaptation_rate: 0.1,
            // Memory optimization defaults
            memory_mode: MemoryMode::Auto,
            effective_mode: MemoryMode::Full, // Until threshold check
            large_model_threshold: 10_000_000_000, // 10B params
            total_tracked_params: 0,
            driver_layers: std::collections::HashSet::new(),
        }
    }
    
    /// Enable adaptive momentum (adjusts per-layer momentum based on prediction error)
    pub fn with_adaptive_momentum(mut self, enabled: bool) -> Self {
        if !enabled {
            self.adaptive_momentum.clear();
            self.error_history.clear();
        }
        self
    }
    
    /// Configure adaptive momentum parameters
    pub fn with_adaptive_config(
        mut self,
        min_momentum: f32,
        max_momentum: f32,
        adaptation_rate: f32,
        error_window: usize,
    ) -> Self {
        self.min_momentum = min_momentum;
        self.max_momentum = max_momentum;
        self.adaptation_rate = adaptation_rate;
        self.error_window = error_window;
        self
    }

    /// Set memory mode explicitly
    pub fn with_memory_mode(mut self, mode: MemoryMode) -> Self {
        self.memory_mode = mode;
        if mode != MemoryMode::Auto {
            self.effective_mode = mode;
        }
        self
    }

    /// Set the large model threshold (params count)
    pub fn with_large_model_threshold(mut self, threshold: usize) -> Self {
        self.large_model_threshold = threshold;
        self
    }
    
    /// Set memory mode based on actual memory usage estimate
    pub fn with_memory_budget(mut self, max_memory_mb: usize) -> Self {
        // Estimate: each layer needs 2 buffers (prev + current) × 4 bytes (i32)
        // max_memory_mb = (total_params × 2 × 4) / (1024 × 1024)
        // total_params = max_memory_mb × 1024 × 1024 / 8
        self.large_model_threshold = max_memory_mb * 1024 * 1024 / 8;
        self
    }

    /// Register driver layers for DriversOnly mode
    pub fn set_driver_layers(&mut self, drivers: impl IntoIterator<Item = LayerId>) {
        self.driver_layers = drivers.into_iter().collect();
    }

    /// Check if a layer should be tracked (based on memory mode)
    pub fn should_track_layer(&self, layer_id: LayerId) -> bool {
        match self.effective_mode {
            MemoryMode::Full => true,
            MemoryMode::DriversOnly => self.driver_layers.contains(&layer_id),
            MemoryMode::Auto => true, // Auto resolves to Full or DriversOnly
        }
    }

    /// Update effective mode based on total params (call after first gradient)
    fn check_memory_threshold(&mut self) {
        if self.memory_mode == MemoryMode::Auto {
            // Calculate actual memory usage:
            // Each tracked param needs 2 buffers (prev + current) × 4 bytes (Q8.23 is i32)
            let estimated_memory_bytes = self.total_tracked_params * 2 * 4;
            let estimated_memory_mb = estimated_memory_bytes / (1024 * 1024);
            
            // Threshold is in params, convert to memory
            let threshold_memory_mb = self.large_model_threshold * 2 * 4 / (1024 * 1024);
            
            if estimated_memory_mb > threshold_memory_mb {
                self.effective_mode = MemoryMode::DriversOnly;
                tracing::info!(
                    "Predictor: Large model detected (~{}MB history > {}MB threshold), \
                     switching to DriversOnly memory mode",
                    estimated_memory_mb,
                    threshold_memory_mb
                );
            } else {
                self.effective_mode = MemoryMode::Full;
            }
        }
    }

    /// Get current memory mode info (mode, total_params, num_layers, estimated_memory_mb)
    pub fn memory_stats(&self) -> (MemoryMode, usize, usize, usize) {
        let estimated_memory_bytes = self.total_tracked_params * 2 * 4;
        let estimated_memory_mb = estimated_memory_bytes / (1024 * 1024);
        (
            self.effective_mode,
            self.total_tracked_params,
            self.history.len(),
            estimated_memory_mb,
        )
    }

    /// Get the effective momentum for a layer (adaptive if enabled)
    pub fn get_momentum(&self, layer_id: LayerId) -> f32 {
        *self.adaptive_momentum.get(&layer_id).unwrap_or(&self.momentum)
    }

    /// Record prediction error for adaptive momentum
    pub fn record_error(&mut self, layer_id: LayerId, error: f32) {
        let history = self.error_history.entry(layer_id).or_insert_with(Vec::new);
        history.push(error);
        
        // Keep only recent history
        if history.len() > self.error_window {
            history.remove(0);
        }
        
        // Adapt momentum based on error trend
        if history.len() >= 3 {
            self.adapt_momentum(layer_id);
        }
    }

    /// Adapt momentum based on prediction error history
    fn adapt_momentum(&mut self, layer_id: LayerId) {
        let history = match self.error_history.get(&layer_id) {
            Some(h) if h.len() >= 3 => h,
            _ => return,
        };
        
        let current_momentum = self.get_momentum(layer_id);
        
        // Calculate error trend (is error increasing or decreasing?)
        let recent: f32 = history.iter().rev().take(3).sum::<f32>() / 3.0;
        let older: f32 = history.iter().take(3).sum::<f32>() / 3.0;
        
        let new_momentum = if recent > older * 1.1 {
            // Error increasing → reduce momentum (more conservative)
            (current_momentum - self.adaptation_rate).max(self.min_momentum)
        } else if recent < older * 0.9 {
            // Error decreasing → increase momentum (more aggressive)
            (current_momentum + self.adaptation_rate * 0.5).min(self.max_momentum)
        } else {
            // Stable → slowly return to base momentum
            current_momentum + (self.momentum - current_momentum) * 0.05
        };
        
        self.adaptive_momentum.insert(layer_id, new_momentum);
    }

    /// Predict the next gradient for a layer
    ///
    /// Uses momentum extrapolation: ĝ_{t+1} = g_t + μ * (g_t - g_{t-1})
    /// All computation is done in Q8.23 fixed-point for determinism.
    /// Uses adaptive momentum if enabled.
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

        // Get adaptive momentum for this layer (or base momentum if not adapted)
        let effective_momentum = self.get_momentum(layer_id);

        // Fixed-point computation:
        // delta = g_t - g_{t-1}
        // momentum_term = delta * μ (using adaptive μ)
        // prediction = g_t + momentum_term
        let delta = g_t.sub(g_t_minus_1);
        let momentum_term = delta.scale(effective_momentum);
        let prediction = g_t.add(&momentum_term);

        // Convert back to f32 tensor for output
        let data = prediction.to_f32_vec();
        let shape = vec![data.len()];
        Ok(Tensor::new(data, shape))
    }

    /// Predict the next gradient for a specific range of indices
    pub fn predict_partial(
        &self,
        layer_id: LayerId,
        start_index: usize,
        end_index: usize,
    ) -> Result<Tensor> {
        let history = self
            .history
            .get(&layer_id)
            .ok_or(NangilaError::LayerNotFound(layer_id))?;

        if !history.can_predict() {
            return Err(NangilaError::InsufficientHistory(layer_id));
        }

        let g_t = history.current.as_ref().unwrap();
        let g_t_minus_1 = history.prev.as_ref().unwrap();

        // Get adaptive momentum for this layer
        let effective_momentum = self.get_momentum(layer_id);
        let momentum = Q8_23::from_f32(effective_momentum);
        
        let mut data = Vec::with_capacity(end_index - start_index);
        let slice_len = (end_index - start_index).min(g_t.len() - start_index);
        
        for i in 0..slice_len {
            let idx = start_index + i;
            if idx >= g_t.len() || idx >= g_t_minus_1.len() {
                break;
            }
            let now = g_t.as_slice()[idx];
            let prev = g_t_minus_1.as_slice()[idx];
            
            let delta = now - prev;
            let term = delta * momentum;
            let pred = now + term;
            data.push(pred.to_f32());
        }

        let len = data.len();
        Ok(Tensor::new(data, vec![len]))
    }

    /// Update predictor state with actual gradient (for synchronized state)
    ///
    /// IMPORTANT: Always call this with the reconstructed/actual gradient,
    /// not speculative values, to maintain bit-perfect consensus.
    pub fn update(&mut self, layer_id: LayerId, gradient: Tensor) {
        // Track total parameters for memory mode auto-detection
        let layer_size = gradient.numel();
        if !self.history.contains_key(&layer_id) {
            self.total_tracked_params += layer_size;
        }
        
        self.history
            .entry(layer_id)
            .or_insert_with(LayerHistory::new)
            .update(&gradient);
    }

    /// Advance the step counter (call once per training step)
    pub fn step(&mut self) {
        self.step += 1;
        
        // Check memory usage once per step (fast check)
        self.check_memory_threshold();

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
