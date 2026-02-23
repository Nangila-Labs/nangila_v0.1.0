//! Ghost Node Buffer
//!
//! Manages boundary conditions for connected partitions.
//! Each ghost node stores:
//!   - Signal state (voltage history + derivatives)
//!   - Selected predictor (auto-chosen by signal classification)
//!   - Confidence score for adaptive speculative depth
//!
//! Phase 2, Sprint 5: Upgraded with predictor selection and accuracy tracking.

use serde::{Deserialize, Serialize};


use crate::predictor::{
    AdaptiveConfidence, PredictionAccuracyLogger, PredictorSelector, SignalState,
};

/// A single ghost node representing a boundary connection to a neighbor partition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GhostNode {
    /// Unique identifier for the circuit net this ghost tracks
    pub net_id: u64,
    /// ID of the neighbor partition that owns this node
    pub owner_partition: u32,
    /// Last known true voltage (from neighbor's residual update)
    pub last_known_voltage: f64,
    /// Estimated rate of change (V/s)
    pub gradient: f64,
    /// Confidence in the current prediction (0.0 = no confidence, 1.0 = exact)
    pub confidence: f64,
    /// Timestamp of the last update
    pub last_update_time: f64,
    /// Signal state for advanced predictors
    #[serde(skip)]
    signal_state: Option<SignalState>,
    /// Name of the currently selected predictor
    pub active_predictor: String,
}

impl GhostNode {
    pub fn new(net_id: u64, owner_partition: u32) -> Self {
        Self {
            net_id,
            owner_partition,
            last_known_voltage: 0.0,
            gradient: 0.0,
            confidence: 0.0,
            last_update_time: 0.0,
            signal_state: Some(SignalState::new()),
            active_predictor: "Linear".to_string(),
        }
    }

    /// Get or initialize the signal state.
    fn state_mut(&mut self) -> &mut SignalState {
        self.signal_state.get_or_insert_with(SignalState::new)
    }

    fn state(&self) -> &SignalState {
        // This is safe because we always initialize in new()
        // For deserialized nodes, we create a fresh state
        static DEFAULT: std::sync::LazyLock<SignalState> =
            std::sync::LazyLock::new(SignalState::new);
        self.signal_state.as_ref().unwrap_or(&DEFAULT)
    }

    /// Apply a residual correction from the neighbor partition.
    /// Returns true if the correction was within tolerance.
    pub fn apply_correction(&mut self, true_voltage: f64, time: f64, reltol: f64) -> bool {
        let predicted = self.predict(time);
        let error = (true_voltage - predicted).abs();

        // Update gradient from new data
        let dt = time - self.last_update_time;
        if dt > 0.0 {
            self.gradient = (true_voltage - self.last_known_voltage) / dt;
        }

        // Update state
        self.last_known_voltage = true_voltage;
        self.last_update_time = time;

        // Record in signal state for advanced predictors
        self.state_mut().record(time, true_voltage);

        // Update confidence based on prediction accuracy
        let relative_error = if true_voltage.abs() > 1e-15 {
            error / true_voltage.abs()
        } else {
            error
        };
        self.confidence = (1.0 - relative_error).clamp(0.0, 1.0);

        // Within tolerance?
        relative_error < reltol
    }

    /// Predict the voltage at a future time using the auto-selected predictor.
    pub fn predict(&self, time: f64) -> f64 {
        let state = self.state();

        if state.history.len() >= 4 {
            // Use advanced predictor selector
            let selector = PredictorSelector::new();
            let (v, _name) = selector.predict(state, time);
            v
        } else {
            // Fallback: linear extrapolation
            let dt = time - self.last_update_time;
            self.last_known_voltage + self.gradient * dt
        }
    }

    /// Predict and also return the predictor name used.
    pub fn predict_named(&self, time: f64) -> (f64, String) {
        let state = self.state();

        if state.history.len() >= 4 {
            let selector = PredictorSelector::new();
            let (v, name) = selector.predict(state, time);
            (v, name.to_string())
        } else {
            let dt = time - self.last_update_time;
            let v = self.last_known_voltage + self.gradient * dt;
            (v, "Linear".to_string())
        }
    }
}

/// The Ghost Buffer manages all ghost nodes for a partition.
/// Sprint 5: Now includes accuracy logging and adaptive confidence.
#[derive(Debug)]
pub struct GhostBuffer {
    /// All ghost nodes indexed by net_id
    nodes: Vec<GhostNode>,
    /// Prediction accuracy tracker
    pub accuracy_logger: PredictionAccuracyLogger,
    /// Adaptive speculative depth controller
    pub adaptive_confidence: AdaptiveConfidence,
}

impl GhostBuffer {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            accuracy_logger: PredictionAccuracyLogger::new(),
            adaptive_confidence: AdaptiveConfidence::new(5),
        }
    }

    /// Create with a specific base speculative depth.
    pub fn with_depth(base_depth: u32) -> Self {
        Self {
            nodes: Vec::new(),
            accuracy_logger: PredictionAccuracyLogger::new(),
            adaptive_confidence: AdaptiveConfidence::new(base_depth),
        }
    }

    pub fn add_ghost(&mut self, node: GhostNode) {
        self.nodes.push(node);
    }

    /// Get the number of ghost nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get predicted values for all ghost nodes at time `t`.
    pub fn predict_all(&self, t: f64) -> Vec<(u64, f64)> {
        self.nodes
            .iter()
            .map(|g| (g.net_id, g.predict(t)))
            .collect()
    }

    /// Apply corrections, track accuracy, and return whether all were within tolerance.
    pub fn apply_corrections(
        &mut self,
        corrections: &[(u64, f64, f64)], // (net_id, true_voltage, time)
        reltol: f64,
    ) -> bool {
        let mut all_ok = true;

        for &(net_id, voltage, time) in corrections {
            if let Some(ghost) = self.nodes.iter_mut().find(|g| g.net_id == net_id) {
                // Get prediction before applying correction (for accuracy tracking)
                let (predicted, predictor_name) = ghost.predict_named(time);

                let was_accurate = ghost.apply_correction(voltage, time, reltol);
                ghost.active_predictor = predictor_name.clone();

                // Log accuracy
                self.accuracy_logger
                    .record(net_id, &predictor_name, predicted, voltage, reltol);

                // Update adaptive confidence
                self.adaptive_confidence.update(net_id, was_accurate);

                if !was_accurate {
                    all_ok = false;
                }
            }
        }

        all_ok
    }

    /// Get the current effective speculative depth.
    pub fn effective_depth(&self) -> u32 {
        self.adaptive_confidence.effective_depth()
    }

    /// Get a summary of prediction accuracy.
    pub fn accuracy_summary(&self) -> String {
        format!(
            "{}\n{}",
            self.accuracy_logger.summary(),
            self.adaptive_confidence.summary()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_prediction() {
        let mut ghost = GhostNode::new(1, 0);
        ghost.last_known_voltage = 1.0;
        ghost.gradient = 1e9; // 1V/ns
        ghost.last_update_time = 0.0;

        let predicted = ghost.predict(1e-9); // predict at t=1ns
        assert!(
            (predicted - 2.0).abs() < 1e-10,
            "Expected ~2.0V, got {predicted}"
        );
    }

    #[test]
    fn test_correction_within_tolerance() {
        let mut ghost = GhostNode::new(1, 0);
        ghost.last_known_voltage = 1.0;
        ghost.gradient = 1e9;
        ghost.last_update_time = 0.0;

        let ok = ghost.apply_correction(2.001, 1e-9, 1e-3);
        assert!(ok, "Correction should be within tolerance");
    }

    #[test]
    fn test_correction_exceeds_tolerance() {
        let mut ghost = GhostNode::new(1, 0);
        ghost.last_known_voltage = 1.0;
        ghost.gradient = 1e9;
        ghost.last_update_time = 0.0;

        let ok = ghost.apply_correction(3.0, 1e-9, 1e-3);
        assert!(!ok, "Correction should exceed tolerance");
    }

    #[test]
    fn test_predictor_selection_with_history() {
        let mut ghost = GhostNode::new(1, 0);

        // Build up enough history for auto-selection
        for i in 0..10 {
            let t = i as f64 * 1e-12;
            let v = 1.8; // Constant → should select Linear
            ghost.apply_correction(v, t, 1e-3);
        }

        let (v, name) = ghost.predict_named(10e-12);
        assert_eq!(name, "Linear", "Constant signal → Linear predictor");
        assert!(
            (v - 1.8).abs() < 0.01,
            "Prediction should be ~1.8V, got {v}"
        );
    }

    #[test]
    fn test_ghost_buffer_accuracy_tracking() {
        let mut buf = GhostBuffer::new();
        buf.add_ghost(GhostNode::new(100, 0));
        buf.add_ghost(GhostNode::new(200, 0));

        // Apply some corrections
        buf.apply_corrections(
            &[(100, 1.0, 1e-12), (200, 0.5, 1e-12)],
            1e-3,
        );
        buf.apply_corrections(
            &[(100, 1.001, 2e-12), (200, 0.6, 2e-12)],
            1e-3,
        );

        assert_eq!(buf.accuracy_logger.node_stats.len(), 2);
        assert!(
            buf.accuracy_logger.overall_hit_rate() >= 0.0,
            "Hit rate should be valid"
        );
    }

    #[test]
    fn test_adaptive_depth_adjustment() {
        let mut buf = GhostBuffer::with_depth(5);
        buf.add_ghost(GhostNode::new(1, 0));

        // All accurate → depth should be >= base
        for i in 0..20 {
            let t = i as f64 * 1e-12;
            buf.apply_corrections(&[(1, 1.8, t)], 1e-3);
        }
        let depth_good = buf.effective_depth();
        assert!(
            depth_good >= 5,
            "Good predictions should keep high depth, got {depth_good}"
        );
    }
}
