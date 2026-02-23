//! Predictor Engine
//!
//! Provides boundary voltage prediction strategies for ghost nodes.
//! Each predictor extrapolates future voltage values from a signal history.
//!
//! Available predictors:
//!   - Linear: Simple dV/dt extrapolation (smooth, slowly-varying signals)
//!   - Exponential: RC decay model (discharge/precharge transients)
//!   - RungeKutta: Multi-point RK4-based extrapolation (stiff circuits)
//!   - Adaptive: Auto-selects best predictor per signal characteristics
//!
//! Phase 2, Sprint 5 deliverable.

// ─── Predictor Trait ───────────────────────────────────────────────

/// Trait for all predictor implementations.
pub trait Predictor: Send + Sync {
    /// Predict voltage at `target_time` given the signal state.
    fn predict(&self, state: &SignalState, target_time: f64) -> f64;

    /// Name of the predictor (for logging/metrics).
    fn name(&self) -> &str;
}

/// Snapshot of a signal's recent state, used by all predictors.
#[derive(Debug, Clone)]
pub struct SignalState {
    /// Most recent voltage
    pub value: f64,
    /// First derivative estimate (V/s)
    pub gradient: f64,
    /// Second derivative estimate (V/s²), if available
    pub second_derivative: f64,
    /// Time of the last update
    pub time: f64,
    /// Recent history: (time, voltage) pairs, newest last
    pub history: Vec<(f64, f64)>,
}

impl SignalState {
    pub fn new() -> Self {
        Self {
            value: 0.0,
            gradient: 0.0,
            second_derivative: 0.0,
            time: 0.0,
            history: Vec::new(),
        }
    }

    /// Record a new data point and update derivatives.
    pub fn record(&mut self, time: f64, voltage: f64) {
        let old_gradient = self.gradient;

        if time > self.time {
            let dt = time - self.time;
            self.gradient = (voltage - self.value) / dt;
            self.second_derivative = (self.gradient - old_gradient) / dt;
        }

        self.value = voltage;
        self.time = time;

        self.history.push((time, voltage));
        // Keep last 16 points for RK predictor
        if self.history.len() > 16 {
            self.history.remove(0);
        }
    }
}

// ─── Linear Predictor ──────────────────────────────────────────────

/// Simple linear extrapolation: V(t) = V₀ + dV/dt × Δt
#[derive(Debug)]
pub struct LinearPredictor;

impl Predictor for LinearPredictor {
    fn predict(&self, state: &SignalState, target_time: f64) -> f64 {
        let dt = target_time - state.time;
        state.value + state.gradient * dt
    }

    fn name(&self) -> &str {
        "Linear"
    }
}

// ─── Exponential Predictor ─────────────────────────────────────────

/// Exponential decay predictor: V(t) = V_∞ + (V₀ − V_∞) × exp(−Δt/τ)
/// Best for RC discharge/charge transients.
pub struct ExponentialPredictor {
    /// Time constant τ (seconds)
    pub tau: f64,
    /// Asymptotic voltage V_∞
    pub v_asymptote: f64,
}

impl Predictor for ExponentialPredictor {
    fn predict(&self, state: &SignalState, target_time: f64) -> f64 {
        let dt = target_time - state.time;
        let delta = state.value - self.v_asymptote;
        self.v_asymptote + delta * (-dt / self.tau).exp()
    }

    fn name(&self) -> &str {
        "Exponential"
    }
}

// ─── Runge-Kutta Predictor ─────────────────────────────────────────

/// 4th-order Runge-Kutta extrapolation predictor.
///
/// Uses the last 4+ history points to estimate the derivative field,
/// then applies RK4 integration to predict forward. Excellent for
/// stiff circuits where linear extrapolation diverges.
#[derive(Debug)]
pub struct RungeKuttaPredictor;

impl RungeKuttaPredictor {
    /// Estimate dV/dt at a given time by interpolating between history points.
    fn estimate_derivative(history: &[(f64, f64)], t: f64) -> f64 {
        if history.len() < 2 {
            return 0.0;
        }

        // Find the two bracketing points
        let n = history.len();
        let mut idx = n - 2;
        for i in 0..n - 1 {
            if history[i + 1].0 >= t {
                idx = i;
                break;
            }
        }

        let (t0, v0) = history[idx];
        let (t1, v1) = history[idx + 1];
        let dt = t1 - t0;

        if dt.abs() < 1e-30 {
            return 0.0;
        }

        // If we have enough points, use a 3-point central difference
        if idx > 0 && idx + 2 < n {
            let (t_prev, v_prev) = history[idx - 1];
            let (t_next, v_next) = history[idx + 2];
            let dt_wide = t_next - t_prev;
            if dt_wide.abs() > 1e-30 {
                return (v_next - v_prev) / dt_wide;
            }
        }

        // Fallback: simple finite difference
        (v1 - v0) / dt
    }
}

impl Predictor for RungeKuttaPredictor {
    fn predict(&self, state: &SignalState, target_time: f64) -> f64 {
        let h = target_time - state.time;

        if h.abs() < 1e-30 || state.history.len() < 2 {
            return state.value;
        }

        // RK4 integration using estimated derivative field
        let t0 = state.time;
        let y0 = state.value;
        let hist = &state.history;

        let k1 = Self::estimate_derivative(hist, t0);
        let k2 = Self::estimate_derivative(hist, t0 + h / 2.0);
        let k3 = k2; // Same midpoint estimate (we don't have future data)
        let k4 = Self::estimate_derivative(hist, t0 + h);

        // If we don't have data beyond current time, use curvature adjustment
        let k4_adjusted = if state.second_derivative.abs() > 1e-20 {
            k1 + state.second_derivative * h
        } else {
            k4
        };

        y0 + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4_adjusted)
    }

    fn name(&self) -> &str {
        "RungeKutta"
    }
}

// ─── Prediction Accuracy Logger ────────────────────────────────────

/// Tracks per-ghost-node prediction accuracy statistics.
#[derive(Debug, Clone)]
pub struct PredictionAccuracyLogger {
    /// Per-node stats: net_id → NodeStats
    pub node_stats: Vec<NodeAccuracyStats>,
}

/// Accuracy stats for a single ghost node.
#[derive(Debug, Clone)]
pub struct NodeAccuracyStats {
    pub net_id: u64,
    pub predictor_name: String,
    pub total_predictions: u64,
    pub hits: u64,               // within tolerance
    pub misses: u64,             // outside tolerance
    pub total_error: f64,        // cumulative absolute error
    pub max_error: f64,          // worst-case error
    pub error_history: Vec<f64>, // recent errors for trend analysis
}

impl NodeAccuracyStats {
    pub fn new(net_id: u64, predictor_name: &str) -> Self {
        Self {
            net_id,
            predictor_name: predictor_name.to_string(),
            total_predictions: 0,
            hits: 0,
            misses: 0,
            total_error: 0.0,
            max_error: 0.0,
            error_history: Vec::new(),
        }
    }

    /// Record a prediction result.
    pub fn record(&mut self, predicted: f64, actual: f64, reltol: f64) {
        let error = (predicted - actual).abs();
        let relative_error = if actual.abs() > 1e-15 {
            error / actual.abs()
        } else {
            error
        };

        self.total_predictions += 1;
        self.total_error += error;
        if error > self.max_error {
            self.max_error = error;
        }

        if relative_error < reltol {
            self.hits += 1;
        } else {
            self.misses += 1;
        }

        self.error_history.push(relative_error);
        if self.error_history.len() > 32 {
            self.error_history.remove(0);
        }
    }

    /// Hit rate (0.0–1.0).
    pub fn hit_rate(&self) -> f64 {
        if self.total_predictions == 0 {
            return 1.0;
        }
        self.hits as f64 / self.total_predictions as f64
    }

    /// Mean absolute error.
    pub fn mean_error(&self) -> f64 {
        if self.total_predictions == 0 {
            return 0.0;
        }
        self.total_error / self.total_predictions as f64
    }

    /// Recent error trend (positive = worsening, negative = improving).
    pub fn error_trend(&self) -> f64 {
        let n = self.error_history.len();
        if n < 4 {
            return 0.0;
        }
        let half = n / 2;
        let recent_avg: f64 = self.error_history[half..].iter().sum::<f64>() / (n - half) as f64;
        let older_avg: f64 = self.error_history[..half].iter().sum::<f64>() / half as f64;
        recent_avg - older_avg
    }
}

impl PredictionAccuracyLogger {
    pub fn new() -> Self {
        Self {
            node_stats: Vec::new(),
        }
    }

    /// Get or create stats entry for a node.
    pub fn get_or_create(&mut self, net_id: u64, predictor_name: &str) -> &mut NodeAccuracyStats {
        let pos = self.node_stats.iter().position(|s| s.net_id == net_id);
        match pos {
            Some(i) => &mut self.node_stats[i],
            None => {
                self.node_stats
                    .push(NodeAccuracyStats::new(net_id, predictor_name));
                self.node_stats.last_mut().unwrap()
            }
        }
    }

    /// Record a prediction outcome.
    pub fn record(
        &mut self,
        net_id: u64,
        predictor_name: &str,
        predicted: f64,
        actual: f64,
        reltol: f64,
    ) {
        let stats = self.get_or_create(net_id, predictor_name);
        stats.record(predicted, actual, reltol);
    }

    /// Overall hit rate across all nodes.
    pub fn overall_hit_rate(&self) -> f64 {
        let total_hits: u64 = self.node_stats.iter().map(|s| s.hits).sum();
        let total: u64 = self.node_stats.iter().map(|s| s.total_predictions).sum();
        if total == 0 {
            return 1.0;
        }
        total_hits as f64 / total as f64
    }

    /// Summary report.
    pub fn summary(&self) -> String {
        let mut lines = vec![format!(
            "Prediction Accuracy: {:.1}% overall ({} nodes)",
            self.overall_hit_rate() * 100.0,
            self.node_stats.len()
        )];
        for s in &self.node_stats {
            lines.push(format!(
                "  net {} [{}]: {:.1}% hit rate, mean_err={:.2e}, max_err={:.2e}, trend={:.2e}",
                s.net_id,
                s.predictor_name,
                s.hit_rate() * 100.0,
                s.mean_error(),
                s.max_error,
                s.error_trend()
            ));
        }
        lines.join("\n")
    }
}

// ─── Signal Classifier ─────────────────────────────────────────────

/// Classification of a signal's behavior for predictor selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SignalClass {
    /// Smooth, slowly varying (e.g., bias voltages) → Linear
    Smooth,
    /// Exponential decay/charge (e.g., RC transients) → Exponential
    Decaying,
    /// Stiff or oscillatory (e.g., ring oscillator, clock) → RungeKutta
    Stiff,
    /// Unknown or insufficient data → Linear (safe default)
    Unknown,
}

/// Classify a signal based on its recent history.
pub fn classify_signal(state: &SignalState) -> SignalClass {
    if state.history.len() < 4 {
        return SignalClass::Unknown;
    }

    let n = state.history.len();

    // Compute recent gradient magnitudes
    let mut gradients = Vec::with_capacity(n - 1);
    for i in 1..n {
        let dt = state.history[i].0 - state.history[i - 1].0;
        if dt.abs() > 1e-30 {
            gradients.push((state.history[i].1 - state.history[i - 1].1) / dt);
        }
    }

    if gradients.len() < 3 {
        return SignalClass::Unknown;
    }

    // Check for sign changes in gradient (oscillatory behavior → Stiff)
    let mut sign_changes = 0;
    for i in 1..gradients.len() {
        if gradients[i] * gradients[i - 1] < 0.0 {
            sign_changes += 1;
        }
    }

    let sign_change_ratio = sign_changes as f64 / (gradients.len() - 1) as f64;
    if sign_change_ratio > 0.3 {
        return SignalClass::Stiff;
    }

    // Check for exponential-like behavior: |d²V/dt²| correlates with |dV/dt|
    // (In an exponential, d²V/dt² = dV/dt / τ, so the ratio is constant)
    let mut second_derivs = Vec::new();
    for i in 1..gradients.len() {
        let dt = state.history[i + 1].0 - state.history[i].0;
        if dt.abs() > 1e-30 {
            second_derivs.push((gradients[i] - gradients[i - 1]) / dt);
        }
    }

    if second_derivs.len() >= 2 {
        // Check if second derivative has consistent sign (monotonic curvature)
        let consistent_sign = second_derivs.windows(2).all(|w| w[0] * w[1] >= 0.0);

        // And gradients are decaying in magnitude
        let avg_gradient_early: f64 = gradients[..gradients.len() / 2]
            .iter()
            .map(|g| g.abs())
            .sum::<f64>()
            / (gradients.len() / 2) as f64;
        let avg_gradient_late: f64 = gradients[gradients.len() / 2..]
            .iter()
            .map(|g| g.abs())
            .sum::<f64>()
            / (gradients.len() - gradients.len() / 2) as f64;

        if consistent_sign && avg_gradient_late < avg_gradient_early * 0.7 {
            return SignalClass::Decaying;
        }
    }

    // Check curvature magnitude relative to gradient (high curvature → Stiff)
    let max_curvature = state.second_derivative.abs();
    let max_gradient = state.gradient.abs();

    if max_gradient > 1e-15 {
        let stiffness_ratio = max_curvature / (max_gradient * max_gradient);
        if stiffness_ratio > 1e6 {
            return SignalClass::Stiff;
        }
    }

    SignalClass::Smooth
}

// ─── Predictor Selector ────────────────────────────────────────────

/// Selects the optimal predictor for a ghost node based on signal classification.
#[derive(Debug)]
pub struct PredictorSelector {
    pub linear: LinearPredictor,
    pub rk: RungeKuttaPredictor,
    /// Default time constant for exponential predictor (estimated from signal)
    pub default_tau: f64,
}

impl PredictorSelector {
    pub fn new() -> Self {
        Self {
            linear: LinearPredictor,
            rk: RungeKuttaPredictor,
            default_tau: 10e-12, // 10ps default
        }
    }

    /// Select the best predictor and return prediction.
    pub fn predict(&self, state: &SignalState, target_time: f64) -> (f64, &str) {
        let class = classify_signal(state);

        match class {
            SignalClass::Smooth | SignalClass::Unknown => {
                (self.linear.predict(state, target_time), "Linear")
            }
            SignalClass::Decaying => {
                // Estimate tau from recent gradient
                let tau = if state.gradient.abs() > 1e-15 && state.value.abs() > 1e-15 {
                    (state.value / state.gradient).abs()
                } else {
                    self.default_tau
                };

                let exp = ExponentialPredictor {
                    tau,
                    v_asymptote: 0.0, // TODO: Estimate from signal trend
                };
                (exp.predict(state, target_time), "Exponential")
            }
            SignalClass::Stiff => (self.rk.predict(state, target_time), "RungeKutta"),
        }
    }
}

// ─── Adaptive Confidence Tuning ────────────────────────────────────

/// Manages adaptive speculative depth based on per-node prediction confidence.
#[derive(Debug, Clone)]
pub struct AdaptiveConfidence {
    /// Base speculative depth (from config)
    pub base_depth: u32,
    /// Minimum allowed depth (never go below this)
    pub min_depth: u32,
    /// Maximum allowed depth
    pub max_depth: u32,
    /// Current confidence scores per node: net_id → confidence (0.0–1.0)
    confidences: Vec<(u64, f64)>,
}

impl AdaptiveConfidence {
    pub fn new(base_depth: u32) -> Self {
        Self {
            base_depth,
            min_depth: 1,
            max_depth: base_depth * 3,
            confidences: Vec::new(),
        }
    }

    /// Update confidence for a node based on prediction accuracy.
    pub fn update(&mut self, net_id: u64, was_accurate: bool) {
        let entry = self.confidences.iter_mut().find(|(id, _)| *id == net_id);

        match entry {
            Some((_, conf)) => {
                if was_accurate {
                    // Slowly increase confidence: EWMA with α=0.1
                    *conf = *conf * 0.9 + 1.0 * 0.1;
                } else {
                    // Rapidly decrease confidence: EWMA with α=0.3
                    *conf = *conf * 0.7;
                }
                *conf = conf.clamp(0.0, 1.0);
            }
            None => {
                let initial = if was_accurate { 0.5 } else { 0.1 };
                self.confidences.push((net_id, initial));
            }
        }
    }

    /// Get the effective speculative depth based on lowest-confidence node.
    pub fn effective_depth(&self) -> u32 {
        if self.confidences.is_empty() {
            return self.base_depth;
        }

        let min_confidence = self
            .confidences
            .iter()
            .map(|(_, c)| *c)
            .fold(1.0f64, |a, b| a.min(b));

        // Scale depth by confidence: high confidence → deeper speculation
        let scaled = (self.base_depth as f64 * (0.5 + min_confidence * 1.5)) as u32;

        scaled.clamp(self.min_depth, self.max_depth)
    }

    /// Get confidence for a specific node.
    pub fn confidence_for(&self, net_id: u64) -> f64 {
        self.confidences
            .iter()
            .find(|(id, _)| *id == net_id)
            .map(|(_, c)| *c)
            .unwrap_or(0.5) // Default 50% confidence
    }

    /// Summary of current confidence state.
    pub fn summary(&self) -> String {
        format!(
            "Adaptive depth: {} (base={}, range={}–{}, {} nodes tracked)",
            self.effective_depth(),
            self.base_depth,
            self.min_depth,
            self.max_depth,
            self.confidences.len()
        )
    }
}

// ─── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_predictor() {
        let mut state = SignalState::new();
        state.value = 1.0;
        state.gradient = 1e9; // 1V/ns
        state.time = 0.0;

        let p = LinearPredictor;
        let v = p.predict(&state, 1e-9);
        assert!((v - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_predictor() {
        let mut state = SignalState::new();
        state.value = 1.0;
        state.time = 0.0;

        let p = ExponentialPredictor {
            tau: 1e-9,
            v_asymptote: 0.0,
        };
        let v = p.predict(&state, 1e-9);
        let expected = (-1.0_f64).exp();
        assert!((v - expected).abs() < 1e-6, "Expected {expected}, got {v}");
    }

    #[test]
    fn test_rk_predictor_linear_signal() {
        // For a perfectly linear signal, RK should match linear predictor
        let mut state = SignalState::new();
        for i in 0..8 {
            let t = i as f64 * 1e-12;
            let v = 0.5 + t * 1e9; // Linear ramp
            state.record(t, v);
        }

        let rk = RungeKuttaPredictor;
        let predicted = rk.predict(&state, 8e-12);
        let expected = 0.5 + 8e-12 * 1e9; // = 8.5
        assert!(
            (predicted - expected).abs() < 0.1,
            "RK on linear signal: expected ~{expected}, got {predicted}"
        );
    }

    #[test]
    fn test_rk_predictor_quadratic_signal() {
        // For a quadratic signal, RK should capture curvature
        let mut state = SignalState::new();
        for i in 0..8 {
            let t = i as f64 * 1e-12;
            let v = t * t * 1e24; // Parabola: v = t²/1e-24
            state.record(t, v);
        }

        let rk = RungeKuttaPredictor;
        let predicted = rk.predict(&state, 8e-12);
        let expected = (8e-12_f64).powi(2) * 1e24; // = 64
                                                   // RK should be closer to truth than linear would be
        let lin = LinearPredictor;
        let linear_pred = lin.predict(&state, 8e-12);

        let rk_err = (predicted - expected).abs();
        let lin_err = (linear_pred - expected).abs();
        assert!(
            rk_err <= lin_err + 1.0,
            "RK error ({rk_err:.2}) should be <= linear error ({lin_err:.2})"
        );
    }

    #[test]
    fn test_signal_classifier_smooth() {
        let mut state = SignalState::new();
        // Constant voltage (very smooth)
        for i in 0..8 {
            state.record(i as f64 * 1e-12, 1.8);
        }
        let class = classify_signal(&state);
        assert_eq!(class, SignalClass::Smooth);
    }

    #[test]
    fn test_signal_classifier_decaying() {
        let mut state = SignalState::new();
        // Exponential decay
        for i in 0..10 {
            let t = i as f64 * 1e-12;
            let v = 1.8 * (-t / 5e-12).exp();
            state.record(t, v);
        }
        let class = classify_signal(&state);
        assert!(
            class == SignalClass::Decaying || class == SignalClass::Smooth,
            "Decaying signal classified as {class:?}"
        );
    }

    #[test]
    fn test_signal_classifier_stiff() {
        let mut state = SignalState::new();
        // Oscillating signal (sign changes in gradient)
        for i in 0..10 {
            let t = i as f64 * 1e-12;
            let v = 0.9 + 0.1 * (t * 1e12 * std::f64::consts::PI).sin();
            state.record(t, v);
        }
        let class = classify_signal(&state);
        assert_eq!(class, SignalClass::Stiff, "Oscillating → Stiff");
    }

    #[test]
    fn test_predictor_selector() {
        let selector = PredictorSelector::new();

        // Smooth signal → Linear
        let mut smooth = SignalState::new();
        for i in 0..8 {
            smooth.record(i as f64 * 1e-12, 1.8);
        }
        let (_, name) = selector.predict(&smooth, 9e-12);
        assert_eq!(name, "Linear");
    }

    #[test]
    fn test_accuracy_logger() {
        let mut logger = PredictionAccuracyLogger::new();

        // Node 1: good predictions
        logger.record(1, "Linear", 1.0, 1.001, 1e-2);
        logger.record(1, "Linear", 1.5, 1.502, 1e-2);

        // Node 2: bad predictions
        logger.record(2, "Linear", 1.0, 2.0, 1e-2);
        logger.record(2, "Linear", 1.5, 3.0, 1e-2);

        assert_eq!(logger.node_stats.len(), 2);

        let stats1 = &logger.node_stats[0];
        assert_eq!(stats1.hits, 2);
        assert_eq!(stats1.misses, 0);
        assert!((stats1.hit_rate() - 1.0).abs() < 1e-6);

        let stats2 = &logger.node_stats[1];
        assert_eq!(stats2.hits, 0);
        assert_eq!(stats2.misses, 2);

        // Overall: 2 hits out of 4
        assert!((logger.overall_hit_rate() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_adaptive_confidence() {
        let mut ac = AdaptiveConfidence::new(5);

        // Initial depth should be base
        assert_eq!(ac.effective_depth(), 5);

        // Good predictions increase depth
        for _ in 0..20 {
            ac.update(1, true);
        }
        assert!(
            ac.effective_depth() >= 5,
            "High confidence should maintain or increase depth"
        );

        // Bad predictions decrease depth
        for _ in 0..10 {
            ac.update(2, false);
        }
        let depth_after_misses = ac.effective_depth();
        assert!(
            depth_after_misses <= 5,
            "Low confidence should reduce depth, got {depth_after_misses}"
        );
    }

    #[test]
    fn test_adaptive_confidence_recovery() {
        let mut ac = AdaptiveConfidence::new(5);

        // Drive confidence down
        for _ in 0..10 {
            ac.update(1, false);
        }
        let low = ac.effective_depth();

        // Gradually recover
        for _ in 0..30 {
            ac.update(1, true);
        }
        let recovered = ac.effective_depth();
        assert!(
            recovered > low,
            "Confidence should recover: low={low}, recovered={recovered}"
        );
    }

    #[test]
    fn test_signal_state_recording() {
        let mut state = SignalState::new();
        state.record(0.0, 0.0);
        state.record(1e-12, 1.0);
        state.record(2e-12, 2.0);

        assert_eq!(state.history.len(), 3);
        assert!((state.value - 2.0).abs() < 1e-15);
        // Gradient should be ~1V/ps = 1e12 V/s
        assert!(
            (state.gradient - 1e12).abs() < 1e6,
            "Gradient: {}",
            state.gradient
        );
    }
}
