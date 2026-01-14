//! Safe Mode: Convergence Monitoring and Auto-Fallback
//!
//! Monitors validation loss to detect divergence and automatically
//! disables compression if training starts to diverge.

/// Configuration for Safe Mode
#[derive(Debug, Clone)]
pub struct SafeModeConfig {
    /// How often to check validation loss (in steps)
    pub check_interval: usize,
    /// Maximum allowed divergence from baseline (0.005 = 0.5%)
    pub divergence_threshold: f32,
    /// Consecutive failures before triggering fallback
    pub max_consecutive_failures: u32,
    /// Steps to wait after recovery before re-enabling compression
    pub recovery_cooldown: usize,
}

impl Default for SafeModeConfig {
    fn default() -> Self {
        Self {
            check_interval: 100,
            divergence_threshold: 0.005, // 0.5%
            max_consecutive_failures: 3,
            recovery_cooldown: 500,
        }
    }
}

impl SafeModeConfig {
    /// Conservative settings (stricter monitoring)
    pub fn conservative() -> Self {
        Self {
            check_interval: 50,
            divergence_threshold: 0.003, // 0.3%
            max_consecutive_failures: 2,
            recovery_cooldown: 1000,
        }
    }

    /// Relaxed settings (more tolerance)
    pub fn relaxed() -> Self {
        Self {
            check_interval: 200,
            divergence_threshold: 0.01, // 1%
            max_consecutive_failures: 5,
            recovery_cooldown: 200,
        }
    }
}

/// Action to take after Safe Mode check
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafeModeAction {
    /// Continue normal compression
    Continue,
    /// Trigger fallback: disable compression
    TriggerFallback,
    /// Recovery complete: re-enable compression
    RecoveryComplete,
    /// No action needed (not time to check yet)
    NoCheck,
}

/// Safe Mode state machine
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafeModeState {
    /// Normal operation, compression enabled
    Active,
    /// Compression disabled due to divergence
    Fallback,
    /// In cooldown period after fallback
    Recovering,
}

/// Safe Mode convergence monitor
#[derive(Debug)]
pub struct SafeMode {
    /// Configuration
    config: SafeModeConfig,
    /// Current state
    state: SafeModeState,
    /// Baseline validation loss (established during warmup)
    baseline_loss: Option<f32>,
    /// Last recorded validation loss
    last_loss: Option<f32>,
    /// Exponential moving average of loss
    ema_loss: Option<f32>,
    /// EMA smoothing factor
    ema_alpha: f32,
    /// Consecutive failure count
    failure_count: u32,
    /// Cooldown counter (steps remaining)
    cooldown_remaining: usize,
    /// Last check step
    last_check_step: usize,
    /// Total fallback count (for diagnostics)
    total_fallbacks: u32,
}

impl SafeMode {
    /// Create a new Safe Mode monitor
    pub fn new(config: SafeModeConfig) -> Self {
        Self {
            config,
            state: SafeModeState::Active,
            baseline_loss: None,
            last_loss: None,
            ema_loss: None,
            ema_alpha: 0.1, // Slow EMA for stability
            failure_count: 0,
            cooldown_remaining: 0,
            last_check_step: 0,
            total_fallbacks: 0,
        }
    }

    /// Check if it's time to evaluate validation loss
    pub fn should_check(&self, step: usize) -> bool {
        step > 0 && step % self.config.check_interval == 0
    }

    /// Report validation loss and get recommended action
    ///
    /// Call this every `check_interval` steps with the validation loss.
    pub fn check(&mut self, step: usize, val_loss: f32) -> SafeModeAction {
        // Update EMA
        self.ema_loss = Some(match self.ema_loss {
            Some(ema) => self.ema_alpha * val_loss + (1.0 - self.ema_alpha) * ema,
            None => val_loss,
        });
        self.last_loss = Some(val_loss);
        self.last_check_step = step;

        match self.state {
            SafeModeState::Active => self.check_active(val_loss),
            SafeModeState::Fallback => self.check_fallback(val_loss),
            SafeModeState::Recovering => self.check_recovering(step),
        }
    }

    /// Check during active compression
    fn check_active(&mut self, val_loss: f32) -> SafeModeAction {
        // Establish baseline if not set
        let baseline = match self.baseline_loss {
            Some(b) => b,
            None => {
                self.baseline_loss = Some(val_loss);
                tracing::info!("Safe Mode: baseline loss established at {:.6}", val_loss);
                return SafeModeAction::Continue;
            }
        };

        // Check for divergence
        let delta = (val_loss - baseline) / baseline;

        if delta > self.config.divergence_threshold {
            self.failure_count += 1;
            tracing::warn!(
                "Safe Mode: divergence detected ({:.2}% > {:.2}%), failure {}/{}",
                delta * 100.0,
                self.config.divergence_threshold * 100.0,
                self.failure_count,
                self.config.max_consecutive_failures
            );

            if self.failure_count >= self.config.max_consecutive_failures {
                self.trigger_fallback();
                return SafeModeAction::TriggerFallback;
            }
        } else {
            // Reset failure count on success
            if self.failure_count > 0 {
                tracing::info!("Safe Mode: loss stabilized, resetting failure count");
            }
            self.failure_count = 0;

            // Update baseline if loss improved (defensive)
            if val_loss < baseline {
                self.baseline_loss = Some(val_loss);
            }
        }

        SafeModeAction::Continue
    }

    /// Check during fallback (compression disabled)
    fn check_fallback(&mut self, val_loss: f32) -> SafeModeAction {
        let baseline = self.baseline_loss.unwrap_or(val_loss);
        let delta = (val_loss - baseline) / baseline;

        // Look for stability (loss back to normal)
        if delta <= self.config.divergence_threshold {
            self.failure_count = 0;
            self.state = SafeModeState::Recovering;
            self.cooldown_remaining = self.config.recovery_cooldown;
            tracing::info!(
                "Safe Mode: loss stabilized, entering recovery cooldown ({} steps)",
                self.config.recovery_cooldown
            );
        }

        SafeModeAction::Continue
    }

    /// Check during recovery cooldown
    fn check_recovering(&mut self, step: usize) -> SafeModeAction {
        // Decrement cooldown by check_interval (we're called every check_interval steps)
        let steps_elapsed = self.config.check_interval;
        self.cooldown_remaining = self.cooldown_remaining.saturating_sub(steps_elapsed);

        if self.cooldown_remaining == 0 {
            self.state = SafeModeState::Active;
            tracing::info!("Safe Mode: recovery complete, re-enabling compression");
            return SafeModeAction::RecoveryComplete;
        }

        SafeModeAction::Continue
    }

    /// Trigger fallback mode
    fn trigger_fallback(&mut self) {
        self.state = SafeModeState::Fallback;
        self.failure_count = 0;
        self.total_fallbacks += 1;
        tracing::warn!(
            "Safe Mode: FALLBACK TRIGGERED (total: {}). Compression disabled.",
            self.total_fallbacks
        );
    }

    /// Check if compression should be enabled
    pub fn should_compress(&self) -> bool {
        matches!(self.state, SafeModeState::Active)
    }

    /// Get current state
    pub fn state(&self) -> SafeModeState {
        self.state
    }

    /// Get diagnostic statistics
    pub fn stats(&self) -> SafeModeStats {
        SafeModeStats {
            state: self.state,
            baseline_loss: self.baseline_loss,
            last_loss: self.last_loss,
            ema_loss: self.ema_loss,
            failure_count: self.failure_count,
            cooldown_remaining: self.cooldown_remaining,
            total_fallbacks: self.total_fallbacks,
        }
    }

    /// Manually trigger fallback (for testing/emergency)
    pub fn force_fallback(&mut self) {
        self.trigger_fallback();
    }

    /// Manually reset to active state
    pub fn reset(&mut self) {
        self.state = SafeModeState::Active;
        self.failure_count = 0;
        self.cooldown_remaining = 0;
    }
}

/// Safe Mode statistics for diagnostics
#[derive(Debug, Clone)]
pub struct SafeModeStats {
    pub state: SafeModeState,
    pub baseline_loss: Option<f32>,
    pub last_loss: Option<f32>,
    pub ema_loss: Option<f32>,
    pub failure_count: u32,
    pub cooldown_remaining: usize,
    pub total_fallbacks: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_baseline_establishment() {
        let mut safe = SafeMode::new(SafeModeConfig::default());

        // First check establishes baseline
        let action = safe.check(100, 1.0);
        assert_eq!(action, SafeModeAction::Continue);
        assert_eq!(safe.baseline_loss, Some(1.0));
    }

    #[test]
    fn test_divergence_detection() {
        let config = SafeModeConfig {
            divergence_threshold: 0.05, // 5%
            max_consecutive_failures: 2,
            ..Default::default()
        };
        let mut safe = SafeMode::new(config);

        // Establish baseline
        safe.check(100, 1.0);

        // First divergence
        let action = safe.check(200, 1.1); // 10% above baseline
        assert_eq!(action, SafeModeAction::Continue);
        assert_eq!(safe.failure_count, 1);

        // Second divergence triggers fallback
        let action = safe.check(300, 1.1);
        assert_eq!(action, SafeModeAction::TriggerFallback);
        assert!(!safe.should_compress());
    }

    #[test]
    fn test_recovery() {
        let config = SafeModeConfig {
            divergence_threshold: 0.05,
            max_consecutive_failures: 1,
            recovery_cooldown: 100,
            check_interval: 50,
        };
        let mut safe = SafeMode::new(config);

        // Establish baseline and trigger fallback
        safe.check(50, 1.0);
        safe.check(100, 1.2); // Diverges

        assert_eq!(safe.state(), SafeModeState::Fallback);

        // Loss stabilizes → enter recovery (cooldown starts at 100 steps)
        safe.check(150, 1.0);
        assert_eq!(safe.state(), SafeModeState::Recovering);
        assert_eq!(safe.cooldown_remaining, 100);

        // Partial cooldown: 50 steps elapsed (150 → 200), 50 remaining
        let action = safe.check(200, 1.0);
        assert_eq!(action, SafeModeAction::Continue);
        assert_eq!(safe.cooldown_remaining, 50);

        // Full cooldown: 100 steps from recovery (150 → 250)
        let action = safe.check(250, 1.0);
        assert_eq!(action, SafeModeAction::RecoveryComplete);
        assert!(safe.should_compress());
    }

    #[test]
    fn test_failure_count_resets() {
        let config = SafeModeConfig {
            divergence_threshold: 0.05,
            max_consecutive_failures: 3,
            ..Default::default()
        };
        let mut safe = SafeMode::new(config);

        safe.check(100, 1.0); // Baseline
        safe.check(200, 1.1); // Failure 1
        safe.check(300, 1.1); // Failure 2
        assert_eq!(safe.failure_count, 2);

        // Good loss resets count
        safe.check(400, 1.0);
        assert_eq!(safe.failure_count, 0);
    }
}
