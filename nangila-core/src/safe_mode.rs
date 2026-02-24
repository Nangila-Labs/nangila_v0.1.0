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
    // === Canary Mode ===
    /// Fraction of layers to enable as "canary" during recovery (0.0-1.0)
    pub canary_fraction: f32,
    /// Steps to monitor canary layers before full re-enable
    pub canary_steps: usize,
    /// Maximum canary failures before aborting recovery
    pub canary_max_failures: u32,
}

impl Default for SafeModeConfig {
    fn default() -> Self {
        Self {
            check_interval: 100,
            divergence_threshold: 0.005, // 0.5%
            max_consecutive_failures: 3,
            recovery_cooldown: 500,
            // Canary mode defaults
            canary_fraction: 0.1,   // Start with 10% of layers
            canary_steps: 100,      // Monitor for 100 steps
            canary_max_failures: 2, // Max 2 failures in canary
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
            canary_fraction: 0.05, // 5% of layers
            canary_steps: 200,
            canary_max_failures: 1,
        }
    }

    /// Relaxed settings (more tolerance)
    pub fn relaxed() -> Self {
        Self {
            check_interval: 200,
            divergence_threshold: 0.01, // 1%
            max_consecutive_failures: 5,
            recovery_cooldown: 200,
            canary_fraction: 0.2, // 20% of layers
            canary_steps: 50,
            canary_max_failures: 3,
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
    /// Canary mode: enable compression for subset of layers
    CanaryTest,
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
    /// Canary mode: testing subset of layers
    Canary,
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
    /// Exponential moving average of loss (short-term)
    ema_loss_short: Option<f32>,
    /// Exponential moving average of loss (long-term, for comparison)
    ema_loss_long: Option<f32>,
    /// EMA smoothing factor for short-term (faster response)
    ema_alpha_short: f32,
    /// EMA smoothing factor for long-term (slower, more stable)
    ema_alpha_long: f32,
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
            ema_loss_short: None,
            ema_loss_long: None,
            ema_alpha_short: 0.3, // Fast response to changes
            ema_alpha_long: 0.05, // Slow, stable baseline
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
        // Update EMAs
        self.ema_loss_short = Some(match self.ema_loss_short {
            Some(ema) => self.ema_alpha_short * val_loss + (1.0 - self.ema_alpha_short) * ema,
            None => val_loss,
        });
        self.ema_loss_long = Some(match self.ema_loss_long {
            Some(ema) => self.ema_alpha_long * val_loss + (1.0 - self.ema_alpha_long) * ema,
            None => val_loss,
        });

        self.last_loss = Some(val_loss);
        self.last_check_step = step;

        match self.state {
            SafeModeState::Active => self.check_active(val_loss),
            SafeModeState::Fallback => self.check_fallback(val_loss),
            SafeModeState::Recovering => self.check_recovering(step),
            SafeModeState::Canary => self.check_canary(val_loss),
        }
    }

    /// Check during canary mode (testing subset of layers)
    fn check_canary(&mut self, val_loss: f32) -> SafeModeAction {
        // Compare short-term EMA to long-term EMA (not fixed baseline)
        let ema_long = self.ema_loss_long.unwrap_or(val_loss);
        let ema_short = self.ema_loss_short.unwrap_or(val_loss);

        // Divergence = short-term trending worse than long-term
        let delta = (ema_short - ema_long) / ema_long.abs().max(1e-6);

        if delta > self.config.divergence_threshold {
            // Canary failed - go back to fallback
            self.failure_count += 1;
            if self.failure_count >= self.config.canary_max_failures {
                tracing::warn!("Safe Mode: canary failed, returning to fallback");
                self.state = SafeModeState::Fallback;
                self.failure_count = 0;
                return SafeModeAction::TriggerFallback;
            }
        } else {
            self.failure_count = 0;
            self.cooldown_remaining = self
                .cooldown_remaining
                .saturating_sub(self.config.check_interval);

            // Canary succeeded for long enough - go fully active
            if self.cooldown_remaining == 0 {
                tracing::info!("Safe Mode: canary successful, enabling full compression");
                self.state = SafeModeState::Active;
                return SafeModeAction::RecoveryComplete;
            }
        }

        SafeModeAction::CanaryTest
    }

    /// Check during active compression
    fn check_active(&mut self, val_loss: f32) -> SafeModeAction {
        // Establish baseline if not set (use long-term EMA)
        let baseline = match self.baseline_loss {
            Some(b) => b,
            None => {
                // Use long-term EMA as baseline once it's established
                if let Some(ema_long) = self.ema_loss_long {
                    self.baseline_loss = Some(ema_long);
                    tracing::info!("Safe Mode: baseline loss established at {:.6}", ema_long);
                }
                return SafeModeAction::Continue;
            }
        };

        // Compare short-term EMA to long-term EMA
        // Divergence = short-term trending significantly worse than long-term
        let ema_long = self.ema_loss_long.unwrap_or(baseline);
        let ema_short = self.ema_loss_short.unwrap_or(val_loss);

        let delta = (ema_short - ema_long) / ema_long.abs().max(1e-6);

        if delta > self.config.divergence_threshold {
            self.failure_count += 1;
            tracing::warn!(
                "Safe Mode: divergence detected (short EMA {:.6} vs long EMA {:.6}, delta {:.2}% > {:.2}%), failure {}/{}",
                ema_short,
                ema_long,
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

            // Update baseline to track improving loss (use long-term EMA)
            if ema_long < baseline {
                self.baseline_loss = Some(ema_long);
            }
        }

        SafeModeAction::Continue
    }

    /// Check during fallback (compression disabled)
    fn check_fallback(&mut self, val_loss: f32) -> SafeModeAction {
        // Compare short-term to long-term EMA
        let ema_long = self.ema_loss_long.unwrap_or(val_loss);
        let ema_short = self.ema_loss_short.unwrap_or(val_loss);

        let delta = (ema_short - ema_long) / ema_long.abs().max(1e-6);

        // Look for stability (short-term not diverging from long-term)
        if delta <= self.config.divergence_threshold {
            self.failure_count = 0;
            self.state = SafeModeState::Recovering;
            self.cooldown_remaining = self.config.recovery_cooldown;
            tracing::info!(
                "Safe Mode: loss stabilized (short EMA {:.6} vs long EMA {:.6}), entering recovery cooldown ({} steps)",
                ema_short,
                ema_long,
                self.config.recovery_cooldown
            );
        }

        SafeModeAction::Continue
    }

    /// Check during recovery cooldown
    fn check_recovering(&mut self, _step: usize) -> SafeModeAction {
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
            ema_loss_short: self.ema_loss_short,
            ema_loss_long: self.ema_loss_long,
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
    pub ema_loss_short: Option<f32>,
    pub ema_loss_long: Option<f32>,
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

        // First few checks establish EMAs
        safe.check(100, 1.0);
        safe.check(200, 1.0);
        safe.check(300, 1.0);

        // Baseline should be established from long-term EMA
        assert!(safe.baseline_loss.is_some());
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
        for i in 0..10 {
            safe.check(i * 100, 1.0);
        }

        // Sudden divergence (short-term EMA will spike)
        for i in 10..15 {
            safe.check(i * 100, 1.5); // 50% higher
        }

        // Should eventually trigger fallback
        assert!(!safe.should_compress());
    }

    #[test]
    fn test_recovery() {
        let config = SafeModeConfig {
            divergence_threshold: 0.05,
            max_consecutive_failures: 1,
            recovery_cooldown: 100,
            check_interval: 50,
            canary_fraction: 0.1,
            canary_steps: 50,
            canary_max_failures: 1,
        };
        let mut safe = SafeMode::new(config);

        // Establish baseline
        for i in 0..10 {
            safe.check(i * 50, 1.0);
        }

        // Trigger divergence
        for i in 10..15 {
            safe.check(i * 50, 1.5);
        }

        assert_eq!(safe.state(), SafeModeState::Fallback);

        // Loss stabilizes → enter recovery
        for i in 15..20 {
            safe.check(i * 50, 1.0);
        }

        // Should eventually recover
        assert!(matches!(
            safe.state(),
            SafeModeState::Recovering | SafeModeState::Active
        ));
    }

    #[test]
    fn test_failure_count_resets() {
        let config = SafeModeConfig {
            divergence_threshold: 0.05,
            max_consecutive_failures: 3,
            ..Default::default()
        };
        let mut safe = SafeMode::new(config);

        // Establish baseline
        for i in 0..10 {
            safe.check(i * 100, 1.0);
        }

        // Spike then recover
        safe.check(1000, 1.5); // Spike
        safe.check(1100, 1.5); // Still high
        assert!(safe.failure_count > 0);

        // Good loss resets count (after EMA stabilizes)
        for i in 12..20 {
            safe.check(i * 100, 1.0);
        }
        assert_eq!(safe.failure_count, 0);
    }
}
