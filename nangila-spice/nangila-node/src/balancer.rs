//! Load Balancer — Work Stealing & Straggler Detection
//!
//! Monitors per-partition solve times and migrates sub-blocks
//! from slow (straggler) partitions to fast ones.
//!
//! Architecture:
//!   - Each partition reports its solve time per timestep
//!   - BalanceMonitor tracks moving averages and detects stragglers
//!   - WorkStealer generates migration plans (which sub-blocks to move)
//!   - Integrates with FabricRouter for live rebalancing
//!
//! Phase 2, Sprint 8 deliverable.

use std::collections::HashMap;
use tracing::{debug, info, warn};

// ─── Partition Metrics ─────────────────────────────────────────────

/// Per-partition performance metrics.
#[derive(Debug, Clone)]
pub struct PartitionMetrics {
    pub partition_id: u32,
    /// Number of circuit elements in this partition
    pub element_count: u32,
    /// Number of boundary (ghost) nodes
    pub ghost_count: u32,
    /// Recent solve times (seconds per timestep)
    solve_times: Vec<f64>,
    /// Exponentially weighted moving average solve time
    ewma_solve_time: f64,
    /// EWMA smoothing factor
    alpha: f64,
    /// Total timesteps completed
    pub steps_completed: u64,
    /// Number of rollbacks
    pub rollback_count: u64,
}

impl PartitionMetrics {
    pub fn new(partition_id: u32, element_count: u32, ghost_count: u32) -> Self {
        Self {
            partition_id,
            element_count,
            ghost_count,
            solve_times: Vec::with_capacity(64),
            ewma_solve_time: 0.0,
            alpha: 0.1,
            steps_completed: 0,
            rollback_count: 0,
        }
    }

    /// Record a solve time for a single timestep.
    pub fn record_solve_time(&mut self, time_secs: f64) {
        self.solve_times.push(time_secs);
        if self.solve_times.len() > 64 {
            self.solve_times.remove(0);
        }

        // Update EWMA
        if self.ewma_solve_time == 0.0 {
            self.ewma_solve_time = time_secs;
        } else {
            self.ewma_solve_time =
                self.alpha * time_secs + (1.0 - self.alpha) * self.ewma_solve_time;
        }

        self.steps_completed += 1;
    }

    /// Get the current EWMA solve time.
    pub fn avg_solve_time(&self) -> f64 {
        self.ewma_solve_time
    }

    /// Get solve time variance (for straggler detection).
    pub fn solve_time_variance(&self) -> f64 {
        if self.solve_times.len() < 2 {
            return 0.0;
        }
        let mean = self.solve_times.iter().sum::<f64>() / self.solve_times.len() as f64;
        let variance = self
            .solve_times
            .iter()
            .map(|t| (t - mean).powi(2))
            .sum::<f64>()
            / (self.solve_times.len() - 1) as f64;
        variance
    }

    /// Throughput: elements solved per second.
    pub fn throughput(&self) -> f64 {
        if self.ewma_solve_time > 0.0 {
            self.element_count as f64 / self.ewma_solve_time
        } else {
            0.0
        }
    }
}

// ─── Straggler Detector ────────────────────────────────────────────

/// Configuration for straggler detection.
#[derive(Debug, Clone)]
pub struct StragglerConfig {
    /// A partition is a straggler if its solve time exceeds
    /// (mean + threshold_sigma * stddev) of all partitions.
    pub threshold_sigma: f64,
    /// Minimum solve time ratio (straggler / fastest) to trigger
    pub min_ratio: f64,
    /// Minimum number of samples before detection activates
    pub min_samples: usize,
}

impl Default for StragglerConfig {
    fn default() -> Self {
        Self {
            threshold_sigma: 1.5,
            min_ratio: 1.3,
            min_samples: 10,
        }
    }
}

/// Detects straggler partitions that are consistently slower.
#[derive(Debug, Clone)]
pub struct StragglerDetector {
    pub config: StragglerConfig,
}

impl StragglerDetector {
    pub fn new(config: StragglerConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(StragglerConfig::default())
    }

    /// Identify straggler partitions from current metrics.
    ///
    /// Returns: Vec of (straggler_id, slowdown_ratio).
    pub fn detect(&self, metrics: &[PartitionMetrics]) -> Vec<(u32, f64)> {
        if metrics.len() < 2 {
            return vec![];
        }

        // Check if we have enough samples
        let min_steps = metrics.iter().map(|m| m.steps_completed).min().unwrap_or(0);
        if (min_steps as usize) < self.config.min_samples {
            return vec![];
        }

        let solve_times: Vec<f64> = metrics.iter().map(|m| m.avg_solve_time()).collect();
        let mean = solve_times.iter().sum::<f64>() / solve_times.len() as f64;
        let variance =
            solve_times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / solve_times.len() as f64;
        let stddev = variance.sqrt();
        let fastest = solve_times.iter().cloned().fold(f64::MAX, f64::min);

        let threshold = mean + self.config.threshold_sigma * stddev;

        let mut stragglers = Vec::new();
        for (_i, m) in metrics.iter().enumerate() {
            let t = m.avg_solve_time();
            let ratio = if fastest > 0.0 { t / fastest } else { 1.0 };

            if t > threshold && ratio > self.config.min_ratio {
                stragglers.push((m.partition_id, ratio));
                debug!(
                    "Straggler detected: P{} ({:.2}x slower, {:.2e}s vs mean {:.2e}s)",
                    m.partition_id, ratio, t, mean
                );
            }
        }

        stragglers
    }
}

// ─── Work Stealing ─────────────────────────────────────────────────

/// A migration plan: move work from a straggler to a fast partition.
#[derive(Debug, Clone)]
pub struct MigrationPlan {
    /// Source partition (straggler)
    pub from_partition: u32,
    /// Target partition (underloaded)
    pub to_partition: u32,
    /// Number of elements to migrate
    pub elements_to_move: u32,
    /// Expected speedup from this migration
    pub expected_speedup: f64,
}

/// Work stealing engine that generates migration plans.
#[derive(Debug, Clone)]
pub struct WorkStealer {
    /// Maximum fraction of a partition's elements that can be migrated
    pub max_steal_fraction: f64,
    /// Minimum improvement ratio to justify a migration
    pub min_improvement: f64,
    /// Stats
    pub stats: WorkStealerStats,
}

#[derive(Debug, Clone, Default)]
pub struct WorkStealerStats {
    pub migrations_planned: u64,
    pub elements_migrated: u64,
    pub total_speedup_estimate: f64,
}

impl WorkStealer {
    pub fn new() -> Self {
        Self {
            max_steal_fraction: 0.25, // Move at most 25% of elements
            min_improvement: 1.1,     // Must improve by at least 10%
            stats: WorkStealerStats::default(),
        }
    }

    /// Generate migration plans to balance load across partitions.
    pub fn plan_migrations(
        &mut self,
        metrics: &[PartitionMetrics],
        stragglers: &[(u32, f64)],
    ) -> Vec<MigrationPlan> {
        if stragglers.is_empty() || metrics.len() < 2 {
            return vec![];
        }

        let mut plans = Vec::new();

        // Find underloaded partitions (fastest solvers)
        let mean_throughput =
            metrics.iter().map(|m| m.throughput()).sum::<f64>() / metrics.len() as f64;

        let mut fast_partitions: Vec<&PartitionMetrics> = metrics
            .iter()
            .filter(|m| m.throughput() > mean_throughput * 1.1)
            .collect();

        // Sort fastest first
        fast_partitions.sort_by(|a, b| {
            b.throughput()
                .partial_cmp(&a.throughput())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for &(straggler_id, slowdown_ratio) in stragglers {
            let straggler = match metrics.iter().find(|m| m.partition_id == straggler_id) {
                Some(m) => m,
                None => continue,
            };

            // Find best target
            if let Some(target) = fast_partitions.first() {
                let elements_to_move = (straggler.element_count as f64
                    * self.max_steal_fraction
                    * (1.0 - 1.0 / slowdown_ratio)) as u32;

                if elements_to_move == 0 {
                    continue;
                }

                let expected_speedup = slowdown_ratio.sqrt(); // Conservative estimate

                if expected_speedup >= self.min_improvement {
                    plans.push(MigrationPlan {
                        from_partition: straggler_id,
                        to_partition: target.partition_id,
                        elements_to_move,
                        expected_speedup,
                    });

                    self.stats.migrations_planned += 1;
                    self.stats.elements_migrated += elements_to_move as u64;
                    self.stats.total_speedup_estimate += expected_speedup;
                }
            }
        }

        plans
    }
}

// ─── Balance Monitor ───────────────────────────────────────────────

/// Top-level load balance monitor.
/// Combines metrics collection, straggler detection, and work stealing.
pub struct BalanceMonitor {
    pub metrics: Vec<PartitionMetrics>,
    pub detector: StragglerDetector,
    pub stealer: WorkStealer,
    /// How often to check (every N timesteps)
    pub check_interval: u64,
    /// Current timestep counter
    step_counter: u64,
    /// History of balance scores
    balance_history: Vec<f64>,
}

impl BalanceMonitor {
    pub fn new(num_partitions: u32) -> Self {
        let metrics: Vec<PartitionMetrics> = (0..num_partitions)
            .map(|i| PartitionMetrics::new(i, 0, 0))
            .collect();

        Self {
            metrics,
            detector: StragglerDetector::with_defaults(),
            stealer: WorkStealer::new(),
            check_interval: 50,
            step_counter: 0,
            balance_history: Vec::new(),
        }
    }

    /// Initialize metrics with actual element counts.
    pub fn set_partition_info(&mut self, partition_id: u32, element_count: u32, ghost_count: u32) {
        if let Some(m) = self
            .metrics
            .iter_mut()
            .find(|m| m.partition_id == partition_id)
        {
            m.element_count = element_count;
            m.ghost_count = ghost_count;
        }
    }

    /// Record a timestep solve time for a partition.
    pub fn record(&mut self, partition_id: u32, solve_time: f64) {
        if let Some(m) = self
            .metrics
            .iter_mut()
            .find(|m| m.partition_id == partition_id)
        {
            m.record_solve_time(solve_time);
        }
        self.step_counter += 1;
    }

    /// Check if rebalancing is needed (called periodically).
    /// Returns migration plans if any.
    pub fn check_balance(&mut self) -> Vec<MigrationPlan> {
        if self.step_counter % self.check_interval != 0 {
            return vec![];
        }

        // Compute balance score
        let score = self.balance_score();
        self.balance_history.push(score);

        // Detect stragglers
        let stragglers = self.detector.detect(&self.metrics);

        if stragglers.is_empty() {
            return vec![];
        }

        info!(
            "Load imbalance detected (score={:.2}): {} stragglers",
            score,
            stragglers.len()
        );

        // Generate plans
        self.stealer.plan_migrations(&self.metrics, &stragglers)
    }

    /// Compute a balance score [0.0 = perfectly balanced, 1.0 = maximally imbalanced].
    pub fn balance_score(&self) -> f64 {
        let times: Vec<f64> = self.metrics.iter().map(|m| m.avg_solve_time()).collect();
        if times.is_empty() {
            return 0.0;
        }

        let max = times.iter().cloned().fold(f64::MIN, f64::max);
        let min = times.iter().cloned().fold(f64::MAX, f64::min);

        if max <= 0.0 {
            return 0.0;
        }

        // Normalized imbalance: 0 = perfect, 1 = max
        (max - min) / max
    }

    /// Get a performance summary.
    pub fn summary(&self) -> String {
        let mut lines = vec![format!(
            "Load Balance Monitor: {} partitions, {} steps",
            self.metrics.len(),
            self.step_counter
        )];

        for m in &self.metrics {
            lines.push(format!(
                "  P{}: {:.2e}s/step, {} elements, {} ghosts, {} rollbacks",
                m.partition_id,
                m.avg_solve_time(),
                m.element_count,
                m.ghost_count,
                m.rollback_count
            ));
        }

        lines.push(format!(
            "  Balance score: {:.3} (0=perfect)",
            self.balance_score()
        ));
        lines.push(format!(
            "  Migrations planned: {}",
            self.stealer.stats.migrations_planned
        ));

        lines.join("\n")
    }
}

// ─── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_metrics_ewma() {
        let mut m = PartitionMetrics::new(0, 1000, 10);

        // Record increasing solve times
        for i in 0..20 {
            m.record_solve_time(0.001 * (i as f64 + 1.0));
        }

        assert!(m.avg_solve_time() > 0.0);
        assert_eq!(m.steps_completed, 20);
        assert!(m.throughput() > 0.0);
    }

    #[test]
    fn test_straggler_detection() {
        let mut metrics = vec![
            PartitionMetrics::new(0, 1000, 5),
            PartitionMetrics::new(1, 1000, 5),
            PartitionMetrics::new(2, 1000, 5),
            PartitionMetrics::new(3, 1000, 5),
        ];

        // P0-P2 are fast, P3 is a straggler (3x slower)
        for _ in 0..20 {
            metrics[0].record_solve_time(0.001);
            metrics[1].record_solve_time(0.0012);
            metrics[2].record_solve_time(0.0011);
            metrics[3].record_solve_time(0.003); // 3x slower
        }

        let detector = StragglerDetector::with_defaults();
        let stragglers = detector.detect(&metrics);

        assert!(
            !stragglers.is_empty(),
            "P3 should be detected as a straggler"
        );
        assert_eq!(stragglers[0].0, 3, "Straggler should be P3");
        assert!(
            stragglers[0].1 > 2.0,
            "Slowdown ratio should be > 2x, got {:.2}",
            stragglers[0].1
        );
    }

    #[test]
    fn test_no_straggler_when_balanced() {
        let mut metrics = vec![
            PartitionMetrics::new(0, 1000, 5),
            PartitionMetrics::new(1, 1000, 5),
            PartitionMetrics::new(2, 1000, 5),
        ];

        // All roughly equal
        for _ in 0..20 {
            metrics[0].record_solve_time(0.001);
            metrics[1].record_solve_time(0.00105);
            metrics[2].record_solve_time(0.00098);
        }

        let detector = StragglerDetector::with_defaults();
        let stragglers = detector.detect(&metrics);

        assert!(stragglers.is_empty(), "No stragglers when balanced");
    }

    #[test]
    fn test_work_stealer_generates_plans() {
        // Need at least one partition with throughput > mean * 1.1
        // P0: 1000 elements, 0.001s → throughput = 1_000_000
        // P1: 1000 elements, 0.001s → throughput = 1_000_000
        // P2: 5000 elements, 0.010s → throughput =   500_000
        // Mean throughput ≈ 833_333. P0, P1 > 833_333 * 1.1 = 916_666 ✓
        let mut metrics = vec![
            PartitionMetrics::new(0, 1000, 5),
            PartitionMetrics::new(1, 1000, 5),
            PartitionMetrics::new(2, 5000, 20), // Heavy straggler
        ];

        for _ in 0..20 {
            metrics[0].record_solve_time(0.001);
            metrics[1].record_solve_time(0.001);
            metrics[2].record_solve_time(0.010); // 10x slower
        }

        let mut stealer = WorkStealer::new();
        let stragglers = vec![(2, 10.0)];
        let plans = stealer.plan_migrations(&metrics, &stragglers);

        assert!(
            !plans.is_empty(),
            "Should generate at least one migration plan"
        );
        assert_eq!(plans[0].from_partition, 2, "Should migrate FROM straggler");
        assert!(
            plans[0].to_partition <= 1,
            "Should migrate TO a fast partition"
        );
        assert!(plans[0].elements_to_move > 0, "Should move some elements");
    }

    #[test]
    fn test_balance_monitor_full_flow() {
        let mut monitor = BalanceMonitor::new(3);
        monitor.set_partition_info(0, 500, 3);
        monitor.set_partition_info(1, 500, 3);
        monitor.set_partition_info(2, 2000, 10);

        // Simulate 100 steps
        for _ in 0..100 {
            monitor.record(0, 0.001);
            monitor.record(1, 0.0011);
            monitor.record(2, 0.004); // P2 is heavy
        }

        let score = monitor.balance_score();
        assert!(score > 0.0, "Balance score should be >0 with unequal loads");

        // Check should produce plans at the right interval
        // step_counter is 300 at this point (100 * 3 partitions)
        let plans = monitor.check_balance();
        // May or may not have plans depending on exact timing

        let summary = monitor.summary();
        assert!(summary.contains("P0"));
        assert!(summary.contains("P2"));
    }

    #[test]
    fn test_balance_score() {
        let mut monitor = BalanceMonitor::new(2);

        // Perfect balance
        for _ in 0..20 {
            monitor.record(0, 0.001);
            monitor.record(1, 0.001);
        }
        let score = monitor.balance_score();
        assert!(
            score < 0.1,
            "Perfect balance should have score near 0, got {score:.3}"
        );
    }

    #[test]
    fn test_partition_metrics_variance() {
        let mut m = PartitionMetrics::new(0, 100, 2);

        // Consistent timing
        for _ in 0..10 {
            m.record_solve_time(0.001);
        }
        let low_var = m.solve_time_variance();

        // Reset with variable timing
        let mut m2 = PartitionMetrics::new(1, 100, 2);
        for i in 0..10 {
            m2.record_solve_time(0.001 * (1.0 + (i as f64 * 0.5)));
        }
        let high_var = m2.solve_time_variance();

        assert!(
            high_var > low_var,
            "Variable timing should have higher variance"
        );
    }
}
