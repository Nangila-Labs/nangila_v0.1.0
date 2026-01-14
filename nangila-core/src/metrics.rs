//! Prometheus-compatible Metrics for Nangila
//!
//! Exports compression statistics in Prometheus text format
//! and JSON for dashboard integration.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Nangila metrics collector
#[derive(Debug)]
pub struct MetricsCollector {
    /// Start time for throughput calculation
    start_time: Instant,
    /// Total steps processed
    steps_total: AtomicU64,
    /// Steps with compression enabled
    steps_compressed: AtomicU64,
    /// Bytes received (uncompressed)
    bytes_received: AtomicU64,
    /// Bytes transmitted (compressed)
    bytes_transmitted: AtomicU64,
    /// Safe Mode fallback count
    fallback_count: AtomicU64,
    /// Current compression ratio (stored as fixed-point * 1000)
    compression_ratio_x1000: AtomicU64,
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            steps_total: AtomicU64::new(0),
            steps_compressed: AtomicU64::new(0),
            bytes_received: AtomicU64::new(0),
            bytes_transmitted: AtomicU64::new(0),
            fallback_count: AtomicU64::new(0),
            compression_ratio_x1000: AtomicU64::new(1000), // Default 1.0
        }
    }

    /// Record a step
    pub fn record_step(&self, compressed: bool) {
        self.steps_total.fetch_add(1, Ordering::Relaxed);
        if compressed {
            self.steps_compressed.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record bytes transmitted
    pub fn record_bytes(&self, received: u64, transmitted: u64) {
        self.bytes_received.fetch_add(received, Ordering::Relaxed);
        self.bytes_transmitted
            .fetch_add(transmitted, Ordering::Relaxed);
    }

    /// Record a fallback event
    pub fn record_fallback(&self) {
        self.fallback_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Update compression ratio
    pub fn set_compression_ratio(&self, ratio: f32) {
        let ratio_x1000 = (ratio * 1000.0) as u64;
        self.compression_ratio_x1000
            .store(ratio_x1000, Ordering::Relaxed);
    }

    /// Get current metrics snapshot
    pub fn snapshot(&self) -> NangilaMetrics {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let bytes_received = self.bytes_received.load(Ordering::Relaxed);
        let bytes_transmitted = self.bytes_transmitted.load(Ordering::Relaxed);

        let compression_ratio = if bytes_transmitted > 0 {
            bytes_received as f64 / bytes_transmitted as f64
        } else {
            self.compression_ratio_x1000.load(Ordering::Relaxed) as f64 / 1000.0
        };

        let virtual_throughput_gbps = if elapsed > 0.0 {
            (bytes_received as f64 * 8.0) / (elapsed * 1e9)
        } else {
            0.0
        };

        let physical_throughput_gbps = if elapsed > 0.0 {
            (bytes_transmitted as f64 * 8.0) / (elapsed * 1e9)
        } else {
            0.0
        };

        NangilaMetrics {
            compression_ratio: compression_ratio as f32,
            virtual_throughput_gbps,
            physical_throughput_gbps,
            steps_total: self.steps_total.load(Ordering::Relaxed),
            steps_compressed: self.steps_compressed.load(Ordering::Relaxed),
            bytes_received,
            bytes_transmitted,
            bytes_saved: bytes_received.saturating_sub(bytes_transmitted),
            fallback_count: self.fallback_count.load(Ordering::Relaxed) as u32,
            uptime_seconds: elapsed,
        }
    }

    /// Reset all metrics
    pub fn reset(&self) {
        self.steps_total.store(0, Ordering::Relaxed);
        self.steps_compressed.store(0, Ordering::Relaxed);
        self.bytes_received.store(0, Ordering::Relaxed);
        self.bytes_transmitted.store(0, Ordering::Relaxed);
        self.fallback_count.store(0, Ordering::Relaxed);
    }
}

/// Metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NangilaMetrics {
    /// Current compression ratio (e.g., 8.0 = 8x compression)
    pub compression_ratio: f32,
    /// Effective bandwidth (as if uncompressed)
    pub virtual_throughput_gbps: f64,
    /// Actual bandwidth used
    pub physical_throughput_gbps: f64,
    /// Total training steps
    pub steps_total: u64,
    /// Steps with compression enabled
    pub steps_compressed: u64,
    /// Total bytes received (before compression)
    pub bytes_received: u64,
    /// Total bytes transmitted (after compression)
    pub bytes_transmitted: u64,
    /// Total bytes saved
    pub bytes_saved: u64,
    /// Safe Mode fallback count
    pub fallback_count: u32,
    /// Uptime in seconds
    pub uptime_seconds: f64,
}

impl NangilaMetrics {
    /// Export as Prometheus text format
    pub fn to_prometheus(&self) -> String {
        let mut output = String::with_capacity(1024);

        // Compression ratio gauge
        output.push_str("# HELP nangila_compression_ratio Current compression ratio\n");
        output.push_str("# TYPE nangila_compression_ratio gauge\n");
        output.push_str(&format!(
            "nangila_compression_ratio {:.3}\n\n",
            self.compression_ratio
        ));

        // Virtual throughput gauge
        output.push_str("# HELP nangila_virtual_throughput_gbps Effective bandwidth in Gbps\n");
        output.push_str("# TYPE nangila_virtual_throughput_gbps gauge\n");
        output.push_str(&format!(
            "nangila_virtual_throughput_gbps {:.6}\n\n",
            self.virtual_throughput_gbps
        ));

        // Physical throughput gauge
        output.push_str("# HELP nangila_physical_throughput_gbps Actual bandwidth in Gbps\n");
        output.push_str("# TYPE nangila_physical_throughput_gbps gauge\n");
        output.push_str(&format!(
            "nangila_physical_throughput_gbps {:.6}\n\n",
            self.physical_throughput_gbps
        ));

        // Steps counters
        output.push_str("# HELP nangila_steps_total Total training steps\n");
        output.push_str("# TYPE nangila_steps_total counter\n");
        output.push_str(&format!("nangila_steps_total {}\n\n", self.steps_total));

        output.push_str("# HELP nangila_steps_compressed Steps with compression enabled\n");
        output.push_str("# TYPE nangila_steps_compressed counter\n");
        output.push_str(&format!(
            "nangila_steps_compressed {}\n\n",
            self.steps_compressed
        ));

        // Bytes counters
        output.push_str("# HELP nangila_bytes_received Total bytes received (uncompressed)\n");
        output.push_str("# TYPE nangila_bytes_received counter\n");
        output.push_str(&format!(
            "nangila_bytes_received {}\n\n",
            self.bytes_received
        ));

        output.push_str("# HELP nangila_bytes_transmitted Total bytes transmitted (compressed)\n");
        output.push_str("# TYPE nangila_bytes_transmitted counter\n");
        output.push_str(&format!(
            "nangila_bytes_transmitted {}\n\n",
            self.bytes_transmitted
        ));

        output.push_str("# HELP nangila_bytes_saved Total bytes saved by compression\n");
        output.push_str("# TYPE nangila_bytes_saved counter\n");
        output.push_str(&format!("nangila_bytes_saved {}\n\n", self.bytes_saved));

        // Fallback counter
        output.push_str("# HELP nangila_fallback_total Safe Mode fallback events\n");
        output.push_str("# TYPE nangila_fallback_total counter\n");
        output.push_str(&format!(
            "nangila_fallback_total {}\n\n",
            self.fallback_count
        ));

        // Uptime gauge
        output.push_str("# HELP nangila_uptime_seconds Time since start\n");
        output.push_str("# TYPE nangila_uptime_seconds gauge\n");
        output.push_str(&format!(
            "nangila_uptime_seconds {:.3}\n",
            self.uptime_seconds
        ));

        output
    }

    /// Export as JSON
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_collector() {
        let collector = MetricsCollector::new();

        collector.record_step(true);
        collector.record_step(true);
        collector.record_step(false);
        collector.record_bytes(1000, 250);

        let metrics = collector.snapshot();
        assert_eq!(metrics.steps_total, 3);
        assert_eq!(metrics.steps_compressed, 2);
        assert_eq!(metrics.bytes_received, 1000);
        assert_eq!(metrics.bytes_transmitted, 250);
        assert!((metrics.compression_ratio - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_prometheus_format() {
        let metrics = NangilaMetrics {
            compression_ratio: 8.5,
            virtual_throughput_gbps: 10.5,
            physical_throughput_gbps: 1.25,
            steps_total: 1000,
            steps_compressed: 900,
            bytes_received: 1_000_000,
            bytes_transmitted: 125_000,
            bytes_saved: 875_000,
            fallback_count: 2,
            uptime_seconds: 3600.0,
        };

        let prom = metrics.to_prometheus();
        assert!(prom.contains("nangila_compression_ratio 8.500"));
        assert!(prom.contains("nangila_steps_total 1000"));
        assert!(prom.contains("# TYPE nangila_compression_ratio gauge"));
    }

    #[test]
    fn test_json_format() {
        let metrics = NangilaMetrics {
            compression_ratio: 4.0,
            virtual_throughput_gbps: 5.0,
            physical_throughput_gbps: 1.25,
            steps_total: 100,
            steps_compressed: 80,
            bytes_received: 10000,
            bytes_transmitted: 2500,
            bytes_saved: 7500,
            fallback_count: 0,
            uptime_seconds: 60.0,
        };

        let json = metrics.to_json();
        assert!(json.contains("\"compression_ratio\""));
        assert!(json.contains("4.0"));
    }
}
