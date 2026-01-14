//! Sculptor: Offline Correlation Analysis for Topology Discovery
//!
//! The Sculptor runs during calibration to discover which layers are
//! highly correlated (Passengers) and can be synthesized from others (Drivers).

use crate::{LayerId, NangilaError, Result, Tensor, TopologyMask};
use std::collections::HashMap;

/// Statistics accumulated for a single layer during calibration
#[derive(Debug, Clone)]
struct LayerStats {
    /// Sum of gradients (for mean calculation)
    sum: Vec<f32>,
    /// Sum of squared gradients (for variance)
    sum_sq: Vec<f32>,
    /// Number of samples
    count: usize,
}

impl LayerStats {
    fn new(size: usize) -> Self {
        Self {
            sum: vec![0.0; size],
            sum_sq: vec![0.0; size],
            count: 0,
        }
    }

    fn update(&mut self, gradient: &Tensor) {
        for (i, &val) in gradient.data.iter().enumerate() {
            self.sum[i] += val;
            self.sum_sq[i] += val * val;
        }
        self.count += 1;
    }

    fn mean(&self) -> Vec<f32> {
        let n = self.count as f32;
        self.sum.iter().map(|&s| s / n).collect()
    }

    fn variance(&self) -> Vec<f32> {
        let n = self.count as f32;
        self.sum
            .iter()
            .zip(&self.sum_sq)
            .map(|(&s, &sq)| (sq / n) - (s / n).powi(2))
            .collect()
    }

    fn std_dev(&self) -> f32 {
        self.variance()
            .iter()
            .map(|v| v.max(0.0))
            .sum::<f32>()
            .sqrt()
    }
}

/// Sculptor for discovering layer topology
#[derive(Debug)]
pub struct Sculptor {
    /// Correlation threshold for Passenger detection
    threshold: f32,
    /// Per-layer statistics
    stats: HashMap<LayerId, LayerStats>,
    /// Raw gradient history for correlation computation
    history: HashMap<LayerId, Vec<Vec<f32>>>,
    /// Maximum history length (for memory efficiency)
    max_history: usize,
}

impl Sculptor {
    /// Create a new Sculptor with given correlation threshold
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold,
            stats: HashMap::new(),
            history: HashMap::new(),
            max_history: 1000, // Store at most 1000 gradient samples per layer
        }
    }

    /// Create a Sculptor with default threshold (0.95)
    pub fn default_threshold() -> Self {
        Self::new(0.95)
    }

    /// Record a gradient sample for a layer
    pub fn record(&mut self, layer_id: LayerId, gradient: &Tensor) {
        // Update running statistics
        let size = gradient.numel();
        self.stats
            .entry(layer_id)
            .or_insert_with(|| LayerStats::new(size))
            .update(gradient);

        // Store flattened gradient for correlation computation
        let history = self.history.entry(layer_id).or_insert_with(Vec::new);
        if history.len() < self.max_history {
            history.push(gradient.data.clone());
        }
    }

    /// Compute Pearson correlation between two layers
    fn compute_correlation(&self, layer_i: LayerId, layer_j: LayerId) -> Option<f32> {
        let hist_i = self.history.get(&layer_i)?;
        let hist_j = self.history.get(&layer_j)?;

        if hist_i.len() < 2 || hist_j.len() < 2 || hist_i.len() != hist_j.len() {
            return None;
        }

        let n = hist_i.len();

        // Flatten and compute per-step "summary" (mean gradient magnitude)
        // This gives us a single value per step to correlate
        let summary_i: Vec<f32> = hist_i
            .iter()
            .map(|g| g.iter().sum::<f32>() / g.len() as f32)
            .collect();
        let summary_j: Vec<f32> = hist_j
            .iter()
            .map(|g| g.iter().sum::<f32>() / g.len() as f32)
            .collect();

        // Pearson correlation
        let mean_i: f32 = summary_i.iter().sum::<f32>() / n as f32;
        let mean_j: f32 = summary_j.iter().sum::<f32>() / n as f32;

        let mut cov = 0.0f32;
        let mut var_i = 0.0f32;
        let mut var_j = 0.0f32;

        for t in 0..n {
            let di = summary_i[t] - mean_i;
            let dj = summary_j[t] - mean_j;
            cov += di * dj;
            var_i += di * di;
            var_j += dj * dj;
        }

        let denom = (var_i * var_j).sqrt();
        if denom < 1e-8 {
            return None;
        }

        Some(cov / denom)
    }

    /// Compute linear regression coupling factor: L_j ≈ α * L_i + β
    fn compute_coupling(&self, driver_id: LayerId, passenger_id: LayerId) -> Option<(f32, f32)> {
        let hist_d = self.history.get(&driver_id)?;
        let hist_p = self.history.get(&passenger_id)?;

        if hist_d.is_empty() || hist_p.is_empty() || hist_d.len() != hist_p.len() {
            return None;
        }

        // Use summary values for regression
        let summary_d: Vec<f32> = hist_d
            .iter()
            .map(|g| g.iter().sum::<f32>() / g.len() as f32)
            .collect();
        let summary_p: Vec<f32> = hist_p
            .iter()
            .map(|g| g.iter().sum::<f32>() / g.len() as f32)
            .collect();

        let n = summary_d.len() as f32;
        let mean_d: f32 = summary_d.iter().sum::<f32>() / n;
        let mean_p: f32 = summary_p.iter().sum::<f32>() / n;

        // α = Σ(d - mean_d)(p - mean_p) / Σ(d - mean_d)²
        let mut num = 0.0f32;
        let mut denom = 0.0f32;

        for (&d, &p) in summary_d.iter().zip(&summary_p) {
            let dd = d - mean_d;
            let dp = p - mean_p;
            num += dd * dp;
            denom += dd * dd;
        }

        if denom.abs() < 1e-8 {
            return None;
        }

        let alpha = num / denom;
        let beta = mean_p - alpha * mean_d;

        Some((alpha, beta))
    }

    /// Generate the topology mask from recorded gradients
    pub fn generate_mask(&self) -> Result<TopologyMask> {
        let layer_ids: Vec<LayerId> = self.history.keys().copied().collect();

        if layer_ids.len() < 2 {
            return Err(NangilaError::InsufficientSamples);
        }

        let mut mask = TopologyMask::new();
        let mut assigned: HashMap<LayerId, bool> = HashMap::new();

        // Sort layers by their gradient variance (higher variance = more likely to be Driver)
        let mut layer_variances: Vec<(LayerId, f32)> = layer_ids
            .iter()
            .filter_map(|&id| self.stats.get(&id).map(|s| (id, s.std_dev())))
            .collect();
        layer_variances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Greedy assignment: high-variance layers become Drivers first
        for (layer_id, _variance) in &layer_variances {
            if assigned.contains_key(layer_id) {
                continue;
            }

            // Check if this layer is highly correlated with any existing Driver
            let mut best_driver: Option<(LayerId, f32, f32, f32)> = None; // (driver_id, correlation, alpha, beta)

            for (&existing_id, &is_driver) in &assigned {
                if !is_driver {
                    continue;
                }

                if let Some(corr) = self.compute_correlation(*layer_id, existing_id) {
                    if corr.abs() > self.threshold {
                        if let Some((alpha, beta)) = self.compute_coupling(existing_id, *layer_id) {
                            if best_driver
                                .as_ref()
                                .map(|(_, c, _, _)| corr.abs() > c.abs())
                                .unwrap_or(true)
                            {
                                best_driver = Some((existing_id, corr, alpha, beta));
                            }
                        }
                    }
                }
            }

            if let Some((driver_id, _corr, alpha, beta)) = best_driver {
                // This layer is a Passenger
                mask.add_passenger(*layer_id, driver_id, alpha, beta);
                assigned.insert(*layer_id, false);
            } else {
                // This layer is a Driver
                mask.add_driver(*layer_id);
                assigned.insert(*layer_id, true);
            }
        }

        tracing::info!(
            "Sculptor generated mask: {} drivers, {} passengers, {:.1}x compression",
            mask.num_drivers(),
            mask.num_passengers(),
            mask.compression_ratio()
        );

        Ok(mask)
    }

    /// Get the number of recorded samples
    pub fn num_samples(&self) -> usize {
        self.stats.values().map(|s| s.count).min().unwrap_or(0)
    }

    /// Get the number of layers being tracked
    pub fn num_layers(&self) -> usize {
        self.stats.len()
    }

    /// Get the correlation threshold
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Compute full correlation matrix
    pub fn correlation_matrix(&self) -> Option<Vec<Vec<f32>>> {
        let layer_ids: Vec<LayerId> = {
            let mut ids: Vec<_> = self.history.keys().copied().collect();
            ids.sort();
            ids
        };

        if layer_ids.len() < 2 {
            return None;
        }

        let n = layer_ids.len();
        let mut matrix = vec![vec![0.0f32; n]; n];

        for (i, &id_i) in layer_ids.iter().enumerate() {
            for (j, &id_j) in layer_ids.iter().enumerate() {
                if i == j {
                    matrix[i][j] = 1.0;
                } else if i < j {
                    let corr = self.compute_correlation(id_i, id_j).unwrap_or(0.0);
                    matrix[i][j] = corr;
                    matrix[j][i] = corr;
                }
            }
        }

        Some(matrix)
    }

    /// Clear all recorded data
    pub fn reset(&mut self) {
        self.stats.clear();
        self.history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tensor(vals: &[f32]) -> Tensor {
        Tensor::new(vals.to_vec(), vec![vals.len()])
    }

    #[test]
    fn test_correlation_detection() {
        let mut sculptor = Sculptor::new(0.9);

        // Layer 0: Driver
        // Layer 1: Highly correlated with Layer 0 (α ≈ 0.5)
        for step in 0..100 {
            let base = step as f32 * 0.1;
            sculptor.record(0, &make_tensor(&[base, base * 2.0, base * 3.0]));
            sculptor.record(
                1,
                &make_tensor(&[base * 0.5, base * 1.0, base * 1.5]), // ~0.5x Layer 0
            );
        }

        let corr = sculptor.compute_correlation(0, 1).unwrap();
        assert!(corr > 0.99, "Expected high correlation, got {}", corr);
    }

    #[test]
    fn test_mask_generation() {
        let mut sculptor = Sculptor::new(0.95);

        // Create correlated layers
        for step in 0..50 {
            let base = (step as f32).sin();
            sculptor.record(0, &make_tensor(&[base, base * 2.0]));
            sculptor.record(1, &make_tensor(&[base * 0.8, base * 1.6])); // Correlated
            sculptor.record(2, &make_tensor(&[(step as f32).cos(), step as f32]));
            // Uncorrelated
        }

        let mask = sculptor.generate_mask().unwrap();
        assert!(mask.num_layers() == 3);
        // At least one driver should exist
        assert!(mask.num_drivers() >= 1);
    }
}
