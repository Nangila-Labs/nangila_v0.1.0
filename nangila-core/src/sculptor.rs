//! Sculptor: Offline Correlation Analysis for Topology Discovery
//!
//! The Sculptor runs during calibration to discover which layers are
//! highly correlated (Passengers) and can be synthesized from others (Drivers).
//!
//! Uses Welford's online algorithm for streaming statistics to avoid
//! storing raw gradient history (critical for large models).

use crate::{LayerId, NangilaError, Result, Tensor, TopologyMask};
use std::collections::HashMap;

/// Sampling strategy for large models to avoid O(n²) correlation computation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingStrategy {
    /// Compute all pairwise correlations (default for <100 layers)
    Full,
    /// Sample k nearest neighbors by gradient magnitude similarity
    KNearestNeighbors { k: usize },
    /// Random sampling of pairs
    RandomSampling { pairs_per_layer: usize },
    /// Hierarchical clustering approach
    Hierarchical { max_cluster_size: usize },
}

impl Default for SamplingStrategy {
    fn default() -> Self {
        SamplingStrategy::Full
    }
}

/// Streaming statistics using Welford's online algorithm
/// Computes running mean and variance of gradient magnitudes
#[derive(Debug, Clone)]
struct StreamingStats {
    /// Number of samples
    count: usize,
    /// Running mean of L2 norm
    mean: f64,
    /// M2 accumulator for variance (Welford's algorithm)
    m2: f64,
}

impl StreamingStats {
    fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    /// Update with new L2 norm value using Welford's algorithm
    fn update(&mut self, l2_norm: f64) {
        self.count += 1;
        let delta = l2_norm - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = l2_norm - self.mean;
        self.m2 += delta * delta2;
    }

    fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / (self.count - 1) as f64
        }
    }

    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

/// Streaming correlation between two layers using online covariance
/// Computes correlation on actual gradient vectors, not just RMS norms
#[derive(Debug, Clone)]
struct StreamingCorrelation {
    /// Number of paired samples
    count: usize,
    /// Running mean vector for layer i (element-wise)
    mean_i: Vec<f64>,
    /// Running mean vector for layer j (element-wise)
    mean_j: Vec<f64>,
    /// Co-moment accumulator: Σ(xi - mean_i)(xj - mean_j) element-wise
    co_moment: Vec<f64>,
    /// M2 accumulator for layer i (element-wise variance)
    m2_i: Vec<f64>,
    /// M2 accumulator for layer j (element-wise variance)
    m2_j: Vec<f64>,
    /// Size of gradient vectors (for validation)
    size: usize,
}

impl StreamingCorrelation {
    fn new(size: usize) -> Self {
        Self {
            count: 0,
            mean_i: vec![0.0; size],
            mean_j: vec![0.0; size],
            co_moment: vec![0.0; size],
            m2_i: vec![0.0; size],
            m2_j: vec![0.0; size],
            size,
        }
    }

    /// Update with paired gradient vectors (element-wise correlation)
    fn update(&mut self, grad_i: &[f32], grad_j: &[f32]) {
        // Validate sizes match
        if grad_i.len() != self.size || grad_j.len() != self.size {
            tracing::warn!(
                "Gradient size mismatch in correlation: expected {}, got {} and {}",
                self.size,
                grad_i.len(),
                grad_j.len()
            );
            return;
        }

        self.count += 1;
        let n = self.count as f64;
        
        // Welford's online covariance algorithm (element-wise)
        for idx in 0..self.size {
            let val_i = grad_i[idx] as f64;
            let val_j = grad_j[idx] as f64;
            
            let delta_i = val_i - self.mean_i[idx];
            let delta_j = val_j - self.mean_j[idx];
            
            self.mean_i[idx] += delta_i / n;
            self.mean_j[idx] += delta_j / n;
            
            let delta2_i = val_i - self.mean_i[idx];
            let delta2_j = val_j - self.mean_j[idx];
            
            self.m2_i[idx] += delta_i * delta2_i;
            self.m2_j[idx] += delta_j * delta2_j;
            self.co_moment[idx] += delta_i * delta2_j;
        }
    }

    /// Compute Pearson correlation coefficient (averaged across elements)
    fn correlation(&self) -> Option<f32> {
        if self.count < 2 {
            return None;
        }
        
        let mut sum_corr = 0.0;
        let mut valid_count = 0;
        
        for idx in 0..self.size {
            let var_i = self.m2_i[idx] / (self.count - 1) as f64;
            let var_j = self.m2_j[idx] / (self.count - 1) as f64;
            let cov = self.co_moment[idx] / (self.count - 1) as f64;
            
            let denom = (var_i * var_j).sqrt();
            if denom > 1e-10 {
                sum_corr += cov / denom;
                valid_count += 1;
            }
        }
        
        if valid_count == 0 {
            return None;
        }
        
        Some((sum_corr / valid_count as f64) as f32)
    }
}

/// Sculptor for discovering layer topology (streaming, memory-efficient)
#[derive(Debug)]
pub struct Sculptor {
    /// Correlation threshold for Passenger detection
    threshold: f32,
    /// Per-layer streaming statistics
    stats: HashMap<LayerId, StreamingStats>,
    /// Pairwise streaming correlation (layer_i, layer_j) -> correlation
    correlations: HashMap<(LayerId, LayerId), StreamingCorrelation>,
    /// Current step's gradients (buffered for pairwise update)
    current_gradients: HashMap<LayerId, Vec<f32>>,
    /// Layer sizes (for validation)
    layer_sizes: HashMap<LayerId, usize>,
    /// Number of times end_step() was called (for validation)
    step_count: usize,
    /// Maximum number of correlation pairs to compute (for large models)
    max_correlation_pairs: Option<usize>,
    /// Sampling strategy for large models
    sampling_strategy: SamplingStrategy,
}

impl Sculptor {
    /// Create a new Sculptor with given correlation threshold
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold,
            stats: HashMap::new(),
            correlations: HashMap::new(),
            current_gradients: HashMap::new(),
            layer_sizes: HashMap::new(),
            step_count: 0,
            max_correlation_pairs: None,
            sampling_strategy: SamplingStrategy::default(),
        }
    }
    
    /// Create a Sculptor optimized for large models (>100 layers)
    pub fn new_large_model(threshold: f32, num_layers: usize) -> Self {
        let strategy = if num_layers < 100 {
            SamplingStrategy::Full
        } else if num_layers < 500 {
            SamplingStrategy::KNearestNeighbors { k: 20 }
        } else {
            SamplingStrategy::KNearestNeighbors { k: 10 }
        };
        
        Self {
            threshold,
            stats: HashMap::new(),
            correlations: HashMap::new(),
            current_gradients: HashMap::new(),
            layer_sizes: HashMap::new(),
            step_count: 0,
            max_correlation_pairs: Some(num_layers * 20), // O(n) instead of O(n²)
            sampling_strategy: strategy,
        }
    }
    
    /// Set the sampling strategy explicitly
    pub fn with_sampling_strategy(mut self, strategy: SamplingStrategy) -> Self {
        self.sampling_strategy = strategy;
        self
    }

    /// Create a Sculptor with default threshold (0.95)
    pub fn default_threshold() -> Self {
        Self::new(0.95)
    }

    /// Record a gradient sample for a layer
    pub fn record(&mut self, layer_id: LayerId, gradient: &Tensor) {
        let size = gradient.numel();
        self.layer_sizes.insert(layer_id, size);
        
        // Compute L2 norm for variance tracking
        let sum_sq: f64 = gradient.data.iter().map(|&x| (x as f64) * (x as f64)).sum();
        let l2_norm = sum_sq.sqrt();
        
        // Update per-layer streaming stats
        self.stats
            .entry(layer_id)
            .or_insert_with(StreamingStats::new)
            .update(l2_norm);
        
        // Buffer gradient for pairwise correlation update
        self.current_gradients.insert(layer_id, gradient.data.clone());
    }

    /// Finalize a calibration step (call after recording all layers for a step)
    /// This updates pairwise correlations with buffered gradients
    pub fn end_step(&mut self) {
        let mut layer_ids: Vec<LayerId> = self.current_gradients.keys().copied().collect();
        // Sort for deterministic ordering
        layer_ids.sort();
        
        match self.sampling_strategy {
            SamplingStrategy::Full => {
                self.end_step_full(&layer_ids);
            }
            SamplingStrategy::KNearestNeighbors { k } => {
                self.end_step_knn(&layer_ids, k);
            }
            SamplingStrategy::RandomSampling { pairs_per_layer } => {
                self.end_step_random(&layer_ids, pairs_per_layer);
            }
            SamplingStrategy::Hierarchical { max_cluster_size } => {
                self.end_step_hierarchical(&layer_ids, max_cluster_size);
            }
        }
        
        // Clear buffer for next step
        self.current_gradients.clear();
        self.step_count += 1;
    }
    
    /// Full O(n²) correlation computation (for small models)
    fn end_step_full(&mut self, layer_ids: &[LayerId]) {
        for (i, &id_i) in layer_ids.iter().enumerate() {
            for &id_j in layer_ids.iter().skip(i + 1) {
                self.update_correlation_pair(id_i, id_j);
            }
        }
    }
    
    /// K-nearest neighbors by gradient magnitude (O(n log n))
    fn end_step_knn(&mut self, layer_ids: &[LayerId], k: usize) {
        // Compute L2 norms for all layers
        let mut layer_norms: Vec<(LayerId, f64)> = layer_ids
            .iter()
            .filter_map(|&id| {
                let grad = self.current_gradients.get(&id)?;
                let norm: f64 = grad.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
                Some((id, norm))
            })
            .collect();
        
        // Sort by norm
        layer_norms.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // For each layer, correlate with k nearest neighbors by norm
        for (idx, &(id_i, _)) in layer_norms.iter().enumerate() {
            // Look at k neighbors before and after
            let start = idx.saturating_sub(k / 2);
            let end = (idx + k / 2 + 1).min(layer_norms.len());
            
            for &(id_j, _) in &layer_norms[start..end] {
                if id_i != id_j {
                    self.update_correlation_pair(id_i, id_j);
                }
            }
        }
    }
    
    /// Random sampling of pairs (O(n))
    fn end_step_random(&mut self, layer_ids: &[LayerId], pairs_per_layer: usize) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        for &id_i in layer_ids {
            // Use deterministic "random" sampling based on step and layer id
            let mut hasher = DefaultHasher::new();
            id_i.hash(&mut hasher);
            self.step_count.hash(&mut hasher);
            let seed = hasher.finish();
            
            // Select pairs_per_layer random partners
            for j in 0..pairs_per_layer {
                let partner_idx = ((seed.wrapping_add(j as u64)) as usize) % layer_ids.len();
                let id_j = layer_ids[partner_idx];
                if id_i != id_j {
                    self.update_correlation_pair(id_i, id_j);
                }
            }
        }
    }
    
    /// Hierarchical clustering (O(n log n))
    fn end_step_hierarchical(&mut self, layer_ids: &[LayerId], max_cluster_size: usize) {
        // Simple hierarchical approach: group by size, then correlate within groups
        let mut size_groups: HashMap<usize, Vec<LayerId>> = HashMap::new();
        
        for &id in layer_ids {
            if let Some(size) = self.layer_sizes.get(&id) {
                size_groups.entry(*size).or_insert_with(Vec::new).push(id);
            }
        }
        
        // Within each size group, do full correlation if small, or sample if large
        for group in size_groups.values() {
            if group.len() <= max_cluster_size {
                // Small group: full correlation
                for (i, &id_i) in group.iter().enumerate() {
                    for &id_j in group.iter().skip(i + 1) {
                        self.update_correlation_pair(id_i, id_j);
                    }
                }
            } else {
                // Large group: sample k neighbors
                let k = max_cluster_size;
                for (i, &id_i) in group.iter().enumerate() {
                    let start = i.saturating_sub(k / 2);
                    let end = (i + k / 2 + 1).min(group.len());
                    for &id_j in &group[start..end] {
                        if id_i != id_j {
                            self.update_correlation_pair(id_i, id_j);
                        }
                    }
                }
            }
        }
    }
    
    /// Helper to update a single correlation pair
    fn update_correlation_pair(&mut self, id_i: LayerId, id_j: LayerId) {
        let grad_i = match self.current_gradients.get(&id_i) {
            Some(g) => g,
            None => return,
        };
        let grad_j = match self.current_gradients.get(&id_j) {
            Some(g) => g,
            None => return,
        };
        
        // Only correlate layers with same size
        if grad_i.len() != grad_j.len() {
            return;
        }
        
        let size = grad_i.len();
        // Ensure key is always (min, max)
        let key = if id_i < id_j { (id_i, id_j) } else { (id_j, id_i) };
        
        self.correlations
            .entry(key)
            .or_insert_with(|| StreamingCorrelation::new(size))
            .update(grad_i, grad_j);
    }

    /// Compute Pearson correlation between two layers (from streaming stats)
    fn compute_correlation(&self, layer_i: LayerId, layer_j: LayerId) -> Option<f32> {
        // Streaming correlations are stored with (min_id, max_id) key
        let key = if layer_i < layer_j {
            (layer_i, layer_j)
        } else {
            (layer_j, layer_i)
        };
        
        self.correlations.get(&key)?.correlation()
    }

    /// Compute linear regression: j ≈ α * i + β (using RMS norms from vectors)
    fn coupling(&self, driver_id: LayerId, passenger_id: LayerId) -> Option<(f32, f32)> {
        let key = if driver_id < passenger_id {
            (driver_id, passenger_id)
        } else {
            (passenger_id, driver_id)
        };
        
        let corr = self.correlations.get(&key)?;
        
        if corr.count < 2 {
            return None;
        }
        
        // Compute element-wise regression, then average
        let mut sum_alpha = 0.0;
        let mut sum_beta = 0.0;
        let mut valid_count = 0;
        
        // Determine which is i (driver) and which is j (passenger)
        let (mean_driver, mean_passenger, m2_driver, co_moment) = if driver_id < passenger_id {
            (&corr.mean_i, &corr.mean_j, &corr.m2_i, &corr.co_moment)
        } else {
            (&corr.mean_j, &corr.mean_i, &corr.m2_j, &corr.co_moment)
        };
        
        for idx in 0..corr.size {
            let var_driver = m2_driver[idx] / (corr.count - 1) as f64;
            let cov = co_moment[idx] / (corr.count - 1) as f64;
            
            if var_driver > 1e-10 {
                let alpha = cov / var_driver;
                let beta = mean_passenger[idx] - alpha * mean_driver[idx];
                sum_alpha += alpha;
                sum_beta += beta;
                valid_count += 1;
            }
        }
        
        if valid_count == 0 {
            return None;
        }
        
        Some((
            (sum_alpha / valid_count as f64) as f32,
            (sum_beta / valid_count as f64) as f32,
        ))
    }


    /// Generate the topology mask from recorded gradients
    pub fn generate_mask(&self) -> Result<TopologyMask> {
        // Fix #3: Warn if end_step() was never called
        if self.step_count == 0 {
            tracing::warn!(
                "generate_mask() called but end_step() was never called. \
                 Correlations are NOT computed. Call end_step() after each calibration step."
            );
        }
        
        let layer_ids: Vec<LayerId> = self.stats.keys().copied().collect();

        if layer_ids.len() < 2 {
            return Err(NangilaError::InsufficientSamples);
        }

        let mut mask = TopologyMask::new();
        let mut assigned: HashMap<LayerId, bool> = HashMap::new();

        // Sort layers by their gradient variance (higher variance = more likely to be Driver)
        let mut layer_variances: Vec<(LayerId, f64)> = layer_ids
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
                        if let Some((alpha, beta)) = self.coupling(existing_id, *layer_id) {
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
            let mut ids: Vec<_> = self.stats.keys().copied().collect();
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
        self.correlations.clear();
        self.current_gradients.clear();
        self.layer_sizes.clear();
        self.step_count = 0;
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
            sculptor.end_step(); // Required for streaming correlation
        }

        let corr = sculptor.compute_correlation(0, 1).unwrap();
        // With actual gradient correlation, we expect very high correlation
        assert!(corr > 0.95, "Expected high correlation, got {}", corr);
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
            sculptor.end_step(); // Required for streaming correlation
        }

        let mask = sculptor.generate_mask().unwrap();
        assert!(mask.num_layers() == 3);
        // At least one driver should exist
        assert!(mask.num_drivers() >= 1);
    }
}
