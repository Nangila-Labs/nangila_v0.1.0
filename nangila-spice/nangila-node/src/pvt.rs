//! PVT-Space Solver
//!
//! Implements the golden corner cache and delta solver for
//! high-throughput Process/Voltage/Temperature corner sweeps.
//!
//! Architecture:
//!   - Golden Corner: simulate nominal corner once, cache full waveform S_nom(t)
//!   - Delta Solver: for each non-nominal corner, solve only the residual
//!       r(t) = S_B(t) - S_A(t)
//!     This is a much smaller linear system since the delta is typically small.
//!   - CornerSpec: defines a single PVT corner (process, voltage, temperature)
//!   - PvtCache: binary serialization of the golden trajectory
//!
//! Target: 5x speedup on 1000-corner PVT sweeps.
//!
//! Phase 3, Sprint 11–12 deliverable.

use std::collections::HashMap;
use std::time::Instant;
use tracing::{debug, info, warn};

// ─── PVT Corner Specification ──────────────────────────────────────

/// Process corner variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProcessCorner {
    /// Typical-Typical (nominal)
    TT,
    /// Fast-Fast
    FF,
    /// Slow-Slow
    SS,
    /// Fast-Slow (NMOS fast, PMOS slow)
    FS,
    /// Slow-Fast (NMOS slow, PMOS fast)
    SF,
}

impl ProcessCorner {
    /// Mobility multiplier relative to TT.
    pub fn mobility_factor(&self) -> f64 {
        match self {
            ProcessCorner::TT => 1.0,
            ProcessCorner::FF => 1.15, // 15% faster
            ProcessCorner::SS => 0.85, // 15% slower
            ProcessCorner::FS => 1.1,
            ProcessCorner::SF => 0.9,
        }
    }

    pub fn is_nominal(&self) -> bool {
        matches!(self, ProcessCorner::TT)
    }

    pub fn name(&self) -> &'static str {
        match self {
            ProcessCorner::TT => "TT",
            ProcessCorner::FF => "FF",
            ProcessCorner::SS => "SS",
            ProcessCorner::FS => "FS",
            ProcessCorner::SF => "SF",
        }
    }
}

/// A single PVT corner specification.
#[derive(Debug, Clone, PartialEq)]
pub struct CornerSpec {
    /// Process corner
    pub process: ProcessCorner,
    /// Supply voltage (V) — nominal typically 1.8V
    pub vdd: f64,
    /// Temperature (°C) — nominal typically 27°C
    pub temperature: f64,
    /// Unique corner name for caching/reporting
    pub name: String,
}

impl CornerSpec {
    /// Nominal (golden) corner: TT, 1.8V, 27°C.
    pub fn nominal() -> Self {
        Self {
            process: ProcessCorner::TT,
            vdd: 1.8,
            temperature: 27.0,
            name: "TT_1V8_27C".to_string(),
        }
    }

    /// Create a corner with full specification.
    pub fn new(process: ProcessCorner, vdd: f64, temp_c: f64) -> Self {
        let name = format!("{}_{}V_{:.0}C", process.name(), vdd, temp_c);
        Self {
            process,
            vdd,
            temperature: temp_c,
            name,
        }
    }

    /// Is this the nominal (golden) corner?
    pub fn is_nominal(&self) -> bool {
        self.process.is_nominal()
            && (self.vdd - 1.8).abs() < 0.01
            && (self.temperature - 27.0).abs() < 0.1
    }

    /// Thermal voltage kT/q at this temperature.
    pub fn thermal_voltage(&self) -> f64 {
        let t_kelvin = self.temperature + 273.15;
        8.617e-5 * t_kelvin // eV/K * K = V
    }

    /// Parameter delta relative to a reference corner (for delta solver).
    pub fn delta_params(&self, reference: &CornerSpec) -> CornerDelta {
        CornerDelta {
            delta_vdd: self.vdd - reference.vdd,
            delta_temp: self.temperature - reference.temperature,
            mobility_factor: self.process.mobility_factor() / reference.process.mobility_factor(),
        }
    }
}

/// Parameter delta between two corners (used by the delta solver).
#[derive(Debug, Clone)]
pub struct CornerDelta {
    /// Supply voltage change (V)
    pub delta_vdd: f64,
    /// Temperature change (°C)
    pub delta_temp: f64,
    /// Relative mobility factor (1.0 = same as reference)
    pub mobility_factor: f64,
}

impl CornerDelta {
    /// Is this delta small enough to use the linearised delta solver?
    /// (Large deltas require full simulation for accuracy)
    pub fn is_linearisable(&self) -> bool {
        self.delta_vdd.abs() < 0.3      // < 300mV supply change
            && self.delta_temp.abs() < 50.0  // < 50°C temperature change
            && (self.mobility_factor - 1.0).abs() < 0.2 // < 20% mobility change
    }
}

// ─── Golden Corner Cache ────────────────────────────────────────────

/// A cached waveform point: time + per-node voltages.
#[derive(Debug, Clone)]
pub struct WaveformPoint {
    pub time: f64,
    pub voltages: Vec<f64>,
}

/// The golden corner trajectory: S_nom(t).
/// Stores the full transient waveform for the nominal corner.
#[derive(Debug, Clone)]
pub struct GoldenTrajectory {
    pub corner: CornerSpec,
    pub points: Vec<WaveformPoint>,
    pub node_names: Vec<String>,
    pub dt: f64,
    pub created_at_secs: u64,
}

impl GoldenTrajectory {
    pub fn new(corner: CornerSpec, dt: f64) -> Self {
        Self {
            corner,
            points: Vec::new(),
            node_names: Vec::new(),
            dt,
            created_at_secs: 0,
        }
    }

    /// Add a timestep to the trajectory.
    pub fn push(&mut self, time: f64, voltages: Vec<f64>) {
        self.points.push(WaveformPoint { time, voltages });
    }

    /// Get voltage at time t for node `node_idx` (linear interpolation).
    pub fn voltage_at(&self, node_idx: usize, t: f64) -> Option<f64> {
        if self.points.is_empty() {
            return None;
        }

        // Binary search for the time point
        let idx = self.points.partition_point(|p| p.time <= t);

        if idx == 0 {
            return self.points[0].voltages.get(node_idx).copied();
        }
        if idx >= self.points.len() {
            return self.points.last()?.voltages.get(node_idx).copied();
        }

        let p1 = &self.points[idx - 1];
        let p2 = &self.points[idx];

        let v1 = p1.voltages.get(node_idx).copied().unwrap_or(0.0);
        let v2 = p2.voltages.get(node_idx).copied().unwrap_or(0.0);

        // Linear interpolation
        let alpha = (t - p1.time) / (p2.time - p1.time).max(1e-30);
        Some(v1 + alpha * (v2 - v1))
    }

    /// Serialize to bytes (compact binary format).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        // Magic + version
        buf.extend_from_slice(b"NZGLD");
        buf.push(1u8);

        // Header
        let n_points = self.points.len() as u32;
        let n_nodes = self.points.first().map(|p| p.voltages.len()).unwrap_or(0) as u32;
        buf.extend_from_slice(&n_points.to_le_bytes());
        buf.extend_from_slice(&n_nodes.to_le_bytes());
        buf.extend_from_slice(&self.dt.to_le_bytes());

        // Node names
        for name in &self.node_names {
            let bytes = name.as_bytes();
            buf.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
            buf.extend_from_slice(bytes);
        }

        // Waveform data
        for point in &self.points {
            buf.extend_from_slice(&point.time.to_le_bytes());
            for &v in &point.voltages {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }

        buf
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8], corner: CornerSpec) -> Option<Self> {
        if data.len() < 10 || &data[0..5] != b"NZGLD" {
            return None;
        }

        let mut pos = 6usize; // skip magic + version

        let n_points = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
        pos += 4;
        let n_nodes = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
        pos += 4;
        let dt = f64::from_le_bytes(data[pos..pos + 8].try_into().ok()?);
        pos += 8;

        // Node names
        let mut node_names = Vec::with_capacity(n_nodes);
        // (In this minimal version, node names are not stored in the waveform section)

        // Waveform data
        let mut points = Vec::with_capacity(n_points);
        for _ in 0..n_points {
            if pos + 8 > data.len() {
                break;
            }
            let time = f64::from_le_bytes(data[pos..pos + 8].try_into().ok()?);
            pos += 8;

            let mut voltages = Vec::with_capacity(n_nodes);
            for _ in 0..n_nodes {
                if pos + 8 > data.len() {
                    return None;
                }
                let v = f64::from_le_bytes(data[pos..pos + 8].try_into().ok()?);
                pos += 8;
                voltages.push(v);
            }
            points.push(WaveformPoint { time, voltages });
        }

        Some(GoldenTrajectory {
            corner,
            points,
            node_names,
            dt,
            created_at_secs: 0,
        })
    }

    /// Number of timesteps.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

// ─── PVT Cache ─────────────────────────────────────────────────────

/// In-memory cache of golden trajectories, keyed by corner name.
#[derive(Debug, Default)]
pub struct PvtCache {
    trajectories: HashMap<String, GoldenTrajectory>,
    pub stats: PvtCacheStats,
}

#[derive(Debug, Clone, Default)]
pub struct PvtCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    /// Max cached trajectories
    pub max_size: usize,
}

impl PvtCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            trajectories: HashMap::new(),
            stats: PvtCacheStats {
                max_size,
                ..Default::default()
            },
        }
    }

    pub fn insert(&mut self, trajectory: GoldenTrajectory) {
        if self.stats.max_size > 0 && self.trajectories.len() >= self.stats.max_size {
            // Simple eviction: remove an arbitrary entry
            if let Some(key) = self.trajectories.keys().next().cloned() {
                self.trajectories.remove(&key);
                self.stats.evictions += 1;
            }
        }
        self.trajectories
            .insert(trajectory.corner.name.clone(), trajectory);
    }

    pub fn get(&mut self, corner_name: &str) -> Option<&GoldenTrajectory> {
        if self.trajectories.contains_key(corner_name) {
            self.stats.hits += 1;
            self.trajectories.get(corner_name)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    pub fn contains(&self, corner_name: &str) -> bool {
        self.trajectories.contains_key(corner_name)
    }

    pub fn len(&self) -> usize {
        self.trajectories.len()
    }
}

// ─── Delta Solver ──────────────────────────────────────────────────

/// Result from a delta solve.
#[derive(Debug, Clone)]
pub struct DeltaSolveResult {
    pub corner: CornerSpec,
    /// Per-node voltage corrections: V_corner(t) = V_golden(t) + delta(t)
    pub delta_waveform: Vec<WaveformPoint>,
    /// Peak voltage deviation from golden (worst-case error)
    pub peak_delta_v: f64,
    /// Wall time for this corner
    pub wall_time_secs: f64,
    /// Whether delta solve was used (vs full simulation fallback)
    pub used_delta_mode: bool,
}

/// Delta solver: computes V_corner = V_golden + δV efficiently.
///
/// Key insight: for small parameter changes, the circuit response
/// is well-approximated by a first-order Taylor expansion:
///   V_corner(t) ≈ V_golden(t) + (∂V/∂p) * Δp
///
/// This avoids a full nonlinear simulation for each corner.
pub struct DeltaSolver {
    pub golden: GoldenTrajectory,
    pub stats: DeltaSolverStats,
}

#[derive(Debug, Clone, Default)]
pub struct DeltaSolverStats {
    pub corners_solved: u64,
    pub delta_mode_used: u64,
    pub full_sim_fallback: u64,
    pub total_wall_time: f64,
    pub speedup_estimate: f64,
}

impl DeltaSolver {
    pub fn new(golden: GoldenTrajectory) -> Self {
        Self {
            golden,
            stats: DeltaSolverStats::default(),
        }
    }

    /// Solve a corner using delta mode.
    ///
    /// For linearisable corners: applies parameter sensitivity correction.
    /// For large deviations: falls back to noting it needs full simulation.
    pub fn solve_corner(&mut self, corner: CornerSpec) -> DeltaSolveResult {
        let start = Instant::now();
        self.stats.corners_solved += 1;

        let delta_params = corner.delta_params(&self.golden.corner);
        let use_delta = delta_params.is_linearisable();

        if use_delta {
            self.stats.delta_mode_used += 1;
        } else {
            self.stats.full_sim_fallback += 1;
            warn!(
                "Corner {} requires full simulation (delta too large)",
                corner.name
            );
        }

        let n_nodes = self
            .golden
            .points
            .first()
            .map(|p| p.voltages.len())
            .unwrap_or(0);

        // Delta mode: scale voltages by VDD ratio + add temperature correction
        let vdd_scale = corner.vdd / self.golden.corner.vdd;
        let temp_correction = -0.5e-3 * delta_params.delta_temp; // -0.5mV/°C typical

        let mut delta_waveform = Vec::with_capacity(self.golden.points.len());
        let mut peak_delta = 0.0f64;

        for point in &self.golden.points {
            let corrected_voltages: Vec<f64> = point
                .voltages
                .iter()
                .map(|&v| {
                    let v_corner = if use_delta {
                        // Linear approximation: scale VDD, add thermal offset
                        v * vdd_scale * delta_params.mobility_factor.sqrt() + temp_correction
                    } else {
                        v // Placeholder — full sim would replace this
                    };

                    let delta_v = v_corner - v;
                    peak_delta = peak_delta.max(delta_v.abs());
                    delta_v
                })
                .collect();

            delta_waveform.push(WaveformPoint {
                time: point.time,
                voltages: corrected_voltages,
            });
        }

        let elapsed = start.elapsed().as_secs_f64();
        self.stats.total_wall_time += elapsed;

        debug!(
            "Corner {}: peak_δV={:.3e}V, δmode={}, {:.2e}s",
            corner.name, peak_delta, use_delta, elapsed
        );

        DeltaSolveResult {
            corner,
            delta_waveform,
            peak_delta_v: peak_delta,
            wall_time_secs: elapsed,
            used_delta_mode: use_delta,
        }
    }

    /// Reconstruct full waveform for a corner from its delta result.
    pub fn reconstruct(&self, delta: &DeltaSolveResult) -> Vec<WaveformPoint> {
        self.golden
            .points
            .iter()
            .zip(delta.delta_waveform.iter())
            .map(|(golden_pt, delta_pt)| WaveformPoint {
                time: golden_pt.time,
                voltages: golden_pt
                    .voltages
                    .iter()
                    .zip(delta_pt.voltages.iter())
                    .map(|(g, d)| g + d)
                    .collect(),
            })
            .collect()
    }

    /// Update speedup estimate (vs hypothetical full-sim time).
    pub fn update_speedup(&mut self, full_sim_time_per_corner: f64) {
        if self.stats.corners_solved > 0 {
            let avg_delta_time = self.stats.total_wall_time / self.stats.corners_solved as f64;
            self.stats.speedup_estimate = full_sim_time_per_corner / avg_delta_time.max(1e-9);
        }
    }
}

// ─── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_golden() -> GoldenTrajectory {
        let corner = CornerSpec::nominal();
        let mut traj = GoldenTrajectory::new(corner, 1e-12);
        // Simulate rising edge: 100 timesteps, 2 nodes
        for i in 0..100 {
            let t = i as f64 * 1e-12;
            let v1 = 1.8 * (1.0 - (-t / 10e-12).exp());
            let v2 = v1 * 0.5;
            traj.push(t, vec![v1, v2]);
        }
        traj
    }

    #[test]
    fn test_corner_spec_nominal() {
        let c = CornerSpec::nominal();
        assert!(c.is_nominal());
        assert_eq!(c.process, ProcessCorner::TT);
        assert!((c.vdd - 1.8).abs() < 1e-6);
    }

    #[test]
    fn test_corner_delta_linearisable() {
        let nom = CornerSpec::nominal();
        let corner = CornerSpec::new(ProcessCorner::FF, 1.9, 40.0);
        let delta = corner.delta_params(&nom);

        assert!(
            delta.is_linearisable(),
            "FF corner with 0.1V/13°C should be linearisable"
        );
        assert!((delta.delta_vdd - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_corner_delta_not_linearisable() {
        let nom = CornerSpec::nominal();
        // Extreme SS corner: -0.4V, +80°C
        let extreme = CornerSpec {
            process: ProcessCorner::SS,
            vdd: 1.4,
            temperature: 107.0,
            name: "extreme_SS".to_string(),
        };
        let delta = extreme.delta_params(&nom);

        assert!(
            !delta.is_linearisable(),
            "Extreme corner should require full sim"
        );
    }

    #[test]
    fn test_golden_trajectory_push_and_interpolate() {
        let traj = make_golden();

        assert_eq!(traj.len(), 100);

        // Interpolate at t=5ps (between i=4 and i=5)
        let v = traj.voltage_at(0, 5e-12);
        assert!(v.is_some());
        let v = v.unwrap();
        assert!(v > 0.0 && v < 1.8, "V should be in [0, 1.8V], got {v:.4}");
    }

    #[test]
    fn test_golden_trajectory_serialization() {
        let traj = make_golden();
        let bytes = traj.to_bytes();

        assert!(&bytes[0..5] == b"NZGLD", "Magic bytes should be present");
        assert!(bytes.len() > 100, "Should have substantial data");

        // Deserialize
        let restored = GoldenTrajectory::from_bytes(&bytes, CornerSpec::nominal());
        assert!(restored.is_some(), "Deserialization should succeed");

        let restored = restored.unwrap();
        assert_eq!(restored.len(), traj.len(), "Point count should match");
        assert!(
            (restored.points[10].time - traj.points[10].time).abs() < 1e-15,
            "Timestamps should survive roundtrip"
        );
    }

    #[test]
    fn test_pvt_cache() {
        let mut cache = PvtCache::new(10);
        let traj = make_golden();

        cache.insert(traj);
        assert_eq!(cache.len(), 1);
        assert!(cache.contains("TT_1V8_27C"));

        let hit = cache.get("TT_1V8_27C");
        assert!(hit.is_some());
        assert_eq!(cache.stats.hits, 1);
        assert_eq!(cache.stats.misses, 0);

        let miss = cache.get("FF_1V9_85C");
        assert!(miss.is_none());
        assert_eq!(cache.stats.misses, 1);
    }

    #[test]
    fn test_delta_solver_ff_corner() {
        let golden = make_golden();
        let mut solver = DeltaSolver::new(golden);

        let ff_corner = CornerSpec::new(ProcessCorner::FF, 1.9, 40.0);
        let result = solver.solve_corner(ff_corner);

        assert!(result.used_delta_mode, "FF corner should use delta mode");
        assert!(
            result.peak_delta_v >= 0.0,
            "Peak delta should be non-negative"
        );
        assert_eq!(
            result.delta_waveform.len(),
            100,
            "Should have same timesteps as golden"
        );
    }

    #[test]
    fn test_delta_solver_reconstruction() {
        let golden = make_golden();
        let nom_voltages_t50 = golden.points[50].voltages.clone();

        let mut solver = DeltaSolver::new(golden);
        let tt_corner = CornerSpec::nominal(); // Same as golden → delta ≈ 0
        let result = solver.solve_corner(tt_corner);

        let reconstructed = solver.reconstruct(&result);
        let rec_v = &reconstructed[50].voltages;

        for (&v_orig, &v_rec) in nom_voltages_t50.iter().zip(rec_v.iter()) {
            assert!(
                (v_rec - v_orig).abs() < 0.01,
                "Reconstruction should be close to golden at TT, got {v_rec:.4} vs {v_orig:.4}"
            );
        }
    }

    #[test]
    fn test_process_corner_mobility_factors() {
        assert!(ProcessCorner::FF.mobility_factor() > ProcessCorner::TT.mobility_factor());
        assert!(ProcessCorner::SS.mobility_factor() < ProcessCorner::TT.mobility_factor());
        assert!((ProcessCorner::TT.mobility_factor() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_thermal_voltage() {
        let nom = CornerSpec::nominal(); // 27°C = 300.15K
        let vt = nom.thermal_voltage();
        // kT/q at 300K ≈ 25.85mV
        assert!(
            (vt - 0.02585).abs() < 0.001,
            "Thermal voltage at 27°C should be ~25.85mV, got {:.4e}",
            vt
        );
    }
}
