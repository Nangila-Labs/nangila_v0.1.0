//! Solver Module
//!
//! Manages the transient simulation loop for a single partition.
//! Orchestrates: MNA stamping → solve → ghost predict → comm → checkpoint.
//!
//! Phase 1, Sprint 3 deliverable.

use tracing::{debug, info, warn};

use crate::comm::CommLayer;
use crate::ghost::GhostBuffer;
use crate::ngspice_ffi::{PartitionNetlist, SolverBackend, SolverEngine};

/// State of a single partition at a point in time.
#[derive(Debug, Clone)]
pub struct PartitionState {
    /// Node voltages (indexed by local node ID)
    pub voltages: Vec<f64>,
    /// Current simulation time
    pub time: f64,
    /// Whether this state has been validated (residuals checked)
    pub validated: bool,
}

impl PartitionState {
    pub fn new(num_nodes: usize) -> Self {
        Self {
            voltages: vec![0.0; num_nodes],
            time: 0.0,
            validated: false,
        }
    }
}

/// Configuration for the simulation loop.
#[derive(Debug, Clone)]
pub struct SimConfig {
    /// Simulation end time (seconds)
    pub tstop: f64,
    /// Time step (seconds)
    pub dt: f64,
    /// Relative tolerance for ghost node predictions
    pub reltol: f64,
    /// Max speculative steps ahead of verified time
    pub predict_depth: u32,
    /// Max checkpoint history depth
    pub max_history: usize,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            tstop: 1e-6,
            dt: 1e-12,
            reltol: 1e-3,
            predict_depth: 5,
            max_history: 100,
        }
    }
}

/// Statistics from a completed simulation run.
#[derive(Debug, Clone, Default)]
pub struct SimStats {
    pub total_steps: u64,
    pub prediction_hits: u64,
    pub prediction_misses: u64,
    pub rollbacks: u64,
    pub wall_time_secs: f64,
}

impl SimStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.prediction_hits + self.prediction_misses;
        if total == 0 {
            1.0
        } else {
            self.prediction_hits as f64 / total as f64
        }
    }
}

/// The main transient simulation loop for a single partition.
pub struct TransientSolver {
    /// Solver engine (ngspice or built-in)
    engine: SolverEngine,
    /// Ghost node buffer for boundary predictions
    ghosts: GhostBuffer,
    /// Communication layer
    comm: CommLayer,
    /// Configuration
    config: SimConfig,
    /// Checkpoint history for rollback
    history: Vec<PartitionState>,
    /// Waveform recording: (time, voltages)
    pub waveform: Vec<(f64, Vec<f64>)>,
    /// Running statistics
    pub stats: SimStats,
}

impl TransientSolver {
    pub fn new(
        netlist: PartitionNetlist,
        ghosts: GhostBuffer,
        comm: CommLayer,
        config: SimConfig,
    ) -> Self {
        Self {
            engine: SolverEngine::with_backend(netlist, SolverBackend::BuiltIn),
            ghosts,
            comm,
            config,
            history: Vec::new(),
            waveform: Vec::new(),
            stats: SimStats::default(),
        }
    }

    /// Run the full transient simulation from t=0 to t=tstop.
    /// Returns the final state and simulation statistics.
    pub fn run(&mut self) -> (PartitionState, SimStats) {
        let start = std::time::Instant::now();

        let mna_size = self.get_mna_size();
        let mut state = PartitionState::new(mna_size);
        let mut step = 0u64;

        info!(
            "Starting transient simulation: tstop={:.2e}s, dt={:.2e}s, steps={}",
            self.config.tstop,
            self.config.dt,
            (self.config.tstop / self.config.dt) as u64
        );

        while state.time < self.config.tstop {
            // 1. Check for incoming residual corrections
            let incoming = self.comm.poll_residuals();
            for msg in &incoming {
                let corrections: Vec<(u64, f64, f64)> = msg
                    .updates
                    .iter()
                    .map(|&(net_id, v)| (net_id, v, msg.time))
                    .collect();

                let all_ok = self
                    .ghosts
                    .apply_corrections(&corrections, self.config.reltol);
                if all_ok {
                    self.stats.prediction_hits += corrections.len() as u64;
                } else {
                    self.stats.prediction_misses += corrections.len() as u64;
                }
            }

            // 2. Check for rollback requests
            if let Some(rollback) = self.comm.poll_rollback() {
                warn!("Rollback requested to t={:.2e}s", rollback.rollback_to_time);
                if let Some(restored) = self.rollback_to(rollback.rollback_to_time) {
                    state = restored;
                    self.stats.rollbacks += 1;
                    continue;
                }
            }

            // 3. Get ghost node predictions for current time + dt
            let ghost_predictions = self.ghosts.predict_all(state.time + self.config.dt);

            // 4. Solve this timestep
            let new_state = self
                .engine
                .solve_step(&state, &ghost_predictions, self.config.dt);

            match new_state {
                Some(mut ns) => {
                    ns.time = state.time + self.config.dt;

                    // 5. Record waveform point
                    self.waveform.push((ns.time, ns.voltages.clone()));

                    // 6. Checkpoint for rollback
                    self.checkpoint(state.clone());

                    // 7. Send boundary values to neighbors
                    // (In single-process mode, orchestrator routes these)
                    // This is a no-op for now until orchestrator wiring

                    state = ns;
                    step += 1;

                    if step % 10000 == 0 {
                        debug!(
                            "Step {}: t={:.2e}s ({:.1}%)",
                            step,
                            state.time,
                            (state.time / self.config.tstop) * 100.0
                        );
                    }
                }
                None => {
                    warn!("Solver failed at t={:.2e}s, reducing timestep", state.time);
                    // TODO: Adaptive timestep reduction
                    break;
                }
            }
        }

        self.stats.total_steps = step;
        self.stats.wall_time_secs = start.elapsed().as_secs_f64();

        info!(
            "Simulation complete: {} steps, {:.3}s wall time, hit_rate={:.1}%, rollbacks={}",
            self.stats.total_steps,
            self.stats.wall_time_secs,
            self.stats.hit_rate() * 100.0,
            self.stats.rollbacks
        );

        (state, self.stats.clone())
    }

    /// Save state for potential rollback.
    fn checkpoint(&mut self, state: PartitionState) {
        if self.history.len() >= self.config.max_history {
            self.history.remove(0);
        }
        self.history.push(state);
    }

    /// Rollback to the state at or before target_time.
    fn rollback_to(&mut self, target_time: f64) -> Option<PartitionState> {
        while let Some(state) = self.history.last() {
            if state.time <= target_time {
                return self.history.last().cloned();
            }
            self.history.pop();
        }
        // Trim waveform
        self.waveform.retain(|(t, _)| *t <= target_time);
        None
    }

    /// Get the MNA system size (nodes + voltage source branch currents).
    fn get_mna_size(&self) -> usize {
        // This is derived from the netlist in the engine
        // For now, use a reasonable default
        3 // Will be properly set from netlist
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mna::Element;

    #[test]
    fn test_transient_rc_simulation() {
        // Simple RC circuit: V=1.8V → R=1k → C=10fF → GND
        let netlist = PartitionNetlist {
            name: "RC test".into(),
            num_nodes: 2,
            elements: vec![
                Element::VoltageSource {
                    pos: 1,
                    neg: 0,
                    v: 1.8,
                },
                Element::Resistor {
                    a: 1,
                    b: 2,
                    r: 1000.0,
                },
                Element::Capacitor {
                    a: 2,
                    b: 0,
                    c: 10e-15,
                },
            ],
            ghost_map: vec![],
        };

        let ghosts = GhostBuffer::new();
        let comm = CommLayer::new(0, 1);
        let config = SimConfig {
            tstop: 100e-12, // 100ps
            dt: 1e-12,      // 1ps steps
            reltol: 1e-3,
            predict_depth: 5,
            max_history: 50,
        };

        let mut solver = TransientSolver::new(netlist, ghosts, comm, config);
        let (final_state, stats) = solver.run();

        // Verify simulation ran
        assert!(stats.total_steps > 0, "Should have completed steps");
        assert!(!solver.waveform.is_empty(), "Should have recorded waveform");

        // After 100ps with RC=10ps (10 tau), cap should be near 1.8V
        let last_v2 = solver.waveform.last().map(|(_, v)| v[1]).unwrap_or(0.0);
        assert!(
            last_v2 > 1.5,
            "V(cap) after 10*tau should be near 1.8V, got {last_v2}"
        );

        // Waveform should show monotonic charging
        for i in 1..solver.waveform.len() {
            assert!(
                solver.waveform[i].1[1] >= solver.waveform[i - 1].1[1] - 1e-15,
                "Capacitor voltage should be monotonically increasing"
            );
        }
    }

    #[test]
    fn test_stats_tracking() {
        let netlist = PartitionNetlist {
            name: "Stats test".into(),
            num_nodes: 2,
            elements: vec![
                Element::VoltageSource {
                    pos: 1,
                    neg: 0,
                    v: 1.0,
                },
                Element::Resistor {
                    a: 1,
                    b: 2,
                    r: 100.0,
                },
                Element::Resistor {
                    a: 2,
                    b: 0,
                    r: 100.0,
                },
            ],
            ghost_map: vec![],
        };

        let config = SimConfig {
            tstop: 10e-12,
            dt: 1e-12,
            ..Default::default()
        };

        let mut solver =
            TransientSolver::new(netlist, GhostBuffer::new(), CommLayer::new(0, 1), config);
        let (_state, stats) = solver.run();

        assert_eq!(stats.total_steps, 10, "Should have 10 steps");
        assert!(stats.wall_time_secs > 0.0, "Wall time should be recorded");
        assert_eq!(stats.rollbacks, 0, "No rollbacks in single partition");
    }
}
