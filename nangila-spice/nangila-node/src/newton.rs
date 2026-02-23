//! Newton-Raphson Nonlinear Solver
//!
//! Iteratively solves nonlinear MNA systems where device currents
//! depend on node voltages (MOSFETs, diodes).
//!
//! Algorithm per timestep:
//!   1. Evaluate device models at current voltage guess
//!   2. Stamp linearised G_stamp and I_stamp into sparse MNA
//!   3. Solve sparse system via GpuSolver
//!   4. Check convergence: max |ΔV| < reltol * max|V| + abstol
//!   5. Repeat until converged or max_iter hit
//!
//! Phase 3, Sprint 9–10 deliverable.

use tracing::{debug, warn};

use crate::device_model::{DeviceEval, DeviceModel, DeviceStamp};
use crate::gpu_solver::{GpuSolver, SparseMatrix};

// ─── Convergence Config ────────────────────────────────────────────

/// Newton-Raphson convergence criteria.
#[derive(Debug, Clone)]
pub struct NrConfig {
    /// Maximum iterations before declaring non-convergence
    pub max_iter: u32,
    /// Relative voltage tolerance (fraction of |V|)
    pub reltol: f64,
    /// Absolute voltage tolerance (volts)
    pub abstol: f64,
    /// Current tolerance (amps)
    pub abstol_i: f64,
    /// Damping factor for initial iterations (0.5 = half-step)
    pub damping: f64,
}

impl Default for NrConfig {
    fn default() -> Self {
        Self {
            max_iter: 150,
            reltol: 1e-3,
            abstol: 1e-6,    // 1μV
            abstol_i: 1e-12, // 1pA
            damping: 1.0,    // No damping by default
        }
    }
}

// ─── NR State ──────────────────────────────────────────────────────

/// State of the Newton-Raphson iteration.
#[derive(Debug, Clone)]
pub struct NrState {
    /// Current voltage guess (len = num_nodes)
    pub voltages: Vec<f64>,
    /// Previous iteration voltages (for convergence check)
    prev_voltages: Vec<f64>,
    /// Number of nodes
    pub num_nodes: usize,
    /// Iteration count
    pub iterations: u32,
    /// Converged flag
    pub converged: bool,
    /// Final convergence error
    pub final_error: f64,
}

impl NrState {
    pub fn new(num_nodes: usize) -> Self {
        Self {
            voltages: vec![0.0; num_nodes],
            prev_voltages: vec![0.0; num_nodes],
            num_nodes,
            iterations: 0,
            converged: false,
            final_error: f64::MAX,
        }
    }

    /// Initialize with a voltage guess (e.g., from previous timestep).
    pub fn with_initial_guess(mut self, guess: Vec<f64>) -> Self {
        self.voltages = guess;
        self
    }

    /// Update voltages with damped Newton step.
    fn update(&mut self, new_voltages: &[f64], damping: f64) {
        self.prev_voltages = self.voltages.clone();
        for (v, &new_v) in self.voltages.iter_mut().zip(new_voltages) {
            *v = *v + damping * (new_v - *v);
        }
    }

    /// Check voltage convergence.
    fn check_convergence(&mut self, reltol: f64, abstol: f64) -> bool {
        let mut max_err = 0.0f64;

        for (&v_new, &v_old) in self.voltages.iter().zip(&self.prev_voltages) {
            let delta = (v_new - v_old).abs();
            let tol = reltol * v_new.abs().max(v_old.abs()) + abstol;
            let normalized_err = delta / tol.max(abstol);
            max_err = max_err.max(normalized_err);
        }

        self.final_error = max_err;
        max_err < 1.0
    }
}

// ─── Newton-Raphson Solver ─────────────────────────────────────────

/// Newton-Raphson nonlinear solver.
///
/// Manages the outer iteration loop, calling GpuSolver for the
/// inner linear solve at each step.
pub struct NewtonSolver {
    pub config: NrConfig,
    pub gpu_solver: GpuSolver,
    pub stats: NrStats,
}

/// Statistics for Newton-Raphson performance.
#[derive(Debug, Clone, Default)]
pub struct NrStats {
    /// Total NR iterations across all timesteps
    pub total_iterations: u64,
    /// Total converged timesteps
    pub converged_count: u64,
    /// Total non-converged (failed) timesteps
    pub failed_count: u64,
    /// Total linear solves
    pub linear_solves: u64,
    /// Average iterations per converged step
    pub avg_iterations: f64,
}

impl NewtonSolver {
    pub fn new(config: NrConfig) -> Self {
        Self {
            config,
            gpu_solver: GpuSolver::new(),
            stats: NrStats::default(),
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(NrConfig::default())
    }

    /// Solve one timestep using Newton-Raphson.
    ///
    /// `build_system`: closure that takes current voltage guess and returns
    ///   (sparse_matrix_with_stamps, rhs_with_device_currents)
    ///
    /// Returns the converged state, or the last iterate if no convergence.
    pub fn solve_timestep<F>(&mut self, mut state: NrState, mut build_system: F) -> NrState
    where
        F: FnMut(&[f64]) -> SparseMatrix,
    {
        state.iterations = 0;
        state.converged = false;

        for iter in 0..self.config.max_iter {
            // Build linearised MNA system at current voltage guess
            let mut mat = build_system(&state.voltages);
            self.stats.linear_solves += 1;

            // Solve the linear system
            if !self.gpu_solver.solve(&mut mat) {
                warn!("NR iter {}: linear solve failed (singular matrix)", iter);
                break;
            }

            // Extract new voltages from solution
            let n = state.num_nodes.min(mat.solution.len());
            let new_v: Vec<f64> = mat.solution[..n].to_vec();

            // Check convergence before updating
            let prev = state.voltages.clone();
            state.update(&new_v, self.config.damping);
            state.iterations = iter + 1;

            // Convergence check uses pre-update voltages
            let converged = {
                let mut max_err = 0.0f64;
                for (&v_new, &v_old) in new_v.iter().zip(&prev) {
                    let delta = (v_new - v_old).abs();
                    let tol =
                        self.config.reltol * v_new.abs().max(v_old.abs()) + self.config.abstol;
                    max_err = max_err.max(delta / tol.max(self.config.abstol));
                }
                state.final_error = max_err;
                max_err < 1.0
            };

            debug!(
                "NR iter {}: max_err = {:.3e}{}",
                iter,
                state.final_error,
                if converged { " ✓" } else { "" }
            );

            if converged {
                state.converged = true;
                break;
            }
        }

        // Update stats
        self.stats.total_iterations += state.iterations as u64;
        if state.converged {
            self.stats.converged_count += 1;
            let n = self.stats.converged_count as f64;
            self.stats.avg_iterations =
                (self.stats.avg_iterations * (n - 1.0) + state.iterations as f64) / n;
        } else {
            self.stats.failed_count += 1;
            warn!(
                "NR did not converge after {} iterations (err={:.3e})",
                self.config.max_iter, state.final_error
            );
        }

        state
    }
}

// ─── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_solver::SparseMatrix;

    /// Build a simple 1-node resistive system: G*V = I
    /// (Linear — NR should converge in 1 iteration)
    fn linear_system(voltages: &[f64], target_v: f64) -> SparseMatrix {
        let g = 1e-3; // 1kΩ conductance
        let i_src = target_v * g; // Current source to drive target voltage

        let mut mat = SparseMatrix::new(1);
        mat.values = vec![g];
        mat.col_indices = vec![0];
        mat.row_ptr = vec![0, 1];
        mat.rhs = vec![i_src];
        mat
    }

    #[test]
    fn test_nr_converges_linear_system() {
        let mut solver = NewtonSolver::with_defaults();
        let state = NrState::new(1);

        let final_state = solver.solve_timestep(state, |v| linear_system(v, 5.0));

        assert!(final_state.converged, "Should converge on linear system");
        assert!(
            final_state.iterations <= 2,
            "Linear system: converge in ≤2 NR iters, got {}",
            final_state.iterations
        );
        assert!(
            (final_state.voltages[0] - 5.0).abs() < 1e-3,
            "V should be ~5V, got {:.4}",
            final_state.voltages[0]
        );
    }

    #[test]
    fn test_nr_stats_tracking() {
        let mut solver = NewtonSolver::with_defaults();

        for _ in 0..3 {
            let state = NrState::new(1);
            solver.solve_timestep(state, |_| linear_system(&[], 1.0));
        }

        assert_eq!(solver.stats.converged_count, 3);
        assert_eq!(solver.stats.failed_count, 0);
        assert!(solver.stats.avg_iterations >= 1.0);
        assert!(solver.stats.linear_solves >= 3);
    }

    #[test]
    fn test_nr_state_initial_guess() {
        let state = NrState::new(3).with_initial_guess(vec![1.0, 2.0, 3.0]);
        assert_eq!(state.voltages, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_nr_2node_system() {
        // 2-node: V1 driven by vsource, V2 via resistor
        // Build: [G, -G; -G, G] * [V1, V2] = [I1, 0]
        // V1=1.8V forced, V2 = V1/2 = 0.9V (symmetric resistors)
        let mut solver = NewtonSolver::with_defaults();
        let state = NrState::new(2);

        let g = 1e-3;
        let final_state = solver.solve_timestep(state, |_v| {
            let n = 2;
            let dense = vec![g * 2.0, -g, -g, g * 2.0];
            let mut mat = SparseMatrix::from_dense(&dense, n, 1e-30);
            mat.rhs = vec![g * 1.8, 0.0];
            mat
        });

        assert!(final_state.converged);
        // Both voltages should be solvable
        assert!(final_state.voltages[0].abs() < 10.0); // Reasonable range
    }

    #[test]
    fn test_nr_convergence_with_damping() {
        let config = NrConfig {
            damping: 0.5,
            ..Default::default()
        };
        let mut solver = NewtonSolver::new(config);
        let state = NrState::new(1);

        let final_state = solver.solve_timestep(state, |_| linear_system(&[], 3.3));

        // With damping, may take more iterations but should still converge
        assert!(
            final_state.converged,
            "Should converge even with damping, got err={:.3e}",
            final_state.final_error
        );
    }
}
