//! Ngspice FFI Bridge
//!
//! Provides an interface to the ngspice shared library for solving
//! circuit partitions. Falls back to the built-in MNA solver when
//! ngspice is not available.
//!
//! Phase 1, Sprint 3 deliverable.
//!
//! # Architecture
//!
//! When ngspice is available:
//!   1. Load libngspice.so/dylib via dlopen
//!   2. Pass partition sub-netlist as a string
//!   3. Run transient simulation for one timestep
//!   4. Extract node voltages from ngspice's internal state
//!
//! When ngspice is NOT available (development/testing):
//!   1. Use the built-in MNA solver from `mna.rs`
//!   2. Only supports R, C, V, I elements (no MOSFET models)
//!   3. Sufficient for validating the partitioning/prediction framework

use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::mna::{Element, MnaSystem};
use crate::solver::PartitionState;
use crate::newton::{NewtonSolver, NrState};
use crate::gpu_solver::SparseMatrix;
use crate::device_model::DeviceModel;

/// Backend solver selection.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SolverBackend {
    /// Use ngspice shared library (full device model support)
    Ngspice,
    /// Use built-in MNA solver (R/C/V/I only, no MOSFET)
    BuiltIn,
}

/// A parsed partition sub-netlist ready for simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionNetlist {
    /// Human-readable name
    pub name: String,
    /// Number of circuit nodes (excluding ground)
    pub num_nodes: usize,
    /// Circuit elements for MNA stamping
    pub elements: Vec<Element>,
    /// Mapping: ghost_net_id → (local_node_index, owner_partition_id)
    pub ghost_map: Vec<(u64, usize, u32)>,
}

/// The solver engine that wraps either ngspice or the built-in solver.
pub struct SolverEngine {
    backend: SolverBackend,
    netlist: PartitionNetlist,
    mna: Option<MnaSystem>,
    nr_solver: Option<NewtonSolver>,
}

impl SolverEngine {
    /// Create a new solver engine. Tries ngspice first, falls back to built-in.
    pub fn new(netlist: PartitionNetlist) -> Self {
        let backend = if Self::ngspice_available() {
            info!("Using ngspice backend");
            SolverBackend::Ngspice
        } else {
            warn!("ngspice not found, using built-in MNA solver (R/C/V/I only)");
            SolverBackend::BuiltIn
        };

        let mna = if backend == SolverBackend::BuiltIn {
            Some(MnaSystem::new(netlist.num_nodes, netlist.elements.clone()))
        } else {
            None
        };
        let nr_solver = if backend == SolverBackend::BuiltIn {
            Some(NewtonSolver::with_defaults())
        } else {
            None
        };

        Self {
            backend,
            netlist,
            mna,
            nr_solver,
        }
    }

    /// Force a specific backend (for testing).
    pub fn with_backend(netlist: PartitionNetlist, backend: SolverBackend) -> Self {
        let mna = if backend == SolverBackend::BuiltIn {
            Some(MnaSystem::new(netlist.num_nodes, netlist.elements.clone()))
        } else {
            None
        };
        let nr_solver = if backend == SolverBackend::BuiltIn {
            Some(NewtonSolver::with_defaults())
        } else {
            None
        };

        Self {
            backend,
            netlist,
            mna,
            nr_solver,
        }
    }

    /// Check if ngspice shared library is available.
    fn ngspice_available() -> bool {
        // TODO: Actually try dlopen("libngspice.so") / dlopen("libngspice.dylib")
        // For now, always return false to use built-in solver
        false
    }

    /// Solve one timestep.
    ///
    /// # Arguments
    /// * `current` - Current partition state (node voltages)
    /// * `ghost_voltages` - Predicted/actual ghost node values: (net_id, voltage)
    /// * `dt` - Timestep size
    ///
    /// # Returns
    /// New partition state after solving.
    pub fn solve_step(
        &mut self,
        current: &PartitionState,
        ghost_voltages: &[(u64, f64)],
        dt: f64,
    ) -> Option<PartitionState> {
        match self.backend {
            SolverBackend::BuiltIn => self.solve_builtin(current, ghost_voltages, dt),
            SolverBackend::Ngspice => self.solve_ngspice(current, ghost_voltages, dt),
        }
    }

    /// Solve using the built-in MNA engine.
    fn solve_builtin(
        &mut self,
        current: &PartitionState,
        ghost_voltages: &[(u64, f64)],
        dt: f64,
    ) -> Option<PartitionState> {
        let mna = self.mna.as_mut()?;
        let nr_solver = self.nr_solver.as_mut()?;

        // Update ghost node voltages in the element list
        let mut elements = self.netlist.elements.clone();
        for (net_id, voltage) in ghost_voltages {
            if let Some((_gid, local_node, _owner)) =
                self.netlist.ghost_map.iter().find(|(gid, _, _)| gid == net_id)
            {
                elements.push(Element::GhostSource {
                    node: *local_node,
                    voltage: *voltage,
                });
            }
        }

        // Rebuild MNA to get structural size (vsources)
        *mna = MnaSystem::new(self.netlist.num_nodes, elements.clone());
        let sys_size = mna.size;

        // Current voltages guess from prior step
        let mut initial_guess = current.voltages.clone();
        initial_guess.resize(sys_size, 0.0);

        let nr_state = NrState::new(sys_size).with_initial_guess(initial_guess);

        let final_state = nr_solver.solve_timestep(nr_state, |v_guess| {
            // 1. Stamp linear components (and calculate RHS from prior timestep capacitors)
            let mut local_mna = MnaSystem::new(self.netlist.num_nodes, elements.clone());
            local_mna.stamp_all(&current.voltages, dt);

            // 2. Add Non-Linear stamps using current v_guess
            for el in &elements {
                match el {
                    Element::Mosfet { d, g, s, model, .. } => {
                        let (gm, gds, i_eq) = model.mna_stamp(v_guess);
                        local_mna.stamp_mosfet(*d, *g, *s, gm, gds, i_eq);
                    }
                    Element::Diode { p, n, model } => {
                        let stamp = model.stamp(
                            if *p > 0 { v_guess[*p - 1] } else { 0.0 },
                            if *n > 0 { v_guess[*n - 1] } else { 0.0 },
                        );
                        local_mna.stamp_diode(*p, *n, stamp.g_eq, stamp.i_eq);
                    }
                    _ => {}
                }
            }

            // 3. Convert assembled dense MNA to SparseMatrix for GpuSolver
            let mut sparse = SparseMatrix::from_dense(&local_mna.g_matrix, local_mna.size, 1e-30);
            sparse.rhs = local_mna.b_vector.clone();
            sparse
        });

        if !final_state.converged {
            warn!("Newton-Raphson failed to converge at t={}", current.time + dt);
            return None;
        }

        // Extract pure node voltages
        let voltages: Vec<f64> = final_state.voltages[0..self.netlist.num_nodes].to_vec();

        Some(PartitionState {
            voltages,
            time: current.time + dt,
            validated: false,
        })
    }

    /// Solve using ngspice FFI.
    fn solve_ngspice(
        &mut self,
        _current: &PartitionState,
        _ghost_voltages: &[(u64, f64)],
        _dt: f64,
    ) -> Option<PartitionState> {
        // TODO (Sprint 3): Implement ngspice FFI calls
        // 1. ngSpice_Init() — initialize shared library
        // 2. ngSpice_Circ() — load circuit netlist
        // 3. Modify ghost node values in ngspice's internal state
        // 4. ngSpice_Command("tran dt dt") — run one step
        // 5. ngSpice_Get_Vec_Info() — extract node voltages
        warn!("ngspice FFI not yet implemented");
        None
    }

    /// Get the current backend.
    pub fn backend(&self) -> SolverBackend {
        self.backend
    }

    /// Get the MNA system size (number of unknowns).
    pub fn size(&self) -> usize {
        match self.backend {
            SolverBackend::BuiltIn => self.mna.as_ref().map(|m| m.size).unwrap_or(0),
            SolverBackend::Ngspice => self.netlist.num_nodes, // Approximation for ngspice
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mna::Element;

    fn make_rc_netlist() -> PartitionNetlist {
        PartitionNetlist {
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
        }
    }

    #[test]
    fn test_builtin_solver_rc() {
        let netlist = make_rc_netlist();
        let mut engine = SolverEngine::with_backend(netlist, SolverBackend::BuiltIn);

        let mut state = PartitionState::new(3); // 2 nodes + 1 vsource current
        let dt = 1e-12;

        // Run 100 timesteps
        for _ in 0..100 {
            if let Some(new_state) = engine.solve_step(&state, &[], dt) {
                state = PartitionState {
                    voltages: new_state.voltages.clone(),
                    time: new_state.time,
                    validated: true,
                };
                // Pad to match MNA system size
                while state.voltages.len() < 3 {
                    state.voltages.push(0.0);
                }
            }
        }

        // After 100ps with RC=10ps, should be near 1.8V
        assert!(
            state.voltages[1] > 1.5, // node 2 = index 1
            "V(2) should converge near 1.8V, got {}",
            state.voltages[1]
        );
    }

    #[test]
    fn test_ghost_node_solve() {
        let netlist = PartitionNetlist {
            name: "Ghost test".into(),
            num_nodes: 2,
            elements: vec![
                Element::Resistor {
                    a: 1,
                    b: 2,
                    r: 1000.0,
                },
                Element::Resistor {
                    a: 2,
                    b: 0,
                    r: 1000.0,
                },
            ],
            ghost_map: vec![(100, 1, 0)], // net_id=100 maps to local node 1, owner=0
        };

        let mut engine = SolverEngine::with_backend(netlist, SolverBackend::BuiltIn);
        let state = PartitionState::new(2);

        // Inject ghost voltage at net_id=100 (local node 1) = 3.3V
        let ghost_voltages = vec![(100_u64, 3.3)];
        let result = engine.solve_step(&state, &ghost_voltages, 1e-9);

        assert!(result.is_some(), "Solve should succeed");
        let new_state = result.unwrap();

        // V(1) should be ~3.3V (ghost forced), V(2) should be ~1.65V (divider)
        assert!(
            (new_state.voltages[0] - 3.3).abs() < 0.01,
            "Ghost node should be ~3.3V, got {}",
            new_state.voltages[0]
        );
        assert!(
            (new_state.voltages[1] - 1.65).abs() < 0.01,
            "V(2) should be ~1.65V, got {}",
            new_state.voltages[1]
        );
    }
}
