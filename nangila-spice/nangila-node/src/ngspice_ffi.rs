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

use tracing::{info, warn};

use crate::mna::{Element, MnaSystem};
use crate::solver::PartitionState;

/// Backend solver selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SolverBackend {
    /// Use ngspice shared library (full device model support)
    Ngspice,
    /// Use built-in MNA solver (R/C/V/I only, no MOSFET)
    BuiltIn,
}

/// A parsed partition sub-netlist ready for simulation.
#[derive(Debug, Clone)]
pub struct PartitionNetlist {
    /// Human-readable name
    pub name: String,
    /// Number of circuit nodes (excluding ground)
    pub num_nodes: usize,
    /// Circuit elements for MNA stamping
    pub elements: Vec<Element>,
    /// Mapping: ghost_net_id → local_node_index
    pub ghost_map: Vec<(u64, usize)>,
}

/// The solver engine that wraps either ngspice or the built-in solver.
pub struct SolverEngine {
    backend: SolverBackend,
    netlist: PartitionNetlist,
    mna: Option<MnaSystem>,
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
            Some(MnaSystem::new(
                netlist.num_nodes,
                netlist.elements.clone(),
            ))
        } else {
            None
        };

        Self {
            backend,
            netlist,
            mna,
        }
    }

    /// Force a specific backend (for testing).
    pub fn with_backend(netlist: PartitionNetlist, backend: SolverBackend) -> Self {
        let mna = if backend == SolverBackend::BuiltIn {
            Some(MnaSystem::new(
                netlist.num_nodes,
                netlist.elements.clone(),
            ))
        } else {
            None
        };

        Self {
            backend,
            netlist,
            mna,
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

        // Update ghost node voltages in the element list
        let mut elements = self.netlist.elements.clone();
        for (net_id, voltage) in ghost_voltages {
            // Find the local node for this ghost
            if let Some((_gid, local_node)) = self
                .netlist
                .ghost_map
                .iter()
                .find(|(gid, _)| gid == net_id)
            {
                // Add/update ghost source
                elements.push(Element::GhostSource {
                    node: *local_node,
                    voltage: *voltage,
                });
            }
        }

        // Rebuild MNA with updated ghost values
        *mna = MnaSystem::new(self.netlist.num_nodes, elements);

        // Stamp and solve
        mna.stamp_all(&current.voltages, dt);

        if !mna.solve() {
            warn!("MNA solve failed (singular matrix)");
            return None;
        }

        // Extract node voltages (only the node voltages, not branch currents)
        let voltages: Vec<f64> = (0..self.netlist.num_nodes)
            .map(|i| mna.x_vector[i])
            .collect();

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
            ghost_map: vec![(100, 1)], // net_id=100 maps to local node 1
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
