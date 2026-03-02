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
    #[serde(default)]
    pub ghost_map: Vec<(u64, usize, u32)>,
    /// Ordered node names for waveform export (index 0 -> node 1)
    #[serde(default)]
    pub node_names: Vec<String>,
    /// Initial node voltages: (1-indexed node ID, volts)
    #[serde(default)]
    pub initial_conditions: Vec<(usize, f64)>,
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

    fn seeded_state(&self) -> PartitionState {
        let mut state = PartitionState::new(self.size());
        for element in &self.netlist.elements {
            match element {
                Element::VoltageSource { pos, neg, source } => {
                    let value = source.value_at(0.0);
                    if *pos > 0 && *neg == 0 && *pos - 1 < state.voltages.len() {
                        state.voltages[*pos - 1] = value;
                    } else if *neg > 0 && *pos == 0 && *neg - 1 < state.voltages.len() {
                        state.voltages[*neg - 1] = -value;
                    }
                }
                _ => {}
            }
        }
        for (node, voltage) in &self.netlist.initial_conditions {
            if *node > 0 && *node - 1 < state.voltages.len() {
                state.voltages[*node - 1] = *voltage;
            }
        }
        state
    }

    pub fn initial_state(&mut self) -> PartitionState {
        let seeded = self.seeded_state();
        if self.backend != SolverBackend::BuiltIn || !self.netlist.initial_conditions.is_empty() {
            return seeded;
        }

        self.solve_dc_operating_point(&seeded).unwrap_or(seeded)
    }

    pub fn node_names(&self) -> &[String] {
        &self.netlist.node_names
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

        let mut method = if current.cap_currents.is_empty() {
            crate::mna::IntegrationMethod::BackwardEuler
        } else {
            crate::mna::IntegrationMethod::Trapezoidal
        };
        let nr_state = NrState::new(sys_size).with_initial_guess(initial_guess);

        let mut final_state = nr_solver.solve_timestep(nr_state.clone(), |v_guess| {
            let mut local_mna = MnaSystem::new(self.netlist.num_nodes, elements.clone());
            local_mna.stamp_all(
                &current.voltages,
                &current.cap_currents,
                dt,
                current.time + dt,
                method,
            );

            // Add Non-Linear stamps using current v_guess
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
            let mut sparse = SparseMatrix::from_dense(&local_mna.g_matrix, local_mna.size, 1e-30);
            sparse.rhs = local_mna.b_vector.clone();
            sparse
        });

        // Fallback Trigger: If TRAP rings/diverges, damp it with Backward-Euler
        if !final_state.converged {
            warn!("TRAP ringing/divergence detected at t={:.2e}s, falling back to Backward-Euler damping", current.time + dt);
            method = crate::mna::IntegrationMethod::BackwardEuler;
            final_state = nr_solver.solve_timestep(nr_state, |v_guess| {
                let mut local_mna = MnaSystem::new(self.netlist.num_nodes, elements.clone());
                local_mna.stamp_all(
                    &current.voltages,
                    &current.cap_currents,
                    dt,
                    current.time + dt,
                    method,
                );
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
                let mut sparse = SparseMatrix::from_dense(&local_mna.g_matrix, local_mna.size, 1e-30);
                sparse.rhs = local_mna.b_vector.clone();
                sparse
            });
        }

        if !final_state.converged {
            warn!("Newton-Raphson failed to converge at t={:.2e}s even with BE damping", current.time + dt);
            return None;
        }

        // Extract full state vector including nodes + dynamic branch variables
        let voltages: Vec<f64> = final_state.voltages[0..sys_size].to_vec();

        // Calculate and cache capacitor historical states for the next Trapezoidal step
        let mut cap_currents = Vec::new();
        let mut cap_idx = 0;
        for el in &self.netlist.elements {
            if let Element::Capacitor { a, b, c } = el {
                let va_new = if *a > 0 { voltages[*a - 1] } else { 0.0 };
                let vb_new = if *b > 0 { voltages[*b - 1] } else { 0.0 };
                let va_old = if *a > 0 { current.voltages.get(*a - 1).copied().unwrap_or(0.0) } else { 0.0 };
                let vb_old = if *b > 0 { current.voltages.get(*b - 1).copied().unwrap_or(0.0) } else { 0.0 };
                
                let prev_ic = current.cap_currents.get(cap_idx).copied().unwrap_or(0.0);
                
                let ic_new = if dt > 0.0 {
                    match method {
                        crate::mna::IntegrationMethod::BackwardEuler => {
                            (c / dt) * ((va_new - vb_new) - (va_old - vb_old))
                        }
                        crate::mna::IntegrationMethod::Trapezoidal => {
                            (2.0 * c / dt) * ((va_new - vb_new) - (va_old - vb_old)) - prev_ic
                        }
                    }
                } else {
                    0.0
                };
                
                cap_currents.push(ic_new);
                cap_idx += 1;
            }
        }

        Some(PartitionState {
            voltages,
            cap_currents,
            time: current.time + dt,
            validated: false,
        })
    }

    fn solve_dc_operating_point(&mut self, seeded: &PartitionState) -> Option<PartitionState> {
        let elements = self.netlist.elements.clone();
        let num_nodes = self.netlist.num_nodes;
        let mna = self.mna.as_mut()?;
        let nr_solver = self.nr_solver.as_mut()?;

        *mna = MnaSystem::new(num_nodes, elements.clone());
        let sys_size = mna.size;

        let mut initial_guess = seeded.voltages.clone();
        initial_guess.resize(sys_size, 0.0);

        let nr_state = NrState::new(sys_size).with_initial_guess(initial_guess);
        let final_state = nr_solver.solve_timestep(nr_state, |v_guess| {
            let mut local_mna = MnaSystem::new(num_nodes, elements.clone());
            local_mna.stamp_all(&[], &[], 0.0, 0.0, crate::mna::IntegrationMethod::BackwardEuler);

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

            let mut sparse = SparseMatrix::from_dense(&local_mna.g_matrix, local_mna.size, 1e-30);
            sparse.rhs = local_mna.b_vector.clone();
            sparse
        });

        if !final_state.converged {
            warn!("DC operating point failed to converge, falling back to seeded initial state");
            return None;
        }

        Some(PartitionState {
            voltages: final_state.voltages[0..sys_size].to_vec(),
            cap_currents: Vec::new(),
            time: 0.0,
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
    use crate::mna::{Element, SourceWaveform};

    fn make_rc_netlist() -> PartitionNetlist {
        PartitionNetlist {
            name: "RC test".into(),
            num_nodes: 2,
            elements: vec![
                Element::VoltageSource {
                    pos: 1,
                    neg: 0,
                    source: SourceWaveform::Dc(1.8),
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
            node_names: vec!["vdd".into(), "out".into()],
            initial_conditions: vec![],
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
                    cap_currents: new_state.cap_currents.clone(),
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
    fn test_initial_state_uses_dc_operating_point_without_ic() {
        let netlist = make_rc_netlist();
        let mut engine = SolverEngine::with_backend(netlist, SolverBackend::BuiltIn);

        let state = engine.initial_state();

        assert!((state.voltages[0] - 1.8).abs() < 1e-9);
        assert!((state.voltages[1] - 1.8).abs() < 1e-6);
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
            node_names: vec!["n1".into(), "n2".into()],
            initial_conditions: vec![],
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
