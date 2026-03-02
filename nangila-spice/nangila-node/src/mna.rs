//! Modified Nodal Analysis (MNA) Engine
//!
//! Builds and stamps the MNA matrix for a sub-circuit partition.
//! Each circuit element contributes "stamps" to the conductance
//! matrix G and the source vector b, producing:
//!
//!   G · x = b
//!
//! Where x is the vector of node voltages.
//!
//! For dynamic elements (capacitors), we use either Backward Euler or Trapezoidal:
//!   - BE:   I_c = C * (V_n - V_{n-1}) / dt
//!   - TRAP: I_c = 2C * (V_n - V_{n-1}) / dt - I_{c, n-1}
//!
//! Phase 1, Sprint 22 deliverable.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntegrationMethod {
    BackwardEuler,
    Trapezoidal,
}

use crate::device_model::{DiodeModel, MosfetLevel1};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SourceWaveform {
    Dc(f64),
    Pulse {
        v1: f64,
        v2: f64,
        td: f64,
        tr: f64,
        tf: f64,
        pw: f64,
        per: f64,
    },
}

impl SourceWaveform {
    pub fn value_at(&self, time: f64) -> f64 {
        match self {
            SourceWaveform::Dc(v) => *v,
            SourceWaveform::Pulse {
                v1,
                v2,
                td,
                tr,
                tf,
                pw,
                per,
            } => {
                if *per <= 0.0 {
                    return *v1;
                }
                if time < *td {
                    return *v1;
                }

                let local = (time - *td) % *per;
                if local < *tr {
                    if *tr <= 0.0 {
                        *v2
                    } else {
                        *v1 + (*v2 - *v1) * (local / *tr)
                    }
                } else if local < *tr + *pw {
                    *v2
                } else if local < *tr + *pw + *tf {
                    if *tf <= 0.0 {
                        *v1
                    } else {
                        let fall = (local - *tr - *pw) / *tf;
                        *v2 + (*v1 - *v2) * fall
                    }
                } else {
                    *v1
                }
            }
        }
    }
}

/// A circuit element in the partition's sub-netlist.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Element {
    /// Resistor: R(node_a, node_b, resistance_ohms)
    Resistor { a: usize, b: usize, r: f64 },
    /// Capacitor: C(node_a, node_b, capacitance_farads)
    Capacitor { a: usize, b: usize, c: f64 },
    /// Inductor: L(node_a, node_b, inductance_henrys)
    Inductor { a: usize, b: usize, l: f64 },
    /// Voltage source: V(node_pos, node_neg, waveform)
    VoltageSource {
        pos: usize,
        neg: usize,
        source: SourceWaveform,
    },
    /// Current source: I(node_pos, node_neg, waveform)
    CurrentSource {
        pos: usize,
        neg: usize,
        source: SourceWaveform,
    },
    /// Ghost node injection: fixed voltage from neighbor partition
    GhostSource { node: usize, voltage: f64 },
    /// Non-linear Diode (node_p, node_n)
    Diode { p: usize, n: usize, model: DiodeModel },
    /// Non-linear MOSFET (Drain, Gate, Source, Bulk)
    Mosfet { d: usize, g: usize, s: usize, b: usize, model: MosfetLevel1 },
}

/// The MNA system: G·x = b with optional companion model for capacitors.
#[derive(Debug, Clone)]
pub struct MnaSystem {
    /// System size (number of unknowns: nodes + voltage source currents)
    pub size: usize,
    /// Number of circuit nodes (excluding ground = 0)
    pub num_nodes: usize,
    /// Number of voltage sources (each adds an extra unknown)
    pub num_vsources: usize,
    /// Number of inductors (each adds an extra unknown)
    pub num_inductors: usize,
    /// Conductance/stamp matrix (row-major, dense for now)
    pub g_matrix: Vec<f64>,
    /// Right-hand side vector
    pub b_vector: Vec<f64>,
    /// Solution vector
    pub x_vector: Vec<f64>,
    /// Circuit elements
    pub elements: Vec<Element>,
}

impl MnaSystem {
    /// Create a new MNA system for a partition.
    pub fn new(num_nodes: usize, elements: Vec<Element>) -> Self {
        // Count voltage sources (each adds one extra equation)
        let num_vsources = elements
            .iter()
            .filter(|e| matches!(e, Element::VoltageSource { .. }))
            .count();
        let num_inductors = elements
            .iter()
            .filter(|e| matches!(e, Element::Inductor { .. }))
            .count();

        let size = num_nodes + num_vsources + num_inductors;

        Self {
            size,
            num_nodes,
            num_vsources,
            num_inductors,
            g_matrix: vec![0.0; size * size],
            b_vector: vec![0.0; size],
            x_vector: vec![0.0; size],
            elements,
        }
    }

    /// Stamp all elements into the G matrix and b vector.
    /// `prev_voltages`: node voltages from previous timestep (for companion).
    /// `prev_cap_currents`: past capacitor branch currents (for Trapezoidal history).
    /// `dt`: timestep for discretization.
    pub fn stamp_all(
        &mut self,
        prev_voltages: &[f64],
        prev_cap_currents: &[f64],
        dt: f64,
        time: f64,
        method: IntegrationMethod,
    ) {
        // Clear matrices
        self.g_matrix.fill(0.0);
        self.b_vector.fill(0.0);

        // Clone elements to avoid borrow conflict with &mut self stamp methods
        let elements = self.elements.clone();
        let mut vsource_idx = 0;
        let mut inductor_idx = 0;
        let mut cap_idx = 0;

        for element in &elements {
            match element {
                Element::Resistor { a, b, r } => {
                    self.stamp_resistor(*a, *b, *r);
                }
                Element::Capacitor { a, b, c } => {
                    let prev_ic = prev_cap_currents.get(cap_idx).copied().unwrap_or(0.0);
                    self.stamp_capacitor(*a, *b, *c, prev_voltages, prev_ic, dt, method);
                    cap_idx += 1;
                }
                Element::Inductor { a, b, l } => {
                    let branch = self.num_nodes + self.num_vsources + inductor_idx;
                    self.stamp_inductor(*a, *b, *l, branch, prev_voltages, dt, method);
                    inductor_idx += 1;
                }
                Element::VoltageSource { pos, neg, source } => {
                    let branch = self.num_nodes + vsource_idx;
                    self.stamp_vsource(*pos, *neg, source.value_at(time), branch);
                    vsource_idx += 1;
                }
                Element::CurrentSource { pos, neg, source } => {
                    self.stamp_current_source(*pos, *neg, source.value_at(time));
                }
                Element::GhostSource { node, voltage } => {
                    self.stamp_ghost(*node, *voltage);
                }
                Element::Mosfet { .. } | Element::Diode { .. } => {
                    // Handled dynamically by Newton-Raphson outer loop
                }
            }
        }
    }

    /// Stamp a resistor: G += 1/R at the (a,a), (b,b), (a,b), (b,a) entries.
    fn stamp_resistor(&mut self, a: usize, b: usize, r: f64) {
        let g = 1.0 / r;
        // Node a (skip if ground = 0)
        if a > 0 {
            let ai = a - 1; // 0-indexed (node 0 = ground, not in matrix)
            self.g_matrix[ai * self.size + ai] += g;
            if b > 0 {
                let bi = b - 1;
                self.g_matrix[ai * self.size + bi] -= g;
            }
        }
        if b > 0 {
            let bi = b - 1;
            self.g_matrix[bi * self.size + bi] += g;
            if a > 0 {
                let ai = a - 1;
                self.g_matrix[bi * self.size + ai] -= g;
            }
        }
    }

    /// Stamp an Inductor (supports BE or Trapezoidal).
    fn stamp_inductor(
        &mut self,
        a: usize,
        b: usize,
        l: f64,
        branch: usize,
        prev: &[f64],
        dt: f64,
        method: IntegrationMethod,
    ) {
        if dt <= 0.0 {
            self.stamp_vsource(a, b, 0.0, branch);
            return;
        }

        let prev_current = if branch < prev.len() { prev[branch] } else { 0.0 };
        
        // TRAP needs previous voltage across L. BE does not, but computing it is cheap.
        let va_prev = if a > 0 && (a - 1) < prev.len() { prev[a - 1] } else { 0.0 };
        let vb_prev = if b > 0 && (b - 1) < prev.len() { prev[b - 1] } else { 0.0 };
        let vl_prev = va_prev - vb_prev;

        let (equiv_res, b_val) = match method {
            IntegrationMethod::BackwardEuler => {
                let eq_r = -l / dt;
                (eq_r, eq_r * prev_current)
            }
            IntegrationMethod::Trapezoidal => {
                let eq_r = -2.0 * l / dt;
                (eq_r, eq_r * prev_current - vl_prev)
            }
        };

        // KCL at A (+1) and B (-1)
        if a > 0 {
            let ai = a - 1;
            self.g_matrix[ai * self.size + branch] += 1.0;
            self.g_matrix[branch * self.size + ai] += 1.0;
        }
        if b > 0 {
            let bi = b - 1;
            self.g_matrix[bi * self.size + branch] -= 1.0;
            self.g_matrix[branch * self.size + bi] -= 1.0;
        }

        self.g_matrix[branch * self.size + branch] += equiv_res;
        self.b_vector[branch] += b_val;
    }

    /// Stamp a capacitor using Backward Euler or Trapezoidal companion models
    fn stamp_capacitor(
        &mut self,
        a: usize,
        b: usize,
        c: f64,
        prev_v: &[f64],
        prev_ic: f64,
        dt: f64,
        method: IntegrationMethod,
    ) {
        if dt <= 0.0 {
            return;
        }

        let va_prev = if a > 0 && (a - 1) < prev_v.len() { prev_v[a - 1] } else { 0.0 };
        let vb_prev = if b > 0 && (b - 1) < prev_v.len() { prev_v[b - 1] } else { 0.0 };

        let (g_eq, i_eq) = match method {
            IntegrationMethod::BackwardEuler => {
                let g = c / dt;
                (g, g * (va_prev - vb_prev))
            }
            IntegrationMethod::Trapezoidal => {
                let g = 2.0 * c / dt;
                (g, g * (va_prev - vb_prev) + prev_ic)
            }
        };

        // Stamp equivalent conductance (same pattern as resistor)
        if a > 0 {
            let ai = a - 1;
            self.g_matrix[ai * self.size + ai] += g_eq;
            if b > 0 {
                let bi = b - 1;
                self.g_matrix[ai * self.size + bi] -= g_eq;
            }
        }
        if b > 0 {
            let bi = b - 1;
            self.g_matrix[bi * self.size + bi] += g_eq;
            if a > 0 {
                let ai = a - 1;
                self.g_matrix[bi * self.size + ai] -= g_eq;
            }
        }

        // Stamp companion current source
        if a > 0 {
            self.b_vector[a - 1] += i_eq;
        }
        if b > 0 {
            self.b_vector[b - 1] -= i_eq;
        }
    }

    /// Stamp a voltage source using MNA augmented row/column.
    fn stamp_vsource(&mut self, pos: usize, neg: usize, v: f64, branch: usize) {
        // KVL equation: V_pos - V_neg = V
        if pos > 0 {
            let pi = pos - 1;
            self.g_matrix[branch * self.size + pi] += 1.0;
            self.g_matrix[pi * self.size + branch] += 1.0;
        }
        if neg > 0 {
            let ni = neg - 1;
            self.g_matrix[branch * self.size + ni] -= 1.0;
            self.g_matrix[ni * self.size + branch] -= 1.0;
        }
        self.b_vector[branch] = v;
    }

    /// Stamp a current source.
    fn stamp_current_source(&mut self, pos: usize, neg: usize, i: f64) {
        if pos > 0 {
            self.b_vector[pos - 1] -= i; // Current flows out of pos
        }
        if neg > 0 {
            self.b_vector[neg - 1] += i; // Current flows into neg
        }
    }

    /// Stamp a ghost node as a strong voltage constraint.
    /// Implemented as a very large conductance to ground (10^12).
    fn stamp_ghost(&mut self, node: usize, voltage: f64) {
        if node > 0 {
            let ni = node - 1;
            let g_large = 1e12;
            self.g_matrix[ni * self.size + ni] += g_large;
            self.b_vector[ni] += g_large * voltage;
        }
    }

    /// Stamp the linearised equivalent conductance and Norton current of a Diode.
    pub fn stamp_diode(&mut self, p: usize, n: usize, g_eq: f64, i_eq: f64) {
        if p > 0 {
            let pi = p - 1;
            self.g_matrix[pi * self.size + pi] += g_eq;
            self.b_vector[pi] -= i_eq; // Equation: sum += I_eq
        }
        if n > 0 {
            let ni = n - 1;
            self.g_matrix[ni * self.size + ni] += g_eq;
            self.b_vector[ni] += i_eq; // Equation: sum -= I_eq
        }
        if p > 0 && n > 0 {
            let pi = p - 1;
            let ni = n - 1;
            self.g_matrix[pi * self.size + ni] -= g_eq;
            self.g_matrix[ni * self.size + pi] -= g_eq;
        }
    }

    /// Stamp the linearised equivalent transconductance and Norton current of a MOSFET.
    pub fn stamp_mosfet(&mut self, d: usize, g: usize, s: usize, gm: f64, gds: f64, i_eq: f64) {
        // D equation: Id leaves drain
        if d > 0 {
            let di = d - 1;
            if g > 0 { self.g_matrix[di * self.size + (g - 1)] += gm; }
            if s > 0 { self.g_matrix[di * self.size + (s - 1)] -= (gm + gds); }
            self.g_matrix[di * self.size + di] += gds;
            self.b_vector[di] -= i_eq;
        }
        // S equation: Id enters source
        if s > 0 {
            let si = s - 1;
            if g > 0 { self.g_matrix[si * self.size + (g - 1)] -= gm; }
            if d > 0 { self.g_matrix[si * self.size + (d - 1)] -= gds; }
            self.g_matrix[si * self.size + si] += (gm + gds);
            self.b_vector[si] += i_eq;
        }
    }

    /// Solve the stamped system G·x = b using Gaussian elimination.
    /// Returns true if solve succeeded.
    pub fn solve(&mut self) -> bool {
        let n = self.size;
        if n == 0 {
            return true;
        }

        // Copy G and b for in-place elimination
        let mut a = self.g_matrix.clone();
        let mut b = self.b_vector.clone();

        // Forward elimination with partial pivoting
        for col in 0..n {
            // Find pivot
            let mut max_val = a[col * n + col].abs();
            let mut max_row = col;
            for row in (col + 1)..n {
                let val = a[row * n + col].abs();
                if val > max_val {
                    max_val = val;
                    max_row = row;
                }
            }

            if max_val < 1e-15 {
                // Singular matrix
                return false;
            }

            // Swap rows if needed
            if max_row != col {
                for j in 0..n {
                    a.swap(col * n + j, max_row * n + j);
                }
                b.swap(col, max_row);
            }

            // Eliminate below
            let pivot = a[col * n + col];
            for row in (col + 1)..n {
                let factor = a[row * n + col] / pivot;
                for j in col..n {
                    a[row * n + j] -= factor * a[col * n + j];
                }
                b[row] -= factor * b[col];
            }
        }

        // Back substitution
        for i in (0..n).rev() {
            let mut sum = b[i];
            for j in (i + 1)..n {
                sum -= a[i * n + j] * self.x_vector[j];
            }
            self.x_vector[i] = sum / a[i * n + i];
        }

        true
    }

    /// Get node voltage from solution (1-indexed, 0 = ground = 0V).
    pub fn node_voltage(&self, node: usize) -> f64 {
        if node == 0 {
            0.0
        } else if node - 1 < self.x_vector.len() {
            self.x_vector[node - 1]
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voltage_divider() {
        // V1=10V, R1=1k (node 1→2), R2=1k (node 2→0)
        // Expected: V(2) = 5V
        let elements = vec![
            Element::VoltageSource {
                pos: 1,
                neg: 0,
                source: SourceWaveform::Dc(10.0),
            },
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
        ];

        let mut mna = MnaSystem::new(2, elements);
        mna.stamp_all(&[], &[], 1e-9, 1e-9, IntegrationMethod::BackwardEuler);
        assert!(mna.solve(), "Solve should succeed");

        let v2 = mna.node_voltage(2);
        assert!((v2 - 5.0).abs() < 1e-6, "V(2) should be 5.0V, got {v2}");
    }

    #[test]
    fn test_rc_charging() {
        // V1=1.8V, R=1k (node 1→2), C=10f (node 2→0)
        // Check that V(2) increases over time toward 1.8V
        let elements = vec![
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
        ];

        let dt = 1e-12; // 1ps
        let mut prev = vec![0.0; 3]; // [V(1), V(2), I_vsource]
        let mut voltages = Vec::new();

        for _step in 0..100 {
            let mut mna = MnaSystem::new(2, elements.clone());
            mna.stamp_all(&prev, &[], dt, (_step as f64 + 1.0) * dt, IntegrationMethod::BackwardEuler);
            assert!(mna.solve(), "Solve should succeed");

            voltages.push(mna.node_voltage(2));
            prev = mna.x_vector.clone();
        }

        // V(2) should be monotonically increasing
        for i in 1..voltages.len() {
            assert!(
                voltages[i] >= voltages[i - 1] - 1e-15,
                "V(2) should be monotonically increasing"
            );
        }

        // After 100ps with RC = 1k * 10fF = 10ps, should be near 1.8V
        let final_v = *voltages.last().unwrap();
        assert!(
            final_v > 1.7,
            "After 10 tau, V(2) should be near 1.8V, got {final_v}"
        );
    }

    #[test]
    fn test_ghost_node_injection() {
        // R1 from node 1 to node 2, ghost at node 1 = 3.3V
        // R2 from node 2 to ground
        let elements = vec![
            Element::GhostSource {
                node: 1,
                voltage: 3.3,
            },
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
        ];

        let mut mna = MnaSystem::new(2, elements);
        mna.stamp_all(&[], &[], 1e-9, 1e-9, IntegrationMethod::BackwardEuler);
        assert!(mna.solve(), "Solve should succeed");

        let v1 = mna.node_voltage(1);
        let v2 = mna.node_voltage(2);

        assert!(
            (v1 - 3.3).abs() < 0.01,
            "Ghost node V(1) should be ~3.3V, got {v1}"
        );
        assert!(
            (v2 - 1.65).abs() < 0.01,
            "V(2) should be ~1.65V (divider), got {v2}"
        );
    }
}
