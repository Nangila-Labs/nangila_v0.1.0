//! Benchmark Harness
//!
//! Generates synthetic benchmark circuits and measures performance.
//! Includes a CiM (Compute-in-Memory) array generator for stress testing
//! the partitioned solver at scale.
//!
//! Phase 2, Sprint 8 deliverable.

use std::time::Instant;
use tracing::info;

use crate::comm::CommLayer;
use crate::ghost::GhostBuffer;
use crate::mna::Element;
use crate::ngspice_ffi::PartitionNetlist;
use crate::solver::{SimConfig, SimStats, TransientSolver};

// ─── Circuit Generators ────────────────────────────────────────────

/// Generate a CiM (Compute-in-Memory) array circuit.
///
/// A CiM array is a crossbar of resistive memory elements.
/// Structure: n_rows × n_cols crossbar with wordline drivers and bitline caps.
pub fn generate_cim_array(n_rows: usize, n_cols: usize) -> Vec<Element> {
    let mut elements = Vec::new();

    // Node numbering:
    //   WL nodes: 1..n_rows (wordlines)
    //   BL nodes: n_rows+1..n_rows+n_cols (bitlines)

    // Wordline drivers (voltage sources to ground)
    for row in 0..n_rows {
        let wl_node = row + 1;
        let voltage = if row == 0 { 1.8 } else { 0.0 }; // Activate WL0 only
        elements.push(Element::VoltageSource {
            pos: wl_node,
            neg: 0,
            v: voltage,
        });
    }

    // CiM cells: resistors between WL_i and BL_j
    for row in 0..n_rows {
        for col in 0..n_cols {
            let wl_node = row + 1;
            let bl_node = n_rows + col + 1;
            // Resistance varies to simulate different stored values (HRS/LRS)
            let resistance = if (row + col) % 3 == 0 {
                1e6 // HRS (high resistance state)
            } else {
                10e3 // LRS (low resistance state)
            };
            elements.push(Element::Resistor {
                a: wl_node,
                b: bl_node,
                r: resistance,
            });
        }
    }

    // Bitline load capacitors (sense amplifier input)
    for col in 0..n_cols {
        let bl_node = n_rows + col + 1;
        elements.push(Element::Capacitor {
            a: bl_node,
            b: 0,
            c: 100e-15, // 100fF
        });
    }

    elements
}

/// Generate an RC ladder circuit (good for testing wave propagation).
pub fn generate_rc_ladder(n_stages: usize, r: f64, c: f64) -> Vec<Element> {
    let mut elements = Vec::new();

    // Input voltage source
    elements.push(Element::VoltageSource {
        pos: 1,
        neg: 0,
        v: 1.8,
    });

    // RC stages
    for i in 0..n_stages {
        let node_in = i + 1;
        let node_out = i + 2;

        elements.push(Element::Resistor {
            a: node_in,
            b: node_out,
            r,
        });

        elements.push(Element::Capacitor {
            a: node_out,
            b: 0,
            c,
        });
    }

    elements
}

/// Generate a mesh resistor network (for testing partitioning).
pub fn generate_resistor_mesh(n: usize, r: f64) -> Vec<Element> {
    let mut elements = Vec::new();
    let node = |row: usize, col: usize| -> usize { row * n + col + 1 };

    // Voltage source at corner
    elements.push(Element::VoltageSource {
        pos: node(0, 0),
        neg: 0,
        v: 1.8,
    });

    // Current sink at opposite corner
    elements.push(Element::CurrentSource {
        pos: 0,
        neg: node(n - 1, n - 1),
        i: 1e-3,
    });

    // Horizontal resistors
    for row in 0..n {
        for col in 0..n - 1 {
            elements.push(Element::Resistor {
                a: node(row, col),
                b: node(row, col + 1),
                r,
            });
        }
    }

    // Vertical resistors
    for row in 0..n - 1 {
        for col in 0..n {
            elements.push(Element::Resistor {
                a: node(row, col),
                b: node(row + 1, col),
                r,
            });
        }
    }

    elements
}

// ─── Benchmark Runner ──────────────────────────────────────────────

/// Result of a benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub element_count: usize,
    pub node_count: usize,
    pub wall_time_secs: f64,
    pub timesteps: u64,
    pub time_per_step: f64,
    pub throughput: f64,
    pub rollbacks: u64,
    pub prediction_hit_rate: f64,
}

impl BenchmarkResult {
    pub fn summary(&self) -> String {
        format!(
            "{}: {} elements, {} nodes | {:.3}s total, {:.2e}s/step, {:.0} elem/s | {} rollbacks, {:.1}% hits",
            self.name,
            self.element_count,
            self.node_count,
            self.wall_time_secs,
            self.time_per_step,
            self.throughput,
            self.rollbacks,
            self.prediction_hit_rate * 100.0,
        )
    }
}

/// Count unique non-ground nodes in an element list.
fn count_nodes(elements: &[Element]) -> usize {
    let mut nodes = std::collections::HashSet::new();
    for e in elements {
        match e {
            Element::Resistor { a, b, .. } => {
                nodes.insert(*a);
                nodes.insert(*b);
            }
            Element::Capacitor { a, b, .. } => {
                nodes.insert(*a);
                nodes.insert(*b);
            }
            Element::VoltageSource { pos, neg, .. } => {
                nodes.insert(*pos);
                nodes.insert(*neg);
            }
            Element::CurrentSource { pos, neg, .. } => {
                nodes.insert(*pos);
                nodes.insert(*neg);
            }
            Element::GhostSource { node, .. } => {
                nodes.insert(*node);
            }
        }
    }
    nodes.remove(&0); // remove ground
    nodes.len()
}

/// Run a benchmark on a circuit.
pub fn run_benchmark(name: &str, elements: Vec<Element>, config: SimConfig) -> BenchmarkResult {
    let element_count = elements.len();
    let node_count = count_nodes(&elements);

    let netlist = PartitionNetlist {
        name: name.to_string(),
        elements,
        num_nodes: node_count,
        ghost_map: vec![],
    };

    let ghost_buffer = GhostBuffer::new();
    let comm = CommLayer::new(0, 1);

    let start = Instant::now();
    let mut solver = TransientSolver::new(netlist, ghost_buffer, comm, config);
    let (_state, stats) = solver.run();
    let elapsed = start.elapsed().as_secs_f64();

    let timesteps = stats.total_steps;
    let time_per_step = if timesteps > 0 {
        elapsed / timesteps as f64
    } else {
        0.0
    };
    let throughput = if elapsed > 0.0 {
        (element_count as f64 * timesteps as f64) / elapsed
    } else {
        0.0
    };

    BenchmarkResult {
        name: name.to_string(),
        element_count,
        node_count,
        wall_time_secs: elapsed,
        timesteps,
        time_per_step,
        throughput,
        rollbacks: stats.rollbacks,
        prediction_hit_rate: stats.hit_rate(),
    }
}

/// Run the full benchmark suite.
pub fn run_benchmark_suite() -> Vec<BenchmarkResult> {
    let config = SimConfig {
        dt: 1e-12,
        tstop: 100e-12,
        reltol: 1e-3,
        predict_depth: 3,
        max_history: 100,
    };

    let mut results = Vec::new();

    info!("Running benchmark: RC Ladder (10 stages)");
    results.push(run_benchmark(
        "RC Ladder (10)",
        generate_rc_ladder(10, 1e3, 1e-12),
        config.clone(),
    ));

    info!("Running benchmark: RC Ladder (100 stages)");
    results.push(run_benchmark(
        "RC Ladder (100)",
        generate_rc_ladder(100, 1e3, 1e-12),
        config.clone(),
    ));

    info!("Running benchmark: CiM Array (4x4)");
    results.push(run_benchmark(
        "CiM 4×4",
        generate_cim_array(4, 4),
        config.clone(),
    ));

    info!("Running benchmark: CiM Array (8x8)");
    results.push(run_benchmark(
        "CiM 8×8",
        generate_cim_array(8, 8),
        config.clone(),
    ));

    info!("Running benchmark: Resistor Mesh (4x4)");
    results.push(run_benchmark(
        "Mesh 4×4",
        generate_resistor_mesh(4, 1e3),
        config.clone(),
    ));

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cim_array_generation() {
        let elements = generate_cim_array(4, 4);
        // 4 voltage sources + 16 resistors + 4 caps = 24
        assert_eq!(elements.len(), 24, "4x4 CiM should have 24 elements");
    }

    #[test]
    fn test_rc_ladder_generation() {
        let elements = generate_rc_ladder(10, 1e3, 1e-12);
        // 1 vsource + 10*(R+C) = 21
        assert_eq!(
            elements.len(),
            21,
            "10-stage RC ladder should have 21 elements"
        );
    }

    #[test]
    fn test_resistor_mesh_generation() {
        let elements = generate_resistor_mesh(3, 1e3);
        // 1 vsource + 1 isource + 6 horizontal + 6 vertical = 14
        assert_eq!(elements.len(), 14, "3x3 mesh should have 14 elements");
    }

    #[test]
    fn test_benchmark_rc_ladder() {
        let config = SimConfig {
            dt: 1e-12,
            tstop: 10e-12,
            reltol: 1e-3,
            predict_depth: 3,
            max_history: 100,
        };

        let result = run_benchmark("Test RC", generate_rc_ladder(5, 1e3, 1e-12), config);

        assert!(result.timesteps > 0, "Should simulate some steps");
        assert!(result.wall_time_secs > 0.0, "Should take some time");
        assert!(result.throughput > 0.0, "Should have nonzero throughput");
    }

    #[test]
    fn test_benchmark_cim_array() {
        let config = SimConfig {
            dt: 1e-12,
            tstop: 10e-12,
            reltol: 1e-3,
            predict_depth: 3,
            max_history: 100,
        };

        let result = run_benchmark("Test CiM", generate_cim_array(3, 3), config);

        assert!(result.timesteps > 0);
        assert!(result.element_count > 0);
        let summary = result.summary();
        assert!(summary.contains("CiM"));
    }
}
