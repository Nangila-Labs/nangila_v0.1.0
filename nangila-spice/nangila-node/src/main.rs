//! Nangila SPICE Solver Node
//!
//! A single solver process responsible for simulating one partition
//! of a circuit netlist using the Predict-Verify architecture.
//!
//! Phase 1, Sprint 3: Full solver loop operational with built-in MNA solver.

use clap::Parser;
use tracing::info;

mod balancer;
mod benchmark;
mod codec;
mod comm;
mod device_model;
mod fabric;
mod ghost;
mod gpu_solver;
mod mna;
mod newton;
mod ngspice_ffi;
mod nz;
mod predictor;
mod pvt;
mod solver;

use comm::CommLayer;
use ghost::GhostBuffer;
use mna::Element;
use ngspice_ffi::PartitionNetlist;
use solver::{SimConfig, TransientSolver};

/// Nangila SPICE Solver Node
#[derive(Parser, Debug)]
#[command(name = "nangila-node")]
#[command(about = "Predictive-Partitioned SPICE Solver Node")]
struct Args {
    /// Path to the partition sub-netlist (JSON format)
    #[arg(short, long)]
    partition: Option<String>,

    /// Node ID within the simulation cluster
    #[arg(short, long, default_value_t = 0)]
    node_id: u32,

    /// Total number of partitions
    #[arg(short = 'k', long, default_value_t = 1)]
    total_partitions: u32,

    /// Maximum speculative depth (steps ahead)
    #[arg(long, default_value_t = 5)]
    predict_depth: u32,

    /// Relative tolerance for accepting speculative steps
    #[arg(long, default_value_t = 1e-3)]
    reltol: f64,

    /// Simulation end time (seconds)
    #[arg(long, default_value_t = 1e-6)]
    tstop: f64,

    /// Time step (seconds)
    #[arg(long, default_value_t = 1e-12)]
    dt: f64,

    /// Run built-in RC demo circuit (for testing)
    #[arg(long, default_value_t = false)]
    demo: bool,
}

fn main() {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    info!(
        "Nangila-Node {} of {} starting | predict_depth: {} | reltol: {:.2e}",
        args.node_id, args.total_partitions, args.predict_depth, args.reltol
    );

    if args.demo {
        run_demo(args);
    } else if let Some(partition_path) = &args.partition {
        info!("Loading partition from: {}", partition_path);
        // TODO: Load partition netlist from JSON file
        info!("Partition loading not yet implemented. Use --demo for a test run.");
    } else {
        info!("No partition specified. Use --partition <path> or --demo for a test run.");
    }
}

/// Run a built-in RC demo circuit for testing the solver loop.
fn run_demo(args: Args) {
    info!("Running RC demo circuit...");

    // Build a simple RC circuit: V=1.8V → R=1kΩ → C=10fF → GND
    //
    //   Node 1 (VDD) ──R1(1kΩ)── Node 2 (cap) ── GND
    //       │                        │
    //    V1(1.8V)                  C1(10fF)
    //       │                        │
    //      GND                     GND
    let netlist = PartitionNetlist {
        name: "RC Demo".into(),
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
    let comm = CommLayer::new(args.node_id, args.total_partitions);
    let config = SimConfig {
        tstop: args.tstop,
        dt: args.dt,
        reltol: args.reltol,
        predict_depth: args.predict_depth,
        max_history: 100,
    };

    let mut solver = TransientSolver::new(netlist, ghosts, comm, config);
    let (final_state, stats) = solver.run();

    // Print results
    info!("=== Simulation Results ===");
    info!("Final voltages:");
    for (i, v) in final_state.voltages.iter().enumerate() {
        info!("  V(node {}) = {:.6}V", i + 1, v);
    }
    info!("Steps: {}", stats.total_steps);
    info!("Wall time: {:.3}s", stats.wall_time_secs);
    info!(
        "Prediction hit rate: {:.1}%",
        stats.hit_rate() * 100.0
    );
    info!("Rollbacks: {}", stats.rollbacks);
    info!("Waveform points: {}", solver.waveform.len());

    // Print a few waveform samples
    let wf = &solver.waveform;
    if wf.len() >= 5 {
        info!("=== Waveform Samples (V_cap) ===");
        let indices = [0, wf.len() / 4, wf.len() / 2, 3 * wf.len() / 4, wf.len() - 1];
        for &i in &indices {
            let (t, ref v) = wf[i];
            info!("  t={:.2e}s  V(cap)={:.6}V", t, v[1]);
        }
    }

    info!("Nangila-Node {} finished.", args.node_id);
}
