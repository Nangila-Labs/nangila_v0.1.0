//! Nangila SPICE Solver Node
//!
//! A single solver process responsible for simulating one partition
//! of a circuit netlist using the Predict-Verify architecture.
//!
//! Phase 1, Sprint 3: Full solver loop operational with built-in MNA solver.

use clap::Parser;
use tracing::info;

use nangila_node::comm::CommLayer;
use nangila_node::ghost::GhostBuffer;
use nangila_node::mna::Element;
use nangila_node::ngspice_ffi::PartitionNetlist;
use nangila_node::parser::SpiceParser;
use nangila_node::solver::{SimConfig, TransientSolver};


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
    } else if let Some(partition_path) = args.partition.clone() {
        let netlist_res = if partition_path.ends_with(".json") {
            info!("Loading partition config from JSON: {}", partition_path);
            std::fs::File::open(&partition_path)
                .map_err(|e| e.to_string())
                .and_then(|file| serde_json::from_reader(file).map_err(|e| e.to_string()))
        } else {
            info!("Parsing SPICE netlist from: {}", partition_path);
            SpiceParser::parse_file(&partition_path).map_err(|e| e.to_string())
        };

        match netlist_res {
            Ok(netlist) => {
                info!("Loaded netlist '{}': {} nodes, {} elements",
                    netlist.name, netlist.num_nodes, netlist.elements.len());
                run_from_netlist(netlist, args);
            }
            Err(e) => {
                eprintln!("ERROR: Failed to load netlist '{}': {}", partition_path, e);
                std::process::exit(1);
            }
        }
    } else {
        info!("No partition specified. Use --partition <path.sp> or --demo for a test run.");
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

    let mut ghosts = GhostBuffer::new();
    for (net_id, _local_node, owner) in &netlist.ghost_map {
        ghosts.add_ghost(nangila_node::ghost::GhostNode::new(*net_id, *owner));
    }

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
    info!("Prediction hit rate: {:.1}%", stats.hit_rate() * 100.0);
    info!("Rollbacks: {}", stats.rollbacks);
    info!("Waveform points: {}", solver.waveform.len());

    // Print a few waveform samples
    let wf = &solver.waveform;
    if wf.len() >= 5 {
        info!("=== Waveform Samples (V_cap) ===");
        let indices = [
            0,
            wf.len() / 4,
            wf.len() / 2,
            3 * wf.len() / 4,
            wf.len() - 1,
        ];
        for &i in &indices {
            let (t, ref v) = wf[i];
            info!("  t={:.2e}s  V(cap)={:.6}V", t, v[1]);
        }
    }

    info!("Nangila-Node {} finished.", args.node_id);
}

/// Run the transient solver on a parsed SPICE netlist.
fn run_from_netlist(netlist: PartitionNetlist, args: Args) {
    info!(
        "Running solver on netlist '{}': {} nodes, {} elements",
        netlist.name,
        netlist.num_nodes,
        netlist.elements.len()
    );

    let mut ghosts = GhostBuffer::new();
    for (net_id, _local_node, owner) in &netlist.ghost_map {
        ghosts.add_ghost(nangila_node::ghost::GhostNode::new(*net_id, *owner));
    }

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

    info!("=== Simulation Results ===");
    info!("Final node voltages:");
    for (i, v) in final_state.voltages.iter().enumerate() {
        info!("  V(node {}) = {:.6}V", i + 1, v);
    }
    info!("Timesteps: {}", stats.total_steps);
    info!("Wall time:  {:.3}s", stats.wall_time_secs);
    info!("Prediction hit rate: {:.1}%", stats.hit_rate() * 100.0);
    info!("Rollbacks: {}", stats.rollbacks);

    // Print a few waveform samples from the first internal node (if any)
    let wf = &solver.waveform;
    if wf.len() >= 5 {
        info!("=== Waveform Samples (node 1) ===");
        let indices = [0, wf.len() / 4, wf.len() / 2, 3 * wf.len() / 4, wf.len() - 1];
        for &i in &indices {
            let (t, ref v) = wf[i];
            if !v.is_empty() {
                info!("  t={:.2e}s  V(1)={:.6}V", t, v[0]);
            }
        }
    }

    info!("Nangila-Node {} finished.", args.node_id);
}
