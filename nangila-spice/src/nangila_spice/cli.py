#!/usr/bin/env python3
"""
Nangila SPICE Command-Line Interface (V2)
"""

import argparse
import sys
import os
import subprocess

def get_project_root():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(this_dir, "..", ".."))

def invoke_build(args):
    root = get_project_root()
    node_dir = os.path.join(root, "nangila-node")
    if not os.path.exists(node_dir):
        print(f"Error: Could not locate nangila-node source at {node_dir}")
        sys.exit(1)
        
    cmd = ["cargo", "build", "--release"]
    if args.gpu:
        cmd.extend(["--features", "cuda"])
        
    print(f"Compiling Nangila Node from {node_dir}...")
    try:
        subprocess.run(cmd, cwd=node_dir, check=True)
        print("Build successful.")
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        sys.exit(e.returncode)

def invoke_run(args):
    from .orchestrator import SimulationConfig, run_simulation

    if not os.path.exists(args.netlist):
        print(f"Error: Netlist not found at {args.netlist}")
        sys.exit(1)

    result = run_simulation(
        SimulationConfig(
            netlist_path=args.netlist,
            partitions=args.partitions,
            method=args.method,
            reltol=args.reltol,
            tstop=args.tstop,
            dt=args.dt,
            predict_depth=args.predict_depth,
            output_dir=args.output_dir,
            output_format=args.output_format,
            validate_partitioned_against_reference=not args.skip_reference_validation,
            reference_vdd=args.vdd,
            verbose=args.verbose,
        )
    )
    if not result.success:
        sys.exit(1)

def invoke_synth(args):
    root = get_project_root()
    synth_script = os.path.join(root, "benchmarks", "synthesize_all.py")
    
    if not os.path.exists(synth_script):
        print(f"Error: Could not find synthesize_all.py at {synth_script}")
        sys.exit(1)
        
    print("Launching Yosys Synthesis Pipeline...")
    subprocess.run(["python3", synth_script])

def invoke_sweep(args):
    from .pvt_orchestrator import PvtOrchestrator, SweepConfig, generate_1000_corner_grid

    if not os.path.exists(args.netlist):
        print(f"Error: Netlist not found at {args.netlist}")
        sys.exit(1)
        
    print(f"Starting PVT Sweep on {args.netlist} with {args.corners} corners (delta_mode={not args.full_sim})...")
    cfg = SweepConfig(
        max_workers=8,
        delta_mode=not args.full_sim,
        tolerance_v=1e-3,
        save_waveforms=False,
    )
    orch = PvtOrchestrator(cfg)
    grid = generate_1000_corner_grid()
    
    if args.corners < len(grid):
        grid = grid[:args.corners]
        
    try:
        orch.run_sweep(corners=grid, netlist_path=args.netlist)
    except Exception as e:
        print(f"Sweep failed: {e}")
        sys.exit(1)

def invoke_phase1_report(args):
    from .phase1_report import main as phase1_report_main

    cmd = []
    if args.include_extended:
        cmd.append("--include-extended")
    cmd.extend(["--output-json", args.output_json, "--output-md", args.output_md])
    raise SystemExit(phase1_report_main(cmd))

def main():
    parser = argparse.ArgumentParser(
        description="Nangila SPICE CLI - High Performance Circuit Verification"
    )
    subparsers = parser.add_subparsers(title="commands", dest="command", required=True)
    
    build_parser = subparsers.add_parser("build", help="Compile underlying Rust/CUDA backends")
    build_parser.add_argument("--gpu", action="store_true", help="Build with NVIDIA cuSPARSE support")
    
    run_parser = subparsers.add_parser("run", help="Run a single transient simulation")
    run_parser.add_argument("netlist", type=str, help="Path to SPICE netlist (.sp/.scs)")
    run_parser.add_argument("--partitions", type=int, default=None, help="Number of partitions (default: auto)")
    run_parser.add_argument("--method", type=str, default="auto", help="Partitioning method")
    run_parser.add_argument("--reltol", type=float, default=1e-3, help="Relative tolerance")
    run_parser.add_argument("--tstop", type=float, default=1e-6, help="Simulation stop time")
    run_parser.add_argument("--dt", type=float, default=1e-12, help="Simulation timestep")
    run_parser.add_argument("--predict-depth", type=int, default=5, help="Speculative prediction depth")
    run_parser.add_argument("--output-dir", type=str, default=None, help="Directory for waveform outputs")
    run_parser.add_argument("--output-format", choices=("csv", "json"), default="csv", help="Waveform export format")
    run_parser.add_argument("--vdd", type=float, default=1.8, help="Reference VDD for validation comparisons")
    run_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show partitioning and solver-progress logs",
    )
    run_parser.add_argument(
        "--skip-reference-validation",
        action="store_true",
        help="Skip single-node reference validation for partitioned runs",
    )
    
    synth_parser = subparsers.add_parser("synth", help="Synthesize Verilog benchmarks to transistor SPICE")
    
    sweep_parser = subparsers.add_parser("sweep", help="Run a massive Monte-Carlo PVT sweep")
    sweep_parser.add_argument("netlist", type=str, help="Baseline netlist to perturb")
    sweep_parser.add_argument("--corners", type=int, default=1000, help="Number of PVT corners to sweep")
    sweep_parser.add_argument("--full-sim", action="store_true", help="Force full massively-parallel Newton-Raphson simulation sweeps (disables delta mode)")

    report_parser = subparsers.add_parser(
        "phase1-report",
        help="Run the Phase 1 ngspice-backed benchmark report",
    )
    report_parser.add_argument(
        "--include-extended",
        action="store_true",
        help="Include the extended correctness gate in the report",
    )
    report_parser.add_argument(
        "--output-json",
        type=str,
        default="artifacts/phase1_benchmark_report.json",
        help="Path for the machine-readable report",
    )
    report_parser.add_argument(
        "--output-md",
        type=str,
        default="artifacts/phase1_benchmark_report.md",
        help="Path for the Markdown summary report",
    )
    
    args = parser.parse_args()
    
    if args.command == "build":
        invoke_build(args)
    elif args.command == "run":
        invoke_run(args)
    elif args.command == "synth":
        invoke_synth(args)
    elif args.command == "sweep":
        invoke_sweep(args)
    elif args.command == "phase1-report":
        invoke_phase1_report(args)

if __name__ == "__main__":
    main()
