#!/usr/bin/env python3
"""
Nangila SPICE Command-Line Interface (V2)
"""

import argparse
import sys
import os
import subprocess
from struct import calcsize

from .pvt_orchestrator import PvtOrchestrator, SweepConfig, generate_1000_corner_grid

def get_project_root():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(this_dir, "..", ".."))

def find_nangila_node_bin():
    root = get_project_root()
    debug_bin = os.path.join(root, "target", "debug", "nangila-node")
    release_bin = os.path.join(root, "target", "release", "nangila-node")
    ws_debug_bin = os.path.join(root, "..", "target", "debug", "nangila-node")
    ws_release_bin = os.path.join(root, "..", "target", "release", "nangila-node")

    for c in [release_bin, ws_release_bin, debug_bin, ws_debug_bin]:
        if os.path.exists(c) and os.access(c, os.X_OK):
            return c
    return None

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
    binary = find_nangila_node_bin()
    if not binary:
        print("Error: Could not find compiled nangila-node binary. Run `nangila build [--gpu]` first.")
        sys.exit(1)
        
    if not os.path.exists(args.netlist):
        print(f"Error: Netlist not found at {args.netlist}")
        sys.exit(1)
        
    cmd = [binary, "--partition", args.netlist]
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd)

def invoke_synth(args):
    root = get_project_root()
    synth_script = os.path.join(root, "benchmarks", "synthesize_all.py")
    
    if not os.path.exists(synth_script):
        print(f"Error: Could not find synthesize_all.py at {synth_script}")
        sys.exit(1)
        
    print("Launching Yosys Synthesis Pipeline...")
    subprocess.run(["python3", synth_script])

def invoke_sweep(args):
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

def main():
    parser = argparse.ArgumentParser(
        description="Nangila SPICE CLI - High Performance Circuit Verification"
    )
    subparsers = parser.add_subparsers(title="commands", dest="command", required=True)
    
    build_parser = subparsers.add_parser("build", help="Compile underlying Rust/CUDA backends")
    build_parser.add_argument("--gpu", action="store_true", help="Build with NVIDIA cuSPARSE support")
    
    run_parser = subparsers.add_parser("run", help="Run a single transient simulation")
    run_parser.add_argument("netlist", type=str, help="Path to SPICE netlist (.sp/.scs)")
    
    synth_parser = subparsers.add_parser("synth", help="Synthesize Verilog benchmarks to transistor SPICE")
    
    sweep_parser = subparsers.add_parser("sweep", help="Run a massive Monte-Carlo PVT sweep")
    sweep_parser.add_argument("netlist", type=str, help="Baseline netlist to perturb")
    sweep_parser.add_argument("--corners", type=int, default=1000, help="Number of PVT corners to sweep")
    sweep_parser.add_argument("--full-sim", action="store_true", help="Force full massively-parallel Newton-Raphson simulation sweeps (disables delta mode)")
    
    args = parser.parse_args()
    
    if args.command == "build":
        invoke_build(args)
    elif args.command == "run":
        invoke_run(args)
    elif args.command == "synth":
        invoke_synth(args)
    elif args.command == "sweep":
        invoke_sweep(args)

if __name__ == "__main__":
    main()
