#!/usr/bin/env python3
"""
nangila-run — Nangila SPICE Command-Line Interface

End-to-end simulation entry point.

Usage:
    python -m nangila_spice.cli simulate netlist.sp
    python -m nangila_spice.cli simulate netlist.sp --partitions 4
    python -m nangila_spice.cli info netlist.sp
"""

import argparse
import sys

from .orchestrator import SimulationConfig, run_simulation, discover_hardware
from .parser import parse_netlist
from .partitioner import partition_netlist
from .graph import build_circuit_graph
from .pvt_orchestrator import SweepConfig, PvtOrchestrator, generate_1000_corner_grid


def main():
    parser = argparse.ArgumentParser(
        prog="nangila-run",
        description="Nangila SPICE — Predictive-Partitioned Circuit Simulator",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --- simulate ---
    sim_parser = subparsers.add_parser(
        "simulate", aliases=["sim"],
        help="Run a transient simulation",
    )
    sim_parser.add_argument("netlist", help="Path to .sp or .cir netlist file")
    sim_parser.add_argument(
        "-k", "--partitions", type=int, default=None,
        help="Number of partitions (auto if not set)",
    )
    sim_parser.add_argument(
        "--method", default="auto", choices=["auto", "metis", "spectral"],
        help="Partitioning method",
    )
    sim_parser.add_argument(
        "--reltol", type=float, default=1e-3,
        help="Relative tolerance (default: 1e-3)",
    )
    sim_parser.add_argument(
        "--tstop", type=float, default=1e-6,
        help="Simulation end time in seconds (default: 1e-6)",
    )
    sim_parser.add_argument(
        "--dt", type=float, default=1e-12,
        help="Time step in seconds (default: 1e-12)",
    )
    sim_parser.add_argument(
        "--predict-depth", type=int, default=5,
        help="Max speculative steps ahead (default: 5)",
    )
    sim_parser.add_argument(
        "-o", "--output", default=None,
        help="Output directory (default: temp)",
    )
    sim_parser.add_argument(
        "--format", default="csv", choices=["csv", "json"],
        help="Output waveform format (default: csv)",
    )

    # --- info ---
    info_parser = subparsers.add_parser(
        "info",
        help="Show netlist info without running simulation",
    )
    info_parser.add_argument("netlist", help="Path to .sp or .cir netlist file")
    info_parser.add_argument(
        "-k", "--partitions", type=int, default=2,
        help="Number of partitions to preview",
    )

    # --- hardware ---
    subparsers.add_parser(
        "hardware", aliases=["hw"],
        help="Show detected hardware",
    )

    # --- pvt ---
    pvt_parser = subparsers.add_parser(
        "pvt",
        help="Run multi-corner PVT sweep (1000 corners)",
    )
    pvt_parser.add_argument("netlist", help="Path to .sp or .cir netlist file")
    pvt_parser.add_argument(
        "--workers", type=int, default=16,
        help="Number of parallel workers (default: 16)",
    )
    pvt_parser.add_argument(
        "--no-delta", action="store_true",
        help="Disable delta-mode approximations (force full sim for all corners)",
    )

    args = parser.parse_args()

    if args.command in ("simulate", "sim"):
        cmd_simulate(args)
    elif args.command == "info":
        cmd_info(args)
    elif args.command in ("hardware", "hw"):
        cmd_hardware()
    elif args.command == "pvt":
        cmd_pvt(args)
    else:
        parser.print_help()
        sys.exit(1)


def cmd_simulate(args):
    """Run a full simulation."""
    config = SimulationConfig(
        netlist_path=args.netlist,
        partitions=args.partitions,
        method=args.method,
        reltol=args.reltol,
        tstop=args.tstop,
        dt=args.dt,
        predict_depth=args.predict_depth,
        output_dir=args.output,
        output_format=args.format,
    )

    result = run_simulation(config)

    if not result.success:
        print(f"\n[ERROR] Simulation failed: {result.error}")
        sys.exit(1)

    print(f"\n[nangila-run] Output: {args.output or 'temp directory'}")


def cmd_info(args):
    """Show netlist information."""
    print(f"[nangila-run] Parsing {args.netlist}...")
    nl = parse_netlist(args.netlist)

    print(f"\n=== Netlist: {nl.title} ===")
    print(f"  Devices: {nl.num_devices}")
    print(f"  Nodes:   {nl.num_nodes}")

    # Device breakdown
    type_counts: dict[str, int] = {}
    for d in nl.devices:
        type_counts[d.dev_type] = type_counts.get(d.dev_type, 0) + 1

    type_names = {
        "R": "Resistors", "C": "Capacitors", "L": "Inductors",
        "M": "MOSFETs", "V": "V Sources", "I": "I Sources",
        "X": "Subcircuits",
    }
    for t, count in sorted(type_counts.items()):
        name = type_names.get(t, t)
        print(f"    {name}: {count}")

    # Subcircuits
    if nl.subcircuits:
        print(f"\n  Subcircuits: {len(nl.subcircuits)}")
        for name, sc in nl.subcircuits.items():
            print(f"    {name}: {len(sc.ports)} ports, {len(sc.devices)} devices")

    # Graph info
    cg = build_circuit_graph(nl)
    print(f"\n  Circuit Graph:")
    print(f"    Device nodes: {cg.num_devices}")
    print(f"    Net nodes:    {cg.num_nets}")
    print(f"    Edges:        {cg.num_edges}")
    print(f"    Feedback groups: {len(cg.feedback_groups)}")

    # Partition preview
    k = args.partitions
    result = partition_netlist(nl, k, method="spectral")
    print(f"\n  Partition Preview (k={k}):")
    print(f"  {result.summary()}")


def cmd_hardware():
    """Show hardware info."""
    hw = discover_hardware()
    print(f"\n=== Hardware Topology ===")
    print(f"  {hw.summary()}")
    print(f"  Recommended partitions: {max(1, hw.cpu_count // 2)}")


def cmd_pvt(args):
    """Run a multi-corner PVT sweep."""
    import os
    print(f"=== Starting 1000-Corner PVT Sweep ===")
    print(f"Netlist: {args.netlist}")
    config = SweepConfig(
        max_workers=args.workers,
        delta_mode=not args.no_delta,
        output_dir="/tmp/nangila_pvt",
    )
    
    # 1. Generate corners
    corners = generate_1000_corner_grid()
    
    # 2. Add the netlist path to the golden corner call directly inside Orchestrator
    # We'll monkeypatch or we can just pass the path. Wait, orchestrator simulates the nominal corner using simulate_corner.
    # Actually, PvtOrchestrator currently calls simulate_corner without netlist_path. Let's fix that.
    
    # Wait, we need to pass netlist_path to run_sweep
    orchestrator = PvtOrchestrator(config)
    orchestrator.netlist_path = args.netlist  # We'll patch PvtOrchestrator to use this
    
    res = orchestrator.run_sweep(corners, netlist_path=args.netlist)
    print("\nSweep completed natively.")
    
if __name__ == "__main__":
    main()
