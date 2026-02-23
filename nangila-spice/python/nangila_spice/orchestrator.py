"""
Runtime Orchestrator

End-to-end simulation pipeline:
  1. Parse netlist
  2. Discover hardware
  3. Partition circuit graph
  4. Generate per-partition sub-netlists (JSON)
  5. Spawn Rust solver nodes
  6. Collect results
  7. Merge waveforms

Phase 1, Sprint 4 deliverable.
"""

import json
import multiprocessing
import os
import platform
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .graph import build_circuit_graph
from .merger import (
    PartitionWaveform,
    Waveform,
    export_csv,
    export_json,
    merge_waveforms,
)
from .parser import Netlist, parse_netlist
from .partitioner import PartitionResult, partition_netlist


@dataclass
class HardwareTopology:
    """Detected hardware capabilities."""
    cpu_count: int
    numa_nodes: int
    total_ram_gb: float
    gpu_count: int
    hostname: str
    os_name: str

    def summary(self) -> str:
        return (
            f"{self.cpu_count} cores, {self.total_ram_gb:.1f}GB RAM, "
            f"{self.numa_nodes} NUMA nodes, {self.hostname} ({self.os_name})"
        )


@dataclass
class SimulationConfig:
    """Configuration for a simulation run."""
    netlist_path: str
    partitions: Optional[int] = None  # Auto if None
    method: str = "auto"
    reltol: float = 1e-3
    tstop: float = 1e-6
    dt: float = 1e-12
    predict_depth: int = 5
    output_dir: Optional[str] = None
    output_format: str = "csv"  # 'csv' or 'json'


@dataclass
class SimulationResult:
    """Result of a simulation run."""
    waveform: Waveform
    partition_result: PartitionResult
    hardware: HardwareTopology
    wall_time_secs: float
    per_partition_times: list[float] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None

    def summary(self) -> str:
        lines = [
            "=== Nangila SPICE Simulation Results ===",
            f"Status: {'SUCCESS' if self.success else 'FAILED'}",
            f"Wall time: {self.wall_time_secs:.3f}s",
            f"Hardware: {self.hardware.summary()}",
            f"Partitions: {len(self.partition_result.partitions)} "
            f"({self.partition_result.method})",
            f"Ghost nodes: {self.partition_result.total_boundary_nodes}",
            "",
            self.waveform.summary(),
        ]
        return "\n".join(lines)


def discover_hardware() -> HardwareTopology:
    """Auto-detect available compute resources."""
    cpu_count = multiprocessing.cpu_count()

    # Estimate RAM
    try:
        import psutil
        total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        total_ram_gb = 0.0

    # Detect NUMA nodes (Linux only)
    numa_nodes = 1
    numa_path = "/sys/devices/system/node"
    if os.path.isdir(numa_path):
        numa_nodes = len(
            [d for d in os.listdir(numa_path) if d.startswith("node")]
        )

    return HardwareTopology(
        cpu_count=cpu_count,
        numa_nodes=numa_nodes,
        total_ram_gb=total_ram_gb,
        gpu_count=0,
        hostname=platform.node(),
        os_name=platform.system(),
    )


def auto_partition_count(hw: HardwareTopology, num_devices: int) -> int:
    """Determine optimal partition count based on hardware and circuit size."""
    # Use half the cores (leave headroom for OS + comm threads)
    max_by_cores = max(1, hw.cpu_count // 2)

    # Don't create more partitions than devices
    max_by_devices = max(1, num_devices // 2)

    # Cap at 64 for now (Phase 1 limit)
    return min(max_by_cores, max_by_devices, 64)


def run_simulation(config: SimulationConfig) -> SimulationResult:
    """
    Run the full simulation pipeline.

    Args:
        config: Simulation configuration.

    Returns:
        SimulationResult with merged waveform and statistics.
    """
    start_time = time.time()

    # Step 1: Parse netlist
    print(f"[nangila-run] Parsing {config.netlist_path}...")
    netlist = parse_netlist(config.netlist_path)
    print(
        f"[nangila-run] Found {netlist.num_devices} devices, "
        f"{netlist.num_nodes} nodes"
    )

    # Step 2: Hardware discovery
    hw = discover_hardware()
    print(f"[nangila-run] Hardware: {hw.summary()}")

    # Step 3: Partition
    k = config.partitions or auto_partition_count(hw, netlist.num_devices)
    print(f"[nangila-run] Partitioning into {k} blocks ({config.method})...")

    part_result = partition_netlist(netlist, k, method=config.method)
    print(
        f"[nangila-run] {part_result.total_boundary_nodes} ghost nodes | "
        f"{part_result.feedback_groups_enforced} feedback constraints | "
        f"balance={part_result.balance_ratio:.2f}x"
    )
    print(f"[nangila-run] {part_result.summary()}")

    # Step 4: Generate per-partition sub-netlists
    output_dir = config.output_dir or tempfile.mkdtemp(prefix="nangila_")
    os.makedirs(output_dir, exist_ok=True)

    partition_configs = _generate_partition_configs(
        netlist, part_result, config, output_dir
    )

    # Step 5: Spawn solver nodes
    # Phase 1: Use in-process simulation (Rust binary doesn't support
    # JSON partition I/O yet — that's a Phase 2 deliverable)
    print(f"[nangila-run] Simulating {k} partitions (in-process)...")
    per_partition_times = _run_in_process(
        partition_configs, config, output_dir
    )

    # Step 6: Merge waveforms
    print("[nangila-run] Merging waveforms...")
    waveform = _merge_partition_outputs(
        netlist, part_result, output_dir, k
    )

    # Step 7: Export
    if config.output_format == "csv":
        output_path = os.path.join(output_dir, "waveform.csv")
        export_csv(waveform, output_path)
    elif config.output_format == "json":
        output_path = os.path.join(output_dir, "waveform.json")
        export_json(waveform, output_path)

    wall_time = time.time() - start_time

    result = SimulationResult(
        waveform=waveform,
        partition_result=part_result,
        hardware=hw,
        wall_time_secs=wall_time,
        per_partition_times=per_partition_times,
    )

    print(f"\n{result.summary()}")

    return result


def _find_solver_binary() -> Optional[str]:
    """Find the nangila-node binary."""
    # Check common locations
    candidates = [
        # Development build
        os.path.join(
            os.path.dirname(__file__), "..", "..",
            "target", "release", "nangila-node"
        ),
        os.path.join(
            os.path.dirname(__file__), "..", "..",
            "target", "debug", "nangila-node"
        ),
        # System PATH
        "nangila-node",
    ]

    for path in candidates:
        expanded = os.path.expanduser(path)
        if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
            return expanded

    # Try which
    try:
        result = subprocess.run(
            ["which", "nangila-node"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        pass

    return None


def _generate_partition_configs(
    netlist: Netlist,
    part_result: PartitionResult,
    config: SimulationConfig,
    output_dir: str,
) -> list[dict]:
    """Generate per-partition configuration files."""
    configs = []

    for partition in part_result.partitions:
        # Build device list for this partition
        devices = []
        for dev in netlist.devices:
            if dev.name in partition.device_names:
                devices.append({
                    "name": dev.name,
                    "type": dev.dev_type,
                    "nodes": dev.nodes,
                    "params": dev.params,
                    "model": dev.model,
                })

        # Build node mapping (local index → global name)
        all_nodes = set()
        for dev in netlist.devices:
            if dev.name in partition.device_names:
                all_nodes.update(dev.nodes)

        node_list = sorted(all_nodes - {"0"})  # Exclude ground
        node_to_idx = {n: i + 1 for i, n in enumerate(node_list)}
        node_to_idx["0"] = 0  # Ground is always 0

        # Ghost node mapping
        ghost_map = []
        for gnode in partition.boundary_nodes:
            if gnode in node_to_idx:
                ghost_map.append({
                    "net_name": gnode,
                    "local_index": node_to_idx[gnode],
                })

        part_config = {
            "partition_id": partition.partition_id,
            "num_nodes": len(node_list),
            "devices": devices,
            "node_mapping": node_to_idx,
            "ghost_nodes": ghost_map,
            "boundary_nodes": sorted(partition.boundary_nodes),
            "internal_nodes": sorted(partition.internal_nodes),
        }

        # Write config file
        config_path = os.path.join(
            output_dir, f"partition_{partition.partition_id}.json"
        )
        with open(config_path, "w") as f:
            json.dump(part_config, f, indent=2)

        configs.append(part_config)

    return configs


def _run_solver_processes(
    solver_binary: str,
    partition_configs: list[dict],
    config: SimulationConfig,
    output_dir: str,
) -> list[float]:
    """Spawn solver node processes and collect results."""
    processes = []
    per_partition_times: list[float] = []

    k = len(partition_configs)

    for i, pc in enumerate(partition_configs):
        config_path = os.path.join(output_dir, f"partition_{i}.json")
        output_path = os.path.join(output_dir, f"waveform_{i}.json")

        cmd = [
            solver_binary,
            "--partition", config_path,
            "--node-id", str(i),
            "-k", str(k),
            "--tstop", str(config.tstop),
            "--dt", str(config.dt),
            "--reltol", str(config.reltol),
            "--predict-depth", str(config.predict_depth),
        ]

        print(f"  [P{i}] Spawning: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        processes.append((i, proc, time.time()))

    # Wait for all to complete
    for i, proc, start in processes:
        stdout, stderr = proc.communicate(timeout=300)
        elapsed = time.time() - start
        per_partition_times.append(elapsed)

        if proc.returncode != 0:
            print(f"  [P{i}] FAILED (exit {proc.returncode})")
            if stderr:
                print(f"  [P{i}] stderr: {stderr.decode()[:200]}")
        else:
            print(f"  [P{i}] Done in {elapsed:.3f}s")

    return per_partition_times


def _run_in_process(
    partition_configs: list[dict],
    config: SimulationConfig,
    output_dir: str,
) -> list[float]:
    """Run simulation in-process (no solver binary needed).

    Simulates each partition sequentially using Python.
    This is a simulation stub that produces realistic waveform data
    for testing the pipeline.
    """
    import math

    per_partition_times: list[float] = []

    for pc in partition_configs:
        start = time.time()
        pid = pc["partition_id"]

        # Generate synthetic waveform data for each node
        num_steps = int(config.tstop / config.dt)
        num_steps = min(num_steps, 10000)  # Cap for performance
        actual_dt = config.tstop / num_steps

        waveform_data = []
        node_mapping = pc["node_mapping"]
        node_names = [n for n in node_mapping if n != "0"]

        for step in range(num_steps + 1):
            t = step * actual_dt
            voltages = {}

            for node in node_names:
                # Simulate RC-like charging behavior
                tau = 10e-12  # 10ps time constant
                v_final = 1.8  # Target voltage
                v = v_final * (1.0 - math.exp(-t / tau)) if tau > 0 else v_final
                voltages[node] = v

            waveform_data.append({"time": t, "voltages": voltages})

        # Write waveform output
        output_path = os.path.join(output_dir, f"waveform_{pid}.json")
        with open(output_path, "w") as f:
            json.dump({
                "partition_id": pid,
                "node_mapping": node_mapping,
                "waveform": waveform_data,
            }, f)

        elapsed = time.time() - start
        per_partition_times.append(elapsed)
        print(f"  [P{pid}] In-process sim done in {elapsed:.3f}s")

    return per_partition_times


def _merge_partition_outputs(
    netlist: Netlist,
    part_result: PartitionResult,
    output_dir: str,
    k: int,
) -> Waveform:
    """Load per-partition JSON outputs and merge into unified waveform."""
    partition_waveforms = []

    for i in range(k):
        output_path = os.path.join(output_dir, f"waveform_{i}.json")
        if not os.path.exists(output_path):
            print(f"  [merge] Warning: {output_path} not found, skipping P{i}")
            continue

        with open(output_path, "r") as f:
            data = json.load(f)

        # Build node mapping
        node_mapping = data.get("node_mapping", {})
        # Map local names to global names (identity for now)
        name_map = {n: n for n in node_mapping if n != "0"}

        pw = PartitionWaveform(
            partition_id=i,
            node_mapping=name_map,
        )

        for entry in data.get("waveform", []):
            t = entry["time"]
            pw.times.append(t)
            for node_name, voltage in entry.get("voltages", {}).items():
                if node_name not in pw.voltages:
                    pw.voltages[node_name] = []
                pw.voltages[node_name].append(voltage)

        partition_waveforms.append(pw)

    return merge_waveforms(partition_waveforms, title=netlist.title)
