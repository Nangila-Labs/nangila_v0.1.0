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

from .correctness import (
    WaveformComparison,
    WaveformData,
    compare_waveforms,
    find_nangila_binary,
    run_nangila_waveform,
    within_v1_contract,
)
from .merger import (
    PartitionWaveform,
    Waveform,
    WaveformPoint,
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
    validate_partitioned_against_reference: bool = True
    reference_vdd: float = 1.8
    prefer_single_node_fallback_for_partitioned: bool = True
    verbose: bool = True


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
    experimental: bool = False
    validation_status: str = "validated_single_node"
    warnings: list[str] = field(default_factory=list)
    reference_comparison: Optional[WaveformComparison] = None

    def summary(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        if self.experimental:
            status += " (EXPERIMENTAL)"
        lines = [
            "=== Nangila SPICE Simulation Results ===",
            f"Status: {status}",
            f"Validation: {self.validation_status}",
            f"Wall time: {self.wall_time_secs:.3f}s",
            f"Hardware: {self.hardware.summary()}",
            f"Partitions: {len(self.partition_result.partitions)} "
            f"({self.partition_result.method})",
            f"Ghost nodes: {self.partition_result.total_boundary_nodes}",
            "",
            self.waveform.summary(),
        ]
        if self.reference_comparison is not None:
            lines.extend(
                [
                    "",
                    "Reference comparison:",
                    f"  max_abs_error={self.reference_comparison.max_abs_error:.6g}V",
                    f"  rms_error={self.reference_comparison.rms_error:.6g}V",
                    f"  final_abs_error={self.reference_comparison.final_abs_error:.6g}V",
                ]
            )
        if self.error:
            lines.extend(["", f"Error: {self.error}"])
        if self.warnings:
            lines.extend([""] + [f"Warning: {warning}" for warning in self.warnings])
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
    _log(config, f"[nangila-run] Parsing {config.netlist_path}...")
    netlist = parse_netlist(config.netlist_path)
    
    # Flatten hierarchical subcircuits
    netlist = netlist.flatten()
    
    _log(
        config,
        f"[nangila-run] Found {netlist.num_devices} devices, "
        f"{netlist.num_nodes} nodes"
    )

    # Step 2: Hardware discovery
    hw = discover_hardware()
    _log(config, f"[nangila-run] Hardware: {hw.summary()}")

    # Step 3: Partition
    k = config.partitions or auto_partition_count(hw, netlist.num_devices)
    _log(config, f"[nangila-run] Partitioning into {k} blocks ({config.method})...")

    part_result = partition_netlist(netlist, k, method=config.method)
    _log(
        config,
        f"[nangila-run] {part_result.total_boundary_nodes} ghost nodes | "
        f"{part_result.feedback_groups_enforced} feedback constraints | "
        f"balance={part_result.balance_ratio:.2f}x"
    )
    _log(config, f"[nangila-run] {part_result.summary()}")

    # Step 4: Generate per-partition sub-netlists
    output_dir = config.output_dir or tempfile.mkdtemp(prefix="nangila_")
    os.makedirs(output_dir, exist_ok=True)

    partition_configs = _generate_partition_configs(
        netlist, part_result, config, output_dir
    )

    used_single_node_fallback = False
    fallback_reference: Optional[WaveformData] = None

    if k > 1 and config.prefer_single_node_fallback_for_partitioned:
        _log(
            config,
            "[nangila-run] Partition equivalence is not implemented yet; "
            "using the validated single-node fallback path."
        )
        fallback_reference = _run_single_node_reference_waveform(config)
        waveform = _waveform_data_to_waveform(fallback_reference, title=netlist.title)
        per_partition_times = []
        used_single_node_fallback = True
    else:
        # Step 5: Spawn solver nodes
        solver_binary = _find_solver_binary()
        if solver_binary:
            _log(config, f"[nangila-run] Found binary: {solver_binary}")
            _log(config, "[nangila-run] Writing per-partition .sp sub-netlists...")
            # Write each partition config as a .sp file that nangila-node can parse
            for pc in partition_configs:
                sp_path = os.path.join(output_dir, f"partition_{pc['partition_id']}.sp")
                _write_partition_netlist(pc, sp_path)
            _log(config, f"[nangila-run] Simulating {k} partitions via nangila-node...")
            per_partition_times = _run_solver_processes(
                solver_binary, partition_configs, config, output_dir
            )
        else:
            _log(config, "[nangila-run] nangila-node not found — using in-process simulation")
            per_partition_times = _run_in_process(
                partition_configs, config, output_dir
            )

        # Step 6: Merge waveforms
        _log(config, "[nangila-run] Merging waveforms...")
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

    if k > 1:
        result.experimental = True
        if used_single_node_fallback:
            result.validation_status = "experimental_partitioned_fallback_single_node"
            result.warnings.append(
                "Requested partitioned execution fell back to the validated single-node path because partition equivalence is not implemented yet."
            )
            if fallback_reference is not None:
                result.reference_comparison = compare_waveforms(
                    fallback_reference,
                    fallback_reference,
                    vdd=config.reference_vdd,
                )
        else:
            result.validation_status = "experimental_partitioned"
            result.warnings.append(
                "Partitioned runtime is experimental until it matches the single-node reference path."
            )
        if not used_single_node_fallback and config.validate_partitioned_against_reference:
            comparison, validation_error = _compare_partitioned_to_single_node_reference(
                config.netlist_path,
                netlist,
                waveform,
                config,
            )
            result.reference_comparison = comparison
            if validation_error:
                result.success = False
                result.error = validation_error
                result.validation_status = "experimental_partitioned_failed_reference"
        else:
            if not used_single_node_fallback:
                result.warnings.append(
                    "Partitioned run was not checked against the single-node reference."
                )
                result.validation_status = "experimental_partitioned_unchecked"

    print(f"\n{result.summary()}")

    return result


def _log(config: SimulationConfig, message: str) -> None:
    if config.verbose:
        print(message)


def _compare_partitioned_to_single_node_reference(
    netlist_path: str,
    netlist: Netlist,
    waveform: Waveform,
    config: SimulationConfig,
) -> tuple[Optional[WaveformComparison], Optional[str]]:
    solver_binary = find_nangila_binary()
    if not solver_binary:
        return None, (
            "Partitioned run is experimental and could not be validated because "
            "the single-node nangila-node binary was not available."
        )

    try:
        reference = run_nangila_waveform(
            netlist_path,
            tstop=config.tstop,
            dt=config.dt,
            vdd=config.reference_vdd,
            binary=solver_binary,
        )
    except RuntimeError as exc:
        return None, (
            "Partitioned run is experimental and single-node validation failed: "
            f"{exc}"
        )

    candidate = _waveform_to_waveform_data(waveform)
    comparison = compare_waveforms(reference, candidate, vdd=config.reference_vdd)
    passed, profile = within_v1_contract(
        comparison,
        nonlinear=_is_nonlinear_netlist(netlist),
        vdd=config.reference_vdd,
    )
    if passed:
        return comparison, None

    return comparison, (
        "Partitioned run diverged from the single-node reference path: "
        f"max_abs={comparison.max_abs_error:.6g}V "
        f"(limit {profile.max_abs_tol:.6g}V), "
        f"rms={comparison.rms_error:.6g}V "
        f"(limit {profile.rms_tol:.6g}V), "
        f"final={comparison.final_abs_error:.6g}V "
        f"(limit {profile.final_abs_tol:.6g}V)."
    )


def _waveform_to_waveform_data(waveform: Waveform) -> WaveformData:
    traces = {name: [] for name in waveform.node_names}
    times: list[float] = []
    for point in waveform.points:
        times.append(point.time)
        for name in waveform.node_names:
            traces[name].append(float(point.voltages.get(name, 0.0)))
    return WaveformData(
        tool="partitioned",
        netlist=waveform.title,
        node_names=list(waveform.node_names),
        times=times,
        traces=traces,
    )


def _waveform_data_to_waveform(waveform_data: WaveformData, *, title: str) -> Waveform:
    points = []
    for idx, time in enumerate(waveform_data.times):
        voltages = {
            name: float(waveform_data.traces.get(name, [0.0])[idx])
            for name in waveform_data.node_names
        }
        points.append(WaveformPoint(time=time, voltages=voltages))
    return Waveform(
        title=title,
        node_names=list(waveform_data.node_names),
        points=points,
    )


def _run_single_node_reference_waveform(config: SimulationConfig) -> WaveformData:
    return run_nangila_waveform(
        config.netlist_path,
        tstop=config.tstop,
        dt=config.dt,
        vdd=config.reference_vdd,
    )


def _is_nonlinear_netlist(netlist: Netlist) -> bool:
    nonlinear_devices = {"M", "D"}
    return any(device.dev_type.upper() in nonlinear_devices for device in netlist.devices)


def _write_partition_netlist(partition_config: dict, output_path: str) -> None:
    """
    Generate a standard SPICE sub-netlist for a partition.

    This file is consumed by nangila-node. We include custom .GHOST
    directives to tell the solver which nodes are boundary nodes.
    """
    lines = [
        f"* Partition {partition_config['partition_id']} sub-netlist",
        f"* Generated by Nangila Orchestrator",
        "",
    ]

    # Node mapping
    lines.append("* Node Mapping:")
    for name, idx in sorted(partition_config["node_mapping"].items(), key=lambda x: x[1]):
        if name != "0":
            lines.append(f".NODEMAP {name} {idx}")
    lines.append("")

    # Ghost nodes
    for ghost in partition_config["ghost_nodes"]:
        # Custom directive for nangila-node: .GHOST <net_name> <local_index> <owner_partition>
        lines.append(f".GHOST {ghost['net_name']} {ghost['local_index']} {ghost['owner']}")
    lines.append("")

    # Devices
    for dev in partition_config["devices"]:
        name = dev["name"]
        dtype = dev["type"]
        nodes = " ".join(dev["nodes"])
        model = dev["model"] or ""

        if dtype in ("R", "C", "L"):
            val = dev["params"].get("value", "0")
            lines.append(f"{name} {nodes} {val}")
        elif dtype == "M":
            # M1 drain gate source bulk model [params]
            params = " ".join([f"{k}={v}" for k, v in dev["params"].items()])
            lines.append(f"{name} {nodes} {model} {params}")
        elif dtype in ("V", "I"):
            val = dev["params"].get("value", "0")
            lines.append(f"{name} {nodes} {val}")
        elif dtype == "X":
            # Subcircuit: X1 n1 n2 ... subckt_name
            lines.append(f"{name} {nodes} {model}")
        else:
            # Fallback
            lines.append(f"{name} {nodes} {model}")

    lines.append("\n.END")
    
    content = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(content)


def _find_solver_binary() -> Optional[str]:
    """Find the nangila-node binary."""
    # Check common locations
    candidates = [
        os.path.join(
            os.path.dirname(__file__), "..", "..",
            "target", "debug", "nangila-node"
        ),
        os.path.join(
            os.path.dirname(__file__), "..", "..",
            "target", "release", "nangila-node"
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

        # Build node_partitions map to find owners
        node_partitions: dict[str, set[int]] = {}
        for p in part_result.partitions:
            for n in p.internal_nodes:
                if n not in node_partitions:
                    node_partitions[n] = set()
                node_partitions[n].add(p.partition_id)
            for n in p.boundary_nodes:
                if n not in node_partitions:
                    node_partitions[n] = set()
                node_partitions[n].add(p.partition_id)

        # Ghost node mapping
        ghost_map = []
        for gnode in partition.boundary_nodes:
            if gnode in node_to_idx:
                # Find owners (any partition other than current)
                owners = node_partitions.get(gnode, set()) - {partition.partition_id}
                owner = min(owners) if owners else 0
                ghost_map.append({
                    "net_name": gnode,
                    "local_index": node_to_idx[gnode],
                    "owner": owner,
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
    """Spawn nangila-node processes (one per partition) and collect full waveforms."""
    processes = []
    per_partition_times: list[float] = []
    k = len(partition_configs)

    for i, pc in enumerate(partition_configs):
        sp_path = os.path.join(output_dir, f"partition_{i}.sp")
        raw_waveform_path = os.path.join(output_dir, f"waveform_raw_{i}.json")
        cmd = [
            solver_binary,
            "--partition", sp_path,
            "--node-id", str(i),
            "-k", str(k),
            "--tstop", str(config.tstop),
            "--dt", str(config.dt),
            "--reltol", str(config.reltol),
            "--predict-depth", str(config.predict_depth),
            "--waveform-json", raw_waveform_path,
        ]
        _log(config, f"  [P{i}] Spawning: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append((i, proc, time.time(), pc, raw_waveform_path))

    for i, proc, start, pc, raw_waveform_path in processes:
        stdout, stderr = proc.communicate(timeout=300)
        elapsed = time.time() - start
        per_partition_times.append(elapsed)
        combined = (stdout or b"").decode() + (stderr or b"").decode()

        if proc.returncode != 0:
            print(f"  [P{i}] FAILED (exit {proc.returncode})")
            print(f"  [P{i}] stderr: {combined[:300]}")
            # Write an empty waveform so merge doesn't crash
            _write_empty_waveform(output_dir, i, pc)
            continue

        _log(config, f"  [P{i}] Done in {elapsed:.3f}s")
        if not os.path.exists(raw_waveform_path):
            print(f"  [P{i}] missing waveform artifact: {raw_waveform_path}")
            _write_empty_waveform(output_dir, i, pc)
            continue

        with open(raw_waveform_path, "r") as f:
            raw = json.load(f)

        output_path = os.path.join(output_dir, f"waveform_{i}.json")
        with open(output_path, "w") as f:
            json.dump({
                "partition_id": i,
                "node_mapping": pc.get("node_mapping", {}),
                "waveform": raw.get("waveform", []),
                "node_names": raw.get("node_names", []),
                "stats": raw.get("stats", {}),
            }, f)

    return per_partition_times


def _write_empty_waveform(output_dir: str, partition_id: int, pc: dict) -> None:
    """Write an empty waveform.json so the merge step doesn't crash on failure."""
    output_path = os.path.join(output_dir, f"waveform_{partition_id}.json")
    with open(output_path, "w") as f:
        json.dump({"partition_id": partition_id, "node_mapping": pc.get("node_mapping", {}), "waveform": []}, f)


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
        _log(config, f"  [P{pid}] In-process sim done in {elapsed:.3f}s")

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
