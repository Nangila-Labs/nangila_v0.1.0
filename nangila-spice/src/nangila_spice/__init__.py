from __future__ import annotations

"""
Nangila SPICE — Predictive-Partitioned Circuit Simulator

Python frontend for the Nangila SPICE solver.

Quick start:
    from nangila_spice import simulate, parse_netlist, partition_netlist

    nl = parse_netlist("circuit.sp")
    result = simulate("circuit.sp", partitions=4)
    print(result.waveform.summary())
"""

__version__ = "0.2.0"

from importlib import import_module
from typing import Any


_EXPORT_MAP = {
    "parse_netlist": (".parser", "parse_netlist"),
    "Netlist": (".parser", "Netlist"),
    "Device": (".parser", "Device"),
    "Subcircuit": (".parser", "Subcircuit"),
    "build_circuit_graph": (".graph", "build_circuit_graph"),
    "CircuitGraph": (".graph", "CircuitGraph"),
    "partition_netlist": (".partitioner", "partition_netlist"),
    "Partition": (".partitioner", "Partition"),
    "PartitionResult": (".partitioner", "PartitionResult"),
    "Waveform": (".merger", "Waveform"),
    "WaveformPoint": (".merger", "WaveformPoint"),
    "merge_waveforms": (".merger", "merge_waveforms"),
    "export_csv": (".merger", "export_csv"),
    "export_json": (".merger", "export_json"),
    "discover_hardware": (".orchestrator", "discover_hardware"),
    "HardwareTopology": (".orchestrator", "HardwareTopology"),
    "SimulationConfig": (".orchestrator", "SimulationConfig"),
    "SimulationResult": (".orchestrator", "SimulationResult"),
    "run_simulation": (".orchestrator", "run_simulation"),
    "PvtOrchestrator": (".pvt_orchestrator", "PvtOrchestrator"),
    "SweepConfig": (".pvt_orchestrator", "SweepConfig"),
    "SweepResult": (".pvt_orchestrator", "SweepResult"),
    "generate_corner_grid": (".pvt_orchestrator", "generate_corner_grid"),
    "simulate_corner": (".pvt_orchestrator", "simulate_corner"),
    "CornerSpec": (".pvt_orchestrator", "CornerSpec"),
    "ProcessCorner": (".pvt_orchestrator", "ProcessCorner"),
    "WaveformData": (".correctness", "WaveformData"),
    "WaveformComparison": (".correctness", "WaveformComparison"),
    "V1ToleranceProfile": (".correctness", "V1ToleranceProfile"),
    "compare_waveforms": (".correctness", "compare_waveforms"),
    "find_ngspice_binary": (".correctness", "find_ngspice_binary"),
    "find_nangila_binary": (".correctness", "find_nangila_binary"),
    "run_ngspice_waveform": (".correctness", "run_ngspice_waveform"),
    "run_nangila_waveform": (".correctness", "run_nangila_waveform"),
    "v1_tolerance_profile": (".correctness", "v1_tolerance_profile"),
    "within_v1_contract": (".correctness", "within_v1_contract"),
}



def simulate(
    netlist_path: str,
    partitions: int | None = None,
    reltol: float = 1e-3,
    tstop: float = 1e-6,
    dt: float = 1e-12,
    output_dir: str | None = None,
    output_format: str = "csv",
    validate_partitioned_against_reference: bool = True,
    reference_vdd: float = 1.8,
) -> SimulationResult:
    """
    Run a simulation on a SPICE netlist.

    This is the primary high-level API.

    Args:
        netlist_path: Path to .sp or .cir file.
        partitions: Number of partitions (auto if None).
        reltol: Relative tolerance.
        tstop: Simulation end time.
        dt: Time step.
        output_dir: Directory for output files.
        output_format: 'csv' or 'json'.

    Returns:
        SimulationResult with merged waveform and statistics.
    """
    SimulationConfig = __getattr__("SimulationConfig")
    run_simulation = __getattr__("run_simulation")
    config = SimulationConfig(
        netlist_path=netlist_path,
        partitions=partitions,
        reltol=reltol,
        tstop=tstop,
        dt=dt,
        output_dir=output_dir,
        output_format=output_format,
        validate_partitioned_against_reference=validate_partitioned_against_reference,
        reference_vdd=reference_vdd,
    )
    return run_simulation(config)


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_EXPORT_MAP.keys()))


__all__ = [
    # Core types
    "Netlist",
    "Device",
    "Subcircuit",
    "CircuitGraph",
    "Partition",
    "PartitionResult",
    "Waveform",
    "WaveformPoint",
    "HardwareTopology",
    "SimulationConfig",
    "SimulationResult",
    # Functions
    "parse_netlist",
    "build_circuit_graph",
    "partition_netlist",
    "merge_waveforms",
    "export_csv",
    "export_json",
    "discover_hardware",
    "run_simulation",
    "simulate",
    "WaveformData",
    "WaveformComparison",
    "V1ToleranceProfile",
    "compare_waveforms",
    "find_ngspice_binary",
    "find_nangila_binary",
    "run_ngspice_waveform",
    "run_nangila_waveform",
    "v1_tolerance_profile",
    "within_v1_contract",
]
