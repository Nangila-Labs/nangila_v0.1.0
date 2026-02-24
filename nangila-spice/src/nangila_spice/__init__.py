"""
Nangila SPICE — Predictive-Partitioned Circuit Simulator

Python frontend for the Nangila SPICE solver.

Quick start:
    from nangila_spice import simulate, parse_netlist, partition_netlist

    nl = parse_netlist("circuit.sp")
    result = simulate("circuit.sp", partitions=4)
    print(result.waveform.summary())
"""

__version__ = "0.1.0"

from .parser import parse_netlist, Netlist, Device, Subcircuit
from .graph import build_circuit_graph, CircuitGraph
from .partitioner import (
    partition_netlist,
    Partition,
    PartitionResult,
)
from .merger import (
    Waveform,
    WaveformPoint,
    merge_waveforms,
    export_csv,
    export_json,
)
from .orchestrator import (
    discover_hardware,
    HardwareTopology,
    SimulationConfig,
    SimulationResult,
    run_simulation,
)
from .pvt_orchestrator import (
    PvtOrchestrator,
    SweepConfig,
    SweepResult,
    generate_corner_grid,
    simulate_corner,
    CornerSpec,
    ProcessCorner,
)



def simulate(
    netlist_path: str,
    partitions: int | None = None,
    reltol: float = 1e-3,
    tstop: float = 1e-6,
    dt: float = 1e-12,
    output_dir: str | None = None,
    output_format: str = "csv",
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
    config = SimulationConfig(
        netlist_path=netlist_path,
        partitions=partitions,
        reltol=reltol,
        tstop=tstop,
        dt=dt,
        output_dir=output_dir,
        output_format=output_format,
    )
    return run_simulation(config)


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
]
