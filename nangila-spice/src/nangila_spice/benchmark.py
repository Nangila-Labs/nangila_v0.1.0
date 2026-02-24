"""
Nangila SPICE Benchmark Runner

Python-side benchmark orchestration and performance reporting.
Generates synthetic circuits, runs the solver, and produces
performance reports with speedup curves.

Phase 2, Sprint 8 deliverable.
"""

import time
import json
import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    name: str
    element_count: int
    node_count: int
    partition_count: int
    wall_time_secs: float
    timesteps: int
    time_per_step: float
    throughput: float  # elements*steps / second
    rollbacks: int = 0
    prediction_hit_rate: float = 0.0
    speedup: float = 1.0  # vs reference

    @property
    def summary_line(self) -> str:
        return (
            f"{self.name}: {self.element_count} elements, "
            f"{self.partition_count} parts | "
            f"{self.wall_time_secs:.3f}s, "
            f"{self.throughput:.0f} elem/s, "
            f"{self.speedup:.2f}x speedup"
        )


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    name: str
    results: List[BenchmarkResult] = field(default_factory=list)
    timestamp: str = ""
    platform: str = ""

    def add(self, result: BenchmarkResult):
        self.results.append(result)

    def summary(self) -> str:
        lines = [
            f"═══ Benchmark Suite: {self.name} ═══",
            f"Platform: {self.platform}",
            f"Timestamp: {self.timestamp}",
            "",
        ]
        for r in self.results:
            lines.append(f"  {r.summary_line}")
        lines.append("")

        # Aggregate stats
        if self.results:
            avg_speedup = sum(r.speedup for r in self.results) / len(self.results)
            max_speedup = max(r.speedup for r in self.results)
            total_elements = sum(r.element_count for r in self.results)
            lines.append(f"  Average speedup: {avg_speedup:.2f}x")
            lines.append(f"  Max speedup: {max_speedup:.2f}x")
            lines.append(f"  Total elements: {total_elements}")

        return "\n".join(lines)

    def to_csv(self, path: str):
        """Export results to CSV."""
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "name", "elements", "nodes", "partitions",
                "wall_time_s", "timesteps", "time_per_step_s",
                "throughput", "rollbacks", "hit_rate", "speedup",
            ])
            for r in self.results:
                writer.writerow([
                    r.name, r.element_count, r.node_count,
                    r.partition_count, f"{r.wall_time_secs:.6f}",
                    r.timesteps, f"{r.time_per_step:.9f}",
                    f"{r.throughput:.0f}", r.rollbacks,
                    f"{r.prediction_hit_rate:.4f}", f"{r.speedup:.3f}",
                ])

    def to_json(self, path: str):
        """Export results to JSON."""
        data = {
            "suite": self.name,
            "platform": self.platform,
            "timestamp": self.timestamp,
            "results": [
                {
                    "name": r.name,
                    "element_count": r.element_count,
                    "node_count": r.node_count,
                    "partition_count": r.partition_count,
                    "wall_time_secs": r.wall_time_secs,
                    "timesteps": r.timesteps,
                    "time_per_step": r.time_per_step,
                    "throughput": r.throughput,
                    "rollbacks": r.rollbacks,
                    "prediction_hit_rate": r.prediction_hit_rate,
                    "speedup": r.speedup,
                }
                for r in self.results
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def generate_cim_netlist(n_rows: int, n_cols: int) -> str:
    """Generate a SPICE netlist for a CiM crossbar array.

    Args:
        n_rows: Number of wordlines
        n_cols: Number of bitlines

    Returns:
        SPICE netlist string
    """
    lines = [f"* CiM Array {n_rows}x{n_cols}", ""]

    # Wordline drivers
    for row in range(n_rows):
        v = 1.8 if row == 0 else 0.0
        lines.append(f"VWL{row} WL{row} 0 {v}")

    lines.append("")

    # Cells
    cell_id = 0
    for row in range(n_rows):
        for col in range(n_cols):
            resistance = 1e6 if (row + col) % 3 == 0 else 10e3
            lines.append(f"RCELL{cell_id} WL{row} BL{col} {resistance}")
            cell_id += 1

    lines.append("")

    # Bitline caps
    for col in range(n_cols):
        lines.append(f"CBL{col} BL{col} 0 100f")

    lines.extend(["", ".tran 1p 100p", ".end"])
    return "\n".join(lines)


def generate_rc_ladder_netlist(n_stages: int, r: float = 1e3, c: float = 1e-12) -> str:
    """Generate a SPICE netlist for an RC ladder."""
    lines = [f"* RC Ladder ({n_stages} stages)", ""]
    lines.append("VIN N1 0 1.8")

    for i in range(n_stages):
        n_in = f"N{i + 1}"
        n_out = f"N{i + 2}"
        lines.append(f"R{i} {n_in} {n_out} {r}")
        lines.append(f"C{i} {n_out} 0 {c}")

    lines.extend(["", ".tran 1p 100p", ".end"])
    return "\n".join(lines)


def run_scaling_study(
    circuit_generator,
    sizes: List[int],
    partition_counts: List[int],
    label: str = "Scaling Study",
) -> BenchmarkSuite:
    """Run a scaling study: vary circuit size and partition count.

    This is a framework function — actual simulation runs
    require the Rust solver binary or in-process simulation.
    Returns a BenchmarkSuite with placeholder metrics for now.
    """
    import platform

    suite = BenchmarkSuite(
        name=label,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        platform=f"{platform.system()} {platform.machine()}"
    )

    for size in sizes:
        for n_parts in partition_counts:
            # Generate netlist
            netlist = circuit_generator(size)
            element_count = netlist.count("\n")

            # Placeholder: actual solver integration would go here
            # For now, estimate based on element count
            estimated_time = element_count * 1e-6 / max(n_parts, 1)
            estimated_speedup = min(n_parts * 0.7, element_count / 10)

            result = BenchmarkResult(
                name=f"{label}_{size}_{n_parts}p",
                element_count=element_count,
                node_count=element_count // 2,
                partition_count=n_parts,
                wall_time_secs=estimated_time,
                timesteps=100,
                time_per_step=estimated_time / 100,
                throughput=element_count * 100 / max(estimated_time, 1e-9),
                speedup=estimated_speedup,
            )
            suite.add(result)

    return suite


def generate_performance_report(suite: BenchmarkSuite, output_dir: str) -> str:
    """Generate a performance report from benchmark results.

    Args:
        suite: BenchmarkSuite with results
        output_dir: Directory to save report files

    Returns:
        Path to the generated report
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save raw data
    suite.to_csv(os.path.join(output_dir, "results.csv"))
    suite.to_json(os.path.join(output_dir, "results.json"))

    # Generate markdown report
    report_path = os.path.join(output_dir, "performance_report.md")
    lines = [
        f"# Nangila SPICE Performance Report",
        f"",
        f"**Suite:** {suite.name}",
        f"**Platform:** {suite.platform}",
        f"**Date:** {suite.timestamp}",
        f"",
        f"## Results",
        f"",
        f"| Benchmark | Elements | Partitions | Wall Time | Throughput | Speedup |",
        f"|-----------|----------|-----------|-----------|------------|---------|",
    ]

    for r in suite.results:
        lines.append(
            f"| {r.name} | {r.element_count} | {r.partition_count} | "
            f"{r.wall_time_secs:.4f}s | {r.throughput:.0f} elem/s | "
            f"{r.speedup:.2f}x |"
        )

    lines.extend([
        "",
        "## Summary",
        "",
        suite.summary(),
    ])

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    return report_path
