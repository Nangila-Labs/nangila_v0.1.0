"""
Waveform Merger

Stitches per-partition waveform outputs into a unified result.
Handles:
  - Time alignment across partitions
  - Node name mapping (local → global)
  - Boundary node reconciliation
  - CSV and JSON export

Phase 1, Sprint 4 deliverable.
"""

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WaveformPoint:
    """A single time-point across all nodes."""
    time: float
    voltages: dict[str, float] = field(default_factory=dict)


@dataclass
class Waveform:
    """A complete simulation waveform."""
    title: str = ""
    node_names: list[str] = field(default_factory=list)
    points: list[WaveformPoint] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def num_points(self) -> int:
        return len(self.points)

    @property
    def num_nodes(self) -> int:
        return len(self.node_names)

    @property
    def time_range(self) -> tuple[float, float]:
        if not self.points:
            return (0.0, 0.0)
        return (self.points[0].time, self.points[-1].time)

    def voltage(self, node: str, time: float) -> Optional[float]:
        """Get voltage at a node at the closest time point."""
        if not self.points:
            return None
        # Binary search for closest time
        lo, hi = 0, len(self.points) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self.points[mid].time < time:
                lo = mid + 1
            else:
                hi = mid
        return self.points[lo].voltages.get(node)

    def node_trace(self, node: str) -> list[tuple[float, float]]:
        """Get the full time-voltage trace for a node."""
        return [
            (p.time, p.voltages.get(node, 0.0))
            for p in self.points
            if node in p.voltages
        ]

    def summary(self) -> str:
        """Human-readable waveform summary."""
        tmin, tmax = self.time_range
        lines = [
            f"Waveform: {self.title}",
            f"  Nodes: {self.num_nodes}",
            f"  Points: {self.num_points}",
            f"  Time: {tmin:.2e}s → {tmax:.2e}s",
        ]
        # Show final voltage at each node
        if self.points:
            last = self.points[-1]
            for name in sorted(self.node_names):
                v = last.voltages.get(name, 0.0)
                lines.append(f"  V({name}) = {v:.6f}V")
        return "\n".join(lines)


@dataclass
class PartitionWaveform:
    """Waveform data from a single partition."""
    partition_id: int
    node_mapping: dict[str, str]  # local_name → global_name
    times: list[float] = field(default_factory=list)
    voltages: dict[str, list[float]] = field(default_factory=dict)


def merge_waveforms(
    partition_waveforms: list[PartitionWaveform],
    title: str = "Merged Waveform",
) -> Waveform:
    """
    Merge per-partition waveforms into a unified waveform.

    For boundary nodes that appear in multiple partitions,
    uses the average of the values (they should be close
    if predictions are good).

    Args:
        partition_waveforms: List of per-partition waveform data.
        title: Title for the merged waveform.

    Returns:
        Unified Waveform with all nodes.
    """
    if not partition_waveforms:
        return Waveform(title=title)

    # Collect all unique time points and sort
    all_times: set[float] = set()
    for pw in partition_waveforms:
        all_times.update(pw.times)
    sorted_times = sorted(all_times)

    # Collect all global node names
    all_nodes: set[str] = set()
    for pw in partition_waveforms:
        all_nodes.update(pw.node_mapping.values())

    waveform = Waveform(
        title=title,
        node_names=sorted(all_nodes),
    )

    # For each time point, merge voltages from all partitions
    for t in sorted_times:
        point = WaveformPoint(time=t)
        node_values: dict[str, list[float]] = {}

        for pw in partition_waveforms:
            # Find closest time index in this partition
            idx = _find_closest_time(pw.times, t)
            if idx is None:
                continue

            for local_name, global_name in pw.node_mapping.items():
                if local_name in pw.voltages and idx < len(pw.voltages[local_name]):
                    v = pw.voltages[local_name][idx]
                    if global_name not in node_values:
                        node_values[global_name] = []
                    node_values[global_name].append(v)

        # Average boundary node values
        for name, values in node_values.items():
            point.voltages[name] = sum(values) / len(values)

        waveform.points.append(point)

    return waveform


def merge_from_json_files(
    json_paths: list[str],
    node_mappings: list[dict[str, str]],
    title: str = "Merged Waveform",
) -> Waveform:
    """
    Merge waveform JSON files produced by solver nodes.

    Each JSON file is expected to have:
    {
        "partition_id": 0,
        "waveform": [[time, [v0, v1, ...]], ...]
    }

    Args:
        json_paths: Paths to per-partition JSON waveform files.
        node_mappings: Per-partition local→global node name maps.
        title: Title for the merged waveform.

    Returns:
        Unified Waveform.
    """
    partition_waveforms = []

    for i, path in enumerate(json_paths):
        with open(path, "r") as f:
            data = json.load(f)

        pw = PartitionWaveform(
            partition_id=data.get("partition_id", i),
            node_mapping=node_mappings[i] if i < len(node_mappings) else {},
        )

        for entry in data.get("waveform", []):
            t = entry[0]
            voltages = entry[1]
            pw.times.append(t)
            for j, v in enumerate(voltages):
                node_name = f"n{j}"
                if node_name not in pw.voltages:
                    pw.voltages[node_name] = []
                pw.voltages[node_name].append(v)

        partition_waveforms.append(pw)

    return merge_waveforms(partition_waveforms, title=title)


def export_csv(waveform: Waveform, output_path: str) -> None:
    """Export waveform to CSV file."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        header = ["time"] + [f"V({n})" for n in waveform.node_names]
        writer.writerow(header)

        # Data rows
        for point in waveform.points:
            row = [f"{point.time:.6e}"]
            for name in waveform.node_names:
                v = point.voltages.get(name, 0.0)
                row.append(f"{v:.6e}")
            writer.writerow(row)

    print(f"[merger] Exported {waveform.num_points} points to {output_path}")


def export_json(waveform: Waveform, output_path: str) -> None:
    """Export waveform to JSON file."""
    data = {
        "title": waveform.title,
        "nodes": waveform.node_names,
        "metadata": waveform.metadata,
        "waveform": [
            {
                "time": p.time,
                "voltages": {n: p.voltages.get(n, 0.0) for n in waveform.node_names},
            }
            for p in waveform.points
        ],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[merger] Exported {waveform.num_points} points to {output_path}")


def _find_closest_time(times: list[float], target: float) -> Optional[int]:
    """Binary search for the closest time index."""
    if not times:
        return None

    lo, hi = 0, len(times) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if times[mid] < target:
            lo = mid + 1
        else:
            hi = mid

    # Check if neighboring index is closer
    if lo > 0 and abs(times[lo - 1] - target) < abs(times[lo] - target):
        return lo - 1
    return lo
