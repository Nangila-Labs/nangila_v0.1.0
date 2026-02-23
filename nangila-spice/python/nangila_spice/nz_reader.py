"""
Nangila .nz Waveform Reader

Transparent decompression of .nz waveform files produced by
the Nangila SPICE solver. Compatible with the Rust nz.rs writer.

Phase 2, Sprint 7 deliverable.
"""

import struct
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Constants matching nz.rs
NZ_MAGIC = b"NZWF"
BLOCK_PREDICTED = 0x01
BLOCK_RAW = 0x02


@dataclass
class NzHeader:
    """Header of a .nz file."""
    version: int = 1
    num_nodes: int = 0
    num_points: int = 0
    t_start: float = 0.0
    t_end: float = 0.0
    error_bound: float = 1e-6
    node_names: List[str] = field(default_factory=list)


@dataclass
class NzWaveform:
    """Decompressed waveform data."""
    header: NzHeader
    time: List[float]
    signals: Dict[str, List[float]]  # node_name → voltage values
    stats: Dict[str, float] = field(default_factory=dict)

    @property
    def num_points(self) -> int:
        return len(self.time)

    @property
    def node_names(self) -> List[str]:
        return list(self.signals.keys())

    def voltage(self, node: str, time: Optional[float] = None) -> float:
        """Get voltage at a node, optionally at a specific time (binary search)."""
        if node not in self.signals:
            raise KeyError(f"Node '{node}' not found. Available: {self.node_names}")

        values = self.signals[node]

        if time is None:
            return values[-1] if values else 0.0

        # Binary search for closest time
        lo, hi = 0, len(self.time) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self.time[mid] < time:
                lo = mid + 1
            else:
                hi = mid

        if lo > 0 and abs(self.time[lo - 1] - time) < abs(self.time[lo] - time):
            return values[lo - 1]
        return values[lo]

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f".nz Waveform: {self.header.num_nodes} nodes, {self.header.num_points} points",
            f"  Time: {self.header.t_start:.2e}s → {self.header.t_end:.2e}s",
            f"  Error bound: {self.header.error_bound:.2e}V",
            f"  Nodes: {', '.join(self.node_names)}",
        ]
        if self.stats:
            lines.append(f"  Compression: {self.stats.get('compression_ratio', 0):.1f}x")
        return "\n".join(lines)


def read_nz(path: str) -> NzWaveform:
    """Read and decompress a .nz file.

    Args:
        path: Path to .nz file

    Returns:
        NzWaveform with decompressed signals
    """
    with open(path, "rb") as f:
        data = f.read()

    return _parse_nz_bytes(data)


def read_nz_bytes(data: bytes) -> NzWaveform:
    """Read and decompress .nz data from bytes."""
    return _parse_nz_bytes(data)


def _parse_nz_bytes(data: bytes) -> NzWaveform:
    """Parse .nz binary format."""
    pos = 0

    # Magic
    if data[0:4] != NZ_MAGIC:
        raise ValueError(f"Invalid .nz magic: {data[0:4]}")
    pos += 4

    # Header
    version = struct.unpack_from("<H", data, pos)[0]
    pos += 2
    num_nodes = struct.unpack_from("<I", data, pos)[0]
    pos += 4
    num_points = struct.unpack_from("<I", data, pos)[0]
    pos += 4
    t_start = struct.unpack_from("<d", data, pos)[0]
    pos += 8
    t_end = struct.unpack_from("<d", data, pos)[0]
    pos += 8
    error_bound = struct.unpack_from("<d", data, pos)[0]
    pos += 8

    # Node names
    node_names = []
    for _ in range(num_nodes):
        name_len = struct.unpack_from("<H", data, pos)[0]
        pos += 2
        name = data[pos : pos + name_len].decode("utf-8")
        pos += name_len
        node_names.append(name)

    # Time values
    time = []
    for _ in range(num_points):
        t = struct.unpack_from("<d", data, pos)[0]
        pos += 8
        time.append(t)

    # Blocks
    num_blocks = struct.unpack_from("<I", data, pos)[0]
    pos += 4

    # Initialize waveform arrays
    waveforms = {name: [0.0] * num_points for name in node_names}

    predicted_blocks = 0
    raw_blocks = 0

    for _ in range(num_blocks):
        block_type = data[pos]
        pos += 1

        if block_type == BLOCK_PREDICTED:
            predicted_blocks += 1
            node_idx = struct.unpack_from("<I", data, pos)[0]
            pos += 4
            start_point = struct.unpack_from("<I", data, pos)[0]
            pos += 4
            count = struct.unpack_from("<I", data, pos)[0]
            pos += 4
            base_value = struct.unpack_from("<d", data, pos)[0]
            pos += 8
            base_gradient = struct.unpack_from("<d", data, pos)[0]
            pos += 8
            scale = struct.unpack_from("<d", data, pos)[0]
            pos += 8

            residuals = []
            for _ in range(count):
                r = struct.unpack_from("<h", data, pos)[0]  # i16
                pos += 2
                residuals.append(r)

            # Decompress
            node_name = node_names[node_idx]
            vals = waveforms[node_name]
            t0 = time[start_point]

            for i in range(count):
                point_idx = start_point + i
                if point_idx >= num_points:
                    break

                dt = time[point_idx] - t0
                predicted = base_value + base_gradient * dt

                # Quadratic correction
                if i >= 2:
                    v0 = vals[point_idx - 2]
                    v1 = vals[point_idx - 1]
                    curvature = v0 - 2.0 * v1 + predicted
                    predicted += curvature * 0.25

                vals[point_idx] = predicted + residuals[i] * scale

        elif block_type == BLOCK_RAW:
            raw_blocks += 1
            node_idx = struct.unpack_from("<I", data, pos)[0]
            pos += 4
            start_point = struct.unpack_from("<I", data, pos)[0]
            pos += 4
            count = struct.unpack_from("<I", data, pos)[0]
            pos += 4

            node_name = node_names[node_idx]
            for i in range(count):
                v = struct.unpack_from("<d", data, pos)[0]
                pos += 8
                point_idx = start_point + i
                if point_idx < num_points:
                    waveforms[node_name][point_idx] = v

    header = NzHeader(
        version=version,
        num_nodes=num_nodes,
        num_points=num_points,
        t_start=t_start,
        t_end=t_end,
        error_bound=error_bound,
        node_names=node_names,
    )

    raw_bytes = num_points * num_nodes * 8
    compressed_bytes = len(data)
    ratio = raw_bytes / compressed_bytes if compressed_bytes > 0 else 1.0

    return NzWaveform(
        header=header,
        time=time,
        signals=waveforms,
        stats={
            "compression_ratio": ratio,
            "predicted_blocks": predicted_blocks,
            "raw_blocks": raw_blocks,
            "file_size_bytes": len(data),
        },
    )
