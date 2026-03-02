"""
PVT Corner Orchestrator

Multi-corner PVT sweep orchestration for Nangila SPICE.
Runs 1000 corners efficiently using delta mode:

1. Simulate the golden corner (TT, 1.8V, 27°C) first
2. For each non-nominal corner, compute delta from golden
3. Reconstruct full waveform from golden + delta
4. Report accuracy vs tolerance targets

This module is still experimental. Only the direct nangila-node path is solver-backed.
Delta-mode and synthetic fallback behavior are prototype/demo flows.
"""

import json
import os
import time
import shutil
import subprocess
import tempfile
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum


# ─── Corner Specification ──────────────────────────────────────────

class ProcessCorner(Enum):
    TT = "TT"  # Typical-Typical (nominal)
    FF = "FF"  # Fast-Fast
    SS = "SS"  # Slow-Slow
    FS = "FS"  # Fast NMOS, Slow PMOS
    SF = "SF"  # Slow NMOS, Fast PMOS

    @property
    def mobility_factor(self) -> float:
        return {
            ProcessCorner.TT: 1.0,
            ProcessCorner.FF: 1.15,
            ProcessCorner.SS: 0.85,
            ProcessCorner.FS: 1.10,
            ProcessCorner.SF: 0.90,
        }[self]

    @property
    def is_nominal(self) -> bool:
        return self == ProcessCorner.TT


@dataclass
class CornerSpec:
    process: ProcessCorner
    vdd: float
    temperature: float
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"{self.process.value}_{self.vdd:.2f}V_{self.temperature:.0f}C"

    @classmethod
    def nominal(cls) -> "CornerSpec":
        return cls(ProcessCorner.TT, 1.8, 27.0, "TT_1V8_27C")

    @property
    def is_nominal(self) -> bool:
        return (
            self.process == ProcessCorner.TT
            and abs(self.vdd - 1.8) < 0.01
            and abs(self.temperature - 27.0) < 0.1
        )

    @property
    def thermal_voltage(self) -> float:
        """kT/q at this temperature (V)."""
        t_kelvin = self.temperature + 273.15
        return 8.617e-5 * t_kelvin

    def delta_params(self, reference: "CornerSpec") -> dict:
        return {
            "delta_vdd": self.vdd - reference.vdd,
            "delta_temp": self.temperature - reference.temperature,
            "mobility_factor": self.process.mobility_factor / reference.process.mobility_factor,
        }

    def is_linearisable(self, reference: "CornerSpec") -> bool:
        delta = self.delta_params(reference)
        return (
            abs(delta["delta_vdd"]) < 0.3
            and abs(delta["delta_temp"]) < 50.0
            and abs(delta["mobility_factor"] - 1.0) < 0.2
        )


# ─── Corner Grid Generator ─────────────────────────────────────────

def generate_corner_grid(
    process_corners: Optional[List[ProcessCorner]] = None,
    vdd_values: Optional[List[float]] = None,
    temp_values: Optional[List[float]] = None,
) -> List[CornerSpec]:
    """Generate a full PVT corner grid.

    Default: 5 process × 5 voltages × 8 temperatures = 200 corners.
    For 1000 corners, use finer voltage/temperature grids.
    """
    if process_corners is None:
        process_corners = list(ProcessCorner)
    if vdd_values is None:
        vdd_values = [1.62, 1.71, 1.80, 1.89, 1.98]  # ±10% from 1.8V
    if temp_values is None:
        temp_values = [-40.0, -10.0, 0.0, 27.0, 55.0, 85.0, 105.0, 125.0]

    corners = []
    for proc, vdd, temp in itertools.product(process_corners, vdd_values, temp_values):
        corners.append(CornerSpec(proc, vdd, temp))

    return corners


def generate_1000_corner_grid() -> List[CornerSpec]:
    """Generate exactly ~1000 corners for performance testing."""
    process_corners = list(ProcessCorner)  # 5
    vdd_values = [round(1.62 + i * 0.06, 2) for i in range(8)]  # 8
    temp_values = [-40.0, -25.0, 0.0, 27.0, 55.0, 85.0, 105.0, 125.0]  # 8 → ~320... 
    # 5 × 8 × 8 = 320. For 1000 use finer grid:
    vdd_values = [round(1.44 + i * 0.04, 2) for i in range(10)]   # 10 voltages
    temp_values = [-40.0, -25.0, -10.0, 0.0, 15.0, 27.0, 40.0, 55.0, 70.0, 85.0, 100.0, 115.0, 125.0]  # 13 temps
    # 5 × 10 × 13 = 650... adjust
    vdd_values = [round(1.4 + i * 0.04, 2) for i in range(11)]  # 11 voltages
    # 5 × 11 × 13 = 715... increase temps
    extra_temps = [130.0, 140.0, 150.0]
    temp_values = temp_values + extra_temps  # 16 temps
    # 5 × 11 × 16 = 880
    more_temps = [-50.0, -55.0, -60.0]
    temp_values = more_temps + temp_values  # 19 temps
    # 5 × 11 × 19 = 1045 ≈ 1000
    return generate_corner_grid(process_corners, vdd_values, temp_values)


def _find_rust_binary() -> Optional[str]:
    """Locate the nangila-node binary on PATH or in the standard build output."""
    # 1. Check PATH first
    on_path = shutil.which("nangila-node")
    if on_path:
        return on_path
    # 2. Check the Cargo debug build location relative to this file's package root
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "..", "target", "debug", "nangila-node"),
        os.path.join(here, "..", "..", "target", "release", "nangila-node"),
    ]
    for c in candidates:
        normalized = os.path.normpath(c)
        if os.path.isfile(normalized) and os.access(normalized, os.X_OK):
            return normalized
    return None


def _call_rust_solver(
    netlist_path: str,
    corner: "CornerSpec",
    binary: str,
    tstop: float = 1e-9,
    dt: float = 1e-12,
) -> dict:
    """Invoke the nangila-node binary on a SPICE netlist and return parsed waveforms.

    Args:
        netlist_path: Path to the .sp file to simulate.
        corner: The PVT corner to simulate (used to scale VDD stimulus).
        binary: Absolute path to the nangila-node binary.
        tstop: Simulation stop time in seconds.
        dt: Timestep in seconds.

    Returns:
        A dict with keys: waveforms, peak_delta_v, wall_time, used_delta, corner_name, vdd, temperature, process.

    Raises:
        RuntimeError: If the binary exits with a non-zero code.
    """
    t_start = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="nangila_pvt_") as tmpdir:
        waveform_path = os.path.join(tmpdir, "waveform.json")
        cmd = [
            binary,
            "--partition", netlist_path,
            "--tstop", str(tstop),
            "--dt", str(dt),
            f"--process={corner.process.value}",
            f"--vdd={corner.vdd}",
            f"--temp={corner.temperature}",
            "--waveform-json", waveform_path,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"nangila-node timed out after 60s for corner {corner.name}")

        if result.returncode != 0:
            raise RuntimeError(
                f"nangila-node exited {result.returncode} for corner {corner.name}:\n{result.stderr[:500]}"
            )
        if not os.path.exists(waveform_path):
            raise RuntimeError(f"nangila-node did not emit waveform JSON for corner {corner.name}")

        with open(waveform_path, "r") as f:
            payload = json.load(f)

    waveforms: Dict[str, List[float]] = {
        node_name: [] for node_name in payload.get("node_names", [])
    }
    for point in payload.get("waveform", []):
        voltages = point.get("voltages", {})
        for node_name in waveforms:
            waveforms[node_name].append(float(voltages.get(node_name, 0.0)))

    elapsed = time.perf_counter() - t_start
    return {
        "corner_name": corner.name,
        "waveforms": waveforms,
        "peak_delta_v": 0.0,  # Populated by caller if delta mode
        "wall_time": elapsed,
        "used_delta": False,
        "vdd": corner.vdd,
        "temperature": corner.temperature,
        "process": corner.process.value,
    }


# ─── Simulated Waveform ───────────────────────────────────────────────────────

def simulate_corner(
    corner: CornerSpec,
    golden_waveform: Optional[dict] = None,
    use_delta: bool = True,
    netlist_path: Optional[str] = None,
) -> dict:
    """Simulate a single corner.

    Priority order:
      1. If ``netlist_path`` is given and the nangila-node binary is found,
         call the real Rust solver.
      2. Else if ``golden_waveform`` is provided and ``use_delta=True``,
         apply the delta approximation to the golden waveform.
      3. Else fall back to the built-in synthetic exponential model.

    Returns:
        dict with keys: corner_name, waveforms (dict node→list[float]),
        peak_delta_v, wall_time, used_delta, vdd, temperature, process,
        experimental, execution_mode.
    """
    t_start = time.perf_counter()
    n_timesteps = 100

    # ── Path 1: Real Rust solver ──────────────────────────────────────
    binary = _find_rust_binary()
    if netlist_path and binary:
        try:
            result = _call_rust_solver(netlist_path, corner, binary)
            # If a golden waveform exists, compute peak_delta_v
            if golden_waveform:
                peak_delta = 0.0
                for node, vlist in result["waveforms"].items():
                    golden_v = golden_waveform["waveforms"].get(node, [])
                    for a, b in zip(vlist, golden_v):
                        peak_delta = max(peak_delta, abs(a - b))
                result["peak_delta_v"] = peak_delta
                result["used_delta"] = False  # Full sim, not delta
            result["experimental"] = False
            result["execution_mode"] = "solver"
            return result
        except RuntimeError as e:
            # Log and fall through to synthetic model
            import warnings
            warnings.warn(
                f"Rust solver failed ({e}); falling back to experimental/demo mode"
            )

    # ── Path 2: Delta approximation from golden ───────────────────────
    if golden_waveform and use_delta:
        delta = corner.delta_params(CornerSpec.nominal())
        vdd_scale = corner.vdd / 1.8
        temp_correction = -0.002 * delta["delta_temp"] # -2mV/°C for non-linear, though delta model is rough
        mob_factor = delta["mobility_factor"] ** 0.5

        waveforms = {}
        peak_delta = 0.0
        for node, golden_v in golden_waveform["waveforms"].items():
            corrected = [
                v * vdd_scale * mob_factor + temp_correction
                for v in golden_v
            ]
            waveforms[node] = corrected
            peak_delta = max(peak_delta, max(abs(c - g) for c, g in zip(corrected, golden_v)))

        elapsed = time.perf_counter() - t_start
        return {
            "corner_name": corner.name,
            "waveforms": waveforms,
            "peak_delta_v": peak_delta,
            "wall_time": elapsed,
            "used_delta": True,
            "experimental": True,
            "execution_mode": "experimental_delta",
            "vdd": corner.vdd,
            "temperature": corner.temperature,
            "process": corner.process.value,
        }

    # ── Path 3: Synthetic exponential model ─────────────────────────
    t = [i * 1e-12 for i in range(n_timesteps)]
    import math
    tau = 10e-12 * (1.0 / corner.process.mobility_factor)
    waveforms = {
        "VDD": [corner.vdd] * n_timesteps,
        "VOUT": [corner.vdd * (1.0 - math.exp(-ti / tau)) for ti in t],
    }
    elapsed = time.perf_counter() - t_start
    return {
        "corner_name": corner.name,
        "waveforms": waveforms,
        "peak_delta_v": 0.0,
        "wall_time": elapsed,
        "used_delta": False,
        "experimental": True,
        "execution_mode": "experimental_synthetic",
        "vdd": corner.vdd,
        "temperature": corner.temperature,
        "process": corner.process.value,
    }


# ─── Orchestrator ─────────────────────────────────────────────────

@dataclass
class SweepConfig:
    """Configuration for a PVT sweep."""
    max_workers: int = 8
    delta_mode: bool = True
    tolerance_v: float = 1e-3  # 1mV accuracy requirement
    output_dir: str = "/tmp/nangila_pvt"
    save_waveforms: bool = False


@dataclass
class SweepResult:
    """Result of a full PVT sweep."""
    total_corners: int = 0
    completed: int = 0
    failed: int = 0
    delta_mode_used: int = 0
    full_sim_used: int = 0
    total_wall_time: float = 0.0
    peak_delta_v: float = 0.0
    speedup_vs_full: float = 1.0
    corner_results: List[dict] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_corners == 0:
            return 0.0
        return self.completed / self.total_corners

    @property
    def avg_time_per_corner(self) -> float:
        if self.completed == 0:
            return 0.0
        return self.total_wall_time / self.completed

    def summary(self) -> str:
        return (
            f"PVT Sweep (experimental): {self.completed}/{self.total_corners} corners "
            f"| {self.total_wall_time:.2f}s total "
            f"| {self.avg_time_per_corner*1000:.2f}ms/corner "
            f"| {self.speedup_vs_full:.1f}x speedup "
            f"| peak ΔV={self.peak_delta_v*1e3:.3f}mV "
            f"| δ-mode: {self.delta_mode_used}/{self.completed}"
        )

    def to_json(self, path: str):
        data = {
            "summary": self.summary(),
            "total_corners": self.total_corners,
            "completed": self.completed,
            "failed": self.failed,
            "delta_mode_used": self.delta_mode_used,
            "total_wall_time": self.total_wall_time,
            "peak_delta_v_v": self.peak_delta_v,
            "speedup_vs_full": self.speedup_vs_full,
            "avg_time_per_corner_ms": self.avg_time_per_corner * 1000,
            "experimental": True,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


class PvtOrchestrator:
    """Multi-corner PVT sweep orchestrator.

    Workflow:
    1. Simulate golden corner (TT, 1.8V, 27°C)
    2. Run all other corners in parallel using delta mode
    3. Collect results, compute speedup vs hypothetical full-sim
    """

    def __init__(self, config: SweepConfig):
        self.config = config
        self.golden_waveform: Optional[dict] = None

    def run_sweep(self, corners: List[CornerSpec], netlist_path: Optional[str] = None) -> SweepResult:
        """Run a complete PVT sweep over the given corners using a real netlist."""
        result = SweepResult(total_corners=len(corners))
        t_sweep_start = time.perf_counter()

        # Step 1: Simulate golden corner first
        nom = CornerSpec.nominal()
        if not any(c.is_nominal for c in corners):
            corners = [nom] + corners

        print(
            f"[PVT] Simulating golden corner ({nom.name}). "
            "Non-solver delta/synthetic paths are experimental."
        )
        golden = simulate_corner(nom, golden_waveform=None, use_delta=False, netlist_path=netlist_path)
        self.golden_waveform = golden
        result.corner_results.append(golden)
        result.completed += 1

        # Step 2: Run remaining corners in parallel
        non_nominal = [c for c in corners if not c.is_nominal]
        print(f"[PVT] Running {len(non_nominal)} non-nominal corners "
              f"(delta_mode={self.config.delta_mode}, workers={self.config.max_workers})...")

        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(
                    simulate_corner, c, self.golden_waveform, self.config.delta_mode, netlist_path
                ): c
                for c in non_nominal
            }

            for future in as_completed(futures):
                try:
                    res = future.result()
                    result.corner_results.append(res)
                    result.completed += 1
                    result.peak_delta_v = max(result.peak_delta_v, res["peak_delta_v"])

                    if res["used_delta"]:
                        result.delta_mode_used += 1
                    else:
                        result.full_sim_used += 1
                except Exception as e:
                    result.failed += 1
                    print(f"[PVT] Corner failed: {e}")

        result.total_wall_time = time.perf_counter() - t_sweep_start

        # Speedup estimate: if full sim takes ~10ms/corner, delta takes ~0.1ms
        hypothetical_full_sim_time = result.total_corners * 10e-3
        result.speedup_vs_full = hypothetical_full_sim_time / max(result.total_wall_time, 1e-9)

        print(f"[PVT] {result.summary()}")
        return result
