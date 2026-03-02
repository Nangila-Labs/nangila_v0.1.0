"""
Correctness harness utilities for Phase 1.

Provides:
  - Nangila waveform execution
  - ngspice oracle execution
  - waveform normalization and comparison
"""

from __future__ import annotations

from bisect import bisect_left
from contextlib import contextmanager
from dataclasses import dataclass
import json
from math import sqrt
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Optional

from .parser import parse_netlist


@dataclass
class WaveformData:
    tool: str
    netlist: str
    node_names: list[str]
    times: list[float]
    traces: dict[str, list[float]]


@dataclass
class WaveformComparison:
    shared_nodes: list[str]
    sample_count: int
    max_abs_error: float
    rms_error: float
    final_abs_error: float
    max_edge_timing_error: Optional[float]

    def within_tolerance(
        self,
        *,
        max_abs_tol: float,
        rms_tol: float,
        final_abs_tol: float,
        edge_tol: Optional[float] = None,
    ) -> bool:
        if self.sample_count == 0:
            return False
        if self.max_abs_error > max_abs_tol:
            return False
        if self.rms_error > rms_tol:
            return False
        if self.final_abs_error > final_abs_tol:
            return False
        if edge_tol is not None and self.max_edge_timing_error is not None:
            if self.max_edge_timing_error > edge_tol:
                return False
        return True


@dataclass(frozen=True)
class V1ToleranceProfile:
    circuit_class: str
    max_abs_tol: float
    rms_tol: float
    final_abs_tol: float
    edge_tol: Optional[float] = None


def v1_tolerance_profile(*, nonlinear: bool, vdd: float = 1.8) -> V1ToleranceProfile:
    if nonlinear:
        return V1ToleranceProfile(
            circuit_class="nonlinear",
            max_abs_tol=max(25e-3, 0.02 * vdd),
            rms_tol=max(10e-3, 0.01 * vdd),
            final_abs_tol=max(10e-3, 0.01 * vdd),
            edge_tol=10e-12,
        )
    return V1ToleranceProfile(
        circuit_class="passive",
        max_abs_tol=max(1e-3, 0.001 * vdd),
        rms_tol=max(0.5e-3, 0.0005 * vdd),
        final_abs_tol=max(1e-3, 0.001 * vdd),
        edge_tol=None,
    )


def within_v1_contract(
    comparison: WaveformComparison,
    *,
    nonlinear: bool,
    vdd: float = 1.8,
) -> tuple[bool, V1ToleranceProfile]:
    profile = v1_tolerance_profile(nonlinear=nonlinear, vdd=vdd)
    return (
        comparison.within_tolerance(
            max_abs_tol=profile.max_abs_tol,
            rms_tol=profile.rms_tol,
            final_abs_tol=profile.final_abs_tol,
            edge_tol=profile.edge_tol,
        ),
        profile,
    )


@dataclass(frozen=True)
class CorrectnessCase:
    name: str
    netlist_path: str
    nodes: list[str]
    tstop: float
    dt: float
    nonlinear: bool
    use_initial_conditions: bool = False
    vdd: float = 1.8


@dataclass(frozen=True)
class BlockedCorrectnessCase:
    name: str
    netlist_path: str
    reason: str


_DEFAULT_LEVEL1_MODEL_BLOCK = """* Default Level-1 models injected by the correctness harness
.MODEL NMOS NMOS (LEVEL=1 VTO=0.5 KP=110u)
.MODEL PMOS PMOS (LEVEL=1 VTO=-0.5 KP=55u)

"""


_MOCK_CELL_FIXTURE_LIBRARY = {
    "AOI221_X1": """.SUBCKT AOI221_X1 A B C D E Z
X1 A B int1 AND2_X1
X2 C D int2 AND2_X1
X3 int1 int2 E Z NOR3_X1
.ENDS
""",
    "OAI221_X1": """.SUBCKT OAI221_X1 A B C D E Z
X1 A B int1 OR2_X1
X2 C D int2 OR2_X1
X3 int1 int2 E Z NAND3_X1
.ENDS
""",
    "AOI222_X1": """.SUBCKT AOI222_X1 A B C D E F Z
X1 A B int1 AND2_X1
X2 C D int2 AND2_X1
X3 E F int3 AND2_X1
X4 int1 int2 int3 Z NOR3_X1
.ENDS
""",
    "OAI222_X1": """.SUBCKT OAI222_X1 A B C D E F Z
X1 A B int1 OR2_X1
X2 C D int2 OR2_X1
X3 E F int3 OR2_X1
X4 int1 int2 int3 Z NAND3_X1
.ENDS
""",
}


_C17_SYNTH_STANDALONE_PREFIX = """* NANGATE 45nm MOCK SUBCIRCUITS FOR LOGIC GATES (2-port to 4-port mappings)
* Note: Global VDD/VSS are hardcoded to node 1 and 0 (GND) for simulation purposes

.SUBCKT INV_X1 A Z
M1 Z A 0 0 NMOS W=0.2u L=0.05u
M2 Z A 1 1 PMOS W=0.4u L=0.05u
.ENDS

.SUBCKT NAND2_X1 A1 A2 Z
M1 Z A1 1 1 PMOS W=0.4u L=0.05u
M2 Z A2 1 1 PMOS W=0.4u L=0.05u
M3 Z A1 int 0 NMOS W=0.2u L=0.05u
M4 int A2 0 0 NMOS W=0.2u L=0.05u
.ENDS

.SUBCKT AND2_X1 A1 A2 Z
X1 A1 A2 Z_bar NAND2_X1
X2 Z_bar Z INV_X1
.ENDS

.SUBCKT OAI21_X1 A B C Z
M1 Z A 1 1 PMOS W=0.4u L=0.05u
M2 Z B int1 1 PMOS W=0.4u L=0.05u
M3 int1 C 1 1 PMOS W=0.4u L=0.05u
M4 Z A 0 0 NMOS W=0.2u L=0.05u
M5 Z B 0 0 NMOS W=0.2u L=0.05u
M6 Z C 0 0 NMOS W=0.2u L=0.05u
.ENDS

.SUBCKT AOI21_X1 A B C Z
M1 Z A int1 1 PMOS W=0.4u L=0.05u
M2 int1 B 1 1 PMOS W=0.4u L=0.05u
M3 Z C 1 1 PMOS W=0.4u L=0.05u
M4 Z A 0 0 NMOS W=0.2u L=0.05u
M5 Z B 0 0 NMOS W=0.2u L=0.05u
M6 Z C 0 0 NMOS W=0.2u L=0.05u
.ENDS

* Provide VDD power source explicitly
Vvdd 1 0 1.8

* Input Signals (N1, N2, N3, N6, N7)
V_N1 N1 0 1.8
V_N2 N2 0 0.0
V_N3 N3 0 1.8
V_N6 N6 0 0.0
V_N7 N7 0 1.8

* No anti-floating shunts are needed here; the logic network is fully driven.

"""


def collect_flattened_nodes(netlist_path: str) -> list[str]:
    with _prepared_netlist_path(netlist_path, include_default_models=False) as prepared_netlist_path:
        netlist = parse_netlist(prepared_netlist_path).flatten()
        nodes: set[str] = set()
        for device in netlist.devices:
            for node in device.nodes:
                lowered = node.lower()
                if lowered in {"0", "gnd"}:
                    continue
                nodes.add(lowered)
        return sorted(nodes)


def collect_observable_nodes(
    netlist_path: str,
    *,
    exclude_zero_volt_alias_sinks: bool = False,
) -> list[str]:
    with _prepared_netlist_path(netlist_path, include_default_models=False) as prepared_netlist_path:
        netlist = parse_netlist(prepared_netlist_path)
        nodes: set[str] = set()
        zero_volt_alias_sinks: set[str] = set()
        for device in netlist.devices:
            for node in device.nodes:
                lowered = node.lower()
                if lowered in {"0", "gnd"}:
                    continue
                nodes.add(lowered)
            if (
                exclude_zero_volt_alias_sinks
                and device.dev_type == "V"
                and len(device.nodes) >= 2
            ):
                value = device.params.get("value", "").strip().lower()
                if device.nodes[1].lower() not in {"0", "gnd"} and value in {
                    "0",
                    "0.0",
                    "dc 0",
                    "dc 0.0",
                }:
                    zero_volt_alias_sinks.add(device.nodes[1].lower())
        return sorted(nodes - zero_volt_alias_sinks)


def official_correctness_cases(repo_root: str) -> list[CorrectnessCase]:
    def p(*parts: str) -> str:
        return str(Path(repo_root, *parts))

    return [
        CorrectnessCase(
            name="simple_rc",
            netlist_path=p("tests", "spice_examples", "simple_rc.sp"),
            nodes=["vdd", "n1"],
            tstop=1e-11,
            dt=1e-12,
            nonlinear=False,
        ),
        CorrectnessCase(
            name="inverter",
            netlist_path=p("benchmarks", "reference_circuits", "inverter.sp"),
            nodes=["in", "out", "vdd"],
            tstop=2e-9,
            dt=1e-11,
            nonlinear=True,
        ),
        CorrectnessCase(
            name="sram_6t",
            netlist_path=p("benchmarks", "reference_circuits", "sram_6t.sp"),
            nodes=["q", "qb", "bl", "blb", "wl", "vdd"],
            tstop=5e-9,
            dt=1e-11,
            nonlinear=True,
            use_initial_conditions=True,
        ),
        CorrectnessCase(
            name="c17_full",
            netlist_path=p("benchmarks", "reference_circuits", "c17_full.sp"),
            nodes=collect_observable_nodes(
                p("benchmarks", "reference_circuits", "c17_full.sp")
            ),
            tstop=1e-11,
            dt=1e-12,
            nonlinear=True,
        ),
        CorrectnessCase(
            name="c17_synth",
            netlist_path=p("benchmarks", "reference_circuits", "c17_synth.sp"),
            nodes=collect_observable_nodes(
                p("benchmarks", "reference_circuits", "c17_synth.sp")
            ),
            tstop=1e-11,
            dt=1e-12,
            nonlinear=True,
        ),
    ]


def phase1_automated_correctness_cases(repo_root: str) -> list[CorrectnessCase]:
    return official_correctness_cases(repo_root) + [
        CorrectnessCase(
            name="c17_auto",
            netlist_path=str(Path(repo_root, "benchmarks", "auto_synth", "c17.sp")),
            nodes=collect_observable_nodes(
                str(Path(repo_root, "benchmarks", "auto_synth", "c17.sp"))
            ),
            tstop=1e-11,
            dt=1e-12,
            nonlinear=True,
        ),
        CorrectnessCase(
            name="c432_auto",
            netlist_path=str(Path(repo_root, "benchmarks", "auto_synth", "c432.sp")),
            nodes=collect_observable_nodes(
                str(Path(repo_root, "benchmarks", "auto_synth", "c432.sp"))
            ),
            tstop=1e-11,
            dt=1e-12,
            nonlinear=True,
        ),
        CorrectnessCase(
            name="s27_auto",
            netlist_path=str(Path(repo_root, "benchmarks", "auto_synth", "s27.sp")),
            nodes=collect_observable_nodes(
                str(Path(repo_root, "benchmarks", "auto_synth", "s27.sp")),
                exclude_zero_volt_alias_sinks=True,
            ),
            tstop=1e-11,
            dt=1e-12,
            nonlinear=True,
        ),
        CorrectnessCase(
            name="s382_auto",
            netlist_path=str(Path(repo_root, "benchmarks", "auto_synth", "s382.sp")),
            nodes=collect_observable_nodes(
                str(Path(repo_root, "benchmarks", "auto_synth", "s382.sp")),
                exclude_zero_volt_alias_sinks=True,
            ),
            tstop=1e-11,
            dt=1e-12,
            nonlinear=True,
        ),
        CorrectnessCase(
            name="s641_auto",
            netlist_path=str(Path(repo_root, "benchmarks", "auto_synth", "s641.sp")),
            nodes=collect_observable_nodes(
                str(Path(repo_root, "benchmarks", "auto_synth", "s641.sp")),
                exclude_zero_volt_alias_sinks=True,
            ),
            tstop=1e-11,
            dt=1e-12,
            nonlinear=True,
        ),
    ]


def extended_phase1_correctness_cases(repo_root: str) -> list[CorrectnessCase]:
    return [
        CorrectnessCase(
            name="c880_auto",
            netlist_path=str(Path(repo_root, "benchmarks", "auto_synth", "c880.sp")),
            nodes=collect_observable_nodes(
                str(Path(repo_root, "benchmarks", "auto_synth", "c880.sp"))
            ),
            tstop=1e-11,
            dt=1e-12,
            nonlinear=True,
        ),
        CorrectnessCase(
            name="c1355_auto",
            netlist_path=str(Path(repo_root, "benchmarks", "auto_synth", "c1355.sp")),
            nodes=collect_observable_nodes(
                str(Path(repo_root, "benchmarks", "auto_synth", "c1355.sp"))
            ),
            tstop=1e-11,
            dt=1e-12,
            nonlinear=True,
        ),
        CorrectnessCase(
            name="c1908_auto",
            netlist_path=str(Path(repo_root, "benchmarks", "auto_synth", "c1908.sp")),
            nodes=collect_observable_nodes(
                str(Path(repo_root, "benchmarks", "auto_synth", "c1908.sp")),
                exclude_zero_volt_alias_sinks=True,
            ),
            tstop=1e-11,
            dt=1e-12,
            nonlinear=True,
        ),
    ]


def blocked_official_correctness_cases(repo_root: str) -> list[BlockedCorrectnessCase]:
    return [
        BlockedCorrectnessCase(
            name="c880",
            netlist_path=str(Path(repo_root, "benchmarks", "auto_synth", "c880.sp")),
            reason="Deck is validated via the opt-in extended correctness job, but remains outside the mandatory Phase 1 CI gate budget.",
        ),
    ]


def find_nangila_binary() -> Optional[str]:
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "target" / "debug" / "nangila-node",
        here.parents[2] / "target" / "release" / "nangila-node",
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_mode & 0o111:
            return str(candidate)
    return shutil.which("nangila-node")


def find_ngspice_binary() -> Optional[str]:
    return shutil.which("ngspice")


def run_nangila_waveform(
    netlist_path: str,
    *,
    tstop: float,
    dt: float,
    process: str = "TT",
    vdd: float = 1.8,
    temp: float = 27.0,
    binary: Optional[str] = None,
) -> WaveformData:
    binary = binary or find_nangila_binary()
    if not binary:
        raise RuntimeError("nangila-node binary not found")

    with tempfile.TemporaryDirectory(prefix="nangila_correctness_") as tmpdir:
        prepared_netlist = Path(tmpdir) / Path(netlist_path).name
        prepared_netlist.write_text(
            _prepare_correctness_source(
                Path(netlist_path).read_text(),
                include_default_models=False,
            )
        )
        output_path = Path(tmpdir) / "waveform.json"
        cmd = [
            binary,
            "--partition",
            str(prepared_netlist),
            "--tstop",
            str(tstop),
            "--dt",
            str(dt),
            "--process",
            process,
            "--vdd",
            str(vdd),
            "--temp",
            str(temp),
            "--waveform-json",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"nangila-node failed with exit code {result.returncode}: {result.stderr[:500]}"
            )
        if not output_path.exists():
            raise RuntimeError("nangila-node did not emit waveform JSON")
        payload = json.loads(output_path.read_text())

    traces = {name: [] for name in payload.get("node_names", [])}
    times: list[float] = []
    for point in payload.get("waveform", []):
        times.append(float(point["time"]))
        voltages = point.get("voltages", {})
        for name in traces:
            traces[name].append(float(voltages.get(name, 0.0)))

    return WaveformData(
        tool="nangila",
        netlist=payload.get("netlist", Path(netlist_path).name),
        node_names=list(traces.keys()),
        times=times,
        traces=traces,
    )


def run_ngspice_waveform(
    netlist_path: str,
    *,
    nodes: list[str],
    tstop: float,
    dt: float,
    use_initial_conditions: bool = False,
    ngspice_bin: Optional[str] = None,
) -> WaveformData:
    ngspice_bin = ngspice_bin or find_ngspice_binary()
    if not ngspice_bin:
        raise RuntimeError("ngspice binary not found")

    source_text = _prepare_correctness_source(
        Path(netlist_path).read_text(),
        include_default_models=True,
    )
    with tempfile.TemporaryDirectory(prefix="ngspice_correctness_") as tmpdir:
        output_path = Path(tmpdir) / "oracle.dat"
        wrapped_path = Path(tmpdir) / "oracle_run.sp"
        control_block = _build_control_block(output_path, dt, tstop, nodes, use_initial_conditions)
        wrapped_path.write_text(_inject_control_block(source_text, control_block))
        result = subprocess.run(
            [ngspice_bin, "-b", str(wrapped_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ngspice failed with exit code {result.returncode}: {result.stderr[:500]}"
            )
        if not output_path.exists():
            raise RuntimeError("ngspice did not emit oracle data")
        times, traces = _parse_wrdata(output_path, nodes)

    return WaveformData(
        tool="ngspice",
        netlist=Path(netlist_path).name,
        node_names=nodes,
        times=times,
        traces=traces,
    )


def compare_waveforms(
    reference: WaveformData,
    candidate: WaveformData,
    *,
    vdd: float = 1.8,
) -> WaveformComparison:
    shared_nodes = sorted(set(reference.node_names) & set(candidate.node_names))
    if not shared_nodes or not reference.times or not candidate.times:
        return WaveformComparison(shared_nodes, 0, float("inf"), float("inf"), float("inf"), None)

    sum_sq = 0.0
    max_abs = 0.0
    final_abs = 0.0
    sample_count = 0
    edge_errors: list[float] = []

    for node in shared_nodes:
        ref_trace = reference.traces[node]
        cand_trace = candidate.traces[node]

        for idx, time in enumerate(reference.times):
            ref_v = ref_trace[idx]
            cand_v = _interpolate(candidate.times, cand_trace, time)
            err = abs(cand_v - ref_v)
            max_abs = max(max_abs, err)
            sum_sq += err * err
            sample_count += 1

        final_time = min(reference.times[-1], candidate.times[-1])
        ref_final = _interpolate(reference.times, ref_trace, final_time)
        cand_final = _interpolate(candidate.times, cand_trace, final_time)
        final_abs = max(final_abs, abs(cand_final - ref_final))

        ref_edge = _first_crossing_time(reference.times, ref_trace, 0.5 * vdd)
        cand_edge = _first_crossing_time(candidate.times, cand_trace, 0.5 * vdd)
        if ref_edge is not None and cand_edge is not None:
            edge_errors.append(abs(cand_edge - ref_edge))

    rms = sqrt(sum_sq / sample_count) if sample_count else float("inf")
    return WaveformComparison(
        shared_nodes=shared_nodes,
        sample_count=sample_count,
        max_abs_error=max_abs,
        rms_error=rms,
        final_abs_error=final_abs,
        max_edge_timing_error=max(edge_errors) if edge_errors else None,
    )


def _build_control_block(
    output_path: Path,
    dt: float,
    tstop: float,
    nodes: list[str],
    use_initial_conditions: bool,
) -> str:
    vectors = " ".join(f"v({node})" for node in nodes)
    if use_initial_conditions:
        return (
            ".control\n"
            "set filetype=ascii\n"
            f"tran {dt} {tstop} uic\n"
            f"wrdata {output_path} {vectors}\n"
            "quit\n"
            ".endc\n"
        )

    return (
        ".control\n"
        "set filetype=ascii\n"
        "op\n"
        f"tran {dt} {tstop}\n"
        f"wrdata {output_path} {vectors}\n"
        "quit\n"
        ".endc\n"
    )


def _inject_control_block(netlist_text: str, control_block: str) -> str:
    lines = netlist_text.splitlines()
    for idx in range(len(lines) - 1, -1, -1):
        if lines[idx].strip().lower() == ".end":
            return "\n".join(lines[:idx] + [control_block.rstrip()] + lines[idx:]) + "\n"
    return netlist_text.rstrip() + "\n" + control_block


@contextmanager
def _prepared_netlist_path(netlist_path: str, *, include_default_models: bool):
    source_text = _prepare_correctness_source(
        Path(netlist_path).read_text(),
        include_default_models=include_default_models,
    )
    with tempfile.TemporaryDirectory(prefix="nangila_prepared_netlist_") as tmpdir:
        prepared_path = Path(tmpdir) / Path(netlist_path).name
        prepared_path.write_text(source_text)
        yield str(prepared_path)


def _prepare_correctness_source(netlist_text: str, *, include_default_models: bool) -> str:
    netlist_text = _inject_reference_scaffold(netlist_text)
    netlist_text = _dedupe_source_instance_names(netlist_text)
    netlist_text = _dedupe_parallel_fixed_sources(netlist_text)
    netlist_text = _inject_mock_cell_fixtures(netlist_text)
    if not include_default_models:
        return netlist_text

    upper = netlist_text.upper()
    uses_mock_mos = " NMOS" in upper or " PMOS" in upper
    has_nmos_model = ".MODEL NMOS" in upper
    has_pmos_model = ".MODEL PMOS" in upper

    if uses_mock_mos and (not has_nmos_model or not has_pmos_model):
        return _DEFAULT_LEVEL1_MODEL_BLOCK + netlist_text.lstrip()

    return netlist_text


def _inject_mock_cell_fixtures(netlist_text: str) -> str:
    upper = netlist_text.upper()
    missing_cells = [
        name
        for name in _MOCK_CELL_FIXTURE_LIBRARY
        if f".SUBCKT {name}" not in upper and name in upper
    ]
    if not missing_cells:
        return netlist_text

    library_block = (
        "* Additional mock-cell fixtures injected by the correctness harness\n"
        + "\n".join(_MOCK_CELL_FIXTURE_LIBRARY[name].rstrip() for name in missing_cells)
        + "\n\n"
    )
    return library_block + netlist_text.lstrip()


def _dedupe_source_instance_names(netlist_text: str) -> str:
    seen: dict[str, int] = {}
    rewritten: list[str] = []

    for line in netlist_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("*") or stripped.startswith("."):
            rewritten.append(line)
            continue

        parts = stripped.split()
        name = parts[0]
        if name and name[0].upper() in {"V", "I"}:
            key = name.upper()
            count = seen.get(key, 0)
            if count > 0:
                parts[0] = f"{name}_{count}"
                leading = line[: len(line) - len(line.lstrip())]
                rewritten.append(leading + " ".join(parts))
            else:
                rewritten.append(line)
            seen[key] = count + 1
        else:
            rewritten.append(line)

    return "\n".join(rewritten) + ("\n" if netlist_text.endswith("\n") else "")


def _dedupe_parallel_fixed_sources(netlist_text: str) -> str:
    seen: set[tuple[str, str, str, str]] = set()
    rewritten: list[str] = []

    for line in netlist_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("*") or stripped.startswith("."):
            rewritten.append(line)
            continue

        parts = stripped.split()
        name = parts[0]
        if len(parts) >= 4 and name and name[0].upper() in {"V", "I"}:
            signature = (
                name[0].upper(),
                parts[1].lower(),
                parts[2].lower(),
                " ".join(parts[3:]).lower(),
            )
            if signature in seen:
                continue
            seen.add(signature)

        rewritten.append(line)

    return "\n".join(rewritten) + ("\n" if netlist_text.endswith("\n") else "")


def _inject_reference_scaffold(netlist_text: str) -> str:
    stripped = netlist_text.lstrip()
    if (
        "X0 N2 1 INV_X1" in stripped
        and ".SUBCKT INV_X1" not in stripped
        and "V_N1 N1 0 1.8" not in stripped
    ):
        return _C17_SYNTH_STANDALONE_PREFIX + stripped
    return netlist_text


def _parse_wrdata(path: Path, nodes: list[str]) -> tuple[list[float], dict[str, list[float]]]:
    times: list[float] = []
    traces = {node: [] for node in nodes}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        values = [float(tok) for tok in stripped.split()]
        expected = len(nodes) * 2
        if len(values) < expected:
            continue
        times.append(values[0])
        for idx, node in enumerate(nodes):
            traces[node].append(values[idx * 2 + 1])
    return times, traces


def _interpolate(times: list[float], values: list[float], target: float) -> float:
    if not times:
        return 0.0
    if target <= times[0]:
        return values[0]
    if target >= times[-1]:
        return values[-1]
    idx = bisect_left(times, target)
    if idx < len(times) and times[idx] == target:
        return values[idx]
    left = idx - 1
    right = idx
    span = times[right] - times[left]
    if span <= 0.0:
        return values[left]
    frac = (target - times[left]) / span
    return values[left] + frac * (values[right] - values[left])


def _first_crossing_time(times: list[float], values: list[float], threshold: float) -> Optional[float]:
    if len(times) < 2:
        return None
    for idx in range(1, len(times)):
        v0 = values[idx - 1]
        v1 = values[idx]
        if v0 == threshold:
            return times[idx - 1]
        if (v0 < threshold <= v1) or (v0 > threshold >= v1):
            dv = v1 - v0
            if abs(dv) < 1e-30:
                return times[idx]
            frac = (threshold - v0) / dv
            return times[idx - 1] + frac * (times[idx] - times[idx - 1])
    return None
