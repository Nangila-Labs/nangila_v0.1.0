from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Iterable

from .correctness import (
    compare_waveforms,
    extended_phase1_correctness_cases,
    find_nangila_binary,
    find_ngspice_binary,
    phase1_automated_correctness_cases,
    run_nangila_waveform,
    run_ngspice_waveform,
    within_v1_contract,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _cargo_version() -> str:
    import subprocess

    result = subprocess.run(
        ["cargo", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return "unavailable"
    return result.stdout.strip()


def _tier_summary(cases: list[dict]) -> dict[str, float | int]:
    return {
        "case_count": len(cases),
        "pass_count": sum(1 for case in cases if case["pass"]),
        "nangila_runtime_s": sum(case["runtime_s"]["nangila"] for case in cases),
        "ngspice_runtime_s": sum(case["runtime_s"]["ngspice"] for case in cases),
    }


def _run_case(case, *, nangila_bin: str, ngspice_bin: str) -> dict:
    ngspice_start = time.perf_counter()
    oracle = run_ngspice_waveform(
        case.netlist_path,
        nodes=case.nodes,
        tstop=case.tstop,
        dt=case.dt,
        use_initial_conditions=case.use_initial_conditions,
        ngspice_bin=ngspice_bin,
    )
    ngspice_elapsed = time.perf_counter() - ngspice_start

    nangila_start = time.perf_counter()
    candidate = run_nangila_waveform(
        case.netlist_path,
        tstop=case.tstop,
        dt=case.dt,
        vdd=case.vdd,
        binary=nangila_bin,
    )
    nangila_elapsed = time.perf_counter() - nangila_start

    comparison = compare_waveforms(oracle, candidate, vdd=case.vdd)
    passed, profile = within_v1_contract(
        comparison,
        nonlinear=case.nonlinear,
        vdd=case.vdd,
    )
    return {
        "name": case.name,
        "netlist_path": case.netlist_path,
        "node_count": len(case.nodes),
        "sample_count": comparison.sample_count,
        "nonlinear": case.nonlinear,
        "pass": passed,
        "metrics": {
            "max_abs_error_v": comparison.max_abs_error,
            "rms_error_v": comparison.rms_error,
            "final_abs_error_v": comparison.final_abs_error,
            "max_edge_timing_error_s": comparison.max_edge_timing_error,
        },
        "tolerances": {
            "max_abs_tol_v": profile.max_abs_tol,
            "rms_tol_v": profile.rms_tol,
            "final_abs_tol_v": profile.final_abs_tol,
            "edge_tol_s": profile.edge_tol,
        },
        "runtime_s": {
            "nangila": nangila_elapsed,
            "ngspice": ngspice_elapsed,
            "ratio_vs_ngspice": (
                nangila_elapsed / ngspice_elapsed if ngspice_elapsed > 0 else None
            ),
        },
    }


def generate_phase1_benchmark_report(
    *,
    repo_root: str | Path | None = None,
    include_extended: bool = False,
) -> dict:
    root = Path(repo_root) if repo_root is not None else _project_root()
    nangila_bin = find_nangila_binary()
    ngspice_bin = find_ngspice_binary()
    if not nangila_bin:
        raise RuntimeError("nangila-node binary not found; build it first")
    if not ngspice_bin:
        raise RuntimeError("ngspice binary not found")

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "cargo": _cargo_version(),
            "nangila_binary": nangila_bin,
            "ngspice_binary": ngspice_bin,
        },
        "mandatory": [],
        "extended": [],
    }

    for case in phase1_automated_correctness_cases(str(root)):
        report["mandatory"].append(
            _run_case(case, nangila_bin=nangila_bin, ngspice_bin=ngspice_bin)
        )

    if include_extended:
        for case in extended_phase1_correctness_cases(str(root)):
            report["extended"].append(
                _run_case(case, nangila_bin=nangila_bin, ngspice_bin=ngspice_bin)
            )

    report["summary"] = {
        "mandatory": _tier_summary(report["mandatory"]),
        "extended": _tier_summary(report["extended"]),
    }
    return report


def _iter_tier_rows(report: dict, tier_name: str) -> Iterable[str]:
    cases = report[tier_name]
    if not cases:
        return []

    header = [
        f"## {tier_name.capitalize()} Gate",
        "",
        "| Case | Pass | Nangila Runtime (s) | ngspice Runtime (s) | Max Error (V) | RMS Error (V) | Final Error (V) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    rows = []
    for case in cases:
        rows.append(
            "| `{name}` | {passed} | {nangila:.3f} | {ngspice:.3f} | {max_err:.3e} | {rms:.3e} | {final:.3e} |".format(
                name=case["name"],
                passed="yes" if case["pass"] else "no",
                nangila=case["runtime_s"]["nangila"],
                ngspice=case["runtime_s"]["ngspice"],
                max_err=case["metrics"]["max_abs_error_v"],
                rms=case["metrics"]["rms_error_v"],
                final=case["metrics"]["final_abs_error_v"],
            )
        )
    summary = report["summary"][tier_name]
    footer = [
        "",
        f"- Nangila total: `{summary['nangila_runtime_s']:.3f} s`",
        f"- ngspice total: `{summary['ngspice_runtime_s']:.3f} s`",
    ]
    return header + rows + footer


def render_markdown_report(report: dict) -> str:
    mandatory = report["summary"]["mandatory"]
    extended = report["summary"]["extended"]
    total_cases = mandatory["case_count"] + extended["case_count"]
    total_passes = mandatory["pass_count"] + extended["pass_count"]
    lines = [
        "# Correctness-First Benchmark Report",
        "",
        "This report was generated from the Nangila Phase 1 correctness harness.",
        "",
        "## Summary",
        "",
        f"- Mandatory gate: `{mandatory['pass_count']} / {mandatory['case_count']}` passed",
        f"- Extended gate: `{extended['pass_count']} / {extended['case_count']}` passed",
        f"- Total: `{total_passes} / {total_cases}` passed",
        "",
        "## Environment",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Platform: `{report['environment']['platform']}`",
        f"- Python: `{report['environment']['python']}`",
        f"- Cargo: `{report['environment']['cargo']}`",
        "",
    ]
    lines.extend(_iter_tier_rows(report, "mandatory"))
    if report["extended"]:
        lines.extend([""])
        lines.extend(_iter_tier_rows(report, "extended"))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the Phase 1 Nangila vs ngspice benchmark report."
    )
    parser.add_argument(
        "--include-extended",
        action="store_true",
        help="Include the extended Phase 1 correctness gate.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/phase1_benchmark_report.json"),
        help="Path for the machine-readable report.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("artifacts/phase1_benchmark_report.md"),
        help="Path for the Markdown summary report.",
    )
    args = parser.parse_args(argv)

    report = generate_phase1_benchmark_report(
        include_extended=args.include_extended,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2))
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown_report(report))

    mandatory = report["summary"]["mandatory"]
    extended = report["summary"]["extended"]
    print(
        "Phase 1 benchmark report written.",
        f"mandatory={mandatory['pass_count']}/{mandatory['case_count']}",
        f"extended={extended['pass_count']}/{extended['case_count']}",
        f"json={args.output_json}",
        f"markdown={args.output_md}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
