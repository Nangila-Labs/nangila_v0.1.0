# Correctness-First Benchmark Report

## Purpose

This report summarizes the measured results of Nangila's correctness program against `ngspice`.

This report serves as the benchmark record for Nangila's correctness-first release. The data below is intended to show:

- pass/fail status against the v1 contract
- waveform error metrics versus `ngspice`
- local runtime characteristics of the current single-node solver

## Measurement Context

- Date: `2026-03-02`
- Platform: `macOS 15.6.1 arm64`
- Python: `3.14.3`
- Cargo: `1.91.1`
- Oracle: local `ngspice`
- Nangila runtime: local `nangila-node`
- Raw artifact: [`artifacts/phase1_benchmark_report.json`](/Users/craigchirara/nangila/nangila-spice/artifacts/phase1_benchmark_report.json)

## Headline Result

All official correctness cases for this release passed the v1 contract against `ngspice`.

- Mandatory gate: `10 / 10` passed
- Extended gate: `3 / 3` passed
- Total official validation cases: `13 / 13` passed

## Mandatory Gate

| Case | Pass | Nangila Runtime (s) | ngspice Runtime (s) | Max Error (V) | RMS Error (V) | Final Error (V) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `simple_rc` | yes | 0.008 | 0.045 | 1.80e-05 | 1.27e-05 | 1.80e-05 |
| `inverter` | yes | 0.053 | 0.020 | 1.40e-02 | 1.19e-03 | 2.69e-09 |
| `sram_6t` | yes | 0.159 | 0.023 | 7.47e-03 | 1.83e-04 | 1.67e-04 |
| `c17_full` | yes | 0.009 | 0.020 | 6.52e-22 | 2.80e-23 | 6.52e-22 |
| `c17_synth` | yes | 0.008 | 0.019 | 6.52e-22 | 2.80e-23 | 6.52e-22 |
| `c17_auto` | yes | 0.010 | 0.020 | 1.45e-31 | 8.06e-33 | 1.45e-31 |
| `c432_auto` | yes | 2.383 | 0.105 | 1.65e-27 | 1.92e-29 | 1.65e-27 |
| `s27_auto` | yes | 0.014 | 0.024 | 1.32e-30 | 4.06e-32 | 1.32e-30 |
| `s382_auto` | yes | 1.048 | 0.080 | 3.34e-31 | 1.67e-32 | 3.34e-31 |
| `s641_auto` | yes | 2.735 | 0.142 | 1.65e-27 | 3.69e-29 | 1.65e-27 |

Mandatory-gate aggregate runtime:

- Nangila total: `6.429 s`
- ngspice total: `0.498 s`

## Extended Gate

These cases are official validation cases, but they remain extended-only because they exceed the per-change CI budget.

| Case | Pass | Nangila Runtime (s) | ngspice Runtime (s) | Max Error (V) | RMS Error (V) | Final Error (V) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `c880_auto` | yes | 37.259 | 0.282 | 1.66e-27 | 3.77e-29 | 1.66e-27 |
| `c1355_auto` | yes | 19.675 | 0.352 | 1.73e-27 | 3.08e-29 | 1.73e-27 |
| `c1908_auto` | yes | 40.079 | 0.542 | 1.66e-27 | 3.21e-29 | 1.66e-27 |

Extended-gate aggregate runtime:

- Nangila total: `97.013 s`
- ngspice total: `1.177 s`

## What This Means

This release establishes that Nangila can now be validated honestly against `ngspice` on its defined v1 subset.

The benchmark data supports these claims:

- the single-node solver passes the official correctness suite
- the project no longer depends on false-green integration behavior to claim success
- larger synthesized decks can be checked reproducibly, even when they are too expensive for the per-change gate

The data does not support a performance claim yet.

On this local release run, Nangila is slower than `ngspice` across both the mandatory and extended suites. That is acceptable at this stage because this release was scoped around correctness infrastructure, not sparse optimization or acceleration.

## Publishable Framing

The defensible public statement is:

`Nangila is shipping a correctness-first release for its v1 transient subset, with oracle-backed agreement against ngspice across a documented mandatory and extended benchmark suite.`

The wrong public statement would be:

`Nangila is already faster than ngspice.`

That is not what the current release data shows.
