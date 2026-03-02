# GitHub Launch Draft

## Repository Description

Nangila SPICE is an open-build transient circuit simulator with ngspice-backed correctness validation for a constrained SPICE subset.

## Repository Short Description

Open-build transient simulator with ngspice-backed correctness validation.

## GitHub About Text

Open-build transient simulator with ngspice-backed correctness validation for a constrained SPICE subset.

## Pinned Project Blurb

Nangila is an in-progress transient circuit simulator with an oracle-validated Phase 1 baseline. The current public claim is correctness on a narrow SPICE subset, not broad compatibility or speed leadership.

## Suggested Topics

- `spice`
- `circuit-simulation`
- `eda`
- `rust`
- `python`
- `transient-analysis`
- `numerical-methods`
- `verification`

## Initial Release Title

`v0.2.0: Phase 1 correctness baseline`

## Recommended Tag

`v0.2.0`

Use this tag for the first public GitHub release once the Phase 1 baseline is committed.

Suggested commands after committing the release state:

```bash
git tag -a v0.2.0 -m "Phase 1 correctness baseline"
git push origin v0.2.0
```

## Initial Release Notes

This release marks Nangila's Phase 1 milestone.

What is new in this release:

- a documented v1 scope and simulator contract
- an `ngspice`-backed correctness harness
- contract-level waveform comparison against the official benchmark suite
- a documented split between mandatory and extended correctness gates
- a reproducible Phase 1 benchmark report
- explicit experimental status for partitioned execution
- install smoke coverage and cleaner public-facing CLI behavior

What this release means:

- Nangila now has an honest correctness baseline for its supported transient subset
- the current validated path is the single-node Rust solver
- the project is ready for public evaluation as an open build

What this release does not claim:

- broad SPICE compatibility
- validated distributed execution
- validated GPU acceleration
- performance superiority over `ngspice`

Recommended first steps after cloning:

```bash
python3 -m pip install .
brew install ngspice
cargo build -p nangila-node --bin nangila-node
nangila run benchmarks/reference_circuits/inverter.sp --partitions 1 --tstop 2e-9 --dt 1e-11
nangila phase1-report
```

Key docs:

- [docs/v1-simulator-contract.md](/Users/craigchirara/nangila/nangila-spice/docs/v1-simulator-contract.md)
- [docs/phase-1-benchmark-report.md](/Users/craigchirara/nangila/nangila-spice/docs/phase-1-benchmark-report.md)
- [docs/production-roadmap.md](/Users/craigchirara/nangila/nangila-spice/docs/production-roadmap.md)

## Launch Post Draft

I just open-sourced Nangila SPICE.

Nangila is an open-build transient simulator focused on a constrained SPICE subset for digital-heavy and near-digital circuits.

The project is not launching as a “faster than ngspice” claim. The current milestone is a correctness one: Phase 1 is complete, with oracle-backed waveform validation against `ngspice` across a documented benchmark suite.

Current state:

- single-node Rust solver is the validated path
- partitioned mode is experimental
- the v1 scope is narrow and explicit
- benchmark reproduction is built into the repo

If you want to evaluate it, the best place to start is the README and the Phase 1 benchmark report.

## Notes

I did not create the git tag in the current workspace state. A release tag should point at a clean committed snapshot of the Phase 1 baseline, not at an uncommitted working tree.
