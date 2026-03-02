# Phase 1 Summary Note

## Title

Nangila Phase 1: From False-Green Tests to Oracle-Validated Correctness

## Draft

Nangila has completed Phase 1 of its v1 program: the correctness foundation.

The important change is not a new feature. It is that the project now has a credible way to tell whether the simulator is right.

Earlier in the build-out, the end-to-end path could report success even when the produced waveforms were incomplete or physically wrong. Phase 1 fixed that by replacing permissive integration checks with an oracle-backed validation system built around `ngspice`.

The current system now does the following:

- runs the single-node Nangila solver as the authoritative Phase 1 path
- runs the same benchmark decks through `ngspice`
- compares full waveforms instead of final-value snapshots
- applies contract-level error tolerances
- fails explicitly when outputs diverge or when execution falls back to non-production paths

The correctness program is split into two tiers:

- a mandatory gate for per-change validation
- an extended gate for larger official decks that are valid but too expensive for every pull request

At the end of Phase 1, all official validation cases in both tiers pass the v1 correctness contract on the supported transient subset.

This is a meaningful milestone because it gives Nangila a trustworthy baseline for the next phases:

- Phase 2 can harden the single-node production path
- Phase 3 can focus on sparse-first matrix infrastructure
- later GPU and partitioned work can be measured against a real reference instead of optimistic demos

What this milestone does not claim is equally important.

Phase 1 is not a broad SPICE compatibility claim.
Phase 1 is not a distributed-runtime success claim.
Phase 1 is not yet a speed win versus `ngspice`.

It is a correctness and validation milestone. For an early simulator project, that is the right thing to finish first.

## Short Version

`Nangila completed Phase 1 correctness validation for its v1 transient subset, replacing false-green integration checks with oracle-backed waveform comparison against ngspice across a documented mandatory and extended benchmark suite.`

## Links

- Benchmark report: [docs/phase-1-benchmark-report.md](/Users/craigchirara/nangila/nangila-spice/docs/phase-1-benchmark-report.md)
- v1 contract: [docs/v1-simulator-contract.md](/Users/craigchirara/nangila/nangila-spice/docs/v1-simulator-contract.md)
- Phase 1 plan: [docs/phase-1-implementation-plan.md](/Users/craigchirara/nangila/nangila-spice/docs/phase-1-implementation-plan.md)
