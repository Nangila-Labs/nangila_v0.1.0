# Nangila SPICE

Nangila SPICE is an open-build transient circuit simulator under active development.

This public snapshot is a correctness-first release: an oracle-validated baseline for a constrained SPICE transient subset.

## Status

- This is a correctness-first release, not a performance release.
- The single-node Rust solver is the current validated path.
- Partitioned mode is still experimental and not part of the production claim.
- The supported scope is a constrained SPICE transient subset focused on digital-heavy and near-digital circuits.

Start here:

- v1 scope and contract: [docs/v1-simulator-contract.md](/Users/craigchirara/nangila/nangila-spice/docs/v1-simulator-contract.md)
- implementation plan: [docs/phase-1-implementation-plan.md](/Users/craigchirara/nangila/nangila-spice/docs/phase-1-implementation-plan.md)
- correctness-first benchmark report: [docs/correctness-first-benchmark-report.md](/Users/craigchirara/nangila/nangila-spice/docs/correctness-first-benchmark-report.md)
- correctness-first release note: [docs/correctness-first-release-note.md](/Users/craigchirara/nangila/nangila-spice/docs/correctness-first-release-note.md)

## What It Is

Nangila v1 is a transient simulator for a constrained SPICE subset, with correctness validated against `ngspice` on a documented benchmark suite.

The current validated scope is:

- transient analysis only
- flattened SPICE netlists
- `R`, `C`, `L`, independent `V/I` sources, simple diode models, and Level-1-style MOS support
- digital-heavy transistor-level circuits, SRAM-class structures, and passive interconnect networks

Nangila v1 is not yet:

- a general-purpose SPICE replacement
- broadly compatible with external `.include` and `.lib` ecosystems
- a validated distributed simulator
- a validated GPU-accelerated production path
- faster than `ngspice` on the published correctness benchmark suite

## Validation

The published correctness program has two tiers:

- mandatory per-change gate:
  `simple_rc`, `inverter`, `sram_6t`, `c17_full`, `c17_synth`, `c17_auto`, `c432_auto`, `s27_auto`, `s382_auto`, `s641_auto`
- extended validation gate:
  `c880_auto`, `c1355_auto`, `c1908_auto`

All `13/13` official cases currently pass the v1 correctness contract against `ngspice`.

Important limitation:

- partitioned runs are explicitly marked experimental
- by default, multi-partition execution falls back to the validated single-node waveform until real partition equivalence is implemented

## Try This First

If you want to test the project quickly, use this path:

1. install the package and `ngspice`
2. build `nangila-node`
3. run one single-node reference circuit
4. run the correctness-first benchmark report

Recommended first commands:

```bash
python3 -m pip install .
brew install ngspice
cargo build -p nangila-node --bin nangila-node
nangila run benchmarks/reference_circuits/inverter.sp --partitions 1 --tstop 2e-9 --dt 1e-11
nangila phase1-report
```

## Quickstart

### 1. Install Python package dependencies

```bash
python3 -m pip install .
```

### 2. Install `ngspice`

You need a working `ngspice` binary on your `PATH` for the oracle-backed correctness harness and the correctness-first benchmark report.

On macOS with Homebrew:

```bash
brew install ngspice
```

On Ubuntu or Debian:

```bash
sudo apt-get update
sudo apt-get install -y ngspice
```

### 3. Build `nangila-node`

The validated public path today is the CPU single-node build:

```bash
cargo build -p nangila-node --bin nangila-node
```

GPU and distributed execution are not required for the current correctness program.

### 4. Run a simulation

Single-node transient run:

```bash
nangila run benchmarks/reference_circuits/inverter.sp --partitions 1 --tstop 2e-9 --dt 1e-11
```

If you request partitioning, the CLI will surface the validation state explicitly.

## Reproducing the Correctness-First Benchmark Report

This command reproduces the published correctness-first benchmark report locally:

```bash
nangila phase1-report --include-extended
```

It writes:

- `artifacts/phase1_benchmark_report.json`
- `artifacts/phase1_benchmark_report.md`

That command runs the official correctness harness against `ngspice`, measures waveform error metrics, and records runtime for both the mandatory and extended validation gates.

If you only want the required per-change gate, omit `--include-extended`.

## Known Limitations

What outside users should expect today:

- `ngspice` is required for the benchmark report and oracle-backed correctness harness
- the validated path is single-node CPU execution
- partitioned mode is explicitly experimental and currently falls back to the validated single-node waveform by default
- the supported SPICE subset is intentionally narrow
- broad `.include` and `.lib` compatibility is not a v1 claim
- GPU support is not yet a validated production path
- current benchmark data supports correctness claims, not performance claims versus `ngspice`

## Running the Correctness Gates Directly

Mandatory gate:

```bash
python3 -m unittest src.tests.test_correctness.CorrectnessHarnessTests -v
```

Extended gate:

```bash
NANGILA_RUN_EXTENDED_CORRECTNESS=1 python3 -m unittest src.tests.test_correctness.ExtendedCorrectnessHarnessTests -v
```

Full Python test suite:

```bash
python3 -m unittest discover -s src/tests
```

Rust test suite:

```bash
cargo test -p nangila-node
```

## Architecture

The repository is split into three main areas:

- `nangila-node` (`Rust`): netlist parsing, device stamping, and the current validated single-node solver
- `src/nangila_spice` (`Python`): orchestration, correctness harnesses, reporting, and CLI
- `benchmarks` and `tests`: reference circuits, auto-synth decks, and correctness/integration coverage

## Open Build Notes

This repository is worth publishing now if it is framed honestly.

What is already credible:

- oracle-backed correctness validation against `ngspice`
- a documented v1 scope
- a reproducible benchmark report
- explicit handling of experimental partitioned mode

What is still in progress:

- sparse-first solver infrastructure
- real partition equivalence
- GPU validation
- performance work needed to compete with `ngspice`

## Current Roadmap

Publicly, this repository should be read as a correctness-first release. Internally, the project has completed Phase 1 and is moving into Phase 2.

- correctness-first release delivered:
  oracle-backed correctness validation against `ngspice`
- Phase 2 next:
  harden the single-node production path and remove remaining placeholder behavior from production-facing execution
- Phase 3 after that:
  replace dense internals with sparse-first solver infrastructure
- Later phases:
  real partition equivalence, measured PVT, and validated GPU acceleration

For the full plan, see [docs/production-roadmap.md](/Users/craigchirara/nangila/nangila-spice/docs/production-roadmap.md).

## Contributing

See [CONTRIBUTING.md](/Users/craigchirara/nangila/nangila-spice/CONTRIBUTING.md).

## License

MIT. See [LICENSE](/Users/craigchirara/nangila/nangila-spice/LICENSE).
