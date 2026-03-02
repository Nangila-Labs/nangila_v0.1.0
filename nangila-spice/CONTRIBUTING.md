# Contributing

Nangila is in active development. Contributions are welcome, but the project is still narrowing and hardening the v1 transient subset.

## Before You Start

Read these first:

- v1 contract: [docs/v1-simulator-contract.md](/Users/craigchirara/nangila/nangila-spice/docs/v1-simulator-contract.md)
- production roadmap: [docs/production-roadmap.md](/Users/craigchirara/nangila/nangila-spice/docs/production-roadmap.md)
- Phase 1 plan: [docs/phase-1-implementation-plan.md](/Users/craigchirara/nangila/nangila-spice/docs/phase-1-implementation-plan.md)

The main rule is simple: do not widen the public claim faster than the code and tests justify it.

## Development Setup

Install the Python package and runtime dependencies:

```bash
python3 -m pip install .
```

Install `ngspice`:

```bash
brew install ngspice
```

or

```bash
sudo apt-get update
sudo apt-get install -y ngspice
```

Build the validated Rust solver path:

```bash
cargo build -p nangila-node --bin nangila-node
```

## Test Commands

Rust tests:

```bash
cargo test -p nangila-node
```

Python tests:

```bash
python3 -m unittest discover -s src/tests
```

Mandatory correctness gate:

```bash
python3 -m unittest src.tests.test_correctness.CorrectnessHarnessTests -v
```

Extended correctness gate:

```bash
NANGILA_RUN_EXTENDED_CORRECTNESS=1 python3 -m unittest src.tests.test_correctness.ExtendedCorrectnessHarnessTests -v
```

Phase 1 benchmark report:

```bash
nangila phase1-report --include-extended
```

## Contribution Priorities

High-value areas:

- single-node solver correctness
- sparse-first matrix infrastructure
- benchmark harnesses and reproducibility
- packaging, CI, and observability
- partition equivalence work

Lower priority areas for external contributions:

- broad new feature claims without matching validation
- speculative GPU work before CPU correctness and sparse infrastructure are stronger
- widening the supported SPICE subset without contract updates

## Pull Request Expectations

- Keep claims and documentation honest.
- Add or update tests for behavior changes.
- If you change the supported subset or validation policy, update the relevant docs.
- Do not treat partitioned mode as production-ready unless equivalence is actually demonstrated.
- Prefer small, reviewable PRs over large mixed-purpose drops.

## Reporting Issues

Good bug reports include:

- the netlist
- the exact command used
- expected versus actual behavior
- whether `ngspice` agrees or disagrees
- any generated waveform or correctness artifacts

If the issue is about correctness, include the benchmark case and the relevant error metrics where possible.
