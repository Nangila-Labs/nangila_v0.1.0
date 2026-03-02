# Nangila SPICE Production Roadmap

## Purpose

This document lays out a research and development plan to move Nangila SPICE from an in-progress prototype into a production-grade system.

For this project, "production mode" should mean:

- Numerically trustworthy on a clearly defined circuit class.
- Stable under long-running workloads and failure conditions.
- Faster than the reference baseline on representative workloads.
- Installable, testable, observable, and releasable by engineers other than the original author.

The immediate goal should not be "general-purpose SPICE replacement." The better target is a production-grade distributed transient simulator for an explicit v1 scope, followed by controlled expansion.

The formal Phase 0 v1 contract is defined in:

- [docs/v1-simulator-contract.md](/Users/craigchirara/nangila/nangila-spice/docs/v1-simulator-contract.md)

The execution plan for Phase 1 is defined in:

- [docs/phase-1-implementation-plan.md](/Users/craigchirara/nangila/nangila-spice/docs/phase-1-implementation-plan.md)

## Recommended Scope for v1

Define a narrow first production target:

- Transient simulation only.
- Flattened SPICE netlists.
- R/C/L, independent sources, Level-1 MOSFETs, and diodes.
- A declared benchmark family such as inverter chains, SRAM cells, RC ladders, ring oscillators, and selected ISCAS-derived circuits.

Explicitly mark the following as out of scope for v1 unless there is a concrete implementation and validation plan:

- Full ngspice compatibility.
- Advanced BSIM-class compact models.
- Arbitrary `.include` and `.lib` resolution trees.
- RF, AC, noise, and sensitivity analyses.
- Signoff-grade Monte Carlo accuracy.

## Locked v1 Scope

The Phase 0 scope for Nangila v1 is locked as follows.

### Product Positioning

Nangila v1 is not a broad SPICE replacement.

Nangila v1 is a high-performance transient simulator for a constrained SPICE subset, optimized for digital-heavy transistor-level and near-digital mixed-signal circuits, with correctness validated against ngspice.

### Benchmark Strategy

- `ngspice` is the primary correctness oracle.
- `ngspice` is the primary public speed baseline.
- `QSPICE` may be used later as a secondary competitive speed comparison, but not as the primary validation target.

### Supported Analysis

- Transient analysis only.

### Supported Netlist Model

- Flattened hierarchy only in the production execution path.
- A defined Nangila-supported SPICE subset, not full SPICE compatibility.

### Supported Device Class

- Resistors.
- Capacitors.
- Inductors.
- Independent voltage sources.
- Independent current sources.
- Diodes.
- Simple MOSFET support consistent with the validated v1 model set.

### Production Reference Path

- Single-node Rust solver is the authoritative production path for v1.
- Partitioned and distributed execution are not treated as production until they reproduce the single-node reference path within declared tolerances.
- GPU acceleration is optional and must be validated against the CPU reference path before being considered production-ready.

### Out of Scope for v1

- Full SPICE language compatibility.
- Broad `.include` and `.lib` ecosystem compatibility.
- Advanced compact models such as BSIM-class production model support.
- AC, noise, sensitivity, RF, and signoff analysis families.
- Precision analog, RF, and power-electronics-first product positioning.

## Supported Circuit Class

The supported circuit class for Nangila v1 is:

Digital-heavy transistor-level and near-digital mixed-signal transient circuits with sparse topology, repeated structure, and moderate analog behavior.

### Included Circuit Families

- CMOS inverter chains.
- NAND, NOR, AOI, OAI, and similar transistor-level logic blocks.
- Ring oscillators.
- Latches and flip-flops.
- Cross-coupled inverter structures.
- SRAM bitcells and small SRAM macros.
- RC and RLC interconnect networks.
- Synthesized transistor-level logic benchmark circuits.
- Near-digital support structures such as level shifters, wordline drivers, and similar blocks that remain within the validated device/model subset.

### Shared Characteristics of the Supported Class

These circuits are expected to have most of the following properties:

- Transient behavior is the primary analysis of interest.
- Large device counts relative to analog complexity.
- Sparse coupling and repeated local structure.
- Strong potential benefit from partitioning and sparse linear algebra.
- Numerical behavior that can be validated credibly against ngspice within declared tolerances.

### Excluded Circuit Families

- Precision analog blocks such as op-amps, bandgaps, and analog references.
- Analog filters requiring broader model fidelity or stronger frequency-domain guarantees.
- RF circuits.
- PLLs and clock/data recovery loops as a primary supported target.
- Power converters and switching power stages.
- Broad post-layout signoff decks that depend on advanced compact-model ecosystems and large external model libraries.

### v1 Circuit-Class Statement

Nangila v1 targets transient simulation of transistor-level digital and near-digital circuits, including SRAM-class feedback structures and passive interconnect networks, while excluding precision analog, RF, and power-centric circuit classes.

## Program Structure

### Phase 0: Product Definition

Objective:

- Freeze the v1 problem statement.
- Define supported analyses, models, input formats, accuracy targets, and deployment assumptions.

Deliverables:

- A v1 simulator specification.
- A benchmark suite definition.
- Explicit acceptance criteria for correctness, performance, and operability.

Exit criteria:

- Every production claim in the README maps to an implemented and testable capability.

### Phase 1: Correctness Foundation

Objective:

- Replace permissive integration checks with physics-based acceptance testing.

Work items:

- Build a golden comparison harness against ngspice for the supported circuit classes.
- Add waveform-based acceptance tests for:
  - RC ladders.
  - Inverter chains.
  - Ring oscillators.
  - SRAM cells.
  - Charge sharing cases.
  - Stiff nonlinear cases.
- Validate:
  - Full waveform shape.
  - Final node voltages.
  - Timestep counts.
  - Convergence behavior.
  - Error tolerances per node and per run.
- Tighten the Python end-to-end tests so one-point or all-zero outputs fail immediately.
- Split the official correctness program into:
  - a mandatory per-change gate
  - an extended nightly or manual gate for larger combinational decks that are valid but too expensive for every PR

Exit criteria:

- Single-partition simulations match the oracle within declared tolerances across the mandatory and extended v1 correctness gates.

### Phase 2: Honest Single-Node Solver

Objective:

- Make the Rust single-node path the authoritative production engine.

Work items:

- Remove synthetic and placeholder behavior from production-facing execution paths.
- Decide the production solver strategy:
  - Recommended: built-in Rust solver as the primary engine.
  - ngspice used as an oracle and validation harness, not as the main runtime.
- Ensure the CLI always emits real transient waveforms for production commands.
- Standardize run artifacts:
  - Full waveforms.
  - Solver statistics.
  - Convergence logs.
  - Run manifest with solver settings and build metadata.

Exit criteria:

- The CLI can run a real transient job end-to-end without synthetic fallback.

### Phase 3: Sparse-First Matrix Infrastructure

Objective:

- Replace dense internal matrix handling with sparse-native infrastructure suitable for scale.

Work items:

- Redesign MNA assembly around sparse data structures from the start.
- Split symbolic structure from numeric values so Newton iterations can reuse topology.
- Add:
  - Fill-in measurement.
  - Reordering support.
  - Residual norms.
  - Factorization reuse.
  - Solver diagnostics.
- Build a strong CPU sparse path before treating GPU acceleration as production-ready.

Exit criteria:

- Sparse CPU solve becomes the default path and outperforms the dense baseline on medium-sized circuits.

### Phase 4: Real Partitioned Execution

Objective:

- Finish the distributed simulation design rather than only partitioning the netlist.

Work items:

- Implement actual boundary exchange between partitions.
- Define partition synchronization semantics.
- Implement rollback and replay behavior with deterministic recovery.
- Add equivalence tests:
  - Single-partition vs multi-partition on the same circuits.
  - Partition count sweeps with bounded numerical divergence.
- Measure the impact of cut quality on convergence, rollback rate, and wall-clock time.

Exit criteria:

- Multi-partition runs reproduce single-node results within declared tolerances and fail predictably when prediction quality degrades.

### Phase 5: Measured PVT Pipeline

Objective:

- Convert the PVT flow from a mostly conceptual framework into a measured production subsystem.

Work items:

- Define when delta-mode is valid.
- Measure error versus full simulation on each benchmark family.
- Add reproducible artifact capture for:
  - Golden corner waveforms.
  - Delta approximations.
  - Full-corner comparison results.
- Replace estimated speedup claims with benchmarked measurements and stored evidence.

Exit criteria:

- PVT performance and accuracy claims are based on recorded experiments with reproducible runs.

### Phase 6: GPU Validation and Acceleration

Objective:

- Move CUDA from "backend exists" to "validated production accelerator."

Work items:

- Validate CPU sparse vs GPU sparse parity on shared matrices and transient workloads.
- Instrument:
  - Transfer overhead.
  - Factorization reuse rates.
  - Kernel utilization.
  - Solver fallback behavior.
- Define workload thresholds where GPU is expected to help.
- Add automatic fallback to CPU when GPU setup, memory pressure, or solve quality makes GPU unsuitable.

Exit criteria:

- GPU acceleration is measurably faster on declared workload classes and safely falls back when needed.

### Phase 7: Packaging, Operations, and Release Engineering

Objective:

- Make the project installable, observable, and releasable by a team.

Work items:

- Fix Python packaging and declare runtime dependencies explicitly.
- Separate runtime, development, benchmarking, and optional visualization dependencies.
- Add reproducible build instructions for:
  - Rust toolchain.
  - CUDA toolchain.
  - Benchmark assets.
- Add CI for:
  - Python unit and integration tests.
  - Rust unit and integration tests.
  - Oracle comparison tests.
  - Benchmark smoke tests.
  - Package installation checks.
- Add structured logging, metrics, failure summaries, and artifact retention policies.

Exit criteria:

- A fresh clone can be installed and validated without manual tribal knowledge.

## Research Tracks

### 1. Numerical Methods

Focus areas:

- Adaptive timestep control.
- Stiffness detection.
- Partition convergence criteria.
- Rollback economics.
- Stability tradeoffs between Trapezoidal and Backward Euler.

### 2. Partitioning Quality

Focus areas:

- Partition cost models tied to convergence, not only device balance.
- Boundary minimization vs solver stability tradeoffs.
- Detection of structures that should never be split.
- Feedback-aware partition scoring.

### 3. Sparse Linear Algebra

Focus areas:

- Symbolic reuse.
- Preconditioning.
- Ordering strategies.
- Factorization caching.
- CPU/GPU crossover thresholds.

### 4. Model Fidelity

Focus areas:

- When Level-1 MOSFET models stop being sufficient.
- How to stage improved compact model support without destabilizing the solver.
- Validation strategy for each added model class.

### 5. Distributed Runtime Design

Focus areas:

- Predictive partitioning vs strict synchronization.
- Latency tolerance.
- Replay determinism.
- Cluster scheduling and straggler handling.

## Suggested 12-Month Sequence

### Months 1-2

- Freeze v1 scope.
- Build the oracle comparison harness.
- Replace fake-success integration tests with correctness-based acceptance tests.
- Remove obviously synthetic success paths from production commands.

### Months 2-4

- Refactor the single-node solver into sparse-first infrastructure.
- Standardize waveform and run artifact outputs.
- Establish the Rust single-node engine as the production reference.

### Months 4-6

- Implement real partition communication and rollback flow.
- Add single-vs-multi partition equivalence tests.
- Start collecting convergence and partition quality metrics.

### Months 6-8

- Turn PVT into a measured subsystem.
- Add delta-mode validation and guardrails.
- Build reproducible benchmark datasets and stored results.

### Months 8-10

- Validate and tune GPU execution.
- Add parity tests between CPU sparse and GPU sparse paths.
- Measure workload classes where GPU is beneficial.

### Months 10-12

- Finish packaging, CI/CD, observability, and release hardening.
- Prepare a release candidate with benchmark evidence and operational documentation.

## Immediate Priorities

The next concrete priorities should be:

1. Rewrite the end-to-end tests so obviously wrong waveforms fail.
2. Declare the current production reference path as single-partition Rust only.
3. Replace dense MNA internals with sparse-first solver infrastructure.
4. Wire real boundary exchange before presenting partitioned transient simulation as operational.
5. Remove or quarantine synthetic benchmark and PVT paths from production-facing commands.

## Suggested Milestones

### Milestone A: Trustworthy Single-Node Engine

- Oracle-backed correctness suite passes.
- Full waveforms emitted reliably.
- Sparse CPU solver established.

### Milestone B: Trustworthy Partitioned Runtime

- Boundary exchange implemented.
- Multi-partition equivalence tests passing.
- Rollback and replay deterministic.

### Milestone C: Measured Acceleration

- PVT measurements reproducible.
- GPU parity and speedup validated.
- Performance claims backed by stored benchmark artifacts.

### Milestone D: Production Release Candidate

- Packaging fixed.
- CI and release gating active.
- Operational documentation complete.

## Final Recommendation

The correct sequencing for Nangila SPICE is:

- correctness first,
- then honest single-node execution,
- then distributed partitioned runtime,
- then performance acceleration,
- then release engineering.

If the project skips that order, it risks optimizing and marketing a simulator whose most important production properties have not yet been proven.
