# Nangila v1 Simulator Contract

## Purpose

This document completes Phase 0 by defining the locked v1 scope for Nangila SPICE:

- the supported SPICE subset
- the device and model contract
- the correctness contract against ngspice
- the official benchmark suites
- the release criteria for calling v1 production-ready

This is a product contract for v1. It is not a claim that every item is already implemented in the current codebase.

## Product Definition

Nangila v1 is a high-performance transient simulator for a constrained SPICE subset, optimized for digital-heavy transistor-level and near-digital circuits.

Nangila v1 is validated against ngspice on the supported subset.

Nangila v1 is not a general-purpose SPICE replacement in v1.

## Analysis Scope

Supported:

- Transient analysis only.

Not supported in v1:

- AC analysis.
- DC sweep analysis as a first-class user workflow.
- Noise analysis.
- Sensitivity analysis.
- RF analysis.
- Monte Carlo signoff flows.

## Supported Circuit Class

Nangila v1 supports:

- transistor-level digital logic
- near-digital mixed-signal support structures
- passive interconnect networks
- SRAM-class feedback structures

Included circuit families:

- inverter chains
- NAND, NOR, AOI, OAI, and similar CMOS logic blocks
- latches and flip-flops
- SRAM bitcells and small SRAM macros
- RC and RLC interconnect networks
- synthesized transistor-level logic blocks
- near-digital support blocks such as level shifters and wordline drivers

Explicitly excluded from the v1 production contract:

- precision analog blocks such as op-amps, references, and bandgaps
- RF circuits
- power converters and switching power stages
- broad post-layout signoff decks with complex external model ecosystems
- circuits whose credibility depends on advanced compact models not included in the v1 device contract
- autonomous oscillator classes as an official release-gating benchmark family

## Supported SPICE Subset

### General Syntax

Supported:

- title line
- `*` comment lines
- `+` line continuations
- case-insensitive keywords and model names
- engineering suffixes:
  - `f`, `p`, `n`, `u`, `m`, `k`, `meg`, `g`, `t`
- ground nodes:
  - `0`
  - `gnd`

### Supported Elements

The official v1 netlist subset includes:

- `Rname n1 n2 value`
- `Cname n1 n2 value`
- `Lname n1 n2 value`
- `Vname nplus nminus DC value`
- `Vname nplus nminus PULSE(v1 v2 td tr tf pw per)`
- `Iname nplus nminus DC value`
- `Dname anode cathode model`
- `Mname drain gate source bulk model [W=...] [L=...]`
- `Xname ... subckt_name`

### Supported Directives

Supported in the v1 contract:

- `.END`
- `.SUBCKT ...`
- `.ENDS`
- `.PARAM name=value`
- `.MODEL name type param=value ...`

Conditionally supported:

- `.TRAN`
  - v1 runtime configuration may be provided by CLI or API settings.
  - when both are present, runtime configuration is authoritative.

### Parameter Rules

Supported parameter behavior:

- scalar numeric literals
- scalar parameter substitution via `.PARAM`
- brace-wrapped scalar parameter references such as `{vdd_val}`

Not supported in v1:

- general arithmetic expressions in parameters
- nested parameter expressions
- behavioral expressions

### Subcircuit Rules

Supported:

- hierarchical `.SUBCKT` definitions
- recursive subcircuit instantiation
- flattening into the production execution path

Not supported in v1:

- unresolved external library references through `.INCLUDE` or `.LIB`
- production reliance on undeclared external model trees

### Explicitly Unsupported Netlist Features

The following are out of the v1 supported subset unless later added by explicit contract revision:

- `.INCLUDE`
- `.LIB`
- behavioral sources
- controlled sources as a production promise
- transmission lines
- Verilog-A and arbitrary external model plugins as a production requirement
- unsupported waveform source types beyond the defined subset
- general-purpose parameter algebra
- broad `.IC` semantics as an official compatibility claim

## Device and Model Contract

### Passive Devices

Supported:

- linear resistor
- linear capacitor
- linear inductor

These are part of the authoritative v1 production contract.

### Independent Sources

Supported:

- DC voltage sources
- DC current sources
- `PULSE(...)` voltage sources for transient digital stimulus

Not part of the v1 production contract:

- general arbitrary waveform sources
- behavioral source expressions
- complex source scripting semantics

### Diode Contract

Nangila v1 supports an ideal Shockley-style diode model.

Supported diode model behavior:

- exponential I-V conduction
- linearized Norton stamping in Newton iterations
- model-card parameters:
  - `IS`
  - `N`

Not promised in v1:

- reverse breakdown modeling
- junction capacitance
- diffusion capacitance
- series resistance
- charge storage
- full temperature-dependent production diode fidelity

### MOSFET Contract

Nangila v1 supports simple transistor-level MOS modeling suitable for digital-heavy transient workloads.

Supported MOS behavior in v1:

- NMOS and PMOS operation
- Level-1/Shichman-Hodges style conduction behavior
- operating regions:
  - cutoff
  - linear
  - saturation
- instance parameters:
  - `W`
  - `L`
- model-card parameters:
  - `VTO`
  - `U0`
  - `TOX`

Allowed v1 usage assumption:

- the bulk terminal may be present in the syntax
- credible v1 behavior assumes bulk is tied to an appropriate supply rail

Not promised in v1:

- BSIM-class compact model fidelity
- body effect as a release-gating correctness requirement
- junction capacitances
- gate capacitances and full charge conservation
- velocity saturation
- subthreshold accuracy as a signoff claim
- DIBL or short-channel production fidelity
- advanced PMOS/NMOS model-card ecosystems

## Correctness Contract Against ngspice

ngspice is the authoritative correctness oracle for the supported v1 subset.

All correctness comparisons must use:

- the same supported netlist subset
- model cards limited to the Nangila v1 contract
- aligned source definitions
- aligned transient runtime parameters

### Accuracy Metrics

The official v1 correctness metrics are:

- maximum absolute node-voltage error
- RMS node-voltage error
- final-time node-voltage error
- edge timing error at the 50% VDD crossing where applicable
- successful completion without fatal non-convergence

### Official Tolerances

#### Passive and Mostly Linear Circuits

For passive or mostly linear benchmark circuits:

- max absolute voltage error: `<= max(1 mV, 0.1% of VDD)`
- RMS voltage error: `<= max(0.5 mV, 0.05% of VDD)`
- final-time node-voltage error: `<= max(1 mV, 0.1% of VDD)`

#### Nonlinear Digital and SRAM-Class Circuits

For digital-heavy nonlinear benchmark circuits:

- max absolute voltage error: `<= max(25 mV, 2% of VDD)`
- RMS voltage error: `<= max(10 mV, 1% of VDD)`
- final-time node-voltage error: `<= max(10 mV, 1% of VDD)`
- 50% VDD edge timing error: `<= max(10 ps, 5% of the reference edge interval)`

### Convergence Expectations

To count as a correctness pass:

- the simulation must complete the benchmark run
- the solver must not terminate with fatal singular-matrix or non-convergence failure
- no `NaN` or `Inf` values may appear in output waveforms
- any damping or timestep-reduction strategy must complete without manual intervention

If Nangila completes only by falling back to a synthetic or placeholder path, the benchmark is an automatic fail.

## Official Benchmark Suites

### Correctness Gate Suite

The official v1 correctness validation program is split into two tiers:

- a mandatory correctness gate for per-change local and CI validation
- an extended correctness gate for nightly or manual validation of larger circuits

Both tiers are part of the official v1 correctness suite.

#### Mandatory Correctness Gate

The mandatory correctness gate is:

- [`tests/spice_examples/simple_rc.sp`](/Users/craigchirara/nangila/nangila-spice/tests/spice_examples/simple_rc.sp)
- [`benchmarks/reference_circuits/inverter.sp`](/Users/craigchirara/nangila/nangila-spice/benchmarks/reference_circuits/inverter.sp)
- [`benchmarks/reference_circuits/sram_6t.sp`](/Users/craigchirara/nangila/nangila-spice/benchmarks/reference_circuits/sram_6t.sp)
- [`benchmarks/reference_circuits/c17_synth.sp`](/Users/craigchirara/nangila/nangila-spice/benchmarks/reference_circuits/c17_synth.sp)
- [`benchmarks/reference_circuits/c17_full.sp`](/Users/craigchirara/nangila/nangila-spice/benchmarks/reference_circuits/c17_full.sp)
- [`benchmarks/auto_synth/c432.sp`](/Users/craigchirara/nangila/nangila-spice/benchmarks/auto_synth/c432.sp)
- [`benchmarks/auto_synth/s27.sp`](/Users/craigchirara/nangila/nangila-spice/benchmarks/auto_synth/s27.sp)
- [`benchmarks/auto_synth/s382.sp`](/Users/craigchirara/nangila/nangila-spice/benchmarks/auto_synth/s382.sp)
- [`benchmarks/auto_synth/s641.sp`](/Users/craigchirara/nangila/nangila-spice/benchmarks/auto_synth/s641.sp)

These are the official pass/fail circuits for the required per-change correctness gate.

#### Extended Correctness Gate

The extended correctness gate is:

- [`benchmarks/auto_synth/c880.sp`](/Users/craigchirara/nangila/nangila-spice/benchmarks/auto_synth/c880.sp)
- [`benchmarks/auto_synth/c1355.sp`](/Users/craigchirara/nangila/nangila-spice/benchmarks/auto_synth/c1355.sp)
- [`benchmarks/auto_synth/c1908.sp`](/Users/craigchirara/nangila/nangila-spice/benchmarks/auto_synth/c1908.sp)

These are official v1 correctness cases, but they are intentionally extended-only in Phase 1 because they exceed the per-change CI/runtime budget.

Promotion from extended to mandatory requires:

- repeated stable passes
- runtime that fits the per-change CI budget
- artifact history showing comfortable margin against the published tolerances

### Performance Gate Suite

The official v1 performance gate suite is:

- [`benchmarks/auto_synth/c2670.sp`](/Users/craigchirara/nangila/nangila-spice/benchmarks/auto_synth/c2670.sp)
- [`benchmarks/auto_synth/c3540.sp`](/Users/craigchirara/nangila/nangila-spice/benchmarks/auto_synth/c3540.sp)
- [`benchmarks/auto_synth/c5315.sp`](/Users/craigchirara/nangila/nangila-spice/benchmarks/auto_synth/c5315.sp)
- [`benchmarks/auto_synth/c6288.sp`](/Users/craigchirara/nangila/nangila-spice/benchmarks/auto_synth/c6288.sp)
- [`benchmarks/auto_synth/c7552.sp`](/Users/craigchirara/nangila/nangila-spice/benchmarks/auto_synth/c7552.sp)
- [`benchmarks/auto_synth/s1238.sp`](/Users/craigchirara/nangila/nangila-spice/benchmarks/auto_synth/s1238.sp)
- [`benchmarks/auto_synth/s5378.sp`](/Users/craigchirara/nangila/nangila-spice/benchmarks/auto_synth/s5378.sp)
- [`benchmarks/auto_synth/s9234.sp`](/Users/craigchirara/nangila/nangila-spice/benchmarks/auto_synth/s9234.sp)
- [`benchmarks/auto_synth/s13207.sp`](/Users/craigchirara/nangila/nangila-spice/benchmarks/auto_synth/s13207.sp)

These are the official benchmark circuits for speed comparison against ngspice on the supported subset.

## Release Criteria

Nangila v1 is production-ready only if all of the following are true.

### Correctness

- The mandatory and extended correctness gates both pass against ngspice within the official tolerances.
- Production commands emit real waveform outputs rather than placeholders or synthetic stand-ins.
- The single-node Rust execution path is the authoritative production path and passes the full correctness suite.

### Performance

- On the official performance gate suite, Nangila demonstrates a median speedup of at least `2.0x` versus ngspice on the reference hardware and benchmark configuration.
- At least `75%` of performance-gate benchmarks must run at `>= 1.25x` ngspice speed.

### Reliability

- No benchmark in the mandatory or extended correctness gates may require manual intervention to complete.
- No production command may silently fall back to fake or synthetic waveform generation.
- Repeated runs on the same benchmark and configuration must be numerically stable within the published tolerances.

### Operability

- A fresh clone can be built and run from documented instructions.
- Runtime dependencies are declared explicitly.
- CI runs the correctness gate suite and release checks automatically.
- Solver output includes waveform artifacts, run metadata, and failure diagnostics.

### Partitioned and GPU Modes

- Partitioned mode is not part of the v1 production claim unless it reproduces the single-node reference path within the correctness tolerances on the correctness gate suite.
- GPU mode is not part of the v1 production claim unless it matches the CPU reference path within the same tolerances and improves runtime on the declared performance suite.

## Phase 0 Completion Statement

Phase 0 is complete when this contract is accepted as the official v1 definition.

After acceptance, all later work should be judged against this contract rather than against broad, unstated SPICE-tool ambitions.
