# Nangila SPICE

**Predictive-Partitioned Circuit Simulator** — A Nangila Framework application for analog chip verification.

## Overview

Nangila SPICE applies the [Nangila Framework](../docs/WHITEPAPER_v2.md) (Predictive-Residual Decomposition) to break the "Communication Wall" in distributed SPICE simulation. It enables **linear scaling** across thousands of cores by replacing global synchronization with local prediction.

## Architecture

```
┌─────────────────────────────────────────────┐
│            Python Orchestrator              │
│  parser → partitioner → orchestrator        │
└──────────┬──────────┬──────────┬────────────┘
           │          │          │
     ┌─────▼───┐ ┌────▼────┐ ┌──▼───────┐
     │ Node 0  │ │ Node 1  │ │ Node k-1 │  ← Rust
     │ (ghost) │ │ (ghost) │ │ (ghost)  │
     │ (solve) │ │ (solve) │ │ (solve)  │
     └────┬────┘ └────┬────┘ └────┬─────┘
          │           │           │
          └─────Residuals─────────┘  ← Nangila Fabric
```

## Project Structure

```
nangila-spice/
├── Cargo.toml                     # Rust workspace root
├── nangila-node/                  # Rust: Solver node binary
│   └── src/
│       ├── main.rs                # CLI entry point
│       ├── ghost.rs               # Ghost node buffer + predictor
│       ├── comm.rs                # Inter-partition communication
│       ├── predictor.rs           # Prediction strategies
│       └── solver.rs              # Local matrix solver
├── python/
│   └── nangila_spice/             # Python: Frontend orchestrator
│       ├── __init__.py
│       ├── parser.py              # SPICE netlist parser
│       ├── partitioner.py         # Hypergraph partitioner
│       ├── orchestrator.py        # Hardware discovery + launcher
│       └── merger.py              # Waveform stitching
├── benchmarks/
│   └── reference_circuits/        # Golden test netlists
└── docs/                          # Design documents
```

## Quick Start

```bash
# Build the Rust solver node
cd nangila-spice
cargo build --release

# Parse a netlist (Python)
python -c "
from nangila_spice.parser import parse_netlist
nl = parse_netlist('benchmarks/reference_circuits/inverter.sp')
print(f'{nl.num_devices} devices, {nl.num_nodes} nodes')
"
```

## Key Features

| Feature | Status | Sprint |
|---------|--------|--------|
| Netlist Parser | 🔨 In Progress | Sprint 1 |
| Graph Partitioner | 📋 Planned | Sprint 2 |
| Ngspice FFI Bridge | 📋 Planned | Sprint 3 |
| Predictive Residuals | 📋 Planned | Sprint 6 |
| .nz Waveform Compression | 📋 Planned | Sprint 7 |
| GPU Native Solver | 📋 Planned | Sprint 9 |

## References

- [V1 Design Spec](../docs/NANGILA_SPICE_ADDENDUM.md)
- [Nangila Whitepaper](../docs/WHITEPAPER_v2.md)
