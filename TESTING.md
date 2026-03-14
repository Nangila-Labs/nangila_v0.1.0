# Nangila Testing Guide

This file documents the active `v0.1` test layout and local validation flow.

## Default `v0.1` baseline

Use the documented release-check path for the stable local baseline:

```bash
make release-check
```

That runs:
- `cargo test`
- a local `.venv` setup
- `maturin develop --release -F python`
- `python -m pytest -q`

## Test layout

- `tests/smoke/`: the default green baseline for `v0.1`
- `tests/integration/`: opt-in distributed, GPU-heavy, and benchmark-oriented validation

Pytest is configured to skip integration collection unless `NANGILA_RUN_INTEGRATION=1` is set.

## Manual commands

### Rust workspace tests

```bash
cargo test
```

### Python smoke tests

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install maturin pytest
./.venv/bin/maturin develop --release -F python
./.venv/bin/python -m pytest -q
```

### Integration collection

```bash
NANGILA_RUN_INTEGRATION=1 python -m pytest
```

### GPU and distributed validation

Examples live under [`tests/integration/`](/Users/craigchirara/nangila/tests/integration/), including:
- [`tests/integration/test_cuda_single_gpu_smoke.py`](/Users/craigchirara/nangila/tests/integration/test_cuda_single_gpu_smoke.py)
- [`tests/integration/gpu_test_suite.sh`](/Users/craigchirara/nangila/tests/integration/gpu_test_suite.sh)
- [`tests/integration/test_ddp_correctness.py`](/Users/craigchirara/nangila/tests/integration/test_ddp_correctness.py)
- [`tests/integration/test_stress.py`](/Users/craigchirara/nangila/tests/integration/test_stress.py)

These are not part of the default `v0.1` support baseline.

### Single-GPU CUDA smoke

If you only have a 1-GPU allocation, you can still run a CUDA pre-qualification pass:

```bash
# load cluster CUDA toolkit first if needed, then:
bash scripts/cuda_single_gpu_smoke.sh
```

This runner auto-resolves `CUDA_HOME`/`CUDA_PATH`, sets `LD_LIBRARY_PATH` for CUDA runtime
loading, and performs a Slurm memory sanity check before invoking `torchrun`.

That validates CUDA build/import, single-rank NCCL setup, and stable DDP hook registration on one GPU.
It does not replace the 2-GPU DDP correctness gate for `v0.1`.

## Related docs

- [`README.md`](/Users/craigchirara/nangila/README.md)
- [`docs/releases/v0.1-support-matrix.md`](/Users/craigchirara/nangila/docs/releases/v0.1-support-matrix.md)
- [`docs/releases/release-checklist.md`](/Users/craigchirara/nangila/docs/releases/release-checklist.md)
