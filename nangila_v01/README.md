# Nangila v0.1

This directory defines the `v0.1` product boundary.

It is intentionally a release snapshot and coordination surface, not a fork of
the runtime source tree. The actual implementation remains in the main
workspace so Rust, Python, CI, and packaging stay consistent.

## Source of Truth

`v0.1` is composed from these paths:

- `nangila-core/`
- `nangila-cuda/`
- `nangila-hook/`
- `python/nangila/`
- `tests/smoke/`
- `tests/integration/test_cuda_single_gpu_smoke.py`
- `tests/integration/test_ddp_correctness.py`
- `docs/releases/`

## Stable Product Surface

`v0.1` means:

- source install with `maturin`
- Rust core compression logic
- PyTorch `DistributedDataParallel` integration
- calibration with `Sculptor`
- documented non-CUDA default build

Everything outside that surface remains experimental for `v0.1`.

## Release Gates

Required before tagging `v0.1.0`:

```bash
make release-check
bash scripts/cuda_single_gpu_smoke.sh
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 tests/integration/test_ddp_correctness.py
```

## Why This Folder Exists

This folder lets the repository say "this is Nangila v0.1" without duplicating
the code into a second package tree such as `python/nangila_v01/` or a parallel
Cargo workspace. That keeps maintenance branches and tags clean while avoiding
version drift inside one repository.

## Commit Scope Check

Use the release-scope checker before cutting the `v0.1.0` commit:

```bash
python scripts/check_v01_commit_scope.py
```

You can also pass an explicit file list:

```bash
python scripts/check_v01_commit_scope.py README.md python/nangila/ddp.py
```
