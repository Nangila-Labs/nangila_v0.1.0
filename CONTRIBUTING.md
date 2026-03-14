# Contributing

This repository is being organized around a narrow `v0.1` product surface.

Before contributing, read:
- [`README.md`](/Users/craigchirara/nangila/README.md)
- [`docs/releases/v0.1-support-matrix.md`](/Users/craigchirara/nangila/docs/releases/v0.1-support-matrix.md)
- [`docs/releases/v0.1-release.md`](/Users/craigchirara/nangila/docs/releases/v0.1-release.md)
- [`docs/releases/versioning-policy.md`](/Users/craigchirara/nangila/docs/releases/versioning-policy.md)

## Repo Structure

- `python/` contains the Python package
- `nangila-core/`, `nangila-hook/`, `nangila-cuda/` contain the Rust crates
- `tests/smoke/` is the default green baseline
- `tests/integration/` is opt-in validation
- `examples/experimental/` contains non-stable examples

## Local Validation

Run before opening a PR:

```bash
cargo test
python3 -m venv .venv
source .venv/bin/activate
python -m pip install maturin pytest
maturin develop --release -F python
python -m pytest -q
```

## Branching and Releases

- `main` is the active development branch
- release tags use the format `vX.Y.Z`
- maintenance branches should use `release/vX.Y` when needed
- future `v0.2` work should branch from `main` after `v0.1` support is stabilized

See [`docs/releases/versioning-policy.md`](/Users/craigchirara/nangila/docs/releases/versioning-policy.md) for the full policy.

## Scope Discipline

For `v0.1`, prefer changes that improve:
- DDP stability
- source-install reliability
- test/CI repeatability
- documentation clarity

If a change expands FSDP, multi-node, or low-level CUDA behavior, treat it as experimental unless the support docs are updated with explicit approval.
