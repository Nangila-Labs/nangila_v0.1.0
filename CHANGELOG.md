# Changelog

All notable user-visible changes for Nangila releases should be recorded here.

Pre-`v0.1` implementation notes were archived to [`docs/archive/pre-v0.1-changelog-report.md`](/Users/craigchirara/nangila/docs/archive/pre-v0.1-changelog-report.md).

## v0.1.0 - 2026-03-14

### Added
- Initial `v0.1` release surface for PyTorch `DistributedDataParallel`.
- Release process documents, support matrix, and contributor/versioning guidance for tagged GitHub releases.
- A local `make release-check` workflow that runs the documented Rust and Python smoke validation steps.

### Changed
- Narrowed the public Python API so stable `v0.1` imports focus on the DDP path.
- Moved FSDP and low-level CUDA helpers under `nangila.experimental`.
- Reorganized the repository around `docs/releases/`, `docs/strategy/`, `docs/archive/`, `tests/smoke/`, `tests/integration/`, and `examples/experimental/`.

### Fixed
- Default Python builds no longer require CUDA for the supported `v0.1` install path.
- Pytest collection now keeps GPU-heavy and distributed validation out of the default smoke run.
- Workspace test coverage now includes a passing `cargo test` baseline and a passing Python smoke baseline.
- The Python DDP all-gather fallback now decodes gathered packets correctly, eliminating stale-packet parity drift in the 2-GPU validation path.

### Validated
- The 1-GPU CUDA smoke gate now passes on supported NVIDIA CUDA hosts.
- The required 2-GPU `DistributedDataParallel` correctness gate now passes on a Linux CUDA setup.

### Support
- `v0.1` supports source installs with `maturin`, the Rust core, calibration with `Sculptor`, and the documented DDP hook path.
- CUDA builds, FSDP, and low-level GPU helper APIs remain outside the stable `v0.1` support contract.

### Experimental
- `nangila.experimental` remains the home for beta FSDP integration and low-level CUDA helper APIs.
