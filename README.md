# Nangila

Nangila is a Rust-backed gradient compression package for PyTorch distributed training.

`v0.1` is intentionally narrow. The supported product surface is:
- PyTorch `DistributedDataParallel` integration
- Python package install from source with `maturin`
- Rust core compression/state logic
- CPU-safe default build

Everything outside that surface, including FSDP and low-level CUDA helper APIs, is considered experimental in `v0.1`.

## Status

Nangila is pre-production software. The current goal is a reliable `v0.1`, not the full long-term framework vision.

- Stable surface: DDP hook, calibration with `Sculptor`, mask-based hook creation, basic stats
- Experimental surface: FSDP integration, low-level CUDA helper APIs, multi-node qualification, broad benchmark claims

See [`docs/releases/v0.1-support-matrix.md`](docs/releases/v0.1-support-matrix.md) for the explicit support contract and [`docs/releases/v0.1-release.md`](docs/releases/v0.1-release.md) for release criteria.

## Installation

### Supported source install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install maturin pytest
maturin develop --release -F python
```

This default build does not require CUDA.

### Experimental CUDA build

```bash
maturin develop --release -F python -F cuda
```

CUDA builds are excluded from the `v0.1` stable support contract.

## Quick Start

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from nangila.ddp import register_nangila_hook

dist.init_process_group("nccl")
model = DDP(MyModel().cuda())

hook = register_nangila_hook(
    model,
    threshold=0.95,
    warmup_steps=100,
)

for batch in dataloader:
    loss = model(batch).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    hook.step()
```

## Calibration Workflow

```python
from nangila import Sculptor, NangilaConfig, NangilaHook

sculptor = Sculptor(threshold=0.95)

for layer_id, param in enumerate(model.parameters()):
    if param.grad is not None:
        sculptor.record(layer_id, param.grad.detach().flatten().cpu().numpy())

mask = sculptor.generate_mask()
hook = NangilaHook(mask_bytes=mask, config=NangilaConfig.conservative())
```

## Public API for `v0.1`

- `nangila.NangilaConfig`
- `nangila.Sculptor`
- `nangila.NangilaHook`
- `nangila.SyncMode`
- `nangila.CompressorType`
- `nangila.cuda_available`
- `nangila.ddp.NangilaDDPHook`
- `nangila.ddp.register_nangila_hook`

Experimental APIs are available under `nangila.experimental`.

## Advanced DDP Options

`register_nangila_hook()` still accepts non-default compressor-selection options for compatibility:
- `compressor_type`
- `dgc_sparsity`
- `power_sgd_rank`

Those knobs are not part of the `v0.1` stable support contract. The stable DDP path for `v0.1` is the default prediction-residual configuration with the documented `threshold`, `warmup_steps`, `prefer_cpp`, and `sync_mode` controls.

## Repository Layout

- [`docs/releases/`](docs/releases/) contains support contracts, release criteria, and release-process templates.
- [`docs/strategy/`](docs/strategy/) contains long-range product and business strategy material.
- [`docs/archive/`](docs/archive/) contains historical research, analysis documents, and archived logs that are not part of the active `v0.1` contract.
- [`examples/experimental/`](examples/experimental/) contains non-stable examples and prototypes.
- [`tests/smoke/`](tests/smoke/) contains the default green baseline for `v0.1`.
- [`tests/integration/`](tests/integration/) contains opt-in distributed and GPU-heavy validation.

## Development

### Rust tests

```bash
cargo test
```

### Python smoke tests

```bash
make release-check
```

### Integration tests

The distributed and GPU-heavy test scripts are excluded from the default test run. To collect them:

```bash
NANGILA_RUN_INTEGRATION=1 python -m pytest
```

For a 1-GPU CUDA smoke pass on a GPU node:

```bash
# load cluster CUDA toolkit first if needed, then:
bash scripts/cuda_single_gpu_smoke.sh
```

The script now auto-resolves `CUDA_HOME`/`CUDA_PATH`, sets `LD_LIBRARY_PATH` for `libcudart`,
and fails early if your Slurm job is clearly under-provisioned on memory.

That is a pre-qualification step only. `v0.1` DDP completion still requires a 2-GPU validation run.
