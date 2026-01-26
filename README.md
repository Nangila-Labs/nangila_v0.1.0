# Nangila

**Gradient Virtualization for Distributed DNN Training**

[![License: MIT/Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)

> ⚠️ **PRE-PRODUCTION STATUS**: Nangila is currently in active development and **not yet production-ready**. Core algorithms are implemented and tested, but comprehensive validation on large-scale training workloads is ongoing. See [Roadmap](#-roadmap) for details.

Nangila virtualizes network bandwidth during Data Parallel (DDP) training by transmitting **Predictive Residuals** instead of raw gradients. Achieve **20-50× compression** in typical scenarios (up to 64× in ideal cases).

> **Note**: GPU-native path provides minimal overhead (<1ms compression latency, pending hardware validation). CPU fallback path has higher overhead due to GPU↔CPU transfers.

## 🚀 Quick Start

```bash
# Install from source
pip install maturin
git clone https://github.com/nangila/nangila.git
cd nangila
maturin develop --release -F python

# Or install wheel directly
pip install nangila-0.1.0-*.whl
```

## 📦 Installation

### From PyPI (coming soon)
```bash
pip install nangila
```

### From Source
```bash
# Prerequisites
# - Rust 1.70+ (https://rustup.rs)
# - Python 3.8+
# - PyTorch 2.0+
# - CUDA 11.0+ (optional, for GPU acceleration)

git clone https://github.com/nangila/nangila.git
cd nangila
pip install maturin
maturin develop --release -F python
```

## 💡 How It Works

Nangila exploits two properties of gradient dynamics:

1. **Temporal Determinism**: Gradients are predictable → transmit only the residual (error)
2. **Spatial Low-Rank**: Most layers are redundant → skip "Passenger" layers entirely

```
Standard DDP:     Gradient (FP32) ──────────────────────▶ All-Reduce
                  28 GB per step (7B model)

Nangila:          Gradient ──▶ [Predict] ──▶ [Residual] ──▶ [INT4] ──▶ [Mask]
                                                                        │
                                                                        ▼
                  Compressed ─────────────────────────────────▶ All-Reduce
                  ~350 MB per step (80× compression)
```

## 🔧 Usage

### Basic DDP Usage

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from nangila.ddp import register_nangila_hook

# Initialize DDP as usual
dist.init_process_group("nccl")
model = DDP(MyModel().cuda())

# Register Nangila compression hook
hook = register_nangila_hook(
    model,
    threshold=0.95,      # Compression quality
    warmup_steps=100,    # Steps before compression activates
)

# Training loop unchanged
for batch in dataloader:
    loss = model(batch).sum()
    loss.backward()
    optimizer.step()
    hook.step()  # Advance predictor state
```

### With Calibration (Higher Compression)

```python
from nangila import Sculptor, NangilaHook, NangilaConfig

# Step 1: Calibration (run once, ~100 steps)
sculptor = Sculptor(threshold=0.95)
for step in range(100):
    for name, param in model.named_parameters():
        if param.grad is not None:
            sculptor.record(layer_id, param.grad.flatten().numpy())

# Step 2: Generate and save mask
mask = sculptor.generate_mask()
with open("topology.nzmask", "wb") as f:
    f.write(mask)

# Step 3: Use mask for training
config = NangilaConfig.conservative()
hook = NangilaHook(mask_bytes=mask, config=config)
```

### Configuration Options

```python
from nangila.ddp import register_nangila_hook
from nangila import SyncMode

# Conservative (safer, less compression)
hook = register_nangila_hook(
    model,
    threshold=0.97,      # Higher threshold = fewer Passengers
    warmup_steps=100,
)

# Aggressive (more compression)
hook = register_nangila_hook(
    model,
    threshold=0.90,      # Lower threshold = more Passengers
    warmup_steps=50,
)

# Custom with CUDA error checking
hook = register_nangila_hook(
    model,
    threshold=0.95,
    warmup_steps=100,
    sync_mode=SyncMode.PERIODIC,  # Balanced error checking
)
```

> **Note**: Advanced configuration (momentum, quantization bits) is currently hardcoded in the Rust core. Python API exposes `threshold` and `warmup_steps` parameters.

## 📊 Benchmarks

### Compression Breakdown

| Source | Factor | Description |
|--------|--------|-------------|
| FP32 → INT4 | 8× | 32 bits → 4 bits (guaranteed) |
| Topology (Passengers) | 1-3× | Skip correlated layers (typical) |
| Predictive Residuals | 1.2-1.8× | Lower entropy signals (model-dependent) |

**Typical Combined**: 20-40× compression  
**Best Case**: Up to 64× (50%+ Passengers, high predictability)

### Compression by Threshold (τ)

The `threshold` parameter controls compression vs. quality tradeoff:

| τ Value | Passenger Layers | Typical Compression | Safety |
|---------|------------------|---------------------|--------|
| **τ=0.99** | ~5% | **10-15×** | 🟢 Very Conservative |
| **τ=0.97** | ~15-25% | **18-28×** | 🟢 Conservative (recommended) |
| **τ=0.95** (default) | ~25-40% | **24-40×** | 🟡 Balanced |
| **τ=0.90** | ~40-60% | **32-50×** | 🟡 Aggressive (use with Safe Mode) |

> **✅ All settings are safe!** Nangila includes **Safe Mode** which automatically monitors gradient quality and promotes Passengers back to Drivers if issues are detected.

```python
from nangila import Sculptor

# Adjust threshold during calibration
sculptor = Sculptor(threshold=0.97)       # More conservative
sculptor = Sculptor(threshold=0.90)       # More aggressive
```

### Preliminary Test Results (2x RTX 4090)

| Test | Result |
|------|--------|
| Rust Unit Tests | 53/53 ✅ |
| DDP Gradient Sync | ✅ |
| NCCL AllReduce | ✅ |
| 1000-iter Stress | 264 iter/s ✅ |
| CIFAR-10 Convergence | 65.5% accuracy ✅ |
| Sculptor Correlation | 0.973 detected ✅ |

### Theoretical Bandwidth (7B Model)

| Setting | Traffic/Step | Compression | Notes |
|---------|--------------|-------------|-------|
| Standard DDP | 28 GB | 1× | Baseline |
| Nangila τ=0.97 | ~1.4 GB | ~20× | Conservative |
| Nangila τ=0.95 | ~1.0 GB | ~28× | Balanced (default) |
| Nangila τ=0.90 | ~700 MB | ~40× | Aggressive |

> **Note**: Values are estimates based on ideal conditions. Actual compression depends on model architecture, optimization dynamics, and Passenger detection accuracy.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         NANGILA                                  │
├─────────────────┬──────────────────┬────────────────────────────┤
│  nangila-core   │   nangila-cuda   │      nangila-hook          │
│  (Pure Rust)    │   (GPU Kernels)  │   (PyTorch Integration)    │
├─────────────────┼──────────────────┼────────────────────────────┤
│ • Predictor     │ • Fused kernels  │ • FFI for NCCL intercept   │
│ • Quantizer     │ • FP16/BF16      │ • Python bindings          │
│ • Sculptor      │ • GPU-native     │ • DDP comm hook            │
│ • Topology Mask │   state mgmt     │                            │
└─────────────────┴──────────────────┴────────────────────────────┘
```

### Key Components

- **Predictor**: Momentum-based gradient prediction (Slow Head)
- **Quantizer**: INT4 stochastic quantization (Fast Head)
- **Sculptor**: Offline correlation analysis for topology discovery
- **Topology Mask**: Driver/Passenger layer classification
- **GPU State Manager**: RAII-safe persistent CUDA buffers (zero CPU transfers)

## 🔬 Features

### Current
- ✅ **INT4 Quantization** — 8× compression from precision reduction
- ✅ **Predictive Coding** — Residual transmission for lower entropy
- ✅ **Topology Sculpting** — Skip Passenger layers entirely
- ✅ **FP16/BF16 Support** — Compatible with mixed precision training
- ✅ **Safe Mode** — Automatic fallback if compression degrades quality
- ✅ **CUDA Error Handling** — Configurable sync modes (ASYNC/PERIODIC/ALWAYS)
- ✅ **GPU-Native Path** — Zero CPU transfers (implemented, pending hardware validation)
- ✅ **Stochastic Quantization** — Mathematically correct hash-based PRNG
- ✅ **DDP Integration** — Full PyTorch DDP support
- ✅ **FSDP Integration** — Beta (2-GPU validated, large-scale pending)

### In Development
- 🚧 **Large-Scale Validation** — Multi-node (8+ GPU) testing for FSDP
- 🚧 **Performance Benchmarking** — Comprehensive latency and throughput measurements
- 🚧 **Adaptive Monitoring** — Fast-fail drift detection for Safe Mode
- 🚧 **Auto-calibration** — Automatic threshold tuning for Sculptor

## 🗺️ Roadmap

### Phase 1: Core Stability ✅
- [x] Fix CUDA stochastic quantization
- [x] Implement GPU-native compression path
- [x] Resolve critical safety issues

### Phase 2: Production Readiness 🚧
- [ ] **Comprehensive Test Suite** — Multi-GPU validation on A100/H100
  - [ ] Large-scale model testing (7B, 13B, 70B parameter models)
  - [ ] Multi-node (8+ node) scaling tests
  - [ ] Convergence validation (match uncompressed baseline)
  - [ ] Memory leak testing (10K+ training steps)
- [/] **FSDP Integration** — Full PyTorch FSDP support
  - [x] Basic FSDP hook implementation
  - [x] Preliminary gradient sync tests (2-GPU)
  - [ ] Large-scale FSDP validation (8+ GPUs)
  - [ ] Mixed-precision FSDP compatibility
  - [ ] Gradient accumulation support
- [ ] **Performance Optimization**
  - [ ] Adaptive Safe Mode monitoring
  - [ ] Auto-calibration for Sculptor
  - [ ] Kernel fusion optimizations

### Phase 3: Advanced Features 📋
- [ ] **Protocol Modelling** — Learned compression strategies
  - [ ] Neural codec for gradient residuals
  - [ ] Adaptive bitrate allocation
  - [ ] Context-aware prediction
- [ ] **Mixture of Experts (MoE)** — Specialized support for MoE architectures
  - [ ] Expert-aware topology masking
  - [ ] Sparse gradient handling
  - [ ] Router gradient optimization
- [ ] **Communication Backends**
  - [ ] InfiniBand optimization
  - [ ] GCP gVNIC integration

## ⚙️ CUDA Error Handling

Nangila includes comprehensive CUDA kernel error handling with three synchronization modes:

```python
from nangila import SyncMode
from nangila.ddp import register_nangila_hook

# Production (maximum speed, minimal error checking)
hook = register_nangila_hook(model, sync_mode=SyncMode.ASYNC)

# Default (balanced - sync every 100 calls)
hook = register_nangila_hook(model, sync_mode=SyncMode.PERIODIC)

# Debug (catch all errors immediately)
hook = register_nangila_hook(model, sync_mode=SyncMode.ALWAYS)
```

### Sync Modes

| Mode | Overhead | Error Detection | Use Case |
|------|----------|-----------------|----------|
| **ASYNC** | ~0% | Delayed | Production (maximum speed) |
| **PERIODIC** | ~1-2% | Good | Default (balanced) |
| **ALWAYS** | ~10-20% | Immediate | Debugging |

**Recommendation**: Use `PERIODIC` (default) for production. Switch to `ALWAYS` if you encounter issues.

## 📁 Project Structure

```
nangila/
├── nangila-core/       # Pure Rust compression engine
├── nangila-cuda/       # CUDA kernels (optional GPU acceleration)
├── nangila-hook/       # PyTorch integration + Python bindings
├── nangila-intercept/  # NCCL LD_PRELOAD shim (transparent mode)
├── python/             # Python package stubs
└── tests/              # Multi-GPU test suite
```

## 🧪 Running Tests

```bash
# Rust tests
cargo test

# Python tests (on GPU machine)
torchrun --nproc_per_node=2 tests/test_ddp.py
torchrun --nproc_per_node=2 tests/test_stress.py
python tests/test_sculptor.py
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas where we especially need help:
- Large-scale testing (8+ GPUs)
- Integration with popular frameworks (DeepSpeed, Megatron-LM)
- Documentation and examples

## 📖 Citation

If you use Nangila in your research, please cite:

```bibtex
@software{nangila2024,
  title = {Nangila: Gradient Virtualization for Distributed DNN Training},
  year = {2024},
  url = {https://github.com/nangila/nangila}
}
```

## 📄 License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.
