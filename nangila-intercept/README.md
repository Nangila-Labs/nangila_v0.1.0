# Nangila NCCL Intercept

LD_PRELOAD-based NCCL interception for transparent gradient compression.

## Overview

This library intercepts NCCL `ncclAllReduce` calls via `LD_PRELOAD` and redirects them through Nangila's gradient compression pipeline. This provides **transparent integration** with any PyTorch DDP training script.

## Building

### Prerequisites
- CUDA Toolkit (nvcc, cuda_runtime.h)
- CMake 3.18+
- Rust toolchain

### Build Steps

```bash
# From nangila root directory
cd nangila-intercept
./build.sh
```

This will:
1. Build the Rust `nangila-hook` static library
2. Compile the C++ intercept shim
3. Link them into `libnangila_intercept.so`

## Usage

### Basic Usage

```bash
# With topology mask (recommended)
LD_PRELOAD=./libnangila_intercept.so \
NANGILA_MASK=topology.nzmask \
torchrun --nproc_per_node=8 train.py

# Without mask (all layers as Drivers)
LD_PRELOAD=./libnangila_intercept.so \
NANGILA_NUM_LAYERS=200 \
torchrun --nproc_per_node=8 train.py
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NANGILA_MASK` | Path to `.nzmask` topology file | None (all drivers) |
| `NANGILA_NUM_LAYERS` | Number of layers if no mask | 1000 |
| `NANGILA_DEBUG` | Enable debug logging | 0 |

### With DeepSpeed/FSDP

```bash
# Works with any framework using NCCL
LD_PRELOAD=./libnangila_intercept.so \
deepspeed --num_gpus=8 train.py
```

## How It Works

```
┌──────────────────────────────────────┐
│          PyTorch DDP                 │
│              │                       │
│        ncclAllReduce()               │
│              │                       │
│              ▼                       │
│  ┌───────────────────────────────┐   │
│  │   libnangila_intercept.so    │   │
│  │   (LD_PRELOAD intercept)     │   │
│  │                               │   │
│  │  1. Compress via Rust FFI    │   │
│  │  2. Call real ncclAllReduce  │   │
│  │  3. Decompress result        │   │
│  └───────────────────────────────┘   │
│              │                       │
│              ▼                       │
│         Real NCCL                    │
└──────────────────────────────────────┘
```

## Files

```
nangila-intercept/
├── CMakeLists.txt       # CMake build configuration
├── build.sh             # Build script
├── intercept.cpp        # NCCL intercept implementation
├── include/
│   └── nangila.h        # C FFI header for Rust functions
└── README.md
```

## Limitations

- Currently only intercepts `ncclAllReduce` (main DDP bottleneck)
- Works with FP32 gradients only (FP16/BF16 pass through unchanged)
- Requires CUDA environment (won't build without CUDA toolkit)
