# Nangila Testing Guide

## Quick Start

### Prerequisites
- 2+ NVIDIA GPUs with CUDA 12.0+
- Python 3.8+
- PyTorch 2.0+
- Rust 1.70+

### Setup on GPU Server

```bash
# 1. Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# 2. Install Python dependencies
pip3 install maturin torch pytest numpy

# 3. Build Nangila with CUDA support
cd ~/nangila
maturin develop --release --features cuda

# 4. Run test suite
bash tests/gpu_test_suite.sh
```

## Test Suite

The test suite (`tests/gpu_test_suite.sh`) runs:

1. **SyncMode.ALWAYS Test** - Catch all errors immediately
2. **Stress Test** - 1000 iterations to verify stability
3. **Benchmark** - Compare ASYNC vs PERIODIC vs ALWAYS overhead
4. **SyncMode.PERIODIC Test** - Production default mode
5. **Python Unit Tests** - pytest test suite

## Manual Testing

### Test Individual Sync Modes

```python
from nangila import SyncMode
from nangila.ddp import register_nangila_hook

# Test ALWAYS mode (debug)
hook = register_nangila_hook(model, sync_mode=SyncMode.ALWAYS)

# Test PERIODIC mode (default)
hook = register_nangila_hook(model, sync_mode=SyncMode.PERIODIC)

# Test ASYNC mode (production)
hook = register_nangila_hook(model, sync_mode=SyncMode.ASYNC)
```

### Run DDP Test

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 tests/test_ddp_gpu.py
```

### Run Stress Test

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 tests/verify_overfit.py
```

## Expected Results

### Compilation
- All Rust code compiles without errors
- Python module builds successfully

### Tests
- All unit tests pass
- DDP gradients sync correctly across ranks
- No CUDA errors with ALWAYS mode
- Compression ratio: 20-40×
- Stress test: 1000 iterations without errors

### Performance
- ASYNC: ~0% overhead
- PERIODIC: ~1-2% overhead
- ALWAYS: ~10-20% overhead

## Troubleshooting

### CUDA Errors

If you see CUDA errors:
1. Switch to `SyncMode.ALWAYS` to get immediate error messages
2. Check GPU memory with `nvidia-smi`
3. Verify CUDA version: `nvcc --version`
4. Check error messages for specific issues

### Build Errors

If build fails:
1. Ensure CUDA is in PATH: `export PATH=/usr/local/cuda/bin:$PATH`
2. Check Rust version: `rustc --version` (need 1.70+)
3. Try clean build: `cargo clean && maturin develop --release --features cuda`

### Import Errors

If Python can't import nangila:
1. Verify build succeeded
2. Check Python version matches build
3. Try: `pip3 install -e .`

## Next Steps

After all tests pass:
1. Review benchmark results
2. Choose sync mode for production (recommend PERIODIC)
3. Run with your actual model
4. Monitor compression ratio and performance

## Questions?

See:
- `README.md` - Usage guide
- `CHANGELOG.md` - Recent changes
- `tests/test_cuda_error_handling.py` - Test examples
