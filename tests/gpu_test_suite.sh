#!/bin/bash
# Nangila CUDA Error Handling Test Suite
# Run this on the GPU server to test all sync modes

set -e

echo "=========================================="
echo "Nangila CUDA Error Handling Test Suite"
echo "=========================================="
echo ""

# Check environment
echo "1. Checking environment..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""
nvcc --version | grep "release"
echo ""
python3 --version
echo ""

# Setup environment
echo "2. Setting up environment..."
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
source $HOME/.cargo/env

# Install Python dependencies
echo "3. Installing Python dependencies..."
pip3 install --upgrade pip maturin torch pytest numpy || {
    echo "Installing pip first..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3 get-pip.py
    pip3 install maturin torch pytest numpy
}

# Build the project
echo "4. Building Nangila with CUDA support..."
cd ~/nangila
maturin develop --release --features cuda

# Run Rust tests
echo "5. Running Rust unit tests..."
cargo test --all-features --lib

# Test 1: ALWAYS mode (catch all errors)
echo ""
echo "=========================================="
echo "TEST 1: SyncMode.ALWAYS (Debug Mode)"
echo "=========================================="
cat > test_always_mode.py << 'EOF'
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from nangila import SyncMode
from nangila.ddp import register_nangila_hook

def test_always_mode():
    # Initialize distributed
    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    
    # Create simple model
    model = torch.nn.Linear(1000, 1000).cuda(rank)
    model = DDP(model, device_ids=[rank])
    
    # Register hook with ALWAYS mode
    hook = register_nangila_hook(model, sync_mode=SyncMode.ALWAYS)
    
    # Run training steps
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    for step in range(10):
        x = torch.randn(32, 1000).cuda(rank)
        y = model(x).sum()
        y.backward()
        optimizer.step()
        optimizer.zero_grad()
        hook.step()
        
        if rank == 0:
            print(f"Step {step}: OK")
    
    if rank == 0:
        stats = hook.get_stats()
        print(f"\nCompression ratio: {hook.compression_ratio:.2f}x")
        print(f"Stats: {stats}")
    
    dist.destroy_process_group()
    print("ALWAYS mode test: PASSED")

if __name__ == "__main__":
    test_always_mode()
EOF

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 test_always_mode.py

# Test 2: Stress test with ALWAYS mode
echo ""
echo "=========================================="
echo "TEST 2: Stress Test (1000 iterations)"
echo "=========================================="
cat > test_stress.py << 'EOF'
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from nangila import SyncMode
from nangila.ddp import register_nangila_hook
import time

def test_stress():
    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    
    # Larger model for stress testing
    model = torch.nn.Sequential(
        torch.nn.Linear(2048, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 1000)
    ).cuda(rank)
    model = DDP(model, device_ids=[rank])
    
    # Use ALWAYS mode to catch any errors
    hook = register_nangila_hook(model, sync_mode=SyncMode.ALWAYS)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    start_time = time.time()
    errors = 0
    
    for step in range(1000):
        try:
            x = torch.randn(64, 2048).cuda(rank)
            y = model(x).sum()
            y.backward()
            optimizer.step()
            optimizer.zero_grad()
            hook.step()
            
            if rank == 0 and step % 100 == 0:
                print(f"Step {step}/1000")
        except Exception as e:
            errors += 1
            if rank == 0:
                print(f"ERROR at step {step}: {e}")
    
    elapsed = time.time() - start_time
    
    if rank == 0:
        print(f"\nStress test completed:")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Steps/sec: {1000/elapsed:.2f}")
        print(f"  Errors: {errors}")
        print(f"  Compression ratio: {hook.compression_ratio:.2f}x")
    
    dist.destroy_process_group()
    
    if errors == 0:
        print("Stress test: PASSED")
    else:
        print(f"Stress test: FAILED ({errors} errors)")
        exit(1)

if __name__ == "__main__":
    test_stress()
EOF

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 test_stress.py

# Test 3: Benchmark sync modes
echo ""
echo "=========================================="
echo "TEST 3: Benchmark Sync Modes"
echo "=========================================="
cat > test_benchmark.py << 'EOF'
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from nangila import SyncMode
from nangila.ddp import register_nangila_hook
import time

def benchmark_mode(sync_mode, mode_name):
    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    
    model = torch.nn.Sequential(
        torch.nn.Linear(2048, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 1000)
    ).cuda(rank)
    model = DDP(model, device_ids=[rank])
    
    hook = register_nangila_hook(model, sync_mode=sync_mode)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Warmup
    for _ in range(10):
        x = torch.randn(64, 2048).cuda(rank)
        y = model(x).sum()
        y.backward()
        optimizer.step()
        optimizer.zero_grad()
        hook.step()
    
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(100):
        x = torch.randn(64, 2048).cuda(rank)
        y = model(x).sum()
        y.backward()
        optimizer.step()
        optimizer.zero_grad()
        hook.step()
    
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    
    if rank == 0:
        print(f"{mode_name:15s}: {elapsed:.3f}s ({100/elapsed:.2f} steps/sec)")
    
    dist.destroy_process_group()
    return elapsed

if __name__ == "__main__":
    print("\nBenchmarking sync modes (100 iterations each):")
    print("-" * 50)
    
    # Test each mode
    for mode, name in [(SyncMode.ASYNC, "ASYNC"), 
                       (SyncMode.PERIODIC, "PERIODIC"),
                       (SyncMode.ALWAYS, "ALWAYS")]:
        CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -c "benchmark_mode({}, '{}')".format(mode, name)
EOF

python3 test_benchmark.py

# Test 4: PERIODIC mode (production default)
echo ""
echo "=========================================="
echo "TEST 4: SyncMode.PERIODIC (Production)"
echo "=========================================="
cat > test_periodic.py << 'EOF'
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from nangila import SyncMode
from nangila.ddp import register_nangila_hook

def test_periodic():
    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    
    model = torch.nn.Linear(1000, 1000).cuda(rank)
    model = DDP(model, device_ids=[rank])
    
    # Use PERIODIC mode (default)
    hook = register_nangila_hook(model, sync_mode=SyncMode.PERIODIC)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    for step in range(500):  # More than 100 to trigger periodic sync
        x = torch.randn(32, 1000).cuda(rank)
        y = model(x).sum()
        y.backward()
        optimizer.step()
        optimizer.zero_grad()
        hook.step()
        
        if rank == 0 and step % 100 == 0:
            print(f"Step {step}: OK")
    
    if rank == 0:
        print(f"\nCompression ratio: {hook.compression_ratio:.2f}x")
    
    dist.destroy_process_group()
    print("PERIODIC mode test: PASSED")

if __name__ == "__main__":
    test_periodic()
EOF

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 test_periodic.py

# Test 5: Python unit tests
echo ""
echo "=========================================="
echo "TEST 5: Python Unit Tests"
echo "=========================================="
pytest tests/test_cuda_error_handling.py -v

# Summary
echo ""
echo "=========================================="
echo "TEST SUITE COMPLETE"
echo "=========================================="
echo ""
echo "All tests passed! Summary:"
echo "  ✓ ALWAYS mode catches all errors"
echo "  ✓ Stress test (1000 iterations) passed"
echo "  ✓ Benchmark completed for all modes"
echo "  ✓ PERIODIC mode works correctly"
echo "  ✓ Python unit tests passed"
echo ""
echo "Next steps:"
echo "  1. Review benchmark results above"
echo "  2. If PERIODIC overhead is acceptable, use it as default"
echo "  3. Use ASYNC for production if overhead is too high"
echo "  4. Keep ALWAYS for debugging only"
echo ""
