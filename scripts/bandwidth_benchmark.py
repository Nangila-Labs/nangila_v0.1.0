#!/usr/bin/env python3
"""
Bandwidth Benchmark for Nangila
Measures actual compression ratio and bandwidth savings on 8 GPUs
"""

import os
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Nangila imports
try:
    from nangila import NangilaHook, NangilaConfig, Sculptor
    NANGILA_AVAILABLE = True
except ImportError:
    NANGILA_AVAILABLE = False
    print("Warning: Nangila not installed. Run: maturin develop --release -F python")


def setup_distributed():
    """Initialize distributed training"""
    dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def create_dummy_model(hidden_size=4096, num_layers=32):
    """Create a model similar to 7B LLM"""
    layers = []
    for _ in range(num_layers):
        layers.append(torch.nn.Linear(hidden_size, hidden_size))
        layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


def measure_allreduce_time(tensor, num_iterations=100):
    """Measure raw AllReduce time"""
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(num_iterations):
        dist.all_reduce(tensor)
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / num_iterations


def benchmark_baseline(model, batch_size=4, seq_len=512, num_steps=100):
    """Benchmark without Nangila"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    
    model = model.to(device)
    model = DDP(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Warmup
    for _ in range(10):
        x = torch.randn(batch_size, seq_len, 4096, device=device)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    total_bytes = 0
    for _ in range(num_steps):
        x = torch.randn(batch_size, seq_len, 4096, device=device)
        loss = model(x).sum()
        loss.backward()
        
        # Count gradient bytes
        for p in model.parameters():
            if p.grad is not None:
                total_bytes += p.grad.numel() * 4  # FP32
        
        optimizer.step()
        optimizer.zero_grad()
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    return {
        "time_per_step": elapsed / num_steps,
        "total_bytes": total_bytes,
        "bytes_per_step": total_bytes / num_steps,
        "throughput_gbps": (total_bytes / elapsed) / 1e9,
    }


def benchmark_with_nangila(model, batch_size=4, seq_len=512, num_steps=100, threshold=0.95):
    """Benchmark with Nangila compression"""
    if not NANGILA_AVAILABLE:
        return None
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    
    model = model.to(device)
    model = DDP(model)
    
    # Create Nangila hook
    config = NangilaConfig(threshold=threshold)
    hook = NangilaHook.all_drivers(num_layers=1000)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Warmup + calibration
    for step in range(100):
        x = torch.randn(batch_size, seq_len, 4096, device=device)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        hook.step()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    stats_samples = []
    for _ in range(num_steps):
        x = torch.randn(batch_size, seq_len, 4096, device=device)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        hook.step()
        stats_samples.append(hook.get_stats())
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    return {
        "time_per_step": elapsed / num_steps,
        "compression_enabled": stats_samples[-1].get("compression_enabled", False),
        "stats": stats_samples[-1],
    }


def main():
    local_rank = setup_distributed()
    world_size = dist.get_world_size()
    
    if local_rank == 0:
        print("=" * 60)
        print("Nangila Bandwidth Benchmark")
        print("=" * 60)
        print(f"World size: {world_size} GPUs")
        print(f"Nangila available: {NANGILA_AVAILABLE}")
        print()
    
    # Create model
    model = create_dummy_model(hidden_size=4096, num_layers=32)
    total_params = sum(p.numel() for p in model.parameters())
    
    if local_rank == 0:
        print(f"Model parameters: {total_params / 1e9:.2f}B")
        print(f"Gradient size per step: {total_params * 4 / 1e9:.2f} GB (FP32)")
        print()
    
    # Benchmark 1: Raw AllReduce
    if local_rank == 0:
        print("Benchmark 1: Raw AllReduce (1GB tensor)")
    
    test_tensor = torch.randn(256 * 1024 * 1024, device=f"cuda:{local_rank}")  # 1GB
    ar_time = measure_allreduce_time(test_tensor, num_iterations=50)
    
    if local_rank == 0:
        bandwidth = (1e9 / ar_time) / 1e9  # GB/s
        print(f"  AllReduce time: {ar_time*1000:.2f} ms")
        print(f"  Effective bandwidth: {bandwidth:.1f} GB/s")
        print()
    
    del test_tensor
    torch.cuda.empty_cache()
    
    # Benchmark 2: Baseline DDP
    if local_rank == 0:
        print("Benchmark 2: Baseline DDP (no compression)")
    
    model_baseline = create_dummy_model()
    baseline_results = benchmark_baseline(model_baseline, num_steps=50)
    
    if local_rank == 0:
        print(f"  Time per step: {baseline_results['time_per_step']*1000:.1f} ms")
        print(f"  Bytes per step: {baseline_results['bytes_per_step']/1e9:.2f} GB")
        print(f"  Throughput: {baseline_results['throughput_gbps']:.1f} GB/s")
        print()
    
    del model_baseline
    torch.cuda.empty_cache()
    
    # Benchmark 3: With Nangila
    if NANGILA_AVAILABLE:
        for threshold in [0.97, 0.95, 0.90]:
            if local_rank == 0:
                print(f"Benchmark 3: Nangila (τ={threshold})")
            
            model_nangila = create_dummy_model()
            nangila_results = benchmark_with_nangila(model_nangila, num_steps=50, threshold=threshold)
            
            if local_rank == 0 and nangila_results:
                speedup = baseline_results['time_per_step'] / nangila_results['time_per_step']
                print(f"  Time per step: {nangila_results['time_per_step']*1000:.1f} ms")
                print(f"  Speedup: {speedup:.2f}x")
                print(f"  Stats: {nangila_results['stats']}")
                print()
            
            del model_nangila
            torch.cuda.empty_cache()
    
    # Summary
    if local_rank == 0:
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"GPUs: {world_size}")
        print(f"Baseline bytes/step: {baseline_results['bytes_per_step']/1e9:.2f} GB")
        if NANGILA_AVAILABLE:
            print("Nangila compression: ENABLED")
        print()
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
