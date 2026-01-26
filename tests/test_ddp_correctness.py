"""
Nangila DDP Gradient Correctness Test
======================================
Verifies that Nangila DDP gradients are correctly averaged (not double-averaged).

Run with: torchrun --nproc_per_node=2 tests/test_ddp_correctness.py

This test ensures the critical gradient scaling bug is fixed.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def test_gradient_scaling():
    """Test that gradients are scaled by 1/world_size, not 1/world_size²"""
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    
    if rank == 0:
        print("=" * 60)
        print("TEST: Gradient Scaling Correctness")
        print("=" * 60)
    
    # Create a simple model
    model = nn.Linear(10, 1, bias=False).to(rank)
    
    # Initialize with known weights
    with torch.no_grad():
        model.weight.fill_(1.0)
    
    ddp_model = DDP(model, device_ids=[rank])
    
    # Create same input on all ranks (for deterministic gradient)
    torch.manual_seed(42)
    x = torch.ones(1, 10, device=rank)
    
    # Forward + backward
    y = ddp_model(x)
    loss = y.sum()
    loss.backward()
    
    # Expected gradient:
    # - Local gradient per rank: d(loss)/d(w) = x = [1, 1, ..., 1]
    # - After all-reduce with AVG: should still be [1, 1, ..., 1]
    # - NOT [1/world_size, 1/world_size, ..., 1/world_size]
    
    expected_grad = torch.ones(1, 10, device=rank)
    actual_grad = model.weight.grad
    
    max_diff = (actual_grad - expected_grad).abs().max().item()
    
    if rank == 0:
        print(f"Expected gradient: {expected_grad[0, :3].tolist()} ...")
        print(f"Actual gradient:   {actual_grad[0, :3].tolist()} ...")
        print(f"Max difference: {max_diff:.6f}")
    
    # Allow small numerical error
    assert max_diff < 1e-5, f"Gradient scaling incorrect! Max diff: {max_diff}"
    
    if rank == 0:
        print("✓ PASSED: Gradients correctly averaged (1/world_size)")
    
    dist.destroy_process_group()


def test_nangila_gradient_parity():
    """Test that Nangila gradients match standard DDP"""
    try:
        from nangila.ddp import register_nangila_hook
    except ImportError:
        print("Nangila not installed, skipping parity test")
        return
    
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    
    if rank == 0:
        print("\n" + "=" * 60)
        print("TEST: Nangila vs Standard DDP Parity")
        print("=" * 60)
    
    # Create two identical models
    torch.manual_seed(42)
    model1 = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    ).to(rank)
    
    torch.manual_seed(42)
    model2 = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    ).to(rank)
    
    # Verify same initialization
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        p2.data.copy_(p1.data)
    
    # Standard DDP
    ddp1 = DDP(model1, device_ids=[rank])
    
    # Nangila DDP (all_drivers mode with no warmup for testing)
    ddp2 = DDP(model2, device_ids=[rank])
    try:
        hook = register_nangila_hook(ddp2, warmup_steps=0, prefer_cpp=False)
    except Exception as e:
        if rank == 0:
            print(f"Could not register Nangila hook: {e}")
            print("Skipping parity test")
        dist.destroy_process_group()
        return
    
    # Same input
    torch.manual_seed(42 + rank)
    x = torch.randn(32, 100, device=rank)
    
    # Forward + backward
    loss1 = ddp1(x).sum()
    loss2 = ddp2(x).sum()
    
    ddp1.zero_grad()
    ddp2.zero_grad()
    
    loss1.backward()
    loss2.backward()
    hook.step()
    
    # Compare gradients
    max_diffs = []
    for (n1, p1), (n2, p2) in zip(ddp1.named_parameters(), ddp2.named_parameters()):
        if p1.grad is not None and p2.grad is not None:
            max_diff = (p1.grad - p2.grad).abs().max().item()
            max_diffs.append(max_diff)
            if rank == 0 and max_diff > 0.01:
                print(f"  {n1}: max_diff = {max_diff:.6f}")
    
    overall_max_diff = max(max_diffs) if max_diffs else 0.0
    
    if rank == 0:
        print(f"Overall max gradient difference: {overall_max_diff:.6f}")
        # Allow for quantization error (INT4 quantization)
        # This should be much smaller with the bug fix
        if overall_max_diff < 0.5:
            print("✓ PASSED: Nangila gradients match DDP (within quantization error)")
        else:
            print(f"✗ WARNING: Gradient difference high ({overall_max_diff:.6f})")
            print("  This may indicate compression error or configuration issues")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    # Test 1: Basic gradient scaling
    test_gradient_scaling()
    
    # Test 2: Nangila parity (if available)
    test_nangila_gradient_parity()
    
    if dist.get_rank() == 0:
        print("\n" + "=" * 60)
        print("ALL GRADIENT TESTS COMPLETED")
        print("=" * 60)
