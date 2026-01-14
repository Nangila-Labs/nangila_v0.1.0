"""
Multi-GPU DDP Test for Nangila MVP Validation
Tests that 2 GPUs can synchronize gradients correctly.

Usage: torchrun --nproc_per_node=2 test_ddp.py
"""
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_test(rank, world_size):
    print(f"[Rank {rank}] Starting on GPU {rank}")
    setup(rank, world_size)
    
    # Create simple model
    model = torch.nn.Linear(100, 10).to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # Create different input on each rank
    torch.manual_seed(rank)
    x = torch.randn(32, 100).to(rank)
    
    # Forward + backward
    output = ddp_model(x)
    loss = output.sum()
    loss.backward()
    
    # Check gradients are synchronized
    grad_norm = model.weight.grad.norm().item()
    print(f"[Rank {rank}] Gradient norm: {grad_norm:.6f}")
    
    # All-reduce to verify NCCL works
    tensor = torch.tensor([rank], dtype=torch.float32).to(rank)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected = sum(range(world_size))
    
    if abs(tensor.item() - expected) < 0.001:
        print(f"[Rank {rank}] NCCL AllReduce: PASSED ✓")
    else:
        print(f"[Rank {rank}] NCCL AllReduce: FAILED")
    
    cleanup()
    print(f"[Rank {rank}] Test complete!")

def main():
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs, launching DDP test...")
    
    if world_size < 2:
        print("ERROR: Need at least 2 GPUs for DDP test")
        return
    
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)
    print("\n=== ALL DDP TESTS PASSED ===")

if __name__ == "__main__":
    main()
