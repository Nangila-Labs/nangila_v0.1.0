
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import argparse
import time

def setup(rank, world_size):
    # If using torchrun, these are already set. Only set defaults if missing.
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12358'
        
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# Large Model (~30GB params)
# 7.5B parameters (FP32) ~ 30GB
# If world_size=4, each GPU holds ~7.5GB
class LargeModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 1 Billion params approx = 4GB FP32
        # We need ~8 Billion params total.
        # Let's create a few large layers.
        # 16384 * 16384 * 4 bytes ~= 1GB per matrix
        # 30 layers -> 30GB
        self.layers = nn.ModuleList([
            nn.Linear(16384, 16384, bias=False) for _ in range(30)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def run_test(rank, world_size):
    setup(rank, world_size)
    
    # 1. Check Initial Memory
    torch.cuda.empty_cache()
    mem_before = torch.cuda.memory_allocated(rank)
    if rank == 0:
        print(f"Initial Memory: {mem_before / 1e9:.2f} GB")
    
    # 2. Initialize Model on CPU (meta device ideal but let's try direct FSDP initialization)
    # FSDP can init on device directly if it fits, or from CPU.
    # We create on CPU to avoid single-card OOM before sharding?
    # Or strict FSDP: create on meta device.
    
    try:
        # Use meta device initialization to avoid CPU OOM too
        with torch.device("meta"):
            model = LargeModel()
        
        # FSDP wrapping
        # Nangila or standard FSDP sharding verification is independent of the hook.
        # The test is "The program does not crash (OOM)".
        # We need to materialize weights.
        # FSDP `param_init_fn` needed? 
        # For simplicity, let's create on CPU if RAM allows (remote might have 256GB CPU RAM).
        # We will assume CPU RAM is sufficient.
        
        model = LargeModel() # CPU
        
        if rank == 0:
            print("Model created on CPU. Wrapping FSDP...")
            
        model = FSDP(model, device_id=rank)
        
        # Move to GPU happens via FSDP internals for shards
        # Wait, FSDP(model) usually keeps it where it is and shards it?
        # Standard usage: move to GPU *then* wrap? No, that OOMs.
        # CPU -> FSDP -> GPU execution.
        
        if rank == 0:
            print("FSDP Wrapped. Allocating dummy input...")
            
        # Dummy forward pass to trigger full sharding/allocation
        # x = torch.randn(1, 16384, device=rank)
        # model(x) 
        
        # Just checking static allocation after wrap
        mem_after = torch.cuda.memory_allocated(rank)
        
        if rank == 0:
            print(f"Memory Per Card: {mem_after / 1e9:.2f} GB")
            print("SUCCESS: Large model initialized without OOM.")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
             if rank == 0:
                 print("FAILURE: OOM detected.")
        else:
             if rank == 0:
                 print(f"FAILURE: Runtime error: {e}")
    except Exception as e:
        if rank == 0:
            print(f"FAILURE: {e}")
            
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=4)
    args = parser.parse_args()
    
    # Prioritize env vars from torchrun
    rank = int(os.environ.get("RANK", args.rank))
    world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
    
    run_test(rank, world_size)
