
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import nangila_ddp_cpp
import argparse
import random
import numpy as np

def setup(rank, world_size):
    # If using torchrun, these are already set. Only set defaults if missing.
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12357'
        
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# Tiny model mimicking Llama structure (MLP blocks)
class TinyLlama(nn.Module):
    def __init__(self, vocab_size=1000, dim=256, layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim)
            ) for _ in range(layers)
        ])
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h) + h # ResNet-ish
        return self.head(h.mean(dim=1)) # Simple pool for classification/next-token

def run_test(use_nangila, rank, world_size):
    setup(rank, world_size)
    
    # Seeds
    torch.manual_seed(42 + rank)
    
    model = TinyLlama().to(rank)
    model = FSDP(model, device_id=rank)
    
    if use_nangila:
        # Aggressive config for testing compression
        # 10 warmup steps to quickly get into compression
        hook = nangila_ddp_cpp.NangilaDDPHook(2, 10) # layers, warmup
        def nangila_fsdp_hook(state, grad):
            pg = state.process_group if hasattr(state, "process_group") else dist.group.WORLD
            return hook(pg, grad)
        model.register_comm_hook(state=model, hook=nangila_fsdp_hook)
        
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Tiny dataset: Overfit to single batch
    data = torch.randint(0, 1000, (100, 10), device=rank) # 100 samples
    target = torch.randint(0, 1000, (100,), device=rank)
    
    print(f"Rank {rank}: Starting Training...")
    
    for step in range(200):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0 and rank == 0:
            print(f"Step {step}, Loss: {loss.item():.6f}")
            
        if use_nangila:
            hook.step()
            
        if loss.item() < 0.001:
            if rank == 0:
                print(f"Converged at step {step} with loss {loss.item()}")
            break
            
    if rank == 0:
        if loss.item() < 0.01:
            print("SUCCESS: Model overfit successfully.")
        else:
            print(f"FAILURE: Model failed to overfit. Final loss: {loss.item()}")
            
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nangila", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=2)
    args = parser.parse_args()
    
    # Prioritize env vars from torchrun
    rank = int(os.environ.get("RANK", args.rank))
    world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
    
    run_test(args.nangila, rank, world_size)
