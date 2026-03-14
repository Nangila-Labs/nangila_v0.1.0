
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

import nangila
from nangila.fsdp import NangilaFSDPState, nangila_fsdp_hook, NangilaConfig

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def train_fsdp(rank, world_size):
    setup(rank, world_size)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # 1. Initialize Model
    model = SimpleModel().to(device)
    
    # 2. Setup Nangila FSDP State
    # Note: State must be shared across hooks if we had multiple FSDP units,
    # but currently FSDP hook API passes state explicitly if use `functools.partial`,
    # OR we follow the signature `hook(state, grad)`.
    # Pytorch FSDP `register_comm_hook` expects `state` and `hook_fn`.
    
    nangila_config = NangilaConfig.aggressive()
    nangila_state = NangilaFSDPState(dist.group.WORLD, nangila_config)
    
    # 3. Wrap with FSDP and Register Hook
    model = FSDP(model, device_id=device, sharding_strategy=ShardingStrategy.FULL_SHARD)
    
    # REGISTER THE HOOK
    model.register_comm_hook(nangila_state, nangila_fsdp_hook)
    
    # 4. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    # 5. Training Loop
    data = torch.randn(64, 1024).to(device)
    target = torch.randn(64, 1024).to(device)
    
    print(f"Rank {rank}: Starting training...")
    
    for step in range(10):
        optimizer.zero_grad()
        output = model(data)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()
        
        # Determine if hooks worked (check Nangila stats if exposed)
        if rank == 0:
            print(f"Step {step}: Loss {loss.item():.4f}")
            
    cleanup()

if __name__ == "__main__":
    # Simulate multi-process launch if running directly
    # In reality, use `torchrun`
    world_size = torch.cuda.device_count()
    if world_size > 1:
        torch.multiprocessing.spawn(train_fsdp, args=(world_size,), nprocs=world_size, join=True)
    else:
        print("Not enough GPUs for FSDP test (need > 1). Running simplified single-proc check.")
        # Single proc logic or warning
