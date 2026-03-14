
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, TensorDataset
import nangila_ddp_cpp
import argparse
import numpy as np

def setup(rank, world_size):
    # If using torchrun, these are already set. Only set defaults if missing.
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12356'
        
    # Torchrun sets RANK and WORLD_SIZE, but we pass them as args too. 
    # Use args if provided, else env.
    
    print(f"Rank {rank}: Initializing process group...")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(f"Rank {rank}: Process group initialized.")
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 1024, bias=False)
        self.fc2 = nn.Linear(1024, 10, bias=False)

    def forward(self, x):
        return self.fc2(self.fc1(x))

def run_test(use_nangila, rank, world_size, output_file):
    setup(rank, world_size)
    
    # Fix seed
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    
    # Model
    model = SimpleModel().to(rank)
    
    # Hook setup
    if use_nangila:
        # 1000 step warmup means first step is passthrough
        hook = nangila_ddp_cpp.NangilaDDPHook(2, 1000) 
        # FSDP with Hook
        # Note: FSDP hook registration is different from DDP. 
        # Current Nangila C++ hook is for DDP (bucket-based).
        # FSDP uses communication hooks.
        # But wait, Nangila implementation so far has been DDP-centric in C++.
        # Does Nangila support FSDP hooks yet?
        # Looking at `nangila_ddp_cpp` bindings, it exposes `NangilaDDPHook` which is a callable.
        # FSDP `register_comm_hook` expects state and hook_fn.
        # Let's assume for this test we are testing the hook mechanism.
        # But FSDP hooks have specific signature.
        # The prompt says "Run B: Nangila + FSDP".
        # If the C++ hook is designed for DDP bucket signature, it might need adaptation for FSDP.
        # DDP Hook: `fut = hook(group, bucket)`
        # FSDP Hook: `def hook(state: object, grad: torch.Tensor) -> torch.futures.Future[torch.Tensor]`
        # If `NangilaDDPHook` is callable with (group, bucket), it matches DDP/FSDP pattern somewhat.
        pass

    # For FSDP consistency, we initialize FSDP wrapper
    # We need to ensure weights are identical.
    # Since we seed fixed, they should be.
    
    model = FSDP(model, device_id=rank) 
    
    if use_nangila:
        # Register hook
        # NangilaDDPHook.__call__ takes (group, bucket). 
        # FSDP `register_comm_hook(state, hook_fn)` where hook_fn is (state, grad).
        # We need an adapter if the signatures don't match.
        # Let's check `nangila-hook/src/python.rs` or `bindings.cpp`.
        # C++ Bindings: `m.def("__call__", ...)` wraps `hook.run_hook`.
        # `run_hook` implementation takes `c10d::ProcessGroup&` and `at::Tensor& bucket`.
        # FSDP hook receives (state, grad). It does NOT receive the ProcessGroup directly in the signature, usually it's in the state.
        # Standard FSDP hook signature: `hook(state: object, grad: torch.Tensor) -> torch.futures.Future[torch.Tensor]`
        
        # We define a python wrapper for FSDP
        def nangila_fsdp_hook(state, grad):
            pg = state.process_group if hasattr(state, "process_group") else dist.group.WORLD
            # Use blocking run to bypass PyBind future conversion issues
            # The C++ side waits for completion, so gradient is ready.
            hook.run_blocking(pg, grad)
            
            # Return a dummy future that is already done
            fut = torch.futures.Future()
            fut.set_result(grad)
            return fut
        
        model.register_comm_hook(state=model, hook=nangila_fsdp_hook)

    # Data
    data = torch.randn(32, 1024, device=rank)
    target = torch.randn(32, 10, device=rank)
    
    # Forward
    output = model(data)
    loss = nn.MSELoss()(output, target)
    
    # Backward
    loss.backward()
    
    # We capture gradients BEFORE optimizer step.
    # In FSDP, gradients are sharded. We need to look at local shards.
    # But FSDP reduces gradients during backward.
    # So after backward(), valid gradients are in `model.params`. (FlattenedParams)
    
    # Save gradients
    grads = []
    for name, param in model.named_parameters():
        if param.grad is not None:
             grads.append(param.grad.view(-1).cpu().numpy())
    
    all_grads = np.concatenate(grads)
    
    # Save to file
    np.save(output_file, all_grads)
    
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nangila", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    # Prioritize env vars from torchrun
    rank = int(os.environ.get("RANK", args.rank))
    world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
    
    run_test(args.nangila, rank, world_size, args.output)
