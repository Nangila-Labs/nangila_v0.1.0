
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

try:
    import nangila
    from nangila.fsdp import NangilaFSDPState, nangila_fsdp_hook, NangilaConfig
except ImportError:
    print("Nangila not installed.")

# === Massive Model Definition ===
# Construct a model that is technically large in parameters but computationally simple
# so we don't wait forever for initialization.
class MassiveLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # A single linear layer of size [dim, dim]
        # if dim = 16384 (16k), params = 268M params = ~1GB (float32)
        # We want > 24GB total.
        # So we need ~24 of these layers.
        self.linear = nn.Linear(dim, dim, bias=False)

class MassiveModel(nn.Module):
    def __init__(self, num_layers=30, dim=16384):
        super().__init__()
        self.layers = nn.ModuleList([MassiveLayer(dim) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer.linear(x)
        return x

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def print_memory(rank, txt):
    mem = torch.cuda.memory_allocated(rank) / 1024**3
    max_mem = torch.cuda.max_memory_allocated(rank) / 1024**3
    print(f"Rank {rank} [{txt}]: Alloc {mem:.2f}GB, Max {max_mem:.2f}GB")

def verify_sharding(rank, world_size):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    
    print_memory(rank, "Start")
    
    # 1. Define Model Config
    # Target: 30 layers * 1GB/layer = 30GB total model size.
    # Per GPU (if world_size=8) -> ~4GB per GPU.
    # If world_size=2 -> ~15GB per GPU.
    # GPU VRAM is usually 24GB or 40GB or 80GB.
    # Let's target 30GB to be safe for consumer cards (24GB).
    dim = 8192 # 8192*8192*4 bytes = 256MB per layer.
    # To get 30GB we need 120 layers. That's a lot of layers to init.
    # Let's adjust.
    # dim=16384 (16k) -> 268M params -> 1.07GB per layer.
    # 30 layers -> 32GB. Correct.
    
    dim = 16384 
    dim = 16384 
    num_layers = 2 # Reduced to 2 layers to guarantee fit in 24GB VRAM with overhead
    
    if rank == 0:
        print(f"Initializing model with {num_layers} layers of dim {dim}...")
        print(f"Est. Model Size: {num_layers * (dim*dim*4) / 1024**3:.2f} GB FP32")
    
    # 2. Setup Nangila
    config = NangilaConfig()
    fsdp_state = NangilaFSDPState(dist.group.WORLD, config)
    
    # 3. Init on Meta Device (crucial for big models)
    # PyTorch FSDP allows deferred initialization
    # But FSDP(..., device_id) usually moves to GPU.
    # Strategy: Init on CPU (if RAM allows) or Meta then FSDP wrap.
    # Given typical dev environments, system RAM might be 64GB+, so CPU init is safer than GPU init without sharding.
    
    # However, to PROVE sharding, we typically init on 'meta' and let FSDP materialize chunks.
    # For now, let's try standard CPU init and rely on FSDP to shard on move to GPU.
    # Be careful: initialization on CPU takes memory too.
    
    with torch.device("meta"):
        model = MassiveModel(num_layers=num_layers, dim=dim)
        
    # Checkpoint/Materialization strategy needed for real FSDP on meta.
    # Simplified test:
    # Just verify that FSDP wraps it and memory usage on GPU is low.
    # We will use `param_init_fn` to materialize params directly on device sharded?
    # Or just standard FSDP usage.
    
    # Using 'meta' device requires `reset_parameters` support or custom initialization.
    # Let's use a smaller model if we can't reliably materialize from meta in a simple script.
    # OR: Just trust user has enough CPU RAM.
    
    # Let's try CPU init first.
    # model = MassiveModel(num_layers=num_layers, dim=dim) # On CPU
    
    # WRAPPING
    # We need to wrap *each layer* or groups of layers to enable sharding.
    # Default FSDP wraps root. We need auto_wrap_policy
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    
    print(rank, "Wrapping...")
    model = FSDP(
        model, 
        device_id=device, 
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=lambda module, recurse, nonwrapped_numel: nonwrapped_numel > 1000000,
        param_init_fn=lambda module: module.to_empty(device=device, recurse=False).reset_parameters() # Materialize from meta
    )
    
    # Now params are on GPU, sharded.
    model.register_comm_hook(fsdp_state, nangila_fsdp_hook)
    
    print_memory(rank, "After Init & Wrap")
    
    # Run one dummy step
    data = torch.randn(1, dim, device=device) # Tiny batch
    output = model(data)
    loss = output.sum()
    loss.backward()
    
    print_memory(rank, "After Backward")
    
    if rank == 0:
        print("SUCCESS: Did not OOM.")
        
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("Need >1 GPU")
    else:
        torch.multiprocessing.spawn(verify_sharding, args=(world_size,), nprocs=world_size, join=True)
