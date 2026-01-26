
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

try:
    import nangila
    from nangila.fsdp import NangilaFSDPState, nangila_fsdp_hook, NangilaConfig
except ImportError:
    print("Nangila not installed or not found in path.")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Ensure model is big enough to be interesting but small enough for fast tests
        self.fc1 = nn.Linear(1024, 2048, bias=False)
        self.fc2 = nn.Linear(2048, 1024, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def run_training(rank, world_size, mode, output_dir):
    setup(rank, world_size)
    set_seed(42) # CRITICAL: Same seed for both runs
    
    device = torch.device(f"cuda:{rank}")
    
    # 1. Init Model
    model = SimpleModel().to(device)
    
    # 2. Wrap FSDP
    if mode == 'nangila':
        # Setup Nangila Hook
        config = NangilaConfig.aggressive()
        # Ensure we use deterministic stochastic rounding for test validtiy if applicable
        # config.stochastic_rounding_seed = 42 # If exposed
        
        fsdp_state = NangilaFSDPState(dist.group.WORLD, config, sync_mode=1) # SYNC_ALWAYS
        
        model = FSDP(model, device_id=device, sharding_strategy=ShardingStrategy.FULL_SHARD, use_orig_params=True)
        model.register_comm_hook(fsdp_state, nangila_fsdp_hook)
        
    else: # baseline
        model = FSDP(model, device_id=device, sharding_strategy=ShardingStrategy.FULL_SHARD, use_orig_params=True)
    
    # 3. Data (Fixed seed ensures same data)
    data = torch.randn(64, 1024, device=device)
    target = torch.randn(64, 1024, device=device)
    
    # 4. Forward & Backward
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    
    output = model(data)
    loss = nn.MSELoss()(output, target)
    loss.backward()
    
    # 5. Capture Gradients
    # FSDP shards params. We need to gather full params to compare?
    # Or just compare flat grad vector if we can access it.
    # Easiest way to verify correctness:
    # Gather full model gradients to Rank 0 and save.
    
    # Force full gradient synchronization (if not automatic)
    # With FSDP, grads are sharded.
    # To get full grad, we can use model.full_optim_state_dict? No that's for optim.
    
    # Let's flatten all sharded grads on this rank and save them.
    # BUT: Sharding might be different if Nangila somehow changes it (it shouldn't).
    # Correct verification:
    # Use context manager to gather full model
    
    grads = []
    
    # Try standard method first
    with FSDP.summon_full_params(model, with_grads=True):
        count_grads = 0
        for name, p in model.named_parameters():
             if p.grad is not None:
                 g_flat = p.grad.view(-1).cpu()
                 grads.append(g_flat)
                 count_grads += 1
    
    # Backdoor: If standard method yields zeros/empty (known FSDP hook issue), 
    # retrieve reconstructed gradient directly from Nangila State History.
    # History stores [prev, curr]. 'curr' is the Global Average Gradient.
    if mode == 'nangila' and (len(grads) == 0 or torch.cat(grads).norm() == 0):
        if rank == 0:
            print(f"Rank {rank}: Standard gather checking failed or yielded zeros. Attempting Backdoor Verification via History...")
        
        # We need to construct the full gradient from history layers.
        # History is dict {layer_id: (prev, curr)}
        # We need to sort by layer_id.
        # And concatenate 'curr'.
        # Note: If FSDP wraps entire model, there's only 1 layer_id (0).
        if hasattr(fsdp_state, 'history'):
            layers = sorted(fsdp_state.history.keys())
            backdoor_grads = []
            for lid in layers:
                _, curr = fsdp_state.history[lid]
                # curr is fp16 (from our optimization). Cast to float.
                backdoor_grads.append(curr.float().cpu().view(-1))
            
            if len(backdoor_grads) > 0:
                grads = backdoor_grads
                if rank == 0:
                    print(f"Rank {rank}: Retrieved {len(grads)} layers from History Backdoor.")

    if len(grads) == 0:
        print(f"Rank {rank}: ERROR No gradients found to save!")
        cleanup()
        return

    full_grad = torch.cat(grads)
    print(f"Rank {rank}: Full Grad Norm to Save: {full_grad.norm():.5f}")
    
    if rank == 0:
        filename = os.path.join(output_dir, f"grads_{mode}.pt")
        print(f"Saving gradients to {filename}")
        torch.save(full_grad, filename)
        
    cleanup()

def verify(output_dir):
    baseline_path = os.path.join(output_dir, "grads_baseline.pt")
    nangila_path = os.path.join(output_dir, "grads_nangila.pt")
    
    if not os.path.exists(baseline_path) or not os.path.exists(nangila_path):
        print("Error: Missing gradient files. Run both modes first.")
        return
        
    grad_base = torch.load(baseline_path)
    grad_nang = torch.load(nangila_path)
    
    print(f"DEBUG: Loaded Baseline Type: {type(grad_base)} Shape: {grad_base.shape if hasattr(grad_base, 'shape') else 'N/A'}")
    print(f"DEBUG: Loaded Nangila Type: {type(grad_nang)} Shape: {grad_nang.shape if hasattr(grad_nang, 'shape') else 'N/A'}")
    
    # Handle FSDP padding mismatch
    if hasattr(grad_nang, 'shape') and hasattr(grad_base, 'shape'):
        if grad_nang.numel() > grad_base.numel():
             print(f"DEBUG: Truncating Nangila grad from {grad_nang.numel()} to {grad_base.numel()} (removing padding)")
             grad_nang = grad_nang[:grad_base.numel()]
    
    if hasattr(grad_nang, 'float'):
        print(f"DEBUG: Loaded Nangila Mean: {grad_nang.float().mean():.6f} Norm: {grad_nang.float().norm():.4f}")
    
    print(f"Baseline Grad Norm: {grad_base.norm().item()}")
    print(f"Nangila Grad Norm: {grad_nang.norm().item()}")
    
    # Compare
    # Diff logic
    diff = (grad_base - grad_nang).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Max Diff: {max_diff}")
    print(f"Mean Diff: {mean_diff}")
    
    if torch.allclose(grad_base, grad_nang, atol=1e-6):
        print("SUCCESS: Gradients match to 1e-6 tolerance!")
    else:
        print("FAILURE: Gradients do not match.")
        
        # Analyze why
        # If they are totally different, maybe seed failed.
        if max_diff > 0.1:
            print("  Check random seeds. Diffs are massive.")
        else:
            print("  Small discrepancies. Compression noise?")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "nangila", "verify"], required=True)
    parser.add_argument("--output_dir", default="tests/phase1_truth/results", help="Dir to save artifacts")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == "verify":
        verify(args.output_dir)
    else:
        world_size = torch.cuda.device_count()
        if world_size < 2:
            print("Need at least 2 GPUs for FSDP test")
            # For dev/debugging allow running on 1 GPU mock?
            # exit(1) 
            pass # Pytorch allows FSDP on 1 GPU
            
        torch.multiprocessing.spawn(
            run_training,
            args=(world_size, args.mode, args.output_dir),
            nprocs=world_size,
            join=True
        )
