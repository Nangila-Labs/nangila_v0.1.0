
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
import argparse

try:
    import nangila
    from nangila.fsdp import NangilaFSDPState, nangila_fsdp_hook, NangilaConfig
except ImportError as e:
    print(f"Nangila import failed: {e}")

# === Scaled-Up Transformer for Bandwidth Stress ===
class ScaledTransformer(nn.Module):
    def __init__(self, vocab_size=50000, embed_dim=1536, num_heads=16, num_layers=12, seq_len=1024):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        
        # 12 Layers (Reduced from 24 due to OOM), 1536 dim
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=4*embed_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        b, s = x.shape
        # Simple pos embed broadcasting
        emb = self.embed(x) + self.pos_embed[:, :s, :]
        out = self.transformer(emb)
        return self.fc(out)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357' # Port for speed tests
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def run_throughput_test(rank, world_size, args):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    
    # Model Config (Reduced)
    SEQ_LEN = 1024 
    BATCH_SIZE = 2 
    
    # 1. Init Model
    model = ScaledTransformer(embed_dim=1536, num_layers=12, seq_len=SEQ_LEN).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"Model Parameters: {num_params / 1e6:.2f}M")
    
    # 2. Setup FSDP / Nangila
    use_nangila = (args.mode == 'nangila')
    
    fsdp_state = None
    if use_nangila:
        config = NangilaConfig.aggressive()
        fsdp_state = NangilaFSDPState(dist.group.WORLD, config, sync_mode=1)
    
    # FSDP Wrap
    # use_orig_params=False for Nangila plumbing fix compatibility if strict, 
    # but our fix handles it via managed_params lookup or attribute bypass.
    # Let's use use_orig_params=False to be safe given Test 1.2 learnings.
    model = FSDP(model, device_id=device, sharding_strategy=ShardingStrategy.FULL_SHARD, use_orig_params=False)
    
    if use_nangila:
        model.register_comm_hook(fsdp_state, nangila_fsdp_hook)
        
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Synthetic Data
    data = torch.randint(0, 50000, (BATCH_SIZE, SEQ_LEN), device=device)
    target = torch.randint(0, 50000, (BATCH_SIZE, SEQ_LEN), device=device)
    
    # Helper for attribute bypass plumbing (Critical for correctness during Nangila runs)
    def nangila_pre_step():
        if use_nangila:
            for p in model.parameters():
                if hasattr(p, '_nangila_grad'):
                    p.grad = p._nangila_grad
                    del p._nangila_grad
                    
            # And step counter
            fsdp_state.step_counter()

    # Warmup
    if rank == 0: print("Warming up...")
    for _ in range(5):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out.view(-1, 50000), target.view(-1))
        loss.backward()
        nangila_pre_step()
        optimizer.step()
        
    torch.cuda.synchronize()
    dist.barrier()
    
    # Measurement
    steps = 50
    if rank == 0: print(f"Measuring {steps} steps...")
    
    start_time = time.time()
    
    for i in range(steps):
        t0 = time.time()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out.view(-1, 50000), target.view(-1))
        loss.backward()
        nangila_pre_step()
        optimizer.step()
        torch.cuda.synchronize() # Wait for step to finish
        t1 = time.time()
        # if rank == 0: print(f"Step {i}: {t1-t0:.3f}s")
        
    end_time = time.time()
    dist.barrier()
    
    total_time = end_time - start_time
    avg_step_time = total_time / steps
    
    # Throughput Calculation
    # Tokens per second per GPU = BATCH * SEQ / Time
    # Total System Throughput = Tokens * WorldSize
    tokens_per_step_per_gpu = BATCH_SIZE * SEQ_LEN
    throughput_per_gpu = tokens_per_step_per_gpu / avg_step_time
    total_throughput = throughput_per_gpu * world_size
    
    if rank == 0:
        print(f"\n=== RESULTS ({args.mode.upper()}) ===")
        print(f"GPUs: {world_size}")
        print(f"Avg Step Time: {avg_step_time:.4f} s")
        print(f"Throughput/GPU: {throughput_per_gpu:.2f} tokens/sec")
        print(f"Total Throughput: {total_throughput:.2f} tokens/sec")
        print("======================\n")

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['baseline', 'nangila'], required=True)
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(run_throughput_test, args=(world_size, args), nprocs=world_size, join=True)
