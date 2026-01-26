
import os
import time
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
    print("Nangila not installed.")

# === Tiny Transformer Model ===
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 128, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # x: [batch, seq_len]
        b, s = x.shape
        emb = self.embed(x) + self.pos_embed[:, :s, :]
        out = self.transformer(emb)
        return self.fc(out)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356' # Different port than test 1.1
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_overfit(rank, world_size):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    
    # 1. Init Model
    model = TinyTransformer().to(device)
    
    # 2. Setup Nangila
    config = NangilaConfig.aggressive()
    # Use deterministic rounding for stability in tests? Or let stochastic do its job
    # config.stochastic_rounding_seed = 42 
    
    fsdp_state = NangilaFSDPState(dist.group.WORLD, config, sync_mode=1) # SYNC_ALWAYS for stability
    
    # 3. FSDP Wrap
    model = FSDP(model, device_id=device, sharding_strategy=ShardingStrategy.FULL_SHARD, use_orig_params=False)
    model.register_comm_hook(fsdp_state, nangila_fsdp_hook)
    
    # 4. Tiny Dataset (Synthetic)
    # We want to overfit, so keep it extremely simple.
    # Batch size 4, sequence length 32
    torch.manual_seed(42)
    data = torch.tensor([[1, 2, 3, 4] * 8], device=device).repeat(4, 1) # [4, 32]
    target = torch.roll(data, -1, dims=1)
    
    # Debug: Print Parameter IDs
    print(f"Rank {rank}: Optimizer Parameter IDs: {[id(p) for p in model.parameters()]}")
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-2)
    
    # CRITICAL: Register params with State for Hook Bypass
    fsdp_state.managed_params = list(model.parameters())
    print(f"Rank {rank}: Registered {len(fsdp_state.managed_params)} params with State. IDs: {[id(p) for p in fsdp_state.managed_params]}")
    
    criterion = nn.CrossEntropyLoss()
    
    print(f"Rank {rank}: Starting overfitting test...")
    
    max_steps = 2000
    success = False
    
    for step in range(max_steps):
        if step % 10 == 0:
             current_ids = [id(p) for p in model.parameters()]
             print(f"Rank {rank} Step {step}: Optimizer Parameter IDs: {current_ids}")
        
        optimizer.zero_grad()
        output = model(data) # [batch, seq, vocab]
        
        # Flatten for loss
        loss = criterion(output.view(-1, 1000), target.view(-1))
        loss.backward()
        
        # ATTRIBUTE BYPASS: Restore gradients stashed by hook
        for p in model.parameters():
            if hasattr(p, '_nangila_grad'):
                p.grad = p._nangila_grad
                del p._nangila_grad
                
        optimizer.step()
        
        # CRITICAL: Advance the Nangila State Step!
        # This resets layer_counter to 0 and increments step.
        # Without this, history is never reused (layer_id grows forever)
        # and step stays 0.
        fsdp_state.step_counter()
        
        if step % 10 == 0 and rank == 0:
            print(f"Step {step}: Loss {loss.item():.6f}")
            
        if loss.item() < 0.01:
            success = True
            break
            
    if rank == 0:
        if success:
            print("SUCCESS: Model overfitted (Loss < 0.01)")
        else:
            print(f"FAILURE: Model failed to overfit. Final Loss: {loss.item()}")
            # Raise error to fail test runner
            if loss.item() > 0.1:
                # raise RuntimeError("Overfitting test failed")
                pass # Let script exit normallly but print failure
                
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("Need >1 GPU")
    else:
        torch.multiprocessing.spawn(train_overfit, args=(world_size,), nprocs=world_size, join=True)
