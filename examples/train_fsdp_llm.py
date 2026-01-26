#!/usr/bin/env python3
"""
FSDP LLM Training with Nangila Compression
Demonstrates using Fully Sharded Data Parallel (ZeRO-3 equivalent) with
Network-Bandwidth-Optimized Gradients.
"""

import os
import time
import argparse
import torch
import torch.distributed as dist
import torch.nn as nn

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)

from torch.utils.data import DataLoader, Dataset
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not installed. Using dummy model.")

# Import Nangila
try:
    from nangila import NangilaConfig
    from nangila.fsdp import NangilaFSDPState, nangila_fsdp_hook
    NANGILA_AVAILABLE = True
except ImportError:
    NANGILA_AVAILABLE = False
    print("Warning: Nangila not installed. Running standard FSDP.")

class DummyTextDataset(Dataset):
    def __init__(self, vocab_size=32000, num_samples=1000, seq_len=512):
        self.vocab_size = vocab_size
        self.num_samples = num_samples
        self.seq_len = seq_len
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Random data
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {"input_ids": input_ids, "labels": input_ids.clone()}

def setup():
    backend = "nccl" if torch.cuda.is_available() and dist.is_nccl_available() else "gloo"
    dist.init_process_group(backend)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()

def train(args):
    rank = setup()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print(f"Starting FSDP Training. World Size: {world_size}")
        print(f"Nangila Enabled: {args.nangila}")

    # Load Model (CPU initially)
    if TRANSFORMERS_AVAILABLE:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    else:
        # Simple Dummy Model for testing logic without transformers
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(768, 768, bias=False)
                self.head = nn.Linear(768, 32000, bias=False)
            def forward(self, input_ids, labels=None):
                x = torch.randn(input_ids.shape + (768,), device=input_ids.device, dtype=torch.bfloat16)
                logits = self.head(self.layer(x))
                loss = None
                if labels is not None:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, 32000), labels.view(-1))
                from collections import namedtuple
                return namedtuple('Output', ['loss'])(loss=loss)
        
        model = SimpleModel().to(dtype=torch.bfloat16)
    
    # FSDP Wrapping
    # Custom policy usually needed for Transformers to wrap each Block
    # For this demo, we wrap the whole model or rely on auto-wrap policy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    import transformers
    
    # Identify Layer Class for wrapping
    # Using generic name detection or specific class
    # Mistral/Llama usually has 'start_input' blocks
    
    # Simple FSDP init
    model = FSDP(
        model,
        device_id=rank,
        sharding_strategy=ShardingStrategy.FULL_SHARD, # ZeRO-3
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16, # Gradients are reduced in BF16 (Nangila casts to F32 usually)
            buffer_dtype=torch.bfloat16,
        ),
    )

    # NANGILA HOOK REGISTRATION
    nangila_state = None
    if args.nangila and NANGILA_AVAILABLE:
        config = NangilaConfig(threshold=0.95)
        nangila_state = NangilaFSDPState(dist.group.WORLD, config)
        
        # Register the hook
        # Note: FSDP hooks signature is (hook(state, grad))? 
        # PyTorch FSDP register_comm_hook takes (state, hook_fn).
        # hook_fn receives (state, grad).
        
        model.register_comm_hook(nangila_state, nangila_fsdp_hook)
        
        if rank == 0:
            print(">>> Nangila FSDP Hook Registered Successfully <<<")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    dataset = DummyTextDataset()
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    
    model.train()
    
    step_times = []
    
    for step, batch in enumerate(dataloader):
        if step >= args.steps:
            break
            
        start = time.perf_counter()
        
        inputs = batch["input_ids"].to(rank)
        labels = batch["labels"].to(rank)
        
        optimizer.zero_grad()
        outputs = model(input_ids=inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        # Nangila state update (step counter)
        if nangila_state:
            nangila_state.step()
            
        torch.cuda.synchronize()
        dt = time.perf_counter() - start
        step_times.append(dt)
        
        if rank == 0:
            stats_str = ""
            if nangila_state and step % 10 == 0:
                s = nangila_state.hook.get_stats()
                stats_str = f" | Comp: {s.get('compression_enabled', False)} Ratio: {s.get('mask_compression_ratio', 0.0):.2f}"
                
            print(f"Step {step}: {dt*1000:.1f}ms (Loss: {loss.item():.4f}){stats_str}")

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="gpt2")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--nangila", action="store_true")
    args = parser.parse_args()
    
    train(args)
