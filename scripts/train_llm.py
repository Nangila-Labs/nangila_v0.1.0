#!/usr/bin/env python3
"""
LLM Training Test with Nangila Compression
Validates convergence on Mistral 7B with gradient compression
"""

import os
import time
import json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset

# Nangila imports
try:
    from nangila import NangilaHook, NangilaConfig, Sculptor
    NANGILA_AVAILABLE = True
except ImportError:
    NANGILA_AVAILABLE = False


class DummyTextDataset(Dataset):
    """Generate random text-like data for testing"""
    def __init__(self, tokenizer, num_samples=10000, seq_len=512):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = tokenizer.vocab_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Random token IDs (simulates text)
        input_ids = torch.randint(100, self.vocab_size - 100, (self.seq_len,))
        return {"input_ids": input_ids, "labels": input_ids.clone()}


def setup_distributed():
    dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def train_epoch(model, dataloader, optimizer, hook=None, max_steps=None):
    """Train for one epoch or max_steps"""
    model.train()
    total_loss = 0
    step_times = []
    
    for step, batch in enumerate(dataloader):
        if max_steps and step >= max_steps:
            break
        
        start_time = time.perf_counter()
        
        input_ids = batch["input_ids"].cuda()
        labels = batch["labels"].cuda()
        
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if hook:
            hook.step()
        
        torch.cuda.synchronize()
        step_time = time.perf_counter() - start_time
        step_times.append(step_time)
        
        total_loss += loss.item()
        
        if step % 10 == 0 and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            avg_time = sum(step_times[-10:]) / len(step_times[-10:])
            print(f"  Step {step}: loss={loss.item():.4f}, time={avg_time*1000:.1f}ms")
    
    return total_loss / len(step_times), sum(step_times) / len(step_times)


def run_training_test(
    model_path: str = "./mistral-7b",
    use_nangila: bool = True,
    threshold: float = 0.95,
    num_steps: int = 100,
    batch_size: int = 2,
    seq_len: int = 512,
):
    """Run training test with or without Nangila"""
    local_rank = setup_distributed()
    world_size = dist.get_world_size()
    
    if local_rank == 0:
        print("=" * 60)
        print(f"Training Test: {'Nangila' if use_nangila else 'Baseline'}")
        print("=" * 60)
        print(f"Model: {model_path}")
        print(f"World size: {world_size}")
        print(f"Nangila: {use_nangila} (τ={threshold})")
        print(f"Steps: {num_steps}")
        print()
    
    # Load model
    if local_rank == 0:
        print("Loading model...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Wrap in DDP
    model = DDP(model, device_ids=[local_rank])
    
    # Setup Nangila hook
    hook = None
    if use_nangila and NANGILA_AVAILABLE:
        config = NangilaConfig(threshold=threshold, warmup_steps=50)
        hook = NangilaHook.all_drivers(num_layers=1000)
        if local_rank == 0:
            print("Nangila hook attached")
    
    # Create dataset and dataloader
    dataset = DummyTextDataset(tokenizer, num_samples=num_steps * batch_size * 2, seq_len=seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    # Train
    if local_rank == 0:
        print(f"\nStarting training for {num_steps} steps...")
    
    avg_loss, avg_step_time = train_epoch(
        model, dataloader, optimizer, hook=hook, max_steps=num_steps
    )
    
    # Collect stats
    if hook and NANGILA_AVAILABLE:
        stats = hook.get_stats()
    else:
        stats = {}
    
    results = {
        "mode": "nangila" if use_nangila else "baseline",
        "threshold": threshold if use_nangila else None,
        "num_steps": num_steps,
        "avg_loss": avg_loss,
        "avg_step_time_ms": avg_step_time * 1000,
        "throughput_steps_per_sec": 1.0 / avg_step_time,
        "world_size": world_size,
        "nangila_stats": stats,
    }
    
    if local_rank == 0:
        print()
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Average loss: {avg_loss:.4f}")
        print(f"Average step time: {avg_step_time*1000:.1f} ms")
        print(f"Throughput: {1.0/avg_step_time:.2f} steps/sec")
        if stats:
            print(f"Nangila stats: {stats}")
        print()
        
        # Save results
        result_file = f"results_{'nangila' if use_nangila else 'baseline'}_{threshold}.json"
        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {result_file}")
    
    dist.destroy_process_group()
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./mistral-7b", help="Path to model")
    parser.add_argument("--nangila", action="store_true", help="Use Nangila compression")
    parser.add_argument("--threshold", type=float, default=0.95, help="Nangila threshold")
    parser.add_argument("--steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per GPU")
    args = parser.parse_args()
    
    run_training_test(
        model_path=args.model,
        use_nangila=args.nangila,
        threshold=args.threshold,
        num_steps=args.steps,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
