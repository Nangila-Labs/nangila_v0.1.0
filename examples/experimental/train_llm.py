#!/usr/bin/env python3
"""
LLM Training Test with Nangila Compression (Optimized Manual Integration)
Validates convergence on GPT-2/Mistral with gradient compression.
Optimized with Parameter Bucketing to maximize throughput while avoiding DDP hook crashes.
"""

import os
import time
import json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Nangila imports
try:
    from nangila import NangilaHook, NangilaConfig, Sculptor
    NANGILA_AVAILABLE = True
except ImportError:
    NANGILA_AVAILABLE = False

class ManualNangilaReducer:
    def __init__(self, model, threshold=0.95, warmup_steps=20, bucket_size_mb=25):
        self.hook = NangilaHook.all_drivers(1000)
        
        # Fast-forward hook internal state to bypass default 1000-step warmup + 100-step shadow run
        # We control warmup manually in Python (warmup_steps=20)
        # So we force Rust hook into 'Active' state immediately.
        # Default config has warmup=1000, shadow=100. Threshold = 1100.
        for _ in range(1150):
            self.hook.step()
            
        self.warmup_steps = warmup_steps
        self.step_count = 0
        self.threshold = threshold
        
        # Create Buckets
        self.buckets = []
        current_bucket = []
        current_size = 0
        
        all_params = [p for p in model.parameters() if p.requires_grad]
        
        # Naive bucketing: accumulate parameters until size limit
        for p in all_params:
            # Estimate size in float32 bytes
            size_bytes = p.numel() * 4 
            if current_size + size_bytes > bucket_size_mb * 1024 * 1024 and current_bucket:
                self.buckets.append(current_bucket)
                current_bucket = []
                current_size = 0
            
            current_bucket.append(p)
            current_size += size_bytes
            
        if current_bucket:
            self.buckets.append(current_bucket)
            
        # Stats
        self.total_original = 0
        self.total_compressed = 0
        
        # Buffer
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.gathered_buffers = [None for _ in range(self.world_size)]
        
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"ManualNangilaReducer: Created {len(self.buckets)} buckets for {len(all_params)} parameters.")

    def step(self):
        self.step_count += 1
        self.hook.step()
    
    def get_stats(self):
        stats = self.hook.get_stats()
        stats['total_original_bytes'] = self.total_original
        stats['total_compressed_bytes'] = self.total_compressed
        if self.total_compressed > 0:
            stats['overall_compression_ratio'] = self.total_original / self.total_compressed
        else:
            stats['overall_compression_ratio'] = 1.0
        return stats

    def reduce(self):
        # Warmup
        if self.step_count < self.warmup_steps:
             for bucket in self.buckets:
                 for p in bucket:
                     if p.grad is not None:
                         dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
             return

        # Compression with Bucketing
        for bucket_idx, bucket in enumerate(self.buckets):
            # 1. Flatten and Concat bucket on GPU first, then move to CPU
            grads = []
            valid_params = []
            for p in bucket:
                if p.grad is not None:
                    # Flatten and cast to float32
                    grads.append(p.grad.flatten().float())
                    valid_params.append(p)
            
            if not grads:
                continue
                
            # Combine into single buffer
            bucket_tensor = torch.cat(grads)
            # Move to CPU for Nangila
            bucket_np = bucket_tensor.cpu().numpy()
            
            self.total_original += bucket_np.nbytes
            
            # 2. Compress
            layer_id = bucket_idx # Stable ID per bucket
            compressed = self.hook.compress(layer_id, bucket_np)
            self.total_compressed += len(compressed)
            
            # 3. Gather
            dist.all_gather_object(self.gathered_buffers, compressed)
            
            # 4. Decompress & Average
            summed = torch.zeros_like(bucket_tensor, dtype=torch.float32)
            
            for comp_bytes in self.gathered_buffers:
                # Decompress
                res_np = self.hook.decompress(layer_id, bytes(comp_bytes))
                res_tensor = torch.from_numpy(res_np).to(bucket_tensor.device)
                summed += res_tensor
            
            summed /= self.world_size
            
            # 5. Scatter back to params
            # Update Predictor (Must use the agreed average)
            self.hook.update(layer_id, summed.cpu().numpy())
            
            offset = 0
            for p in valid_params:
                numel = p.numel()
                # Slice and reshape
                p_grad_new = summed[offset:offset+numel].view(p.shape)
                # Copy back to parameter grad (cast to original dtype, e.g. bf16)
                p.grad.copy_(p_grad_new.to(p.dtype))
                offset += numel


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


def train_epoch(model, dataloader, optimizer, reducer=None, max_steps=None):
    """Train for one epoch or max_steps"""
    model.train()
    total_loss = 0
    step_times = []
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    for step, batch in enumerate(dataloader):
        if max_steps and step >= max_steps:
            break
        
        start_time = time.perf_counter()
        
        input_ids = batch["input_ids"].cuda()
        labels = batch["labels"].cuda()
        
        optimizer.zero_grad()

        if reducer:
            with model.no_sync():
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                loss.backward()
            reducer.reduce()
        else:
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
        
        optimizer.step()
        
        if reducer:
            reducer.step()
        
        torch.cuda.synchronize()
        step_time = time.perf_counter() - start_time
        step_times.append(step_time)
        
        total_loss += loss.item()
        
        if step % 10 == 0 and local_rank == 0:
            avg_time = sum(step_times[-10:]) / len(step_times[-10:])
            ratio_str = ""
            if reducer:
                s = reducer.get_stats()
                ratio = s.get('overall_compression_ratio', 1.0)
                mask_ratio = s.get('mask_compression_ratio', 1.0)
                enabled = s.get('compression_enabled', False)
                ratio_str = f", CR={ratio:.1f}x (Mask={mask_ratio:.3f}, Active={enabled})"
                
            print(f"  Step {step}: loss={loss.item():.4f}, time={avg_time*1000:.1f}ms{ratio_str}")
    
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
        print(f"Training Test: {'Nangila (Manual+Bucketing)' if use_nangila else 'Baseline'}")
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
        dtype=torch.bfloat16, 
        device_map={"": local_rank},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = DDP(model, device_ids=[local_rank])
    
    reducer = None
    if use_nangila and NANGILA_AVAILABLE:
        try:
            reducer = ManualNangilaReducer(
                model, 
                threshold=threshold, 
                warmup_steps=20,
                bucket_size_mb=40 # Larger buckets (40MB) often better for throughput
            )
            if local_rank == 0:
                print("ManualNangilaReducer (Bucketed) initialized")
        except Exception as e:
            if local_rank == 0:
                print(f"Warning: Could not init Nangila reducer: {e}")
            raise e
    
    dataset = DummyTextDataset(tokenizer, num_samples=num_steps * batch_size * 2, seq_len=seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    if local_rank == 0:
        print(f"\nStarting training for {num_steps} steps...")
    
    avg_loss, avg_step_time = train_epoch(
        model, dataloader, optimizer, reducer=reducer, max_steps=num_steps
    )
    
    if reducer:
        stats = reducer.get_stats()
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
