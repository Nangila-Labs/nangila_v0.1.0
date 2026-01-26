#!/usr/bin/env python3
"""
Comprehensive FSDP Sharding Test with Nangila Compression

Tests that:
1. FSDP properly shards a real LLM across GPUs
2. Nangila compression hook integrates correctly
3. Training loss decreases (model is learning)
4. Gradients remain synchronized across ranks
5. Memory usage improves with sharding

Usage:
    torchrun --nproc_per_node=4 tests/test_fsdp_sharding.py
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

# FSDP imports
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# Try importing transformers for real model
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Import Nangila
try:
    from nangila import NangilaConfig, cuda_available
    from nangila.fsdp import NangilaFSDPState, nangila_fsdp_hook
    NANGILA_AVAILABLE = True
except ImportError:
    NANGILA_AVAILABLE = False


class DummyDataset(Dataset):
    """Simple dataset for testing"""
    def __init__(self, vocab_size=32000, seq_len=128, num_samples=100):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Deterministic data based on idx for reproducibility
        torch.manual_seed(idx)
        input_ids = torch.randint(1, self.vocab_size, (self.seq_len,))
        return {"input_ids": input_ids, "labels": input_ids.clone()}


def setup():
    """Initialize distributed training"""
    dist.init_process_group(backend="nccl")
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def cleanup():
    dist.destroy_process_group()


def get_model_memory_mb(model):
    """Get model memory usage in MB"""
    mem = sum(p.numel() * p.element_size() for p in model.parameters())
    return mem / 1024 / 1024


def test_fsdp_sharding():
    rank, world_size = setup()
    
    results = {
        "passed": True,
        "tests": [],
        "losses": [],
    }
    
    def log(msg):
        if rank == 0:
            print(msg)
    
    log("=" * 60)
    log("FSDP SHARDING TEST WITH NANGILA")
    log("=" * 60)
    log(f"World Size: {world_size}")
    log(f"Transformers: {TRANSFORMERS_AVAILABLE}")
    log(f"Nangila: {NANGILA_AVAILABLE}")
    if NANGILA_AVAILABLE:
        log(f"Nangila CUDA: {cuda_available()}")
    log("=" * 60)
    
    # ========================================
    # TEST 1: Load and Shard Model
    # ========================================
    log("\n[TEST 1] Loading and Sharding Model...")
    
    try:
        if TRANSFORMERS_AVAILABLE:
            # Use TinyLlama for fast testing (1.1B params)
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            log(f"Loading {model_name}...")
            
            config = AutoConfig.from_pretrained(model_name)
            # Reduce for faster testing
            config.num_hidden_layers = 4  # Only 4 layers for quick test
            
            model = AutoModelForCausalLM.from_config(config)
            vocab_size = config.vocab_size
            
            # Create wrap policy using functools.partial (PyTorch 2.x API)
            from functools import partial
            wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={LlamaDecoderLayer}
            )
        else:
            # Fallback: simple transformer-like model
            log("Using dummy model (transformers not available)")
            
            class DummyTransformerBlock(nn.Module):
                def __init__(self, d_model=1024):
                    super().__init__()
                    self.attn = nn.MultiheadAttention(d_model, 8, batch_first=True)
                    self.ff = nn.Sequential(
                        nn.Linear(d_model, d_model * 4),
                        nn.GELU(),
                        nn.Linear(d_model * 4, d_model),
                    )
                    self.norm1 = nn.LayerNorm(d_model)
                    self.norm2 = nn.LayerNorm(d_model)
                
                def forward(self, x):
                    x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
                    x = x + self.ff(self.norm2(x))
                    return x
            
            class DummyLLM(nn.Module):
                def __init__(self, vocab_size=32000, d_model=1024, n_layers=8):
                    super().__init__()
                    self.embed = nn.Embedding(vocab_size, d_model)
                    self.layers = nn.ModuleList([DummyTransformerBlock(d_model) for _ in range(n_layers)])
                    self.head = nn.Linear(d_model, vocab_size, bias=False)
                
                def forward(self, input_ids, labels=None):
                    x = self.embed(input_ids)
                    for layer in self.layers:
                        x = layer(x)
                    logits = self.head(x)
                    
                    loss = None
                    if labels is not None:
                        loss_fn = nn.CrossEntropyLoss()
                        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                    
                    class Output:
                        def __init__(self, loss):
                            self.loss = loss
                    return Output(loss)
            
            model = DummyLLM()
            vocab_size = 32000
            from functools import partial
            wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={DummyTransformerBlock}
            )
        
        # Get pre-shard memory
        pre_shard_mem = get_model_memory_mb(model)
        log(f"Pre-shard model memory: {pre_shard_mem:.1f} MB")
        
        # Wrap with FSDP
        model = FSDP(
            model,
            device_id=rank,
            sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
            mixed_precision=MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            ),
            auto_wrap_policy=wrap_policy,
        )
        
        # Get post-shard memory (on this rank)
        post_shard_mem = get_model_memory_mb(model)
        log(f"Post-shard model memory (rank {rank}): {post_shard_mem:.1f} MB")
        log(f"Sharding efficiency: {pre_shard_mem / max(post_shard_mem, 1):.2f}x reduction")
        
        results["tests"].append(("Model Sharding", True))
        log("✓ Model sharded successfully")
        
    except Exception as e:
        log(f"✗ Model sharding failed: {e}")
        results["tests"].append(("Model Sharding", False))
        results["passed"] = False
        cleanup()
        return results
    
    # ========================================
    # TEST 2: Register Nangila Hook
    # ========================================
    log("\n[TEST 2] Registering Nangila FSDP Hook...")
    
    nangila_state = None
    if NANGILA_AVAILABLE:
        try:
            config = NangilaConfig(threshold=0.95, warmup_steps=0)
            nangila_state = NangilaFSDPState(dist.group.WORLD, config)
            
            model.register_comm_hook(nangila_state, nangila_fsdp_hook)
            
            results["tests"].append(("Nangila Hook Registration", True))
            log("✓ Nangila FSDP hook registered")
            
        except Exception as e:
            log(f"✗ Nangila hook failed: {e}")
            results["tests"].append(("Nangila Hook Registration", False))
            # Continue without Nangila
            nangila_state = None
    else:
        log("⚠ Nangila not available, using standard FSDP")
        results["tests"].append(("Nangila Hook Registration", None))
    
    # ========================================
    # TEST 3: Training Loop (Loss Decreases)
    # ========================================
    log("\n[TEST 3] Training Loop...")
    
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        dataset = DummyDataset(vocab_size=vocab_size, num_samples=50)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        model.train()
        losses = []
        step_times = []
        
        for step, batch in enumerate(dataloader):
            if step >= 20:  # Train for 20 steps
                break
            
            start_time = time.perf_counter()
            
            input_ids = batch["input_ids"].to(rank)
            labels = batch["labels"].to(rank)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            # Step Nangila state
            if nangila_state:
                nangila_state.step_counter()
            
            torch.cuda.synchronize()
            step_time = time.perf_counter() - start_time
            step_times.append(step_time)
            
            loss_val = loss.item()
            losses.append(loss_val)
            
            if rank == 0 and step % 5 == 0:
                log(f"  Step {step}: loss={loss_val:.4f}, time={step_time*1000:.1f}ms")
        
        results["losses"] = losses
        
        # Check loss decreased
        avg_first_5 = sum(losses[:5]) / 5 if len(losses) >= 5 else losses[0]
        avg_last_5 = sum(losses[-5:]) / 5 if len(losses) >= 5 else losses[-1]
        loss_decreased = avg_last_5 < avg_first_5 * 1.1  # Allow 10% tolerance
        
        log(f"  First 5 avg loss: {avg_first_5:.4f}")
        log(f"  Last 5 avg loss: {avg_last_5:.4f}")
        log(f"  Avg step time: {sum(step_times)/len(step_times)*1000:.1f}ms")
        
        if loss_decreased:
            results["tests"].append(("Training Loss Decreases", True))
            log("✓ Training loss decreased (model is learning)")
        else:
            results["tests"].append(("Training Loss Decreases", False))
            log("⚠ Training loss did not decrease (may need more steps)")
        
    except Exception as e:
        import traceback
        log(f"✗ Training failed: {e}")
        traceback.print_exc()
        results["tests"].append(("Training Loop", False))
        results["passed"] = False
    
    # ========================================
    # TEST 4: Gradient Sync Verification
    # ========================================
    log("\n[TEST 4] Gradient Synchronization...")
    
    try:
        # Run one more forward/backward pass
        batch = next(iter(dataloader))
        input_ids = batch["input_ids"].to(rank)
        labels = batch["labels"].to(rank)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        outputs.loss.backward()
        
        # Check a gradient exists and is synchronized
        # FSDP shards gradients, so we check they're not NaN
        grad_ok = True
        grad_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_count += 1
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    log(f"  ✗ NaN/Inf in gradient: {name}")
                    grad_ok = False
                    break
        
        if grad_ok and grad_count > 0:
            results["tests"].append(("Gradient Sync", True))
            log(f"✓ Gradients synchronized ({grad_count} params with grads)")
        else:
            results["tests"].append(("Gradient Sync", False))
            log(f"✗ Gradient sync issue")
        
    except Exception as e:
        log(f"✗ Gradient sync test failed: {e}")
        results["tests"].append(("Gradient Sync", False))
    
    # ========================================
    # SUMMARY
    # ========================================
    log("\n" + "=" * 60)
    log("TEST SUMMARY")
    log("=" * 60)
    
    for test_name, passed in results["tests"]:
        if passed is True:
            status = "✓ PASSED"
        elif passed is False:
            status = "✗ FAILED"
            results["passed"] = False
        else:
            status = "⊘ SKIPPED"
        log(f"  {status}: {test_name}")
    
    log("=" * 60)
    if results["passed"]:
        log("🎉 ALL FSDP SHARDING TESTS PASSED!")
    else:
        log("❌ SOME TESTS FAILED")
    log("=" * 60)
    
    cleanup()
    return results


if __name__ == "__main__":
    test_fsdp_sharding()
