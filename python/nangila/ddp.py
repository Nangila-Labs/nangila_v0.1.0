"""
Nangila DDP Communication Hook for PyTorch

This module provides a PyTorch-native way to integrate Nangila gradient
compression with DistributedDataParallel (DDP) training.

Usage:
    from nangila.ddp import register_nangila_hook
    from nangila import SyncMode
    
    model = DDP(model, ...)
    hook = register_nangila_hook(
        model, 
        threshold=0.95,
        sync_mode=SyncMode.PERIODIC  # Default: balanced error checking
    )
    
    # Training loop as usual
    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
        hook.step()  # Update Nangila state
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional, Callable, Any

try:
    from .nangila import NangilaHook, NangilaConfig, SyncMode
    NANGILA_AVAILABLE = True
except ImportError:
    NANGILA_AVAILABLE = False
    # Define stub for type hints
    class SyncMode:
        ASYNC = 0
        ALWAYS = 1
        PERIODIC = 2

# Try to import native C++ hook (much faster than Python)
try:
    from nangila_ddp_cpp import NangilaDDPHook as NangilaCppHook
    CPP_HOOK_AVAILABLE = True
except ImportError:
    CPP_HOOK_AVAILABLE = False

# Import GradBucket for type hinting (required by register_comm_hook)
try:
    from torch.distributed import GradBucket
except ImportError:
    try:
        from torch.distributed.grad_bucket import GradBucket
    except ImportError:
        GradBucket = Any



class NangilaDDPHook:
    """
    DDP communication hook that compresses gradients using Nangila.
    
    This hook intercepts gradient synchronization in PyTorch DDP and
    applies Nangila's predictive compression to reduce bandwidth usage.
    
    Args:
        process_group: PyTorch distributed process group (default: global)
        threshold: Compression quality threshold (0.90-0.99)
        warmup_steps: Steps before compression activates
        num_layers: Estimated number of gradient layers
        sync_mode: CUDA kernel synchronization mode (SyncMode.ASYNC, ALWAYS, or PERIODIC)
    """
    
    def __init__(
        self,
        process_group: Optional[Any] = None,
        threshold: float = 0.95,
        warmup_steps: int = 100,
        num_layers: int = 1000,
        sync_mode: int = 2,  # SyncMode.PERIODIC
        compressor_type: int = 0,
        dgc_sparsity: float = 0.999,
        power_sgd_rank: int = 1,
    ):
        """
        Initialize the Nangila DDP hook.
        """
        if not NANGILA_AVAILABLE:
            raise ImportError("Nangila is not available. Install with: pip install nangila")
        
        self.process_group = process_group or dist.distributed_c10d._get_default_group()
        self.threshold = threshold
        self.warmup_steps = warmup_steps
        self.step_count = 0
        self.sync_mode = sync_mode
        
        # Create Nangila hook
        # Get CompressorType logic
        try:
             from .nangila import CompressorType
             # Ensure valid type (default to PredictionResidual if not specified)
             if isinstance(compressor_type, int):
                 ctype_map = {0: CompressorType.PredictionResidual, 1: CompressorType.DGC, 2: CompressorType.PowerSGD}
                 c_type = ctype_map.get(compressor_type, CompressorType.PredictionResidual)
             else:
                 c_type = compressor_type
        except ImportError:
             # Fallback if module not fully loaded (shouldn't happen)
             c_type = 0
             
        self.config = NangilaConfig(
            threshold=threshold,
            warmup_steps=warmup_steps,
            compressor_type=c_type,
            dgc_sparsity=dgc_sparsity,
            power_sgd_rank=power_sgd_rank,
        )
        self.__name__ = "nangila_hook"
        self.__qualname__ = "nangila_hook"
        self.hook = NangilaHook.all_drivers(num_layers)
        self.layer_counter = 0
        self.step_count = 0
        
        # Track compression stats
        self.total_original_bytes = 0
        self.total_compressed_bytes = 0
        
        # Log sync mode
        sync_mode_name = {0: "ASYNC", 1: "ALWAYS", 2: "PERIODIC"}.get(sync_mode, "UNKNOWN")
        if dist.get_rank() == 0:
            print(f"Nangila DDP Hook initialized with sync_mode={sync_mode_name}")
            if sync_mode == 0:  # ASYNC
                print("  WARNING: ASYNC mode has no error checking. Use PERIODIC or ALWAYS for debugging.")
    
    def __call__(self, state: Any, bucket: GradBucket) -> torch.futures.Future[torch.Tensor]:
        """
        Communication hook called by DDP for each gradient bucket.
        
        During warmup: standard all-reduce (no compression)
        After warmup: Nangila compression -> reduced all-reduce -> decompress
        """
        try:
            self.layer_counter += 1
            layer_id = self.layer_counter % 1000
            
            # Get the gradient tensor
            tensor = bucket.buffer()
            
            # Warmup check (use standard all_reduce with AVG)
            if self.step_count < self.warmup_steps:
                # During warmup, use standard all-reduce
                fut = dist.all_reduce(tensor, op=dist.ReduceOp.AVG, group=self.process_group, async_op=True).get_future()
                # Wrap in a new future to ensure proper type for PyTorch 2.10
                result_fut = torch.futures.Future()
                
                def _on_complete(fut_result):
                    result_fut.set_result(tensor)
                
                fut.then(_on_complete)
                return result_fut
            
            # === NANGILA GPU-NATIVE COMPRESSION ===
            
            # Check if we can use GPU-native path
            if tensor.is_cuda and hasattr(self.hook, 'compress_gpu'):
                fut = self._compress_gpu_native(tensor, layer_id)
            else:
                # Fallback to CPU path (slow but works)
                fut = self._compress_cpu_fallback(tensor, layer_id)
            
            return fut
            
        except Exception as e:
            print(f"\nCRITICAL NANGILA DDP HOOK ERROR: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    def _compress_gpu_native(self, tensor: torch.Tensor, layer_id: int) -> torch.futures.Future[torch.Tensor]:
        """GPU-native compression path - keeps data on GPU throughout"""
        world_size = dist.get_world_size(self.process_group) if self.process_group else dist.get_world_size()
        
        # 1. Compress on GPU (returns GPU tensor)
        # Pass sync_mode to CUDA kernel for error checking
        compressed_gpu = self.hook.compress_gpu(layer_id, tensor, sync_mode=self.sync_mode)
        self.total_original_bytes += tensor.numel() * 4
        self.total_compressed_bytes += compressed_gpu.numel()
        
        # 2. Gather compressed sizes
        local_size = torch.tensor([compressed_gpu.numel()], dtype=torch.long, device=tensor.device)
        all_sizes = [torch.empty_like(local_size) for _ in range(world_size)]
        dist.all_gather(all_sizes, local_size, group=self.process_group)
        
        max_size = max(s.item() for s in all_sizes)
        
        # 3. Pad and all-gather compressed data
        if compressed_gpu.numel() < max_size:
            padded = torch.zeros(max_size, dtype=torch.uint8, device=tensor.device)
            padded[:compressed_gpu.numel()] = compressed_gpu
            compressed_gpu = padded
        
        gathered_tensors = [torch.empty(max_size, dtype=torch.uint8, device=tensor.device) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, compressed_gpu, group=self.process_group)
        
        # 4. Decompress all ranks on GPU (batch operation)
        decompressed_list = []
        for i, comp_tensor in enumerate(gathered_tensors):
            actual_size = all_sizes[i].item()
            comp_slice = comp_tensor[:actual_size]
            # Pass sync_mode to CUDA kernel for error checking
            decompressed = self.hook.decompress_gpu(layer_id, comp_slice, tensor.numel(), sync_mode=self.sync_mode)
            decompressed_list.append(decompressed)
        
        # 5. Stack and average on GPU
        stacked = torch.stack(decompressed_list, dim=0)
        averaged = stacked.mean(dim=0)
        
        # 6. Update tensor with averaged gradient
        tensor.copy_(averaged)
        
        # 7. Update Predictor with averaged gradient (async, non-blocking)
        self.hook.update_gpu(layer_id, averaged, sync_mode=self.sync_mode)
        
        # 8. Return completed future
        future = torch.futures.Future()
        future.set_result(tensor)
        return future
    
    def _compress_cpu_fallback(self, tensor: torch.Tensor, layer_id: int) -> torch.futures.Future[torch.Tensor]:
        """CPU fallback path (original implementation)"""
        world_size = dist.get_world_size(self.process_group) if self.process_group else dist.get_world_size()
        
        # 1. Compress (single contiguous CPU transfer)
        tensor_cpu = tensor.detach().cpu().contiguous()
        tensor_np = tensor_cpu.numpy()
        self.total_original_bytes += tensor_np.nbytes
        
        compressed = self.hook.compress(layer_id, tensor_np)
        compressed_bytes = bytes(compressed)
        self.total_compressed_bytes += len(compressed_bytes)
        
        # 2. Gather compressed data from all ranks
        # First, gather sizes
        local_size = torch.tensor([len(compressed_bytes)], dtype=torch.long, device=tensor.device)
        all_sizes = [torch.empty_like(local_size) for _ in range(world_size)]
        dist.all_gather(all_sizes, local_size, group=self.process_group)
        
        max_size = max(s.item() for s in all_sizes)
        
        # Pad local compressed data to max_size
        padded_compressed = torch.zeros(max_size, dtype=torch.uint8, device=tensor.device)
        local_tensor = torch.frombuffer(compressed_bytes, dtype=torch.uint8).to(tensor.device)
        padded_compressed[:len(compressed_bytes)] = local_tensor
        
        # All-gather compressed tensors
        gathered_tensors = [torch.empty(max_size, dtype=torch.uint8, device=tensor.device) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, padded_compressed, group=self.process_group)
        
        # 3. Decompress all ranks (batch CPU operations)
        import numpy as np
        decompressed_list = []
        for i, comp_tensor in enumerate(gathered_tensors):
            actual_size = all_sizes[i].item()
            comp_bytes = bytes(comp_tensor[:actual_size].cpu().numpy())
            res_np = self.hook.decompress(layer_id, comp_bytes)
            decompressed_list.append(res_np)
        
        # 4. Stack and average on CPU
        stacked = np.stack(decompressed_list, axis=0)
        averaged = stacked.mean(axis=0).astype(np.float32)
        
        # 5. Single GPU transfer of averaged result
        result_tensor = torch.from_numpy(averaged).to(
            device=tensor.device,
            dtype=tensor.dtype,
            non_blocking=True
        )
        
        # 6. Update tensor with averaged gradient
        tensor.copy_(result_tensor, non_blocking=True)
        
        # 7. Update Predictor with averaged gradient
        self.hook.update(layer_id, averaged)
        
        # 8. Return completed future
        future = torch.futures.Future()
        future.set_result(tensor)
        return future
    
    def step(self):
        """Call after each optimizer step to update Nangila state."""
        self.step_count += 1
        self.layer_counter = 0  # Reset layer counter each step
        self.hook.step()
        
        # Report compression ratio every 10 steps
        if self.step_count % 10 == 0 and self.step_count > self.warmup_steps:
            ratio = self.compression_ratio
            if dist.get_rank() == 0 and ratio > 1.0:
                orig_mb = self.total_original_bytes / 1e6
                comp_mb = self.total_compressed_bytes / 1e6
                print(f"[Step {self.step_count}] Compression: {orig_mb:.1f} MB → {comp_mb:.1f} MB ({ratio:.1f}×)")
        
        # Auto predictor hash verification every 100 steps
        if self.step_count % 100 == 0 and self.step_count > 0:
            self._verify_predictor_hash()
    
    def get_stats(self) -> dict:
        """Get compression statistics."""
        stats = self.hook.get_stats()
        stats['total_original_bytes'] = self.total_original_bytes
        stats['total_compressed_bytes'] = self.total_compressed_bytes
        if self.total_original_bytes > 0:
            stats['overall_compression_ratio'] = (
                self.total_original_bytes / max(self.total_compressed_bytes, 1)
            )
        return stats
    
    @property
    def compression_ratio(self) -> float:
        """Current compression ratio (original / compressed)."""
        if self.total_compressed_bytes == 0:
            return 1.0
        return self.total_original_bytes / self.total_compressed_bytes

    def _verify_predictor_hash(self):
        """Verify predictor hash across all ranks, trigger recovery on mismatch."""
        try:
            # Get local predictor hash
            local_hash = self.hook.predictor_hash()
            
            # All-reduce to check if all ranks have same hash (XOR should be 0)
            hash_tensor = torch.tensor([local_hash], dtype=torch.long, device='cpu')
            
            # Gather all hashes
            world_size = dist.get_world_size(self.process_group)
            all_hashes = [torch.zeros_like(hash_tensor) for _ in range(world_size)]
            dist.all_gather(all_hashes, hash_tensor, group=self.process_group)
            
            # Check if all hashes match
            first_hash = all_hashes[0].item()
            all_match = all(h.item() == first_hash for h in all_hashes)
            
            if not all_match:
                rank = dist.get_rank()
                print(f"[Rank {rank}] PREDICTOR HASH MISMATCH at step {self.step_count}!")
                print(f"[Rank {rank}] Hashes: {[h.item() for h in all_hashes]}")
                # Trigger force sync on next compression
                # The hook.verify_hash method handles recovery internally
                self.hook.verify_hash(first_hash)  # Forces recovery mode
        except Exception as e:
            # Non-fatal - just log and continue
            print(f"Hash verification failed: {e}")


def register_nangila_hook(
    ddp_model: DDP,
    threshold: float = 0.95,
    warmup_steps: int = 100,
    prefer_cpp: bool = True,
    sync_mode: int = 2,  # SyncMode.PERIODIC
    compressor_type: int = 0, # CompressorType.PredictionResidual
    dgc_sparsity: float = 0.999,
    power_sgd_rank: int = 1,
) -> "NangilaDDPHook":
    """
    Register Nangila compression hook with a DDP model.
    
    Automatically uses native C++ hook if available (recommended for production),
    otherwise falls back to Python implementation.
    
    Args:
        ddp_model: A DistributedDataParallel wrapped model
        threshold: Compression quality threshold (0.90-0.99)
        warmup_steps: Steps before compression activates
        prefer_cpp: If True, use C++ hook when available (default: True)
        sync_mode: CUDA kernel synchronization mode
            - SyncMode.ASYNC (0): No sync, maximum performance (production)
            - SyncMode.ALWAYS (1): Always sync, catch all errors (debug)
            - SyncMode.PERIODIC (2): Sync every 100 calls (default, balanced)
        compressor_type: CompressorType (0=PredictionResidual, 1=DGC, 2=PowerSGD)
        dgc_sparsity: Sparsity for DGC compressor (default 0.999)
        power_sgd_rank: Rank for PowerSGD compressor (default 1)
    
    Returns:
        Hook instance (call .step() after each optimizer step)
    
    Example:
        from nangila import SyncMode
        
        model = DDP(model, device_ids=[local_rank])
        
        # For production (maximum speed, minimal error checking)
        hook = register_nangila_hook(model, sync_mode=SyncMode.ASYNC)
        
        # For debugging (catch all errors immediately)
        hook = register_nangila_hook(model, sync_mode=SyncMode.ALWAYS)
        
        # Default (balanced)
        hook = register_nangila_hook(model)  # Uses SyncMode.PERIODIC
        
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            optimizer.step()
            hook.step()  # Important!
    """
    if not isinstance(ddp_model, DDP):
        raise TypeError(f"Expected DistributedDataParallel, got {type(ddp_model)}")
    
    # Estimate number of layers from model parameters
    num_layers = len(list(ddp_model.parameters()))
    
    # Try C++ hook first for better performance (and stability)
    if prefer_cpp and CPP_HOOK_AVAILABLE:
        try:
            # NangilaCppHook signature: (num_layers, warmup_steps)
            hook = NangilaCppHook(num_layers, warmup_steps)
            ddp_model.register_comm_hook(state=None, hook=hook)
            if dist.get_rank() == 0:
                print("Using native C++ DDP hook (NangilaCppHook)")
            return hook
        except Exception as e:
            if dist.get_rank() == 0:
                print(f"C++ hook failed ({e}), falling back to Python")
    
    # Fall back to Python hook
    hook = NangilaDDPHook(
        process_group=ddp_model.process_group,
        threshold=threshold,
        warmup_steps=warmup_steps,
        num_layers=num_layers,
        sync_mode=sync_mode,
        compressor_type=compressor_type,
        dgc_sparsity=dgc_sparsity,
        power_sgd_rank=power_sgd_rank,
    )
    
    # Register as communication hook
    ddp_model.register_comm_hook(state=None, hook=hook)
    
    if dist.get_rank() == 0:
        print("Using Python DDP hook")
    
    return hook


__all__ = ['NangilaDDPHook', 'register_nangila_hook', 'CPP_HOOK_AVAILABLE']

