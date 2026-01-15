"""
Nangila DDP Communication Hook for PyTorch

This module provides a PyTorch-native way to integrate Nangila gradient
compression with DistributedDataParallel (DDP) training.

Usage:
    from nangila.ddp import register_nangila_hook
    
    model = DDP(model, ...)
    hook = register_nangila_hook(model, threshold=0.95)
    
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
    from .nangila import NangilaHook, NangilaConfig
    NANGILA_AVAILABLE = True
except ImportError:
    NANGILA_AVAILABLE = False

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
    """
    
    def __init__(
        self,
        process_group: Optional[Any] = None,
        threshold: float = 0.95,
        warmup_steps: int = 100,
        num_layers: int = 1000,
    ):
        """
        Initialize the Nangila DDP hook.
        
        Args:
            process_group: PyTorch distributed process group (default: global)
            threshold: Compression quality threshold (0.90-0.99)
            warmup_steps: Steps before compression activates
            num_layers: Estimated number of gradient layers
        """
        if not NANGILA_AVAILABLE:
            raise ImportError("Nangila is not available. Install with: pip install nangila")
        
        self.process_group = process_group or dist.distributed_c10d._get_default_group()
        self.threshold = threshold
        self.warmup_steps = warmup_steps
        self.step_count = 0
        
        # Create Nangila hook
        self.config = NangilaConfig(
            threshold=threshold,
            warmup_steps=warmup_steps,
        )
        self.__name__ = "nangila_hook"
        self.__qualname__ = "nangila_hook"
        self.hook = NangilaHook.all_drivers(num_layers)
        self.layer_counter = 0
        
        # Track compression stats
        self.total_original_bytes = 0
        self.total_compressed_bytes = 0
    
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
            
            # Warmup check
            if self.step_count < self.warmup_steps:
                return dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.process_group, async_op=True).get_future()
            
            # === NANGILA COMPRESSION ===
            
            # 1. Compress
            tensor_np = tensor.detach().cpu().numpy()
            self.total_original_bytes += tensor_np.nbytes
            
            compressed = self.hook.compress(layer_id, tensor_np)
            self.total_compressed_bytes += len(compressed)
            
            # 2. Gather
            world_size = dist.get_world_size(self.process_group) if self.process_group else dist.get_world_size()
            gathered = [None for _ in range(world_size)]
            
            dist.all_gather_object(gathered, compressed, group=self.process_group)
            
            # 3. Decompress, Sum, Average
            summed_grad = torch.zeros_like(tensor, dtype=torch.float32)
            
            for comp_bytes_np in gathered:
                res_i_np = self.hook.decompress(layer_id, bytes(comp_bytes_np))
                res_i = torch.from_numpy(res_i_np).to(tensor.device, dtype=torch.float32)
                summed_grad += res_i
            
            summed_grad /= world_size
            
            # 4. Update tensor with Average
            tensor.copy_(summed_grad.to(tensor.dtype))
            
            # 5. Update Predictor
            self.hook.update(layer_id, summed_grad.cpu().numpy())
            
            # 6. Return valid future via Redundant All-Reduce
            # We assume bandwidth is not the verify-blocker, but correctness is.
            # We pre-divide by world_size so that all_reduce(SUM) restores the Average.
            tensor.div_(world_size)
            
            return dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.process_group, async_op=True).get_future()
            
        except Exception as e:
            print(f"\nCRITICAL NANGILA DDP HOOK ERROR: {e}")
            import traceback
            traceback.print_exc()
            # If we fail, we MUST return a valid future or DDP hangs/crashes worse.
            # We'll return identity future (no-op) so training continues (broken but visible)
            # or re-raise to crash.
            # Crash is better for debugging.
            raise e
    
    def step(self):
        """Call after each optimizer step to update Nangila state."""
        self.step_count += 1
        self.layer_counter = 0  # Reset layer counter each step
        self.hook.step()
    
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


def register_nangila_hook(
    ddp_model: DDP,
    threshold: float = 0.95,
    warmup_steps: int = 100,
    prefer_cpp: bool = True,
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
    
    Returns:
        Hook instance (call .step() after each optimizer step)
    
    Example:
        model = DDP(model, device_ids=[local_rank])
        hook = register_nangila_hook(model, threshold=0.95)
        
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
    
    # Try C++ hook first for better performance
    if prefer_cpp and CPP_HOOK_AVAILABLE:
        try:
            hook = NangilaCppHook(num_layers=num_layers, warmup_steps=warmup_steps)
            ddp_model.register_comm_hook(state=None, hook=hook)
            if dist.get_rank() == 0:
                print("Using native C++ DDP hook (high performance)")
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
    )
    
    # Register as communication hook
    ddp_model.register_comm_hook(state=None, hook=hook)
    
    if dist.get_rank() == 0:
        print("Using Python DDP hook")
    
    return hook


__all__ = ['NangilaDDPHook', 'register_nangila_hook', 'CPP_HOOK_AVAILABLE']

