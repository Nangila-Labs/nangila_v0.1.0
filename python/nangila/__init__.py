"""
Nangila: Gradient Virtualization for Distributed DNN Training

This package provides gradient compression for PyTorch DDP training,
reducing communication bandwidth by 32-80x with minimal overhead.
"""

# The actual implementation is in the Rust extension module
from .nangila import (
    NangilaConfig,
    NangilaHook,
    Sculptor,
    NangilaHook,
    Sculptor,
    SyncMode,
    CompressorType,
    cuda_available,
    __version__,
)

# CUDA kernel functions (for GPU-native FSDP path)
try:
    from .nangila import (
        cuda_predict_and_quantize,
        cuda_dequantize_and_reconstruct,
    )
except ImportError:
    # CUDA not compiled - provide stub functions that raise
    def cuda_predict_and_quantize(*args, **kwargs):
        raise RuntimeError("CUDA not compiled. Rebuild with CUDA support.")
    def cuda_dequantize_and_reconstruct(*args, **kwargs):
        raise RuntimeError("CUDA not compiled. Rebuild with CUDA support.")

# PyTorch DDP integration
from .ddp import NangilaDDPHook, register_nangila_hook, CPP_HOOK_AVAILABLE

# Try to import native C++ hook
try:
    from nangila_ddp_cpp import NangilaDDPHook as NangilaCppHook
except ImportError:
    NangilaCppHook = None

# PyTorch FSDP integration
from .fsdp import NangilaFSDPState, nangila_fsdp_hook

__all__ = [
    "NangilaConfig",
    "NangilaHook", 
    "Sculptor",
    "Sculptor",
    "SyncMode",
    "CompressorType",
    "cuda_available",
    "cuda_predict_and_quantize",
    "cuda_dequantize_and_reconstruct",
    "__version__",
    # DDP integration
    "NangilaDDPHook",
    "NangilaCppHook",
    "CPP_HOOK_AVAILABLE",
    "register_nangila_hook",
    # FSDP integration
    "NangilaFSDPState",
    "nangila_fsdp_hook",
]
