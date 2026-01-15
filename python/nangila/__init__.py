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
    cuda_available,
    __version__,
)

# PyTorch DDP integration
from .ddp import NangilaDDPHook, register_nangila_hook

__all__ = [
    "NangilaConfig",
    "NangilaHook", 
    "Sculptor",
    "cuda_available",
    "__version__",
    # DDP integration
    "NangilaDDPHook",
    "register_nangila_hook",
]
