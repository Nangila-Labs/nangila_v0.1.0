import importlib

"""
Nangila: Gradient Virtualization for Distributed DNN Training

The v0.1 stable surface targets PyTorch DDP training.
"""

from .nangila import (
    NangilaConfig,
    NangilaHook,
    Sculptor,
    SyncMode,
    CompressorType,
    cuda_available,
    __version__,
)

from .ddp import NangilaDDPHook, register_nangila_hook, CPP_HOOK_AVAILABLE

try:
    from nangila_ddp_cpp import NangilaDDPHook as NangilaCppHook
except ImportError:
    NangilaCppHook = None


def __getattr__(name):
    if name == "experimental":
        return importlib.import_module(".experimental", __name__)
    raise AttributeError(f"module 'nangila' has no attribute {name!r}")

__all__ = [
    "NangilaConfig",
    "NangilaHook",
    "Sculptor",
    "SyncMode",
    "CompressorType",
    "cuda_available",
    "__version__",
    "NangilaDDPHook",
    "NangilaCppHook",
    "CPP_HOOK_AVAILABLE",
    "register_nangila_hook",
    "experimental",
]
