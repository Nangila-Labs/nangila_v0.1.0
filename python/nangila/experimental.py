"""Experimental APIs that are not part of the Nangila v0.1 support contract."""

import warnings


warnings.warn(
    "nangila.experimental exposes beta and unstable APIs that are excluded from the v0.1 "
    "support contract.",
    stacklevel=2,
)

from .fsdp import NangilaFSDPState, nangila_fsdp_hook

try:
    from .nangila import (
        cuda_predict_and_quantize,
        cuda_dequantize_and_reconstruct,
    )
except ImportError:
    def cuda_predict_and_quantize(*args, **kwargs):
        raise RuntimeError("CUDA not compiled. Rebuild with CUDA support.")

    def cuda_dequantize_and_reconstruct(*args, **kwargs):
        raise RuntimeError("CUDA not compiled. Rebuild with CUDA support.")


__all__ = [
    "NangilaFSDPState",
    "nangila_fsdp_hook",
    "cuda_predict_and_quantize",
    "cuda_dequantize_and_reconstruct",
]
