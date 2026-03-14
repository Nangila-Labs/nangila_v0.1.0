#!/usr/bin/env python3
"""Lightweight Python API smoke tests for the v0.1 surface."""

import nangila
from nangila import NangilaConfig, SyncMode


def test_python_api_smoke():
    config = NangilaConfig(
        momentum=0.9,
        threshold=0.95,
        warmup_steps=10,
        shadow_run_steps=0,
        quantize_bits=4,
    )

    assert config is not None
    assert isinstance(nangila.cuda_available(), bool)
    assert SyncMode.ASYNC == 0
    assert SyncMode.ALWAYS == 1
    assert SyncMode.PERIODIC == 2

if __name__ == "__main__":
    test_python_api_smoke()
