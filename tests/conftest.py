"""Shared pytest hooks: skip `gpu`-marked tests when CUDA is unavailable."""
from __future__ import annotations

import pytest


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def pytest_collection_modifyitems(config, items):
    gpu_items = [item for item in items if item.get_closest_marker("gpu")]
    if not gpu_items or _cuda_available():
        return
    skip = pytest.mark.skip(reason="needs a CUDA GPU (torch missing or torch.cuda.is_available() is false)")
    for item in gpu_items:
        item.add_marker(skip)
