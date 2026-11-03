"""Shared pytest hooks: skip `gpu`-marked tests without CUDA; hide real weights from the rest."""
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


@pytest.fixture(autouse=True)
def _no_weights_unless_gpu(request, tmp_path_factory, monkeypatch):
    """Non-gpu tests see an empty weights dir, so the enhance stage falls back to bicubic
    the same way on every machine, whether or not real weights are installed."""
    if request.node.get_closest_marker("gpu"):
        yield
        return
    monkeypatch.setenv("R2V_WEIGHTS_DIR", str(tmp_path_factory.mktemp("no_weights")))
    from engine.ml import models

    models.load_model.cache_clear()  # a model cached by a gpu test must not leak in
    yield
