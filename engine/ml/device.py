"""Compute device selection; falls back to CPU instead of raising."""
from __future__ import annotations

import importlib
from typing import Literal

DEVICE_PREFS = ("auto", "cuda", "cpu")


def _cuda_available() -> bool:
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return False
    return bool(torch.cuda.is_available())  # pragma: no cover


def pick_device(pref: Literal["auto", "cuda", "cpu"] = "auto") -> str:
    """Return "cuda" or "cpu". A "cuda" request without a usable GPU yields "cpu"."""
    if pref not in DEVICE_PREFS:
        raise ValueError(f"device must be one of {DEVICE_PREFS}, got {pref!r}")
    if pref == "cpu":
        return "cpu"
    return "cuda" if _cuda_available() else "cpu"
