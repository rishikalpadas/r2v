"""Optional ML support. Torch and spandrel are imported lazily, never at import time."""
from __future__ import annotations

import importlib


def ml_available() -> bool:
    """True when both torch and spandrel can be imported."""
    try:
        importlib.import_module("torch")
        importlib.import_module("spandrel")
    except ImportError:
        return False
    return True
