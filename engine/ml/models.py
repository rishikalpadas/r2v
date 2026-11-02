"""Spandrel model loading, wrapped as numpy-in/numpy-out callables. Torch is imported lazily."""
from __future__ import annotations

import functools
import importlib
from pathlib import Path
from typing import Any, Callable

import numpy as np

from engine.ml import weights

ModelFn = Callable[[np.ndarray], np.ndarray]


def _fetch_hint(name: str) -> str:
    return f"python -m engine.ml.fetch_weights {name}"


def _load_descriptor(path: Path) -> Any:
    """Read a weight file safely and build its spandrel ImageModelDescriptor (on CPU).

    spandrel 0.4.2's own file loader calls torch.load with a custom restricted unpickler,
    which makes torch use weights_only=False; we load with weights_only=True instead.
    """
    torch = importlib.import_module("torch")
    spandrel = importlib.import_module("spandrel")
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    return spandrel.ModelLoader().load_from_state_dict(state_dict)  # canonicalizes the dict


def _pad(torch: Any, t: Any, pad_w: int, pad_h: int) -> Any:
    """Pad right/bottom: reflect where possible (needs size - 1 >= pad), replicate the rest."""
    h, w = t.shape[-2:]
    rw, rh = min(pad_w, w - 1), min(pad_h, h - 1)
    if rw or rh:
        t = torch.nn.functional.pad(t, (0, rw, 0, rh), mode="reflect")
    if pad_w - rw or pad_h - rh:
        t = torch.nn.functional.pad(t, (0, pad_w - rw, 0, pad_h - rh), mode="replicate")
    return t


def _padding(size: int, minimum: int, multiple_of: int) -> int:
    target = max(size, minimum)
    target = -(-target // multiple_of) * multiple_of
    return target - size


def _validate(rgb: Any) -> None:
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3 or 0 in rgb.shape:
        raise ValueError(f"rgb must be a non-empty HxWx3 array, got shape {getattr(rgb, 'shape', None)}")
    if rgb.dtype != np.float32:
        raise ValueError(f"rgb must be float32 in 0..1, got dtype {rgb.dtype}")


@functools.lru_cache(maxsize=None)
def load_model(name: str, device: str) -> ModelFn:
    """Load the manifest weight `name` onto `device` ("cuda" or "cpu"), cached per (name, device).

    Returns fn(rgb) -> rgb: HxWx3 float32 0..1 in, (H*scale)x(W*scale)x3 float32 0..1 out
    (a CPU numpy array). `fn.scale` is the model's upscale factor (4 for SR, 1 for
    restoration). Inputs below the model's size requirements are padded (reflect, then
    replicate) and the output cropped back. Runs fp16 on CUDA when the architecture
    supports it, fp32 otherwise. GPU out-of-memory errors propagate unchanged so
    run_tiled can back off.

    Raises KeyError for an unknown name, FileNotFoundError when the weight is missing and
    ValueError when its sha256 does not match the manifest. Failures are not cached.
    """
    path = weights.weight_path(name)
    if not path.is_file():
        raise FileNotFoundError(f"weight {name!r} not found at {path}; run: {_fetch_hint(name)}")
    if not weights.verify(name):
        raise ValueError(
            f"weight {name!r} at {path} fails its sha256 check (corrupt or stale); "
            f"re-download it with: {_fetch_hint(name)}"
        )
    torch = importlib.import_module("torch")
    descriptor = _load_descriptor(path)
    use_half = device.startswith("cuda") and bool(descriptor.supports_half)
    descriptor.to(device)
    descriptor.eval()
    if use_half:
        descriptor.half()
    else:
        descriptor.float()
    dtype = torch.float16 if use_half else torch.float32
    scale = int(descriptor.scale)
    req = descriptor.size_requirements
    minimum, multiple_of = int(req.minimum), int(req.multiple_of)

    def run(rgb: np.ndarray) -> np.ndarray:
        _validate(rgb)
        h, w = rgb.shape[:2]
        with torch.inference_mode():
            t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0)
            t = t.to(device=device, dtype=dtype)
            t = _pad(torch, t, _padding(w, minimum, multiple_of), _padding(h, minimum, multiple_of))
            out = descriptor(t)
            out = out[..., : h * scale, : w * scale].clamp(0, 1)
            arr = out.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        return np.ascontiguousarray(arr, dtype=np.float32)

    run.scale = scale  # type: ignore[attr-defined]
    return run
