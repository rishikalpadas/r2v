"""Tiled model inference with feathered blending and GPU out-of-memory back-off. Pure numpy."""
from __future__ import annotations

import math
import sys
from typing import Callable, NamedTuple

import numpy as np

ModelFn = Callable[[np.ndarray], np.ndarray]


class TiledResult(NamedTuple):
    image: np.ndarray  # (H*scale)x(W*scale)x3 float32
    final_tile: int  # tile size that succeeded after any OOM halving


def _is_oom(exc: BaseException) -> bool:
    return type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower()


def _free_cuda_cache() -> None:
    torch = sys.modules.get("torch")  # never import torch just for this
    if torch is None:
        return
    try:
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - best effort; the retry proceeds regardless
        pass


def _starts(length: int, win: int, stride: int) -> list[int]:
    """Window starts covering [0, length), evenly spaced, last window flush with the end."""
    if length <= win:
        return [0]
    n = math.ceil((length - win) / stride) + 1
    return [round(i * (length - win) / (n - 1)) for i in range(n)]


def _ramps(starts: list[int], win: int, scale: int) -> list[np.ndarray]:
    """Per-window 1D output weights: linear ramps across each overlap with a neighbour."""
    out = []
    size = win * scale
    for i, s in enumerate(starts):
        x = np.arange(size, dtype=np.float64)
        w = np.ones(size)
        if i > 0:
            ov = (starts[i - 1] + win - s) * scale
            if ov > 0:
                w = np.minimum(w, (x + 0.5) / ov)
        if i < len(starts) - 1:
            ov = (s + win - starts[i + 1]) * scale
            if ov > 0:
                w = np.minimum(w, (size - x - 0.5) / ov)
        out.append(w)
    return out


def _run_once(model_fn: ModelFn, rgb: np.ndarray, scale: int, tile: int, overlap: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    ov = min(overlap, tile // 4)  # keep the core at least half the tile
    stride = tile - 2 * ov
    win_h, win_w = min(tile, h), min(tile, w)
    ys, xs = _starts(h, win_h, stride), _starts(w, win_w, stride)
    wy, wx = _ramps(ys, win_h, scale), _ramps(xs, win_w, scale)

    # float32 accumulator; the weights are separable, so their sum is sy x sx
    acc = np.zeros((h * scale, w * scale, 3), dtype=np.float32)
    sy = np.zeros(h * scale, dtype=np.float32)
    sx = np.zeros(w * scale, dtype=np.float32)
    for y, ry in zip(ys, wy):
        sy[y * scale:(y + win_h) * scale] += ry
    for x, rx in zip(xs, wx):
        sx[x * scale:(x + win_w) * scale] += rx
    expected = (win_h * scale, win_w * scale, 3)
    for y, ry in zip(ys, wy):
        for x, rx in zip(xs, wx):
            part = model_fn(np.ascontiguousarray(rgb[y:y + win_h, x:x + win_w]))
            if getattr(part, "shape", None) != expected:
                raise ValueError(f"model output shape {getattr(part, 'shape', None)} != {expected}")
            weight = (ry[:, None] * rx[None, :]).astype(np.float32)[..., None]
            oy, ox = y * scale, x * scale
            acc[oy:oy + expected[0], ox:ox + expected[1]] += part * weight
    acc /= sy[:, None, None]
    acc /= sx[None, :, None]
    return acc


def run_tiled(
    model_fn: ModelFn,
    rgb: np.ndarray,
    scale: int,
    tile: int = 512,
    overlap: int = 32,
    min_tile: int = 64,
) -> TiledResult:
    """Run `model_fn` over `rgb` (HxWx3 float32 0..1) in feather-blended tiles.

    `tile` is the model input window (context included), clamped at the start to
    max(min_tile, min(tile, long side)). Windows are evenly spaced, stay inside the image
    and overlap by at least `overlap` px of context per side (capped at tile // 4); up to
    three windows can overlap along one axis. Each window's output is weighted by linear
    ramps across its overlaps and the blend is normalized by the summed weights. On GPU
    OOM the tile is halved and the whole image restarted; below `min_tile` raises
    MemoryError. Returns TiledResult(image, final_tile); unpacks as a tuple.
    """
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3 or 0 in rgb.shape:
        raise ValueError(f"rgb must be HxWx3, got shape {getattr(rgb, 'shape', None)}")
    if isinstance(scale, bool) or not isinstance(scale, int) or scale < 1:
        raise ValueError(f"scale must be an int >= 1, got {scale!r}")
    for name, value in (("tile", tile), ("overlap", overlap), ("min_tile", min_tile)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an int, got {value!r}")
    if min_tile < 1 or tile < min_tile:
        raise ValueError(f"need 1 <= min_tile <= tile, got min_tile={min_tile}, tile={tile}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    rgb = rgb.astype(np.float32, copy=False)
    tile = max(min_tile, min(tile, max(rgb.shape[:2])))  # no identical retries on small images
    while True:
        try:
            return TiledResult(_run_once(model_fn, rgb, scale, tile, overlap), tile)
        except Exception as exc:
            if not _is_oom(exc):
                raise
            if tile // 2 < min_tile:
                raise MemoryError(f"out of memory at tile {tile}, floor is {min_tile}") from exc
        # outside the except block, so the traceback no longer pins GPU tensors
        _free_cuda_cache()
        tile //= 2
