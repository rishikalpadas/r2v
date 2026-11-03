"""Enhance stage: JPEG deblock, denoise and 4x super-resolution before segmentation.

Every ML step falls back to a bicubic upscale (reason recorded in the report); this stage never
raises for ML problems. Torch is never imported here: models come from engine.ml as
numpy-in/numpy-out callables.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable

import numpy as np
from PIL import Image
from scipy import ndimage

from engine.ml import ml_available
from engine.ml.device import pick_device
from engine.ml.models import load_model
from engine.ml.tiling import run_tiled
from engine.presets import EnhanceParams
from engine.types import RasterImage

DEBLOCK_MODEL = "fbcnn_color"
DENOISE_MODEL = "scunet_color_real_psnr"
MIN_ML_SIDE = 16  # below this short side the models have too little context: bicubic
FLAT_MAX_COLORS = 64  # unique colours after 5-bit quantization that still count as flat art
ALPHA_THRESHOLD = 128
OVERLAP = 32


def _report(**values: object) -> dict[str, object]:
    out: dict[str, object] = dict.fromkeys(
        ("model_used", "fallback_reason", "final_tile", "effective_scale", "device",
         "skipped_reason")
    )
    out.update(values)
    return out


def effective_scale(params: EnhanceParams, width: int, height: int) -> int:
    """params.scale, reduced so the long side stays within max_output_side; at least 1."""
    return max(1, min(params.scale, params.max_output_side // max(width, height)))


def _is_exact_colour(px: np.ndarray, max_colors: int) -> bool:
    """Alpha only 0/255 and at most max_colors distinct RGB colours among opaque pixels.

    Such input takes segment's exact-colour path, where there is nothing to deblock and no
    anti-aliasing to place sub-pixel; upscaling only shifts flat colours (measured on
    flat_shapes: SSIM -0.005..-0.011 and roughly double the dE with ESRGAN 4x).
    """
    alpha = px[..., 3]
    if not np.isin(alpha, (0, 255)).all():
        return False
    rgb = px[..., :3][alpha == 255].astype(np.uint32)
    packed = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
    return len(np.unique(packed)) <= max_colors


def _is_flat(rgb: np.ndarray, alpha: np.ndarray) -> bool:
    """At most FLAT_MAX_COLORS colours after 5-bit-per-channel quantization (visible pixels)."""
    q = (rgb[alpha > 0] >> 3).astype(np.uint32)
    packed = (q[:, 0] << 10) | (q[:, 1] << 5) | q[:, 2]
    return len(np.unique(packed)) <= FLAT_MAX_COLORS


def _fill_transparent(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Give fully transparent pixels the colour of the nearest visible pixel, so their
    arbitrary RGB cannot bleed into shape edges through the models' receptive field."""
    hidden = alpha == 0
    if not hidden.any() or hidden.all():
        return rgb
    idx = ndimage.distance_transform_edt(hidden, return_distances=False, return_indices=True)
    return rgb[idx[0], idx[1]]


def _resize(arr: np.ndarray, size: tuple[int, int], method: Image.Resampling) -> np.ndarray:
    """Resize an HxW or HxWxC uint8/float32 array to size=(w, h) channel by channel."""
    if arr.ndim == 2:
        return np.asarray(Image.fromarray(arr).resize(size, method))
    return np.stack([_resize(arr[..., c], size, method) for c in range(arr.shape[2])], axis=-1)


def _upscale_alpha(alpha: np.ndarray, s: int) -> np.ndarray:
    h, w = alpha.shape
    up = _resize(alpha, (w * s, h * s), Image.Resampling.BICUBIC)
    return np.where(up >= ALPHA_THRESHOLD, 255, 0).astype(np.uint8)


def _bicubic(rgb: np.ndarray, s: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    return np.asarray(Image.fromarray(rgb).resize((w * s, h * s), Image.Resampling.BICUBIC))


def _rescaled(model_fn: Callable[[np.ndarray], np.ndarray], s: int) -> Callable:
    """Wrap a model so each tile's output is box-downsampled to s x its input (s < model scale).
    Done per tile, so a capped image never materializes at the model's full 4x."""

    def run(rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        return _resize(model_fn(rgb), (w * s, h * s), Image.Resampling.BOX).astype(np.float32)

    return run


def _ml_rgb(rgb: np.ndarray, alpha: np.ndarray, image: RasterImage, params: EnhanceParams,
            s: int, device: str) -> tuple[np.ndarray, list[str], int]:
    """Run deblock (JPEG only) -> denoise (if enabled) -> SR. Returns (uint8 RGB, names, tile)."""
    steps = []
    if params.deblock and image.source_format == "JPEG":
        steps.append(DEBLOCK_MODEL)
    if params.denoise:
        steps.append(DENOISE_MODEL)
    x = rgb.astype(np.float32) / 255.0
    names: list[str] = []
    tiles: list[int] = []

    def apply(name: str, scale: int | None = None) -> np.ndarray:
        model = load_model(name, device)
        fn, out_scale = model, model.scale
        if scale is not None and scale != model.scale:
            fn, out_scale = _rescaled(model, scale), scale
        tile = params.tile * 2 if model.tiling == "DISCOURAGED" else params.tile
        fn.tiling = model.tiling  # type: ignore[attr-defined]
        result = run_tiled(fn, x, out_scale, tile, overlap=OVERLAP)
        names.append(name)
        tiles.append(result.final_tile)
        return result.image

    for name in steps:
        x = apply(name)
    restored = np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)
    sr = params.sr_model_flat if _is_flat(restored, alpha) else params.sr_model_general
    x = apply(sr, s)
    return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8), names, min(tiles)


def _fallback_reason(exc: Exception) -> str:
    if isinstance(exc, FileNotFoundError):
        return f"weights missing: {exc}"
    return f"model error ({type(exc).__name__})"


def enhance(
    image: RasterImage, params: EnhanceParams, max_colors: int | None = None
) -> tuple[RasterImage, dict[str, object]]:
    """Clean and upscale `image`; returns (enhanced image, report dict for the StageReport).

    `max_colors` is segment's exact-colour bound (the runner passes `SegmentParams.max_colors`):
    input that segment would treat as exact colour is returned unchanged. None disables that skip.
    The output has scale `image.scale * effective_scale` and the same original_size.
    """
    px = image.pixels
    h, w = px.shape[:2]
    if not params.enabled:
        return image, _report(effective_scale=1, skipped_reason="disabled")
    s = effective_scale(params, w, h)
    if s == 1:
        return image, _report(effective_scale=1, skipped_reason="scale 1")
    if max_colors is not None and _is_exact_colour(px, max_colors):
        return image, _report(effective_scale=1, skipped_reason="exact colours")

    alpha = px[..., 3]
    rgb = _fill_transparent(px[..., :3], alpha)
    report = _report(effective_scale=s, model_used="bicubic")
    out_rgb = None
    if min(h, w) < MIN_ML_SIDE:
        report["fallback_reason"] = "too small"
    elif not ml_available():
        report["fallback_reason"] = "ml unavailable"
    else:
        device = pick_device(params.device)
        report["device"] = device
        try:
            out_rgb, names, tile = _ml_rgb(rgb, alpha, image, params, s, device)
        except Exception as exc:  # noqa: BLE001 - any ML failure falls back to bicubic
            report["fallback_reason"] = _fallback_reason(exc)
        else:
            report.update(model_used="+".join(names), final_tile=tile)
    if out_rgb is None:
        out_rgb = _bicubic(rgb, s)
    pixels = np.concatenate([out_rgb, _upscale_alpha(alpha, s)[..., None]], axis=2)
    return replace(image, pixels=pixels, scale=image.scale * s), report
