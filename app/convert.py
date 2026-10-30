"""Shared conversion used by the CLI and the preview server."""
from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass

from engine import to_svg, vectorize
from engine.loader import load_image
from eval.metrics import count_nodes, mean_delta_e, render_svg, ssim

# skimage's SSIM window needs at least 7 px per side.
MIN_SSIM_SIDE = 7

OPTION_KEYS: Mapping[str, str] = {
    "max_colors": "segment.max_colors",
    "layer_mode": "export.layer_mode",
    "fit": "fit.enabled",
    "tolerance": "fit.tolerance_px",
}


@dataclass(frozen=True)
class Conversion:
    svg: str
    width: int
    height: int
    nodes: int
    paths: int
    seconds: float
    ssim: float | None
    delta_e: float | None
    stage_seconds: Mapping[str, float]


def to_overrides(options: Mapping[str, object]) -> dict[str, object]:
    """Map friendly option names to engine override keys, dropping unset ones."""
    return {OPTION_KEYS[k]: v for k, v in options.items() if k in OPTION_KEYS and v is not None}


def convert(image_bytes: bytes, options: Mapping[str, object] | None = None) -> Conversion:
    """Vectorize `image_bytes` and score the result against the source.

    Raises engine InputError / StageError unchanged.
    """
    start = time.perf_counter()
    result = vectorize(image_bytes, overrides=to_overrides(options or {}))
    seconds = time.perf_counter() - start
    svg = to_svg(result.doc)
    nodes, paths = count_nodes(svg)

    source = load_image(image_bytes).pixels
    height, width = source.shape[:2]
    score = delta = None
    if min(width, height) >= MIN_SSIM_SIDE:
        rendered = render_svg(svg, width, height)
        score = ssim(source, rendered)
        delta = mean_delta_e(source, rendered)

    return Conversion(
        svg=svg,
        width=result.doc.width,
        height=result.doc.height,
        nodes=nodes,
        paths=paths,
        seconds=seconds,
        ssim=score,
        delta_e=delta,
        stage_seconds={r.stage: r.duration_s for r in result.reports},
    )
