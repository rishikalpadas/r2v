"""Image-fidelity and complexity metrics for comparing rasters and SVGs."""
from __future__ import annotations

import io

import numpy as np
import resvg_py
import svgelements
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.color import deltaE_ciede2000, rgb2gray, rgb2lab
from skimage.feature import canny
from skimage.metrics import structural_similarity

ALPHA_THRESHOLD = 128


def render_svg(svg: str, width: int, height: int) -> np.ndarray:
    png = resvg_py.svg_to_bytes(svg_string=svg, width=width, height=height)
    return np.array(Image.open(io.BytesIO(bytes(png))).convert("RGBA"), dtype=np.uint8)


def _over_white(px: np.ndarray) -> np.ndarray:
    """Composite RGBA over white; returns float RGB in [0, 255]."""
    alpha = px[..., 3:4].astype(float) / 255.0
    return px[..., :3].astype(float) * alpha + 255.0 * (1.0 - alpha)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        structural_similarity(_over_white(a), _over_white(b), channel_axis=2, data_range=255)
    )


def mean_delta_e(a: np.ndarray, b: np.ndarray) -> float:
    lab_a = rgb2lab(_over_white(a) / 255.0)
    lab_b = rgb2lab(_over_white(b) / 255.0)
    return float(np.mean(deltaE_ciede2000(lab_a, lab_b)))


def _edges(px: np.ndarray) -> np.ndarray:
    return canny(rgb2gray(_over_white(px) / 255.0), sigma=1)


def edge_f1(a: np.ndarray, b: np.ndarray, tolerance_px: int = 2) -> float:
    ea, eb = _edges(a), _edges(b)
    if not ea.any() and not eb.any():
        return 1.0
    if not ea.any() or not eb.any():
        return 0.0
    near_a = distance_transform_edt(~ea) <= tolerance_px
    near_b = distance_transform_edt(~eb) <= tolerance_px
    precision = float(near_a[eb].mean())  # b edges that match an a edge
    recall = float(near_b[ea].mean())  # a edges that match a b edge
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def count_nodes(svg: str) -> tuple[int, int]:
    """Return (node_count, path_count). A Close counts only if it has nonzero length."""
    doc = svgelements.SVG.parse(io.StringIO(svg))
    nodes = paths = 0
    for el in doc.elements():
        if not isinstance(el, svgelements.Shape):
            continue
        paths += 1
        if isinstance(el, svgelements.Path):
            for seg in el:
                if isinstance(seg, svgelements.Move):
                    continue
                if isinstance(seg, svgelements.Close) and seg.start == seg.end:
                    continue
                nodes += 1
        elif isinstance(el, (svgelements.Polygon, svgelements.Polyline)):
            nodes += len(el.points)
        else:
            nodes += 1
    return nodes, paths


def alpha_agreement(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a[..., 3] >= ALPHA_THRESHOLD) == (b[..., 3] >= ALPHA_THRESHOLD)))
