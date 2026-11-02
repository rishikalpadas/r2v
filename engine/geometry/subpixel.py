"""Sub-pixel boundary placement from anti-aliased edge pixels.

Boundary curves sit on the integer crack grid: every pixel belongs to exactly one region, so an
edge that really runs 30% of the way across a pixel is traced along that pixel's side. The edge
pixels themselves still record where the edge is: a pixel 30% covered by region R is (close to) a
30/70 blend of R's colour and its neighbour's. This module reads that blend back.

For each dense sample (one per unit crack edge, the layout of `corners.dense_samples`) between
pixel `a` (left side, label L) and pixel `b` (right side, label R), see `BoundaryCurve` for the
orientation convention:

* Walk outward along the edge normal over up to `WINDOW` pixels per side (a1 = a, a2, ...;
  b1 = b, b2, ...). Beyond the first pixel a pixel is used only while it carries its side's label
  and its R-coverage does not move back toward the crack (coverage of R falls going into A and
  rises going into B), so the far edge of a thin stroke is never read as this edge's blend.
* Each pixel's coverage of R, t, is its colour projected onto the segment palette[L] ->
  palette[R] in sRGB (where box-filter anti-aliasing and JPEG blend colours), clamped to [0, 1].
  A pixel that is not close to any such blend (perceptual distance > `RESIDUAL_MAX_DE`) is
  treated as pure. Against TRANSPARENT the coverage comes from alpha instead.
* With m B-side pixels used, the window holds m - e of R when the edge sits at e (measured from
  the crack toward B), so e = m - sum(t), clamped to +-`OFFSET_LIMIT`.
* Pairs whose palette colours are closer than `MIN_CONTRAST_DE` are not moved.

Non-anti-aliased input (every pixel exactly its region's palette colour) gives offsets of exactly
0, so the samples equal `dense_samples` bit for bit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.color import rgb2lab

from engine.geometry.corners import dense_samples
from engine.types import OUTSIDE, TRANSPARENT, BoundaryCurve, Point, RasterImage, RegionMap

WINDOW = 2  # pixels read per side: a 30-60 degree edge spreads its blend over ~2 pixels
OFFSET_LIMIT = 1.0  # px; segmentation puts the crack within ~0.5 px of the true edge
RESIDUAL_MAX_DE = 12.0  # CIE76; farther from every L/R blend -> another colour, not a blend
MIN_CONTRAST_DE = 10.0  # CIE76; below this, blend fractions drown in noise and quantisation
_SIDE_STEPS = np.arange(-WINDOW, WINDOW) + 0.5  # pixel centres along the normal: a2 a1 b1 b2
_A1, _B1 = WINDOW - 1, WINDOW


@dataclass(frozen=True, eq=False)  # holds ndarrays: compare by identity
class EdgeContext:
    """Per-image data the offset estimator reads (views of the processing image, no copies)."""

    labels: np.ndarray  # HxW int32 region ids, TRANSPARENT = -1
    rgb: np.ndarray  # HxWx3 uint8
    alpha: np.ndarray  # HxW uint8
    palette_rgb: np.ndarray  # Kx3 float64
    palette_lab: np.ndarray  # Kx3 float64


def make_edge_context(image: RasterImage, regions: RegionMap) -> EdgeContext:
    labels = np.asarray(regions.labels)
    pixels = np.asarray(image.pixels)
    if pixels.ndim != 3 or pixels.shape[2] != 4 or labels.shape != pixels.shape[:2]:
        raise ValueError(
            f"labels {labels.shape} do not match image pixels {pixels.shape} (expected HxW, HxWx4)"
        )
    return EdgeContext(
        labels=labels,
        rgb=pixels[..., :3],
        alpha=pixels[..., 3],
        palette_rgb=np.asarray(regions.palette_rgb, np.float64).reshape(-1, 3),
        palette_lab=np.asarray(regions.palette_lab, np.float64).reshape(-1, 3),
    )


def _right_normals(points: tuple[Point, ...], closed: bool) -> np.ndarray:
    """Unit normal toward the `right` side for every midpoint of `dense_samples(points, closed)`."""
    pts = np.asarray(points, np.float64)
    if closed:
        if tuple(pts[0]) == tuple(pts[-1]):
            pts = pts[:-1]
        pts = np.vstack([pts, pts[:1]])
    vec = np.diff(pts, axis=0)
    length = np.hypot(vec[:, 0], vec[:, 1])
    pieces = np.maximum(1, np.rint(length)).astype(np.int64)
    unit = vec / np.where(length > 0, length, 1.0)[:, None]
    normals = np.stack([unit[:, 1], -unit[:, 0]], axis=1)  # right of (dx, dy) is (dy, -dx)
    return np.repeat(normals, pieces, axis=0)


def _lab(rgb: np.ndarray) -> np.ndarray:
    return rgb2lab(rgb.reshape(-1, 1, 3) / 255.0).reshape(rgb.shape)


def _opaque_coverage(rgb: np.ndarray, left: np.ndarray, right: np.ndarray, ctx: EdgeContext):
    """R-coverage of each window pixel (N x 4) for opaque L/R pairs, plus the pair contrast."""
    p_l, p_r = ctx.palette_rgb[left], ctx.palette_rgb[right]
    d = p_r - p_l
    dd = np.einsum("ij,ij->i", d, d)
    rel = rgb - p_l[:, None, :]
    t = np.einsum("ijk,ik->ij", rel, d) / np.where(dd > 0, dd, 1.0)[:, None]
    t = np.clip(t, 0.0, 1.0)
    blend = p_l[:, None, :] + t[..., None] * d[:, None, :]
    residual = np.linalg.norm(_lab(rgb) - _lab(blend), axis=2)
    pure = np.broadcast_to(_SIDE_STEPS > 0, t.shape).astype(np.float64)  # A: 0, B: 1
    t = np.where(residual > RESIDUAL_MAX_DE, pure, t)
    contrast = np.linalg.norm(ctx.palette_lab[left] - ctx.palette_lab[right], axis=1)
    return t, contrast


def _offsets(mids: np.ndarray, normals: np.ndarray, ctx: EdgeContext) -> np.ndarray:
    """Signed edge offset along `normals` (toward the right side) for every crack-edge midpoint."""
    h, w = ctx.labels.shape
    centres = mids[:, None, :] + _SIDE_STEPS[None, :, None] * normals[:, None, :]
    ix = np.floor(centres[..., 0]).astype(np.int64)
    iy = np.floor(centres[..., 1]).astype(np.int64)
    inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    ix, iy = np.clip(ix, 0, w - 1), np.clip(iy, 0, h - 1)
    labels = np.where(inside, ctx.labels[iy, ix], OUTSIDE)
    left, right = labels[:, _A1], labels[:, _B1]
    valid = (left != OUTSIDE) & (right != OUTSIDE) & (left != right)

    t = np.zeros(labels.shape)
    readable = np.zeros(len(mids), bool)
    opaque = valid & (left >= 0) & (right >= 0)
    if opaque.any():
        rgb = ctx.rgb[iy[opaque], ix[opaque]].astype(np.float64)
        t[opaque], contrast = _opaque_coverage(rgb, left[opaque], right[opaque], ctx)
        readable[opaque] = contrast >= MIN_CONTRAST_DE
    clear = valid & ((left == TRANSPARENT) | (right == TRANSPARENT))
    if clear.any():
        opacity = ctx.alpha[iy[clear], ix[clear]] / 255.0
        t[clear] = np.where((right[clear] == TRANSPARENT)[:, None], 1.0 - opacity, opacity)
        readable[clear] = True

    # Outer pixels: same label as their side and coverage monotone away from the crack.
    use = np.ones(labels.shape, bool)
    for k in range(_A1):  # A side, outward = decreasing index
        use[:, k] = (labels[:, k] == left) & (t[:, k] <= t[:, k + 1]) & use[:, k + 1]
    for k in range(_B1 + 1, 2 * WINDOW):
        use[:, k] = (labels[:, k] == right) & (t[:, k] >= t[:, k - 1]) & use[:, k - 1]
    m = use[:, _B1:].sum(axis=1)
    e = m - np.where(use, t, 0.0).sum(axis=1)
    e = np.clip(e, -OFFSET_LIMIT, OFFSET_LIMIT)  # k, m >= 1, so this is inside (-k, m) too
    return np.where(valid & readable, e, 0.0)


def subpixel_dense(curve: BoundaryCurve, ctx: EdgeContext) -> tuple[Point, ...]:
    """`dense_samples` of `curve` with each midpoint moved to the anti-aliased edge position.

    Open curves keep their exact junction endpoints. Curves on the canvas edge (OUTSIDE) and
    samples without a readable blend are returned unmoved.
    """
    dense = dense_samples(curve.points, curve.closed)
    if OUTSIDE in (curve.left, curve.right) or len(curve.points) < 2:
        return dense
    inner = dense if curve.closed else dense[1:-1]
    mids = np.asarray(inner, np.float64)
    normals = _right_normals(curve.points, curve.closed)
    offsets = _offsets(mids, normals, ctx)
    if not offsets.any():
        return dense
    moved = mids + offsets[:, None] * normals
    points = tuple(
        p if o == 0 else (float(x), float(y)) for p, o, (x, y) in zip(inner, offsets, moved)
    )
    return points if curve.closed else (dense[0], *points, dense[-1])
