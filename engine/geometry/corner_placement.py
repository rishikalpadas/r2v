"""Corner placement: move each detected corner sample to where the edge really turns.

Smoothing pulls a corner sample inwards. With sub-pixel samples nearby, the corner goes where lines
fitted to the raw samples on its two sides meet; otherwise it snaps to the nearest crack-grid
vertex of the original curve. Lengths here are original pixels, multiplied by `scale`.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from engine.types import Point

CORNER_SNAP_PX = 1.5  # a corner moves to the nearest original vertex within this distance
# Sub-pixel corners: a corner goes where lines fitted to the sub-pixel samples on its two sides
# meet. Samples right next to the corner are skipped: their pixels blend both sides.
CORNER_LINE_SAMPLES = 4  # samples per side for each line fit
CORNER_LINE_GAP = 1  # samples skipped on each side of the corner index
MIN_CORNER_LINE_SAMPLES = 2
MIN_CORNER_LINE_ANGLE_DEG = 20.0  # flatter corners make the intersection ill-conditioned


def _snap_corners(
    dense: Sequence[Point], corners: Sequence[int], original: Sequence[Point], scale: float
) -> list[Point]:
    """Replace each corner sample by the nearest original crack-grid vertex, if close.

    Smoothing pulls a corner inwards; fitting between the true vertices keeps straight edges
    where the pixels put them.
    """
    out = list(dense)
    for c in corners:
        x, y = dense[c]
        best = min(original, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)
        if math.hypot(best[0] - x, best[1] - y) <= CORNER_SNAP_PX * scale:
            out[c] = best
    return out


def _fit_direction(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Total-least-squares line through `pts`: (centroid, unit direction)."""
    centroid = pts.mean(axis=0)
    return centroid, np.linalg.svd(pts - centroid, full_matrices=False)[2][0]


def _line_intersection(side_a: np.ndarray, side_b: np.ndarray) -> Point | None:
    """Where the lines fitted to two point runs cross; None when they are too close to parallel."""
    pa, da = _fit_direction(side_a)
    pb, db = _fit_direction(side_b)
    cross = da[0] * db[1] - da[1] * db[0]
    if abs(cross) < math.sin(math.radians(MIN_CORNER_LINE_ANGLE_DEG)):
        return None
    diff = pb - pa
    s = (diff[0] * db[1] - diff[1] * db[0]) / cross
    return float(pa[0] + s * da[0]), float(pa[1] + s * da[1])


def _behind(hit: Point, run: np.ndarray, nearest: Point) -> bool:
    """True when `hit` is not ahead of `run`'s last point or of `nearest` along the run.

    `run` is ordered toward the corner; `nearest` is the smoothed sample next to the corner on
    that side, which the fitted span also has to pass through before reaching the corner.
    """
    direction = run[-1] - run[0]
    h = np.asarray(hit)
    return bool((h - run[-1]) @ direction < 0.0 or (h - np.asarray(nearest)) @ direction < 0.0)


def _corner_neighbours(corners: Sequence[int], k: int, n: int, closed: bool) -> tuple[int, int]:
    """Indices of the corners before and after corners[k], unwrapped around a closed ring."""
    if not closed:
        return corners[k - 1], corners[k + 1]
    prev = corners[k - 1] - (n if k == 0 else 0)
    nxt = corners[(k + 1) % len(corners)] + (n if k == len(corners) - 1 else 0)
    return prev, nxt


def _subpixel_corner(
    dense: Sequence[Point], raw: Sequence[Point], c: int, neighbours: tuple[int, int], scale: float
) -> Point | None:
    """Intersection of lines fitted to the raw sub-pixel samples on each side of corner `c`.

    Uses up to CORNER_LINE_SAMPLES samples per side, stopping short of the neighbouring corners
    (and so of an open curve's junction endpoints). None if a side is too short, the lines are
    near-parallel, the intersection lies behind the samples of either side (the span would have
    to double back to reach it), or it is farther than CORNER_SNAP_PX from the smoothed corner.
    """
    n = len(raw)
    prev, nxt = neighbours
    reach = CORNER_LINE_GAP + CORNER_LINE_SAMPLES
    before = range(max(prev + 1, c - reach), c - CORNER_LINE_GAP)
    after = range(c + CORNER_LINE_GAP + 1, min(nxt, c + reach + 1))
    if min(len(before), len(after)) < MIN_CORNER_LINE_SAMPLES:
        return None
    side_a = np.asarray([raw[i % n] for i in before], np.float64)
    side_b = np.asarray([raw[i % n] for i in after], np.float64)
    hit = _line_intersection(side_a, side_b)
    if hit is None:
        return None
    before_corner, after_corner = dense[(c - 1) % n], dense[(c + 1) % n]
    if _behind(hit, side_a, before_corner) or _behind(hit, side_b[::-1], after_corner):
        return None
    x, y = dense[c]
    return hit if math.hypot(hit[0] - x, hit[1] - y) <= CORNER_SNAP_PX * scale else None


def _has_subpixel_samples(
    raw: Sequence[Point], plain: Sequence[Point], c: int, closed: bool
) -> bool:
    """True when any raw sample within reach of corner `c` was moved off the crack grid."""
    n = len(raw)
    reach = CORNER_LINE_GAP + CORNER_LINE_SAMPLES
    window = range(c - reach, c + reach + 1)
    if closed:
        return any(raw[i % n] != plain[i % n] for i in window)
    return any(raw[i] != plain[i] for i in window if 0 <= i < n)


def place_corners(
    dense: Sequence[Point],
    corners: Sequence[int],
    all_corners: Sequence[int],
    samples: tuple[Sequence[Point], Sequence[Point]],
    original: Sequence[Point],
    closed: bool,
    scale: float,
) -> list[Point]:
    """Move each of `corners` (a subset of `all_corners`) from its smoothed sample to the corner.

    `samples` is (raw, plain): the unsmoothed sub-pixel samples and their crack-grid originals.
    Near sub-pixel samples the corner is `_subpixel_corner`. Elsewhere, and wherever that finds
    no usable intersection, it is grid-snapped by `_snap_corners` exactly as without sub-pixel
    input (leaving the smoothed sample would keep the corner rounded in).
    """
    raw, plain = samples
    order = list(all_corners)
    hits: dict[int, Point] = {}
    for c in corners:
        if not _has_subpixel_samples(raw, plain, c, closed):
            continue
        neighbours = _corner_neighbours(order, order.index(c), len(dense), closed)
        hit = _subpixel_corner(dense, raw, c, neighbours, scale)
        if hit is not None:
            hits[c] = hit
    out = _snap_corners(dense, [c for c in corners if c not in hits], original, scale)
    for c, hit in hits.items():
        out[c] = hit
    return out
