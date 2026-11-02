"""Corner placement: move each detected corner sample to where the edge really turns.

Smoothing pulls a corner sample inwards. With sub-pixel samples nearby, the corner goes where lines
fitted to the raw samples on its two sides meet; otherwise it snaps to the nearest crack-grid
vertex of the original curve. Lengths here are original pixels, multiplied by `scale`.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from engine.types import Point

CORNER_SNAP_PX = 1.5  # a corner moves to the nearest original vertex within this distance
# Sub-pixel corners: a corner goes where lines fitted to the sub-pixel samples on its two sides
# meet. Samples right next to the corner are skipped: their pixels blend both sides.
CORNER_LINE_SAMPLES = 4  # samples per side for each line fit
CORNER_LINE_GAP = 1  # samples skipped on each side of the corner index
MIN_CORNER_LINE_SAMPLES = 2
MIN_CORNER_LINE_ANGLE_DEG = 20.0  # flatter corners make the intersection ill-conditioned


@dataclass(frozen=True)
class SideLines:
    """How the side lines of a sub-pixel corner are fitted.

    `gap` samples next to the corner are skipped (their pixels blend both sides) and up to
    `samples` per side are used. With `allow_behind` an intersection behind the smoothed samples
    next to the corner is accepted; the caller must then pull those samples onto the corner
    (see `collapse_behind`) or the fitted span doubles back.
    """

    gap: int = CORNER_LINE_GAP
    samples: int = CORNER_LINE_SAMPLES
    allow_behind: bool = False


DEFAULT_SIDE_LINES = (SideLines(),)


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


def line_intersection(side_a: np.ndarray, side_b: np.ndarray) -> Point | None:
    """Where total-least-squares lines through two point runs cross (None if near-parallel)."""
    return _line_intersection(side_a, side_b)


def _subpixel_corner(
    dense: Sequence[Point], raw: Sequence[Point], c: int, neighbours: tuple[int, int], scale: float,
    lines: SideLines = SideLines(),
) -> Point | None:
    """Intersection of lines fitted to the raw sub-pixel samples on each side of corner `c`.

    Uses up to `lines.samples` samples per side after `lines.gap`, stopping short of the
    neighbouring corners (and so of an open curve's junction endpoints). None if a side is too
    short, the lines are near-parallel, the intersection lies behind the samples of either side
    (the span would have to double back to reach it; see `SideLines.allow_behind`), or it is
    farther than CORNER_SNAP_PX from the smoothed corner.
    """
    n = len(raw)
    prev, nxt = neighbours
    reach = lines.gap + lines.samples
    before = range(max(prev + 1, c - reach), c - lines.gap)
    after = range(c + lines.gap + 1, min(nxt, c + reach + 1))
    if min(len(before), len(after)) < MIN_CORNER_LINE_SAMPLES:
        return None
    side_a = np.asarray([raw[i % n] for i in before], np.float64)
    side_b = np.asarray([raw[i % n] for i in after], np.float64)
    hit = _line_intersection(side_a, side_b)
    if hit is None:
        return None
    if _behind(hit, side_a, side_a[-1]) or _behind(hit, side_b[::-1], side_b[0]):
        return None
    before_corner, after_corner = dense[(c - 1) % n], dense[(c + 1) % n]
    if not lines.allow_behind and (
        _behind(hit, side_a, before_corner) or _behind(hit, side_b[::-1], after_corner)
    ):
        return None
    x, y = dense[c]
    return hit if math.hypot(hit[0] - x, hit[1] - y) <= CORNER_SNAP_PX * scale else None


def _has_subpixel_samples(
    raw: Sequence[Point], plain: Sequence[Point], c: int, closed: bool,
    reach: int = CORNER_LINE_GAP + CORNER_LINE_SAMPLES,
) -> bool:
    """True when any raw sample within `reach` of corner `c` was moved off the crack grid."""
    n = len(raw)
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
    lines: Sequence[SideLines] = DEFAULT_SIDE_LINES,
) -> list[Point]:
    """Move each of `corners` (a subset of `all_corners`) from its smoothed sample to the corner.

    `samples` is (raw, plain): the unsmoothed sub-pixel samples and their crack-grid originals.
    Near sub-pixel samples the corner is `_subpixel_corner`. Elsewhere, and wherever that finds
    no usable intersection, it is grid-snapped by `_snap_corners` exactly as without sub-pixel
    input (leaving the smoothed sample would keep the corner rounded in). Each entry of `lines`
    is tried in turn until one gives an intersection.
    """
    raw, plain = samples
    order = list(all_corners)
    hits: dict[int, Point] = {}
    for c in corners:
        neighbours = _corner_neighbours(order, order.index(c), len(dense), closed)
        for side_lines in lines:
            if not _has_subpixel_samples(raw, plain, c, closed, side_lines.gap + side_lines.samples):
                continue
            hit = _subpixel_corner(dense, raw, c, neighbours, scale, side_lines)
            if hit is not None:
                hits[c] = hit
                break
    out = _snap_corners(dense, [c for c in corners if c not in hits], original, scale)
    for c, hit in hits.items():
        out[c] = hit
    return out


def collapse_behind(
    seq: Sequence[Point], corners: Sequence[int], all_corners: Sequence[int], closed: bool,
    reach: int,
) -> list[Point]:
    """Pull samples next to each of `corners` that lie past it onto the corner.

    `corners` is a subset of `all_corners` (an open curve's end points must not be in it). A
    sample within `reach` of corner c is past it when it is ahead of c along the direction from
    the sample `reach` away on its side toward c; it becomes a duplicate of the corner, which
    the fitters drop, so the span does not double back.
    """
    out = list(seq)
    n = len(out)
    order = list(all_corners)
    for c in corners:
        neighbours = _corner_neighbours(order, order.index(c), n, closed)
        corner = np.asarray(out[c])
        for step, limit in zip((-1, 1), neighbours):
            far = c + step * reach
            if (limit - far) * step <= 0:
                continue  # the run to the next corner is too short to give a direction
            direction = corner - np.asarray(out[far % n])
            for i in range(c + step, far, step):
                if (np.asarray(out[i % n]) - corner) @ direction <= 0.0:
                    break
                out[i % n] = out[c]
    return out
