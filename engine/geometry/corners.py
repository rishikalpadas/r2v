"""Polyline smoothing and corner detection for crack-grid boundary curves.

Input curves come from `build_boundary_graph`: points sit on the integer pixel-corner grid and
collinear runs are collapsed, so a straight 20 px edge is only its two endpoints. Both functions
here therefore work on a *dense* representation with one sample per unit crack edge.

Conventions
-----------
* `smooth_polyline` resamples at unit-edge midpoints (`dense_samples`), Gaussian-smooths the
  coordinates (`smooth_dense`) and returns the smoothed dense sequence. The two halves are
  separate so callers can move the dense samples (e.g. to sub-pixel edges) before smoothing.
* Closed curves are treated as cyclic and the output does NOT repeat the first point (unlike
  `BoundaryCurve.points`, which does). `find_corners(closed=True)` expects that same convention.
* Open curves keep their original first and last points exactly (they are shared junctions); the
  smoothed interior sits between them.
* Corner indices refer to the sequence returned by `smooth_polyline`.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

from engine.types import Point

_MIN_CLOSED_POINTS = 3
# Scale-space corners (find_corners_multiscale)
WINDOW_PER_SIGMA = 2.0  # turning-angle window at the coarse scale: +-2 sigma spans the blur
TRACK_RADIUS = 2  # samples; a coarse corner response this close confirms a fine corner


def samples_at_scale(count: int, scale: float) -> int:
    """A count of dense samples at scale 1 as the count covering the same length at `scale`.

    Dense samples are one per processing pixel, so a window of `count` samples spans `count`
    original pixels only when multiplied by the processing scale. Exactly `count` at scale 1.
    Pipeline scales are integers (enhance factors 1-4), so the product is exact; for other
    scales a half-integer product rounds half to even (Python `round`), e.g. 2.5 -> 2.
    """
    return max(1, round(count * scale))


def unit_pieces(a: Point, b: Point) -> int:
    """Number of unit-length pieces segment a->b is resampled into (at least one)."""
    return max(1, round(math.hypot(b[0] - a[0], b[1] - a[1])))


def _unit_midpoints(a: Point, b: Point) -> list[Point]:
    """Midpoints of the unit-length pieces of segment a->b (at least one piece)."""
    pieces = unit_pieces(a, b)
    dx, dy = (b[0] - a[0]) / pieces, (b[1] - a[1]) / pieces
    return [(a[0] + dx * (k + 0.5), a[1] + dy * (k + 0.5)) for k in range(pieces)]


def dense_samples(points: tuple[Point, ...], closed: bool) -> tuple[Point, ...]:
    """One sample per unit edge, at its midpoint.

    Open: the exact first and last points are added at the ends. Closed: cyclic, the first
    sample is not repeated (a repeated closing point in `points` is ignored).
    """
    if len(points) < 2:
        return tuple(points)
    ring = points[:-1] if closed and points[0] == points[-1] else points
    segments = zip(ring, ring[1:] + ring[:1]) if closed else zip(points, points[1:])
    dense: list[Point] = []
    for a, b in segments:
        dense.extend(_unit_midpoints(a, b))
    if not closed:
        dense = [points[0], *dense, points[-1]]
    return tuple(dense)


def smooth_dense(dense: tuple[Point, ...], closed: bool, sigma: float = 1.0) -> tuple[Point, ...]:
    """Gaussian-smooth a dense sequence (see `dense_samples`); open endpoints stay exactly fixed."""
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    if (closed and len(dense) < _MIN_CLOSED_POINTS) or len(dense) < 2:
        return tuple(dense)
    arr = np.asarray(dense, np.float64)
    if sigma > 0:
        arr = gaussian_filter1d(arr, sigma, axis=0, mode="wrap" if closed else "nearest")
    out = [(float(x), float(y)) for x, y in arr]
    if not closed:
        out[0], out[-1] = dense[0], dense[-1]
    return tuple(out)


def smooth_polyline(points: tuple[Point, ...], closed: bool, sigma: float = 1.0) -> tuple[Point, ...]:
    """Resample at unit-edge midpoints and Gaussian-smooth; open endpoints stay exactly fixed."""
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    if len(points) < 2:
        return tuple(points)
    return smooth_dense(dense_samples(points, closed), closed, sigma)


def _turning_angles(arr: np.ndarray, closed: bool, window: int) -> np.ndarray:
    """Angle (degrees) between vectors to the +-window neighbors; NaN where undefined (open ends)."""
    n = len(arr)
    idx = np.arange(n)
    prev_i, next_i = idx - window, idx + window
    valid = np.ones(n, bool)
    if closed:
        prev_i, next_i = prev_i % n, next_i % n
    else:
        valid = (prev_i >= 0) & (next_i < n)
        prev_i, next_i = np.clip(prev_i, 0, n - 1), np.clip(next_i, 0, n - 1)
    v1 = arr[idx] - arr[prev_i]
    v2 = arr[next_i] - arr[idx]
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    dot = np.einsum("ij,ij->i", v1, v2)
    angles = np.degrees(np.abs(np.arctan2(cross, dot)))
    return np.where(valid, angles, np.nan)


def find_corners(
    points: tuple[Point, ...], closed: bool, angle_deg: float = 35.0, window: int = 3
) -> tuple[int, ...]:
    """Indices whose turning angle exceeds `angle_deg` and is a local maximum within +-window.

    `points` is a smoothed dense sequence (see `smooth_polyline`; closed = cyclic, no repeated
    first point). Open curves always include 0 and len-1.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    n = len(points)
    if n == 0:
        return ()
    if n == 1:
        return (0,)
    arr = np.asarray(points, np.float64)
    if closed and n < 2 * window + 1:
        return ()
    angles = _turning_angles(arr, closed, window)
    found: list[int] = []
    for i in range(n):
        a = angles[i]
        if not a > angle_deg:
            continue
        neighbors = [(i + d) % n if closed else i + d for d in range(-window, window + 1) if d]
        if any(0 <= j < n and angles[j] > a for j in neighbors):
            continue
        # Plateau tie: keep only the first index of equal-angle neighbors on the left.
        left = [(i - d) % n if closed else i - d for d in range(1, window + 1)]
        if any(0 <= j < n and angles[j] == a for j in left):
            continue
        found.append(i)
    if not closed:
        found = sorted({0, n - 1, *found})
    return tuple(found)


class MultiscaleCorners(NamedTuple):
    """Corner indices (fine-scale positions) and each one's coarse-scale turning angle.

    Open curves include their end points, whose coarse angle is NaN.
    """

    indices: tuple[int, ...]
    coarse_angles: tuple[float, ...]


def _track_window(i: int, n: int, closed: bool) -> np.ndarray:
    """Sample indices within TRACK_RADIUS of i (an open curve's end points excluded)."""
    idx = np.arange(i - TRACK_RADIUS, i + TRACK_RADIUS + 1)
    return idx % n if closed else idx[(idx > 0) & (idx < n - 1)]


def _coarse_response(angles: np.ndarray, i: int, n: int, closed: bool) -> float:
    """Largest coarse turning angle within TRACK_RADIUS of sample i (NaN where undefined)."""
    window = angles[_track_window(i, n, closed)]
    return float(np.nanmax(window)) if np.isfinite(window).any() else math.nan


def _track_to_fine(fine_angles: np.ndarray, i: int, n: int, closed: bool) -> int:
    """Index of the strongest fine-scale turn within TRACK_RADIUS of coarse corner i."""
    idx = _track_window(i, n, closed)
    window = fine_angles[idx]
    return int(idx[int(np.nanargmax(window))]) if np.isfinite(window).any() else i


def _dedupe_close(found: Sequence[int], fine_angles: np.ndarray, gap: int) -> list[int]:
    """Sorted unique indices; of two closer than `gap` samples the weaker fine turn is dropped."""
    out: list[int] = []
    for i in sorted(set(found)):
        if out and i - out[-1] < gap:
            if fine_angles[i] > fine_angles[out[-1]]:
                out[-1] = i
            continue
        out.append(i)
    return out


def find_corners_multiscale(
    raw: Sequence[Point],
    closed: bool,
    angle_deg: float,
    fine_sigma: float,
    coarse_sigma: float,
    fine_window: int = 2,
) -> MultiscaleCorners:
    """Corners that persist from the fine scale up to the coarse scale.

    `raw` is an unsmoothed dense sequence (see `dense_samples`). At the coarse scale (`raw`
    smoothed at `coarse_sigma`, turning angles over WINDOW_PER_SIGMA * coarse_sigma samples)
    jitter bumps flatten out while real corners keep their turn. A corner is

    * a coarse-scale corner (a local maximum turning by more than `angle_deg`), tracked down to
      the strongest fine-scale turn within TRACK_RADIUS samples, or
    * a fine-scale corner (`find_corners` at `fine_sigma`, window `fine_window`) whose coarse turn
      within TRACK_RADIUS also exceeds `angle_deg` -- so the two corners of a narrow end, which
      the coarse scale sees as one, both survive -- or where the coarse turn is undefined (too
      close to an open curve's end for the coarse window), which the coarse scale cannot judge.

    Indices are fine-scale positions; corners closer than `fine_window` collapse into one.
    """
    pts = tuple((float(x), float(y)) for x, y in raw)
    n = len(pts)
    window = max(fine_window, math.ceil(WINDOW_PER_SIGMA * coarse_sigma))
    if closed and n < 2 * window + 1:
        return MultiscaleCorners((), ())  # too short to show a coarse-scale turn
    fine_pts = smooth_dense(pts, closed, fine_sigma)
    fine = find_corners(fine_pts, closed, angle_deg, fine_window)
    ends = [i for i in fine if not closed and i in (0, n - 1)]
    if n < 3:
        return MultiscaleCorners(tuple(ends), tuple(math.nan for _ in ends))
    fine_angles = _turning_angles(np.asarray(fine_pts, np.float64), closed, fine_window)
    coarse_pts = smooth_dense(pts, closed, coarse_sigma)
    coarse_angles = _turning_angles(np.asarray(coarse_pts, np.float64), closed, window)
    tracked = [
        _track_to_fine(fine_angles, i, n, closed)
        for i in find_corners(coarse_pts, closed, angle_deg, window)
        if closed or 0 < i < n - 1
    ]

    def response(i: int) -> float:  # NaN: an end, or the coarse window does not fit at i
        if i in ends or math.isnan(coarse_angles[i]):
            return math.nan
        return _coarse_response(coarse_angles, i, n, closed)

    confirmed = [i for i in fine if i not in ends and not response(i) <= angle_deg]
    corners = sorted(ends + _dedupe_close(tracked + confirmed, fine_angles, fine_window))
    responses = [response(i) for i in corners]
    return MultiscaleCorners(tuple(corners), tuple(responses))


def even_breaks(n: int, spans: int) -> list[int]:
    """Indices 0..n splitting a corner-free ring of n samples into `spans` near-equal runs."""
    return [n * k // spans for k in range(spans + 1)]


def smooth_between(
    points: Sequence[Point], corners: Sequence[int], closed: bool, sigma: float
) -> tuple[Point, ...]:
    """Gaussian-smooth each corner-to-corner run on its own; corner samples stay exactly put.

    Smoothing never mixes samples across a corner, so it cannot round one off. `corners` are
    sorted indices into `points`; for an open curve they must include both end points. A closed
    curve without corners is smoothed as one ring (`smooth_dense`).
    """
    pts = tuple(points)
    n = len(pts)
    if not corners:
        return smooth_dense(pts, closed, sigma)
    order = sorted(corners)
    out = list(pts)
    ends = list(zip(order, order[1:]))
    if closed:
        ends.append((order[-1], order[0] + n))
    for a, b in ends:
        if b - a < 2:
            continue
        run = tuple(pts[i % n] for i in range(a, b + 1))
        for i, p in zip(range(a + 1, b), smooth_dense(run, False, sigma)[1:-1]):
            out[i % n] = p
    return tuple(out)
