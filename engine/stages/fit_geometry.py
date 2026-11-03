"""Fit stage: turn boundary-graph curves into segment lists."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from engine.geometry.corners import dense_samples, find_corners, samples_at_scale, smooth_dense
from engine.geometry.crossing_guard import SpanPlan, guard_crossings, plan_spans
from engine.geometry.primitives import detect_primitive
from engine.geometry.segment_fit import fit_span
from engine.geometry.subpixel import EdgeContext, subpixel_dense
from engine.presets import FitParams
from engine.types import OUTSIDE, BoundaryCurve, BoundaryGraph, FittedCurve, Line, Point

_COLLINEAR_EPS = 1e-9
# Lengths below are in original-image pixels and are multiplied by the processing scale where
# used. Curves shorter than this are kept as exact polylines: smoothing shrinks tiny closed
# regions and rounds away the true corners of thin features.
MIN_CLOSED_LENGTH_PX = 12.0
MIN_OPEN_LENGTH_PX = 4.0
CORNER_SNAP_PX = 1.5  # a corner moves to the nearest original vertex within this distance
# Sample counts below are dense samples at scale 1 (one per original-image pixel); at a larger
# processing scale there is one sample per processing pixel, so they are converted with
# `samples_at_scale` where used and keep covering the same original-pixel length.
CORNER_WINDOW = 2  # samples; find_corners window: 3 merges the two corners of a 4 px wide end
CORNERLESS_SPANS = 4  # a closed curve without corners is split into this many spans (unscaled)
# Sub-pixel corners: a corner goes where lines fitted to the sub-pixel samples on its two sides
# meet. Samples right next to the corner are skipped: their pixels blend both sides.
CORNER_LINE_SAMPLES = 4  # samples per side for each line fit
CORNER_LINE_GAP = 1  # samples skipped on each side of the corner index
MIN_CORNER_LINE_SAMPLES = 2  # samples
MIN_CORNER_LINE_ANGLE_DEG = 20.0  # flatter corners make the intersection ill-conditioned


def _is_straight_through(a: Point, b: Point, c: Point) -> bool:
    """True when b lies on segment a-c continuing in the same direction."""
    abx, aby = b[0] - a[0], b[1] - a[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
    cross = abx * bcy - aby * bcx
    dot = abx * bcx + aby * bcy
    return abs(cross) <= _COLLINEAR_EPS and dot > 0


def _merge_collinear(points: tuple[Point, ...]) -> list[Point]:
    out: list[Point] = []
    for p in points:
        if out and p == out[-1]:
            continue
        while len(out) >= 2 and _is_straight_through(out[-2], out[-1], p):
            out.pop()
        out.append(p)
    return out


def _fit_curve(curve: BoundaryCurve) -> FittedCurve:
    pts = _merge_collinear(curve.points)
    segments = tuple(Line(a, b) for a, b in zip(pts, pts[1:]))
    return FittedCurve(curve_id=curve.id, segments=segments, primitive=None)


def polyline_fit(graph: BoundaryGraph) -> tuple[FittedCurve, ...]:
    """Exact polyline of every curve with redundant collinear nodes removed.

    Endpoints are never moved, so shared junctions stay shared.
    """
    return tuple(_fit_curve(c) for c in graph.curves)


# ---------------------------------------------------------------- curve fitting


def _length(points: Sequence[Point]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))


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
    gap = samples_at_scale(CORNER_LINE_GAP, scale)
    reach = gap + samples_at_scale(CORNER_LINE_SAMPLES, scale)
    before = range(max(prev + 1, c - reach), c - gap)
    after = range(c + gap + 1, min(nxt, c + reach + 1))
    if min(len(before), len(after)) < samples_at_scale(MIN_CORNER_LINE_SAMPLES, scale):
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
    raw: Sequence[Point], plain: Sequence[Point], c: int, closed: bool, scale: float
) -> bool:
    """True when any raw sample within reach of corner `c` was moved off the crack grid."""
    n = len(raw)
    reach = samples_at_scale(CORNER_LINE_GAP, scale) + samples_at_scale(CORNER_LINE_SAMPLES, scale)
    window = range(c - reach, c + reach + 1)
    if closed:
        return any(raw[i % n] != plain[i % n] for i in window)
    return any(raw[i] != plain[i] for i in window if 0 <= i < n)


def _place_corners(
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
        if not _has_subpixel_samples(raw, plain, c, closed, scale):
            continue
        neighbours = _corner_neighbours(order, order.index(c), len(dense), closed)
        hit = _subpixel_corner(dense, raw, c, neighbours, scale)
        if hit is not None:
            hits[c] = hit
    out = _snap_corners(dense, [c for c in corners if c not in hits], original, scale)
    for c, hit in hits.items():
        out[c] = hit
    return out


def _samples(
    curve: BoundaryCurve, sigma: float, context: EdgeContext | None
) -> tuple[tuple[Point, ...], tuple[tuple[Point, ...], tuple[Point, ...]]]:
    """(smoothed, (raw, plain)): plain = crack-grid dense samples, raw = their sub-pixel version.

    Without a context raw is plain, and smoothing it is exactly `smooth_polyline`.
    """
    plain = dense_samples(curve.points, curve.closed)
    raw = plain if context is None else subpixel_dense(curve, context)
    return smooth_dense(raw, curve.closed, sigma), (raw, plain)


_Fit = tuple[FittedCurve, SpanPlan | None]  # the plan lets the crossing guard refit spans


def _fit_open(
    curve: BoundaryCurve, params: FitParams, scale: float, context: EdgeContext | None
) -> _Fit:
    dense, samples = _samples(curve, params.smooth_sigma, context)
    window = samples_at_scale(CORNER_WINDOW, scale)
    corners = find_corners(dense, closed=False, angle_deg=params.corner_angle_deg, window=window)
    inner = corners[1:-1]  # the end points are already the exact junctions
    snapped = _place_corners(dense, inner, corners, samples, curve.points, False, scale)
    inner_breaks = frozenset(range(1, len(corners) - 1))
    plan, segments = plan_spans(
        curve, tuple(snapped), corners, inner_breaks, 0, params, scale, fit_span
    )
    return FittedCurve(curve.id, segments, primitive=None), plan


def _fit_closed(
    curve: BoundaryCurve, params: FitParams, scale: float, context: EdgeContext | None
) -> _Fit:
    ring, samples = _samples(curve, params.smooth_sigma, context)
    n = len(ring)
    window = samples_at_scale(CORNER_WINDOW, scale)
    corners = find_corners(ring, closed=True, angle_deg=params.corner_angle_deg, window=window)
    start = corners[0] if corners else 0
    # Rotate so the chain starts (and, by repeating the start sample, ends) at a corner.
    snapped = _place_corners(ring, corners, corners, samples, curve.points, True, scale)
    seq = (*snapped[start:], *snapped[:start], snapped[start])
    if corners:
        breaks = sorted((c - start) % n for c in corners) + [n]
    else:
        breaks = [n * k // CORNERLESS_SPANS for k in range(CORNERLESS_SPANS + 1)]
    corner_breaks = frozenset(range(len(breaks))) if corners else frozenset()
    plan, segments = plan_spans(
        curve, seq, tuple(breaks), corner_breaks, start, params, scale, fit_span
    )
    primitive = detect_primitive(ring, corners, scale)
    return FittedCurve(curve.id, segments, primitive=primitive), plan


def _fit_smooth(
    curve: BoundaryCurve, params: FitParams, scale: float, context: EdgeContext | None
) -> _Fit:
    if OUTSIDE in (curve.left, curve.right):
        return _fit_curve(curve), None  # the canvas edge is exact; smoothing would round it
    min_length = MIN_CLOSED_LENGTH_PX if curve.closed else MIN_OPEN_LENGTH_PX
    if _length(curve.points) < min_length * scale:
        return _fit_curve(curve), None
    fit = _fit_closed if curve.closed else _fit_open
    return fit(curve, params, scale, context)


def _processing_params(params: FitParams, scale: float) -> FitParams:
    """`params` with its lengths converted from original pixels to processing pixels.

    Smoothing runs on one sample per processing pixel, so sigma (in samples) scales too.
    """
    return replace(
        params, tolerance_px=params.tolerance_px * scale, smooth_sigma=params.smooth_sigma * scale
    )


def fit_geometry(
    graph: BoundaryGraph,
    params: FitParams,
    scale: float = 1.0,
    context: EdgeContext | None = None,
) -> tuple[FittedCurve, ...]:
    """Fit lines, arcs and Bezier chains (and whole-shape primitives for closed curves).

    Per curve: dense samples -> smooth -> corners -> `fit_span` between consecutive corners ->
    `snap_axis`. Curve endpoints are never moved, so curves shared by two regions stay
    stitched. With `params.enabled` False this is `polyline_fit`.

    With an `EdgeContext` (see `make_edge_context`) and `params.subpixel`, the dense samples are
    first moved to the edge positions read from anti-aliased edge pixels (`subpixel_dense`) and
    corners are placed from them; otherwise samples stay on the crack grid.

    `graph` is in processing pixels (original pixels x `scale`); every length threshold in
    `params` and in this module is in original pixels and is converted here, and every
    sample-count window is converted with `samples_at_scale`. `params.complexity_weight` is
    deliberately left unscaled: scaling it by scale**3 cost fidelity on enhanced input.

    Curves are fitted independently, so where a band is thinner than the fit's allowances a
    fitted curve can cross its neighbour; `guard_crossings` then refits the offending spans
    (output without crossings is unchanged).
    """
    if not params.enabled:
        return polyline_fit(graph)
    scaled = _processing_params(params, scale)
    ctx = context if params.subpixel else None
    fits = [_fit_smooth(c, scaled, scale, ctx) for c in graph.curves]
    plans = {fit.curve_id: plan for fit, plan in fits if plan is not None}
    return guard_crossings(graph.curves, [fit for fit, _ in fits], plans, scale)
