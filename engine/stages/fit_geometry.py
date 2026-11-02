"""Fit stage: turn boundary-graph curves into segment lists."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace

from engine.geometry.corner_placement import place_corners
from engine.geometry.corners import dense_samples, find_corners, smooth_dense
from engine.geometry.primitives import detect_primitive
from engine.geometry.segment_fit import fit_span, snap_axis
from engine.geometry.subpixel import EdgeContext, subpixel_dense
from engine.presets import FitParams
from engine.types import OUTSIDE, BoundaryCurve, BoundaryGraph, FittedCurve, Line, Point, Segment

_COLLINEAR_EPS = 1e-9
# Lengths below are in original-image pixels and are multiplied by the processing scale where
# used. Curves shorter than this are kept as exact polylines: smoothing shrinks tiny closed
# regions and rounds away the true corners of thin features.
MIN_CLOSED_LENGTH_PX = 12.0
MIN_OPEN_LENGTH_PX = 4.0
CORNER_WINDOW = 2  # find_corners window: 3 merges the two corners of a 4 px wide end
CORNERLESS_SPANS = 4  # a closed curve without corners is split into this many spans


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


def _fit_spans(points: Sequence[Point], breaks: Sequence[int], params: FitParams) -> tuple[Segment, ...]:
    """Fit each run points[breaks[k]..breaks[k+1]] and concatenate the chains."""
    chain: list[Segment] = []
    for a, b in zip(breaks, breaks[1:]):
        chain.extend(fit_span(points[a : b + 1], params))
    return snap_axis(chain, params)


def _samples(
    curve: BoundaryCurve, sigma: float, context: EdgeContext | None
) -> tuple[tuple[Point, ...], tuple[tuple[Point, ...], tuple[Point, ...]]]:
    """(smoothed, (raw, plain)): plain = crack-grid dense samples, raw = their sub-pixel version.

    Without a context raw is plain, and smoothing it is exactly `smooth_polyline`.
    """
    plain = dense_samples(curve.points, curve.closed)
    raw = plain if context is None else subpixel_dense(curve, context)
    return smooth_dense(raw, curve.closed, sigma), (raw, plain)


def _fit_open(
    curve: BoundaryCurve, params: FitParams, scale: float, context: EdgeContext | None
) -> FittedCurve:
    dense, samples = _samples(curve, params.smooth_sigma, context)
    corners = find_corners(dense, closed=False, angle_deg=params.corner_angle_deg, window=CORNER_WINDOW)
    inner = corners[1:-1]  # the end points are already the exact junctions
    snapped = place_corners(dense, inner, corners, samples, curve.points, False, scale)
    return FittedCurve(curve.id, _fit_spans(snapped, corners, params), primitive=None)


def _fit_closed(
    curve: BoundaryCurve, params: FitParams, scale: float, context: EdgeContext | None
) -> FittedCurve:
    ring, samples = _samples(curve, params.smooth_sigma, context)
    n = len(ring)
    corners = find_corners(ring, closed=True, angle_deg=params.corner_angle_deg, window=CORNER_WINDOW)
    start = corners[0] if corners else 0
    # Rotate so the chain starts (and, by repeating the start sample, ends) at a corner.
    snapped = place_corners(ring, corners, corners, samples, curve.points, True, scale)
    seq = (*snapped[start:], *snapped[:start], snapped[start])
    if corners:
        breaks = sorted((c - start) % n for c in corners) + [n]
    else:
        breaks = [n * k // CORNERLESS_SPANS for k in range(CORNERLESS_SPANS + 1)]
    segments = _fit_spans(seq, breaks, params)
    return FittedCurve(curve.id, segments, primitive=detect_primitive(ring, corners, scale))


def _fit_smooth(
    curve: BoundaryCurve, params: FitParams, scale: float, context: EdgeContext | None
) -> FittedCurve:
    if OUTSIDE in (curve.left, curve.right):
        return _fit_curve(curve)  # the canvas edge is exact already; smoothing would round it
    min_length = MIN_CLOSED_LENGTH_PX if curve.closed else MIN_OPEN_LENGTH_PX
    if _length(curve.points) < min_length * scale:
        return _fit_curve(curve)
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
    `params` and in this module is in original pixels and is converted here.
    """
    if not params.enabled:
        return polyline_fit(graph)
    scaled = _processing_params(params, scale)
    ctx = context if params.subpixel else None
    return tuple(_fit_smooth(c, scaled, scale, ctx) for c in graph.curves)
