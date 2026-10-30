"""Fit stage: turn boundary-graph curves into segment lists."""
from __future__ import annotations

import math
from collections.abc import Sequence

from engine.geometry.corners import find_corners, smooth_polyline
from engine.geometry.primitives import detect_primitive
from engine.geometry.segment_fit import fit_span, snap_axis
from engine.presets import FitParams
from engine.types import OUTSIDE, BoundaryCurve, BoundaryGraph, FittedCurve, Line, Point, Segment

_COLLINEAR_EPS = 1e-9
# Curves shorter than this are kept as exact polylines: smoothing shrinks tiny closed regions
# and rounds away the true corners of thin features.
MIN_CLOSED_LENGTH_PX = 12.0
MIN_OPEN_LENGTH_PX = 4.0
CORNER_SNAP_PX = 1.5  # a corner moves to the nearest original vertex within this distance
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


def _snap_corners(
    dense: Sequence[Point], corners: Sequence[int], original: Sequence[Point]
) -> list[Point]:
    """Replace each corner sample by the nearest original crack-grid vertex, if close.

    Smoothing pulls a corner inwards; fitting between the true vertices keeps straight edges
    where the pixels put them.
    """
    out = list(dense)
    for c in corners:
        x, y = dense[c]
        best = min(original, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)
        if math.hypot(best[0] - x, best[1] - y) <= CORNER_SNAP_PX:
            out[c] = best
    return out


def _fit_spans(points: Sequence[Point], breaks: Sequence[int], params: FitParams) -> tuple[Segment, ...]:
    """Fit each run points[breaks[k]..breaks[k+1]] and concatenate the chains."""
    chain: list[Segment] = []
    for a, b in zip(breaks, breaks[1:]):
        chain.extend(fit_span(points[a : b + 1], params))
    return snap_axis(chain, params)


def _fit_open(curve: BoundaryCurve, params: FitParams) -> FittedCurve:
    dense = smooth_polyline(curve.points, closed=False, sigma=params.smooth_sigma)
    corners = find_corners(dense, closed=False, angle_deg=params.corner_angle_deg, window=CORNER_WINDOW)
    inner = corners[1:-1]  # the end points are already the exact junctions
    snapped = _snap_corners(dense, inner, curve.points)
    return FittedCurve(curve.id, _fit_spans(snapped, corners, params), primitive=None)


def _fit_closed(curve: BoundaryCurve, params: FitParams) -> FittedCurve:
    ring = smooth_polyline(curve.points, closed=True, sigma=params.smooth_sigma)
    n = len(ring)
    corners = find_corners(ring, closed=True, angle_deg=params.corner_angle_deg, window=CORNER_WINDOW)
    start = corners[0] if corners else 0
    # Rotate so the chain starts (and, by repeating the start sample, ends) at a corner.
    snapped = _snap_corners(ring, corners, curve.points)
    seq = (*snapped[start:], *snapped[:start], snapped[start])
    if corners:
        breaks = sorted((c - start) % n for c in corners) + [n]
    else:
        breaks = [n * k // CORNERLESS_SPANS for k in range(CORNERLESS_SPANS + 1)]
    segments = _fit_spans(seq, breaks, params)
    return FittedCurve(curve.id, segments, primitive=detect_primitive(ring, corners))


def _fit_smooth(curve: BoundaryCurve, params: FitParams) -> FittedCurve:
    if OUTSIDE in (curve.left, curve.right):
        return _fit_curve(curve)  # the canvas edge is exact already; smoothing would round it
    min_length = MIN_CLOSED_LENGTH_PX if curve.closed else MIN_OPEN_LENGTH_PX
    if _length(curve.points) < min_length:
        return _fit_curve(curve)
    return _fit_closed(curve, params) if curve.closed else _fit_open(curve, params)


def fit_geometry(graph: BoundaryGraph, params: FitParams) -> tuple[FittedCurve, ...]:
    """Fit lines, arcs and Bezier chains (and whole-shape primitives for closed curves).

    Per curve: smooth -> corners -> `fit_span` between consecutive corners -> `snap_axis`.
    Curve endpoints are never moved, so curves shared by two regions stay stitched. With
    `params.enabled` False this is `polyline_fit`.
    """
    if not params.enabled:
        return polyline_fit(graph)
    return tuple(_fit_smooth(c, params) for c in graph.curves)
