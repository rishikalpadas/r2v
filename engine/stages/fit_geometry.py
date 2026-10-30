"""Fit stage: turn boundary-graph curves into segment lists."""
from __future__ import annotations

from engine.types import BoundaryCurve, BoundaryGraph, FittedCurve, Line, Point

_COLLINEAR_EPS = 1e-9


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
