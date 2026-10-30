"""Fit lines, circular arcs and Bezier chains to a corner-to-corner span and choose by MDL."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from engine.geometry.bezier import fit_cubic
from engine.presets import FitParams
from engine.types import Arc, CubicBezier, Line, Point, Segment

_LINE_PARAMS = 4
_ARC_PARAMS = 7
_BEZIER_PARAMS = 8
_MAX_RADIUS_TO_CHORD = 10.0
_EPS = 1e-12
_CURVE_SAMPLES = 64


def _as_array(points: Sequence[Point]) -> np.ndarray:
    return np.asarray(points, dtype=float).reshape(-1, 2)


def _dist_to_segment(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = float(ab @ ab)
    if denom < _EPS:
        return np.hypot(*(pts - a).T)
    t = np.clip(((pts - a) @ ab) / denom, 0.0, 1.0)
    return np.hypot(*(pts - (a + t[:, None] * ab)).T)


def _exact(p: Point) -> Point:
    return (p[0], p[1])


def fit_line(points: Sequence[Point]) -> tuple[Line, float]:
    pts = _as_array(points)
    d = _dist_to_segment(pts, pts[0], pts[-1])
    return Line(_exact(points[0]), _exact(points[-1])), float(np.sum(d**2))


def _kasa_circle(pts: np.ndarray) -> tuple[np.ndarray, float] | None:
    mean = pts.mean(axis=0)
    q = pts - mean
    a = np.column_stack((2 * q[:, 0], 2 * q[:, 1], np.ones(len(q))))
    b = (q**2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    r_sq = sol[2] + sol[0] ** 2 + sol[1] ** 2
    if not np.isfinite(r_sq) or r_sq <= _EPS:
        return None
    return np.array([sol[0], sol[1]]) + mean, float(math.sqrt(r_sq))


def _angular_travel(pts: np.ndarray, center: np.ndarray) -> float:
    """Signed total angle swept about center, in the y-down atan2 frame (positive = sweep flag)."""
    ang = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    step = (np.diff(ang) + math.pi) % (2 * math.pi) - math.pi
    return float(step.sum())


def fit_arc(points: Sequence[Point]) -> tuple[Arc, float] | None:
    """Kasa circle fit. None if degenerate or radius > 10x the chord length."""
    if len(points) < 3:
        return None
    pts = _as_array(points)
    chord = float(np.hypot(*(pts[-1] - pts[0])))
    fit = _kasa_circle(pts)
    if fit is None:
        return None
    center, radius = fit
    if chord < _EPS or radius > _MAX_RADIUS_TO_CHORD * chord:
        return None
    travel = _angular_travel(pts, center)
    if abs(travel) < _EPS:
        return None
    resid = np.hypot(*(pts - center).T) - radius
    arc = Arc(
        _exact(points[0]), _exact(points[-1]), (float(center[0]), float(center[1])), radius,
        large_arc=abs(travel) > math.pi, sweep=travel > 0,
    )
    return arc, float(np.sum(resid**2))


def _arc_max_error(arc: Arc, pts: np.ndarray) -> float:
    resid = np.hypot(pts[:, 0] - arc.center[0], pts[:, 1] - arc.center[1]) - arc.radius
    return float(np.max(np.abs(resid)))


def _bezier_distances(chain: Sequence[CubicBezier], pts: np.ndarray) -> np.ndarray:
    t = np.linspace(0.0, 1.0, _CURVE_SAMPLES)
    s = 1.0 - t
    samples = []
    for seg in chain:
        c = _as_array([seg.p0, seg.c1, seg.c2, seg.p1])
        samples.append(
            (s**3)[:, None] * c[0] + (3 * s * s * t)[:, None] * c[1]
            + (3 * s * t * t)[:, None] * c[2] + (t**3)[:, None] * c[3]
        )
    curve = np.vstack(samples)
    diff = pts[:, None, :] - curve[None, :, :]
    return np.sqrt((diff**2).sum(axis=2)).min(axis=1)


def fit_span(points: Sequence[Point], params: FitParams) -> tuple[Segment, ...]:
    """Pick the cheapest of line / arc / Bezier chain by sse + complexity_weight * n_params."""
    if len(points) < 2:
        return ()
    if len(points) == 2:
        return (Line(_exact(points[0]), _exact(points[-1])),)
    pts = _as_array(points)
    tol = params.tolerance_px
    w = params.complexity_weight
    candidates: list[tuple[float, tuple[Segment, ...]]] = []

    line, line_sse = fit_line(points)
    if float(np.max(_dist_to_segment(pts, pts[0], pts[-1]))) <= tol:
        candidates.append((line_sse + w * _LINE_PARAMS, (line,)))

    arc_fit = fit_arc(points)
    if arc_fit is not None and _arc_max_error(arc_fit[0], pts) <= tol:
        candidates.append((arc_fit[1] + w * _ARC_PARAMS, (arc_fit[0],)))

    chain = fit_cubic(points, tol)
    dists = _bezier_distances(chain, pts)
    bez_cost = float(np.sum(dists**2)) + w * _BEZIER_PARAMS * len(chain)
    candidates.append((bez_cost, tuple(chain)))
    return min(candidates, key=lambda c: c[0])[1]


def _shift(p: Point, delta: Point) -> Point:
    return (p[0] + delta[0], p[1] + delta[1])


def _move_end(seg: Segment, at_end: bool, delta: Point) -> Segment:
    """Move a Line/Bezier endpoint (and its adjacent control point) by delta."""
    if isinstance(seg, Line):
        return replace(seg, p1=_shift(seg.p1, delta)) if at_end else replace(seg, p0=_shift(seg.p0, delta))
    if at_end:
        return replace(seg, c2=_shift(seg.c2, delta), p1=_shift(seg.p1, delta))
    return replace(seg, p0=_shift(seg.p0, delta), c1=_shift(seg.c1, delta))


def _snap_deltas(line: Line, tol_rad: float) -> tuple[Point, Point] | None:
    """Endpoint displacements that make `line` exactly horizontal/vertical, or None."""
    dx, dy = line.p1[0] - line.p0[0], line.p1[1] - line.p0[1]
    if math.hypot(dx, dy) < _EPS:
        return None
    angle = math.atan2(abs(dy), abs(dx))  # 0 = horizontal, pi/2 = vertical
    if angle <= tol_rad:
        y = (line.p0[1] + line.p1[1]) / 2
        return (0.0, y - line.p0[1]), (0.0, y - line.p1[1])
    if math.pi / 2 - angle <= tol_rad:
        x = (line.p0[0] + line.p1[0]) / 2
        return (x - line.p0[0], 0.0), (x - line.p1[0], 0.0)
    return None


def snap_axis(segments: Sequence[Segment], params: FitParams) -> tuple[Segment, ...]:
    """Snap near-axis interior Lines to exact horizontal/vertical.

    Only Lines that are neither the first nor the last segment qualify (their endpoints are
    interior corners); the neighbours' shared endpoints move identically. Lines next to an Arc
    are left alone.
    """
    segs = list(segments)
    tol_rad = math.radians(params.axis_snap_deg)
    for i in range(1, len(segs) - 1):
        seg, prev, nxt = segs[i], segs[i - 1], segs[i + 1]
        if not isinstance(seg, Line) or isinstance(prev, Arc) or isinstance(nxt, Arc):
            continue
        deltas = _snap_deltas(seg, tol_rad)
        if deltas is None:
            continue
        d0, d1 = deltas
        segs[i - 1] = _move_end(prev, True, d0)
        segs[i] = Line(_shift(seg.p0, d0), _shift(seg.p1, d1))
        segs[i + 1] = _move_end(nxt, False, d1)
    return tuple(segs)
