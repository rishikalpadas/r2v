"""Schneider cubic Bezier fitting (Graphics Gems, 1990) with G1 joins between chained pieces."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from engine.types import CubicBezier, Point

_MAX_REPARAM_ITERATIONS = 4
_REPARAM_ERROR_FACTOR = 16.0  # only reparameterize when squared error < tol_sq * this
_EPS = 1e-12
MIN_CHORD_PX = 8.0  # processing px; don't subdivide into micro-splines smaller than this


def _dedupe(points: Sequence[Point]) -> list[Point]:
    """Drop consecutive duplicates; the first and last points are kept exactly."""
    out: list[Point] = []
    for p in points:
        q = (float(p[0]), float(p[1]))
        if not out or q != out[-1]:
            out.append(q)
    return out


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.hypot(v[0], v[1]))
    return v / n if n > _EPS else np.zeros(2)


def _bezier_at(ctrl: np.ndarray, t: np.ndarray) -> np.ndarray:
    s = 1.0 - t
    return (
        (s**3)[:, None] * ctrl[0]
        + (3 * s * s * t)[:, None] * ctrl[1]
        + (3 * s * t * t)[:, None] * ctrl[2]
        + (t**3)[:, None] * ctrl[3]
    )


def sample(seg: CubicBezier, ts: np.ndarray) -> np.ndarray:
    """Points of `seg` at parameters `ts` as an Nx2 array."""
    ctrl = np.array([seg.p0, seg.c1, seg.c2, seg.p1], dtype=float)
    return _bezier_at(ctrl, np.asarray(ts, dtype=float))


def evaluate(seg: CubicBezier, t: float) -> Point:
    """Point of `seg` at parameter t in [0, 1]."""
    x, y = sample(seg, np.array([t]))[0]
    return (float(x), float(y))


def _chord_params(d: np.ndarray) -> np.ndarray:
    steps = np.hypot(*np.diff(d, axis=0).T)
    u = np.concatenate(([0.0], np.cumsum(steps)))
    return u / u[-1] if u[-1] > _EPS else np.linspace(0.0, 1.0, len(d))


def _generate(d: np.ndarray, u: np.ndarray, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    p0, p3 = d[0], d[-1]
    s = 1.0 - u
    b0, b1, b2, b3 = s**3, 3 * s * s * u, 3 * s * u * u, u**3
    a1 = b1[:, None] * t1
    a2 = b2[:, None] * t2
    c = np.array([[np.sum(a1 * a1), np.sum(a1 * a2)], [np.sum(a1 * a2), np.sum(a2 * a2)]])
    tmp = d - (b0 + b1)[:, None] * p0 - (b2 + b3)[:, None] * p3
    x = np.array([np.sum(a1 * tmp), np.sum(a2 * tmp)])
    det = c[0, 0] * c[1, 1] - c[0, 1] * c[1, 0]
    seg_len = float(np.hypot(*(p3 - p0)))
    alpha1 = alpha2 = 0.0
    if abs(det) > _EPS:
        alpha1 = (x[0] * c[1, 1] - x[1] * c[0, 1]) / det
        alpha2 = (c[0, 0] * x[1] - c[1, 0] * x[0]) / det
    eps = 1e-6 * seg_len
    if alpha1 < eps or alpha2 < eps:
        alpha1 = alpha2 = seg_len / 3.0
    return np.array([p0, p0 + t1 * alpha1, p3 + t2 * alpha2, p3])


def _reparameterize(ctrl: np.ndarray, d: np.ndarray, u: np.ndarray) -> np.ndarray:
    d1 = 3 * (ctrl[1:] - ctrl[:-1])
    d2 = 2 * (d1[1:] - d1[:-1])
    out = u.copy()
    for i, t in enumerate(u):
        q = _bezier_at(ctrl, np.array([t]))[0]
        s = 1.0 - t
        q1 = s * s * d1[0] + 2 * s * t * d1[1] + t * t * d1[2]
        q2 = s * d2[0] + t * d2[1]
        diff = q - d[i]
        num = float(diff @ q1)
        den = float(q1 @ q1 + diff @ q2)
        if abs(den) > _EPS:
            out[i] = min(1.0, max(0.0, t - num / den))
    return out


def _max_error(ctrl: np.ndarray, d: np.ndarray, u: np.ndarray) -> tuple[float, int]:
    """Max squared distance at the given parameters and the (interior) index where it occurs."""
    dist = np.sum((_bezier_at(ctrl, u) - d) ** 2, axis=1)
    interior = dist[1:-1]
    if interior.size == 0:
        return 0.0, len(d) // 2
    k = int(np.argmax(interior))
    return float(interior[k]), k + 1


def _to_bezier(p0: Point, ctrl: np.ndarray, p1: Point) -> CubicBezier:
    return CubicBezier(
        p0, (float(ctrl[1][0]), float(ctrl[1][1])), (float(ctrl[2][0]), float(ctrl[2][1])), p1
    )


def _fit_piece(
    pts: list[Point], d: np.ndarray, t1: np.ndarray, t2: np.ndarray, tol_sq: float, min_chord: float = 0.0
) -> list[CubicBezier]:
    steps = np.hypot(*(d[1:] - d[:-1]).T)
    arc_len = float(steps.sum())
    chord_len = float(np.hypot(d[-1, 0] - d[0, 0], d[-1, 1] - d[0, 1]))
    if len(d) <= 3 or (min_chord > 0 and arc_len < min_chord):
        dist = chord_len / 3.0
        ctrl = np.array([d[0], d[0] + t1 * dist, d[-1] + t2 * dist, d[-1]])
        return [_to_bezier(pts[0], ctrl, pts[-1])]
    u = _chord_params(d)
    ctrl = _generate(d, u, t1, t2)
    err, split = _max_error(ctrl, d, u)
    if err < tol_sq:
        return [_to_bezier(pts[0], ctrl, pts[-1])]
    if err < tol_sq * _REPARAM_ERROR_FACTOR:
        for _ in range(_MAX_REPARAM_ITERATIONS):
            u = _reparameterize(ctrl, d, u)
            ctrl = _generate(d, u, t1, t2)
            err, split = _max_error(ctrl, d, u)
            if err < tol_sq:
                return [_to_bezier(pts[0], ctrl, pts[-1])]
    
    if min_chord > 0:
        left_arc = float(steps[:split].sum())
        right_arc = float(steps[split:].sum())
        if left_arc < min_chord * 0.5 or right_arc < min_chord * 0.5:
            return [_to_bezier(pts[0], ctrl, pts[-1])]

    k = min(3, split, len(d) - 1 - split) if min_chord > 0 else 1
    centre = _unit(d[split - k] - d[split + k])
    if not centre.any():
        chord = d[-1] - d[0]
        centre = _unit(np.array([-chord[1], chord[0]]))
    left = _fit_piece(pts[: split + 1], d[: split + 1], t1, centre, tol_sq, min_chord)
    right = _fit_piece(pts[split:], d[split:], -centre, t2, tol_sq, min_chord)
    return left + right


def fit_cubic(
    points: Sequence[Point], tolerance_px: float, tangent_samples: int = 1, min_chord: float = 0.0
) -> tuple[CubicBezier, ...]:
    """Fit a G1-joined chain of cubic Beziers from points[0] to points[-1] (exact endpoints).

    The end tangents point from each end to the mean of its next `tangent_samples` points (at
    most a third of the run); 1 is Schneider's one-sample estimate.
    """
    if len(points) < 2:
        return ()
    pts = _dedupe(points)
    if len(pts) < 2:  # all points coincide
        return (CubicBezier(pts[0], pts[0], pts[0], pts[0]),)
    d = np.array(pts, dtype=float)
    k = max(1, min(tangent_samples, len(d) // 3))
    t1 = _unit(d[1 : k + 1].mean(axis=0) - d[0])
    t2 = _unit(d[-k - 1 : -1].mean(axis=0) - d[-1])
    return tuple(_fit_piece(pts, d, t1, t2, tolerance_px**2, min_chord))
