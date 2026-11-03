"""Fit lines, circular arcs and Bezier chains to a corner-to-corner span and choose by MDL."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree

from engine.geometry.bezier import fit_cubic, sample
from engine.geometry.noise import NOISE_FLOOR_PX
from engine.presets import FitParams
from engine.types import Arc, CubicBezier, Line, Point, Segment

_LINE_PARAMS = 4
_ARC_PARAMS = 7
_BEZIER_PARAMS = 8
_MAX_RADIUS_TO_CHORD = 10.0
_EPS = 1e-12
_CURVE_SAMPLES = 64
_G1_COS = math.cos(math.radians(10.0))  # handles within 10 deg of a line count as G1


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


def _bisector_circle(pts: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Circle through p0 and p1 (center on their perpendicular bisector) best fitting the points.

    This is the circle an SVG renderer reconstructs from `A r r 0 large sweep p1` at p0, so
    fitting and scoring against it means the tolerance gate sees what is actually drawn.
    """
    p0, p1 = pts[0], pts[-1]
    chord = p1 - p0
    length = float(np.hypot(*chord))
    if length < _EPS:
        return None
    mid = (p0 + p1) / 2
    normal = np.array([-chord[1], chord[0]]) / length
    limit = 2 * _MAX_RADIUS_TO_CHORD * length  # search past the cutoff so it rejects, not clips

    def center(s: float) -> np.ndarray:
        return mid + s * normal

    def cost(s: float) -> float:
        c = center(s)
        r = float(np.hypot(*(p0 - c)))
        return float(np.sum((np.hypot(*(pts - c).T) - r) ** 2))

    # algebraic start: |p-c|^2 - |p0-c|^2 = a_i - s*b_i is linear in the offset s
    q = pts - p0
    a = (pts**2).sum(axis=1) - (p0**2).sum() - 2 * (q @ mid)
    b = 2 * (q @ normal)
    denom = float(b @ b)
    s0 = float(a @ b) / denom if denom > _EPS else 0.0
    best = min((min(max(s0, -limit), limit), 0.0), key=cost)
    for lo, hi in ((best - length, best + length), (-limit, limit)):
        res = minimize_scalar(cost, bounds=(max(lo, -limit), min(hi, limit)), method="bounded")
        if cost(float(res.x)) < cost(best):
            best = float(res.x)
    c = center(best)
    return c, float(np.hypot(*(p0 - c)))


def _angular_travel(pts: np.ndarray, center: np.ndarray) -> float:
    """Signed total angle swept about center, in the y-down atan2 frame (positive = sweep flag)."""
    ang = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    step = (np.diff(ang) + math.pi) % (2 * math.pi) - math.pi
    return float(step.sum())


def fit_arc(points: Sequence[Point]) -> tuple[Arc, float] | None:
    """Circular arc through points[0] and points[-1]; None if degenerate or radius > 10x chord."""
    if len(points) < 3:
        return None
    pts = _as_array(points)
    chord = float(np.hypot(*(pts[-1] - pts[0])))
    fit = _bisector_circle(pts)
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
    curve = np.vstack([sample(seg, t) for seg in chain])
    diff = pts[:, None, :] - curve[None, :, :]
    return np.sqrt((diff**2).sum(axis=2)).min(axis=1)


def _model_cost(sse: float, n_params: int, params: FitParams, noise: float | None) -> float:
    """MDL cost: sse + w * n_params, or sse / sigma^2 + w * n_params when the noise is known.

    Dividing by the noise variance makes the cost scale-free (sse grows with scale squared, the
    parameter count does not) and lets noisier curves trade residual for fewer parameters.
    """
    w = params.complexity_weight
    if noise is None:
        return sse + w * n_params
    sigma = max(noise, NOISE_FLOOR_PX)
    return sse / (sigma * sigma) + w * n_params


def fit_span_scored(
    points: Sequence[Point], params: FitParams, noise: float | None = None, tangent_samples: int = 1
) -> tuple[tuple[Segment, ...], float]:
    """The cheapest of line / arc / Bezier chain within tolerance, and its `_model_cost`.

    `tangent_samples` is passed to `fit_cubic` for the Bezier chain's end tangents.
    """
    if len(points) < 2:
        return (), 0.0
    if len(points) == 2:
        line = Line(_exact(points[0]), _exact(points[-1]))
        return (line,), _model_cost(0.0, _LINE_PARAMS, params, noise)
    pts = _as_array(points)
    tol = params.tolerance_px
    candidates: list[tuple[float, tuple[Segment, ...]]] = []

    line, line_sse = fit_line(points)
    if float(np.max(_dist_to_segment(pts, pts[0], pts[-1]))) <= tol:
        candidates.append((_model_cost(line_sse, _LINE_PARAMS, params, noise), (line,)))

    arc_fit = fit_arc(points)
    if arc_fit is not None and _arc_max_error(arc_fit[0], pts) <= tol:
        candidates.append((_model_cost(arc_fit[1], _ARC_PARAMS, params, noise), (arc_fit[0],)))

    min_chord = max(6.0, tol * 3.0) if noise is not None else 0.0
    chain = fit_cubic(points, tol, tangent_samples, min_chord)
    sse = float(np.sum(_bezier_distances(chain, pts) ** 2))
    candidates.append((_model_cost(sse, _BEZIER_PARAMS * len(chain), params, noise), tuple(chain)))
    cost, segments = min(candidates, key=lambda c: c[0])
    return segments, cost


def fit_span(
    points: Sequence[Point], params: FitParams, noise: float | None = None, tangent_samples: int = 1
) -> tuple[Segment, ...]:
    """Pick the cheapest of line / arc / Bezier chain by `_model_cost` (see `fit_span_scored`)."""
    return fit_span_scored(points, params, noise, tangent_samples)[0]


_CHAIN_SAMPLES_PER_PX = 10  # chain_cost samples the chain this densely (0.05 px max error)


def _n_params(seg: Segment) -> int:
    if isinstance(seg, Line):
        return _LINE_PARAMS
    return _ARC_PARAMS if isinstance(seg, Arc) else _BEZIER_PARAMS


def _segment_points(seg: Segment) -> np.ndarray:
    """Points along `seg`, about 1 / _CHAIN_SAMPLES_PER_PX apart."""
    if isinstance(seg, Arc):
        a0 = math.atan2(seg.p0[1] - seg.center[1], seg.p0[0] - seg.center[0])
        a1 = math.atan2(seg.p1[1] - seg.center[1], seg.p1[0] - seg.center[0])
        travel = (a1 - a0) % (2 * math.pi) if seg.sweep else -((a0 - a1) % (2 * math.pi))
        count = max(2, math.ceil(abs(travel) * seg.radius * _CHAIN_SAMPLES_PER_PX) + 1)
        ang = a0 + np.linspace(0.0, 1.0, count) * travel
        return np.column_stack([seg.center[0] + seg.radius * np.cos(ang),
                                seg.center[1] + seg.radius * np.sin(ang)])
    ctrl = _as_array((seg.p0, seg.p1) if isinstance(seg, Line) else (seg.p0, seg.c1, seg.c2, seg.p1))
    length = float(np.hypot(*np.diff(ctrl, axis=0).T).sum())  # bounds the curve length
    t = np.linspace(0.0, 1.0, max(2, math.ceil(length * _CHAIN_SAMPLES_PER_PX) + 1))
    if isinstance(seg, Line):
        return ctrl[0] + t[:, None] * (ctrl[1] - ctrl[0])
    return sample(seg, t)


def chain_cost(
    segments: Sequence[Segment], points: Sequence[Point], params: FitParams, noise: float | None
) -> float:
    """`_model_cost` of a whole chain against `points` (each point's squared distance to it).

    Unlike the cost `fit_span_scored` reports, this can score different chains against the same
    data, e.g. one fit across a corner against two fits that meet at it.
    """
    n_params = sum(_n_params(s) for s in segments)
    if not segments or not points:
        return _model_cost(0.0, n_params, params, noise)
    curve = np.vstack([_segment_points(s) for s in segments])
    dist, _ = cKDTree(curve).query(_as_array(points))
    return _model_cost(float(np.sum(dist**2)), n_params, params, noise)


def _shift(p: Point, delta: Point) -> Point:
    return (p[0] + delta[0], p[1] + delta[1])


def _move_end(seg: Segment, at_end: bool, delta: Point, direction: Point) -> Segment:
    """Move a Line/Bezier endpoint by delta; `direction` is the snapped line's unit direction.

    A Bezier handle that was (nearly) collinear with the line keeps G1: it is re-aimed along
    `direction`, preserving its length. Other handles just follow their endpoint.
    """
    if isinstance(seg, Line):
        return replace(seg, p1=_shift(seg.p1, delta)) if at_end else replace(seg, p0=_shift(seg.p0, delta))
    anchor, handle = (seg.p1, seg.c2) if at_end else (seg.p0, seg.c1)
    vec = (handle[0] - anchor[0], handle[1] - anchor[1])  # points away from the junction
    length = math.hypot(*vec)
    new_anchor = _shift(anchor, delta)
    new_handle = _shift(handle, delta)
    if length > _EPS:
        # leaving the junction, the next segment's handle runs along +direction and the
        # previous segment's handle along -direction
        wanted = (-direction[0], -direction[1]) if at_end else direction
        if (vec[0] * wanted[0] + vec[1] * wanted[1]) / length >= _G1_COS:
            new_handle = (new_anchor[0] + wanted[0] * length, new_anchor[1] + wanted[1] * length)
    if at_end:
        return replace(seg, c2=new_handle, p1=new_anchor)
    return replace(seg, p0=new_anchor, c1=new_handle)


def _snap_deltas(line: Line, tol_rad: float, max_move: float) -> tuple[Point, Point] | None:
    """Endpoint displacements that make `line` exactly horizontal/vertical, or None.

    None when the line is not within tol_rad of an axis or an endpoint would move > max_move.
    """
    dx, dy = line.p1[0] - line.p0[0], line.p1[1] - line.p0[1]
    if math.hypot(dx, dy) < _EPS:
        return None
    angle = math.atan2(abs(dy), abs(dx))  # 0 = horizontal, pi/2 = vertical
    if angle <= tol_rad:
        y = (line.p0[1] + line.p1[1]) / 2
        deltas = ((0.0, y - line.p0[1]), (0.0, y - line.p1[1]))
    elif math.pi / 2 - angle <= tol_rad:
        x = (line.p0[0] + line.p1[0]) / 2
        deltas = ((x - line.p0[0], 0.0), (x - line.p1[0], 0.0))
    else:
        return None
    if any(math.hypot(*d) > max_move for d in deltas):
        return None
    return deltas


def snap_axis(segments: Sequence[Segment], params: FitParams, locked: frozenset[int] = frozenset()) -> tuple[Segment, ...]:
    """Snap near-axis interior Lines to exact horizontal/vertical.

    Only Lines that are neither the first nor the last segment qualify (their endpoints are
    interior corners); the neighbours' shared endpoints move identically. Lines next to an Arc
    are left alone, as are lines whose snap would move an endpoint more than tolerance_px.
    """
    segs = list(segments)
    tol_rad = math.radians(params.axis_snap_deg)
    for i in range(1, len(segs) - 1):
        seg, prev, nxt = segs[i], segs[i - 1], segs[i + 1]
        if not isinstance(seg, Line) or isinstance(prev, Arc) or isinstance(nxt, Arc):
            continue
        if locked and not locked.isdisjoint((i - 1, i, i + 1)):
            continue
        deltas = _snap_deltas(seg, tol_rad, params.tolerance_px)
        if deltas is None:
            continue
        d0, d1 = deltas
        snapped = Line(_shift(seg.p0, d0), _shift(seg.p1, d1))
        length = math.hypot(snapped.p1[0] - snapped.p0[0], snapped.p1[1] - snapped.p0[1])
        direction = ((snapped.p1[0] - snapped.p0[0]) / length, (snapped.p1[1] - snapped.p0[1]) / length)
        segs[i - 1] = _move_end(prev, True, d0, direction)
        segs[i] = snapped
        segs[i + 1] = _move_end(nxt, False, d1, direction)
    return tuple(segs)


def _unit_vec(dx: float, dy: float) -> Point | None:
    length = math.hypot(dx, dy)
    return (dx / length, dy / length) if length > _EPS else None


def _arc_tangent(arc: Arc, p: Point) -> Point | None:
    """Direction of travel along `arc` at its end point `p`."""
    rx, ry = p[0] - arc.center[0], p[1] - arc.center[1]
    sign = 1.0 if arc.sweep else -1.0  # sweep = increasing angle in the y-down frame
    return _unit_vec(-ry * sign, rx * sign)


def _end_tangent(seg: Segment, at_end: bool) -> Point | None:
    """Unit direction of travel where `seg` ends (at_end) or starts."""
    if isinstance(seg, Line):
        return _unit_vec(seg.p1[0] - seg.p0[0], seg.p1[1] - seg.p0[1])
    if isinstance(seg, Arc):
        return _arc_tangent(seg, seg.p1 if at_end else seg.p0)
    ctrl = (seg.p0, seg.c1, seg.c2, seg.p1)
    if at_end:
        anchor, others = seg.p1, ctrl[2::-1]
        return next((u for q in others if (u := _unit_vec(anchor[0] - q[0], anchor[1] - q[1]))), None)
    anchor, others = seg.p0, ctrl[1:]
    return next((u for q in others if (u := _unit_vec(q[0] - anchor[0], q[1] - anchor[1]))), None)


def _aim_handle(seg: CubicBezier, at_end: bool, direction: Point) -> CubicBezier:
    """Re-aim the handle at one end of `seg` along the travel `direction`, keeping its length."""
    anchor, handle = (seg.p1, seg.c2) if at_end else (seg.p0, seg.c1)
    length = math.hypot(handle[0] - anchor[0], handle[1] - anchor[1])
    if length <= _EPS:
        return seg
    sign = -1.0 if at_end else 1.0  # the end handle points back against the travel
    aimed = (anchor[0] + sign * direction[0] * length, anchor[1] + sign * direction[1] * length)
    return replace(seg, c2=aimed) if at_end else replace(seg, c1=aimed)


def align_joins(segments: Sequence[Segment], joins: Sequence[int]) -> tuple[Segment, ...]:
    """Make the chain G1 at each join index j (between segments[j - 1] and segments[j]).

    j = 0 is the wrap-around join of a closed chain. Lines and arcs are rigid: a Bezier next to
    one has its handle aimed along the rigid tangent; two Beziers share the bisecting direction;
    joins between two rigid segments (or at a cusp) are left alone. End points never move.
    """
    segs = list(segments)
    for j in joins:
        i = (j - 1) % len(segs)
        prev, nxt = segs[i], segs[j]
        t_out, t_in = _end_tangent(prev, True), _end_tangent(nxt, False)
        prev_bez, next_bez = isinstance(prev, CubicBezier), isinstance(nxt, CubicBezier)
        if t_out is None or t_in is None or not (prev_bez or next_bez):
            continue
        if not prev_bez:
            direction = t_out
        elif not next_bez:
            direction = t_in
        else:
            direction = _unit_vec(t_out[0] + t_in[0], t_out[1] + t_in[1])
            if direction is None:
                continue
        if prev_bez:
            segs[i] = _aim_handle(prev, True, direction)
        if next_bez:
            segs[j] = _aim_handle(segs[j], False, direction)
    return tuple(segs)
