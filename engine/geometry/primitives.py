"""Whole-shape primitive detection (rect, circle, ellipse, polygon) for closed boundary loops.

Input is the smoothed closed sequence from `smooth_polyline` (cyclic, first point not repeated)
and its corner indices from `find_corners`. Candidates are tried rect -> circle -> ellipse ->
polygon and the first match wins.

Parameter conventions match `engine.stages.export`: `Rect(x, y, w, h, angle_deg)` is an
axis-aligned rect at top-left (x, y) rotated by `angle_deg` about (x, y) (SVG `rotate`, y-down,
positive = clockwise on screen); `Ellipse.angle_deg` rotates about the centre the same way.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from engine.types import Circle, Ellipse, Point, PrimitiveShape, Polygon, Rect

RIGHT_ANGLE_TOL_DEG = 5.0
AXIS_ALIGNED_TOL_DEG = 3.0
SIDE_TOL_PX = 1.0  # *_TOL_PX are original-image pixels; detect_primitive scales them
CIRCLE_TOL_PX = 1.0
CIRCLE_TOL_FRAC = 0.03
ELLIPSE_TOL_PX = 1.0
MIN_POLY_CORNERS = 3
MAX_POLY_CORNERS = 12
CORNER_TRIM = 4  # samples dropped at each side end: smoothing rounds the corners
MIN_SIDE_POINTS = 3
MIN_LOOP_POINTS = 8
ELLIPSE_SAMPLES = 1440
_PARALLEL_SIN = 1e-3


def _fit_line(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Total-least-squares line: (centroid, unit direction, max perpendicular deviation)."""
    centroid = pts.mean(axis=0)
    direction = np.linalg.svd(pts - centroid, full_matrices=False)[2][0]
    offsets = pts - centroid
    dev = np.abs(offsets[:, 0] * direction[1] - offsets[:, 1] * direction[0]).max()
    return centroid, direction, float(dev)


def _sides(arr: np.ndarray, corners: tuple[int, ...]) -> list[np.ndarray] | None:
    """Trimmed point runs between consecutive corners (cyclic); None if any run is too short."""
    n = len(arr)
    ordered = sorted(corners)
    sides = []
    for a, b in zip(ordered, ordered[1:] + [ordered[0] + n]):
        run = arr[np.arange(a, b + 1) % n]
        run = run[CORNER_TRIM : len(run) - CORNER_TRIM]
        if len(run) < MIN_SIDE_POINTS:
            return None
        sides.append(run)
    return sides


def _angle_between_lines(d1: np.ndarray, d2: np.ndarray) -> float:
    cos = min(1.0, abs(float(d1 @ d2)))
    return math.degrees(math.acos(cos))


def _rotate(pts: np.ndarray, deg: float) -> np.ndarray:
    """Rotate row-vector points by +deg (SVG rotate, y-down) about the origin."""
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return pts @ np.array([[c, s], [-s, c]])  # row-vector form of R(deg)


def _rect_angle(directions: list[np.ndarray]) -> float:
    """Mean side orientation folded into (-45, 45] degrees (4-fold symmetric circular mean)."""
    quad = [4.0 * math.atan2(d[1], d[0]) for d in directions]
    mean = math.atan2(sum(math.sin(a) for a in quad), sum(math.cos(a) for a in quad)) / 4.0
    return math.degrees(mean)


def _detect_rect(arr: np.ndarray, corners: tuple[int, ...], scale: float) -> Rect | None:
    if len(corners) != 4:
        return None
    sides = _sides(arr, corners)
    if sides is None:
        return None
    fits = [_fit_line(s) for s in sides]
    if any(dev > SIDE_TOL_PX * scale for _, _, dev in fits):
        return None
    dirs = [d for _, d, _ in fits]
    for i in range(4):
        if abs(_angle_between_lines(dirs[i], dirs[(i + 1) % 4]) - 90.0) > RIGHT_ANGLE_TOL_DEG:
            return None
    theta = _rect_angle(dirs)
    if abs(theta) < AXIS_ALIGNED_TOL_DEG:
        theta = 0.0
    horizontal = [abs(d[1] * math.cos(math.radians(theta)) - d[0] * math.sin(math.radians(theta)))
                  < math.sqrt(0.5) for d in dirs]
    if sum(horizontal) != 2 or horizontal[0] == horizontal[1]:
        return None
    local = [_rotate(s, -theta) for s in sides]  # side points in the rect's own frame
    ys = sorted(float(s[:, 1].mean()) for s, h in zip(local, horizontal) if h)
    xs = sorted(float(s[:, 0].mean()) for s, h in zip(local, horizontal) if not h)
    x0, y0 = _rotate(np.array([[xs[0], ys[0]]]), theta)[0]
    return Rect(float(x0), float(y0), xs[1] - xs[0], ys[1] - ys[0], theta)


def _detect_circle(arr: np.ndarray, scale: float) -> Circle | None:
    x, y = arr[:, 0], arr[:, 1]
    design = np.column_stack([2 * x, 2 * y, np.ones(len(arr))])
    sol, *_ = np.linalg.lstsq(design, x * x + y * y, rcond=None)
    cx, cy = float(sol[0]), float(sol[1])
    r_sq = sol[2] + cx * cx + cy * cy
    if r_sq <= 0:
        return None
    r = math.sqrt(r_sq)
    dev = np.abs(np.hypot(x - cx, y - cy) - r).max()
    return Circle(cx, cy, r) if dev <= max(CIRCLE_TOL_PX * scale, CIRCLE_TOL_FRAC * r) else None


def _conic_fit(arr: np.ndarray) -> np.ndarray | None:
    """Direct least-squares ellipse conic (Halir-Flusser) on centred data: [A, B, C, D, E, F]."""
    x, y = arr[:, 0], arr[:, 1]
    d1 = np.column_stack([x * x, x * y, y * y])
    d2 = np.column_stack([x, y, np.ones(len(arr))])
    s1, s2, s3 = d1.T @ d1, d1.T @ d2, d2.T @ d2
    try:
        t = -np.linalg.solve(s3, s2.T)
        m = s1 + s2 @ t
        m = np.array([m[2] / 2.0, -m[1], m[0] / 2.0])
        _, vecs = np.linalg.eig(m)
    except np.linalg.LinAlgError:
        return None
    vecs = vecs.real
    valid = 4.0 * vecs[0] * vecs[2] - vecs[1] ** 2 > 0
    if not valid.any():
        return None
    a1 = vecs[:, int(np.argmax(valid))]
    return np.concatenate([a1, t @ a1])


def _conic_to_ellipse(conic: np.ndarray) -> tuple[float, float, float, float, float] | None:
    """(cx, cy, rx, ry, angle_deg) with rx the major semi-axis, or None if degenerate."""
    a, b, c, d, e, f = conic
    quad = np.array([[a, b / 2.0], [b / 2.0, c]])
    try:
        centre = np.linalg.solve(2.0 * quad, [-d, -e])
    except np.linalg.LinAlgError:
        return None
    rhs = -(f + (d * centre[0] + e * centre[1]) / 2.0)
    vals, vecs = np.linalg.eigh(quad)
    if rhs == 0 or np.any(vals / rhs <= 0):
        return None
    axes = np.sqrt(rhs / vals)  # semi-axis along each eigenvector
    major = int(np.argmax(axes))
    vx, vy = vecs[:, major]
    angle = (math.degrees(math.atan2(vy, vx)) + 90.0) % 180.0 - 90.0
    return float(centre[0]), float(centre[1]), float(axes[major]), float(axes[1 - major]), angle


def _ellipse_max_deviation(arr: np.ndarray, ell: tuple[float, float, float, float, float]) -> float:
    cx, cy, rx, ry, angle = ell
    t = np.linspace(0.0, 2.0 * np.pi, ELLIPSE_SAMPLES, endpoint=False)
    curve = _rotate(np.column_stack([rx * np.cos(t), ry * np.sin(t)]), angle) + [cx, cy]
    return float(cKDTree(curve).query(arr)[0].max())


def _detect_ellipse(arr: np.ndarray, scale: float) -> Ellipse | None:
    # skimage's EllipseModel is unusable here (it raises on exact ellipses with numpy 2), so the
    # conic fit is done directly; the geometric deviation check below is what gates acceptance.
    centre = arr.mean(axis=0)
    conic = _conic_fit(arr - centre)
    ell = _conic_to_ellipse(conic) if conic is not None else None
    if ell is None:
        return None
    ell = (ell[0] + float(centre[0]), ell[1] + float(centre[1]), *ell[2:])
    if _ellipse_max_deviation(arr, ell) > ELLIPSE_TOL_PX * scale:
        return None
    return Ellipse(*ell)


def _line_intersection(f1: tuple, f2: tuple) -> np.ndarray | None:
    (p1, d1, _), (p2, d2, _) = f1, f2
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < _PARALLEL_SIN:
        return None
    t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / denom
    return p1 + t * d1


def _detect_polygon(arr: np.ndarray, corners: tuple[int, ...], scale: float) -> Polygon | None:
    if not MIN_POLY_CORNERS <= len(corners) <= MAX_POLY_CORNERS:
        return None
    sides = _sides(arr, corners)
    if sides is None:
        return None
    fits = [_fit_line(s) for s in sides]
    if any(dev > SIDE_TOL_PX * scale for _, _, dev in fits):
        return None
    ordered = sorted(corners)
    verts: list[Point] = []
    for k, idx in enumerate(ordered):  # vertex k joins side k-1 and side k
        hit = _line_intersection(fits[k - 1], fits[k])
        pt = hit if hit is not None else arr[idx]
        verts.append((float(pt[0]), float(pt[1])))
    return Polygon(tuple(verts))


def detect_primitive(
    points: tuple[Point, ...], corners: tuple[int, ...], scale: float = 1.0
) -> PrimitiveShape | None:
    """Return the first matching primitive (rect, circle, ellipse, polygon) or None.

    `points` are in processing pixels (original pixels x `scale`).
    """
    if len(points) < MIN_LOOP_POINTS:
        return None
    arr = np.asarray(points, np.float64)
    rect = _detect_rect(arr, corners, scale)
    if rect is not None:
        return rect
    if len(corners) < MIN_POLY_CORNERS:
        # Up to two corners are tolerated: the tips of a pixelated ellipse can read as corners.
        # The fit tolerances below reject lenses, half-discs and other genuinely cornered shapes.
        return _detect_circle(arr, scale) or _detect_ellipse(arr, scale)
    return _detect_polygon(arr, corners, scale)
