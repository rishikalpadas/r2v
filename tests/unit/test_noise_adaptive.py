"""Noise-adaptive fitting: noisy JPEG edges fit without fake corners, real corners survive."""
import math

import numpy as np
import pytest

from engine.geometry.subpixel import make_edge_context
from engine.presets import FitParams, SegmentParams
from engine.stages.fit_geometry import fit_geometry
from engine.stages.segment import segment
from engine.stages.trace import trace
from engine.types import OUTSIDE, Arc, Circle, CubicBezier, Line, RasterImage
from eval.synthetic import (
    aa_disc,
    aa_mask,
    aa_polygon,
    aa_shapes,
    add_noise,
    flat_shapes,
    jpeg_roundtrip,
)

SIZE = 128
PIXEL_NOISE = 6.0  # 8-bit units
JPEG_QUALITY = 50
CORNER_BREAK_DEG = 30.0  # a join whose tangent turns more than this is a corner
DISC = dict(cx=64.3, cy=63.7, r=40.4)


def _noisy(pixels: np.ndarray, seed: int = 0) -> np.ndarray:
    return jpeg_roundtrip(add_noise(pixels, PIXEL_NOISE, seed), JPEG_QUALITY)


def _fit_noisy(pixels: np.ndarray, params: FitParams = FitParams()):
    """The one interior closed fitted curve of a two-colour noisy image."""
    h, w = pixels.shape[:2]
    image = RasterImage(pixels, (w, h))
    regions = segment(image, SegmentParams(max_colors=2))
    graph = trace(regions)
    fitted = fit_geometry(graph, params, context=make_edge_context(image, regions))
    inner = [
        f for f, c in zip(fitted, graph.curves) if c.closed and OUTSIDE not in (c.left, c.right)
    ]
    assert len(inner) == 1, f"expected one closed shape outline, got {len(inner)}"
    return inner[0]


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = float(np.hypot(*v))
    return v / n if n > 1e-12 else v


def _tangents(seg) -> tuple[np.ndarray, np.ndarray]:
    """Unit direction leaving p0 and arriving at p1."""
    if isinstance(seg, Line):
        d = _unit(np.subtract(seg.p1, seg.p0))
        return d, d
    if isinstance(seg, CubicBezier):
        pts = [np.asarray(p, float) for p in (seg.p0, seg.c1, seg.c2, seg.p1)]
        start = next((q - pts[0] for q in pts[1:] if np.hypot(*(q - pts[0])) > 1e-9), pts[3] - pts[0])
        end = next((pts[3] - q for q in pts[2::-1] if np.hypot(*(pts[3] - q)) > 1e-9), pts[3] - pts[0])
        return _unit(start), _unit(end)
    assert isinstance(seg, Arc)
    c = np.asarray(seg.center, float)
    sign = 1.0 if seg.sweep else -1.0

    def tangent(p):
        r = np.asarray(p, float) - c
        return _unit(sign * np.array([-r[1], r[0]]))

    return tangent(seg.p0), tangent(seg.p1)


def _joins(segments) -> list[tuple[tuple[float, float], float]]:
    """(position, tangent break in degrees) at every join of a closed chain."""
    out = []
    for prev, nxt in zip(segments, segments[1:] + segments[:1]):
        a, b = _tangents(prev)[1], _tangents(nxt)[0]
        cos = float(np.clip(a @ b, -1.0, 1.0))
        out.append((nxt.p0, math.degrees(math.acos(cos))))
    return out


def _corners(segments) -> list[tuple[float, float]]:
    return [p for p, brk in _joins(segments) if brk > CORNER_BREAK_DEG]


def _nearest(points, target) -> float:
    return min(math.hypot(p[0] - target[0], p[1] - target[1]) for p in points)


# ---------------------------------------------------------------- scenes


def _rotated_rect_vertices(cx=64.2, cy=63.6, w=72.0, h=48.0, deg=15.0):
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    half = ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))
    return tuple((cx + x * c - y * s, cy + x * s + y * c) for x, y in half)


# Heart: two lobes (discs) whose overlap leaves a ~6 px deep cusp on top, and a 100 degree tip
# at the bottom where the tangents from the tip touch both lobes.
HEART_LOBES = ((50.0, 55.0), (78.0, 55.0))
HEART_R = 20.0
HEART_TIP_HALF_DEG = 50.0


def _heart_geometry():
    (x1, y1), (x2, _) = HEART_LOBES
    half = math.radians(HEART_TIP_HALF_DEG)
    mid = (x1 + x2) / 2
    tip_y = y1 + (HEART_R + (x2 - mid) * math.cos(half)) / math.sin(half)
    tip = (mid, tip_y)
    direction = np.array([math.sin(half), -math.cos(half)])
    along = float((np.array([x2 - mid, y1 - tip_y])) @ direction)
    tangent_r = (mid + along * direction[0], tip_y + along * direction[1])
    tangent_l = (2 * mid - tangent_r[0], tangent_r[1])
    cusp = (mid, y1 - math.sqrt(HEART_R**2 - (x2 - mid) ** 2))
    return tip, cusp, (tip, tangent_r, HEART_LOBES[1], HEART_LOBES[0], tangent_l)


def _heart() -> np.ndarray:
    from eval.synthetic import _polygon_inside

    _, _, body = _heart_geometry()
    verts = np.asarray(body)

    def inside(x, y):
        lobes = [(x - cx) ** 2 + (y - cy) ** 2 <= HEART_R**2 for cx, cy in HEART_LOBES]
        return lobes[0] | lobes[1] | _polygon_inside(verts, x, y)

    return aa_mask(inside, SIZE)


def _blob() -> np.ndarray:
    """Sine-perturbed circle: smooth everywhere, curvature varies."""

    def inside(x, y):
        dx, dy = x - 64.3, y - 63.8
        return np.hypot(dx, dy) <= 36.0 + 4.0 * np.sin(3.0 * np.arctan2(dy, dx))

    return aa_mask(inside, SIZE)


# ---------------------------------------------------------------- behaviour


def test_noisy_disc_is_one_smooth_circle():
    fitted = _fit_noisy(_noisy(aa_disc(SIZE, **DISC)))
    assert _corners(fitted.segments) == []
    assert len(fitted.segments) <= 8
    shape = fitted.primitive
    assert isinstance(shape, Circle)
    assert math.hypot(shape.cx - DISC["cx"], shape.cy - DISC["cy"]) <= 0.4
    assert abs(shape.r - DISC["r"]) <= 0.4


def test_noisy_rotated_rect_has_exactly_its_four_corners():
    verts = _rotated_rect_vertices()
    fitted = _fit_noisy(_noisy(aa_polygon(verts, SIZE)))
    corners = _corners(fitted.segments)
    assert len(corners) == 4
    for v in verts:
        assert _nearest(corners, v) <= 0.5, v


def test_noisy_heart_keeps_tip_and_cusp():
    tip, cusp, _ = _heart_geometry()
    corners = _corners(_fit_noisy(_noisy(_heart())).segments)
    assert _nearest(corners, tip) <= 1.0
    assert _nearest(corners, cusp) <= 1.0


def test_noisy_blob_has_no_corners_and_smooth_joins():
    fitted = _fit_noisy(_noisy(_blob()))
    breaks = [brk for _, brk in _joins(fitted.segments)]
    assert max(breaks) <= 5.0


# ---------------------------------------------------------------- adaptive=False is the old fit


def _fingerprint(fitted) -> str:
    """Hash of every segment and primitive coordinate, rounded to 1e-6 px."""
    import hashlib

    def norm(value):
        if isinstance(value, float):
            return round(value, 6) + 0.0  # + 0.0 folds -0.0 into 0.0
        if isinstance(value, (tuple, list)):
            return tuple(norm(v) for v in value)
        if hasattr(value, "__dataclass_fields__"):
            return (type(value).__name__, *(norm(getattr(value, k)) for k in value.__dataclass_fields__))
        return value

    return hashlib.sha256(repr(norm(tuple(fitted))).encode()).hexdigest()[:16]


def _fit_image(pixels, params):
    h, w = pixels.shape[:2]
    image = RasterImage(pixels, (w, h))
    regions = segment(image, SegmentParams())
    return fit_geometry(trace(regions), params, context=make_edge_context(image, regions))


LEGACY_SCENES = {
    **{f"flat_shapes_{s}": (lambda s=s: flat_shapes(s)) for s in range(10)},
    **{f"aa_shapes_{s}": (lambda s=s: aa_shapes(s)) for s in range(4)},
    "aa_disc": aa_disc,
}
# Fingerprints of `fit_geometry(..., FitParams())` before adaptive fitting existed (master 2f293c9,
# i.e. with fix-3 segmentation; flat_shapes are unchanged from d1f3181).
LEGACY_FINGERPRINTS = {
    "flat_shapes_0": "c319cabe2c17ef50",
    "flat_shapes_1": "f4f6a10d441d486b",
    "flat_shapes_2": "60241681ba920415",
    "flat_shapes_3": "666bae0d9c254367",
    "flat_shapes_4": "e9bae56143436573",
    "flat_shapes_5": "ddc75123d21722c8",
    "flat_shapes_6": "3bb1827fdd870ac0",
    "flat_shapes_7": "92f6b40b56d2b81f",
    "flat_shapes_8": "5c1291687cf08695",
    "flat_shapes_9": "38e3018e7ef2de6b",
    "aa_shapes_0": "746534025274d358",
    "aa_shapes_1": "566cd5310ac89655",
    "aa_shapes_2": "16f14b497adb4022",
    "aa_shapes_3": "6b7e8d6f74cfee8d",
    "aa_disc": "522f2717b4fb6d45",
}


@pytest.mark.parametrize("name", list(LEGACY_SCENES))
def test_adaptive_off_reproduces_the_previous_fit(name):
    fitted = _fit_image(LEGACY_SCENES[name](), FitParams(adaptive=False))
    assert _fingerprint(fitted) == LEGACY_FINGERPRINTS[name]


# ---------------------------------------------------------------- parameter and contract


def test_fit_params_adaptive_is_a_validated_bool_and_overridable():
    from engine.errors import InputError
    from engine.presets import apply_overrides, resolve_preset

    assert FitParams().adaptive is True
    with pytest.raises(ValueError):
        FitParams(adaptive=1)
    preset = apply_overrides(resolve_preset("auto"), {"fit.adaptive": False})
    assert preset.fit.adaptive is False
    with pytest.raises(InputError):
        apply_overrides(resolve_preset("auto"), {"fit.adaptive": "no"})


@pytest.mark.parametrize("seed", range(3))
def test_noisy_open_curves_keep_their_junctions_exactly(seed):
    """A rect over a split background: its outline is two open curves between junctions."""
    verts = _rotated_rect_vertices(w=60.0, h=40.0)
    rect = aa_polygon(verts, SIZE)
    rect[:, SIZE // 2 :][(rect[:, SIZE // 2 :, :3] == 255).all(axis=2)] = (250, 220, 60, 255)
    pixels = _noisy(rect, seed)
    image = RasterImage(pixels, (SIZE, SIZE))
    regions = segment(image, SegmentParams(max_colors=3))
    graph = trace(regions)
    fitted = fit_geometry(graph, FitParams(), context=make_edge_context(image, regions))
    for f, c in zip(fitted, graph.curves):
        if c.closed:
            assert f.segments[-1].p1 == f.segments[0].p0
        else:
            assert f.segments[0].p0 == c.points[0] and f.segments[-1].p1 == c.points[-1]
        assert all(a.p1 == b.p0 for a, b in zip(f.segments, f.segments[1:]))


# ---------------------------------------------------------------- real corners on clean art


def _rotate_about(points, deg, centre=(64.0, 64.0)):
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    cx, cy = centre
    return tuple((cx + (x - cx) * c - (y - cy) * s, cy + (x - cx) * s + (y - cy) * c) for x, y in points)


# A thick L stroke: five convex 90 degree corners and one concave (inner) 90 degree corner.
L_STROKE = ((22.3, 18.6), (44.7, 18.6), (44.7, 84.2), (104.4, 84.2), (104.4, 106.7), (22.3, 106.7))
LIGHT = dict(noise=3.0, quality=80)


SOFT_BLUR_PX = 1.0  # a soft-edged scan / resampled print: corners read below 60 deg coarse


def _lightly_noisy(pixels, seed=0):
    return jpeg_roundtrip(add_noise(pixels, LIGHT["noise"], seed), LIGHT["quality"])


def _soft(pixels):
    from PIL import Image, ImageFilter

    out = pixels.copy()
    out[..., :3] = np.asarray(Image.fromarray(pixels[..., :3]).filter(ImageFilter.GaussianBlur(SOFT_BLUR_PX)))
    return jpeg_roundtrip(out, LIGHT["quality"])


@pytest.mark.parametrize("deg", [0.0, 20.0])
@pytest.mark.parametrize("degrade", ["clean", "light", "soft"])
def test_l_stroke_keeps_every_right_angle(deg, degrade):
    verts = _rotate_about(L_STROKE, deg)
    pixels = aa_polygon(verts, SIZE, fg=(40, 36, 36, 255), bg=(232, 186, 196, 255))
    pixels = {"clean": lambda p: p, "light": _lightly_noisy, "soft": _soft}[degrade](pixels)
    corners = _corners(_fit_noisy(pixels).segments)
    for v in verts:
        assert _nearest(corners, v) <= 1.0, v



# ---------------------------------------------------------------- merge pass internals


def _straight_then_up(overshoot: float):
    """Open curve along y = 0 to a corner at (20, 0), then up; samples 18, 19 overshoot it."""
    raw = [(float(x), 0.0) for x in range(21)] + [(20.0, -float(y)) for y in range(1, 21)]
    raw[18], raw[19] = (20.0 + overshoot / 2, 0.0), (20.0 + overshoot, 0.0)
    return tuple(raw)


def test_merging_a_corner_keeps_neighbouring_corners_collapsed():
    from engine.geometry.adaptive_fit import _Curve, _Merge, _merge_spans
    from engine.geometry.corner_placement import collapse_behind

    raw = _straight_then_up(0.8)
    params = FitParams()
    curve = _Curve(raw, False, params, replace_tol(params, 0.375), 0.15)
    corners = (0, 10, 20, 40)  # 10 is a fake corner in the middle of the straight run
    seq = collapse_behind(raw, [10, 20], list(corners), False, 4)
    merged = _merge_spans(curve, _Merge(raw, tuple(seq), corners), {10: 36.0, 20: 61.0})
    assert merged.corners == (0, 20, 40)
    corner = np.asarray(merged.seq[20])
    for i in range(16, 20):
        assert (np.asarray(merged.seq[i]) - corner) @ np.array([1.0, 0.0]) <= 1e-9, i


def replace_tol(params, tol):
    from dataclasses import replace

    return replace(params, tolerance_px=tol)


def test_one_corner_ring_has_no_zero_chord_segment():
    """A teardrop has one corner, so its single run starts and ends at the same point."""

    def inside(x, y):
        disc = (x - 64.2) ** 2 + (y - 74.1) ** 2 <= 26.0**2
        cone = (np.abs(x - 64.2) <= (y - 30.3) * 0.75) & (y <= 74.1)
        return disc | cone

    fitted = _fit_noisy(aa_mask(inside, SIZE))
    assert all(s.p0 != s.p1 for s in fitted.segments)


def test_a_run_that_returns_to_its_start_is_split_not_fitted_as_a_zero_chord():
    from engine.geometry.adaptive_fit import _fit_run

    t = np.linspace(0.0, 2 * math.pi, 40)
    loop = [(10.0 + 5.0 * math.sin(a), 10.0 + 5.0 - 5.0 * math.cos(a)) for a in t]
    loop[-1] = loop[0]
    segs = _fit_run(loop, FitParams(), 0.15)
    assert len(segs) >= 2 and all(s.p0 != s.p1 for s in segs)
    assert segs[0].p0 == loop[0] and segs[-1].p1 == loop[-1]
    assert _fit_run([(3.0, 4.0)] * 5, FitParams(), 0.15) == ()


# ---------------------------------------------------------------- shared boundaries stay disjoint


def _arc_points(s: Arc) -> np.ndarray:
    a0 = math.atan2(s.p0[1] - s.center[1], s.p0[0] - s.center[0])
    a1 = math.atan2(s.p1[1] - s.center[1], s.p1[0] - s.center[0])
    travel = (a1 - a0) % (2 * math.pi) if s.sweep else -((a0 - a1) % (2 * math.pi))
    ang = a0 + np.linspace(0.0, 1.0, max(2, int(abs(travel) * s.radius / 0.5) + 2)) * travel
    pts = np.column_stack([s.center[0] + s.radius * np.cos(ang), s.center[1] + s.radius * np.sin(ang)])
    pts[0], pts[-1] = s.p0, s.p1
    return pts


def _polyline(fitted) -> np.ndarray:
    from engine.geometry.bezier import sample

    parts = []
    for s in fitted.segments:
        if isinstance(s, Line):
            parts.append(np.array([s.p0, s.p1], float))
        elif isinstance(s, Arc):
            parts.append(_arc_points(s))
        else:
            parts.append(sample(s, np.linspace(0.0, 1.0, 24)))
    return np.vstack([parts[0]] + [p[1:] for p in parts[1:]]) if parts else np.zeros((0, 2))


def _properly_cross(a: np.ndarray, b: np.ndarray) -> bool:
    """True when polylines a and b cross somewhere other than at their segment end points."""
    if min(len(a), len(b)) < 2 or (a.min(0) > b.max(0)).any() or (b.min(0) > a.max(0)).any():
        return False
    p, r = a[:-1, None, :], (a[1:] - a[:-1])[:, None, :]
    q, s = b[None, :-1, :], (b[1:] - b[:-1])[None, :, :]
    rxs = r[..., 0] * s[..., 1] - r[..., 1] * s[..., 0]
    qp = q - p
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qp[..., 0] * s[..., 1] - qp[..., 1] * s[..., 0]) / rxs
        u = (qp[..., 0] * r[..., 1] - qp[..., 1] * r[..., 0]) / rxs
    eps = 1e-6
    inside = (t > eps) & (t < 1 - eps) & (u > eps) & (u < 1 - eps)
    return bool(((np.abs(rxs) > 1e-12) & inside).any())


def _crossing_pairs(fitted) -> list[tuple[int, int]]:
    polys = [_polyline(f) for f in fitted]
    return [
        (fitted[i].curve_id, fitted[j].curve_id)
        for i in range(len(polys))
        for j in range(i + 1, len(polys))
        if _properly_cross(polys[i], polys[j])
    ]


def _outlined_blob() -> np.ndarray:
    """1.jpg-like lettering: a pink blob in a thin white band in a ~3 px dark outline, on grey."""
    from eval.synthetic import supersample

    def draw(n):
        k = n / SIZE
        ys, xs = np.mgrid[0:n, 0:n]
        x, y = (xs + 0.5) / k - 64.0, (ys + 0.5) / k - 64.0
        r = np.hypot(x, y) + 6.0 * np.sin(3.0 * np.arctan2(y, x))
        px = np.empty((n, n, 4), np.uint8)
        px[:] = (120, 116, 116, 255)
        px[r <= 44.0] = (40, 36, 36, 255)
        px[r <= 41.0] = (250, 250, 250, 255)
        px[r <= 38.5] = (236, 190, 200, 255)
        return px

    return supersample(draw, SIZE)


CROSSING_SCENES = {
    **{f"aa_shapes_{s}": (lambda s=s: _noisy(aa_shapes(s))) for s in range(4)},
    "outlined_blob": lambda: _noisy(_outlined_blob()),
    "outlined_blob_soft": lambda: _soft(_outlined_blob()),
}


@pytest.mark.parametrize("name", list(CROSSING_SCENES))
def test_distinct_fitted_curves_never_cross(name):
    pixels = CROSSING_SCENES[name]()
    image = RasterImage(pixels, pixels.shape[1::-1])
    regions = segment(image, SegmentParams())
    fitted = fit_geometry(trace(regions), FitParams(), context=make_edge_context(image, regions))
    assert _crossing_pairs(fitted) == []
    assert all(s.p0 != s.p1 for f in fitted for s in f.segments)


def test_no_hairpin_from_corners_next_to_a_junction():
    """Phone-photo case: sub-pixel samples beside a junction put two corners on top of it."""
    from engine.geometry.adaptive_fit import fit_curve_adaptive
    from engine.geometry.corners import dense_samples
    from engine.types import BoundaryCurve

    curve = BoundaryCurve(0, ((361.0, 274.0), (360.0, 274.0), (360.0, 298.0)), 1, 2, False)
    xs = [360.5, 360.56, 360.48, 360.88, 360.92, 360.96, 360.9, 360.9, 360.82, 360.87, 360.95,
          360.89, 360.82, 360.8, 360.78, 360.61, 360.69, 360.73, 360.74, 360.77, 360.81, 360.74,
          360.63, 360.72, 360.38]
    raw = ((361.0, 274.0), (360.5, 274.93), *[(x, 273.5 + i) for i, x in enumerate(xs[1:], 1)],
           (360.0, 298.0))
    plain = dense_samples(curve.points, False)
    assert len(raw) == len(plain)
    fitted = fit_curve_adaptive(curve, (raw, plain), FitParams(), 1.0, 2, 4)
    breaks = [brk for _, brk in _joins(list(fitted.segments))[:-1]]  # open: no wrap-around join
    assert max(breaks, default=0.0) < 150.0
    assert all(s.p1 != curve.points[0] for s in fitted.segments)
