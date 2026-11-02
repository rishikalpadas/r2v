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
