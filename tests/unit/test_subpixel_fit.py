"""Fit stage and pipeline with sub-pixel edge samples (anti-aliased input)."""
import io
import math

import numpy as np
import pytest
from PIL import Image

from engine import to_svg, vectorize
from engine.geometry.subpixel import make_edge_context
from engine.presets import FitParams, SegmentParams
from engine.stages.fit_geometry import fit_geometry
from engine.stages.segment import _to_lab, segment
from engine.stages.trace import trace
from engine.types import OUTSIDE, Arc, Circle, CubicBezier, Line, RasterImage, Rect, RegionMap
from eval.metrics import count_nodes
from eval.synthetic import aa_disc, flat_shapes, ring_in_ring, solid, supersample, transparent_logo

WHITE = (255, 255, 255, 255)
BLUE = (30, 60, 200, 255)
DISC = dict(cx=64.3, cy=63.7, r=40.4)
# aa_disc has ~60 blend colours; with the default max_colors the colour merge chains through the
# blend clusters and fuses disc and background (a segmentation issue, out of scope here). Two
# colours is what the image really has.
TWO_COLORS = {"segment.max_colors": 2}


def _ideal_regions(pixels: np.ndarray, colors) -> RegionMap:
    rgb = pixels[..., :3].astype(float)
    dist = np.stack([np.linalg.norm(rgb - np.array(c[:3], float), axis=2) for c in colors])
    labels = np.argmin(dist, axis=0).astype(np.int32)
    palette = np.array([c[:3] for c in colors], np.uint8)
    return RegionMap(labels, palette, _to_lab(palette), frozenset())


def _fit(pixels, regions, params=FitParams()):
    h, w = pixels.shape[:2]
    graph = trace(regions)
    ctx = make_edge_context(RasterImage(pixels, (w, h)), regions)
    return graph, fit_geometry(graph, params, context=ctx)


def _inner_closed(graph, fitted):
    return [f for f, c in zip(fitted, graph.curves) if c.closed and OUTSIDE not in (c.left, c.right)]


def _sample_segment(seg, n=16) -> np.ndarray:
    ts = np.linspace(0.0, 1.0, n)
    if isinstance(seg, Line):
        p0, p1 = np.array(seg.p0), np.array(seg.p1)
        return p0 + ts[:, None] * (p1 - p0)
    if isinstance(seg, CubicBezier):
        p = [np.array(q) for q in (seg.p0, seg.c1, seg.c2, seg.p1)]
        u = 1 - ts[:, None]
        t = ts[:, None]
        return u**3 * p[0] + 3 * u * u * t * p[1] + 3 * u * t * t * p[2] + t**3 * p[3]
    assert isinstance(seg, Arc)
    c = np.array(seg.center)
    a0 = math.atan2(seg.p0[1] - c[1], seg.p0[0] - c[0])
    a1 = math.atan2(seg.p1[1] - c[1], seg.p1[0] - c[0])
    sweep = (a1 - a0) % (2 * math.pi) if seg.sweep else -((a0 - a1) % (2 * math.pi))
    ang = a0 + ts * sweep
    return c + seg.radius * np.column_stack([np.cos(ang), np.sin(ang)])


def _radial_error(segments, cx, cy, r) -> float:
    pts = np.vstack([_sample_segment(s) for s in segments])
    return float(np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r).max())


# ---------------------------------------------------------------- fit stage


def test_fit_params_subpixel_is_a_validated_bool():
    assert FitParams().subpixel is True
    with pytest.raises(ValueError):
        FitParams(subpixel=1)


def test_disc_outline_follows_the_true_circle():
    pixels = aa_disc(**DISC, fg=BLUE, bg=WHITE)
    regions = _ideal_regions(pixels, (WHITE, BLUE))
    graph, fitted = _fit(pixels, regions)
    (disc,) = _inner_closed(graph, fitted)
    assert _radial_error(disc.segments, **DISC) <= 0.06  # crack-grid samples: 0.116
    assert isinstance(disc.primitive, Circle)
    assert abs(disc.primitive.r - DISC["r"]) <= 0.05


def test_subpixel_off_ignores_context():
    pixels = aa_disc(**DISC, fg=BLUE, bg=WHITE)
    regions = _ideal_regions(pixels, (WHITE, BLUE))
    h, w = pixels.shape[:2]
    graph = trace(regions)
    ctx = make_edge_context(RasterImage(pixels, (w, h)), regions)
    off = FitParams(subpixel=False)
    assert fit_geometry(graph, off, context=ctx) == fit_geometry(graph, off)
    assert fit_geometry(graph, FitParams(), context=ctx) != fit_geometry(graph, FitParams())


RECT = (10.3, 12.6, 40.7, 35.2)  # x0, y0, x1, y1 at sub-pixel positions


def _aa_rect(size=48):
    x0, y0, x1, y1 = RECT

    def draw(n):
        k = n / size
        c = (np.arange(n) + 0.5) / k
        px = solid(n, WHITE)
        inside = ((c >= y0) & (c < y1))[:, None] & ((c >= x0) & (c < x1))[None, :]
        px[inside] = BLUE
        return px

    return supersample(draw, size, factor=16)


def test_rect_corners_land_on_true_corners():
    pixels = _aa_rect()
    graph, fitted = _fit(pixels, _ideal_regions(pixels, (WHITE, BLUE)))
    (rect,) = _inner_closed(graph, fitted)
    x0, y0, x1, y1 = RECT
    ends = np.array([s.p0 for s in rect.segments])
    for corner in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        assert np.hypot(*(ends - corner).T).min() <= 0.1, corner
    shape = rect.primitive
    assert isinstance(shape, Rect)
    assert np.allclose((shape.x, shape.y, shape.w, shape.h), (x0, y0, x1 - x0, y1 - y0), atol=0.1)


NON_AA = {
    **{f"flat_shapes_{s}": (lambda s=s: flat_shapes(s)) for s in range(10)},
    "ring_in_ring": ring_in_ring,
    "transparent_logo": transparent_logo,
}


@pytest.mark.parametrize("name", list(NON_AA))
def test_non_antialiased_fit_is_unchanged(name):
    pixels = NON_AA[name]()
    h, w = pixels.shape[:2]
    image = RasterImage(pixels, (w, h))
    regions = segment(image, SegmentParams())
    graph = trace(regions)
    ctx = make_edge_context(image, regions)
    assert fit_geometry(graph, FitParams(), context=ctx) == fit_geometry(graph, FitParams())


# ---------------------------------------------------------------- whole pipeline


def _circle_of(result) -> Circle:
    circles = [p.shape for p in result.doc.primitives if isinstance(p.shape, Circle)]
    assert len(circles) == 1
    return circles[0]


def _png(pixels):
    buf = io.BytesIO()
    Image.fromarray(pixels, "RGBA").save(buf, format="PNG")
    return buf.getvalue()


def _jpeg(pixels, quality=75):
    buf = io.BytesIO()
    Image.fromarray(pixels[..., :3], "RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def test_pipeline_aa_disc_circle():
    shape = _circle_of(vectorize(_png(aa_disc(**DISC)), overrides=TWO_COLORS))
    assert abs(shape.cx - DISC["cx"]) <= 0.15 and abs(shape.cy - DISC["cy"]) <= 0.15
    assert abs(shape.r - DISC["r"]) <= 0.15


def test_pipeline_transparent_aa_disc_circle():
    pixels = aa_disc(**DISC, fg=(30, 90, 200, 255), bg=(0, 0, 0, 0))
    shape = _circle_of(vectorize(_png(pixels)))
    assert abs(shape.cx - DISC["cx"]) <= 0.2 and abs(shape.cy - DISC["cy"]) <= 0.2
    assert abs(shape.r - DISC["r"]) <= 0.2


def test_pipeline_jpeg_aa_disc():
    data = _jpeg(aa_disc(**DISC))
    on = vectorize(data, overrides=TWO_COLORS)
    off = vectorize(data, overrides={**TWO_COLORS, "fit.subpixel": False})
    assert abs(_circle_of(on).r - DISC["r"]) <= 0.3
    assert count_nodes(to_svg(on.doc))[0] <= count_nodes(to_svg(off.doc))[0]


def test_pipeline_subpixel_override_reaches_fit_stage():
    data = _png(aa_disc(**DISC))
    on = vectorize(data, overrides=TWO_COLORS)
    off = vectorize(data, overrides={**TWO_COLORS, "fit.subpixel": False})
    assert on.doc != off.doc
    fit_report = next(r for r in on.reports if r.stage == "fit")
    assert fit_report.params["subpixel"] is True


YELLOW = (250, 220, 60, 255)


def _aa_rect_on_split_background(size=48):
    """RECT over a background that is white left of x = 24.5 and yellow right of it."""
    x0, y0, x1, y1 = RECT

    def draw(n):
        k = n / size
        c = (np.arange(n) + 0.5) / k
        px = solid(n, WHITE)
        px[:, c >= 24.5] = YELLOW
        inside = ((c >= y0) & (c < y1))[:, None] & ((c >= x0) & (c < x1))[None, :]
        px[inside] = BLUE
        return px

    return supersample(draw, size, factor=16)


def test_open_curve_corners_land_on_true_corners():
    pixels = _aa_rect_on_split_background()
    graph, fitted = _fit(pixels, _ideal_regions(pixels, (WHITE, YELLOW, BLUE)))
    blue = 2
    open_rect = [
        f for f, c in zip(fitted, graph.curves) if not c.closed and blue in (c.left, c.right)
        and OUTSIDE not in (c.left, c.right)
    ]
    assert len(open_rect) == 2  # the rect outline is cut in two by the background split
    ends = np.array([s.p0 for f in open_rect for s in f.segments])
    x0, y0, x1, y1 = RECT
    for corner in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        assert np.hypot(*(ends - corner).T).min() <= 0.1, corner


def test_corner_line_intersection_rejects_near_parallel_lines():
    from engine.stages.fit_geometry import _line_intersection

    a = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    shallow = np.array([[3.0, 0.0], [4.0, 0.2], [5.0, 0.4]])  # ~11 degrees
    steep = np.array([[3.0, 1.0], [3.0, 2.0], [3.0, 3.0]])
    assert _line_intersection(a, shallow) is None
    assert _line_intersection(a, steep) == pytest.approx((3.0, 0.0))


def test_subpixel_corner_needs_samples_on_both_sides_and_a_near_hit():
    from engine.stages.fit_geometry import _subpixel_corner

    raw = [(x + 0.5, 0.2) for x in range(6)] + [(6.2, y + 0.5) for y in range(6)]
    dense = list(raw)
    assert _subpixel_corner(dense, raw, 6, (-1, 12), 1.0) == pytest.approx((6.2, 0.2))
    assert _subpixel_corner(dense, raw, 6, (5, 12), 1.0) is None  # no samples before
    far = [(p[0] + 3.0, p[1]) for p in dense]  # smoothed corner 3 px away from the hit
    assert _subpixel_corner(far, raw, 6, (-1, 12), 1.0) is None
