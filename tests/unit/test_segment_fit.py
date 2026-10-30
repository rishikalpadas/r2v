import math

import numpy as np
import pytest

from engine.geometry.segment_fit import fit_arc, fit_line, fit_span, snap_axis
from engine.presets import FitParams
from engine.types import Arc, CubicBezier, Line
from eval.metrics import render_svg

PARAMS = FitParams()


def _circle_pts(cx, cy, r, a0, a1, n=30):
    return [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in np.linspace(a0, a1, n)]


def _svg_center(arc: Arc):
    """SVG implementation-notes F.6.5 center for `A r r 0 large sweep p1` starting at p0."""
    (x1, y1), (x2, y2), r = arc.p0, arc.p1, arc.radius
    dx, dy = (x1 - x2) / 2, (y1 - y2) / 2
    k = math.sqrt(max(0.0, (r * r - dx * dx - dy * dy) / (dx * dx + dy * dy)))
    sign = -1 if arc.large_arc == arc.sweep else 1
    return ((x1 + x2) / 2 + sign * k * dy, (y1 + y2) / 2 - sign * k * dx)


def test_fit_line_exact():
    pts = [(float(i), 2.0 * i + 1) for i in range(10)]
    line, sse = fit_line(pts)
    assert isinstance(line, Line) and line.p0 == pts[0] and line.p1 == pts[-1]
    assert sse == pytest.approx(0.0, abs=1e-9)


def test_fit_arc_recovers_circle():
    arc, sse = fit_arc(_circle_pts(50, 40, 20, 0, math.pi / 2))
    assert math.dist(arc.center, (50, 40)) < 0.1
    assert abs(arc.radius - 20) < 0.1 and sse < 1e-6


def test_fit_arc_rejects_near_straight():
    assert fit_arc([(float(i), 0.0003 * i * i) for i in range(20)]) is None
    assert fit_arc([(0.0, 0.0), (1.0, 1.0)]) is None


@pytest.mark.parametrize("a0,a1", [
    (0.0, math.pi / 2),            # positive-angle direction (clockwise on screen)
    (math.pi / 2, 0.0),            # reversed (counter-clockwise on screen)
    (0.0, 1.5 * math.pi),          # >180 degrees, positive
    (1.5 * math.pi, 0.0),          # >180 degrees, negative
])
def test_arc_flags_match_svg_convention(a0, a1):
    pts = _circle_pts(50, 40, 20, a0, a1, n=41)
    arc, _ = fit_arc(pts)
    assert arc.sweep == (a1 > a0)
    assert arc.large_arc == (abs(a1 - a0) > math.pi)
    # the center an SVG renderer derives from (p0, p1, r, flags) is the fitted center
    assert math.dist(_svg_center(arc), (50, 40)) < 0.2
    # and the rendered stroke passes through the input mid-point
    mid = pts[len(pts) // 2]
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">'
        '<rect width="100" height="100" fill="white"/>'
        f'<path d="M{arc.p0[0]} {arc.p0[1]} A{arc.radius} {arc.radius} 0 '
        f'{int(arc.large_arc)} {int(arc.sweep)} {arc.p1[0]} {arc.p1[1]}" '
        'fill="none" stroke="black" stroke-width="2"/></svg>'
    )
    img = render_svg(svg, 100, 100)
    x, y = round(mid[0]), round(mid[1])
    assert img[y - 1:y + 2, x - 1:x + 2, 0].min() < 100


def test_fit_cubic_via_span_within_tolerance():
    t = np.linspace(0, 1, 40)
    pts = [(float(40 * x), float(10 * math.sin(2 * math.pi * x))) for x in t]
    segs = fit_span(pts, PARAMS)
    assert all(isinstance(s, CubicBezier) for s in segs) and len(segs) <= 2


def test_mdl_prefers_line():
    rng = np.random.default_rng(0)
    pts = [(float(i), 0.3 * i + float(rng.uniform(-0.2, 0.2))) for i in range(30)]
    segs = fit_span(pts, PARAMS)
    assert len(segs) == 1 and isinstance(segs[0], Line)


def test_mdl_prefers_arc_for_circle():
    segs = fit_span(_circle_pts(50, 40, 20, 0, math.pi, 40), PARAMS)
    assert len(segs) == 1 and isinstance(segs[0], Arc)


def test_endpoints_preserved():
    rng = np.random.default_rng(1)
    cases = [
        [(float(i), 0.5 * i) for i in range(12)],
        _circle_pts(50, 40, 20, 0, math.pi, 25),
        [(float(i), 9 * math.sin(i / 3) + float(rng.uniform(-.1, .1))) for i in range(40)],
    ]
    for pts in cases:
        pts = [(float(x), float(y)) for x, y in pts]
        segs = fit_span(pts, PARAMS)
        assert segs[0].p0 == pts[0] and segs[-1].p1 == pts[-1]
        for a, b in zip(segs, segs[1:]):
            assert a.p1 == b.p0
        assert fit_line(pts)[0].p0 == pts[0] and fit_line(pts)[0].p1 == pts[-1]
        arc = fit_arc(pts)
        if arc:
            assert arc[0].p0 == pts[0] and arc[0].p1 == pts[-1]


def test_degenerate_spans_do_not_crash():
    assert fit_span([(0.0, 0.0), (5.0, 1.0)], PARAMS) == (Line((0.0, 0.0), (5.0, 1.0)),)
    segs = fit_span([(3.0, 3.0)] * 5, PARAMS)
    assert segs and segs[0].p0 == (3.0, 3.0) and segs[-1].p1 == (3.0, 3.0)
    assert fit_span([(1.0, 1.0)], PARAMS) == ()


def test_axis_snap():
    l0 = Line((0.0, 10.0), (0.0, 0.0))
    l1 = Line((0.0, 0.0), (20.0, 0.7))  # about 2 degrees
    b2 = CubicBezier((20.0, 0.7), (20.0, 5.0), (25.0, 8.0), (30.0, 10.0))
    out = snap_axis((l0, l1, b2), PARAMS)
    assert out[1].p0[1] == out[1].p1[1] == pytest.approx(0.35)
    assert out[0].p1 == out[1].p0 and out[1].p1 == out[2].p0
    assert out[0].p0 == l0.p0 and out[2].p1 == b2.p1  # chain ends never move
    assert out[2].c1 == (20.0, pytest.approx(5.0 - 0.35))  # control point follows its endpoint


def test_axis_snap_leaves_far_and_boundary_lines():
    a = Line((0.0, 0.0), (20.0, 5.0))
    far = Line((20.0, 5.0), (40.0, 15.0))  # 27 degrees from horizontal: not snapped
    c = Line((40.0, 15.0), (40.0, 30.0))
    assert snap_axis((a, far, c), PARAMS) == (a, far, c)
    first = Line((0.0, 0.0), (20.0, 0.5))  # first/last segments are never moved
    assert snap_axis((first, c), PARAMS) == (first, c)
    assert snap_axis((first,), PARAMS) == (first,)
    arc = Arc((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), 1.0, False, True)
    mid = Line((0.0, 0.0), (20.0, 0.5))
    assert snap_axis((arc, mid, a), PARAMS)[1] == mid  # arc neighbour: untouched


def _rendered_arc_samples(arc: Arc, n=2000):
    cx, cy = _svg_center(arc)
    r = arc.radius
    a0 = math.atan2(arc.p0[1] - cy, arc.p0[0] - cx)
    a1 = math.atan2(arc.p1[1] - cy, arc.p1[0] - cx)
    d = a1 - a0
    if arc.sweep and d < 0:
        d += 2 * math.pi
    if not arc.sweep and d > 0:
        d -= 2 * math.pi
    a = a0 + d * np.linspace(0, 1, n)
    return np.column_stack((cx + r * np.cos(a), cy + r * np.sin(a)))


def _disc_ring(radius):
    from engine.geometry.corners import smooth_polyline
    from engine.geometry.stitching import build_boundary_graph
    from engine.presets import SegmentParams
    from engine.stages.segment import segment
    from engine.types import RasterImage
    from eval.synthetic import circle

    size = 2 * radius + 24
    px = circle(size=size, cx=size / 2 + 0.3, cy=size / 2 - 0.2, r=radius)
    regions = segment(RasterImage(pixels=px, original_size=(size, size)), SegmentParams())
    curves = [c for c in build_boundary_graph(regions).curves if c.closed and c.left >= 0 and c.right >= 0]
    return smooth_polyline(curves[0].points, closed=True, sigma=PARAMS.smooth_sigma)


@pytest.mark.parametrize("radius", [20, 40])
def test_chosen_arc_renders_within_tolerance_of_noisy_input(radius):
    ring = _disc_ring(radius)
    n = len(ring)
    checked = 0
    for deg in (150, 165, 180, 195, 210):
        length = int(n * deg / 360)
        for off in range(0, n, max(1, n // 40)):
            span = [ring[(off + k) % n] for k in range(length + 1)]
            segs = fit_span(span, PARAMS)
            if len(segs) != 1 or not isinstance(segs[0], Arc):
                continue
            curve = _rendered_arc_samples(segs[0])
            pts = np.array(span)
            worst = np.sqrt(((pts[:, None, :] - curve[None]) ** 2).sum(2)).min(1).max()
            assert worst <= PARAMS.tolerance_px + 1e-6, (radius, deg, off, worst)
            checked += 1
    assert checked > 0


def test_fit_arc_center_lies_on_endpoint_bisector():
    rng = np.random.default_rng(3)
    pts = [(x + float(rng.uniform(-.3, .3)), y + float(rng.uniform(-.3, .3)))
           for x, y in _circle_pts(50, 40, 20, 0.2, 0.2 + math.pi, 40)]
    arc, _ = fit_arc(pts)
    assert math.dist(arc.center, arc.p0) == pytest.approx(arc.radius, abs=1e-9)
    assert math.dist(arc.center, arc.p1) == pytest.approx(arc.radius, abs=1e-9)


def test_axis_snap_displacement_is_capped():
    long_line = Line((0.0, 0.0), (100.0, 5.0))  # ~2.9 degrees: ends would move 2.5 px
    out = snap_axis((Line((0.0, 20.0), (0.0, 0.0)), long_line, Line((100.0, 5.0), (100.0, 30.0))), PARAMS)
    assert out[1] == long_line
    small = Line((0.0, 0.0), (20.0, 0.7))  # moves 0.35 px: allowed
    out = snap_axis((Line((0.0, 20.0), (0.0, 0.0)), small, Line((20.0, 0.7), (20.0, 30.0))), PARAMS)
    assert out[1].p0[1] == out[1].p1[1]


def test_axis_snap_keeps_g1_with_bezier_neighbour():
    # smooth join: line direction (20, 0.7); bezier handle collinear with it
    line = Line((0.0, 0.0), (20.0, 0.7))
    d = (20.0, 0.7)
    ln = math.hypot(*d)
    handle = (20.0 + 6 * d[0] / ln, 0.7 + 6 * d[1] / ln)
    bez = CubicBezier((20.0, 0.7), handle, (35.0, 12.0), (40.0, 20.0))
    prev = Line((0.0, 20.0), (0.0, 0.0))
    out = snap_axis((prev, line, bez), PARAMS)
    assert out[1].p0[1] == out[1].p1[1]
    b = out[2]
    assert b.p0 == out[1].p1
    assert b.c1[1] == pytest.approx(b.p0[1], abs=1e-9)  # handle follows the now-horizontal line
    assert math.dist(b.c1, b.p0) == pytest.approx(6.0)  # handle length preserved
    assert b.c1[0] > b.p0[0]
