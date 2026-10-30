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
    assert fit_arc([(float(i), 0.001 * i * i) for i in range(20)]) is None
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
