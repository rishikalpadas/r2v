import numpy as np
import pytest

from engine.presets import ExportParams, FitParams, SegmentParams
from engine.stages.export import _reverse_segment, build_doc
from engine.stages.fit_geometry import fit_geometry, polyline_fit
from engine.stages.segment import segment
from engine.stages.trace import trace
from engine.types import OUTSIDE, BoundaryCurve, BoundaryGraph, Circle, Ellipse, Line, RasterImage
from eval.synthetic import circle, flat_shapes, ring_in_ring, transparent_logo


def _regions_graph(pixels):
    h, w = pixels.shape[:2]
    regions = segment(RasterImage(pixels=pixels, original_size=(w, h)), SegmentParams())
    return regions, trace(regions)


def _graph(pixels) -> BoundaryGraph:
    return _regions_graph(pixels)[1]


CASES = {
    "flat_shapes_0": lambda: flat_shapes(0),
    "flat_shapes_3": lambda: flat_shapes(3),
    "flat_shapes_9": lambda: flat_shapes(9),
    "ring_in_ring": ring_in_ring,
    "transparent_logo": transparent_logo,
}


@pytest.mark.parametrize("name", list(CASES))
def test_endpoints_unchanged(name):
    """Open curves end at shared junctions, which must not move. A closed curve has no
    junction (its start is arbitrary and the fit restarts it at a corner); it must close."""
    graph = _graph(CASES[name]())
    fitted = {f.curve_id: f for f in fit_geometry(graph, FitParams())}
    for curve in graph.curves:
        segs = fitted[curve.id].segments
        assert segs, f"curve {curve.id} has no segments"
        if curve.closed:
            assert segs[-1].p1 == segs[0].p0
        else:
            assert segs[0].p0 == curve.points[0]
            assert segs[-1].p1 == curve.points[-1]


def test_canvas_edge_curves_stay_exact():
    graph = _graph(flat_shapes(0))
    fitted = fit_geometry(graph, FitParams())
    poly = polyline_fit(graph)
    for f, p, c in zip(fitted, poly, graph.curves):
        if OUTSIDE in (c.left, c.right):
            assert f == p


@pytest.mark.parametrize("name", list(CASES))
def test_chains_are_continuous(name):
    graph = _graph(CASES[name]())
    for f in fit_geometry(graph, FitParams()):
        for a, b in zip(f.segments, f.segments[1:]):
            assert a.p1 == b.p0


def test_one_fitted_curve_per_boundary_curve():
    graph = _graph(flat_shapes(2))
    fitted = fit_geometry(graph, FitParams())
    assert [f.curve_id for f in fitted] == [c.id for c in graph.curves]


def test_disabled_returns_polyline_fit():
    graph = _graph(flat_shapes(4))
    assert fit_geometry(graph, FitParams(enabled=False)) == polyline_fit(graph)


def test_fitting_reduces_segments_on_curved_shapes():
    graph = _graph(ring_in_ring())
    fitted = sum(len(f.segments) for f in fit_geometry(graph, FitParams()))
    poly = sum(len(f.segments) for f in polyline_fit(graph))
    assert fitted * 4 <= poly


def test_closed_circle_gets_primitive_and_segments():
    graph = _graph(circle())
    pairs = zip(fit_geometry(graph, FitParams()), graph.curves)
    closed = [f for f, c in pairs if c.closed and OUTSIDE not in (c.left, c.right)]
    assert len(closed) == 1
    shape = closed[0].primitive
    assert isinstance(shape, Circle)
    assert abs(shape.cx - 64) <= 1.0 and abs(shape.cy - 64) <= 1.0 and abs(shape.r - 40) <= 1.0
    assert not all(isinstance(s, Line) for s in closed[0].segments)


def test_open_curves_get_no_primitive():
    graph = _graph(flat_shapes(0))
    for f, c in zip(fit_geometry(graph, FitParams()), graph.curves):
        if not c.closed:
            assert f.primitive is None


def test_short_closed_curve_falls_back_to_polyline():
    square = ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (0.0, 0.0))
    graph = BoundaryGraph(curves=(BoundaryCurve(0, square, 0, 1, True),), region_loops={})
    (fitted,) = fit_geometry(graph, FitParams())
    assert fitted == polyline_fit(graph)[0]
    assert fitted.primitive is None


def test_short_open_curve_falls_back_to_polyline():
    pts = ((3.0, 1.0), (4.0, 1.0), (4.0, 2.0))
    graph = BoundaryGraph(curves=(BoundaryCurve(0, pts, 0, 1, False),), region_loops={})
    assert fit_geometry(graph, FitParams()) == polyline_fit(graph)


def test_closed_curve_without_corners_closes_exactly():
    graph = _graph(circle(r=20))
    for f, c in zip(fit_geometry(graph, FitParams()), graph.curves):
        if c.closed:
            assert f.segments[-1].p1 == f.segments[0].p0


def test_same_curve_both_sides():
    regions, graph = _regions_graph(ring_in_ring())
    fitted = fit_geometry(graph, FitParams())
    doc = build_doc(regions, graph, fitted, ExportParams(layer_mode="cutout"))
    by_curve = {f.curve_id: f for f in fitted}
    holed = [p for p in doc.paths if len(p.contours) == 2]
    assert len(holed) >= 2
    shared = 0
    for outer in holed:
        for inner in doc.paths:
            hole, contour = outer.contours[1], inner.contours[0]
            if hole == tuple(_reverse_segment(s) for s in reversed(contour)):
                own = [f.segments for f in by_curve.values()]
                assert contour in own or hole in own
                assert not all(isinstance(s, Line) for s in contour)
                shared += 1
    assert shared >= 2


def _rect_image(x0, y0, w, h, size=64):
    px = np.full((size, size, 4), 255, np.uint8)
    px[y0 : y0 + h, x0 : x0 + w] = (30, 60, 200, 255)
    return px


def _closed_inner(graph):
    pairs = zip(fit_geometry(graph, FitParams()), graph.curves)
    return [f for f, c in pairs if c.closed and OUTSIDE not in (c.left, c.right)]


def test_corners_snap_to_pixel_grid_vertices():
    (fitted,) = _closed_inner(_graph(_rect_image(10, 12, 30, 20)))
    ends = {s.p0 for s in fitted.segments}
    assert {(10, 12), (40, 12), (40, 32), (10, 32)} <= ends


def test_narrow_rect_keeps_both_end_corners():
    # A 4 px wide strip: with the default corner window the two corners at each end merge and
    # the strip was turned into an ellipse.
    (fitted,) = _closed_inner(_graph(_rect_image(11, 12, 4, 39)))
    assert not isinstance(fitted.primitive, (Circle, Ellipse))
    ends = {s.p0 for s in fitted.segments}
    assert {(11, 12), (15, 12), (15, 51), (11, 51)} <= ends
