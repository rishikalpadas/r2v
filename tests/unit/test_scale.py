"""Processing scale: thresholds are in original pixels, the document is in original units."""
import numpy as np
import pytest

from engine import to_svg
from engine.pipeline import run_pipeline
from engine.presets import PRESETS, ExportParams, FitParams, SegmentParams
from engine.stages.export import build_doc
from engine.stages.fit_geometry import fit_geometry, polyline_fit
from engine.stages.segment import segment
from engine.types import (
    OUTSIDE,
    Arc,
    BoundaryCurve,
    BoundaryGraph,
    Circle,
    CubicBezier,
    Ellipse,
    FittedCurve,
    Line,
    Polygon,
    RasterImage,
    Rect,
    RegionMap,
)
from eval.metrics import render_svg, ssim
from eval.synthetic import circle, flat_shapes


def _upscale(pixels: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(pixels, factor, axis=0), factor, axis=1)


def _image(pixels: np.ndarray, scale: float = 1.0, original_size=None) -> RasterImage:
    h, w = pixels.shape[:2]
    return RasterImage(pixels=pixels, original_size=original_size or (w, h), scale=scale)


@pytest.mark.parametrize("pixels", [flat_shapes(0), circle()], ids=["flat_shapes_0", "circle"])
def test_upscaled_image_maps_back_to_original_units(pixels):
    h, w = pixels.shape[:2]
    preset = PRESETS["logo"]
    base = run_pipeline(_image(pixels), preset).doc
    scaled = run_pipeline(_image(_upscale(pixels, 2), 2.0, (w, h)), preset).doc
    assert (scaled.width, scaled.height) == (w, h)
    base_render = render_svg(to_svg(base), w, h)
    scaled_render = render_svg(to_svg(scaled), w, h)
    assert ssim(base_render, scaled_render) >= 0.98


# ---------------------------------------------------------------- export: divide exactly once

_SQUARE = ((0, 0), (8, 0), (8, 8), (0, 8), (0, 0))


def _single_region(fitted_curve: FittedCurve):
    regions = RegionMap(
        np.zeros((8, 8), np.int32), np.array([[9, 9, 9]], np.uint8), np.zeros((1, 3)), frozenset()
    )
    graph = BoundaryGraph((BoundaryCurve(0, _SQUARE, 0, OUTSIDE, True),), {0: (((0, False),),)})
    return regions, graph, (fitted_curve,)


def test_segments_divided_by_scale_once():
    segs = (
        Line((0, 0), (8, 0)),
        Arc((8, 0), (8, 8), (8, 4), 4.0, False, True),
        CubicBezier((8, 8), (6, 10), (2, 10), (0, 8)),
        Line((0, 8), (0, 0)),
    )
    regions, graph, fitted = _single_region(FittedCurve(0, segs, None))
    doc = build_doc(regions, graph, fitted, ExportParams(), scale=2.0, original_size=(4, 4))
    assert (doc.width, doc.height) == (4, 4)
    assert doc.paths[0].contours[0] == (
        Line((0, 0), (4, 0)),
        Arc((4, 0), (4, 4), (4, 2), 2.0, False, True),
        CubicBezier((4, 4), (3, 5), (1, 5), (0, 4)),
        Line((0, 4), (0, 0)),
    )


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        (Circle(4, 4, 2), Circle(2, 2, 1)),
        (Ellipse(4, 4, 4, 2, 30.0), Ellipse(2, 2, 2, 1, 30.0)),
        (Rect(2, 2, 4, 6, 10.0), Rect(1, 1, 2, 3, 10.0)),
        (Polygon(((0, 0), (8, 0), (4, 6))), Polygon(((0, 0), (4, 0), (2, 3)))),
    ],
)
def test_primitives_divided_by_scale_once(shape, expected):
    regions, graph, fitted = _single_region(FittedCurve(0, (Line((0, 0), (8, 0)),), shape))
    doc = build_doc(regions, graph, fitted, ExportParams(), scale=2.0, original_size=(4, 4))
    assert doc.primitives[0].shape == expected


def test_canvas_underlay_uses_original_size():
    px = np.full((32, 32, 4), 255, np.uint8)
    px[0:12, 4:14] = (220, 30, 30, 255)  # touches the top edge -> background needs an underlay
    regions = segment(_image(px), SegmentParams())
    from engine.stages.trace import trace

    graph = trace(regions)
    doc = build_doc(regions, graph, polyline_fit(graph), ExportParams(), scale=2.0, original_size=(16, 16))
    bottom = min(doc.paths, key=lambda p: p.z_order)
    assert bottom.contours == (
        (Line((0, 0), (16, 0)), Line((16, 0), (16, 16)), Line((16, 16), (0, 16)), Line((0, 16), (0, 0))),
    )


# ---------------------------------------------------------------- thresholds in original pixels


def test_min_region_area_scales_with_scale_squared():
    px = np.full((16, 16, 4), 255, np.uint8)
    px[:, 8:] = (0, 0, 255, 255)
    # 3x3 dot (has an interior pixel, so it is not a speck) of a colour with another component
    px[2:5, 2:5] = (0, 0, 255, 255)
    params = SegmentParams(min_region_px=9)
    kept = segment(_image(px), params)
    assert kept.labels[3, 3] != kept.labels[0, 0]
    # The same 9 processing px are only 2.25 original px at scale 2: below min_region_px.
    merged = segment(_image(px, 2.0, (8, 8)), params)
    assert merged.labels[3, 3] == merged.labels[0, 0]


def test_speck_area_scales_with_scale_squared():
    px = np.full((24, 24, 4), 255, np.uint8)
    px[:, 12:] = (0, 0, 255, 255)
    px[3, 1:21] = (255, 0, 0, 255)  # 1-px line, 20 px, the only red: no interior pixel
    px[23, :, :3] = (np.arange(24) * 3 + 100)[:, None]  # grey ramp row: the noisy path
    kept = segment(_image(px), SegmentParams())
    assert kept.labels[3, 5] != kept.labels[0, 0]
    # 20 processing px are 5 original px at scale 2: below SPECK_MAX_PX, so a speck.
    merged = segment(_image(px, 2.0, (12, 12)), SegmentParams())
    assert merged.labels[3, 5] == merged.labels[0, 0]


def _closed_square_graph(side: int) -> BoundaryGraph:
    pts = ((0, 0), (side, 0), (side, side), (0, side), (0, 0))
    return BoundaryGraph((BoundaryCurve(0, pts, 1, 0, True),), {})


def test_min_curve_length_measured_in_original_pixels():
    graph = _closed_square_graph(5)  # length 20 processing px
    exact = polyline_fit(graph)
    assert fit_geometry(graph, FitParams()) != exact  # 20 px >= MIN_CLOSED_LENGTH_PX: fitted
    assert fit_geometry(graph, FitParams(), scale=2.0) == exact  # only 10 original px


@pytest.mark.parametrize("adaptive", [False, True])
def test_fit_tolerance_measured_in_original_pixels(monkeypatch, adaptive):
    seen = []

    def spy(points, params, noise=None, tangent_samples=1):
        seen.append(params.tolerance_px)
        return (Line(points[0], points[-1]),)

    monkeypatch.setattr("engine.stages.fit_geometry.fit_span", spy)
    monkeypatch.setattr("engine.geometry.adaptive_fit.fit_span", spy)
    graph = _closed_square_graph(10)
    fit_geometry(graph, FitParams(tolerance_px=0.75, adaptive=adaptive), scale=2.0)
    assert seen and all(t == pytest.approx(1.5) for t in seen)


# ---------------------------------------------------------------- sample-count windows


@pytest.mark.parametrize(
    ("count", "scale", "expected"), [(2, 1.0, 2), (4, 1.0, 4), (2, 4.0, 8), (3, 2.5, 8), (1, 0.25, 1)]
)
def test_samples_at_scale(count, scale, expected):
    from engine.geometry.corners import samples_at_scale

    assert samples_at_scale(count, scale) == expected


@pytest.mark.parametrize("scale", [1.0, 4.0])
def test_corner_window_covers_the_same_original_length(monkeypatch, scale):
    from engine.geometry import corners
    from engine.stages.fit_geometry import CORNER_WINDOW

    seen = []

    def spy(points, closed, angle_deg, window):
        seen.append(window)
        return corners.find_corners(points, closed, angle_deg, window)

    monkeypatch.setattr("engine.stages.fit_geometry.find_corners", spy)
    graph = _closed_square_graph(round(20 * scale))
    fit_geometry(graph, FitParams(adaptive=False), scale=scale)  # the fixed fit's find_corners
    assert seen == [round(CORNER_WINDOW * scale)]


def test_square_corners_found_at_scale4():
    # 40 px square at scale 4: sigma 4 spreads each turn over ~16 samples, so a 2-sample
    # window saw only a fraction of it. Four line sides need all four corners.
    (fitted,) = fit_geometry(_closed_square_graph(160), FitParams(), scale=4.0)
    assert len(fitted.segments) == 4 and all(isinstance(s, Line) for s in fitted.segments)


def test_subpixel_corner_side_samples_scale():
    from engine.geometry.corner_placement import _subpixel_corner

    # Two samples on each side past the gap: enough at scale 1, too few at scale 2.
    raw = [(x + 0.5, 0.2) for x in range(4)] + [(4.2, y + 0.5) for y in range(4)]
    assert _subpixel_corner(raw, raw, 4, (-1, 8), 1.0) == pytest.approx((4.2, 0.2))
    assert _subpixel_corner(raw, raw, 4, (-1, 8), 2.0) is None


def test_primitive_side_trim_scales():
    from engine.geometry.primitives import CORNER_TRIM, _sides

    arr = np.column_stack([np.arange(40.0), np.zeros(40)])
    (side_1,) = _sides(arr, (0,), 1.0)
    (side_2,) = _sides(arr, (0,), 2.0)
    assert len(side_1) == 41 - 2 * CORNER_TRIM
    assert len(side_2) == 41 - 4 * CORNER_TRIM
    # 21 samples: 5 left after trimming 8 per end, fewer than MIN_SIDE_POINTS x 2 = 6.
    assert _sides(arr[:20], (0,), 1.0) is not None
    assert _sides(arr[:20], (0,), 2.0) is None
