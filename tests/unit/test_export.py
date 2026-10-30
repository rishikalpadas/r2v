import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from engine.errors import InputError
from engine.pipeline import vectorize
from engine.presets import PRESETS, ExportParams, SegmentParams, apply_overrides
from engine.stages.export import build_doc, to_svg
from engine.stages.fit_geometry import polyline_fit
from engine.stages.segment import segment
from engine.stages.trace import trace
from engine.types import (
    Arc,
    BoundaryCurve,
    BoundaryGraph,
    Circle,
    CubicBezier,
    FittedCurve,
    Line,
    Path,
    Primitive,
    RasterImage,
    RegionMap,
    SolidFill,
    VectorDoc,
)
from eval.synthetic import circle, ring_in_ring, to_png_bytes

RED = SolidFill((255, 0, 0))


def _doc(paths=(), primitives=(), mode="stacked", precision=2) -> VectorDoc:
    return VectorDoc(10, 8, tuple(paths), tuple(primitives), mode, precision)


def _regions_graph(pixels, mode="stacked"):
    h, w = pixels.shape[:2]
    regions = segment(RasterImage(pixels=pixels, original_size=(w, h)), SegmentParams())
    graph = trace(regions)
    doc = build_doc(regions, graph, polyline_fit(graph), ExportParams(layer_mode=mode))
    return regions, graph, doc


def test_to_svg_parses():
    tri = (Line((0, 0), (5, 0)), Line((5, 0), (5, 5)), Line((5, 5), (0, 0)))
    root = ET.fromstring(to_svg(_doc([Path((tri,), RED, 0)])))
    assert root.tag.endswith("svg")
    assert root.attrib["viewBox"] == "0 0 10 8"
    assert root.attrib["width"] == "10" and root.attrib["height"] == "8"


def test_precision_rounding():
    contour = (Line((1.23456, 2), (3, 2)), Line((3, 2), (1.23456, 2)))
    svg = to_svg(_doc([Path((contour,), RED, 0)]))
    assert "M1.23 2" in svg and "1.23456" not in svg


def test_precision_param_respected():
    contour = (Line((1.23456, 2), (3, 2)),)
    assert "M1.2346 2" in to_svg(_doc([Path((contour,), RED, 0)], precision=4))


def test_arc_bezier_and_primitives_serialise():
    arc = Arc((0, 0), (10, 0), (5, 0), 5.0, False, True)
    bez = CubicBezier((10, 0), (10, 5), (0, 5), (0, 0))
    svg = to_svg(_doc([Path(((arc, bez),), RED, 1)], [Primitive(Circle(2, 3, 1.5), RED, 0)]))
    root = ET.fromstring(svg)
    assert [k.tag.split("}")[1] for k in root] == ["circle", "path"]  # z-order
    d = root[1].attrib["d"]
    assert "A5 5 0 0 1 10 0" in d and "C10 5 0 5 0 0" in d
    assert root[1].attrib["fill"] == "#ff0000"


def test_cutout_uses_evenodd_and_holes():
    _, _, doc = _regions_graph(ring_in_ring(), "cutout")
    root = ET.fromstring(to_svg(doc))
    paths = [e for e in root if e.tag.endswith("path")]
    assert all(p.attrib["fill-rule"] == "evenodd" for p in paths)
    assert sorted(p.attrib["d"].count("M") for p in paths) == [1, 2, 2, 2]


def test_stacked_emits_only_outer_loop_largest_first():
    _, _, doc = _regions_graph(ring_in_ring(), "stacked")
    assert [p.z_order for p in doc.paths] == [0, 1, 2, 3]
    assert all(len(p.contours) == 1 for p in doc.paths)
    assert "fill-rule" not in to_svg(doc)


def test_primitive_emitted_for_single_curve_region():
    regions, graph, _ = _regions_graph(circle())
    fitted = tuple(
        FittedCurve(f.curve_id, f.segments, Circle(64, 64, 40)) for f in polyline_fit(graph)
    )
    doc = build_doc(regions, graph, fitted, ExportParams())
    assert len(doc.primitives) >= 1
    assert len(doc.paths) + len(doc.primitives) == 2


def test_cutout_region_with_hole_stays_a_path_even_with_primitive():
    regions, graph, _ = _regions_graph(ring_in_ring())
    fitted = tuple(
        FittedCurve(f.curve_id, f.segments, Circle(0, 0, 1)) for f in polyline_fit(graph)
    )
    stacked = build_doc(regions, graph, fitted, ExportParams("stacked"))
    cutout = build_doc(regions, graph, fitted, ExportParams("cutout"))
    assert len(stacked.primitives) == 4 and not stacked.paths
    assert len(cutout.primitives) < len(stacked.primitives)


def test_reversed_loop_segments_are_reversed():
    graph = BoundaryGraph(
        curves=(BoundaryCurve(0, ((0, 0), (4, 0), (4, 4), (0, 4), (0, 0)), 0, -2, True),),
        region_loops={0: (((0, True),),)},
    )
    fitted = (
        FittedCurve(0, (Line((0, 0), (4, 0)), Arc((4, 0), (0, 0), (2, 0), 2, False, True)), None),
    )
    regions = RegionMap(
        np.zeros((4, 4), np.int32), np.array([[1, 2, 3]], np.uint8), np.zeros((1, 3)), frozenset()
    )
    segs = build_doc(regions, graph, fitted, ExportParams()).paths[0].contours[0]
    assert segs[0] == Arc((0, 0), (4, 0), (2, 0), 2, False, False)
    assert segs[1] == Line((4, 0), (0, 0))


def test_polyline_fit_merges_collinear_and_keeps_endpoints():
    pts = ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (0, 2), (0, 0))
    graph = BoundaryGraph((BoundaryCurve(0, pts, 0, -2, True),), {})
    (fc,) = polyline_fit(graph)
    assert [s.p0 for s in fc.segments] == [(0, 0), (2, 0), (2, 2), (0, 2)]
    assert fc.segments[-1].p1 == (0, 0)
    assert fc.primitive is None


def test_polyline_fit_keeps_backtrack_points():
    graph = BoundaryGraph((BoundaryCurve(0, ((0, 0), (2, 0), (1, 0)), 0, -2, False),), {})
    (fc,) = polyline_fit(graph)
    assert len(fc.segments) == 2


def test_apply_overrides():
    base = PRESETS["logo"]
    out = apply_overrides(base, {"segment.max_colors": 8, "export.layer_mode": "cutout"})
    assert out.segment.max_colors == 8 and out.export.layer_mode == "cutout"
    assert base.segment.max_colors == 16 and base.export.layer_mode == "stacked"
    bad_overrides = (
        {"segment.nope": 1},
        {"nogroup.x": 1},
        {"max_colors": 1},
        {"segment.max_colors": 0},
        {"export.layer_mode": "x"},
        {"segment.max_colors": "a"},
    )
    for bad in bad_overrides:
        with pytest.raises(InputError):
            apply_overrides(base, bad)


def test_override_reaches_vectorize():
    result = vectorize(to_png_bytes(ring_in_ring()), overrides={"export.layer_mode": "cutout"})
    assert result.doc.layer_mode == "cutout"
    assert re.search(r"fill-rule", to_svg(result.doc))


def _transparent_ring() -> np.ndarray:
    px = np.zeros((32, 32, 4), np.uint8)
    yy, xx = np.mgrid[0:32, 0:32]
    d2 = (xx - 15.5) ** 2 + (yy - 15.5) ** 2
    px[(d2 <= 12**2) & (d2 >= 6**2)] = (200, 30, 30, 255)
    return px


def test_stacked_keeps_transparent_hole():
    _, _, doc = _regions_graph(_transparent_ring(), "stacked")
    root = ET.fromstring(to_svg(doc))
    (ring,) = list(root)
    assert ring.tag.endswith("path")
    assert ring.attrib["d"].count("M") == 2
    assert ring.attrib["fill-rule"] == "evenodd"
    assert not doc.primitives


def test_stacked_transparent_hole_blocks_primitive():
    regions, graph, _ = _regions_graph(_transparent_ring())
    fitted = tuple(FittedCurve(f.curve_id, f.segments, Circle(0, 0, 1)) for f in polyline_fit(graph))
    doc = build_doc(regions, graph, fitted, ExportParams("stacked"))
    assert len(doc.paths) == 1 and not doc.primitives


def _nested_frame() -> np.ndarray:
    px = np.full((96, 96, 4), 255, np.uint8)
    px[60:92, 60:92] = (30, 60, 200, 255)
    px[64:88, 64:88] = (40, 170, 60, 255)
    for x in (68, 74, 80):
        px[64:88, x : x + 3] = (220, 30, 30, 255)
    return px


def test_nested_region_painted_after_container():
    regions, _, doc = _regions_graph(_nested_frame())
    z_by_color = {}
    for p in doc.paths:
        z_by_color[p.fill.rgb] = p.z_order
    assert z_by_color[(30, 60, 200)] < z_by_color[(40, 170, 60)] < z_by_color[(220, 30, 30)]


def test_empty_contours_skipped_in_svg():
    root = ET.fromstring(to_svg(_doc([Path(((),), RED, 0)])))
    assert len(root) == 0


def test_fit_params_validation():
    from engine.presets import FitParams

    FitParams()
    for bad in ({"tolerance_px": 0}, {"complexity_weight": -1}, {"axis_snap_deg": 45},
                {"smooth_sigma": -0.1}, {"corner_angle_deg": 0}, {"corner_angle_deg": 180},
                {"enabled": 1}, {"tolerance_px": "a"}):
        with pytest.raises(ValueError):
            FitParams(**bad)
    with pytest.raises(InputError):
        apply_overrides(PRESETS["logo"], {"fit.tolerance_px": -1})


def test_precision_must_be_int():
    for bad in (2.5, True, "2"):
        with pytest.raises(InputError):
            apply_overrides(PRESETS["logo"], {"export.precision": bad})
