import math

import numpy as np
import pytest
from PIL import Image, ImageDraw

from engine.geometry.corners import find_corners, smooth_polyline
from engine.geometry.primitives import detect_primitive
from engine.geometry.stitching import build_boundary_graph
from engine.stages.export import to_svg
from engine.types import Circle, Ellipse, Polygon, Primitive, Rect, RegionMap, SolidFill, VectorDoc
from eval.metrics import render_svg
from eval.synthetic import circle

SIZE = 128
IOU_MIN = 0.95


def _regions(mask: np.ndarray) -> RegionMap:
    return RegionMap(
        labels=mask.astype(np.int32),
        palette_rgb=np.zeros((2, 3), np.uint8),
        palette_lab=np.zeros((2, 3)),
        adjacency=frozenset(),
    )


def _detect(mask: np.ndarray):
    curves = [
        c for c in build_boundary_graph(_regions(mask)).curves
        if c.closed and c.left >= 0 and c.right >= 0
    ]
    assert len(curves) == 1
    smooth = smooth_polyline(curves[0].points, closed=True)
    return detect_primitive(smooth, find_corners(smooth, closed=True))


def _draw(fn) -> np.ndarray:
    img = Image.new("L", (SIZE, SIZE), 0)
    fn(ImageDraw.Draw(img))
    return np.asarray(img) > 0


def _iou_with_render(shape, mask: np.ndarray) -> float:
    doc = VectorDoc(SIZE, SIZE, (), (Primitive(shape, SolidFill((255, 0, 0)), 0),), "stacked")
    rendered = render_svg(to_svg(doc), SIZE, SIZE)[..., 3] > 127
    return (rendered & mask).sum() / (rendered | mask).sum()


def _rotated_rect_polygon(x, y, w, h, deg):
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return [(x + px * c - py * s, y + px * s + py * c) for px, py in [(0, 0), (w, 0), (w, h), (0, h)]]


def test_detects_circle():
    mask = circle(r=40)[..., 1] < 128
    shape = _detect(mask)
    assert isinstance(shape, Circle)
    assert abs(shape.cx - 64) <= 1.0 and abs(shape.cy - 64) <= 1.0 and abs(shape.r - 40) <= 1.0
    assert _iou_with_render(shape, mask) >= IOU_MIN


@pytest.mark.parametrize("deg", [0, 30, -20])
def test_detects_rect_and_rotated_rect(deg):
    quad = _rotated_rect_polygon(40, 50, 60, 30, deg) if deg else None
    if deg:
        mask = _draw(lambda d: d.polygon(quad, fill=255))
    else:
        mask = _draw(lambda d: d.rectangle([40, 50, 99, 79], fill=255))
    shape = _detect(mask)
    assert isinstance(shape, Rect)
    if deg == 0:
        assert shape.angle_deg == 0
        assert abs(shape.w - 60) <= 1.0 and abs(shape.h - 30) <= 1.0
    assert _iou_with_render(shape, mask) >= IOU_MIN


@pytest.mark.parametrize("deg", [0, 35])
def test_detects_ellipse(deg):
    def draw(d):
        cx, cy, rx, ry, t = 64, 64, 40, 20, math.radians(deg)
        ang = np.linspace(0, 2 * np.pi, 720)
        pts = [(cx + rx * math.cos(a) * math.cos(t) - ry * math.sin(a) * math.sin(t),
                cy + rx * math.cos(a) * math.sin(t) + ry * math.sin(a) * math.cos(t)) for a in ang]
        d.polygon(pts, fill=255)

    mask = _draw(draw)
    shape = _detect(mask)
    assert isinstance(shape, Ellipse)
    assert abs(max(shape.rx, shape.ry) - 40) <= 1.0 and abs(min(shape.rx, shape.ry) - 20) <= 1.0
    assert _iou_with_render(shape, mask) >= IOU_MIN


def test_detects_triangle_polygon():
    tri = [(20, 100), (100, 90), (50, 20)]
    mask = _draw(lambda d: d.polygon(tri, fill=255))
    shape = _detect(mask)
    assert isinstance(shape, Polygon)
    assert len(shape.points) == 3
    for t in tri:
        assert min(math.hypot(p[0] - t[0], p[1] - t[1]) for p in shape.points) <= 1.5
    assert _iou_with_render(shape, mask) >= IOU_MIN


def test_blob_returns_none():
    def draw(d):
        ang = np.linspace(0, 2 * np.pi, 720)
        r = 35 + 8 * np.sin(3 * ang) + 5 * np.cos(5 * ang + 1)
        d.polygon([(64 + ri * math.cos(a), 64 + ri * math.sin(a)) for ri, a in zip(r, ang)], fill=255)

    assert _detect(_draw(draw)) is None


def test_degenerate_inputs_return_none():
    assert detect_primitive((), ()) is None
    assert detect_primitive(((0.0, 0.0), (1.0, 0.0)), ()) is None
