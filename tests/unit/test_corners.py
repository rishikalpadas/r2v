import numpy as np
import pytest

from engine.geometry.corners import dense_samples, find_corners, smooth_dense, smooth_polyline
from engine.geometry.stitching import build_boundary_graph
from engine.presets import SegmentParams
from engine.stages.segment import segment
from engine.types import RasterImage, RegionMap
from eval.synthetic import circle


def _label_map(labels: np.ndarray) -> RegionMap:
    arr = np.asarray(labels, np.int32)
    k = int(arr.max()) + 1
    return RegionMap(
        labels=arr,
        palette_rgb=np.zeros((k, 3), np.uint8),
        palette_lab=np.zeros((k, 3)),
        adjacency=frozenset(),
    )


# Interior boundaries only: the image frame (right == OUTSIDE) is a true square, not a circle.
def _closed_curves(regions: RegionMap):
    return [c for c in build_boundary_graph(regions).curves if c.closed and c.left >= 0 and c.right >= 0]


def test_square_has_four_corners():
    labels = np.zeros((40, 40), np.int32)
    labels[10:30, 10:30] = 1
    (curve,) = _closed_curves(_label_map(labels))
    smooth = smooth_polyline(curve.points, closed=True)
    corners = find_corners(smooth, closed=True)
    assert len(corners) == 4
    truth = [(10, 10), (30, 10), (30, 30), (10, 30)]
    for t in truth:
        assert min(np.hypot(smooth[i][0] - t[0], smooth[i][1] - t[1]) for i in corners) <= 1.5


def test_circle_has_no_corners():
    h = w = 128
    px = circle(r=40)
    regions = segment(RasterImage(pixels=px, original_size=(w, h)), SegmentParams())
    curves = _closed_curves(regions)
    assert curves
    for c in curves:
        smooth = smooth_polyline(c.points, closed=True)
        assert find_corners(smooth, closed=True) == ()


def test_open_endpoints_fixed():
    pts = ((0.0, 0.0), (5.0, 0.0), (5.0, 4.0), (9.0, 4.0))
    out = smooth_polyline(pts, closed=False)
    assert out[0] == pts[0]
    assert out[-1] == pts[-1]
    assert len(out) > len(pts)


def test_closed_output_does_not_repeat_first_point():
    pts = ((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0), (0.0, 0.0))
    out = smooth_polyline(pts, closed=True)
    assert len(out) == 16
    assert out[0] != out[-1]


def test_staircase_line_no_corners():
    # Crack edges of a 1:3 slope: runs of 3 horizontal, 1 vertical.
    pts = [(0.0, 0.0)]
    x = y = 0.0
    for _ in range(10):
        x += 3.0
        pts.append((x, y))
        y += 1.0
        pts.append((x, y))
    smooth = smooth_polyline(tuple(pts), closed=False)
    assert find_corners(smooth, closed=False) == (0, len(smooth) - 1)


def test_open_curve_includes_endpoints_and_real_corner():
    pts = ((0.0, 0.0), (20.0, 0.0), (20.0, 20.0))
    smooth = smooth_polyline(pts, closed=False)
    corners = find_corners(smooth, closed=False)
    assert corners[0] == 0 and corners[-1] == len(smooth) - 1
    assert len(corners) == 3


def test_degenerate_inputs():
    assert smooth_polyline((), closed=False) == ()
    assert find_corners((), closed=True) == ()
    assert find_corners(((0.0, 0.0),), closed=False) == (0,)
    with pytest.raises(ValueError):
        smooth_polyline(((0.0, 0.0), (1.0, 0.0)), closed=False, sigma=-1.0)


def test_dense_samples_layout():
    open_pts = ((0.0, 0.0), (2.0, 0.0), (2.0, 1.0))
    assert dense_samples(open_pts, closed=False) == (
        (0.0, 0.0), (0.5, 0.0), (1.5, 0.0), (2.0, 0.5), (2.0, 1.0)
    )
    square = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0))
    assert dense_samples(square, closed=True) == ((0.5, 0.0), (1.0, 0.5), (0.5, 1.0), (0.0, 0.5))


@pytest.mark.parametrize("closed", [False, True])
def test_smooth_polyline_composes_dense_and_smooth(closed):
    pts = ((0.0, 0.0), (6.0, 0.0), (6.0, 3.0), (2.0, 5.0), (0.0, 0.0))
    for sigma in (0.0, 1.0, 2.5):
        expected = smooth_dense(dense_samples(pts, closed), closed, sigma)
        assert smooth_polyline(pts, closed, sigma) == expected


def test_smooth_dense_keeps_open_endpoints_and_rejects_negative_sigma():
    dense = ((0.0, 0.0), (0.5, 0.3), (1.5, -0.2), (2.0, 0.0))
    out = smooth_dense(dense, closed=False, sigma=1.0)
    assert out[0] == dense[0] and out[-1] == dense[-1]
    with pytest.raises(ValueError):
        smooth_dense(dense, closed=False, sigma=-0.5)
