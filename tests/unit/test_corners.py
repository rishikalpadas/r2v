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


# ---------------------------------------------------------------- scale-space corners


def _jittered_ring(points, sigma, seed):
    """Unit-spaced samples of a closed polyline with Gaussian jitter along the edge normal."""
    rng = np.random.default_rng(seed)
    dense = np.asarray(dense_samples(tuple(points), closed=True), float)
    tangent = np.roll(dense, -1, axis=0) - np.roll(dense, 1, axis=0)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    normal /= np.hypot(*normal.T)[:, None]
    return tuple(map(tuple, dense + rng.normal(0.0, sigma, len(dense))[:, None] * normal))


def _circle_ring(r=30.0, n=96):
    t = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return [(64 + r * np.cos(a), 64 + r * np.sin(a)) for a in t] + [(64 + r, 64.0)]


SQUARE = [(10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0), (10.0, 10.0)]


@pytest.mark.parametrize("seed", range(3))
def test_multiscale_drops_noise_corners_on_a_circle(seed):
    from engine.geometry.corners import find_corners_multiscale

    raw = _jittered_ring(_circle_ring(), 0.8, seed)
    assert find_corners(smooth_dense(raw, True, 1.0), True, 35.0, 2)  # fine scale alone is fooled
    found = find_corners_multiscale(raw, True, 35.0, fine_sigma=1.0, coarse_sigma=2.4)
    assert found.indices == ()


@pytest.mark.parametrize("seed", range(3))
def test_multiscale_keeps_real_corners_at_their_fine_position(seed):
    from engine.geometry.corners import find_corners_multiscale

    raw = _jittered_ring(SQUARE, 0.3, seed)
    found = find_corners_multiscale(raw, True, 35.0, fine_sigma=1.0, coarse_sigma=2.0)
    assert len(found.indices) == 4
    truth = [(10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)]
    for t in truth:
        # midpoint samples sit 0.5 px from the vertex; allow one sample of index slack plus jitter
        assert min(np.hypot(raw[i][0] - t[0], raw[i][1] - t[1]) for i in found.indices) <= 2.0
    assert all(a > 35.0 for a in found.coarse_angles)


def test_multiscale_keeps_both_corners_of_a_narrow_end():
    from engine.geometry.corners import find_corners_multiscale

    bar = [(10.0, 10.0), (40.0, 10.0), (40.0, 14.0), (10.0, 14.0), (10.0, 10.0)]
    raw = dense_samples(tuple(bar), closed=True)
    found = find_corners_multiscale(raw, True, 35.0, fine_sigma=1.0, coarse_sigma=2.0)
    assert len(found.indices) == 4


def test_multiscale_open_curve_keeps_its_end_points():
    from engine.geometry.corners import find_corners_multiscale

    raw = dense_samples(((0.0, 0.0), (20.0, 0.0), (20.0, 20.0)), closed=False)
    found = find_corners_multiscale(raw, False, 35.0, fine_sigma=1.0, coarse_sigma=2.0)
    assert found.indices[0] == 0 and found.indices[-1] == len(raw) - 1
    assert len(found.indices) == 3
    assert len(found.coarse_angles) == 3 and np.isnan(found.coarse_angles[0])


def test_smooth_between_pins_corners_and_keeps_straight_sides_straight():
    from engine.geometry.corners import smooth_between

    raw = [(float(x), 10.0) for x in range(10, 50)] + [(50.0, float(y)) for y in range(10, 50)]
    raw += [(float(x), 50.0) for x in range(50, 10, -1)] + [(10.0, float(y)) for y in range(50, 10, -1)]
    corners = [0, 40, 80, 120]  # the four vertices
    out = smooth_between(raw, corners, True, sigma=3.0)
    assert len(out) == len(raw)
    for c in corners:
        assert out[c] == raw[c]
    top = [out[i] for i in range(corners[0], corners[1] + 1)]
    assert all(y == pytest.approx(10.0) for _, y in top)  # nothing pulled round the corner


def test_smooth_between_without_corners_is_smooth_dense():
    from engine.geometry.corners import smooth_between

    raw = _jittered_ring(_circle_ring(), 0.5, 0)
    assert smooth_between(raw, [], True, 2.0) == smooth_dense(raw, True, 2.0)


def test_multiscale_keeps_a_fine_corner_too_near_an_open_end_for_the_coarse_scale():
    from engine.geometry.corners import find_corners_multiscale

    # A right angle 1 px before the end of an open curve: the coarse window does not fit there.
    raw = dense_samples(((0.0, 0.0), (30.0, 0.0), (30.0, 1.0)), closed=False)
    found = find_corners_multiscale(raw, False, 35.0, fine_sigma=1.0, coarse_sigma=2.0)
    assert len(found.indices) == 3
    corner = found.indices[1]
    assert abs(raw[corner][0] - 30.0) <= 1.0 and np.isnan(found.coarse_angles[1])


def test_multiscale_track_radius_is_configurable():
    from engine.geometry.corners import find_corners_multiscale

    # An L whose fine corner is 1 sample from the coarse response either way: tracking works
    # for both radii; the parameter only widens the search window.
    pts = [(float(x), 0.0) for x in range(30)] + [(30.0, float(y)) for y in range(30)]
    a = find_corners_multiscale(pts, False, 35.0, 1.0, 4.0, 2)
    b = find_corners_multiscale(pts, False, 35.0, 1.0, 4.0, 2, track_radius=8)
    assert len(a.indices) == len(b.indices) == 3
    assert abs(a.indices[1] - 30) <= 1 and abs(b.indices[1] - 30) <= 1
