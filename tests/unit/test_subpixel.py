"""Sub-pixel edge placement from anti-aliased edge pixels."""
import math

import numpy as np
import pytest

from engine.geometry.corners import dense_samples
from engine.geometry.stitching import build_boundary_graph
from engine.geometry.subpixel import EdgeContext, make_edge_context, subpixel_dense
from engine.presets import SegmentParams
from engine.stages.segment import _to_lab, segment
from engine.types import OUTSIDE, TRANSPARENT, RasterImage, RegionMap
from eval.synthetic import aa_half_plane, flat_shapes, solid, supersample

WHITE = (255, 255, 255, 255)
BLUE = (30, 60, 200, 255)
CLEAR = (0, 0, 0, 0)


def _ideal_regions(pixels: np.ndarray, colors) -> RegionMap:
    """Label each pixel with its nearest color (alpha < 128 -> TRANSPARENT): a perfect segmenter."""
    opaque = [c for c in colors if c[3] == 255]
    rgb = pixels[..., :3].astype(float)
    dist = np.stack([np.linalg.norm(rgb - np.array(c[:3], float), axis=2) for c in opaque])
    labels = np.argmin(dist, axis=0).astype(np.int32)
    labels[pixels[..., 3] < 128] = TRANSPARENT
    palette = np.array([c[:3] for c in opaque], np.uint8)
    return RegionMap(labels, palette, _to_lab(palette), frozenset())


def _context(pixels: np.ndarray, colors) -> tuple[EdgeContext, RegionMap]:
    h, w = pixels.shape[:2]
    regions = _ideal_regions(pixels, colors)
    return make_edge_context(RasterImage(pixels, (w, h)), regions), regions


def _inner_curves(regions: RegionMap):
    return [c for c in build_boundary_graph(regions).curves if OUTSIDE not in (c.left, c.right)]


def _interior(curve, ctx) -> np.ndarray:
    """Sub-pixel samples without the exact junction endpoints of an open curve."""
    pts = np.asarray(subpixel_dense(curve, ctx))
    return pts if curve.closed else pts[1:-1]


# ---------------------------------------------------------------- straight edges


@pytest.mark.parametrize("x0", [10.3, 10.7])
@pytest.mark.parametrize("colors", [(WHITE, BLUE), (BLUE, WHITE)], ids=["white_left", "blue_left"])
def test_vertical_edge_samples_sit_on_true_edge(x0, colors):
    pixels = aa_half_plane(32, x0=x0, angle_deg=0.0, left=colors[0], right=colors[1])
    ctx, regions = _context(pixels, colors)
    (curve,) = _inner_curves(regions)
    pts = _interior(curve, ctx)
    assert len(pts) == 32
    assert np.abs(pts[:, 0] - x0).max() <= 0.05
    # The along-edge coordinate is untouched: samples stay at the unit-edge midpoints.
    assert np.allclose(np.sort(pts[:, 1]), np.arange(32) + 0.5)


@pytest.mark.parametrize("angle", [30.0, -30.0])
def test_slanted_edge_samples_sit_on_true_line(angle):
    size, x0 = 64, 30.4
    pixels = aa_half_plane(size, x0=x0, angle_deg=angle, left=WHITE, right=BLUE)
    ctx, regions = _context(pixels, (WHITE, BLUE))
    (curve,) = _inner_curves(regions)
    pts = _interior(curve, ctx)
    t = math.radians(angle)
    dist = (pts[:, 0] - x0) * math.cos(t) - (pts[:, 1] - size / 2) * math.sin(t)
    assert np.abs(dist).max() <= 0.1
    # Sanity: the crack-grid samples themselves are much further off.
    grid = np.asarray(dense_samples(curve.points, curve.closed))[1:-1]
    grid_dist = (grid[:, 0] - x0) * math.cos(t) - (grid[:, 1] - size / 2) * math.sin(t)
    assert np.abs(grid_dist).max() >= 0.4


# ---------------------------------------------------------------- thin strokes

LIGHT_A = (245, 235, 200, 255)
LIGHT_B = (200, 235, 245, 255)
DARK = (20, 20, 30, 255)


def _stroke(size: int, x_left: float, width: float) -> np.ndarray:
    def draw(n: int) -> np.ndarray:
        k = n / size
        xs = (np.arange(n) + 0.5) / k
        px = solid(n, LIGHT_A)
        px[:, xs >= x_left + width] = LIGHT_B
        px[:, (xs >= x_left) & (xs < x_left + width)] = DARK
        return px

    return supersample(draw, size, factor=32)


@pytest.mark.parametrize("x_left", [20.3, 20.5, 20.8])
def test_thin_stroke_edges_do_not_read_each_other(x_left):
    width = 2.0
    pixels = _stroke(48, x_left, width)
    ctx, regions = _context(pixels, (LIGHT_A, LIGHT_B, DARK))
    curves = sorted(_inner_curves(regions), key=lambda c: c.points[0][0])
    assert len(curves) == 2
    left, right = (_interior(c, ctx) for c in curves)
    by_row_left = left[np.argsort(left[:, 1]), 0]
    by_row_right = right[np.argsort(right[:, 1]), 0]
    assert (by_row_left < by_row_right).all()  # ordered: the two edges never cross
    measured = by_row_right - by_row_left
    assert np.abs(measured - width).max() <= 0.3
    assert np.abs(by_row_left - x_left).max() <= 0.1
    assert np.abs(by_row_right - (x_left + width)).max() <= 0.1


# ---------------------------------------------------------------- alpha edges


@pytest.mark.parametrize("colors", [(CLEAR, BLUE), (BLUE, CLEAR)], ids=["clear_left", "clear_right"])
def test_transparent_edge_uses_alpha(colors):
    x0 = 12.35
    pixels = aa_half_plane(32, x0=x0, angle_deg=0.0, left=colors[0], right=colors[1])
    ctx, regions = _context(pixels, colors)
    (curve,) = _inner_curves(regions)
    assert TRANSPARENT in (curve.left, curve.right)
    assert np.abs(_interior(curve, ctx)[:, 0] - x0).max() <= 0.05


# ---------------------------------------------------------------- no reliable blend -> no move


def _plain_dense(curve):
    return dense_samples(curve.points, curve.closed)


def test_non_antialiased_edges_have_zero_offset_exactly():
    pixels = flat_shapes(2)
    h, w = pixels.shape[:2]
    regions = segment(RasterImage(pixels, (w, h)), SegmentParams())
    ctx = make_edge_context(RasterImage(pixels, (w, h)), regions)
    for curve in build_boundary_graph(regions).curves:
        assert subpixel_dense(curve, ctx) == _plain_dense(curve)


def test_low_contrast_edge_is_not_moved():
    a, b = (200, 200, 200, 255), (206, 206, 206, 255)  # deltaE ~2: blends are unreadable
    pixels = aa_half_plane(32, x0=10.3, left=a, right=b)
    ctx, regions = _context(pixels, (a, b))
    (curve,) = _inner_curves(regions)
    assert subpixel_dense(curve, ctx) == _plain_dense(curve)


def test_pixel_that_is_not_a_blend_counts_as_pure():
    # A green column between white and blue is not a mix of them: it must not shift the edge.
    pixels = solid(16, WHITE)
    pixels[:, 8:] = BLUE
    pixels[:, 8] = (0, 200, 0, 255)
    h, w = pixels.shape[:2]
    labels = np.zeros((h, w), np.int32)
    labels[:, 8:] = 1
    palette = np.array([WHITE[:3], BLUE[:3]], np.uint8)
    regions = RegionMap(labels, palette, _to_lab(palette), frozenset())
    ctx = make_edge_context(RasterImage(pixels, (w, h)), regions)
    (curve,) = _inner_curves(regions)
    assert subpixel_dense(curve, ctx) == _plain_dense(curve)


def test_outside_curves_are_returned_unmoved():
    pixels = aa_half_plane(16, x0=6.3, left=WHITE, right=BLUE)
    ctx, regions = _context(pixels, (WHITE, BLUE))
    for curve in build_boundary_graph(regions).curves:
        if OUTSIDE in (curve.left, curve.right):
            assert subpixel_dense(curve, ctx) == _plain_dense(curve)


def test_make_edge_context_rejects_mismatched_shapes():
    pixels = solid(8, WHITE)
    palette = np.array([WHITE[:3]], np.uint8)
    regions = RegionMap(np.zeros((4, 4), np.int32), palette, _to_lab(palette), frozenset())
    with pytest.raises(ValueError):
        make_edge_context(RasterImage(pixels, (8, 8)), regions)


def test_sample_and_normal_counts_must_agree(monkeypatch):
    import engine.geometry.subpixel as sp

    pixels = aa_half_plane(16, x0=6.3, left=WHITE, right=BLUE)
    ctx, regions = _context(pixels, (WHITE, BLUE))
    (curve,) = _inner_curves(regions)
    real = sp._right_normals
    monkeypatch.setattr(sp, "_right_normals", lambda pts, closed: real(pts, closed)[:-1])
    with pytest.raises(ValueError, match="normals"):
        subpixel_dense(curve, ctx)
