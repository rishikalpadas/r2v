"""Anti-aliased synthetic images: box-filtered supersampling with known sub-pixel geometry."""
import numpy as np
import pytest

from eval.make_synthetic_dataset import make_dataset
from eval.synthetic import aa_disc, aa_half_plane, aa_shapes, flat_shapes, solid, supersample


def test_supersample_box_filters_each_block():
    def draw(n):
        px = solid(n, (0, 0, 0, 255))
        px[:, : n // 4] = (200, 100, 40, 255)  # hi-res column 0 only
        return px

    out = supersample(draw, 2, factor=2)
    assert out.shape == (2, 2, 4) and out.dtype == np.uint8
    # Output pixel 0 covers hi-res columns 0..1; only column 0 is colored -> 50% blend.
    np.testing.assert_array_equal(out[0, 0], (100, 50, 20, 255))
    np.testing.assert_array_equal(out[0, 1], (0, 0, 0, 255))


def test_supersample_averages_premultiplied_alpha():
    def draw(n):
        px = solid(n, (0, 0, 0, 0))
        px[:, : n // 2] = (30, 90, 200, 255)
        return px

    out = supersample(draw, 1, factor=4)
    # Half covered: color stays the shape color (not darkened), alpha is half.
    np.testing.assert_array_equal(out[0, 0], (30, 90, 200, 128))


@pytest.mark.parametrize("x0", [10.3, 10.7])
def test_aa_half_plane_vertical_edge_coverage(x0):
    px = aa_half_plane(32, x0=x0, angle_deg=0.0, left=(255, 255, 255, 255), right=(0, 0, 0, 255))
    row = px[16, :, 0].astype(float)
    coverage_right = 1.0 - row / 255.0  # black = right side
    assert np.allclose(coverage_right[:10], 0.0)
    assert np.allclose(coverage_right[11:], 1.0)
    assert abs(coverage_right[10] - (11 - x0)) <= 0.02


def test_aa_disc_area_matches_radius():
    px = aa_disc(fg=(0, 0, 0, 255), bg=(255, 255, 255, 255))
    coverage = 1.0 - px[..., 0].astype(float) / 255.0
    assert abs(coverage.sum() - np.pi * 40.4**2) <= 0.002 * np.pi * 40.4**2
    assert 0 < ((coverage > 0.02) & (coverage < 0.98)).sum()  # it is anti-aliased


def test_aa_shapes_is_box_filtered_flat_shapes():
    px = aa_shapes(3, size=32)
    hi = flat_shapes(3, size=256).astype(float)
    expected = hi.reshape(32, 8, 32, 8, 4).mean(axis=(1, 3))
    assert np.abs(px.astype(float) - expected).max() <= 0.5 + 1e-9


def test_dataset_writes_aa_logos(tmp_path):
    root = make_dataset(tmp_path / "ds", size=64)
    files = sorted(p.name for p in (root / "aa_logos").glob("*.png"))
    assert "aa_disc.png" in files and "aa_shapes_00.png" in files
    assert len(files) == 6
