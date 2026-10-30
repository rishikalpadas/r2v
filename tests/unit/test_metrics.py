import numpy as np
import pytest
from skimage.color import deltaE_ciede2000, rgb2lab

from eval.metrics import (
    alpha_agreement,
    count_nodes,
    edge_f1,
    mean_delta_e,
    render_svg,
    ssim,
)
from eval.synthetic import (
    circle,
    flat_shapes,
    ring_in_ring,
    solid,
    to_png_bytes,
    transparent_logo,
)


def test_ssim_identical_is_one():
    x = flat_shapes(3)
    assert ssim(x, x) == pytest.approx(1.0)


def test_delta_e_identical_is_zero():
    x = flat_shapes(3)
    assert mean_delta_e(x, x) == pytest.approx(0.0)


def test_edge_f1_identical_is_one():
    x = circle()
    assert edge_f1(x, x) == pytest.approx(1.0)


def test_edge_f1_both_empty_is_one():
    a = solid(32, (10, 20, 30, 255))
    b = solid(32, (200, 20, 30, 255))
    assert edge_f1(a, b) == 1.0


def test_edge_f1_disjoint_is_low():
    assert edge_f1(circle(cx=30, cy=30, r=15), circle(cx=100, cy=100, r=15)) < 0.1


def test_render_svg_rect():
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><rect width="10" height="10" fill="#ff0000"/></svg>'
    px = render_svg(svg, 10, 10)
    assert px.shape == (10, 10, 4)
    assert (px == np.array([255, 0, 0, 255], dtype=np.uint8)).all()


def test_count_nodes():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20">'
        '<path d="M0 0 L10 0 L5 8 Z"/><circle cx="5" cy="5" r="3"/></svg>'
    )
    assert count_nodes(svg) == (4, 2)


def test_count_nodes_polygon_points():
    svg = '<svg xmlns="http://www.w3.org/2000/svg"><polygon points="0,0 10,0 5,8"/></svg>'
    assert count_nodes(svg) == (3, 1)


def test_alpha_agreement():
    a = solid(4, (0, 0, 0, 255))
    b = solid(4, (0, 0, 0, 0))
    assert alpha_agreement(a, a) == 1.0
    assert alpha_agreement(a, b) == 0.0


def test_synthetic_deterministic():
    assert np.array_equal(flat_shapes(1), flat_shapes(1))
    px = flat_shapes(1, n_colors=4)
    assert px.shape == (128, 128, 4) and px.dtype == np.uint8
    assert len(np.unique(px.reshape(-1, 4), axis=0)) <= 4


def test_flat_shapes_palette_is_well_separated():
    for seed in range(5):
        colors = np.unique(flat_shapes(seed, n_colors=5).reshape(-1, 4), axis=0)[:, :3]
        lab = rgb2lab(colors.reshape(1, -1, 3) / 255.0).reshape(-1, 3)
        for i in range(len(lab)):
            for j in range(i + 1, len(lab)):
                d = deltaE_ciede2000(lab[i : i + 1], lab[j : j + 1])[0]
                assert d >= 20


def test_other_synthetics():
    c = circle()
    assert c.shape == (128, 128, 4) and c[64, 64, 3] == 255
    assert len(np.unique(ring_in_ring().reshape(-1, 4), axis=0)) == 4
    logo = transparent_logo()
    assert logo[0, 0, 3] == 0 and logo[64, 64, 3] == 255
    assert to_png_bytes(c).startswith(b"\x89PNG")
