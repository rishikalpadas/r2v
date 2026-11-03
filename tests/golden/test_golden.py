import xml.etree.ElementTree as ET

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from engine import to_svg, vectorize
from engine.pipeline import run_pipeline
from engine.presets import PRESETS, apply_overrides
from engine.types import RasterImage
from eval.metrics import alpha_agreement, count_nodes, mean_delta_e, render_svg, ssim
from eval.synthetic import circle, flat_shapes, solid, to_png_bytes, transparent_logo


def _roundtrip(pixels: np.ndarray, **kwargs):
    h, w = pixels.shape[:2]
    svg = to_svg(vectorize(to_png_bytes(pixels), **kwargs).doc)
    return svg, render_svg(svg, w, h)


@pytest.mark.parametrize("seed", range(10))
def test_flat_shapes_fidelity(seed):
    pixels = flat_shapes(seed)
    _, render = _roundtrip(pixels)
    assert ssim(pixels, render) >= 0.95
    assert mean_delta_e(pixels, render) <= 3.0


@pytest.mark.parametrize("seed", range(10))
def test_stacked_no_gaps(seed):
    _, render = _roundtrip(flat_shapes(seed))
    assert (render[..., 3] == 255).all()


def test_cutout_fidelity():
    pixels = flat_shapes(3)
    _, render = _roundtrip(pixels, overrides={"export.layer_mode": "cutout"})
    assert ssim(pixels, render) >= 0.95


def test_transparent_logo_alpha():
    pixels = transparent_logo()
    _, render = _roundtrip(pixels)
    assert alpha_agreement(pixels, render) >= 0.98


def _one_px_line() -> np.ndarray:
    px = solid(32, (255, 255, 255, 255))
    px[:, 16] = (0, 0, 0, 255)
    return px


DEGENERATE = {
    "1x1": solid(1, (10, 20, 30, 255)),
    "2x2": np.array([[[255, 0, 0, 255], [0, 0, 255, 255]]] * 2, np.uint8),
    "solid_opaque": solid(16, (40, 160, 90, 255)),
    "solid_transparent": solid(16, (0, 0, 0, 0)),
    "one_px_line": _one_px_line(),
}


@pytest.mark.parametrize("name", list(DEGENERATE))
def test_degenerate_inputs(name):
    svg, _ = _roundtrip(DEGENERATE[name])
    shapes = list(ET.fromstring(svg))
    if name == "solid_transparent":
        assert shapes == []
    else:
        assert shapes


_COLORS = [(220, 30, 30), (30, 160, 60), (30, 60, 200), (240, 200, 20), (20, 20, 20)]
_rect = st.tuples(
    st.integers(0, 52), st.integers(0, 52), st.integers(4, 40), st.integers(4, 40),
    st.integers(0, len(_COLORS) - 1),
)


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(_rect, min_size=1, max_size=5))
def test_property_random_rects(rects):
    pixels = solid(64, (255, 255, 255, 255))
    for x, y, w, h, c in rects:
        pixels[y : y + h, x : x + w] = (*_COLORS[c], 255)
    _, render = _roundtrip(pixels)
    assert ssim(pixels, render) >= 0.95


def test_transparent_ring_hole_stays_transparent():
    px = solid(32, (0, 0, 0, 0))
    yy, xx = np.mgrid[0:32, 0:32]
    d2 = (xx - 15.5) ** 2 + (yy - 15.5) ** 2
    px[(d2 <= 12**2) & (d2 >= 6**2)] = (200, 30, 30, 255)
    _, render = _roundtrip(px)
    assert alpha_agreement(px, render) >= 0.98
    assert render[16, 16, 3] == 0


def test_nested_regions_keep_colors():
    px = np.full((96, 96, 4), 255, np.uint8)
    px[60:92, 60:92] = (30, 60, 200, 255)
    px[64:88, 64:88] = (40, 170, 60, 255)
    for x in (68, 74, 80):
        px[64:88, x : x + 3] = (220, 30, 30, 255)
    _, render = _roundtrip(px)
    inner = (slice(64, 88), slice(64, 88))
    assert mean_delta_e(px[inner], render[inner]) <= 3.0
    assert tuple(render[85, 72, :3]) == (40, 170, 60)


def test_circle_emitted_as_primitive():
    svg = to_svg(vectorize(to_png_bytes(circle())).doc)
    circles = [el for el in ET.fromstring(svg) if el.tag.endswith("circle")]
    assert any(
        abs(float(c.get("cx")) - 64) <= 1.0
        and abs(float(c.get("cy")) - 64) <= 1.0
        and abs(float(c.get("r")) - 40) <= 1.0
        for c in circles
    )


def _mean_nodes(overrides=None) -> float:
    counts = [
        count_nodes(to_svg(vectorize(to_png_bytes(flat_shapes(seed)), overrides=overrides).doc))[0]
        for seed in range(10)
    ]
    return sum(counts) / len(counts)


def test_fit_reduces_nodes():
    assert _mean_nodes() <= 0.5 * _mean_nodes({"fit.enabled": False})


def test_regression_narrow_rects_stay_rects():
    # Hypothesis-found case: 4 px wide rects were fitted as a circle and an ellipse (SSIM 0.9496).
    pixels = solid(64, (255, 255, 255, 255))
    pixels[1:6, 1:5] = (*_COLORS[0], 255)
    pixels[12:51, 11:15] = (*_COLORS[1], 255)
    _, render = _roundtrip(pixels)
    assert ssim(pixels, render) >= 0.95


# ---------------------------------------------------------------- processing scale 4
# Measured at commit de88906: worst SSIM 0.9825 (flat_shapes_4), worst node ratio 1.24
# (flat_shapes_7, 25 -> 31). Scaling complexity_weight with scale (x scale, scale^2, scale^3)
# was measured too: it widens these margins (ratio <= 1.05) but cost fidelity on enhanced
# input (scale^3: degraded SSIM -0.004..-0.006), so the weight stays unscaled.

_SCALE = 4


def _scaled_pair(pixels: np.ndarray) -> tuple[str, str]:
    """(scale-1 SVG, SVG of the same image nearest-upscaled 4x and run at scale 4), enhance off."""
    h, w = pixels.shape[:2]
    preset = apply_overrides(PRESETS["logo"], {"enhance.enabled": False})
    base = to_svg(run_pipeline(RasterImage(pixels, (w, h)), preset).doc)
    up = np.repeat(np.repeat(pixels, _SCALE, axis=0), _SCALE, axis=1)
    scaled = to_svg(run_pipeline(RasterImage(up, (w, h), scale=float(_SCALE)), preset).doc)
    return base, scaled


@pytest.mark.parametrize(
    "pixels",
    [*(flat_shapes(seed) for seed in range(10)), circle()],
    ids=[*(f"flat_shapes_{seed}" for seed in range(10)), "circle"],
)
def test_scale4_matches_scale1(pixels):
    h, w = pixels.shape[:2]
    base, scaled = _scaled_pair(pixels)
    assert ssim(render_svg(base, w, h), render_svg(scaled, w, h)) >= 0.98
    assert count_nodes(scaled)[0] <= 1.3 * count_nodes(base)[0]


def test_scale4_circle_stays_primitive():
    _, scaled = _scaled_pair(circle())
    assert any(el.tag.endswith("circle") for el in ET.fromstring(scaled))
