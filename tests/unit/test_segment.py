import io
import time

import numpy as np
import pytest
from scipy import ndimage
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab

from engine.presets import SegmentParams
from engine.stages.segment import (
    _components,
    _merge_small_components,
    segment,
)
from engine.stages.segment_palette import _complete_linkage_groups, _drop_blend_clusters
from engine.types import TRANSPARENT, RasterImage
from eval.synthetic import _circle_mask, flat_shapes, solid, supersample, transparent_logo


def _img(pixels: np.ndarray) -> RasterImage:
    h, w = pixels.shape[:2]
    return RasterImage(pixels=pixels, original_size=(w, h))


def test_exact_colors_recovered():
    px = flat_shapes(3, n_colors=4)
    params = SegmentParams()
    rm = segment(_img(px), params)
    assert rm.labels.shape == px.shape[:2]
    assert (rm.labels >= 0).all()
    wrong = (px[..., :3] != rm.palette_rgb[rm.labels]).any(axis=-1)
    # Only source specks smaller than min_region_px (merged into a neighbor by design) may differ.
    for color in np.unique(px[..., :3].reshape(-1, 3), axis=0):
        groups, n = ndimage.label((px[..., :3] == color).all(-1), structure=np.ones((3, 3)))
        sizes = np.bincount(groups.ravel())
        bad_groups = np.unique(groups[wrong & (groups > 0)])
        assert all(sizes[g] < params.min_region_px for g in bad_groups)
    # every region's palette color is a source color
    source = {tuple(c) for c in np.unique(px[..., :3].reshape(-1, 3), axis=0)}
    assert {tuple(c) for c in rm.palette_rgb} <= source
    assert wrong.sum() <= 8


def test_single_color_image():
    rm = segment(_img(solid(16, (10, 20, 30, 255))), SegmentParams())
    assert len(rm.palette_rgb) == 1
    assert tuple(rm.palette_rgb[0]) == (10, 20, 30)
    assert (rm.labels == 0).all()
    assert rm.adjacency == frozenset()


def test_fully_transparent():
    rm = segment(_img(solid(16, (0, 0, 0, 0))), SegmentParams())
    assert (rm.labels == TRANSPARENT).all()
    assert len(rm.palette_rgb) == 0
    assert rm.palette_lab.shape == (0, 3)


def test_transparent_background():
    px = transparent_logo()
    rm = segment(_img(px), SegmentParams())
    assert ((rm.labels == TRANSPARENT) == (px[..., 3] < 128)).all()
    assert len(rm.palette_rgb) == 1


@pytest.mark.parametrize("size", [1, 2])
def test_tiny_images(size):
    px = solid(size, (200, 100, 50, 255))
    rm = segment(_img(px), SegmentParams())
    assert rm.labels.shape == (size, size)
    assert rm.labels.dtype == np.int32
    assert rm.palette_rgb.shape[1] == 3 and rm.palette_lab.shape[1] == 3


def test_one_px_line_survives():
    px = solid(32, (255, 255, 255, 255))
    px[16, :, :3] = 0
    rm = segment(_img(px), SegmentParams())
    ids = np.unique(rm.labels[16, :])
    assert len(ids) == 1
    assert tuple(rm.palette_rgb[ids[0]]) == (0, 0, 0)
    assert (rm.labels == ids[0]).sum() == 32


def test_jpeg_noise_merged():
    px = flat_shapes(5, n_colors=3)
    buf = io.BytesIO()
    Image.fromarray(px[..., :3], "RGB").save(buf, format="JPEG", quality=85)
    rgb = np.array(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
    noisy = np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])
    rm = segment(_img(noisy), SegmentParams())
    assert len(np.unique(rm.palette_rgb, axis=0)) <= 4


def test_does_not_mutate_input():
    px = flat_shapes(7, n_colors=4)
    before = px.copy()
    segment(_img(px), SegmentParams())
    assert (px == before).all()


def test_adjacency_and_contiguous_ids():
    px = flat_shapes(3, n_colors=4)
    rm = segment(_img(px), SegmentParams())
    k = len(rm.palette_rgb)
    assert set(np.unique(rm.labels)) == set(range(k))
    assert all(a < b for a, b in rm.adjacency)
    h = rm.labels[:, :-1] != rm.labels[:, 1:]
    a, b = rm.labels[:, :-1][h], rm.labels[:, 1:][h]
    expected = {(min(x, y), max(x, y)) for x, y in zip(a, b)}
    v = rm.labels[:-1] != rm.labels[1:]
    a, b = rm.labels[:-1][v], rm.labels[1:][v]
    expected |= {(min(x, y), max(x, y)) for x, y in zip(a, b)}
    assert rm.adjacency == frozenset(expected)


def test_kmeans_path_deterministic():
    rng = np.random.default_rng(0)
    px = np.dstack([rng.integers(0, 256, (24, 24, 3), dtype=np.uint8),
                    np.full((24, 24), 255, np.uint8)])
    p = SegmentParams(max_colors=4)
    a, b = segment(_img(px), p), segment(_img(px), p)
    assert (a.labels == b.labels).all()
    assert len(np.unique(a.palette_rgb, axis=0)) <= 4


def test_drop_blend_clusters_never_yields_invalid_ids():
    palette = np.array([[0, 0, 0], [255, 255, 255], [120, 120, 120], [127, 127, 127]], np.uint8)
    labels = np.repeat(np.array([0, 1, 2, 3]), [1000, 1000, 30, 10]).reshape(1, -1)
    counts = np.bincount(labels.ravel())
    cluster = np.array([0, 1, 2, 3])
    new, kept = _drop_blend_clusters(cluster, palette, counts, labels)
    assert (new >= 0).all() and (new < len(kept)).all()


def _blend_drop_case() -> tuple[np.ndarray, np.ndarray]:
    """Black | white halves; a solid 20x20 dark-grey block (2%) and a 1-px light-grey ramp."""
    palette = np.array([[0, 0, 0], [255, 255, 255], [90, 90, 90], [160, 160, 160]], np.uint8)
    labels = np.zeros((100, 200), np.int32)
    labels[:, 101:] = 1
    labels[:, 100] = 3  # AA ramp along the black/white edge: no interior pixel
    labels[40:60, 40:60] = 2  # genuine minor colour between black and white
    return palette, labels


def test_blend_drop_keeps_solid_minor_colour():
    palette, labels = _blend_drop_case()
    counts = np.bincount(labels.ravel())
    assert counts[2] / counts.sum() == 0.02
    new, kept = _drop_blend_clusters(np.arange(4), palette, counts, labels)
    kept_set = {tuple(c) for c in kept}
    assert (90, 90, 90) in kept_set
    assert (160, 160, 160) not in kept_set
    assert len(kept) == 3
    assert (new >= 0).all() and (new < len(kept)).all()


def test_noisy_opaque_pixels_never_transparent():
    rng = np.random.default_rng(1)
    base = flat_shapes(2, n_colors=3)
    noise = rng.integers(-12, 13, base[..., :3].shape)
    px = base.copy()
    px[..., :3] = np.clip(base[..., :3].astype(int) + noise, 0, 255).astype(np.uint8)
    rm = segment(_img(px), SegmentParams())
    assert (rm.labels >= 0).all()


def test_large_noisy_image_is_fast():
    base = np.kron(flat_shapes(4, n_colors=5), np.ones((16, 16, 1), np.uint8))
    buf = io.BytesIO()
    Image.fromarray(base[..., :3], "RGB").save(buf, format="JPEG", quality=85)
    rgb = np.array(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
    px = np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])
    assert px.shape[:2] == (2048, 2048)
    t0 = time.perf_counter()
    rm = segment(_img(px), SegmentParams())
    elapsed = time.perf_counter() - t0
    print(f"segment 2048x2048 noisy: {elapsed:.2f}s")
    assert (rm.labels >= 0).all()
    assert elapsed < 20


@pytest.mark.parametrize(
    "kwargs",
    [{"max_colors": 0}, {"min_region_px": -1}, {"merge_delta_e": -1.0}, {"sample_px": 0}],
)
def test_params_validated(kwargs):
    with pytest.raises(ValueError):
        SegmentParams(**kwargs)


def test_speck_with_only_up_left_neighbours_is_merged():
    px = np.zeros((10, 10, 4), np.uint8)
    px[..., 3] = 255
    px[..., :3] = (255, 0, 0)
    px[:5, :, :3] = (0, 255, 0)
    px[9, 9, :3] = (0, 255, 0)  # 1 px green speck: neighbours only above and to the left
    rm = segment(_img(px), SegmentParams(min_region_px=4))
    assert rm.labels.max() + 1 == 2
    assert rm.labels[9, 9] == rm.labels[8, 9]


def test_small_component_joins_longest_border_neighbour_in_any_direction():
    lab = np.ones((10, 10), np.int32)
    lab[:, 5:] = 2
    lab[8:, :] = 0
    lab[3:5, 4] = 0  # 2 px speck: 4 edges to colour 1 (left/up/down), 2 edges to colour 2
    comp, comp_cluster = _components(lab)
    roots = _merge_small_components(comp, comp_cluster, 4)
    speck = comp[3, 4]
    assert comp_cluster[roots[speck]] == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_colors": 2.5},
        {"max_colors": True},
        {"min_region_px": 2.5},
        {"sample_px": 1.5},
        {"seed": -1},
        {"seed": 1.0},
        {"merge_delta_e": float("inf")},
        {"merge_delta_e": float("nan")},
        {"merge_delta_e": "8"},
        {"merge_delta_e": True},
    ],
)
def test_params_types_validated(kwargs):
    with pytest.raises(ValueError):
        SegmentParams(**kwargs)


# Near-white pinks like a real logo: white-light pink and pink-dark pink are each ~5.5-6 dE2000
# apart, close enough that a coarse threshold or transitive (AA-bridged) merging fuses them.
CHAIN_COLORS = {
    "white": (255, 255, 255),
    "light_pink": (249, 239, 242),
    "pink": (237, 197, 205),
    "dark_pink": (222, 174, 184),
}


def _chain_image(size: int = 128) -> np.ndarray:
    """Concentric anti-aliased discs: white > light pink > pink > dark pink (AA ramps between)."""
    rings = [(52.3, "light_pink"), (36.4, "pink"), (18.2, "dark_pink")]

    def draw(n: int) -> np.ndarray:
        k = n / size
        px = solid(n, (*CHAIN_COLORS["white"], 255))
        for r, name in rings:
            px[_circle_mask(n, 64.3 * k, 63.6 * k, r * k)] = (*CHAIN_COLORS[name], 255)
        return px

    return supersample(draw, size)


def _accent_image(size: int = 128) -> np.ndarray:
    """White page, pink disc with dark pink core, and a small solid light-pink block (~3%)."""

    def draw(n: int) -> np.ndarray:
        k = n / size
        px = solid(n, (*CHAIN_COLORS["white"], 255))
        px[_circle_mask(n, 70.3 * k, 70.6 * k, 40.2 * k)] = (*CHAIN_COLORS["pink"], 255)
        px[_circle_mask(n, 70.3 * k, 70.6 * k, 20.4 * k)] = (*CHAIN_COLORS["dark_pink"], 255)
        y0, x0, side = round(6.4 * k), round(8.6 * k), round(22 * k)
        px[y0:y0 + side, x0:x0 + side] = (*CHAIN_COLORS["light_pink"], 255)
        return px

    return supersample(draw, size)


def _de(a, b) -> float:
    lab = rgb2lab(np.array([a, b], np.float64).reshape(1, 2, 3) / 255.0)[0]
    return float(deltaE_ciede2000(lab[0], lab[1]))


def test_aa_bridged_near_colours_stay_distinct():
    px = _chain_image()
    assert len(np.unique(px[..., :3].reshape(-1, 3), axis=0)) > SegmentParams().max_colors
    rm = segment(_img(px), SegmentParams())
    palette = np.unique(rm.palette_rgb, axis=0)
    for name, rgb in CHAIN_COLORS.items():
        nearest = min(_de(rgb, p) for p in palette)
        assert nearest < 3, f"{name} {rgb} lost; palette {palette.tolist()}"


def test_minor_solid_accent_colour_survives_blend_drop():
    px = _accent_image()
    accent = (px[..., :3] == CHAIN_COLORS["light_pink"]).all(-1).mean()
    assert 0.02 < accent < 0.05  # minor colour, lying between white and pink
    rm = segment(_img(px), SegmentParams())
    palette = np.unique(rm.palette_rgb, axis=0)
    for name, rgb in CHAIN_COLORS.items():
        nearest = min(_de(rgb, p) for p in palette)
        assert nearest < 3, f"{name} {rgb} lost; palette {palette.tolist()}"


def test_complete_linkage_never_chains():
    # A-B 4, B-C 4, A-C 8 at threshold 5: single linkage would chain A..C into one group.
    dist = np.array([[0.0, 4.0, 8.0], [4.0, 0.0, 4.0], [8.0, 4.0, 0.0]])
    groups = _complete_linkage_groups(dist, 5.0)
    assert groups[0] != groups[2]
    assert len(np.unique(groups)) == 2


def test_complete_linkage_merges_tight_group_and_respects_strict_threshold():
    dist = np.array([[0.0, 1.0, 2.0, 5.0], [1.0, 0.0, 1.5, 9.0], [2.0, 1.5, 0.0, 9.0],
                     [5.0, 9.0, 9.0, 0.0]])
    groups = _complete_linkage_groups(dist, 5.0)
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]  # exactly at threshold: not merged


def _speck_image() -> tuple[np.ndarray, dict[str, tuple[int, int, int]]]:
    """White | black halves with specks, a fringe sliver, a 3x3 dot and thin lines.

    Every speck has a colour of its own, so the old "last of its colour" rule kept them all.
    """
    c = {
        "red": (230, 20, 20), "green": (20, 200, 40), "grey": (128, 128, 128),
        "blue": (20, 40, 220), "orange": (250, 150, 0), "purple": (140, 30, 160),
    }
    px = solid(64, (255, 255, 255, 255))
    px[:, 32:, :3] = 0
    px[5:7, 5:7, :3] = c["red"]  # 2x2 speck (4 px)
    px[50, 10:13, :3] = c["green"]  # 1x3 speck
    px[20:30, 31, :3] = c["grey"]  # 1-px fringe sliver along the white/black edge
    px[10:13, 18:21, :3] = c["blue"]  # solid 3x3 dot: has an interior pixel
    px[60, 12:52, :3] = c["orange"]  # 1-px wide, 40-px long line
    for i in range(30):  # 1-px diagonal staircase line: 4-connected pieces of 1 px
        px[14 + i, 34 + i, :3] = c["purple"]
    px[40:42, 45, :3] = 255  # 2-px white speck inside black (white has other components)
    px[41, 46, :3] = 255
    px[42, 46, :3] = 255
    return px, c


def test_fringe_specks_merged_but_dots_and_lines_survive():
    px, c = _speck_image()
    rm = segment(_img(px), SegmentParams())
    palette = {tuple(int(v) for v in p) for p in rm.palette_rgb}
    for gone in ("red", "green", "grey"):
        assert c[gone] not in palette, gone
    for kept in ("blue", "orange", "purple"):
        assert c[kept] in palette, kept
    colour = rm.palette_rgb[rm.labels]
    assert (colour[10:13, 18:21] == c["blue"]).all()
    assert (colour[60, 12:52] == c["orange"]).all()
    assert (colour[40:43, 45:47] == 0).all()  # 4-px white speck absorbed by black
    assert (colour[20:30, 31] == 255).all()  # sliver joins its longest border (white)


def test_pixels_diagonally_touching_a_solid_block_are_specks():
    px = solid(40, (255, 255, 255, 255))
    px[0, :, :3] = np.arange(40)[:, None] + 100  # grey ramp row: > max_colors, the noisy path
    px[10:30, 10:30, :3] = (200, 30, 90)  # solid block: its 8-connected reach has interior
    px[9, 9, :3] = (200, 30, 90)  # 1 px touching the top-left corner diagonally
    px[30:32, 30, :3] = (200, 30, 90)  # 2 px touching the bottom-right corner diagonally
    rm = segment(_img(px), SegmentParams())
    colour = rm.palette_rgb[rm.labels]
    assert (colour[9, 9] == 255).all()
    assert (colour[30:32, 30] == 255).all()
    assert (colour[10:30, 10:30] == (200, 30, 90)).all()


def test_exact_colour_tips_diagonally_touching_a_shape_survive():
    # Exact flat art (<= max_colors colors): a 1-px tip is part of the aliased shape, not noise.
    px = solid(40, (255, 255, 255, 255))
    px[10:30, 10:30, :3] = (200, 30, 90)
    px[9, 9, :3] = (200, 30, 90)
    rm = segment(_img(px), SegmentParams())
    assert tuple(rm.palette_rgb[rm.labels[9, 9]]) == (200, 30, 90)
