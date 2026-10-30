import io
import time

import numpy as np
import pytest
from scipy import ndimage
from PIL import Image

from engine.presets import SegmentParams
from engine.stages.segment import _drop_blend_clusters, segment
from engine.types import TRANSPARENT, RasterImage
from eval.synthetic import flat_shapes, solid, transparent_logo


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
    counts = np.array([1000, 1000, 30, 10])
    cluster = np.array([0, 1, 2, 3])
    new, kept = _drop_blend_clusters(cluster, palette, counts)
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
