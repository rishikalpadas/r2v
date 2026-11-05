"""Region-level cleanup of quantized input (engine.stages.segment_regions, quality fix Q2)."""
import io

import numpy as np
import pytest
from PIL import Image
from scipy import ndimage
from skimage.color import deltaE_ciede2000, rgb2lab

from engine.presets import SegmentParams
from engine.stages.segment import segment
from engine.stages.segment_regions import (
    absorb_slivers,
    absorb_specks,
    contrast_image,
    lab_image,
    merge_near_regions,
)
from engine.types import RasterImage

GROUND = (246, 214, 226)  # pale pink
NEAR = (232, 188, 206)  # dE2000 6.6 from GROUND: above the palette merge (5), a "near" shade


def _de(a, b) -> float:
    lab = rgb2lab(np.array([a, b], np.float64).reshape(1, 2, 3) / 255.0)[0]
    return float(deltaE_ciede2000(lab[0], lab[1]))


def _img(pixels: np.ndarray, scale: float = 1.0) -> RasterImage:
    h, w = pixels.shape[:2]
    return RasterImage(pixels=pixels, original_size=(round(w / scale), round(h / scale)),
                       scale=scale)


def _noisy_rgba(rgb: np.ndarray, amp: int = 3, quality: int | None = 90, seed: int = 0):
    rng = np.random.default_rng(seed)
    out = np.clip(np.rint(rgb) + rng.integers(-amp, amp + 1, rgb.shape), 0, 255).astype(np.uint8)
    if quality is not None:
        buf = io.BytesIO()
        Image.fromarray(out, "RGB").save(buf, format="JPEG", quality=quality)
        out = np.array(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
    return np.dstack([out, np.full(out.shape[:2], 255, np.uint8)])


def _blob_mask(size: int) -> np.ndarray:
    """An irregular blob of about 10% of the canvas (a banding/JPEG blotch shape)."""
    yy, xx = np.mgrid[0:size, 0:size]
    angle = np.arctan2(yy - size / 2, xx - size / 2)
    radius = size * (0.17 + 0.03 * np.sin(5 * angle) + 0.02 * np.cos(3 * angle))
    return np.hypot(yy - size / 2, xx - size / 2) < radius


def _connected(labels: np.ndarray) -> bool:
    return all(ndimage.label(labels == r)[1] == 1 for r in np.unique(labels[labels >= 0]))


def test_near_colours_are_distinct_shades():
    assert 5.0 < _de(GROUND, NEAR) < 9.0


def test_noisy_flat_fill_with_soft_blotch_is_one_region():
    """A flat fill split by a soft-edged blotch of a near shade (banding) becomes one region."""
    size = 160
    blend = ndimage.gaussian_filter(_blob_mask(size).astype(float), 4.0)[..., None]
    rgb = np.array(GROUND) * (1 - blend) + np.array(NEAR) * blend
    rm = segment(_img(_noisy_rgba(rgb)), SegmentParams())
    assert len(rm.palette_rgb) == 1
    assert _de(tuple(rm.palette_rgb[0]), GROUND) < 2.0


def test_merge_is_disabled_by_zero_delta_e():
    size = 160
    blend = ndimage.gaussian_filter(_blob_mask(size).astype(float), 4.0)[..., None]
    rgb = np.array(GROUND) * (1 - blend) + np.array(NEAR) * blend
    params = SegmentParams(region_merge_delta_e=0.0)
    assert len(segment(_img(_noisy_rgba(rgb)), params).palette_rgb) >= 2


def test_pale_on_pale_designed_edge_stays_two_regions():
    """Pale pink lettering-like shape with a crisp edge on a slightly different pink ground."""
    size, factor = 160, 4
    hi = np.empty((size * factor, size * factor, 3))
    hi[:] = GROUND
    yy, xx = np.mgrid[0:size * factor, 0:size * factor] / factor
    hi[(np.abs(xx - 80) < 50) & (np.abs(yy - 80) < 14)] = NEAR  # a bar, about 14% of the area
    rgb = hi.reshape(size, factor, size, factor, 3).mean((1, 3))
    rm = segment(_img(_noisy_rgba(rgb)), SegmentParams())
    palette = [tuple(int(v) for v in c) for c in rm.palette_rgb]
    assert len(palette) == 2
    assert min(_de(c, NEAR) for c in palette) < 3.0
    assert min(_de(c, GROUND) for c in palette) < 3.0


def test_merge_uses_area_weighted_lab_mean_and_keeps_other_colours():
    labels = np.zeros((20, 20), np.int32)
    labels[5:9, 5:9] = 1  # 16 px blotch
    labels[:, 15:] = 2  # an unrelated dark region
    palette = np.array([GROUND, NEAR, (30, 30, 30)], np.uint8)
    contrast = np.zeros(labels.shape, np.float32)
    out, pal = merge_near_regions(labels, palette, contrast)
    assert len(pal) == 2 and _connected(out)
    lab = rgb2lab(palette[:2].reshape(1, 2, 3) / 255.0)[0]
    area = np.array([(labels == 0).sum(), 16.0])
    expected = (lab * area[:, None]).sum(0) / area.sum()
    merged = rgb2lab(pal[out[0, 0]].reshape(1, 1, 3) / 255.0)[0, 0]
    assert np.abs(merged - expected).max() < 1.0
    assert tuple(pal[out[0, 19]]) == (30, 30, 30)


def test_merge_blocked_by_edge_and_by_colour_shift():
    labels = np.zeros((20, 20), np.int32)
    labels[5:9, 5:9] = 1
    palette = np.array([GROUND, NEAR], np.uint8)
    loud = np.full(labels.shape, 5.0, np.float32)  # every boundary edge is a real edge
    assert len(merge_near_regions(labels, palette, loud)[1]) == 2
    halves = np.zeros((20, 20), np.int32)
    halves[:, 10:] = 1  # equal areas: each would move by half their distance
    quiet = np.zeros(labels.shape, np.float32)
    assert len(merge_near_regions(halves, palette, quiet)[1]) == 2


def test_equal_halves_merge_when_the_source_shows_one_flat_fill():
    """Noise-split halves: the palette colours differ, but the smoothed source means agree."""
    halves = np.zeros((20, 20), np.int32)
    halves[:, 10:] = 1
    palette = np.array([GROUND, NEAR], np.uint8)
    quiet = np.zeros(halves.shape, np.float32)
    fill = rgb2lab(np.array(GROUND, np.float64).reshape(1, 1, 3) / 255.0)[0, 0]
    same = np.array([fill * 200, fill * 200])  # sums over 200 px each
    out, pal = merge_near_regions(halves, palette, quiet, smooth_sums=same)
    assert len(pal) == 1 and (out == 0).all()
    lab = rgb2lab(palette.reshape(1, 2, 3) / 255.0)[0]
    banded = lab * 200  # the source really has two bands: they stay
    assert len(merge_near_regions(halves, palette, quiet, smooth_sums=banded)[1]) == 2


def test_merge_never_chains_along_a_gradient():
    """Three bands, each near its neighbour but the ends far apart: complete linkage stops."""
    labels = np.zeros((30, 60), np.int32)
    labels[:, 28:30] = 1
    labels[:, 30:] = 2
    palette = np.array([(252, 232, 240), (243, 212, 224), (232, 190, 208)], np.uint8)
    assert _de(palette[0], palette[2]) > 9.0
    out, pal = merge_near_regions(labels, palette, np.zeros(labels.shape, np.float32),
                                  max_shift=100.0)
    assert len(pal) == 2


def _three_columns(middle_rgb, width: int = 1):
    labels = np.zeros((40, 41), np.int32)
    labels[:, 20:20 + width] = 1
    labels[:, 20 + width:] = 2
    palette = np.array([(250, 250, 250), middle_rgb, (40, 40, 160)], np.uint8)
    rgb = palette[labels].astype(np.float64)
    return labels, palette, lab_image(rgb.astype(np.uint8))


@pytest.mark.parametrize("width", [1, 2])  # 3 px has half width 2 > 1.5 at scale 1
def test_anti_aliasing_sliver_is_absorbed(width):
    labels, palette, _ = _three_columns((145, 145, 205), width)  # halfway white -> blue
    rgb = palette[labels].copy()
    if width == 2:  # a 2 px AA ribbon is a ramp, not a flat stripe
        rgb[:, 20], rgb[:, 21] = (195, 195, 225), (95, 95, 185)
    out, pal = absorb_slivers(labels, palette, lab_image(rgb), scale=1.0)
    assert len(pal) == 2 and _connected(out)
    assert {tuple(c) for c in pal} == {(250, 250, 250), (40, 40, 160)}


def test_sliver_split_follows_the_blend():
    """A two-pixel ribbon: the near-white column joins white, the near-blue one blue."""
    labels, palette, _ = _three_columns((145, 145, 205), 2)
    rgb = palette[labels].copy()
    rgb[:, 20] = (205, 205, 230)
    rgb[:, 21] = (90, 90, 180)
    out, pal = absorb_slivers(labels, palette, lab_image(rgb), scale=1.0)
    assert (out[:, 20] == out[0, 0]).all() and (out[:, 21] == out[0, 40]).all()


def test_dark_one_px_outline_stays():
    labels, palette, lab = _three_columns((20, 20, 20))
    palette[2] = (246, 200, 220)  # pink on the other side: the line is darker than both
    out, pal = absorb_slivers(labels, palette, lab, scale=1.0)
    assert len(pal) == 3


def test_wide_middle_band_is_not_a_sliver():
    labels, palette, lab = _three_columns((145, 145, 205), 5)  # 5 px: half width 3 > 1.5
    assert len(absorb_slivers(labels, palette, lab, scale=1.0)[1]) == 3


def test_sliver_width_is_in_original_px():
    labels, palette, lab = _three_columns((145, 145, 205), 5)
    assert len(absorb_slivers(labels, palette, lab, scale=4.0)[1]) == 2


def _dot(colour, side: int = 3):
    labels = np.zeros((20, 20), np.int32)
    labels[8:8 + side, 8:8 + side] = 1
    return labels, np.array([(250, 250, 250), colour], np.uint8)


def test_strong_contrast_dot_stays():
    labels, palette = _dot((20, 20, 20))  # 9 px < 12 px^2, but black on white (an i-dot)
    assert len(absorb_specks(labels, palette, scale=1.0)[1]) == 2


def test_low_contrast_speck_is_merged():
    labels, palette = _dot((240, 236, 238))  # dE ~3 from the ground
    out, pal = absorb_specks(labels, palette, scale=1.0)
    assert len(pal) == 1 and tuple(pal[0]) == (250, 250, 250) and (out == 0).all()


def test_speck_joins_most_similar_neighbour():
    labels = np.zeros((20, 20), np.int32)
    labels[:, 10:] = 1
    labels[9:11, 9:11] = 2  # 4 px straddling both
    palette = np.array([(250, 250, 250), (200, 40, 40), (210, 60, 60)], np.uint8)
    out, pal = absorb_specks(labels, palette, scale=1.0)
    assert len(pal) == 2 and out[9, 9] == out[0, 19]


def test_speck_area_is_in_original_px():
    labels, palette = _dot((240, 236, 238), side=6)  # 36 px = 2.25 original px^2 at scale 4
    assert len(absorb_specks(labels, palette, scale=4.0)[1]) == 1
    assert len(absorb_specks(labels, palette, scale=1.0)[1]) == 2


def test_contrast_is_per_original_px():
    step = np.zeros((40, 80, 3), np.uint8)
    step[:, 40:] = 100
    c1 = contrast_image(lab_image(step), 1.0).max()
    wide = np.repeat(np.repeat(step, 4, 0), 4, 1)
    c4 = contrast_image(lab_image(wide), 4.0).max()
    assert abs(c1 - c4) / c1 < 0.25  # same units; a 1 px sharp step is sampled coarsely at 1x


def test_exact_colour_input_is_untouched():
    """Exact-colour input (<= max_colors colours) keeps its exact segmentation: a 3x3 speck of a
    near colour is a real source detail there."""
    px = np.zeros((20, 20, 4), np.uint8)
    px[..., :3] = GROUND
    px[..., 3] = 255
    px[8:11, 8:11, :3] = NEAR
    rm = segment(_img(px), SegmentParams())
    assert len(rm.palette_rgb) == 2


@pytest.mark.parametrize("kwargs", [
    {"region_merge_delta_e": -1.0}, {"speck_area_px": -1.0},
    {"region_merge_delta_e": float("nan")}, {"speck_area_px": "12"},
])
def test_new_params_validated(kwargs):
    with pytest.raises(ValueError):
        SegmentParams(**kwargs)


# ---------------------------------------------------------------- review fixes (dd1872a)


def _lab(rgb) -> np.ndarray:
    return rgb2lab(np.array(rgb, np.float64).reshape(1, 1, 3) / 255.0)[0, 0]


def test_merge_after_repush_with_higher_keep_id():
    """Strips 0 | 1 | 2: 1 joins the large 2 first (cost 0); 0 then needs the re-pushed pair
    (keep=2, nb=0) whose version stamps used to be stored swapped and dropped as stale."""
    labels = np.zeros((20, 20), np.int32)
    labels[:, 1:2] = 1
    labels[:, 2:] = 2
    palette = np.array([NEAR, GROUND, GROUND], np.uint8)
    quiet = np.zeros(labels.shape, np.float32)
    out, pal = merge_near_regions(labels, palette, quiet)
    assert len(pal) == 1 and (out == 0).all()


def test_merge_drawn_colour_stays_near_each_regions_source():
    """Two large regions whose palette colours differ (dE 6.6) but whose source means agree:
    they merge, and the drawn colour comes from the source, not from the palette mean."""
    labels = np.zeros((30, 30), np.int32)
    labels[:, 16:] = 1  # areas 480 / 420
    palette = np.array([GROUND, NEAR], np.uint8)
    src = _lab(GROUND)
    sums = np.array([src * 480, src * 420])
    out, pal = merge_near_regions(labels, palette, np.zeros(labels.shape, np.float32),
                                  smooth_sums=sums)
    assert len(pal) == 1
    assert _de(tuple(pal[0]), GROUND) < 2.0


def test_large_region_does_not_merge_into_darker_large_neighbour():
    """The smaller of two large regions would be drawn far from its own source colour."""
    labels = np.zeros((30, 30), np.int32)
    labels[:, 18:] = 1  # areas 540 / 360
    palette = np.array([NEAR, GROUND], np.uint8)
    sums = np.array([_lab(NEAR) * 540, _lab(GROUND) * 360])
    out, pal = merge_near_regions(labels, palette, np.zeros(labels.shape, np.float32),
                                  smooth_sums=sums)
    if len(pal) == 1:  # if it merged, every region must still be drawn close to its source
        assert _de(tuple(pal[0]), GROUND) < 2.0 and _de(tuple(pal[0]), NEAR) < 2.0
    else:
        assert {tuple(c) for c in pal} == {GROUND, NEAR}


def _stripe_scene(stripe_rgb, width: int, left=(0, 0, 0), right=(255, 255, 255)):
    labels = np.zeros((40, 41), np.int32)
    labels[:, 20:20 + width] = 1
    labels[:, 20 + width:] = 2
    palette = np.array([left, stripe_rgb, right], np.uint8)
    return labels, palette, lab_image(palette[labels])


def test_flat_mid_grey_pinstripe_between_black_and_white_stays():
    labels, palette, lab = _stripe_scene((128, 128, 128), 2)
    assert len(absorb_slivers(labels, palette, lab, scale=1.0)[1]) == 3


def test_rim_darker_than_its_fill_next_to_white_stays():
    fill = (238, 143, 203)
    rim = (214, 115, 180)  # 7.7 dE2000 darker than the fill: the e (11) petal rim
    assert 6.5 < _de(fill, rim) < 9.0
    labels, palette, lab = _stripe_scene(rim, 1, left=fill, right=(253, 253, 253))
    assert len(absorb_slivers(labels, palette, lab, scale=1.0)[1]) == 3


def test_ringing_ribbon_slightly_beyond_the_darker_side_is_absorbed():
    fill = (238, 143, 203)
    ribbon = (236, 140, 201)  # ~1 dE beyond the fill, away from white: JPEG/SR ringing
    labels, palette, lab = _stripe_scene(ribbon, 1, left=fill, right=(253, 253, 253))
    assert len(absorb_slivers(labels, palette, lab, scale=1.0)[1]) == 2


def test_pale_on_pale_edge_is_two_regions_before_the_merge_too():
    """Control: the bar scene is two regions with the merge disabled, so the pale-on-pale test
    exercises the merge refusing, not the clustering."""
    size, factor = 160, 4
    hi = np.empty((size * factor, size * factor, 3))
    hi[:] = GROUND
    yy, xx = np.mgrid[0:size * factor, 0:size * factor] / factor
    hi[(np.abs(xx - 80) < 50) & (np.abs(yy - 80) < 14)] = NEAR
    rgb = hi.reshape(size, factor, size, factor, 3).mean((1, 3))
    rm = segment(_img(_noisy_rgba(rgb)), SegmentParams(region_merge_delta_e=0.0))
    assert len(rm.palette_rgb) == 2


def test_sliver_that_cannot_be_split_cleanly_is_kept():
    """A sliver pixel whose colour says A but which only touches B and another sliver piece
    that goes to B: the A piece is cut off from A, flips to B... unless B is also unreachable.
    Here a 1 px ribbon isolates a pixel enclosed by a third region: the split returns None."""
    from engine.stages.segment_regions import _split_sliver
    window = np.full((5, 7), 2, np.int32)  # region 2 everywhere (a third colour)
    window[:, 0] = 0  # A on the left edge
    window[:, 6] = 3  # B on the right edge
    window[2, 3] = 1  # a one-pixel sliver touching neither A nor B
    lab = np.zeros((5, 7, 3), np.float32)
    assert _split_sliver(window, lab, 1, 0, 3, np.zeros(3), np.ones(3) * 50) is None


def test_segment_is_deterministic():
    size = 160
    blend = ndimage.gaussian_filter(_blob_mask(size).astype(float), 4.0)[..., None]
    rgb = np.array(GROUND) * (1 - blend) + np.array(NEAR) * blend
    px = _noisy_rgba(rgb)
    a, b = segment(_img(px), SegmentParams()), segment(_img(px.copy()), SegmentParams())
    assert np.array_equal(a.labels, b.labels) and np.array_equal(a.palette_rgb, b.palette_rgb)


def test_prune_spurs_orthogonal_and_peninsula():
    from engine.stages.segment_regions import prune_spurs
    grid = np.zeros((10, 10), np.int32)
    # 90-degree square corner at [1:5, 1:5]
    grid[1:5, 1:5] = 1
    # 1-pixel spur sticking out from square at (5, 2)
    grid[5, 2] = 1
    # 2-pixel wide terminal peninsula at (5, 6) and (5, 7) attached to [1:5, 6:8]
    grid[1:5, 6:8] = 1
    grid[5, 6] = 1
    grid[5, 7] = 1

    cleaned = prune_spurs(grid, max_iterations=4)
    # The 90-degree square corner [1:5, 1:5] must be fully preserved
    assert (cleaned[1:5, 1:5] == 1).all()
    # The 1-pixel spur at (5, 2) must be pruned to 0
    assert cleaned[5, 2] == 0
    # The 2-pixel peninsula at (5, 6) and (5, 7) must be pruned to 0
    assert cleaned[5, 6] == 0
    assert cleaned[5, 7] == 0


def test_absorb_specks_compact_vs_sliver():
    # 1. Compact dot (3x3 square, area 9, aspect 1.0) with high delta-e to background stays
    labels = np.zeros((20, 20), np.int32)
    labels[8:11, 8:11] = 1
    palette = np.array([[0, 0, 0], [255, 255, 255]], np.uint8)
    out, pal = absorb_specks(labels, palette, scale=1.0, min_area_px=12.0, keep_delta_e=15.0)
    assert len(pal) == 2
    assert (out == 1).sum() == 9

    # 2. Elongated thin sliver (1x16, area 16, aspect 16.0) with high delta-e is absorbed
    labels_sliver = np.zeros((30, 30), np.int32)
    labels_sliver[5:21, 10] = 1
    out_sl, pal_sl = absorb_specks(labels_sliver, palette, scale=1.0, min_area_px=12.0, keep_delta_e=15.0)
    assert len(pal_sl) == 1
    assert (out_sl == 0).all()
