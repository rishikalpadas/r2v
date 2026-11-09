"""Segment regions: clean up the region map of quantized (noisy) input.

Colour clustering splits one visually flat fill into near-identical shades (JPEG noise, gradient
banding), leaves anti-aliasing ribbons along edges as thin regions of their own, and leaves
low-contrast specks. Their boundaries are arbitrary and ragged, and every one becomes a curve the
fit stage has to draw. Three passes on the region map fix that:

1. `merge_near_regions`: adjacent regions whose colours are perceptually near-identical merge,
   least distinct pair first, unless the source shows a real edge between them.
2. `absorb_slivers`: regions that are thin everywhere and whose colour lies between their two
   main neighbours (an anti-aliasing ribbon) are split between those neighbours.
3. `absorb_specks`: regions below a minimum area merge into their most similar neighbour unless
   they contrast strongly with every neighbour (dots, highlights).

Regions stay 4-connected throughout (trace needs that).
"""
from __future__ import annotations

import heapq
from collections import defaultdict

import numpy as np
from scipy import ndimage
from skimage.color import deltaE_ciede2000, lab2rgb, rgb2lab

from engine.types import TRANSPARENT

# Boundary contrast = mean source gradient magnitude (CIE76 Lab units per original px) on the
# pixels either side of a shared boundary, after Gaussian smoothing at this sigma (original px).
# One original px averages away JPEG/SR noise but keeps a step edge's peak: a step of dE76 d reads
# about 0.4 d, while banding of a smooth gradient reads its slope (measured 1-3 on the logos).
CONTRAST_SIGMA_PX = 1.0
# A boundary edge is "quiet" when the contrast there is below QUIET_GRADIENT. Measured: a designed
# step of dE2000 4 (JPEG q85, upscaled) reads 1.6, dE2000 2 reads 0.8-1.0; flat JPEG/SR noise
# 0.1-0.3; band boundaries of a banded gradient and noise splits on the logos have medians 0.1-0.5.
QUIET_GRADIENT = 1.0
# Near-identical colour merge: complete-linkage dE2000 bound over the members' original colours
# (no chaining along a gradient). Measured on the logos: the split shades of one flat fill are
# dE2000 5.3-8.5 apart (clustering already merges below 5).
MERGE_DELTA_E = 9.0
# ...and at least this share of their shared boundary must be quiet. Measured per boundary on
# the logos: banding/noise splits 0.5-1.0, designed edges and pieces hugging an outline 0-0.3.
MERGE_MIN_QUIET_SHARE = 0.5
# ...and the merge may add at most this much colour error, dE2000 averaged over the pixels of
# both regions: each region's mean smoothed-source colour (noise averages out) against the merged
# one, which is also the colour drawn. A blotch inside a fill and the noise-split halves of one
# fill merge; broad bands of a designed gradient (e.g. the e (13) banner) and a large region
# beside a darker large one stay apart. Measured on the logos: see the Q2 rulings note.
MERGE_MAX_SHIFT = 1.6
# Anti-aliasing slivers: largest inscribed distance (distance transform, original px) of a
# region that is thin everywhere; 1.5 holds a ribbon up to 3 original px wide.
SLIVER_MAX_HALF_WIDTH_PX = 1.5
# "Between" test on the sliver's mean source colour S against its two main neighbours A, B (Lab):
# S projected on the line A-B may lie beyond the nearer end by at most SLIVER_MAX_OVERSHOOT (JPEG/SR
# ringing undershoots by ~1-3; the e (11) petal rims, 7.7 dE2000 darker than their fill, sit 5-8
# beyond and are designed), and off the line by at most SLIVER_MAX_OFFSET x |AB| (absorbed
# ribbons on the logos measured <= 0.18; a dark rim beside two light colours is far off it).
SLIVER_MAX_OVERSHOOT = 3.0
SLIVER_MAX_OFFSET = 0.25
# A designed pinstripe is flat across its width while an AA ribbon is a ramp: a sliver at least
# SLIVER_FLAT_MIN_WIDTH_PX wide (original px, mean width = 2 area / perimeter) whose pixels lie
# within SLIVER_FLAT_DE (Lab) of its median colour for SLIVER_FLAT_SHARE of them stays. A 1 px
# ribbon cannot show a ramp, so it is judged on colour alone.
SLIVER_FLAT_MIN_WIDTH_PX = 1.5
SLIVER_FLAT_DE = 3.0
SLIVER_FLAT_SHARE = 0.5
# Leftover specks: regions under this area (original px^2) merge into their most similar
# neighbour unless their dE2000 to every neighbour is at least SPECK_KEEP_DELTA_E (real dots).
SPECK_AREA_PX = 12.0
SPECK_KEEP_DELTA_E = 15.0


def lab_image(rgb: np.ndarray) -> np.ndarray:
    """float32 CIELAB of an HxWx3 uint8 image, converted once per distinct colour."""
    flat = rgb.reshape(-1, 3).astype(np.uint32)
    packed = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
    keys, inverse = np.unique(packed, return_inverse=True)
    colours = np.stack([keys >> 16, (keys >> 8) & 255, keys & 255], axis=1).astype(np.float64)
    lab = rgb2lab(colours.reshape(-1, 1, 3) / 255.0).reshape(-1, 3).astype(np.float32)
    return lab[inverse.ravel()].reshape(rgb.shape)


def contrast_image(
    lab: np.ndarray, scale: float, labels: np.ndarray | None = None
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Gradient magnitude of the smoothed Lab image, in Lab units per original px.

    With `labels`, also returns the per-label sums (n, 3) of the smoothed Lab image: noise
    averages out there, so two halves of one noisy flat fill have nearly the same smoothed
    mean while two bands of a gradient keep their colour difference."""
    sigma = CONTRAST_SIGMA_PX * scale
    total = np.zeros(lab.shape[:2], np.float32)
    sums = None
    if labels is not None:
        opaque = labels >= 0
        n = int(labels.max(initial=-1)) + 1
        sums = np.zeros((n, 3))
    for c in range(3):
        smooth = ndimage.gaussian_filter(lab[..., c], sigma, mode="nearest", truncate=3.0)
        gy, gx = np.gradient(smooth)
        total += gx * gx + gy * gy
        if sums is not None:
            sums[:, c] = np.bincount(labels[opaque], weights=smooth[opaque], minlength=n)
    contrast = np.sqrt(total) * np.float32(scale)
    return contrast if sums is None else (contrast, sums)


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    return rgb2lab(np.asarray(rgb, np.float64).reshape(-1, 1, 3) / 255.0).reshape(-1, 3)


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    rgb = lab2rgb(np.asarray(lab, np.float64).reshape(-1, 1, 3)).reshape(-1, 3)
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def _de(a: np.ndarray, b: np.ndarray) -> float:
    return float(deltaE_ciede2000(np.asarray(a, np.float64), np.asarray(b, np.float64)))


def _edge_arrays(labels: np.ndarray, values: np.ndarray | None = None):
    """4-neighbour edges between differing labels >= 0: (a, b, per-edge mean of `values`)."""
    a = np.concatenate([labels[:, :-1].ravel(), labels[:-1, :].ravel()])
    b = np.concatenate([labels[:, 1:].ravel(), labels[1:, :].ravel()])
    keep = (a >= 0) & (b >= 0) & (a != b)
    v = None
    if values is not None:
        v = np.concatenate([
            (values[:, :-1] + values[:, 1:]).ravel(), (values[:-1, :] + values[1:, :]).ravel()
        ])[keep] * 0.5
    return a[keep], b[keep], v


def _relabel(labels: np.ndarray, root: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Labels mapped through `root` and made contiguous; returns (labels, kept old ids)."""
    survivors, new = np.unique(root, return_inverse=True)
    out = np.full(labels.shape, TRANSPARENT, dtype=np.int32)
    opaque = labels >= 0
    out[opaque] = new.ravel()[labels[opaque]].astype(np.int32)
    return out, survivors


# ------------------------------------------------------------------ 1. near-identical merge


def merge_near_regions(
    labels: np.ndarray, palette_rgb: np.ndarray, contrast: np.ndarray,
    max_delta_e: float = MERGE_DELTA_E, min_quiet_share: float = MERGE_MIN_QUIET_SHARE,
    max_shift: float = MERGE_MAX_SHIFT, smooth_sums: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge adjacent regions with near-identical colours and no real edge between them.

    Greedy, cheapest pair first. A pair merges when every pair of its members' original colours
    is within `max_delta_e` (complete linkage, so a banded gradient cannot chain into one
    colour), at least `min_quiet_share` of the boundary they share has `contrast` below
    QUIET_GRADIENT (a designed pale-on-pale edge stays), and its cost is at most `max_shift`:
    the area-weighted mean dE2000 between each region's source colour and the merged one,
    (A_x dE(m, src_x) + A_y dE(m, src_y)) / (A_x + A_y). Source colours come from
    `smooth_sums` (per-region sums of the smoothed source Lab, see `contrast_image`), so the
    noise-split halves of one flat fill merge whatever their areas while broad bands of a
    gradient stay; without it, from the palette colours. A merged region is drawn in its merged
    source colour m, so the gate and the output agree; unmerged regions keep their colour.
    Returns (labels, palette_rgb).
    """
    n = len(palette_rgb)
    if n < 2 or max_delta_e <= 0:
        return labels, palette_rgb
    a, b, c = _edge_arrays(labels, contrast)
    quiet = (c < QUIET_GRADIENT) if c is not None else None
    if len(a) == 0:
        return labels, palette_rgb
    lo, hi = np.minimum(a, b).astype(np.int64), np.maximum(a, b).astype(np.int64)
    keys, inverse, counts = np.unique(lo * n + hi, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse.ravel(), weights=quiet.astype(np.float64))
    # Original colours: distinct palette entries, with their pairwise dE2000.
    distinct, colour_of = np.unique(palette_rgb, axis=0, return_inverse=True)
    colour_of = colour_of.ravel()
    distinct_lab = _rgb_to_lab(distinct)
    dist = deltaE_ciede2000(distinct_lab[:, None, :], distinct_lab[None, :, :])
    pi, pj = keys // n, keys % n
    near = dist[colour_of[pi], colour_of[pj]] < max_delta_e
    if not near.any():
        return labels, palette_rgb

    area = np.bincount(labels[labels >= 0], minlength=n).astype(np.float64)
    if smooth_sums is None:
        src_sum = _rgb_to_lab(palette_rgb) * area[:, None]
    else:
        src_sum = np.asarray(smooth_sums, np.float64).copy()
    members: list[frozenset[int]] = [frozenset((int(colour_of[i]),)) for i in range(n)]
    border: list[dict[int, list[float]]] = [dict() for _ in range(n)]
    for i, j, s, k in zip(pi.tolist(), pj.tolist(), sums.tolist(), counts.tolist()):
        border[i][j] = [s, k]
        border[j][i] = [s, k]
    parent = np.arange(n)
    version = np.zeros(n, dtype=np.int64)

    def linkage(x: int, y: int) -> float:
        return max(dist[p, q] for p in members[x] for q in members[y])

    def push(heap: list, x: int, y: int) -> None:
        x, y = min(x, y), max(x, y)  # canonical order: the version stamps follow x, y
        s, k = border[x][y]
        if s / k < min_quiet_share or linkage(x, y) >= max_delta_e:
            return
        total = area[x] + area[y]
        merged = (src_sum[x] + src_sum[y]) / total
        cost = (area[x] * _de(merged, src_sum[x] / area[x])
                + area[y] * _de(merged, src_sum[y] / area[y])) / total
        if cost > max_shift:
            return
        heapq.heappush(heap, (cost, x, y, int(version[x]), int(version[y])))

    heap: list = []
    for i, j in zip(pi[near].tolist(), pj[near].tolist()):
        push(heap, i, j)
    while heap:
        _, x, y, vx, vy = heapq.heappop(heap)
        if parent[x] != x or parent[y] != y or version[x] != vx or version[y] != vy:
            continue
        keep, gone = (x, y) if area[x] >= area[y] else (y, x)
        parent[gone] = keep
        area[keep] += area[gone]
        src_sum[keep] += src_sum[gone]
        members[keep] = members[keep] | members[gone]
        del border[keep][gone]
        for nb, (s, k) in border[gone].items():
            if nb == keep:
                continue
            del border[nb][gone]
            entry = border[keep].setdefault(nb, [0.0, 0])
            entry[0] += s
            entry[1] += k
            border[nb][keep] = entry
        border[gone] = {}
        version[keep] += 1
        for nb in border[keep]:
            push(heap, keep, nb)

    root = parent.copy()
    for i in range(n):
        r = i
        while root[r] != r:
            r = root[r]
        root[i] = r
    out, survivors = _relabel(labels, root)
    merged = np.bincount(root, minlength=n)[survivors] > 1
    palette = palette_rgb[survivors].copy()
    if merged.any():
        means = src_sum[survivors[merged]] / area[survivors[merged], None]
        palette[merged] = _lab_to_rgb(means)
    return out, palette


# ------------------------------------------------------------------ 2. anti-aliasing slivers


def _interior(labels: np.ndarray) -> np.ndarray:
    """Pixels whose 4 neighbours share their label (outside the array counts as different)."""
    pad = np.pad(labels, 1, mode="constant", constant_values=labels.min(initial=0) - 1)
    return (
        (pad[:-2, 1:-1] == labels) & (pad[2:, 1:-1] == labels)
        & (pad[1:-1, :-2] == labels) & (pad[1:-1, 2:] == labels)
    )


def _thin_regions(labels: np.ndarray, n: int, limit: float) -> np.ndarray:
    """Per region 0..n-1: is its half width at most `limit` (processing px)? The half width is
    the largest distance from one of its pixels to the nearest pixel outside it (outside the
    canvas counts as outside); a w px ribbon has ceil(w / 2)."""
    interior = _interior(labels)
    # Chessboard distance never exceeds the Euclidean one: a region deep by it is thick for sure.
    rough = ndimage.distance_transform_cdt(interior, metric="chessboard") + 1
    thin = np.bincount(labels[(rough > limit) & (labels >= 0)], minlength=n)[:n] == 0
    # The rest are checked exactly: each in its own box (the nearest non-interior pixel to a
    # pixel of a region is always on that region's own border), or all at once when those boxes
    # would cover more than the image.
    boxes = ndimage.find_objects(labels + 1, max_label=n)
    todo = [r for r in np.flatnonzero(thin) if boxes[r] is not None]
    cover = sum((bx[0].stop - bx[0].start) * (bx[1].stop - bx[1].start)
                for bx in (boxes[r] for r in todo))
    if cover > labels.size:
        dist = ndimage.distance_transform_edt(interior) + 1.0
        deep = (dist > limit) & (labels >= 0)
        return thin & (np.bincount(labels[deep], minlength=n)[:n] == 0)
    for r in todo:
        inside = interior[boxes[r]] & (labels[boxes[r]] == r)
        if inside.any() and ndimage.distance_transform_edt(np.pad(inside, 1)).max() + 1 > limit:
            thin[r] = False
    return thin


def _neighbour_borders(window: np.ndarray, mask: np.ndarray) -> dict[int, int]:
    """4-neighbour edge counts from the `mask` pixels to each other label >= 0 in `window`."""
    tally: dict[int, int] = defaultdict(int)
    for inside, outside in (
        (mask[:, :-1], window[:, 1:]), (mask[:, 1:], window[:, :-1]),
        (mask[:-1, :], window[1:, :]), (mask[1:, :], window[:-1, :]),
    ):
        other = outside[inside & (outside >= 0)]
        for lab_id, cnt in zip(*np.unique(other, return_counts=True)):
            tally[int(lab_id)] += int(cnt)
    return tally


def _detached(target: np.ndarray, region: np.ndarray, own: int) -> np.ndarray:
    """Mask of the `own` pieces of `target` with no pixel 4-adjacent to `region`."""
    pieces, count = ndimage.label(target == own)
    if count == 0:
        return np.zeros(target.shape, bool)
    attached = np.zeros(count + 1, bool)
    for near, other in (
        (pieces[:, :-1], region[:, 1:]), (pieces[:, 1:], region[:, :-1]),
        (pieces[:-1, :], region[1:, :]), (pieces[1:, :], region[:-1, :]),
    ):
        attached[near[other]] = True
    attached[0] = True
    return ~attached[pieces]


def _split_sliver(window: np.ndarray, lab_win: np.ndarray, sliver: int, a: int, b: int,
                  lab_a: np.ndarray, lab_b: np.ndarray) -> np.ndarray | None:
    """New labels for the sliver's pixels in `window`: each goes to whichever of a, b its source
    colour is closer to; pieces cut off from their region flip to the other one. None when some
    piece touches neither (the sliver is then kept)."""
    full = window == sliver
    ys, xs = np.nonzero(full)
    crop = (slice(max(ys.min() - 1, 0), ys.max() + 2), slice(max(xs.min() - 1, 0), xs.max() + 2))
    window, lab_win = window[crop], lab_win[crop]  # every piece and its neighbours lie in here
    mask = window == sliver
    px = lab_win[mask]
    to_a = ((px - lab_a) ** 2).sum(1) <= ((px - lab_b) ** 2).sum(1)
    target = np.full(window.shape, -1, dtype=np.int64)
    target[mask] = np.where(to_a, a, b)
    region = {a: window == a, b: window == b}
    for _ in range(3):
        moved = False
        for own, other in ((a, b), (b, a)):
            cut = _detached(target, region[own], own)
            if cut.any():
                target[cut] = other
                moved = True
        if not moved:
            break
    if any(_detached(target, region[own], own).any() for own in (a, b)):
        return None
    return target[mask]


def _is_blend(px: np.ndarray, lab_a: np.ndarray, lab_b: np.ndarray, width_px: float) -> bool:
    """Do the sliver's source pixels `px` (Lab) read as a blend between A and B?

    Their mean must project onto the line A-B no further than SLIVER_MAX_OVERSHOOT beyond the
    nearer end and lie within SLIVER_MAX_OFFSET x |AB| of the line; a sliver at least
    SLIVER_FLAT_MIN_WIDTH_PX wide (original px) must also not be flat (a designed pinstripe)."""
    ab = np.asarray(lab_b, np.float64) - lab_a
    length2 = float(ab @ ab)
    if length2 <= 1e-9:
        return False
    length = length2 ** 0.5
    mean = px.mean(0) - lab_a
    t = float(mean @ ab) / length2
    if max(0.0, -t, t - 1.0) * length > SLIVER_MAX_OVERSHOOT:
        return False
    if np.linalg.norm(mean - t * ab) > SLIVER_MAX_OFFSET * length:
        return False
    if width_px >= SLIVER_FLAT_MIN_WIDTH_PX:
        median = np.median(px, axis=0)
        flat = np.mean(((px - median) ** 2).sum(1) <= SLIVER_FLAT_DE ** 2)
        if flat >= SLIVER_FLAT_SHARE:
            return False
    return True


def absorb_slivers(
    labels: np.ndarray, palette_rgb: np.ndarray, lab: np.ndarray, scale: float,
    max_half_width_px: float = SLIVER_MAX_HALF_WIDTH_PX,
) -> tuple[np.ndarray, np.ndarray]:
    """Split anti-aliasing slivers between the two regions they separate.

    A sliver is thin everywhere (half width, see `_thin_regions`, at most `max_half_width_px`
    original px, i.e. `max_half_width_px` x `scale` processing px) and its source pixels read as a blend between its two
    longest-border neighbours A and B (`_is_blend`): a real thin stroke darker or lighter than
    both sides, or a flat pinstripe, stays. Each sliver pixel goes to A or B by its source colour,
    so the cut sits where the blend crosses halfway. A sliver that received pixels from another
    one is re-measured before it is judged. Returns (labels, palette_rgb).
    """
    n = len(palette_rgb)
    if n < 3:
        return labels, palette_rgb
    limit = max_half_width_px * scale
    thin = np.flatnonzero(_thin_regions(labels, n, limit))
    if len(thin) == 0:
        return labels, palette_rgb
    pal_lab = _rgb_to_lab(palette_rgb)
    out = labels.copy()
    area = np.bincount(labels[labels >= 0], minlength=n)
    boxes = ndimage.find_objects(labels + 1)
    h, w = labels.shape
    alive = np.ones(n, dtype=bool)
    grown = np.zeros(n, dtype=bool)
    # Grown boxes: a sliver that receives pixels from another one spans both boxes.
    span = {int(i): boxes[i] for i in thin if boxes[i] is not None}
    for s in sorted(span, key=lambda i: (area[i], i)):
        ys, xs = span[s]
        y0, y1 = max(ys.start - 1, 0), min(ys.stop + 1, h)
        x0, x1 = max(xs.start - 1, 0), min(xs.stop + 1, w)
        window = out[y0:y1, x0:x1]
        mask = window == s
        if not mask.any():
            continue
        if grown[s]:  # it received pixels: is it still thin everywhere? (taxicab distance is
            # never below the Euclidean one, so this only errs towards keeping the sliver)
            own = np.where(mask, 0, -1)
            local = ndimage.distance_transform_cdt(_interior(own) & mask, metric="taxicab")
            if local[mask].max() + 1 > limit:
                continue
        tally = _neighbour_borders(window, mask)
        tally.pop(s, None)
        if len(tally) < 2:
            continue
        (a, _), (b, _) = sorted(tally.items(), key=lambda t: (-t[1], t[0]))[:2]
        lab_win = lab[y0:y1, x0:x1]
        width = 2.0 * float(mask.sum()) / max(sum(tally.values()), 1) / scale
        if not _is_blend(lab_win[mask].astype(np.float64), pal_lab[a], pal_lab[b], width):
            continue
        new = _split_sliver(window, lab_win, s, a, b, pal_lab[a], pal_lab[b])
        if new is None:
            continue
        window[mask] = new
        alive[s] = False
        for r in (a, b):
            if r in span:
                grown[r] = True
                ry, rx = span[r]
                span[r] = (slice(min(ry.start, ys.start), max(ry.stop, ys.stop)),
                           slice(min(rx.start, xs.start), max(rx.stop, xs.stop)))
    if alive.all():
        return labels, palette_rgb
    keep = np.flatnonzero(alive)  # absorbed ids no longer occur in `out`
    remap = np.full(n, -1, dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    final = np.full(labels.shape, TRANSPARENT, dtype=np.int32)
    opaque = out >= 0
    final[opaque] = remap[out[opaque]].astype(np.int32)
    return final, palette_rgb[keep]


# ------------------------------------------------------------------ 3. specks


def absorb_specks(
    labels: np.ndarray, palette_rgb: np.ndarray, scale: float,
    min_area_px: float = SPECK_AREA_PX, keep_delta_e: float = SPECK_KEEP_DELTA_E,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge regions under `min_area_px` (original px^2) into their most similar neighbour,
    unless every neighbour is at least `keep_delta_e` away (a real dot or highlight).
    The receiving region keeps its colour. Returns (labels, palette_rgb)."""
    n = len(palette_rgb)
    limit = min_area_px * scale * scale
    area = np.bincount(labels[labels >= 0], minlength=n).astype(np.float64)
    sliver_limit = max(limit, 50.0 * scale * scale)
    boxes = ndimage.find_objects(labels + 1, max_label=n)

    a, b, _ = _edge_arrays(labels)
    keys = np.unique(np.concatenate([a.astype(np.int64) * n + b, b.astype(np.int64) * n + a]))
    border: dict[int, set[int]] = defaultdict(set)
    for key in keys.tolist():
        border[key // n].add(key % n)

    def _is_candidate(i: int) -> bool:
        if area[i] < limit:
            return True
        if area[i] < sliver_limit and boxes[i] is not None:
            h = boxes[i][0].stop - boxes[i][0].start
            w = boxes[i][1].stop - boxes[i][1].start
            aspect = max(h, w) / max(min(h, w), 1)
            thickness = min(h, w)
            # Standalone islands inside a single region are only slivers if needle-like (aspect >= 8)
            if len(border[i]) < 2:
                return bool(aspect >= 8.0 or thickness <= 1.0 * scale)
            return bool(aspect >= 3.0 and thickness <= 3.5 * scale)
        return False

    small = [i for i in range(n) if _is_candidate(i)]
    if n < 2 or len(small) == 0:
        return labels, palette_rgb
    pal_lab = _rgb_to_lab(palette_rgb)
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return int(i)

    for s in sorted(small, key=lambda i: (area[i], i)):
        if find(s) != s:
            continue
        bx = boxes[s] if s < len(boxes) else None
        if bx is not None:
            h = bx[0].stop - bx[0].start
            w = bx[1].stop - bx[1].start
            aspect = max(h, w) / max(min(h, w), 1)
            fill_ratio = area[s] / max(w * h, 1)
            thickness = min(h, w)
            is_compact = (aspect <= 2.5) and (fill_ratio >= 0.35)
            is_sliver = (aspect >= 3.0) and (thickness <= 3.5 * scale)
        else:
            is_compact = True
            is_sliver = False
        nbs = {find(nb) for nb in border[s]} - {s}
        if not nbs:
            continue
        # Standalone islands inside 1 neighbor: sliver limit only absorbs if aspect >= 8
        if len(nbs) < 2 and area[s] >= limit and aspect < 8.0:
            continue
        max_allowed = sliver_limit if is_sliver else limit
        if area[s] >= max_allowed:
            continue
        de = {nb: _de(pal_lab[s], pal_lab[nb]) for nb in nbs}
        target = min(de, key=lambda nb: (de[nb], -area[nb], nb))
        if (is_compact or (len(nbs) == 1 and aspect < 8.0)) and de[target] >= keep_delta_e:
            continue
        parent[s] = target
        area[target] += area[s]
        border[target] |= border[s]
    root = np.array([find(i) for i in range(n)])
    if (root == np.arange(n)).all():
        return labels, palette_rgb
    out, survivors = _relabel(labels, root)
    return out, palette_rgb[survivors]


def prune_spurs(labels: np.ndarray, max_iterations: int = 4) -> np.ndarray:
    """Prune 1-pixel wide spurs and 2-pixel wide peninsulas protruding from regions.

    A pixel is pruned if 3 of its 4 orthogonal neighbours agree on a different label,
    or if it forms a 2-pixel wide terminal peninsula bounded by an adjacent label.
    Preserves 90-degree corners (which have at most 2 matching orthogonal neighbours).
    """
    out = labels.copy()
    for _ in range(max_iterations):
        up = np.pad(out[:-1, :], ((1, 0), (0, 0)), mode="edge")
        down = np.pad(out[1:, :], ((0, 1), (0, 0)), mode="edge")
        left = np.pad(out[:, :-1], ((0, 0), (1, 0)), mode="edge")
        right = np.pad(out[:, 1:], ((0, 0), (0, 1)), mode="edge")

        m1 = (up == down) & (down == left) & (up != out)
        m2 = (up == down) & (down == right) & (up != out)
        m3 = (up == left) & (left == right) & (up != out)
        m4 = (down == left) & (left == right) & (down != out)

        cand = m1 | m2 | m3 | m4
        repl = np.where(m1 | m2 | m3, up, down)

        left2 = np.pad(out[:, :-2], ((0, 0), (2, 0)), mode="edge")
        right2 = np.pad(out[:, 2:], ((0, 0), (0, 2)), mode="edge")
        up2 = np.pad(out[:-2, :], ((2, 0), (0, 0)), mode="edge")
        down2 = np.pad(out[2:, :], ((0, 2), (0, 0)), mode="edge")

        p_down_l = (left == right2) & (left == down) & (left != out)
        p_down_r = (right == left2) & (right == down) & (right != out)
        p_up_l = (left == right2) & (left == up) & (left != out)
        p_up_r = (right == left2) & (right == up) & (right != out)
        p_right_u = (up == down2) & (up == right) & (up != out)
        p_right_d = (down == up2) & (down == right) & (down != out)
        p_left_u = (up == down2) & (up == left) & (up != out)
        p_left_d = (down == up2) & (down == left) & (down != out)

        p_cand = p_down_l | p_down_r | p_up_l | p_up_r | p_right_u | p_right_d | p_left_u | p_left_d
        p_repl = np.where(p_down_l | p_up_l, left,
                 np.where(p_down_r | p_up_r, right,
                 np.where(p_right_u | p_left_u, up, down)))

        all_cand = cand | p_cand
        if not all_cand.any():
            break
        final_repl = np.where(cand, repl, p_repl)
        out[all_cand] = final_repl[all_cand]
    return out


def refine_regions(
    labels: np.ndarray, palette_rgb: np.ndarray, rgb: np.ndarray, scale: float,
    merge_delta_e: float = MERGE_DELTA_E, speck_area_px: float = SPECK_AREA_PX,
    lab: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """All three passes: near-identical merge, sliver absorption, speck absorption.
    `lab`: the float32 CIELAB of `rgb` when the caller has it (else computed here)."""
    if len(palette_rgb) < 2:
        return labels, palette_rgb
    if lab is None:
        lab = lab_image(rgb)
    contrast, smooth_sums = contrast_image(lab, scale, labels)
    labels, palette_rgb = merge_near_regions(labels, palette_rgb, contrast, merge_delta_e,
                                             smooth_sums=smooth_sums)
    labels, palette_rgb = absorb_slivers(labels, palette_rgb, lab, scale)
    labels, palette_rgb = absorb_specks(labels, palette_rgb, scale, speck_area_px)
    labels = prune_spurs(labels)
    return labels, palette_rgb
