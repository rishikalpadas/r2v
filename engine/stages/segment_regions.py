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
# ...and merging may move the larger region's mean source colour (smoothed source, so noise
# averages out) by at most this dE2000. A blotch inside a fill and the noise-split halves of one
# fill merge; broad bands of a designed gradient (e.g. the e (13) banner) stay apart. Measured
# on the logos (full pipeline, vs 1.0): 1.6 merges the soft-shaded letter fills of e (5), e (9)
# and e (17) (their halves read 1.3-1.6) for dE <= +0.02 per image and SSIM/edge F1 up on all.
MERGE_MAX_SHIFT = 1.6
# Anti-aliasing slivers: largest inscribed distance (distance transform, original px) of a
# region that is thin everywhere; 1.5 holds a ribbon up to 3 original px wide.
SLIVER_MAX_HALF_WIDTH_PX = 1.5
# "Between" test: dE(A, S) + dE(S, B) <= tol x dE(A, B). JPEG/SR ringing leaves edge ribbons a
# little darker than the darker side; measured on the logos: those read 1.3-1.45, real dark rims
# and outlines >= 2.0 (a black outline between pink and white ~4). 1.2 (the 3x3 blend vote's)
# kept the ringing ribbons as ragged fragments: specks +75, wobble +54 over 1.5 on the logos.
SLIVER_BETWEEN_TOLERANCE = 1.5
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
        smooth = ndimage.gaussian_filter(lab[..., c], sigma, mode="nearest")
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

    Greedy, least distinct pair (dE2000 of the current colours) first. A pair merges when every
    pair of its members' original colours is within `max_delta_e` (complete linkage, so a banded
    gradient cannot chain into one colour) and at least `min_quiet_share` of the boundary they
    share has `contrast` below QUIET_GRADIENT (a designed pale-on-pale edge stays), and merging
    moves the larger region's mean source colour by at most `max_shift`. With `smooth_sums`
    (per-region sums of the smoothed source Lab, see `contrast_image`) that mean is read from
    the source, so two noise-split halves of one flat fill merge whatever their areas while
    broad bands of a gradient stay; without, from the palette colours. The merged colour is the
    area-weighted mean in Lab; unmerged regions keep their colour exactly.
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
    lab = _rgb_to_lab(palette_rgb)
    lab_sum = lab * area[:, None]
    src_sum = (lab_sum if smooth_sums is None else np.asarray(smooth_sums, np.float64)).copy()
    members: list[frozenset[int]] = [frozenset((int(colour_of[i]),)) for i in range(n)]
    border: list[dict[int, list[float]]] = [dict() for _ in range(n)]
    for i, j, s, k in zip(pi.tolist(), pj.tolist(), sums.tolist(), counts.tolist()):
        border[i][j] = [s, k]
        border[j][i] = [s, k]
    parent = np.arange(n)
    version = np.zeros(n, dtype=np.int64)

    def linkage(x: int, y: int) -> float:
        return max(dist[p, q] for p in members[x] for q in members[y])

    def colour(x: int) -> np.ndarray:
        return lab_sum[x] / area[x]

    def push(heap: list, x: int, y: int) -> None:
        s, k = border[x][y]
        if s / k < min_quiet_share or linkage(x, y) >= max_delta_e:
            return
        big = x if area[x] >= area[y] else y
        merged = (src_sum[x] + src_sum[y]) / (area[x] + area[y])
        if _de(merged, src_sum[big] / area[big]) > max_shift:
            return
        heapq.heappush(heap, (_de(colour(x), colour(y)), min(x, y), max(x, y),
                              int(version[x]), int(version[y])))

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
        lab_sum[keep] += lab_sum[gone]
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
        means = lab_sum[survivors[merged]] / area[survivors[merged], None]
        palette[merged] = _lab_to_rgb(means)
    return out, palette


# ------------------------------------------------------------------ 2. anti-aliasing slivers


def _half_widths(labels: np.ndarray, n: int) -> np.ndarray:
    """Per region: largest distance (px) from one of its pixels to the nearest pixel outside it
    (outside the canvas counts as outside); a w px wide ribbon gives ceil(w / 2)."""
    pad = np.pad(labels, 1, mode="constant", constant_values=labels.min(initial=0) - 1)
    interior = (
        (pad[:-2, 1:-1] == labels) & (pad[2:, 1:-1] == labels)
        & (pad[1:-1, :-2] == labels) & (pad[1:-1, 2:] == labels)
    )
    dist = ndimage.distance_transform_edt(interior).astype(np.float32) + 1.0
    opaque = labels >= 0
    out = np.zeros(n, np.float64)
    np.maximum.at(out, labels[opaque], dist[opaque])
    return out


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


def _touches(piece: np.ndarray, region: np.ndarray) -> bool:
    """Is any `piece` pixel 4-adjacent to a `region` pixel? (both boolean, same window)."""
    grown = ndimage.binary_dilation(piece, ndimage.generate_binary_structure(2, 1))
    return bool((grown & region).any())


def _split_sliver(window: np.ndarray, lab_win: np.ndarray, sliver: int, a: int, b: int,
                  lab_a: np.ndarray, lab_b: np.ndarray) -> np.ndarray | None:
    """New labels for the sliver's pixels in `window`: each goes to whichever of a, b its source
    colour is closer to; pieces cut off from their region flip to the other one. None when some
    piece touches neither (the sliver is then kept)."""
    mask = window == sliver
    px = lab_win[mask]
    to_a = ((px - lab_a) ** 2).sum(1) <= ((px - lab_b) ** 2).sum(1)
    target = np.full(window.shape, -1, dtype=np.int64)
    target[mask] = np.where(to_a, a, b)
    for _ in range(3):
        moved = False
        for own, other in ((a, b), (b, a)):
            pieces, count = ndimage.label(target == own)
            if count == 0:
                continue
            region = window == own
            for k in range(1, count + 1):
                piece = pieces == k
                if not _touches(piece, region):
                    target[piece] = other
                    moved = True
        if not moved:
            break
    for own in (a, b):
        pieces, count = ndimage.label(target == own)
        region = window == own
        for k in range(1, count + 1):
            if not _touches(pieces == k, region):
                return None
    return target[mask]


def absorb_slivers(
    labels: np.ndarray, palette_rgb: np.ndarray, lab: np.ndarray, scale: float,
    max_half_width_px: float = SLIVER_MAX_HALF_WIDTH_PX,
    tolerance: float = SLIVER_BETWEEN_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray]:
    """Split anti-aliasing slivers between the two regions they separate.

    A sliver is thin everywhere (half width <= max_half_width_px original px) and its colour lies
    between its two longest-border neighbours A and B: dE(A, S) + dE(S, B) <= tolerance x dE(A, B).
    A real thin stroke darker (or lighter) than both sides fails that test and stays. Each sliver
    pixel goes to A or B by its source colour, so the cut sits where the blend crosses halfway.
    Returns (labels, palette_rgb).
    """
    n = len(palette_rgb)
    if n < 3:
        return labels, palette_rgb
    thin = np.flatnonzero(_half_widths(labels, n) <= max_half_width_px * scale)
    if len(thin) == 0:
        return labels, palette_rgb
    pal_lab = _rgb_to_lab(palette_rgb)
    out = labels.copy()
    area = np.bincount(labels[labels >= 0], minlength=n)
    boxes = ndimage.find_objects(labels + 1)
    h, w = labels.shape
    alive = np.ones(n, dtype=bool)
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
        tally = _neighbour_borders(window, mask)
        tally.pop(s, None)
        if len(tally) < 2:
            continue
        (a, _), (b, _) = sorted(tally.items(), key=lambda t: (-t[1], t[0]))[:2]
        d_ab = _de(pal_lab[a], pal_lab[b])
        if d_ab <= 0 or _de(pal_lab[a], pal_lab[s]) + _de(pal_lab[s], pal_lab[b]) > tolerance * d_ab:
            continue
        new = _split_sliver(window, lab[y0:y1, x0:x1], s, a, b, pal_lab[a], pal_lab[b])
        if new is None:
            continue
        window[mask] = new
        alive[s] = False
        for r in (a, b):
            if r in span:
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
    small = np.flatnonzero(area < limit)
    if n < 2 or len(small) == 0:
        return labels, palette_rgb
    a, b, _ = _edge_arrays(labels)
    keys = np.unique(np.concatenate([a.astype(np.int64) * n + b, b.astype(np.int64) * n + a]))
    border: dict[int, set[int]] = defaultdict(set)
    for key in keys.tolist():
        border[key // n].add(key % n)
    pal_lab = _rgb_to_lab(palette_rgb)
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return int(i)

    for s in sorted(small.tolist(), key=lambda i: (area[i], i)):
        if find(s) != s or area[s] >= limit:
            continue
        nbs = {find(nb) for nb in border[s]} - {s}
        if not nbs:
            continue
        de = {nb: _de(pal_lab[s], pal_lab[nb]) for nb in nbs}
        target = min(de, key=lambda nb: (de[nb], -area[nb], nb))
        if de[target] >= keep_delta_e:
            continue
        parent[s] = target
        area[target] += area[s]
        border[target] |= border[s]
    root = np.array([find(i) for i in range(n)])
    if (root == np.arange(n)).all():
        return labels, palette_rgb
    out, survivors = _relabel(labels, root)
    return out, palette_rgb[survivors]


def refine_regions(
    labels: np.ndarray, palette_rgb: np.ndarray, rgb: np.ndarray, scale: float,
    merge_delta_e: float = MERGE_DELTA_E, speck_area_px: float = SPECK_AREA_PX,
) -> tuple[np.ndarray, np.ndarray]:
    """All three passes: near-identical merge, sliver absorption, speck absorption."""
    if len(palette_rgb) < 2:
        return labels, palette_rgb
    lab = lab_image(rgb)
    contrast, smooth_sums = contrast_image(lab, scale, labels)
    labels, palette_rgb = merge_near_regions(labels, palette_rgb, contrast, merge_delta_e,
                                             smooth_sums=smooth_sums)
    labels, palette_rgb = absorb_slivers(labels, palette_rgb, lab, scale)
    labels, palette_rgb = absorb_specks(labels, palette_rgb, scale, speck_area_px)
    return labels, palette_rgb
