"""Segment stage: quantize flat-art pixels into connected, color-labelled regions."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import ndimage
from skimage.color import deltaE_ciede2000

from engine.presets import SegmentParams
from engine.stages.segment_palette import (
    _cluster_colors,
    _drop_blend_clusters,
    _merge_clusters,
    _to_lab,
)
from engine.types import TRANSPARENT, RasterImage, RegionMap

OPAQUE_ALPHA = 128
# The 3x3 majority pass is deliberately conservative so corners, tips and thin lines of flat
# shapes survive. It repaints (a) isolated speckles and (b) anti-alias ribbons, i.e. pixels
# whose color lies between their two strongest neighbor colors.
MAX_SPECKLE_OWN_VOTES = 1
MIN_WINNER_VOTES = 5
MAX_BLEND_OWN_VOTES = 5
BLEND_TOLERANCE = 1.2


def _blend_vote_changes(
    labels: np.ndarray, votes: np.ndarray, own: np.ndarray, palette: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pixels whose color lies between their two strongest neighbor colors (anti-alias ribbons).

    Returns (is_blend, target label = the stronger of the two neighbor labels).
    """
    lab = _to_lab(palette)
    dist = deltaE_ciede2000(lab[:, None, :], lab[None, :, :])
    others = votes.copy()
    np.put_along_axis(others, labels[None], -1, axis=0)
    first = others.argmax(axis=0)
    first_votes = others.max(axis=0)
    np.put_along_axis(others, first[None], -1, axis=0)
    second = others.argmax(axis=0)
    second_votes = others.max(axis=0)
    own_lab = labels
    via = dist[first, own_lab] + dist[own_lab, second]
    between = via <= BLEND_TOLERANCE * dist[first, second]
    is_blend = (own <= MAX_BLEND_OWN_VOTES) & (second_votes >= 1) & (first_votes >= own) & between
    return is_blend, first


def _majority_cleanup(labels: np.ndarray, palette: np.ndarray, noisy: bool) -> np.ndarray:
    """One 3x3 majority pass on speckles and anti-alias ribbons; never erases a label entirely."""
    n_labels = len(palette)
    opaque = labels >= 0
    pad = np.pad(labels, 1, mode="edge")
    same_4 = (
        (pad[:-2, 1:-1] == labels) & (pad[2:, 1:-1] == labels)
        & (pad[1:-1, :-2] == labels) & (pad[1:-1, 2:] == labels)
    )
    candidate = opaque & ~same_4
    ys, xs = np.nonzero(candidate)
    if len(ys) == 0:
        return labels
    # 3x3 neighborhood labels for candidate pixels only: (N, 9); votes: (K, N).
    window = np.stack([pad[ys + dy, xs + dx] for dy in range(3) for dx in range(3)], axis=1)
    votes = np.stack([(window == c).sum(axis=1, dtype=np.int16) for c in range(n_labels)])
    cand_labels = labels[ys, xs]
    own = np.take_along_axis(votes, cand_labels[None], axis=0)[0]
    winner = votes.argmax(axis=0)
    win_votes = votes.max(axis=0)
    speckle = (own <= MAX_SPECKLE_OWN_VOTES) & (win_votes >= MIN_WINNER_VOTES)
    target = winner
    change = speckle & (winner != cand_labels)
    if noisy and n_labels >= 3:
        is_blend, blend_target = _blend_vote_changes(cand_labels, votes, own, palette)
        blend = is_blend & ~speckle
        change |= blend
        target = np.where(blend, blend_target, winner)
    out = labels.copy()
    out[ys[change], xs[change]] = target[change]
    lost = np.setdiff1d(np.unique(labels[opaque]), np.unique(out[out >= 0]))
    if len(lost):
        out = np.where(np.isin(labels, lost), labels, out)
    return out


def _edge_pairs(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pairs (a, b) of differing positive ids across 4-neighbor edges."""
    a = np.concatenate([ids[:, :-1].ravel(), ids[:-1, :].ravel()])
    b = np.concatenate([ids[:, 1:].ravel(), ids[1:, :].ravel()])
    keep = (a > 0) & (b > 0) & (a != b)
    return a[keep], b[keep]


def _components(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-cluster 4-connected components: comp ids (0 = transparent) and cluster per comp id."""
    comp = np.zeros(labels.shape, dtype=np.int64)
    comp_cluster = [-1]
    for c in np.unique(labels[labels >= 0]):
        lab, n = ndimage.label(labels == c)
        mask = lab > 0
        comp[mask] = lab[mask] + len(comp_cluster) - 1
        comp_cluster.extend([int(c)] * n)
    return comp, np.array(comp_cluster)


def _diagonal_reach(comp: np.ndarray, comp_cluster: np.ndarray) -> np.ndarray:
    """Pixel count of each component's 8-connected same-color group.

    Staircase pieces of a thin diagonal line are tiny 4-connected components but belong to
    one long feature; judging smallness on the 8-connected group keeps them.
    """
    reach = np.zeros(len(comp_cluster), dtype=np.int64)
    eight = np.ones((3, 3), dtype=int)
    cluster_map = comp_cluster[comp]
    for c in np.unique(comp_cluster[1:]):
        mask = cluster_map == c
        lab, n = ndimage.label(mask, structure=eight)
        group_sizes = np.bincount(lab.ravel(), minlength=n + 1)
        ids, first = np.unique(comp[mask], return_index=True)
        reach[ids] = group_sizes[lab[mask][first]]
    return reach


def _merge_small_components(comp: np.ndarray, comp_cluster: np.ndarray, min_px: int) -> np.ndarray:
    """Root id per component id; small ones join their longest-border neighbor.

    A component that is the only one of its color is always kept.
    """
    sizes = np.bincount(comp.ravel(), minlength=len(comp_cluster)).astype(np.int64)
    reach = _diagonal_reach(comp, comp_cluster)
    a, b = _edge_pairs(comp)
    borders: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    n_ids = len(comp_cluster)
    # Each edge counts for both components: a small one must see its left/up neighbours too.
    keys, cnts = np.unique(np.concatenate([a * n_ids + b, b * n_ids + a]), return_counts=True)
    for key, cnt in zip(keys.tolist(), cnts.tolist()):
        borders[key // n_ids][key % n_ids] = cnt
    per_cluster = np.bincount(comp_cluster[1:], minlength=int(comp_cluster.max()) + 1)
    parent = np.arange(len(comp_cluster))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return int(i)

    small = [i for i in range(1, len(sizes)) if reach[i] < min_px and per_cluster[comp_cluster[i]] > 1]
    for i in sorted(small, key=lambda i: (sizes[i], i)):
        if find(i) != i:
            continue
        tally: dict[int, int] = defaultdict(int)
        for nb, cnt in borders[i].items():
            r = find(nb)
            if r != i:
                tally[r] += cnt
        if not tally:
            continue
        target = max(tally, key=lambda r: (tally[r], -r))
        parent[i] = target
        sizes[target] += sizes[i]
        for r, cnt in tally.items():
            if r != target:
                borders[target][r] += cnt
                borders[r][target] += cnt
    return np.array([find(i) for i in range(len(comp_cluster))])


def _adjacency(labels: np.ndarray) -> frozenset[tuple[int, int]]:
    a, b = _edge_pairs(labels + 1)
    if len(a) == 0:
        return frozenset()
    pairs = np.unique(np.stack([np.minimum(a, b) - 1, np.maximum(a, b) - 1], axis=1), axis=0)
    return frozenset((int(x), int(y)) for x, y in pairs)


def _empty_map(shape: tuple[int, int]) -> RegionMap:
    return RegionMap(
        labels=np.full(shape, TRANSPARENT, dtype=np.int32),
        palette_rgb=np.zeros((0, 3), dtype=np.uint8),
        palette_lab=np.zeros((0, 3), dtype=np.float64),
        adjacency=frozenset(),
    )


def segment(image: RasterImage, params: SegmentParams) -> RegionMap:
    px = image.pixels
    h, w = px.shape[:2]
    opaque = px[..., 3] >= OPAQUE_ALPHA
    if not opaque.any():
        return _empty_map((h, w))

    rgb = px[..., :3][opaque].astype(np.uint32)
    packed = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
    keys, pixel_colors, counts = np.unique(packed, return_inverse=True, return_counts=True)
    pixel_colors = pixel_colors.ravel()
    colors = np.stack([keys >> 16, (keys >> 8) & 255, keys & 255], axis=1).astype(np.uint8)
    cluster = _cluster_colors(colors, pixel_colors, params)
    cluster, palette = _merge_clusters(colors, counts, cluster, params.merge_delta_e)
    noisy = len(colors) > params.max_colors  # quantized (KMeans) input
    if noisy and len(palette) >= 3:
        per_cluster = np.bincount(cluster[pixel_colors], minlength=len(palette))
        cluster, palette = _drop_blend_clusters(cluster, palette, per_cluster)

    labels = np.full((h, w), TRANSPARENT, dtype=np.int32)
    labels[opaque] = cluster[pixel_colors]
    labels = _majority_cleanup(labels, palette, noisy)

    comp, comp_cluster = _components(labels)
    # min_region_px is an area in original pixels; one original pixel covers scale^2 here.
    min_px = round(params.min_region_px * image.scale**2)
    merged = _merge_small_components(comp, comp_cluster, min_px)[comp]
    survivors, region = np.unique(merged[comp > 0], return_inverse=True)
    out = np.full((h, w), TRANSPARENT, dtype=np.int32)
    out[comp > 0] = region.ravel().astype(np.int32)

    palette_rgb = palette[comp_cluster[survivors]]
    return RegionMap(
        labels=out,
        palette_rgb=palette_rgb,
        palette_lab=_to_lab(palette_rgb),
        adjacency=_adjacency(out),
    )
