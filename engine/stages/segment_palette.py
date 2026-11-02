"""Segment palette: cluster pixel colours into a palette of real (non-blend) colours."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import ndimage
from skimage.color import deltaE_ciede2000, rgb2lab
from sklearn.cluster import KMeans

from engine.presets import SegmentParams

BLEND_CLUSTER_TOLERANCE = 1.05
BLEND_MAX_SHARE = 0.05
# Fraction of a cluster's pixels whose whole 3x3 is that cluster. Measured on a real JPEG logo:
# anti-alias blend clusters 0.00 (<= 0.03 at k=32), every real colour >= 0.43.
BLEND_MAX_INTERIOR = 0.10
KMEANS_N_INIT = 4
# Re-merge of one noisy colour that complete linkage split (its noise cloud is wider than the
# merge threshold). Split halves interleave pixel by pixel and are spread over their whole
# common area; a real thin band (outline) also interleaves along its edge but never reaches
# the core of the union. Measured: noise splits share >= 0.85 4-neighbour edges per pixel of
# the smaller half and its sparser half has interior <= 0.29; real near colours on a JPEG logo
# <= 0.39 edges/px and interior >= 0.54; a 2-4 px outline beside a near colour shares 0.8
# edges/px with interior 0.05, but has no pixel REMERGE_CORE_RADIUS px inside the union, while
# a noise half reaches the core as often as its partner (share ratio ~1; outline ratio 0). A +-10 noise cloud spans dE ~15, so the colour bound
# (max linkage over the original colours, so merged groups cannot chain) is 4 x merge_delta_e.
REMERGE_MIN_SHARED_EDGES = 0.5
REMERGE_MAX_INTERIOR = 0.5
REMERGE_CORE_RADIUS = 4
REMERGE_MIN_CORE_RATIO = 0.5
REMERGE_DELTA_E_FACTOR = 4.0
# Noise-split halves overlap: their centroid gap is about their own spread. Measured centroid
# dE / pooled RMS within-cluster dE: noise splits 1.5-2.5 (page splits under JPEG up to 4.1);
# real two-colour patterns (stripes, checkers, dots at dE 9-19, raw and JPEG q85) >= 5.9.
REMERGE_MAX_SEPARATION = 5.0


def _to_lab(rgb: np.ndarray) -> np.ndarray:
    return rgb2lab(rgb.reshape(-1, 1, 3).astype(np.float64) / 255.0).reshape(-1, 3)


def _cluster_colors(
    colors: np.ndarray, pixel_colors: np.ndarray, params: SegmentParams
) -> np.ndarray:
    """Cluster id per unique color. pixel_colors: per-opaque-pixel index into colors."""
    if len(colors) <= params.max_colors:
        return np.arange(len(colors))
    rng = np.random.default_rng(params.seed)
    if len(pixel_colors) > params.sample_px:
        sample = rng.choice(pixel_colors, size=params.sample_px, replace=False)
    else:
        sample = pixel_colors
    lab = _to_lab(colors)
    k = min(params.max_colors, len(np.unique(sample)))
    km = KMeans(n_clusters=k, n_init=KMEANS_N_INIT, random_state=params.seed)
    km.fit(lab[sample])
    return km.predict(lab)


def _complete_linkage_groups(dist: np.ndarray, threshold: float) -> np.ndarray:
    """Group id per item: agglomerative complete linkage on a pairwise distance matrix.

    Repeatedly merges the closest pair of groups while every pairwise distance inside the
    merged group stays < threshold. Unlike transitive (single-linkage) merging, a chain of
    close neighbours (white ~ blend ~ pink) can never fuse colours that are far apart.
    """
    n = len(dist)
    link = dist.astype(np.float64).copy()
    np.fill_diagonal(link, np.inf)
    group = np.arange(n)
    active = np.ones(n, dtype=bool)
    while active.sum() > 1:
        masked = np.where(active[:, None] & active[None, :], link, np.inf)
        flat = int(np.argmin(masked))
        i, j = divmod(flat, n)
        if not masked[i, j] < threshold:
            break
        i, j = min(i, j), max(i, j)
        # Lance-Williams update for complete linkage: distance to the union is the max.
        link[i, :] = np.maximum(link[i, :], link[j, :])
        link[:, i] = link[i, :]
        link[i, i] = np.inf
        active[j] = False
        group[group == j] = i
    return group


def _merge_clusters(
    colors: np.ndarray, counts: np.ndarray, cluster: np.ndarray, delta_e: float
) -> tuple[np.ndarray, np.ndarray]:
    """Complete-linkage merge of clusters whose centroids are all within deltaE2000 < delta_e.

    Returns (cluster ids, palette rgb); merged colours are pixel-weighted means.
    """
    k = int(cluster.max()) + 1
    w = counts.astype(np.float64)
    sums = np.zeros((k, 3))
    np.add.at(sums, cluster, colors * w[:, None])
    tot = np.bincount(cluster, weights=w, minlength=k)
    ids = np.flatnonzero(tot > 0)
    cent_rgb = sums[ids] / tot[ids, None]
    cent_lab = _to_lab(np.clip(np.rint(cent_rgb), 0, 255).astype(np.uint8))
    dist = deltaE_ciede2000(cent_lab[:, None, :], cent_lab[None, :, :])
    root = np.zeros(k, dtype=np.int64)
    root[ids] = ids[_complete_linkage_groups(dist, delta_e)]
    _, new_id = np.unique(root[cluster], return_inverse=True)
    new_id = new_id.ravel()
    m = int(new_id.max()) + 1
    msum = np.zeros((m, 3))
    np.add.at(msum, new_id, colors * w[:, None])
    mtot = np.bincount(new_id, weights=w, minlength=m)
    palette = np.clip(np.rint(msum / mtot[:, None]), 0, 255).astype(np.uint8)
    return new_id, palette


def _interior_mask(labels: np.ndarray) -> np.ndarray:
    """Pixels whose whole 3x3 neighbourhood has their label.

    Outside the image counts as a different label, so a 1-2 px strip along the border is not
    mistaken for solid area.
    """
    pad = np.pad(labels, 1, mode="constant", constant_values=labels.min(initial=0) - 1)
    h, w = labels.shape
    inside = np.ones(labels.shape, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            inside &= pad[dy:dy + h, dx:dx + w] == labels
    return inside


def _interior_fraction(labels: np.ndarray, n: int) -> np.ndarray:
    """Per label 0..n-1: share of its pixels that are interior (1.0 for absent labels)."""
    opaque = labels >= 0
    total = np.bincount(labels[opaque], minlength=n).astype(np.float64)
    inner = np.bincount(labels[opaque & _interior_mask(labels)], minlength=n)
    return np.divide(inner, total, out=np.ones(n), where=total > 0)


def _drop_blend_clusters(
    cluster: np.ndarray, palette: np.ndarray, counts: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Remove minor clusters that are JPEG/AA blends between two larger clusters.

    A blend lies between two larger clusters in colour AND has (almost) no interior pixels in
    `labels` (per-pixel cluster ids, TRANSPARENT < 0): blends live in 1-2 px edge ribbons, while
    a genuine minor colour has solid area. Only used on the noisy (KMeans) path.
    Returns (remapped cluster ids, palette).
    """
    interior = _interior_fraction(labels, len(palette))
    lab = _to_lab(palette)
    dist = deltaE_ciede2000(lab[:, None, :], lab[None, :, :])
    alive = np.ones(len(palette), dtype=bool)
    total = counts.sum()
    for c in np.argsort(counts, kind="stable"):
        if counts[c] > BLEND_MAX_SHARE * total:
            break
        if interior[c] >= BLEND_MAX_INTERIOR:
            continue  # has solid area: a genuine fill, not an edge ribbon
        bigger = np.flatnonzero(alive & (counts > counts[c]))
        if len(bigger) < 2:
            continue
        pair = dist[np.ix_(bigger, bigger)]
        via = dist[bigger, c][:, None] + dist[c, bigger][None, :]
        if np.any((via <= BLEND_CLUSTER_TOLERANCE * pair) & (pair > 0)):
            alive[c] = False
    kept = np.flatnonzero(alive)
    # Nearest *surviving* cluster for every cluster (kept ones map to themselves).
    nearest = np.argmin(dist[:, kept], axis=1)
    nearest[kept] = np.arange(len(kept))
    return nearest[cluster], palette[kept]


def _shared_edges(labels: np.ndarray, n: int) -> np.ndarray:
    """Symmetric (n, n) count of 4-neighbour pixel edges between differing labels >= 0."""
    a = np.concatenate([labels[:, :-1].ravel(), labels[:-1, :].ravel()])
    b = np.concatenate([labels[:, 1:].ravel(), labels[1:, :].ravel()])
    keep = (a >= 0) & (b >= 0) & (a != b)
    edges = np.bincount(a[keep] * n + b[keep], minlength=n * n).reshape(n, n)
    return edges + edges.T


def _core_share(labels: np.ndarray, small: int, other: int) -> tuple[float, float]:
    """Shares of `small`'s and `other`'s pixels lying REMERGE_CORE_RADIUS px inside their union.

    Pin holes left by further noise splits are closed first. Outside the canvas is not part
    of the union, so a band along the canvas edge has no core either.
    """
    union = (labels == small) | (labels == other)
    union = ndimage.binary_closing(union, np.ones((3, 3), bool), border_value=0)
    size = 2 * REMERGE_CORE_RADIUS + 1
    core = ndimage.binary_erosion(union, np.ones((size, size), bool), border_value=0)
    shares = []
    for c in (small, other):
        own = labels == c
        shares.append(float((core & own).sum()) / max(int(own.sum()), 1))
    return shares[0], shares[1]


def _spread(colors_lab: np.ndarray, weights: np.ndarray, cluster: np.ndarray,
            palette_lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per cluster: pixel count and RMS dE2000 of its pixels to its own palette colour."""
    n = len(palette_lab)
    d2 = deltaE_ciede2000(colors_lab, palette_lab[cluster]) ** 2
    count = np.bincount(cluster, weights, n)
    msq = np.bincount(cluster, weights * d2, n) / np.maximum(count, 1)
    return count, msq


def _separation(palette_lab: np.ndarray, count: np.ndarray, msq: np.ndarray, i: int, j: int
                ) -> float:
    """Centroid dE2000 of clusters i, j over their pooled RMS within-cluster spread."""
    pooled = np.sqrt((count[i] * msq[i] + count[j] * msq[j]) / max(count[i] + count[j], 1))
    return float(deltaE_ciede2000(palette_lab[i], palette_lab[j])) / max(float(pooled), 1e-6)


def _closest_co_located_pair(
    labels: np.ndarray, area: np.ndarray, candidate: np.ndarray, link: np.ndarray,
    separable: Callable[[int, int], bool],
) -> tuple[int, int] | None:
    """Closest candidate pair (i < j) whose colours overlap (not `separable`) and whose smaller
    cluster is spread through their union."""
    ii, jj = np.nonzero(np.triu(candidate, 1))
    for k in np.argsort(link[ii, jj], kind="stable"):
        i, j = int(ii[k]), int(jj[k])
        if separable(i, j):
            continue
        small, other = (i, j) if area[i] <= area[j] else (j, i)
        small_core, other_core = _core_share(labels, small, other)
        if other_core > 0 and small_core >= REMERGE_MIN_CORE_RATIO * other_core:
            return i, j
    return None


def _merge_intermixed(
    cluster: np.ndarray,
    palette: np.ndarray,
    colors: np.ndarray,
    counts: np.ndarray,
    labels_of: Callable[[np.ndarray], np.ndarray],
    delta_e: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-merge clusters that are pixel-interleaved splits of one noisy colour.

    Greedy: merge the closest pair whose original colours are all within
    REMERGE_DELTA_E_FACTOR * delta_e (max linkage), that shares many edges relative to the
    smaller cluster's area, where one has little solid interior, whose pixel colours overlap
    (centroid gap <= REMERGE_MAX_SEPARATION x pooled spread), and whose smaller cluster
    reaches the core of their union (not only one border); repeat. The merged
    colour is the area-weighted mean; unmerged colours are unchanged.
    `colors`/`counts`: the unique pixel colours `cluster` indexes, with pixel counts.
    `labels_of(cluster)` builds the label map. Returns (cluster ids, palette).
    """
    palette = palette.copy()
    colors_lab = _to_lab(colors)
    weights = counts.astype(np.float64)
    lab = _to_lab(palette)
    link = deltaE_ciede2000(lab[:, None, :], lab[None, :, :])  # max over original members
    np.fill_diagonal(link, np.inf)
    bound = REMERGE_DELTA_E_FACTOR * delta_e
    while len(palette) > 1:
        n = len(palette)
        labels = labels_of(cluster)
        area = np.bincount(labels[labels >= 0], minlength=n)
        interior = _interior_fraction(labels, n)
        mixed = _shared_edges(labels, n) / np.maximum(np.minimum.outer(area, area), 1)
        candidate = (
            (link < bound)
            & (mixed >= REMERGE_MIN_SHARED_EDGES)
            & (np.minimum.outer(interior, interior) < REMERGE_MAX_INTERIOR)
        )
        palette_lab = _to_lab(palette)
        count, msq = _spread(colors_lab, weights, cluster, palette_lab)

        def separable(i: int, j: int) -> bool:
            return _separation(palette_lab, count, msq, i, j) > REMERGE_MAX_SEPARATION

        pair = _closest_co_located_pair(labels, area, candidate, link, separable)
        if pair is None:
            break
        i, j = pair
        mean = (palette[i] * float(area[i]) + palette[j] * float(area[j])) / max(area[i] + area[j], 1)
        palette[i] = np.clip(np.rint(mean), 0, 255).astype(np.uint8)
        link[i, :] = np.maximum(link[i, :], link[j, :])
        link[:, i] = link[i, :]
        link[i, i] = np.inf
        palette = np.delete(palette, j, axis=0)
        link = np.delete(np.delete(link, j, axis=0), j, axis=1)
        cluster = np.where(cluster == j, i, cluster)
        cluster = np.where(cluster > j, cluster - 1, cluster)
    return cluster, palette
