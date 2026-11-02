"""Segment palette: cluster pixel colours into a palette of real (non-blend) colours."""
from __future__ import annotations

import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab
from sklearn.cluster import KMeans

from engine.presets import SegmentParams

BLEND_CLUSTER_TOLERANCE = 1.05
BLEND_MAX_SHARE = 0.05
KMEANS_N_INIT = 4


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


def _merge_clusters(
    colors: np.ndarray, counts: np.ndarray, cluster: np.ndarray, delta_e: float
) -> tuple[np.ndarray, np.ndarray]:
    """Merge clusters with centroid deltaE2000 < delta_e. Returns (cluster ids, palette rgb)."""
    k = int(cluster.max()) + 1
    w = counts.astype(np.float64)
    sums = np.zeros((k, 3))
    np.add.at(sums, cluster, colors * w[:, None])
    tot = np.bincount(cluster, weights=w, minlength=k)
    present = tot > 0
    parent = list(range(k))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    cent_rgb = sums / np.maximum(tot, 1)[:, None]
    cent_lab = _to_lab(np.clip(np.rint(cent_rgb), 0, 255).astype(np.uint8))
    ids = np.flatnonzero(present)
    for ai, a in enumerate(ids):
        for b in ids[ai + 1:]:
            if deltaE_ciede2000(cent_lab[a], cent_lab[b]) < delta_e:
                parent[find(int(b))] = find(int(a))
    roots = np.array([find(i) for i in range(k)])
    _, new_id = np.unique(roots[cluster], return_inverse=True)
    new_id = new_id.ravel()
    m = int(new_id.max()) + 1
    msum = np.zeros((m, 3))
    np.add.at(msum, new_id, colors * w[:, None])
    mtot = np.bincount(new_id, weights=w, minlength=m)
    palette = np.clip(np.rint(msum / mtot[:, None]), 0, 255).astype(np.uint8)
    return new_id, palette


def _drop_blend_clusters(
    cluster: np.ndarray, palette: np.ndarray, counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Remove minor clusters whose color lies between two larger clusters (JPEG/AA blends).

    Only used on the noisy (KMeans) path. Returns (remapped cluster ids, palette).
    """
    lab = _to_lab(palette)
    dist = deltaE_ciede2000(lab[:, None, :], lab[None, :, :])
    alive = np.ones(len(palette), dtype=bool)
    total = counts.sum()
    for c in np.argsort(counts, kind="stable"):
        if counts[c] > BLEND_MAX_SHARE * total:
            break
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
