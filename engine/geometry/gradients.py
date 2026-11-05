"""Linear gradient detection and fitting for multi-tone gradient regions."""
from __future__ import annotations

from typing import Mapping
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csg
import scipy.ndimage as ndi
from skimage.color import deltaE_ciede2000

from engine.stages.segment_regions import (
    QUIET_GRADIENT,
    _edge_arrays,
    _rgb_to_lab,
    contrast_image,
    lab_image,
)
from engine.types import GradientStop, LinearGradient


def fit_gradient_axis(
    source_rgb: np.ndarray, mask: np.ndarray, scale: float = 1.0
) -> tuple[float, float, float, float, tuple[int, int, int], tuple[int, int, int]] | None:
    """Fit a 2D linear gradient model to source_rgb within boolean mask.

    Returns (x1, y1, x2, y2, rgb_start, rgb_end) in processing-pixel units,
    or None if the region does not exhibit a significant linear gradient.
    """
    ys, xs = np.where(mask)
    if len(xs) < 20:
        return None

    # Subsample for speed if the region is very large
    if len(xs) > 10000:
        sub = np.random.choice(len(xs), 10000, replace=False)
        xs_sub = xs[sub].astype(np.float64)
        ys_sub = ys[sub].astype(np.float64)
        colors_sub = source_rgb[ys[sub], xs[sub], :3].astype(np.float64)
    else:
        xs_sub = xs.astype(np.float64)
        ys_sub = ys.astype(np.float64)
        colors_sub = source_rgb[ys, xs, :3].astype(np.float64)

    # Luminance as standard perceptual proxy
    lum = 0.299 * colors_sub[:, 0] + 0.587 * colors_sub[:, 1] + 0.114 * colors_sub[:, 2]

    x_mean = xs_sub.mean()
    y_mean = ys_sub.mean()
    lum_mean = lum.mean()

    dx = xs_sub - x_mean
    dy = ys_sub - y_mean
    dlum = lum - lum_mean

    # Solve [dx, dy] @ [a, b] = dlum via least squares
    A = np.stack([dx, dy], axis=1)
    res, _, _, _ = np.linalg.lstsq(A, dlum, rcond=None)
    a_coeff, b_coeff = float(res[0]), float(res[1])
    norm = np.hypot(a_coeff, b_coeff)
    if norm < 1e-5:
        return None

    vx, vy = a_coeff / norm, b_coeff / norm
    proj = dx * vx + dy * vy
    p_min = float(np.percentile(proj, 2.0))
    p_max = float(np.percentile(proj, 98.0))

    # Reject if spatial extent along gradient axis is too thin
    if p_max - p_min < 4.0 * scale:
        return None

    # Extract start and end colors
    start_mask = proj <= (p_min + 0.15 * (p_max - p_min))
    end_mask = proj >= (p_max - 0.15 * (p_max - p_min))

    if not start_mask.any() or not end_mask.any():
        return None

    start_rgb = tuple(int(np.clip(c, 0, 255)) for c in np.median(colors_sub[start_mask], axis=0))
    end_rgb = tuple(int(np.clip(c, 0, 255)) for c in np.median(colors_sub[end_mask], axis=0))

    # Verify that there is a genuine perceptual color difference
    start_lab = _rgb_to_lab(np.array([start_rgb]))
    end_lab = _rgb_to_lab(np.array([end_rgb]))
    if float(deltaE_ciede2000(start_lab, end_lab)[0]) < 4.0:
        return None

    x1 = float(x_mean + p_min * vx)
    y1 = float(y_mean + p_min * vy)
    x2 = float(x_mean + p_max * vx)
    y2 = float(y_mean + p_max * vy)

    return (x1, y1, x2, y2, start_rgb, end_rgb)


def detect_and_merge_gradients(
    labels: np.ndarray,
    palette_rgb: np.ndarray,
    source_rgb: np.ndarray,
    scale: float,
    max_delta_e: float = 15.0,
    min_quiet_share: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, dict[int, LinearGradient]]:
    """Detect adjacent segmented bands that form a linear gradient and merge their geometry.

    Returns (merged_labels, palette_rgb, dict_of_gradients).
    """
    n = len(palette_rgb)
    if n < 2:
        return labels, palette_rgb, {}

    lab = lab_image(source_rgb)
    contrast, _ = contrast_image(lab, scale, labels)

    a, b, c = _edge_arrays(labels, contrast)
    if len(a) == 0:
        return labels, palette_rgb, {}

    quiet = c < QUIET_GRADIENT
    lo, hi = np.minimum(a, b).astype(np.int64), np.maximum(a, b).astype(np.int64)
    keys, inverse, counts = np.unique(lo * n + hi, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse.ravel(), weights=quiet.astype(np.float64))
    quiet_share = sums / counts

    pi, pj = keys // n, keys % n
    palette_lab = _rgb_to_lab(palette_rgb)
    dE = deltaE_ciede2000(palette_lab[pi], palette_lab[pj])

    # Filter quiet candidate boundary edges
    min_count = round(16 * scale)
    cand_mask = (quiet_share >= min_quiet_share) & (counts >= min_count) & (dE < max_delta_e)
    cand_i = pi[cand_mask]
    cand_j = pj[cand_mask]

    if len(cand_i) == 0:
        return labels, palette_rgb, {}

    adj = sp.coo_matrix((np.ones(len(cand_i)), (cand_i, cand_j)), shape=(n, n))
    adj = adj + adj.T
    n_comp, comp_labels = csg.connected_components(adj, directed=False)

    out_labels = labels.copy()
    gradients: dict[int, LinearGradient] = {}

    for c_id in range(n_comp):
        members = np.where(comp_labels == c_id)[0]
        if len(members) < 2:
            continue

        # Total color variation across members must indicate a gradient
        sub_dE = deltaE_ciede2000(palette_lab[members, None, :], palette_lab[None, members, :])
        if sub_dE.max() < 4.0:
            continue

        group_mask = np.isin(out_labels, members)
        fit = fit_gradient_axis(source_rgb, group_mask, scale)
        if fit is None:
            continue

        # Absorb adjacent boundary bands into this gradient group
        # (prevents isolated slivers on letter curves where outline contrast suppresses quiet_share)
        dilated = ndi.binary_dilation(group_mask)
        touching_labels = np.unique(out_labels[dilated & ~group_mask])
        for t_id in touching_labels:
            if t_id < 0 or t_id in gradients:
                continue
            t_dE = deltaE_ciede2000(palette_lab[[t_id]], palette_lab[members]).min()
            if t_dE < max_delta_e:
                t_mask = out_labels == t_id
                combined_mask = group_mask | t_mask
                new_fit = fit_gradient_axis(source_rgb, combined_mask, scale)
                if new_fit is not None:
                    group_mask = combined_mask
                    fit = new_fit
                    members = np.append(members, t_id)

        x1, y1, x2, y2, start_rgb, end_rgb = fit
        target_id = int(members[0])

        # Merge region labels into target_id so internal dividing lines vanish
        out_labels[group_mask] = target_id

        grad = LinearGradient(
            id=f"grad_{target_id}",
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            stops=(
                GradientStop(0.0, start_rgb),
                GradientStop(1.0, end_rgb),
            ),
        )
        gradients[target_id] = grad

    return out_labels, palette_rgb, gradients
