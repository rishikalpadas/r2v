"""Per-curve edge-noise estimate for dense boundary samples.

The noise of a curve is the robust spread of its raw samples across the edge: the residuals of
the samples against a strongly smoothed copy, measured along the copy's normals. Real shape
(curvature, corners) is what the strong smoothing keeps; jitter from JPEG blocks, sensor noise or
pixel quantisation is what it removes. Corners are where the smoothed copy cuts inside the true
shape, so samples next to strong corners are left out, and the spread is measured with the median
absolute deviation so the few residuals that remain from shape detail do not inflate it.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d

from engine.geometry.corners import find_corners, samples_at_scale, smooth_dense
from engine.types import Point

NOISE_FLOOR_PX = 0.15  # below this, sample jitter is beneath what the fit tolerance resolves
NOISE_SMOOTH_SIGMA = 4.0  # samples; removes jitter while keeping shape detail of ~10 px and up
MIN_NOISE_SAMPLES = 16  # shorter curves have too few residuals for a spread estimate
CORNER_PROBE_SIGMA = 2.0  # samples; light smoothing so strong corners are found despite noise
CORNER_PROBE_WINDOW = 3  # samples
STRONG_CORNER_DEG = 40.0  # turning angle at the probe scale: a 90 deg corner reads ~52 deg
CORNER_EXCLUDE = 3  # samples dropped each side of a strong corner; the MAD absorbs the rest
_MAD_TO_STD = 1.4826  # median absolute deviation of a normal distribution -> its std
_KERNEL_RADIUS = 4.0  # gaussian_filter1d default truncation, in sigmas


def _residual_scale(sigma: float) -> float:
    """Std of (white noise - its Gaussian smoothing) per unit noise std."""
    radius = int(_KERNEL_RADIUS * sigma + 0.5)
    impulse = np.zeros(2 * radius + 1)
    impulse[radius] = 1.0
    kernel = gaussian_filter1d(impulse, sigma, mode="constant")
    centre = kernel[radius]
    return float(np.sqrt(1.0 - 2.0 * centre + float(kernel @ kernel)))


def _normals(arr: np.ndarray, closed: bool) -> np.ndarray:
    if closed:
        tangent = (np.roll(arr, -1, axis=0) - np.roll(arr, 1, axis=0)) / 2.0
    else:
        tangent = np.gradient(arr, axis=0)
    length = np.hypot(tangent[:, 0], tangent[:, 1])
    length[length < 1e-12] = 1.0
    return np.column_stack([-tangent[:, 1], tangent[:, 0]]) / length[:, None]


def _usable(n: int, corners: Sequence[int], closed: bool, scale: float = 1.0) -> np.ndarray:
    """Mask of samples away from strong corners (and, for open curves, from the end points)."""
    keep = np.ones(n, bool)
    exclude = samples_at_scale(CORNER_EXCLUDE, scale)
    for c in corners:
        idx = np.arange(c - exclude, c + exclude + 1)
        keep[idx % n if closed else idx[(idx >= 0) & (idx < n)]] = False
    if not closed:
        edge = int(NOISE_SMOOTH_SIGMA * scale)  # the smoothed copy is pinned near the ends
        keep[:edge] = keep[n - edge :] = False
    return keep


def estimate_noise(dense: Sequence[Point], closed: bool, scale: float = 1.0) -> float:
    """Robust std (processing px) of the samples' normal jitter, at least NOISE_FLOOR_PX.

    `dense` is a raw (unsmoothed) dense sample sequence in the `corners.dense_samples` layout,
    ideally already moved to sub-pixel edge positions. The sample counts and sigmas above are
    at scale 1 and are converted to `scale` (one sample per processing pixel), so they keep
    covering the same original-pixel length.
    """
    n = len(dense)
    min_samples = samples_at_scale(MIN_NOISE_SAMPLES, scale)
    if n < min_samples:
        return NOISE_FLOOR_PX
    raw = tuple((float(x), float(y)) for x, y in dense)
    arr = np.asarray(raw, np.float64)
    sigma = NOISE_SMOOTH_SIGMA * scale
    smooth = np.asarray(smooth_dense(raw, closed, sigma), np.float64)
    probe = smooth_dense(raw, closed, CORNER_PROBE_SIGMA * scale)
    window = samples_at_scale(CORNER_PROBE_WINDOW, scale)
    strong = find_corners(probe, closed, STRONG_CORNER_DEG, window)
    if not closed:
        strong = strong[1:-1]  # open-curve ends are reported as corners; they are handled below
    keep = _usable(n, strong, closed, scale)
    if keep.sum() < min_samples:
        return NOISE_FLOOR_PX
    resid = np.einsum("ij,ij->i", arr - smooth, _normals(smooth, closed))[keep]
    spread = _MAD_TO_STD * float(np.median(np.abs(resid - np.median(resid))))
    return max(NOISE_FLOOR_PX, spread / _residual_scale(sigma))
