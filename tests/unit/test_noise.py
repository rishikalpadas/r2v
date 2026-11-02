"""Per-curve edge-noise estimate."""
import math

import numpy as np
import pytest

from engine.geometry.noise import NOISE_FLOOR_PX, estimate_noise


def _circle(n=250, r=40.0, sigma=0.0, seed=0):
    """Closed ring of ~unit-spaced samples with Gaussian jitter along the normal."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    rad = r + rng.normal(0.0, sigma, n)
    return tuple(zip(64 + rad * np.cos(t), 64 + rad * np.sin(t)))


def _rect(w=60, h=40, sigma=0.0, seed=0):
    """Closed axis-aligned rectangle outline, unit-spaced, jittered along each side's normal."""
    rng = np.random.default_rng(seed)
    pts = [(x + 0.5, 0.0) for x in range(w)] + [(float(w), y + 0.5) for y in range(h)]
    pts += [(w - x - 0.5, float(h)) for x in range(w)] + [(0.0, h - y - 0.5) for y in range(h)]
    normals = [(0, 1)] * w + [(1, 0)] * h + [(0, 1)] * w + [(1, 0)] * h
    noise = rng.normal(0.0, sigma, len(pts))
    return tuple((x + e * nx, y + e * ny) for (x, y), (nx, ny), e in zip(pts, normals, noise))


def _line(n=120, sigma=0.0, seed=0, angle_deg=23.0):
    rng = np.random.default_rng(seed)
    t = math.radians(angle_deg)
    d, nrm = np.array([math.cos(t), math.sin(t)]), np.array([-math.sin(t), math.cos(t)])
    s = np.arange(n, dtype=float)
    pts = s[:, None] * d + rng.normal(0.0, sigma, n)[:, None] * nrm
    return tuple(map(tuple, pts))


@pytest.mark.parametrize("sigma", [0.3, 0.8])
@pytest.mark.parametrize("seed", [0, 1])
def test_estimate_matches_known_jitter(sigma, seed):
    for dense, closed in ((_circle(sigma=sigma, seed=seed), True),
                          (_rect(sigma=sigma, seed=seed), True),
                          (_line(sigma=sigma, seed=seed), False)):
        assert estimate_noise(dense, closed) == pytest.approx(sigma, rel=0.3)


def test_clean_edges_estimate_the_floor():
    for dense, closed in ((_circle(), True), (_rect(), True), (_line(), False)):
        assert estimate_noise(dense, closed) <= NOISE_FLOOR_PX


def test_short_curves_estimate_the_floor():
    assert estimate_noise(_line(n=10, sigma=1.0), False) == NOISE_FLOOR_PX
    assert estimate_noise((), True) == NOISE_FLOOR_PX
