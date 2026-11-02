"""Deterministic synthetic test images (HxWx4 uint8).

The plain generators are non-anti-aliased (every pixel is exactly one palette color). The `aa_*`
generators draw the same kind of geometry at `factor`x resolution and box-filter it down, so edge
pixels carry the true sub-pixel edge position as a color (or alpha) blend.
"""
from __future__ import annotations

import io
import math
from collections.abc import Callable

import numpy as np
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab

MIN_PALETTE_DELTA_E = 20.0
_MAX_PALETTE_TRIES = 10_000


def solid(size: int, rgba: tuple[int, int, int, int]) -> np.ndarray:
    px = np.empty((size, size, 4), dtype=np.uint8)
    px[:] = rgba
    return px


def to_png_bytes(pixels: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(pixels, "RGBA").save(buf, format="PNG")
    return buf.getvalue()


def _grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-center coordinates (x, y)."""
    ys, xs = np.mgrid[0:size, 0:size]
    return xs + 0.5, ys + 0.5


def _circle_mask(size: int, cx: float, cy: float, r: float) -> np.ndarray:
    xs, ys = _grid(size)
    return (xs - cx) ** 2 + (ys - cy) ** 2 <= r**2


def _lab(rgb: tuple[int, int, int] | np.ndarray) -> np.ndarray:
    return rgb2lab(np.asarray(rgb, dtype=float).reshape(1, 1, 3) / 255.0).reshape(3)


def _separated_palette(rng: np.random.Generator, n: int) -> list[tuple[int, int, int]]:
    """n colors with pairwise CIEDE2000 >= MIN_PALETTE_DELTA_E, via rejection sampling."""
    colors: list[tuple[int, int, int]] = []
    labs: list[np.ndarray] = []
    for _ in range(_MAX_PALETTE_TRIES):
        if len(colors) == n:
            return colors
        rgb = tuple(int(v) for v in rng.integers(0, 256, size=3))
        lab = _lab(rgb)
        if all(
            deltaE_ciede2000(lab, other) >= MIN_PALETTE_DELTA_E for other in labs
        ):
            colors.append(rgb)
            labs.append(lab)
    if len(colors) == n:
        return colors
    raise ValueError(f"could not find {n} colors with deltaE >= {MIN_PALETTE_DELTA_E}")


def _triangle_mask(size: int, pts: np.ndarray) -> np.ndarray:
    xs, ys = _grid(size)
    (x0, y0), (x1, y1), (x2, y2) = pts

    def side(ax, ay, bx, by):
        return (bx - ax) * (ys - ay) - (by - ay) * (xs - ax)

    d0, d1, d2 = side(x0, y0, x1, y1), side(x1, y1, x2, y2), side(x2, y2, x0, y0)
    neg = (d0 < 0) | (d1 < 0) | (d2 < 0)
    pos = (d0 > 0) | (d1 > 0) | (d2 > 0)
    return ~(neg & pos)


def flat_shapes(seed: int, size: int = 128, n_colors: int = 4) -> np.ndarray:
    """Background plus random rects/ellipses/triangles in n_colors total colors."""
    rng = np.random.default_rng(seed)
    palette = _separated_palette(rng, n_colors)
    px = solid(size, (*palette[0], 255))
    xs, ys = _grid(size)
    for _ in range(int(rng.integers(3, 7))):
        color = (*palette[int(rng.integers(1, n_colors))], 255) if n_colors > 1 else (*palette[0], 255)
        kind = int(rng.integers(0, 3))
        if kind == 0:
            x0, y0 = rng.uniform(0, size * 0.6, 2)
            w, h = rng.uniform(size * 0.15, size * 0.4, 2)
            mask = (xs >= x0) & (xs < x0 + w) & (ys >= y0) & (ys < y0 + h)
        elif kind == 1:
            cx, cy = rng.uniform(size * 0.2, size * 0.8, 2)
            rx, ry = rng.uniform(size * 0.08, size * 0.25, 2)
            mask = ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 <= 1
        else:
            mask = _triangle_mask(size, rng.uniform(0, size, (3, 2)))
        px[mask] = color
    return px


def circle(size: int = 128, cx: float = 64, cy: float = 64, r: float = 40) -> np.ndarray:
    px = solid(size, (255, 255, 255, 255))
    px[_circle_mask(size, cx, cy, r)] = (220, 40, 40, 255)
    return px


def ring_in_ring(size: int = 128) -> np.ndarray:
    """Three concentric colored rings on a background."""
    px = solid(size, (255, 255, 255, 255))
    c = size / 2
    for frac, color in ((0.45, (200, 30, 30, 255)), (0.32, (30, 160, 60, 255)), (0.18, (30, 60, 200, 255))):
        px[_circle_mask(size, c, c, size * frac)] = color
    return px


def transparent_logo(size: int = 128) -> np.ndarray:
    """Opaque shape on an alpha-0 background."""
    px = solid(size, (0, 0, 0, 0))
    px[_circle_mask(size, size / 2, size / 2, size * 0.3)] = (30, 90, 200, 255)
    return px


# ---------------------------------------------------------------- anti-aliased images

AA_FACTOR = 8  # default supersampling: coverage resolves to 1/64 of a pixel
HALF_PLANE_FACTOR = 32  # straight-edge tests need ~0.01 px coverage resolution


def supersample(
    draw_fn: Callable[[int], np.ndarray], size: int, factor: int = AA_FACTOR
) -> np.ndarray:
    """Box-filter `draw_fn(size * factor)` (an RGBA image) down to size x size.

    Colors are averaged premultiplied by alpha, so a half-covered pixel over a transparent
    background keeps the shape color with alpha ~128.
    """
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    hi = draw_fn(size * factor).astype(np.float64)
    alpha = hi[..., 3:4] / 255.0
    blocks = np.concatenate([hi[..., :3] * alpha, alpha], axis=2)
    mean = blocks.reshape(size, factor, size, factor, 4).mean(axis=(1, 3))
    a = mean[..., 3:4]
    rgb = np.divide(mean[..., :3], a, out=np.zeros_like(mean[..., :3]), where=a > 0)
    out = np.concatenate([rgb, a * 255.0], axis=2)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def aa_disc(
    size: int = 128,
    cx: float = 64.3,
    cy: float = 63.7,
    r: float = 40.4,
    fg: tuple[int, int, int, int] = (220, 40, 40, 255),
    bg: tuple[int, int, int, int] = (255, 255, 255, 255),
    factor: int = AA_FACTOR,
) -> np.ndarray:
    """Anti-aliased disc at a sub-pixel centre and radius."""

    def draw(n: int) -> np.ndarray:
        k = n / size
        px = solid(n, bg)
        px[_circle_mask(n, cx * k, cy * k, r * k)] = fg
        return px

    return supersample(draw, size, factor)


def aa_shapes(seed: int, size: int = 128, factor: int = AA_FACTOR) -> np.ndarray:
    """`flat_shapes(seed)` drawn at `factor`x and box-filtered down (same shapes, scaled)."""
    return supersample(lambda n: flat_shapes(seed, size=n), size, factor)


def aa_half_plane(
    size: int = 32,
    x0: float = 10.3,
    angle_deg: float = 0.0,
    left: tuple[int, int, int, int] = (255, 255, 255, 255),
    right: tuple[int, int, int, int] = (30, 60, 200, 255),
) -> np.ndarray:
    """Straight edge through (x0, size / 2), tilted `angle_deg` from vertical.

    Points with (x - x0) cos(a) - (y - size / 2) sin(a) >= 0 get `right`, the rest `left`; at
    angle 0 that is the vertical edge x = x0.
    """
    t = math.radians(angle_deg)

    def draw(n: int) -> np.ndarray:
        k = n / size
        xs, ys = _grid(n)
        side = (xs / k - x0) * math.cos(t) - (ys / k - size / 2) * math.sin(t) >= 0
        px = solid(n, left)
        px[side] = right
        return px

    return supersample(draw, size, HALF_PLANE_FACTOR)
