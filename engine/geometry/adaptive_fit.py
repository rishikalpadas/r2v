"""Noise-adaptive fitting of one boundary curve.

Fixed fitting parameters suit clean edges; on JPEG art the sub-pixel samples jitter, single-scale
corner detection finds fake corners every few pixels and a tight tolerance makes Beziers follow
the jitter. Here every step is scaled to the curve's own measured noise (`estimate_noise`):

1. corners must persist from the fine scale up to a noise-dependent coarse scale;
2. each corner-to-corner run is smoothed on its own (corners pinned) with sigma >= 2 x noise;
3. the tolerance is at least 2.5 x noise and spans are chosen by noise-normalised MDL;
4. a gentle corner is dropped (its two runs merged) when, judged by MDL on the samples around
   it, one smooth fit explains them better than two fits meeting at the corner;
5. a corner between two straight runs moves to where lines through the whole runs meet;
6. joins that are not corners (the splits of a corner-free ring) are made G1.

All lengths are processing pixels; one dense sample is one processing pixel of boundary.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from engine.geometry.corner_placement import (
    CORNER_SNAP_PX,
    SideLines,
    collapse_behind,
    line_intersection,
    place_corners,
)
from engine.geometry.corners import find_corners_multiscale, smooth_between, smooth_dense
from engine.geometry.noise import estimate_noise
from engine.geometry.primitives import detect_primitive
from engine.geometry.segment_fit import align_joins, chain_cost, fit_span, snap_axis
from engine.presets import FitParams
from engine.types import BoundaryCurve, FittedCurve, Line, Point, Segment

SMOOTH_PER_NOISE = 2.0  # fit smoothing sigma (samples) per px of noise: halves white jitter
TOL_PER_NOISE = 2.5  # fit tolerance per px of noise: ~99% of Gaussian jitter lies within it
COARSE_PER_SIGMA = 2.0  # coarse corner scale is at least this x the fine smoothing sigma
COARSE_PER_NOISE = 3.0  # ... and this x the noise: jitter bumps are flat at 3 sigma
MERGE_MAX_ANGLE_DEG = 60.0  # coarse turn above which a corner is never merged away
# Corner placement fits longer side lines than on clean input: under JPEG noise 4 samples give
# an unstable direction (rotated-rect corners up to 0.85 px off), while much longer lines bend
# toward curved sides (a heart's cusp 1.1 px off at 12 samples). 8 measured best for both.
# The hit may lie behind the blended samples next to the corner; those are pulled onto it.
SIDE_LINES = (SideLines(gap=1, samples=8, allow_behind=True),)
BEHIND_REACH = 4  # samples each side of a corner checked for lying past it
LINE_CORNER_TRIM = 2  # samples next to a corner left out of a straight side's line fit
MIN_LINE_SIDE_SAMPLES = 6  # shorter sides keep the corner `place_corners` gave them
JUDGE_REACH = 8  # samples each side of a corner that `_corner_is_fake` compares models on
MIN_JUDGE_SAMPLES = 3  # a corner closer than this to its neighbour is kept unjudged


@dataclass(frozen=True)
class _Curve:
    """Per-curve fitting inputs: samples, noise-adapted params and the noise itself."""

    raw: tuple[Point, ...]  # sub-pixel (or crack-grid) samples, unsmoothed
    closed: bool
    params: FitParams  # tolerance_px / smooth_sigma already adapted to the noise
    judge: FitParams  # params with a noise-limited tolerance, for comparing alternative fits
    noise: float


def _run(seq: Sequence[Point], a: int, b: int) -> tuple[Point, ...]:
    """Samples a..b inclusive; b may run past the end of a closed ring."""
    n = len(seq)
    return tuple(seq[i % n] for i in range(a, b + 1))


def _bounds(corners: Sequence[int], n: int, closed: bool) -> list[tuple[int, int]]:
    """(start, end) sample indices of every corner-to-corner run, unwrapped for closed rings."""
    spans = list(zip(corners, corners[1:]))
    if closed:
        spans.append((corners[-1], corners[0] + n))
    return spans


def _beats(curve: _Curve, data: Sequence[Point], a: Sequence[Segment], b: Sequence[Segment]) -> bool:
    """True when chain `a` explains `data` more cheaply than chain `b` (noise-normalised MDL)."""
    cost_a = chain_cost(a, data, curve.judge, curve.noise)
    return cost_a < chain_cost(b, data, curve.judge, curve.noise)


def _corner_is_fake(curve: _Curve, pinned: Sequence[Point], c: int, lo: int, hi: int) -> bool:
    """MDL test of corner c on the samples lo..hi around it (unwrapped indices).

    The corner model is two fits meeting at the placed corner, the smooth model one fit through
    it, both with the noise-limited `judge` tolerance, both scored by `chain_cost` against the
    same unsmoothed samples (corner sample left raw). A real corner costs the smooth model
    residual or extra pieces where it has to turn sharply; a fake one costs the corner model
    parameters it does not need. Judging only near the corner keeps the comparison about the
    corner rather than about how the rest of two long runs happened to split into pieces.
    """
    n = len(curve.raw)
    sigma = curve.params.smooth_sigma
    data = list(_run(pinned, lo, hi))
    data[c - lo] = curve.raw[c % n]
    left = smooth_dense(_run(pinned, lo, c), False, sigma)
    right = smooth_dense(_run(pinned, c, hi), False, sigma)
    cornered = fit_span(left, curve.judge, curve.noise) + fit_span(right, curve.judge, curve.noise)
    smooth = fit_span(smooth_dense(tuple(data), False, sigma), curve.judge, curve.noise)
    return _beats(curve, data, smooth, cornered)


def _try_merge(
    curve: _Curve, pinned: list[Point], seq: list[Point], corners: list[int], k: int
) -> None:
    """Drop corners[k] when `_corner_is_fake`, joining the runs on either side of it.

    `pinned` holds the raw samples with every remaining corner placed; `seq` the samples being
    fitted. On success the union run is re-smoothed without corners[k] pinned.
    """
    n = len(seq)
    c = corners[k]
    a = corners[k - 1] if k > 0 else corners[-1] - n
    b = corners[k + 1] if k + 1 < len(corners) else corners[0] + n
    lo, hi = max(a, c - JUDGE_REACH), min(b, c + JUDGE_REACH)
    if min(c - lo, hi - c) < MIN_JUDGE_SAMPLES or not _corner_is_fake(curve, pinned, c, lo, hi):
        return
    pinned[c % n] = curve.raw[c % n]
    corners.pop(k)
    if not corners:
        return  # the ring is now corner-free and is fitted as such
    smoothed = smooth_dense(_run(pinned, a, b), False, curve.params.smooth_sigma)
    for i, p in zip(range(a + 1, b), smoothed[1:-1]):
        seq[i % n] = p


def _merge_spans(
    curve: _Curve, pinned: list[Point], seq: list[Point], corners: list[int],
    angles: dict[int, float],
) -> None:
    """Greedily drop fake corners among the gentle ones, gentlest first (see `_try_merge`)."""
    lo, hi = (0, len(corners)) if curve.closed else (1, len(corners) - 1)
    candidates = sorted(
        (angles[c], c) for c in corners[lo:hi] if angles[c] < MERGE_MAX_ANGLE_DEG
    )
    for _, c in candidates:
        _try_merge(curve, pinned, seq, corners, corners.index(c))


def _side_line(curve: _Curve, a: int, b: int) -> np.ndarray | None:
    """Raw samples of run a..b without the corner-blended ones at its ends, if enough remain."""
    side = _run(curve.raw, a + LINE_CORNER_TRIM, b - LINE_CORNER_TRIM)
    return np.asarray(side, np.float64) if len(side) >= MIN_LINE_SIDE_SAMPLES else None


def _sharpen_line_corners(
    curve: _Curve, runs: list[tuple[Segment, ...]], bounds: Sequence[tuple[int, int]], snap: float
) -> list[tuple[Segment, ...]]:
    """Move each corner between two single-Line runs to where lines through their samples meet.

    Whole straight sides pin a polygon corner far better than the few samples next to it. The
    corner moves only if the lines cross within `snap` of it (open-curve end points never move).
    """
    out = list(runs)
    joins = range(len(out)) if curve.closed else range(1, len(out))
    for j in joins:
        i = j - 1
        prev, nxt = out[i], out[j]
        straight = len(prev) == len(nxt) == 1 and isinstance(prev[0], Line)
        if not (straight and isinstance(nxt[0], Line)):
            continue
        side_a, side_b = _side_line(curve, *bounds[i]), _side_line(curve, *bounds[j])
        if side_a is None or side_b is None:
            continue
        hit = line_intersection(side_a, side_b)
        if hit is None or math.dist(hit, prev[0].p1) > snap:
            continue
        out[i] = (Line(prev[0].p0, hit),)
        out[j] = (Line(hit, nxt[0].p1),)
    return out


def _fit_runs(
    curve: _Curve, seq: Sequence[Point], corners: Sequence[int], snap: float
) -> tuple[Segment, ...]:
    """The chain of every corner-to-corner run, starting at the first corner."""
    bounds = _bounds(corners, len(seq), curve.closed)
    runs = [fit_span(_run(seq, a, b), curve.params, curve.noise) for a, b in bounds]
    return tuple(s for run in _sharpen_line_corners(curve, runs, bounds, snap) for s in run)


def _cornerless(curve: _Curve, spans: int) -> tuple[tuple[Segment, ...], list[int]]:
    """Fit a corner-free closed ring as `spans` runs; returns the chain and its join indices."""
    ring = smooth_dense(curve.raw, True, curve.params.smooth_sigma)
    n = len(ring)
    seq = (*ring, ring[0])
    breaks = [n * k // spans for k in range(spans + 1)]
    chain: list[Segment] = []
    joins: list[int] = []
    for a, b in zip(breaks, breaks[1:]):
        joins.append(len(chain))
        chain.extend(fit_span(seq[a : b + 1], curve.params, curve.noise))
    return tuple(chain), joins


def _adapt(params: FitParams, noise: float) -> FitParams:
    return replace(
        params,
        tolerance_px=max(params.tolerance_px, TOL_PER_NOISE * noise),
        smooth_sigma=max(params.smooth_sigma, SMOOTH_PER_NOISE * noise),
    )


def fit_curve_adaptive(
    curve: BoundaryCurve,
    samples: tuple[Sequence[Point], Sequence[Point]],
    params: FitParams,
    scale: float,
    corner_window: int,
    cornerless_spans: int,
) -> FittedCurve:
    """Fit one curve with smoothing, corners, tolerance and model cost scaled to its noise.

    `samples` is (raw, plain): sub-pixel dense samples and their crack-grid originals (equal
    without sub-pixel input). `params` are in processing pixels. Open-curve end points (shared
    junctions) are kept exactly; a closed chain starts at a corner and closes exactly.
    """
    raw, plain = tuple(samples[0]), tuple(samples[1])
    closed, n = curve.closed, len(raw)
    noise = estimate_noise(raw, closed)
    adapted = _adapt(params, noise)
    judge = replace(adapted, tolerance_px=TOL_PER_NOISE * noise)
    c = _Curve(raw, closed, adapted, judge, noise)
    coarse = max(COARSE_PER_SIGMA * params.smooth_sigma, COARSE_PER_NOISE * noise)
    found = find_corners_multiscale(
        raw, closed, params.corner_angle_deg, params.smooth_sigma, coarse, corner_window
    )
    corners = list(found.indices)
    angles = dict(zip(corners, found.coarse_angles))
    inner = corners if closed else corners[1:-1]
    seq = place_corners(
        smooth_between(raw, corners, closed, c.params.smooth_sigma),
        inner, corners, (raw, plain), curve.points, closed, scale, SIDE_LINES,
    )
    seq = collapse_behind(seq, inner, corners, closed, BEHIND_REACH)
    pinned = list(raw)
    for i in inner:
        pinned[i] = seq[i]
    _merge_spans(c, pinned, seq, corners, angles)
    joins: list[int] = []
    if closed and not corners:
        chain, joins = _cornerless(c, cornerless_spans)
    else:
        chain = _fit_runs(c, seq, corners, CORNER_SNAP_PX * scale)
    chain = align_joins(snap_axis(chain, c.params), joins)
    primitive = None
    if closed:
        ring = smooth_dense(raw, True, c.params.smooth_sigma)
        primitive = detect_primitive(ring, tuple(corners), scale)
    return FittedCurve(curve.id, chain, primitive=primitive)
