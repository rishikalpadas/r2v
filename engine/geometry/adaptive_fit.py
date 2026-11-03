"""Noise-adaptive fitting of one boundary curve.

Fixed fitting parameters suit clean edges; on JPEG art the sub-pixel samples jitter, single-scale
corner detection finds fake corners every few pixels and a tight tolerance makes Beziers follow
the jitter. Here every step is scaled to the curve's own measured noise (`estimate_noise`):

0. samples next to a junction stay on the crack grid (their pixels mix in a third region);
1. corners must persist from the fine scale up to a noise-dependent coarse scale, and are not
   kept right beside a junction or on top of the previous corner;
2. each corner-to-corner run is smoothed on its own (corners pinned) with sigma >= 2 x noise;
3. the tolerance is at least 2.5 x noise and spans are chosen by noise-normalised MDL;
4. a gentle corner (coarse turn below that of any right angle) is dropped, its two runs merged,
   when by MDL on the samples around it one smooth fit beats two fits meeting at the corner;
5. a corner between two straight runs moves to where lines through the whole runs meet;
6. joins that are not corners (the splits of a corner-free ring) are made G1.

All lengths are processing pixels; one dense sample is one processing pixel of boundary.
Every sample count below is at scale 1 (one sample per original-image pixel) and is converted
with `samples_at_scale`, so it keeps covering the same original-pixel length on enhanced input.
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
    corner_neighbours,
    line_intersection,
    place_corners,
)
from engine.geometry.corners import (
    TRACK_RADIUS,
    even_breaks,
    find_corners_multiscale,
    samples_at_scale,
    smooth_between,
    smooth_dense,
)
from engine.geometry.noise import estimate_noise
from engine.geometry.primitives import detect_primitive
from engine.geometry.segment_fit import align_joins, chain_cost, fit_span, snap_axis
from engine.presets import FitParams
from engine.types import BoundaryCurve, FittedCurve, Line, Point, Segment

SMOOTH_PER_NOISE = 2.0  # fit smoothing sigma (samples) per px of noise: halves white jitter
TOL_PER_NOISE = 2.5  # fit tolerance per px of noise: ~99% of Gaussian jitter lies within it
COARSE_PER_SIGMA = 2.0  # coarse corner scale is at least this x the fine smoothing sigma
COARSE_PER_NOISE = 3.0  # ... and this x the noise: jitter bumps are flat at 3 sigma
# Coarse turn above which a corner is never merged away. The coarse response is ~0.65 x the
# true turn: an ideal 90 deg corner reads 61 deg, anti-aliased / soft ones 52-58 deg, a 60 deg
# turn 38 deg. 40 keeps every right angle well clear of the merge pass.
MERGE_MAX_ANGLE_DEG = 40.0
# Corner placement fits longer side lines than on clean input: under JPEG noise 4 samples give
# an unstable direction (rotated-rect corners up to 0.85 px off). Lines that meet inside the raw
# corner have bent toward curved sides (sharp tips between arcs); those retry with the short
# clean-input lines before falling back to the grid snap.
# The hit may lie behind the blended samples next to the corner; those are pulled onto it.
SIDE_LINES = (
    SideLines(gap=1, samples=8, allow_behind=True, outward_only=True),
    SideLines(allow_behind=True, outward_only=True),
)
BEHIND_REACH = 4  # samples each side of a corner checked for lying past it
LINE_CORNER_TRIM = 2  # samples next to a corner left out of a straight side's line fit
MIN_LINE_SIDE_SAMPLES = 6  # shorter sides keep the corner `place_corners` gave them
JUDGE_REACH = 8  # samples each side of a corner that `_corner_is_fake` compares models on
MIN_JUDGE_SAMPLES = 3  # a corner closer than this to its neighbour is kept unjudged
# Next to a junction a sample's pixels belong to a third region, so its sub-pixel offset reads
# the wrong colour pair and can push the curve across its neighbour (1jpg__scan: inter-curve
# crossings 44 -> 20 with these samples kept on the crack grid; adaptive off has 23).
JUNCTION_GRID_SAMPLES = 2
# With sub-pixel samples, Bezier end tangents are read over 3 samples, not 1: the sample beside
# a placed corner is often a blended one off the side, and a one-sample tangent then smooths the
# corner away (1.jpg: right-angle corners emitted with < 30 deg breaks 9 -> 4). Exact crack-grid
# samples keep the one-sample tangent (3 costs flat_shapes(3) cutout SSIM 0.952 -> 0.949), and
# their grid snaps may go inward: there the crack vertex is the exact pixel corner.
END_TANGENT_SAMPLES = 3
COINCIDENT_PX = 0.5  # corners placed closer than this to the previous corner are one corner


@dataclass(frozen=True)
class _Curve:
    """Per-curve fitting inputs: samples, noise-adapted params and the noise itself."""

    raw: tuple[Point, ...]  # sub-pixel (or crack-grid) samples, unsmoothed
    closed: bool
    params: FitParams  # tolerance_px / smooth_sigma already adapted to the noise
    judge: FitParams  # params with a noise-limited tolerance, for comparing alternative fits
    noise: float
    subpixel: bool = False  # samples were moved off the crack grid (anti-aliased input)
    scale: float = 1.0  # processing scale: sample counts are converted with it

    def count(self, samples: int) -> int:
        """A sample count at scale 1 as the count covering the same length here."""
        return samples_at_scale(samples, self.scale)

    @property
    def tangent_samples(self) -> int:
        """Bezier end tangents are read over this many samples (see END_TANGENT_SAMPLES)."""
        return self.count(END_TANGENT_SAMPLES) if self.subpixel else 1


def _fit(curve: _Curve, points: Sequence[Point], params: FitParams) -> tuple[Segment, ...]:
    return fit_span(points, params, curve.noise, curve.tangent_samples)


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

    The corner model is two fits meeting at the placed corner, the smooth model one fit of the
    samples with the raw (rounded) corner sample, both with the noise-limited `judge` tolerance.
    Both are scored by `chain_cost` against the same samples with the corner placed: the
    question is whether the edge really turns there, and the raw sample next to an anti-aliased
    corner is rounded off by the pixels, not by the shape. A real corner costs the smooth model
    residual or extra pieces; a fake one costs the corner model parameters it does not need.
    Judging only near the corner keeps the comparison about the corner rather than about how
    two long runs happened to split into pieces.
    """
    n = len(curve.raw)
    sigma = curve.params.smooth_sigma
    data = _run(pinned, lo, hi)
    unpinned = list(data)
    unpinned[c - lo] = curve.raw[c % n]
    left = smooth_dense(_run(pinned, lo, c), False, sigma)
    right = smooth_dense(_run(pinned, c, hi), False, sigma)
    cornered = _fit(curve, left, curve.judge) + _fit(curve, right, curve.judge)
    smooth = _fit(curve, smooth_dense(tuple(unpinned), False, sigma), curve.judge)
    return _beats(curve, data, smooth, cornered)


@dataclass(frozen=True)
class _Merge:
    """State of the merge pass: placed-raw samples, samples to fit, and the corners left."""

    pinned: tuple[Point, ...]  # raw samples with every remaining corner placed
    seq: tuple[Point, ...]  # the samples being fitted (smoothed between corners)
    corners: tuple[int, ...]


def _try_merge(curve: _Curve, state: _Merge, k: int) -> _Merge:
    """`state` without corners[k] when `_corner_is_fake`, else `state` unchanged.

    The union run is re-smoothed without corners[k] pinned. Re-smoothing brings back samples that
    `collapse_behind` had pulled onto the neighbouring corners, so they are collapsed again.
    """
    n = len(state.seq)
    corners = state.corners
    c = corners[k]
    a, b = corner_neighbours(corners, k, n, curve.closed)
    reach = curve.count(JUDGE_REACH)
    lo, hi = max(a, c - reach), min(b, c + reach)
    if min(c - lo, hi - c) < curve.count(MIN_JUDGE_SAMPLES):
        return state
    if not _corner_is_fake(curve, state.pinned, c, lo, hi):
        return state
    pinned = list(state.pinned)
    pinned[c % n] = curve.raw[c % n]
    left = corners[:k] + corners[k + 1 :]
    if not left:
        return _Merge(tuple(pinned), state.seq, left)  # now corner-free; fitted as such
    seq = list(state.seq)
    smoothed = smooth_dense(_run(pinned, a, b), False, curve.params.smooth_sigma)
    for i, p in zip(range(a + 1, b), smoothed[1:-1]):
        seq[i % n] = p
    inner = left if curve.closed else left[1:-1]
    ends = [i for i in {a % n, b % n} if i in inner]
    seq = collapse_behind(seq, ends, left, curve.closed, curve.count(BEHIND_REACH))
    return _Merge(tuple(pinned), tuple(seq), left)


def _merge_spans(curve: _Curve, state: _Merge, angles: dict[int, float]) -> _Merge:
    """Greedily drop fake corners among the gentle ones, gentlest first (see `_try_merge`)."""
    corners = state.corners
    lo, hi = (0, len(corners)) if curve.closed else (1, len(corners) - 1)
    candidates = sorted(
        (angles[c], c) for c in corners[lo:hi] if angles[c] < MERGE_MAX_ANGLE_DEG
    )
    for _, c in candidates:
        state = _try_merge(curve, state, state.corners.index(c))
    return state


def _side_line(curve: _Curve, a: int, b: int) -> np.ndarray | None:
    """Raw samples of run a..b without the corner-blended ones at its ends, if enough remain."""
    trim = curve.count(LINE_CORNER_TRIM)
    side = _run(curve.raw, a + trim, b - trim)
    return np.asarray(side, np.float64) if len(side) >= curve.count(MIN_LINE_SIDE_SAMPLES) else None


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


def _fit_run(
    points: Sequence[Point], params: FitParams, noise: float, tangent_samples: int = 1
) -> tuple[Segment, ...]:
    """`fit_span` of one corner-to-corner run, never as a zero-chord segment.

    A run whose ends coincide (a lone corner of a ring, or two corners placed on the same
    vertex) is split at the sample farthest from its end into two fits; one that never leaves
    its end point has no segments.
    """
    def fit(run: Sequence[Point]) -> tuple[Segment, ...]:
        return fit_span(run, params, noise, tangent_samples)

    if points[0] != points[-1]:
        return fit(points)
    start = np.asarray(points[0])
    far = int(np.argmax(np.hypot(*(np.asarray(points) - start).T)))
    if points[far] == points[0]:
        return ()
    return fit(points[: far + 1]) + fit(points[far:])


def _fit_runs(
    curve: _Curve, seq: Sequence[Point], corners: Sequence[int], snap: float
) -> tuple[Segment, ...]:
    """The chain of every corner-to-corner run, starting at the first corner."""
    bounds = _bounds(corners, len(seq), curve.closed)
    runs = [
        _fit_run(_run(seq, a, b), curve.params, curve.noise, curve.tangent_samples)
        for a, b in bounds
    ]
    return tuple(s for run in _sharpen_line_corners(curve, runs, bounds, snap) for s in run)


def _cornerless(curve: _Curve, spans: int) -> tuple[tuple[Segment, ...], list[int]]:
    """Fit a corner-free closed ring as `spans` runs; returns the chain and its join indices."""
    ring = smooth_dense(curve.raw, True, curve.params.smooth_sigma)
    n = len(ring)
    seq = (*ring, ring[0])
    breaks = even_breaks(n, spans)
    chain: list[Segment] = []
    joins: list[int] = []
    for a, b in zip(breaks, breaks[1:]):
        joins.append(len(chain))
        chain.extend(_fit(curve, seq[a : b + 1], curve.params))
    return tuple(chain), joins


def _clear_of_ends(corners: Sequence[int], n: int, closed: bool, k: int) -> list[int]:
    """Corners without the inner ones within `k` samples of an open curve's ends.

    The junction is already a vertex; a corner right next to it only adds a sliver or hairpin.
    """
    if closed:
        return list(corners)
    return [i for i in corners if i in (0, n - 1) or k <= i <= n - 1 - k]


def _place(
    curve: _Curve, plain: tuple[Point, ...], original: Sequence[Point], corners: Sequence[int],
    scale: float,
) -> list[Point]:
    """Samples to fit: smoothed between `corners`, corners placed, overshoots collapsed."""
    inner = list(corners) if curve.closed else list(corners[1:-1])
    seq = place_corners(
        smooth_between(curve.raw, corners, curve.closed, curve.params.smooth_sigma),
        inner, corners, (curve.raw, plain), original, curve.closed, scale, SIDE_LINES,
        outward_snap=curve.subpixel,
    )
    return collapse_behind(seq, inner, corners, curve.closed, curve.count(BEHIND_REACH))


def _placed(
    curve: _Curve, plain: tuple[Point, ...], original: Sequence[Point], corners: list[int],
    scale: float,
) -> tuple[tuple[Point, ...], list[int]]:
    """`_place`, then without any corner placed onto the corner before it, and placed again.

    Two corners on one point (both snapped to the same vertex) leave a run that leaves and
    returns to it: a hairpin. The later corner of such a pair is dropped (never a curve end).
    """
    seq = _place(curve, plain, original, corners, scale)
    n = len(seq)

    def distinct(k: int, c: int) -> bool:
        if len(corners) == 1 or (not curve.closed and c in (0, n - 1)):
            return True
        return math.dist(seq[c], seq[corners[k - 1]]) > COINCIDENT_PX * scale

    keep = [c for k, c in enumerate(corners) if distinct(k, c)]
    if len(keep) == len(corners):
        return tuple(seq), corners
    return tuple(_place(curve, plain, original, keep, scale)), keep


def _grid_near_junctions(
    raw: tuple[Point, ...], plain: tuple[Point, ...], closed: bool, k: int
) -> tuple[Point, ...]:
    """`raw` with the samples within `k` of an open curve's ends on the crack grid."""
    if closed:
        return raw
    n = len(raw)
    return tuple(plain[i] if i < k or i >= n - k else p for i, p in enumerate(raw))


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
    junction = samples_at_scale(JUNCTION_GRID_SAMPLES, scale)
    plain = tuple(samples[1])
    raw = _grid_near_junctions(tuple(samples[0]), plain, curve.closed, junction)
    closed, n = curve.closed, len(raw)
    noise = estimate_noise(raw, closed, scale)
    adapted = _adapt(params, noise)
    judge = replace(adapted, tolerance_px=TOL_PER_NOISE * noise)
    c = _Curve(raw, closed, adapted, judge, noise, subpixel=raw != plain, scale=scale)
    coarse = max(COARSE_PER_SIGMA * params.smooth_sigma, COARSE_PER_NOISE * noise)
    found = find_corners_multiscale(
        raw, closed, params.corner_angle_deg, params.smooth_sigma, coarse,
        c.count(corner_window), c.count(TRACK_RADIUS),
    )
    angles = dict(zip(found.indices, found.coarse_angles))
    clear = _clear_of_ends(found.indices, n, closed, junction)
    seq, corners = _placed(c, plain, curve.points, clear, scale)
    inner = set(corners if closed else corners[1:-1])
    pinned = tuple(seq[i] if i in inner else p for i, p in enumerate(raw))
    merged = _merge_spans(c, _Merge(pinned, tuple(seq), tuple(corners)), angles)
    seq, corners = merged.seq, list(merged.corners)
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
