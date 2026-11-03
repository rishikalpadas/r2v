"""Automatic defect detectors for vector output (quality roadmap step Q1).

Each `detect_*` function is pure: it takes plain geometry (or rasters) in original-image pixels
and returns a `DetectorResult`, a score (higher is worse, 0 is clean) plus `Spot`s (boxes in
original-image px with a severity), worst first. `geometry_from_state` turns the pipeline's
final state (see `engine.pipeline.run_stages`) into that geometry.

| detector                 | score                                                         |
|--------------------------|---------------------------------------------------------------|
| `detect_crossings`       | proper crossings between drawn curves (count)                 |
| `detect_specks`          | output shapes below `SPECK_AREA_PX` (count)                   |
| `detect_wobble`          | excess over the flag line, in flag units x window lengths     |
| `detect_thin_bands`      | same, for band width std (halved: each band is seen from both sides) |
| `detect_junction_spikes` | excess turning (rad) of flagged junction ends                 |
| `detect_straightness`    | same as wobble, for deviation of a straight fit from the source edge |
| `detect_worst_patches`   | mean badness of the worst `PATCH_TOP_K` windows               |

Sliding-window scores are continuous: each window adds max(0, value / flag - 1) times its share
of the curve length, so a score does not jump when a window boundary moves; the flag threshold
only decides which windows become spots. Thresholds are named constants below, each with the
reason for its value, tuned so clean synthetic shapes do not flag and the known happiness.jpg
defects do.
"""
from __future__ import annotations

import itertools
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates, uniform_filter
from scipy.spatial import cKDTree
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity

from engine.geometry.crossing_guard import (
    FITTED,
    PRIMITIVE,
    Track,
    find_crossings,
    sample_segments,
    sample_shape,
)
from engine.types import OUTSIDE, TRANSPARENT, Circle, Ellipse, Rect, VectorDoc

Box = tuple[float, float, float, float]  # (x0, y0, x1, y1) in original-image px

# ---------------------------------------------------------------- thresholds (original px)

# Geometry sampling: chord error of sampled Beziers/arcs; far below the 0.5 px resampling step.
GEOM_SAMPLE_ERROR_PX = 0.05

# Crossings: same clearance as the engine's guard, so a legit meeting at a junction is ignored.
CROSSING_CLEARANCE_PX = 1.0

# Specks: shapes under this area (px^2) are specks; real dots/periods in logos are far larger,
# and segment.min_region_px (4 processing px) lets 1-6 px^2 noise regions through.
SPECK_AREA_PX = 12.0

# Wobble: short wiggles (turning excess over a faired copy).
WOBBLE_STEP_PX = 0.5  # arc-length resampling step; fine enough to see a 3 px wiggle
WOBBLE_MIN_CURVE_PX = 12.0  # shorter curves are tiny shapes, not "long edges" (= fit's closed minimum)
WOBBLE_FAIR_SIGMA_PX = 6.0  # fairing scale: removes wiggles up to ~15-20 px wavelength (4 missed the L edge of happiness), keeps letter-scale curves
WOBBLE_CORNER_RAD = math.radians(40.0)  # turn within 1.5 px that counts as a true corner (fit's corner is 35 deg)
WOBBLE_CORNER_GAP = 2  # samples dropped either side of a corner (its turn is split over ~2 samples)
WOBBLE_MIN_PIECE_PX = 6.0  # corner-to-corner pieces shorter than this are serifs/ticks, not edges
WOBBLE_WINDOW_PX = 24.0  # local window: about two to three wiggles of a visible wobble
WOBBLE_STRIDE = 4  # samples (2 px) between sliding windows: spots and scores move smoothly
WOBBLE_FLAT_RAD = math.radians(0.25)  # per-sample turn treated as straight (radius > 115 px)
WOBBLE_MIN_SIGN_CHANGES = 2  # one inflection is a clean S-curve; wobble alternates repeatedly
WOBBLE_FLAG_RAD = math.radians(20.0)  # excess turning per window that is visibly wavy
WOBBLE_MIN_AMPLITUDE_PX = 0.2  # wiggles closer than this to the faired curve are invisible at 1x
# Wobble: long low-amplitude bends (reviewer's measure; the inner L edge of happiness).
BEND_WINDOW_PX = 40.0  # long enough to hold a 30-40 px wave the fairing scale leaves alone
BEND_MAX_TURN_RAD = math.radians(20.0)  # windows that turn more are real curves, not bent lines
BEND_CURVATURE_SIGMA_PX = 4.0  # curvature smoothing before counting inflections
BEND_FLAT_CURVATURE = 1.5e-3  # 1/px; flatter is straight (radius > 650 px) and carries no sign
BEND_RESIDUAL_PX = 0.3  # quadratic-fit residual that reads as a bend at 1x
BEND_MAX_RESIDUAL_PX = 1.0  # larger residuals are drawn S-shapes (script and letter curves on happiness hit the old 1.5 cap); the L inner edge is 0.92

# Straightness against the source edge (segmentation, not fit).
STRAIGHT_WINDOW_PX = 60.0  # long straight runs only; short ones are too noisy to judge
STRAIGHT_STRIDE_PX = 5.0
STRAIGHT_MAX_TURN_RAD = math.radians(5.0)  # source edge turning end to end; the L's outer edge reads 3.9 deg
STRAIGHT_MAX_FIT_TURN_RAD = math.radians(40.0)  # fitted windows past a corner are not compared
STRAIGHT_STEP_PX = 1.0  # samples along the window
STRAIGHT_PROFILE_PX = 3.0  # source profile half-length across the edge
STRAIGHT_MIN_CONTRAST = 30.0  # grey levels; weaker edges give no reliable position
STRAIGHT_SOURCE_RMS_PX = 0.35  # the source edge itself must be straight (JPEG noise ~0.2 px)
STRAIGHT_BORDER_PX = 4.0  # windows this close to the image edge are skipped: the profile is clipped
#                            there (happiness' 1-3 px frame lines read up to 2 px off)
STRAIGHT_FLAG_PX = 0.4  # how much farther the fit strays from the source's line than the source does

# Uneven thin bands.
BAND_STEP_PX = 1.0  # width sample spacing along each curve
BAND_MAX_PX = 6.5  # spec said ~4; the black letter outlines of happiness (LOOKS, L) are 5.3-6 px wide
BAND_END_TRIM_PX = 2.0  # drop run ends: corners and tips change width legitimately there
BAND_MIN_RUN_PX = 8.0  # shorter thin runs are corners/tips, not bands
BAND_WINDOW_PX = 30.0  # local window for width variation
BAND_STRIDE = 3  # samples between sliding windows
BAND_TREND_SIGMA_PX = 4.0  # width trend scale: a straight taper (wedge, tip) is trend, not unevenness
BAND_STD_FLAG_PX = 0.3  # std of width about its trend that reads as uneven at 1x (a third of a px)
BAND_SIDES = 2.0  # every band is measured from both of its sides; the score is divided by this
_RAY_EPS = 0.05  # hits closer than this are the ray's own or a touching curve, not a width

# Junction spikes.
JUNCTION_STEP_PX = 0.25  # resampling step near curve ends
JUNCTION_REACH_PX = 4.0  # "within a few px" of the junction
JUNCTION_TURN_RAD = math.radians(60.0)  # excess end turning that makes a visible spike/hook
# The source's own end turning is the NET heading change of its crack path (smoothed by
# JUNCTION_RAW_SIGMA_PX) over the reach. Pixel stairs alternate and cancel in a net sum, so the
# baseline barely depends on the processing scale; the sum of |turn| used before left 25-42 deg
# of stair noise at scale 1 against 6-7 deg at scale 4. A true tip or corner still counts.
JUNCTION_RAW_SIGMA_PX = 1.0

# Worst patches.
PATCH_PX = 24  # window side; about a letter stroke plus its edges at 1x
PATCH_DE_REF = 2.0  # mean dE2000 per window counted as one unit (just noticeable is ~1-2)
PATCH_SSIM_REF = 0.1  # 1 - SSIM per window counted as one unit
PATCH_TOP_K = 5  # image score = mean of this many worst windows
PATCH_SPOTS = 10  # windows reported


# ---------------------------------------------------------------- data


@dataclass(frozen=True)
class Spot:
    box: Box
    severity: float
    note: str = ""


@dataclass(frozen=True)
class DetectorResult:
    score: float
    spots: tuple[Spot, ...]  # worst first


@dataclass(frozen=True, eq=False)  # holds ndarrays
class CurveGeom:
    """One boundary curve as drawn, in original px.

    `points` is the drawn outline: the fitted polyline, or the primitive's outline when the
    curve is drawn only as a primitive (closed curves repeat the first point last). Walking it,
    `left` is on the left as in `engine.types.BoundaryCurve`. `smooth` is False for curves kept
    as exact pixel polylines (canvas edge, too short). `primitive` is the drawn primitive's
    outline, if export draws one for this curve; `segments_drawn` says whether the fitted
    segments are drawn too (as another region's loop). `raw` is the crack-grid path."""

    id: int
    points: np.ndarray
    left: int
    right: int
    closed: bool
    smooth: bool = True
    primitive: np.ndarray | None = None
    raw: np.ndarray | None = None
    segments_drawn: bool = True


@dataclass(frozen=True)
class ShapeGeom:
    area: float  # original px^2
    box: Box


@dataclass(frozen=True, eq=False)
class Geometry:
    curves: tuple[CurveGeom, ...]
    shapes: tuple[ShapeGeom, ...]
    scale: float  # processing px per original px


DETECTORS = ("crossings", "specks", "wobble", "thin_bands", "junction_spikes", "straightness",
             "worst_patches")


# ---------------------------------------------------------------- polyline helpers


def _bbox(pts: np.ndarray, pad: float = 0.0) -> Box:
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return (float(lo[0] - pad), float(lo[1] - pad), float(hi[0] + pad), float(hi[1] + pad))


def _dedupe(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float64)
    if len(pts) < 2:
        return pts
    keep = np.r_[True, np.hypot(*np.diff(pts, axis=0).T) > 1e-9]
    return pts[keep]


def _resample(pts: np.ndarray, step: float) -> np.ndarray:
    """Points every `step` of arc length, keeping both ends (a closed input stays closed)."""
    pts = _dedupe(pts)
    if len(pts) < 2:
        return pts
    seg = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.r_[0.0, np.cumsum(seg)]
    total = cum[-1]
    n = max(1, int(round(total / step)))
    s = np.linspace(0.0, total, n + 1)
    return np.stack([np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])], axis=1)


def _turns(pts: np.ndarray) -> np.ndarray:
    """Signed turning angle at each interior vertex of an open polyline."""
    d = np.diff(pts, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    return (np.diff(ang) + math.pi) % (2 * math.pi) - math.pi


def _length(pts: np.ndarray) -> float:
    return float(np.hypot(*np.diff(pts, axis=0).T).sum()) if len(pts) > 1 else 0.0


def _sign_changes(values: np.ndarray, flat: float) -> int:
    signs = np.sign(values[np.abs(values) > flat])
    return int(np.count_nonzero(signs[1:] != signs[:-1])) if len(signs) > 1 else 0


def _fair(pts: np.ndarray, sigma: float, closed: bool) -> np.ndarray:
    """Gaussian-faired copy of `pts`. Open pieces keep their end points: they are padded by odd
    reflection about each end, which continues a straight or inflecting end unchanged."""
    if closed:
        return np.stack([gaussian_filter1d(pts[:, i], sigma, mode="wrap") for i in (0, 1)], axis=1)
    pad = min(len(pts) - 1, int(math.ceil(4 * sigma)))
    if pad < 1:
        return pts.copy()
    head = 2 * pts[0] - pts[pad:0:-1]
    tail = 2 * pts[-1] - pts[-2 : -pad - 2 : -1]
    ext = np.vstack([head, pts, tail])
    out = np.stack([gaussian_filter1d(ext[:, i], sigma, mode="nearest") for i in (0, 1)], axis=1)
    return out[pad : pad + len(pts)]


def _windows(n: int, size: int, stride: int) -> list[tuple[int, int]]:
    """Sliding windows [a, b) over n samples; one window when n <= size."""
    if n <= size:
        return [(0, n)] if n else []
    starts = list(range(0, n - size + 1, stride))
    if starts[-1] != n - size:
        starts.append(n - size)
    return [(a, a + size) for a in starts]


def _window_weight(n: int, size: int, count: int) -> float:
    """Each window's share of the piece length in window units: the windows of a piece add up
    to n / size, however many there are, so the score is an integral along the curve."""
    return (n / size) / count if count else 0.0


def _merge_overlapping(items: list[tuple[int, int, float, np.ndarray, str]]) -> list[Spot]:
    """Merge flagged windows (start, end, severity, points, note) that overlap into spots;
    a spot keeps its worst window's severity and note."""
    spots: list[Spot] = []
    group: list = []

    def flush():
        if group:
            worst = max(group, key=lambda g: g[2])
            pts = np.vstack([g[3] for g in group])
            spots.append(Spot(_bbox(pts, 0.5), float(worst[2]), worst[4]))

    for item in sorted(items, key=lambda g: g[0]):
        if group and item[0] >= max(g[1] for g in group):
            flush()
            group = []
        group.append(item)
    flush()
    return spots


def _sorted(score: float, spots: list[Spot]) -> DetectorResult:
    return DetectorResult(float(score), tuple(sorted(spots, key=lambda s: -s.severity)))


# ---------------------------------------------------------------- crossings


def detect_crossings(
    curves: Sequence[CurveGeom], clearance: float = CROSSING_CLEARANCE_PX
) -> DetectorResult:
    """Proper crossings between the drawn outlines of different curves, ignoring points within
    `clearance` of a junction both curves end at (`crossing_guard.find_crossings`)."""
    tracks: list[Track] = []
    ends: dict[int, tuple] = {}
    for c in curves:
        if c.segments_drawn:
            pts = _dedupe(c.points)
            if len(pts) >= 2:
                tracks.append(Track(c.id, FITTED, pts, np.zeros(len(pts) - 1, np.int64)))
        if c.primitive is not None and len(c.primitive) >= 2:
            tracks.append(
                Track(c.id, PRIMITIVE, c.primitive, np.zeros(len(c.primitive) - 1, np.int64))
            )
        if not c.closed and len(c.points):
            ends[c.id] = (tuple(map(float, c.points[0])), tuple(map(float, c.points[-1])))
    found = find_crossings(tracks, ends, clearance)
    spots = [
        Spot((x.point[0] - 2, x.point[1] - 2, x.point[0] + 2, x.point[1] + 2), 1.0,
             f"curves {x.a[0]} x {x.b[0]}")
        for x in found
    ]
    return _sorted(len(found), spots)


# ---------------------------------------------------------------- specks


def _shoelace(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]) + x[-1] * y[0] - x[0] * y[-1])


def _shape_area(shape) -> float:
    if isinstance(shape, Circle):
        return math.pi * shape.r**2
    if isinstance(shape, Ellipse):
        return math.pi * shape.rx * shape.ry
    if isinstance(shape, Rect):
        return abs(shape.w * shape.h)
    return abs(_shoelace(np.asarray(shape.points, np.float64)))


def shapes_from_doc(doc: VectorDoc) -> tuple[ShapeGeom, ...]:
    """Area (outer minus emitted holes) and box of every drawn path and primitive."""
    out: list[ShapeGeom] = []
    for path in doc.paths:
        if not path.contours or not path.contours[0]:
            continue
        rings = [sample_segments(c, GEOM_SAMPLE_ERROR_PX)[0] for c in path.contours if c]
        area = abs(_shoelace(rings[0])) - sum(abs(_shoelace(r)) for r in rings[1:])
        out.append(ShapeGeom(max(area, 0.0), _bbox(rings[0])))
    for prim in doc.primitives:
        pts = sample_shape(prim.shape, GEOM_SAMPLE_ERROR_PX)
        out.append(ShapeGeom(_shape_area(prim.shape), _bbox(pts)))
    return tuple(out)


def detect_specks(shapes: Sequence[ShapeGeom], min_area: float = SPECK_AREA_PX) -> DetectorResult:
    """Shapes smaller than `min_area`; score = their count. Severity = area / `min_area`: a
    bigger speck is more visible (one under 1 px^2 is degenerate junk, nearly invisible at 1x)."""
    spots = [
        Spot(s.box, s.area / min_area, f"area {s.area:.1f} px2")
        for s in shapes
        if s.area < min_area
    ]
    return _sorted(len(spots), spots)


# ---------------------------------------------------------------- wobble


def _corner_runs(turns: np.ndarray, ring: bool = False) -> list[tuple[int, int]]:
    """(first, last) turn indices of each run whose 3-sample turning reaches a corner."""
    if ring and len(turns) >= 3:
        window = turns + np.roll(turns, 1) + np.roll(turns, -1)
    else:
        window = np.convolve(turns, np.ones(3), mode="same")
    hot = np.abs(window) >= WOBBLE_CORNER_RAD
    runs, start = [], None
    for i, h in enumerate(hot):
        if h and start is None:
            start = i
        if not h and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(hot) - 1))
    return runs


def _pieces(pts: np.ndarray, closed: bool) -> list[tuple[np.ndarray, bool]]:
    """Corner-to-corner pieces of a resampled curve, each (points, is_ring). A closed curve
    without corners is one ring (no repeated end point); corners and `WOBBLE_CORNER_GAP`
    samples either side of them are left out."""
    if closed:
        ring = pts[:-1]
        m = len(ring)
        turns = _turns(np.vstack([ring[-1:], ring, ring[:1]]))  # turn at every ring vertex
        runs = _corner_runs(turns, ring=True)
        if not runs:
            return [(ring, True)]
        # rotate so the ring starts just before a corner run, then cut it open there
        first = runs[0][0] if not (runs[0][0] == 0 and runs[-1][1] == m - 1) else runs[-1][0]
        ring, turns = np.roll(ring, -first, axis=0), np.roll(turns, -first)
        pts = np.vstack([ring, ring[:1]])
        vert_runs = [*_corner_runs(turns), (m, m)]  # ring vertex k = turn k; m closes the ring
    else:
        vert_runs = [(a + 1, b + 1) for a, b in _corner_runs(_turns(pts))]
    pieces, start = [], 0
    for a, b in vert_runs:
        lo, hi = a - WOBBLE_CORNER_GAP, b + WOBBLE_CORNER_GAP
        if lo > start:
            pieces.append(pts[start : lo + 1])
        start = max(start, hi)
    if start < len(pts) - 1:
        pieces.append(pts[start:])
    return [(p, False) for p in pieces]


def _wiggle_windows(piece: np.ndarray, ring: bool) -> tuple[float, list]:
    """(score, flagged windows) of short wiggles on one piece: excess turning over a faired copy
    per `WOBBLE_WINDOW_PX` window, gated by sign changes and visible amplitude."""
    faired = _fair(piece, WOBBLE_FAIR_SIGMA_PX / WOBBLE_STEP_PX, ring)
    if ring:
        ext = lambda p: np.vstack([p[-1:], p, p[:1]])  # noqa: E731 - turn at every vertex
        t_raw, t_fair, verts = _turns(ext(piece)), _turns(ext(faired)), piece
        dev = np.hypot(*(piece - faired).T)
    else:
        t_raw, t_fair, verts = _turns(piece), _turns(faired), piece[1:-1]
        dev = np.hypot(*(piece - faired).T)[1:-1]
    excess = np.abs(t_raw) - np.abs(t_fair)
    size = int(round(WOBBLE_WINDOW_PX / WOBBLE_STEP_PX))
    wins = _windows(len(excess), size, WOBBLE_STRIDE)
    weight = _window_weight(len(excess), size, len(wins))
    score, flagged = 0.0, []
    for a, b in wins:
        ex = float(excess[a:b].sum())
        if ex <= 0 or float(dev[a:b].max()) < WOBBLE_MIN_AMPLITUDE_PX:
            continue
        changes = _sign_changes(t_raw[a:b], WOBBLE_FLAT_RAD)
        if changes < WOBBLE_MIN_SIGN_CHANGES:
            continue
        score += max(0.0, ex / WOBBLE_FLAG_RAD - 1.0) * weight
        if ex >= WOBBLE_FLAG_RAD:
            note = (f"wiggle: excess {math.degrees(ex):.0f} deg, amplitude "
                    f"{dev[a:b].max():.2f} px, {changes} sign changes")
            flagged.append((a, b, ex / WOBBLE_FLAG_RAD, verts[a:b], note))
    return score, flagged


def _bend_windows(piece: np.ndarray) -> tuple[float, list]:
    """(score, flagged windows) of long low-amplitude bends: `BEND_WINDOW_PX` windows that turn
    little overall, inflect at least twice, and leave a quadratic-fit residual between
    `BEND_RESIDUAL_PX` and `BEND_MAX_RESIDUAL_PX`."""
    size = int(round(BEND_WINDOW_PX / WOBBLE_STEP_PX))
    if len(piece) < size:
        return 0.0, []
    k = gaussian_filter1d(_turns(piece) / WOBBLE_STEP_PX, BEND_CURVATURE_SIGMA_PX / WOBBLE_STEP_PX,
                          mode="nearest")
    k = np.r_[0.0, k, 0.0]  # curvature per vertex
    wins = _windows(len(piece), size, WOBBLE_STRIDE)
    weight = _window_weight(len(piece), size, len(wins))
    score, flagged = 0.0, []
    for a, b in wins:
        kw = k[a:b]
        if abs(kw.sum() * WOBBLE_STEP_PX) > BEND_MAX_TURN_RAD:
            continue
        if _sign_changes(kw, BEND_FLAT_CURVATURE) < 2:
            continue
        w = piece[a:b]
        chord = w[-1] - w[0]
        length = math.hypot(*chord)
        if length < 1e-9:
            continue
        t = chord / length
        u, v = (w - w[0]) @ t, (w - w[0]) @ np.array([-t[1], t[0]])
        if np.any(np.diff(u) <= 0):
            continue
        r = float(np.abs(v - np.polyval(np.polyfit(u, v, 2), u)).max())
        if r > BEND_MAX_RESIDUAL_PX:
            continue
        score += max(0.0, r / BEND_RESIDUAL_PX - 1.0) * weight
        if r >= BEND_RESIDUAL_PX:
            flagged.append((a, b, r / BEND_RESIDUAL_PX, w, f"bend: {r:.2f} px off a smooth arc"))
    return score, flagged


def detect_wobble(curves: Sequence[CurveGeom]) -> DetectorResult:
    """Jitter and wavy bends along smooth curves.

    Each curve is resampled every `WOBBLE_STEP_PX` and split at true corners (turning of at
    least `WOBBLE_CORNER_RAD` within 3 samples, which are excluded). On each piece:

    - wiggles: per sliding `WOBBLE_WINDOW_PX` window, excess turning = sum|turn| - sum|faired
      turn| against a Gaussian-faired copy (near 0 for circles, S-curves and straight runs),
      counted when the curvature changes sign `WOBBLE_MIN_SIGN_CHANGES` times and the piece
      strays `WOBBLE_MIN_AMPLITUDE_PX` from its faired copy;
    - bends: see `_bend_windows`.

    Score = sum over windows of max(0, value / flag - 1) x the window's share of the length
    (unitless); windows at or above the flag become spots (overlapping ones merged).
    """
    spots: list[Spot] = []
    score = 0.0
    for c in curves:
        if not c.smooth or OUTSIDE in (c.left, c.right) or _length(c.points) < WOBBLE_MIN_CURVE_PX:
            continue
        pts = _resample(c.points, WOBBLE_STEP_PX)
        if len(pts) < 5:
            continue
        closed = c.closed and np.allclose(pts[0], pts[-1])
        for piece, ring in _pieces(pts, closed):
            if _length(piece) < WOBBLE_MIN_PIECE_PX or len(piece) < 5:
                continue
            s1, f1 = _wiggle_windows(piece, ring)
            s2, f2 = _bend_windows(piece) if not ring else _bend_windows(np.vstack([piece, piece[:1]]))
            score += s1 + s2
            for s in _merge_overlapping(f1) + _merge_overlapping(f2):
                spots.append(Spot(s.box, s.severity, f"curve {c.id} {s.note}"))
    return _sorted(score, spots)


# ---------------------------------------------------------------- straightness


def _luma(px: np.ndarray) -> np.ndarray:
    alpha = px[..., 3].astype(np.float64) / 255.0
    rgb = px[..., :3].astype(np.float64) * alpha[..., None] + 255.0 * (1.0 - alpha[..., None])
    return rgb @ np.array([0.299, 0.587, 0.114])


def _source_edge(luma: np.ndarray, pts: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Sub-pixel source edge position along `normal` through each point (NaN where the source
    has no clear edge there): the mid-level crossing of the luminance profile nearest the point."""
    offs = np.arange(-STRAIGHT_PROFILE_PX, STRAIGHT_PROFILE_PX + 1e-9, 0.25)
    xy = pts[:, None, :] + offs[None, :, None] * normal[:, None, :]
    prof = map_coordinates(luma, [xy[..., 1] - 0.5, xy[..., 0] - 0.5], order=1, mode="nearest")
    lo, hi = prof[:, :3].mean(axis=1), prof[:, -3:].mean(axis=1)
    out = np.full(len(pts), np.nan)
    for i in np.flatnonzero(np.abs(hi - lo) >= STRAIGHT_MIN_CONTRAST):
        f = prof[i] - (lo[i] + hi[i]) / 2
        cross = np.flatnonzero(np.sign(f[:-1]) * np.sign(f[1:]) < 0)
        if not len(cross):
            continue
        j = cross[np.argmin(np.abs(offs[cross]))]
        out[i] = offs[j] + (offs[j + 1] - offs[j]) * f[j] / (f[j] - f[j + 1])
    return out


def _turn_between(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between the total-least-squares directions of two point runs."""
    if len(a) < 3 or len(b) < 3:
        return math.pi
    da = np.linalg.svd(a - a.mean(axis=0), full_matrices=False)[2][0]
    db = np.linalg.svd(b - b.mean(axis=0), full_matrices=False)[2][0]
    return math.acos(min(1.0, abs(float(da @ db))))


def detect_straightness(curves: Sequence[CurveGeom], source: np.ndarray) -> DetectorResult:
    """Straight source edges drawn bent.

    Along smooth curves, the source edge is read across each sample (`_source_edge`). Sliding
    `STRAIGHT_WINDOW_PX` windows where that source edge is straight (line-fit RMS at most
    `STRAIGHT_SOURCE_RMS_PX`, turning less than `STRAIGHT_MAX_TURN_RAD` end to end) are
    straight edges; the window's value is how much farther the fitted outline strays from the
    source's line than the source edge itself does, flagged at `STRAIGHT_FLAG_PX`. The straightness gate is on the source, not
    the fit: a fit that bends off a straight source edge is exactly the defect. Windows whose
    fit turns more than `STRAIGHT_MAX_FIT_TURN_RAD` (past a corner) are skipped. This measures where segmentation followed edge
    noise, so it is kept apart from `detect_wobble`. `source` is HxWx4 in the curves' px.
    """
    luma = _luma(source)
    size = int(round(STRAIGHT_WINDOW_PX / STRAIGHT_STEP_PX))
    stride = max(1, int(round(STRAIGHT_STRIDE_PX / STRAIGHT_STEP_PX)))
    spots: list[Spot] = []
    score = 0.0
    for c in curves:
        if not c.smooth or OUTSIDE in (c.left, c.right) or _length(c.points) < STRAIGHT_WINDOW_PX:
            continue
        pts = _resample(c.points, STRAIGHT_STEP_PX)
        tangent = np.gradient(pts, axis=0)
        tangent /= np.maximum(np.hypot(*tangent.T), 1e-12)[:, None]
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
        heading = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
        edge = None
        wins = _windows(len(pts), size, stride)
        if len(pts) < size:
            continue
        weight = _window_weight(len(pts), size, len(wins))
        flagged = []
        h_img, w_img = luma.shape
        for a, b in wins:
            if np.ptp(heading[a:b]) > STRAIGHT_MAX_FIT_TURN_RAD:  # a corner inside
                continue
            win = pts[a:b]
            if (win.min() < STRAIGHT_BORDER_PX or win[:, 0].max() > w_img - STRAIGHT_BORDER_PX
                    or win[:, 1].max() > h_img - STRAIGHT_BORDER_PX):
                continue
            if edge is None:
                edge = _source_edge(luma, pts, normal)
            ok = np.isfinite(edge[a:b])
            if ok.sum() < 0.8 * (b - a):
                continue
            src = pts[a:b][ok] + edge[a:b][ok, None] * normal[a:b][ok]
            centre = src.mean(axis=0)
            direction = np.linalg.svd(src - centre, full_matrices=False)[2][0]
            line_n = np.array([-direction[1], direction[0]])
            if float(np.sqrt((((src - centre) @ line_n) ** 2).mean())) > STRAIGHT_SOURCE_RMS_PX:
                continue
            if _turn_between(src[: len(src) // 4], src[-(len(src) // 4) :]) > STRAIGHT_MAX_TURN_RAD:
                continue
            # the fit's excess over the source's own spread: a faithfully fitted gentle arc
            # (sagitta up to ~0.6 px at 5 deg over 60 px) is not a defect
            dev = float(np.abs((pts[a:b] - centre) @ line_n).max()
                        - np.abs((src - centre) @ line_n).max())
            score += max(0.0, dev / STRAIGHT_FLAG_PX - 1.0) * weight
            if dev >= STRAIGHT_FLAG_PX:
                flagged.append((a, b, dev / STRAIGHT_FLAG_PX, pts[a:b],
                                f"{dev:.2f} px off the straight source edge"))
        for s in _merge_overlapping(flagged):
            spots.append(Spot(s.box, s.severity, f"curve {c.id}: {s.note}"))
    return _sorted(score, spots)


# ---------------------------------------------------------------- thin bands


class _Segments:
    """All curve polylines as segments, with a KD-tree on their midpoints for ray casting."""

    def __init__(self, curves: Sequence[CurveGeom]) -> None:
        p0, p1, left, right = [], [], [], []
        for c in curves:
            pts = _resample(c.points, BAND_STEP_PX)  # short pieces keep the KD-tree reach small
            if len(pts) < 2:
                continue
            p0.append(pts[:-1])
            p1.append(pts[1:])
            left.append(np.full(len(pts) - 1, c.left))
            right.append(np.full(len(pts) - 1, c.right))
        self.ok = bool(p0)
        if not self.ok:
            return
        self.p0, self.p1 = np.vstack(p0), np.vstack(p1)
        self.left, self.right = np.concatenate(left), np.concatenate(right)
        self.reach = float(np.hypot(*(self.p1 - self.p0).T).max()) / 2
        self.tree = cKDTree((self.p0 + self.p1) / 2)

    def cast(self, origin: np.ndarray, direction: np.ndarray, region: int) -> np.ndarray:
        """Distance along each ray to the first boundary of `region`, or inf beyond BAND_MAX_PX."""
        out = np.full(len(origin), np.inf)
        if not self.ok or not len(origin):
            return out
        lists = self.tree.query_ball_point(origin, BAND_MAX_PX + self.reach)
        counts = np.fromiter((len(x) for x in lists), np.int64, len(lists))
        if not counts.sum():
            return out
        ray = np.repeat(np.arange(len(origin)), counts)
        seg = np.fromiter(itertools.chain.from_iterable(lists), np.int64, int(counts.sum()))
        bounding = (self.left[seg] == region) | (self.right[seg] == region)
        ray, seg = ray[bounding], seg[bounding]
        o, d = origin[ray], direction[ray]
        a, e = self.p0[seg], self.p1[seg] - self.p0[seg]
        denom = d[:, 0] * e[:, 1] - d[:, 1] * e[:, 0]
        ao = a - o
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (ao[:, 0] * e[:, 1] - ao[:, 1] * e[:, 0]) / denom
            u = (ao[:, 0] * d[:, 1] - ao[:, 1] * d[:, 0]) / denom
        hit = (np.abs(denom) > 1e-12) & (t > _RAY_EPS) & (u >= 0) & (u <= 1) & (t <= BAND_MAX_PX)
        np.minimum.at(out, ray[hit], t[hit])
        return out


def _trend(w: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian trend of a width profile, padded by odd reflection so a linear taper is its own
    trend right up to the run ends."""
    pad = min(len(w) - 1, int(math.ceil(4 * sigma)))
    if pad < 1:
        return np.full_like(w, w.mean())
    ext = np.r_[2 * w[0] - w[pad:0:-1], w, 2 * w[-1] - w[-2 : -pad - 2 : -1]]
    return gaussian_filter1d(ext, sigma, mode="nearest")[pad : pad + len(w)]


def _band_windows(pts: np.ndarray, width: np.ndarray) -> tuple[float, list]:
    thin = width <= BAND_MAX_PX
    trim = int(round(BAND_END_TRIM_PX / BAND_STEP_PX))
    size = max(2, int(round(BAND_WINDOW_PX / BAND_STEP_PX)))
    score, flagged = 0.0, []
    for is_thin, group in itertools.groupby(range(len(thin)), key=lambda i: thin[i]):
        idx = np.fromiter(group, np.int64)
        if not is_thin:
            continue
        idx = idx[trim : len(idx) - trim]
        if len(idx) * BAND_STEP_PX < BAND_MIN_RUN_PX:
            continue
        run_w = width[idx]
        resid = run_w - _trend(run_w, BAND_TREND_SIGMA_PX / BAND_STEP_PX)
        wins = _windows(len(idx), size, BAND_STRIDE)
        weight = _window_weight(len(idx), size, len(wins))
        for a, b in wins:
            std = float(resid[a:b].std())
            score += max(0.0, std / BAND_STD_FLAG_PX - 1.0) * weight / BAND_SIDES
            if std >= BAND_STD_FLAG_PX:
                w = run_w[a:b]
                note = f"width {w.min():.1f}-{w.max():.1f} px, std about trend {std:.2f}"
                flagged.append((int(idx[a]), int(idx[b - 1]) + 1, std / BAND_STD_FLAG_PX,
                                pts[idx[a:b]], note))
    return score, flagged


def detect_thin_bands(curves: Sequence[CurveGeom]) -> DetectorResult:
    """Uneven width along thin bands.

    From samples every `BAND_STEP_PX` along each curve, a ray is cast along the normal into the
    region on each side; the distance to that region's next boundary is the band width there.
    Runs where the width is at most `BAND_MAX_PX` (trimmed by `BAND_END_TRIM_PX`, at least
    `BAND_MIN_RUN_PX` long) are bands. Width is compared with its own Gaussian trend
    (`BAND_TREND_SIGMA_PX`, so a straight taper does not count); each sliding `BAND_WINDOW_PX`
    window scores max(0, std / `BAND_STD_FLAG_PX` - 1) x its share of the run length, and flags
    at the threshold. Every band is seen from both of its sides, so the score is divided by
    `BAND_SIDES`; spots stay per side (a band can show two overlapping spots).
    """
    segs = _Segments(curves)
    spots: list[Spot] = []
    score = 0.0
    for c in curves:
        pts = _resample(c.points, BAND_STEP_PX)
        if len(pts) < 3:
            continue
        tangent = np.gradient(pts, axis=0)
        norm = np.hypot(*tangent.T)
        good = norm > 1e-9
        tangent[good] /= norm[good, None]
        left_normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
        for sign, region in ((1.0, c.left), (-1.0, c.right)):
            if region in (TRANSPARENT, OUTSIDE):
                continue
            width = segs.cast(pts, sign * left_normal, region)
            width[~good] = np.inf
            s, flagged = _band_windows(pts, width)
            score += s
            for spot in _merge_overlapping(flagged):
                spots.append(Spot(spot.box, spot.severity, f"region {region}: {spot.note}"))
    return _sorted(score, spots)


# ---------------------------------------------------------------- junction spikes


def _abs_turn(pts: np.ndarray) -> float:
    return float(np.abs(_turns(pts)).sum()) if len(pts) >= 3 else 0.0


def _raw_end_turn(raw: np.ndarray, reach: int) -> float:
    """Net heading change |sum turn| over the first `reach` samples of the crack path smoothed
    by `JUNCTION_RAW_SIGMA_PX`. Pixel stairs alternate and cancel in the net sum at any scale;
    a true tip or corner near the junction does not."""
    dense = _resample(_dedupe(raw), JUNCTION_STEP_PX)
    if len(dense) < 3:
        return 0.0
    smooth = _fair(dense, JUNCTION_RAW_SIGMA_PX / JUNCTION_STEP_PX, False)
    return abs(float(_turns(smooth[: reach + 2]).sum()))


def detect_junction_spikes(curves: Sequence[CurveGeom]) -> DetectorResult:
    """Sharp turns or short hooks within `JUNCTION_REACH_PX` of a curve's end at a junction
    shared with another curve.

    Per end: near = sum|turn| over the first `JUNCTION_REACH_PX` of the fitted curve; baseline =
    the larger of the curve's own turning rate over the next 2x reach (times the reach) and,
    when `raw` is known, the net heading change of the smoothed crack path over the reach
    (`_raw_end_turn`: a true tip or corner turns there too; pixel stairs cancel).
    A spike is near - baseline >= `JUNCTION_TURN_RAD`. Score = summed excess (rad). Only
    smoothly fitted curves are checked (exact pixel polylines turn 90 deg at every step).
    """
    ends = Counter()
    for c in curves:
        if not c.closed and len(c.points) >= 2:
            ends[tuple(map(float, c.points[0]))] += 1
            ends[tuple(map(float, c.points[-1]))] += 1
    reach = int(round(JUNCTION_REACH_PX / JUNCTION_STEP_PX))
    spots: list[Spot] = []
    seen: set = set()
    for c in curves:
        if c.closed or not c.smooth or len(c.points) < 2:
            continue
        for pts in (c.points, c.points[::-1]):
            if ends[tuple(map(float, pts[0]))] < 2:
                continue
            dense = _resample(pts, JUNCTION_STEP_PX)
            near = _abs_turn(dense[: reach + 2])
            far_pts = dense[reach : 3 * reach + 1]
            far_len = _length(far_pts)
            baseline = _abs_turn(far_pts) / far_len * JUNCTION_REACH_PX if far_len > 1 else 0.0
            if c.raw is not None and len(c.raw) >= 2:
                raw = c.raw if np.allclose(c.raw[0], pts[0]) else c.raw[::-1]
                baseline = max(baseline, _raw_end_turn(raw, reach))
            excess = near - baseline
            box = _bbox(dense[: reach + 2], 0.5)
            if excess >= JUNCTION_TURN_RAD and (c.id, box) not in seen:  # short curves: once
                seen.add((c.id, box))
                note = f"curve {c.id}: {math.degrees(near):.0f} deg turn near junction"
                spots.append(Spot(box, excess, note))
    return _sorted(sum(s.severity for s in spots), spots)


# ---------------------------------------------------------------- worst patches


def _over_white(px: np.ndarray) -> np.ndarray:
    alpha = px[..., 3:4].astype(np.float64) / 255.0
    return px[..., :3].astype(np.float64) * alpha + 255.0 * (1.0 - alpha)


def detect_worst_patches(
    source: np.ndarray, render: np.ndarray, window: int = PATCH_PX
) -> DetectorResult:
    """Local windows where the render differs most from the source (both HxWx4 at one size).

    badness = mean dE2000 / `PATCH_DE_REF` + (1 - mean SSIM) / `PATCH_SSIM_REF` per window;
    the worst `PATCH_SPOTS` non-overlapping windows are spots; score = mean badness of the worst
    `PATCH_TOP_K`.
    """
    a, b = _over_white(source), _over_white(render)
    h, w = a.shape[:2]
    win = max(1, min(window, h, w))
    de = deltaE_ciede2000(rgb2lab(a / 255.0), rgb2lab(b / 255.0))
    ssim_map = np.ones((h, w))
    if min(h, w) >= 7:
        _, full = structural_similarity(a, b, channel_axis=2, data_range=255, full=True)
        ssim_map = full.mean(axis=2)
    de_w = uniform_filter(de, win, mode="nearest")
    ss_w = uniform_filter(ssim_map, win, mode="nearest")
    bad = de_w / PATCH_DE_REF + (1.0 - ss_w) / PATCH_SSIM_REF
    work = np.full_like(bad, -np.inf)  # only windows that lie fully inside the image
    lo, hi = win // 2, (win - 1) // 2
    work[lo : h - hi, lo : w - hi] = bad[lo : h - hi, lo : w - hi]
    spots: list[Spot] = []
    for _ in range(PATCH_SPOTS):
        y, x = np.unravel_index(int(np.argmax(work)), work.shape)
        if not np.isfinite(work[y, x]):
            break
        x0, y0 = float(x - lo), float(y - lo)
        box = (x0, y0, x0 + win, y0 + win)
        note = f"dE {de_w[y, x]:.2f}, SSIM {ss_w[y, x]:.3f}"
        spots.append(Spot(box, float(bad[y, x]), note))
        work[max(0, y - win + 1) : y + win, max(0, x - win + 1) : x + win] = -np.inf
    top = [s.severity for s in spots[:PATCH_TOP_K]]
    return DetectorResult(float(np.mean(top)) if top else 0.0, tuple(spots))


# ---------------------------------------------------------------- pipeline state -> geometry


def _signed_area(pts: np.ndarray) -> float:
    return _shoelace(pts) if len(pts) >= 3 else 0.0


def geometry_from_state(state) -> Geometry:
    """Detector geometry, in original px, from a finished `engine.pipeline.PipelineState`.

    Which curves are drawn as primitives and which as path segments is export's own decision
    (`export.plan_layers`), so the geometry is what the document draws."""
    from engine.stages.export import plan_layers
    from engine.stages.fit_geometry import MIN_CLOSED_LENGTH_PX, MIN_OPEN_LENGTH_PX

    scale = state.image.scale
    fitted = {f.curve_id: f for f in state.fitted}
    plans = plan_layers(state.regions, state.graph, state.fitted, state.preset.export)
    primitive_ids = {p.primitive_curve for p in plans if p.primitive_curve is not None}
    segment_ids = {
        cid
        for p in plans
        if not p.underlay and p.primitive_curve is None
        for loop in p.loops
        for cid, _ in loop
    }
    curves = []
    for c in state.graph.curves:
        f = fitted[c.id]
        pts = sample_segments(f.segments, GEOM_SAMPLE_ERROR_PX * scale)[0] / scale
        raw = np.asarray(c.points, np.float64)
        min_len = MIN_CLOSED_LENGTH_PX if c.closed else MIN_OPEN_LENGTH_PX
        smooth = (state.preset.fit.enabled and OUTSIDE not in (c.left, c.right)
                  and _length(raw) >= min_len * scale)
        prim = None
        drawn_segments = c.id in segment_ids
        if c.id in primitive_ids:
            prim = sample_shape(f.primitive, GEOM_SAMPLE_ERROR_PX * scale) / scale
            # walk it the curve's way round so `left` / `right` still hold
            if np.sign(_signed_area(prim)) != np.sign(_signed_area(raw)):
                prim = prim[::-1].copy()
            if not drawn_segments:
                pts, smooth = prim, True
        curves.append(CurveGeom(c.id, pts, c.left, c.right, c.closed, smooth, prim, raw / scale,
                                drawn_segments))
    return Geometry(tuple(curves), shapes_from_doc(state.doc), scale)


def detect_all(
    geometry: Geometry, source: np.ndarray, render: np.ndarray,
    edge_source: np.ndarray | None = None,
) -> dict[str, DetectorResult]:
    """Every detector. `source` and `render` are HxWx4 at one size for the patch comparison;
    `edge_source` (default `source`) is the input image in the geometry's px for straightness."""
    return {
        "crossings": detect_crossings(geometry.curves),
        "specks": detect_specks(geometry.shapes),
        "wobble": detect_wobble(geometry.curves),
        "thin_bands": detect_thin_bands(geometry.curves),
        "junction_spikes": detect_junction_spikes(geometry.curves),
        "straightness": detect_straightness(
            geometry.curves, source if edge_source is None else edge_source),
        "worst_patches": detect_worst_patches(source, render),
    }
