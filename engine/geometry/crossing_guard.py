"""No-crossing guard: stop fitted boundary curves from crossing their neighbours.

Every boundary curve is fitted on its own (see `engine.stages.fit_geometry`), so nothing ties a
fitted curve to the curves beside it. Where a region is a band thinner than the fit's
allowances (sub-pixel corner extrapolation and corner snapping up to `CORNER_SNAP_PX`, plus
`tolerance_px` on each side), a fitted curve can cross a neighbour and the region above it in
the stacked SVG pokes through the band (a fill spiking through a thin outline).

Raw boundaries (the crack-grid polylines from `build_boundary_graph`) never cross: each unit
crack edge belongs to exactly one curve, and curves meet only at junction endpoints. So a fitted
curve crossing another curve's raw boundary is always wrong, and so is two fitted curves crossing
each other away from the junctions they share.

`guard_crossings` runs after every curve is fitted:

1. Sample each fitted curve (and its primitive, if any) into a polyline in processing pixels
   (`sample_segments`, `sample_shape`).
2. `find_crossings`: proper intersections between a fitted polyline and another curve's raw or
   fitted polyline, ignoring points within a clearance of a junction both curves end at. A
   uniform grid keeps this near-linear in the total polyline length.
3. Repair the offending span (corner to corner, as fitted) of each fitted side, escalating per
   span: `LEVEL_CORNERS` refits it with its corners placed on the raw crack path (no sub-pixel
   extrapolation or snap overshoot), `LEVEL_REFIT` also halves the tolerance, and `LEVEL_RAW`
   replaces it with the crack path itself (through crack-edge midpoints). A crossing primitive is
   dropped (the region falls back to its fitted segments). Detection reruns after each round,
   since a repair can create a crossing with a third curve; after `MAX_REPAIR_ROUNDS` every
   still-crossing curve goes fully raw, which always terminates because raw paths never cross.

A curve that never crosses is returned as the very same `FittedCurve` object, so output without
crossings is byte-identical to the unguarded fit. Junction endpoints are never moved and closed
curves stay closed.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np

from engine.geometry.bezier import sample
from engine.geometry.corners import dense_samples, unit_pieces
from engine.geometry.segment_fit import fit_span, snap_axis
from engine.presets import FitParams
from engine.types import (
    Arc,
    BoundaryCurve,
    Circle,
    CubicBezier,
    Ellipse,
    FittedCurve,
    Line,
    Point,
    PrimitiveShape,
    Rect,
    Segment,
)

logger = logging.getLogger(__name__)

# Polyline kinds.
RAW, FITTED, PRIMITIVE = 0, 1, 2
# Span repair levels.
LEVEL_FIT, LEVEL_CORNERS, LEVEL_REFIT, LEVEL_RAW = 0, 1, 2, 3

SAMPLE_ERROR_PX = 0.25  # processing px; chord error of sampled curves, well under a pixel
# Original px (x scale where used). Curves ending at the same junction leave it in different
# directions, but within the fit tolerance of it their fitted shapes may brush each other there.
JUNCTION_CLEARANCE_PX = 1.0
CORNER_REACH_PX = 1.5  # original px; a repaired corner moves to a crack vertex this close, as in
#                        fit_geometry.CORNER_SNAP_PX, but measured along the curve's own path
# LEVEL_REFIT tolerance as a fraction of the fit tolerance. LEVEL_CORNERS keeps the full
# tolerance first: on happiness.jpg it clears every crossing for +10 nodes where halving costs +53.
REFIT_TOLERANCE_FACTOR = 0.5
MAX_REPAIR_ROUNDS = 6  # rounds of span-by-span escalation before whole curves go raw
# Hard stop. Escalation is monotone and raw paths never cross, so the loop ends long before
# this; it only guards against a bug looping forever.
MAX_GUARD_ROUNDS = 64
GRID_CELL_PX = 8.0  # processing px; detection grid cell (segments are split to fit cells)


# ---------------------------------------------------------------- sampling


def _bezier_points(seg: CubicBezier, max_error: float) -> np.ndarray:
    ctrl = np.array([seg.p0, seg.c1, seg.c2, seg.p1], np.float64)
    second = max(
        float(np.hypot(*(ctrl[0] - 2 * ctrl[1] + ctrl[2]))),
        float(np.hypot(*(ctrl[1] - 2 * ctrl[2] + ctrl[3]))),
    )
    # chord error of n uniform pieces <= max|B''| / (8 n^2), with max|B''| <= 6 * second
    n = max(1, math.ceil(math.sqrt(6.0 * second / (8.0 * max_error))))
    pts = sample(seg, np.linspace(0.0, 1.0, n + 1))
    pts[0], pts[-1] = seg.p0, seg.p1
    return pts


def _circle_steps(radius: float, angle: float, max_error: float) -> int:
    """Pieces needed to follow `angle` radians of a circle within `max_error` (sagitta)."""
    if radius <= max_error:
        return max(1, math.ceil(abs(angle) / (math.pi / 2)))
    step = 2.0 * math.acos(1.0 - max_error / radius)
    return max(1, math.ceil(abs(angle) / step))


def _arc_points(seg: Arc, max_error: float) -> np.ndarray:
    cx, cy = seg.center
    a0 = math.atan2(seg.p0[1] - cy, seg.p0[0] - cx)
    a1 = math.atan2(seg.p1[1] - cy, seg.p1[0] - cx)
    travel = (a1 - a0) % (2 * math.pi) if seg.sweep else -((a0 - a1) % (2 * math.pi))
    n = _circle_steps(seg.radius, travel, max_error)
    ang = a0 + travel * np.linspace(0.0, 1.0, n + 1)
    pts = np.stack([cx + seg.radius * np.cos(ang), cy + seg.radius * np.sin(ang)], axis=1)
    pts[0], pts[-1] = seg.p0, seg.p1
    return pts


def _segment_points(seg: Segment, max_error: float) -> np.ndarray:
    if isinstance(seg, Line):
        return np.array([seg.p0, seg.p1], np.float64)
    if isinstance(seg, Arc):
        return _arc_points(seg, max_error)
    return _bezier_points(seg, max_error)


def sample_segments(
    segments: Sequence[Segment], max_error: float = SAMPLE_ERROR_PX
) -> tuple[np.ndarray, np.ndarray]:
    """(points Nx2, tags N-1): a polyline within `max_error` of the chain; tags[i] is the index
    of the segment polyline edge i belongs to. Lines are exact; segment endpoints are kept."""
    if not segments:
        return np.zeros((0, 2)), np.zeros(0, np.int64)
    parts = [_segment_points(s, max_error) for s in segments]
    tags = np.concatenate([np.full(len(p) - 1, i, np.int64) for i, p in enumerate(parts)])
    pts = np.vstack([parts[0]] + [p[1:] for p in parts[1:]])
    return pts, tags


def _rotate_about(pts: np.ndarray, deg: float, origin: Point) -> np.ndarray:
    """Rotate by +deg about origin (SVG `rotate`, y-down: clockwise on screen)."""
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    d = pts - origin
    return np.stack([d[:, 0] * c - d[:, 1] * s, d[:, 0] * s + d[:, 1] * c], axis=1) + origin


def sample_shape(shape: PrimitiveShape, max_error: float = SAMPLE_ERROR_PX) -> np.ndarray:
    """Closed polyline (first point repeated last) of a primitive, in its own units."""
    if isinstance(shape, (Circle, Ellipse)):
        rx, ry = (shape.r, shape.r) if isinstance(shape, Circle) else (shape.rx, shape.ry)
        n = max(8, _circle_steps(max(rx, ry), 2 * math.pi, max_error))
        t = np.linspace(0.0, 2 * math.pi, n + 1)
        pts = np.stack([shape.cx + rx * np.cos(t), shape.cy + ry * np.sin(t)], axis=1)
        if isinstance(shape, Ellipse) and shape.angle_deg:
            pts = _rotate_about(pts, shape.angle_deg, (shape.cx, shape.cy))
        pts[-1] = pts[0]
        return pts
    if isinstance(shape, Rect):
        x, y = shape.x, shape.y
        box = np.array([(x, y), (x + shape.w, y), (x + shape.w, y + shape.h), (x, y + shape.h)])
        pts = _rotate_about(box, shape.angle_deg, (x, y)) if shape.angle_deg else box
        return np.vstack([pts, pts[:1]])
    pts = np.asarray(shape.points, np.float64)
    return np.vstack([pts, pts[:1]])


# ---------------------------------------------------------------- detection


class Track(NamedTuple):
    """A polyline of one curve: kind RAW / FITTED / PRIMITIVE, tags[i] labels edge i."""

    curve_id: int
    kind: int
    points: np.ndarray  # Nx2
    tags: np.ndarray  # N-1


Tag = tuple[int, int, int]  # (curve_id, kind, edge tag)


class Crossing(NamedTuple):
    a: Tag
    b: Tag
    point: Point


_CELL_OFFSET = 1 << 24  # cells; keeps packed cell keys positive for any sane coordinate


def _cell_entries(p0: np.ndarray, p1: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """(cell key, edge index) for every grid cell each edge passes through, without duplicates,
    sorted by cell key. Keys are absolute, so grids built separately can be joined.

    Edges are split into pieces no longer than a cell, so each piece's box spans <= 2x2 cells.
    """
    length = np.hypot(*(p1 - p0).T)
    pieces = np.maximum(1, np.ceil(length / cell)).astype(np.int64)
    edge = np.repeat(np.arange(len(p0)), pieces)
    k = np.arange(len(edge)) - np.repeat(np.cumsum(pieces) - pieces, pieces)
    t0 = (k / pieces[edge])[:, None]
    t1 = ((k + 1) / pieces[edge])[:, None]
    d = (p1 - p0)[edge]
    q0, q1 = p0[edge] + t0 * d, p0[edge] + t1 * d
    lo = np.floor(np.minimum(q0, q1) / cell).astype(np.int64) + _CELL_OFFSET
    hi = np.floor(np.maximum(q0, q1) / cell).astype(np.int64) + _CELL_OFFSET
    keys, owners = [], []
    for dx in (0, 1):
        for dy in (0, 1):
            cx, cy = lo[:, 0] + dx, lo[:, 1] + dy
            ok = (cx <= hi[:, 0]) & (cy <= hi[:, 1])
            keys.append(cx[ok] * (2 * _CELL_OFFSET) + cy[ok])
            owners.append(edge[ok])
    key, owner = np.concatenate(keys), np.concatenate(owners)
    order = np.lexsort((owner, key))
    key, owner = key[order], owner[order]
    first = np.r_[True, (key[1:] != key[:-1]) | (owner[1:] != owner[:-1])]
    return key[first], owner[first]


def _ranges(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Concatenated ranges starts[i] .. starts[i] + counts[i] - 1."""
    total = int(counts.sum())
    offsets = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    return np.repeat(starts, counts) + offsets


def _cell_pairs(keys: np.ndarray, owners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All (i, j) edge pairs, i before j, sharing a cell; `keys` must be sorted."""
    n = len(keys)
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    group_end = np.repeat(np.r_[starts[1:], n], np.diff(np.r_[starts, n]))
    count = group_end - np.arange(n) - 1
    first = np.repeat(np.arange(n), count)
    return owners[first], owners[_ranges(np.arange(n) + 1, count)]


class EdgeGrid(NamedTuple):
    """The edges of some tracks, bucketed by grid cell (see `grid_tracks`)."""

    p0: np.ndarray
    p1: np.ndarray
    curve: np.ndarray
    kind: np.ndarray
    tag: np.ndarray
    keys: np.ndarray  # sorted cell keys
    owners: np.ndarray  # edge index of each key


def grid_tracks(tracks: Sequence[Track]) -> EdgeGrid:
    """Bucket the edges of `tracks` into `GRID_CELL_PX` cells."""
    tracks = [t for t in tracks if len(t.points) >= 2]
    if not tracks:
        empty = np.zeros(0, np.int64)
        return EdgeGrid(np.zeros((0, 2)), np.zeros((0, 2)), empty, empty, empty, empty, empty)
    p0 = np.vstack([t.points[:-1] for t in tracks]).astype(np.float64)
    p1 = np.vstack([t.points[1:] for t in tracks]).astype(np.float64)
    curve = np.concatenate([np.full(len(t.points) - 1, t.curve_id, np.int64) for t in tracks])
    kind = np.concatenate([np.full(len(t.points) - 1, t.kind, np.int64) for t in tracks])
    tag = np.concatenate([np.asarray(t.tags, np.int64) for t in tracks])
    return EdgeGrid(p0, p1, curve, kind, tag, *_cell_entries(p0, p1, GRID_CELL_PX))


def _orient(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sign of the side of line a->b each p lies on (0 = on the line)."""
    cross = (b[:, 0] - a[:, 0]) * (p[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (p[:, 0] - a[:, 0])
    return np.sign(cross)


def find_crossings(
    tracks: Sequence[Track],
    ends: Mapping[int, tuple[Point, ...]],
    clearance: float,
    raw: EdgeGrid | None = None,
) -> list[Crossing]:
    """Proper crossings between polylines of different curves where at least one is not RAW.

    `ends` maps an open curve's id to its junction endpoints; a crossing closer than `clearance`
    to an endpoint both curves share is where they legitimately meet and is not reported. Each
    crossing is reported once per pair of edges, ordered by (a, b) tag. `raw`, if given, is the
    prebuilt `grid_tracks` of the RAW tracks (RAW tracks in `tracks` are then ignored), so a
    caller checking repeatedly grids the unchanging raw paths once. RAW-RAW pairs are never
    generated.
    """
    if raw is None:
        raw = grid_tracks([t for t in tracks if t.kind == RAW])
    other = grid_tracks([t for t in tracks if t.kind != RAW])
    if not len(other.p0):
        return []
    n = len(other.p0)
    i, j = _cell_pairs(other.keys, other.owners)  # non-RAW x non-RAW
    lo = np.searchsorted(raw.keys, other.keys, side="left")  # non-RAW x RAW in the same cell
    counts = np.searchsorted(raw.keys, other.keys, side="right") - lo
    i = np.concatenate([i, np.repeat(other.owners, counts)])
    j = np.concatenate([j, n + raw.owners[_ranges(lo, counts)]])
    p0, p1 = np.vstack([other.p0, raw.p0]), np.vstack([other.p1, raw.p1])
    curve = np.concatenate([other.curve, raw.curve])
    kind = np.concatenate([other.kind, raw.kind])
    tag = np.concatenate([other.tag, raw.tag])
    keep = curve[i] != curve[j]
    i, j = i[keep], j[keep]
    # Proper crossings only: each edge strictly straddles the other's line. Touching and
    # collinear overlap are not crossings; an exact pass through the other path's vertex is
    # missed too, but exact coincidences only arise where curves meet at a junction.
    hit = (_orient(p0[i], p0[j], p1[j]) * _orient(p1[i], p0[j], p1[j]) < 0) & (
        _orient(p0[j], p0[i], p1[i]) * _orient(p1[j], p0[i], p1[i]) < 0
    )
    i, j = i[hit], j[hit]
    if not len(i):
        return []
    lo, hi = np.minimum(i, j), np.maximum(i, j)
    pairs = np.unique(np.stack([lo, hi], axis=1), axis=0)
    found: list[Crossing] = []
    for a, b in pairs.tolist():
        da, db = p1[a] - p0[a], p1[b] - p0[b]
        denom = da[0] * db[1] - da[1] * db[0]
        s = ((p0[b][0] - p0[a][0]) * db[1] - (p0[b][1] - p0[a][1]) * db[0]) / denom
        x, y = p0[a] + s * da
        shared = set(ends.get(int(curve[a]), ())) & set(ends.get(int(curve[b]), ()))
        if any(math.hypot(x - e[0], y - e[1]) < clearance for e in shared):
            continue
        ta = (int(curve[a]), int(kind[a]), int(tag[a]))
        tb = (int(curve[b]), int(kind[b]), int(tag[b]))
        found.append(Crossing(min(ta, tb), max(ta, tb), (float(x), float(y))))
    return sorted(found)


# ---------------------------------------------------------------- span plans


@dataclass(frozen=True)
class SpanPlan:
    """How a curve's fitted chain was built, so single spans can be refitted.

    `points` is the sequence the spans were fitted on (corners placed; closed curves rotated to
    start at `offset` in the dense ring with the start repeated at the end) and span k is
    points[breaks[k]..breaks[k+1]]. `corners` holds the positions in `breaks` that are detected
    corners (the others are junction endpoints or the arbitrary splits of a cornerless loop).
    `spans` are the unsnapped per-span fits; the curve's segments are `snap_axis` of their
    concatenation, which keeps the segment count, so `span_sizes` maps segments to spans.
    """

    curve: BoundaryCurve
    points: tuple[Point, ...]
    breaks: tuple[int, ...]
    corners: frozenset[int]
    offset: int
    params: FitParams  # in processing pixels
    scale: float
    spans: tuple[tuple[Segment, ...], ...]

    @property
    def span_sizes(self) -> tuple[int, ...]:
        return tuple(len(s) for s in self.spans)


def plan_spans(
    curve: BoundaryCurve,
    points: tuple[Point, ...],
    breaks: tuple[int, ...],
    corners: frozenset[int],
    offset: int,
    params: FitParams,
    scale: float,
    fit: Callable[[Sequence[Point], FitParams], tuple[Segment, ...]] = fit_span,
) -> tuple[SpanPlan, tuple[Segment, ...]]:
    """Fit each span points[breaks[k]..breaks[k+1]] with `fit` and snap the chain:
    (plan, segments)."""
    spans = tuple(fit(points[a : b + 1], params) for a, b in zip(breaks, breaks[1:]))
    plan = SpanPlan(curve, points, breaks, corners, offset, params, scale, spans)
    return plan, snap_axis([s for span in spans for s in span], params)


class _CrackPath:
    """A curve's crack-grid polyline with arc-length positions of its vertices and dense samples.

    Closed curves are unrolled over three laps (-1, 0, +1) so a span or a corner's vertex may
    lie past the closing point.
    """

    def __init__(self, curve: BoundaryCurve) -> None:
        pts = curve.points
        lengths = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:])]
        cum = np.concatenate([[0.0], np.cumsum(lengths)])
        mids = [
            cum[m] + (k + 0.5) * lengths[m] / unit_pieces(a, b)
            for m, (a, b) in enumerate(zip(pts, pts[1:]))
            for k in range(unit_pieces(a, b))
        ]
        self.closed = curve.closed
        self.total = float(cum[-1])
        self.plain = dense_samples(pts, curve.closed)
        self.dense_s = np.asarray(mids if curve.closed else [0.0, *mids, self.total])
        verts = list(pts[:-1]) if curve.closed else list(pts)
        vs = cum[:-1] if curve.closed else cum
        plain, ps = list(self.plain), self.dense_s
        if curve.closed:
            verts, vs = verts * 3, np.concatenate([vs - self.total, vs, vs + self.total])
            plain, ps = plain * 3, np.concatenate([ps - self.total, ps, ps + self.total])
        self.verts, self.vert_s = verts, np.asarray(vs)  # both sorted by arc position
        self._mids, self._mid_s = plain, ps

    def position(self, dense: int) -> float:
        """Arc position of dense sample `dense`; closed indices past one lap continue the path."""
        if not self.closed:
            return float(self.dense_s[dense])
        n = len(self.dense_s)
        return float(self.dense_s[dense % n] + self.total * (dense // n))

    def sample_at(self, dense: int) -> Point:
        return self.plain[dense % len(self.plain)]

    def nearest_vertex(
        self, s: float, reach: float, lo: float, hi: float
    ) -> tuple[Point, float] | None:
        """The vertex closest to arc position s within `reach` and in [lo, hi) (the first on
        a tie), or None."""
        i0 = int(np.searchsorted(self.vert_s, max(lo, s - reach), side="left"))
        i1 = int(np.searchsorted(self.vert_s, s + reach, side="right"))
        cand = [i for i in range(i0, i1) if self.vert_s[i] < hi]
        if not cand:
            return None
        best = min(cand, key=lambda i: abs(self.vert_s[i] - s))
        return self.verts[best], float(self.vert_s[best])

    def midpoints_between(self, s0: float, s1: float) -> list[Point]:
        """Crack-edge midpoints strictly between arc positions s0 < s1, in path order.

        Chords between consecutive midpoints only cut the corner of the pixel inside each turn,
        where no other curve's crack path can run, so the polyline never crosses a neighbour's
        raw path; on a regular staircase the midpoints are collinear and merge away.
        """
        i0 = int(np.searchsorted(self._mid_s, s0, side="right"))
        i1 = int(np.searchsorted(self._mid_s, s1, side="left"))
        return self._mids[i0:i1]


def _merge_collinear(points: Sequence[Point]) -> list[Point]:
    out: list[Point] = []
    for p in points:
        if out and p == out[-1]:
            continue
        while len(out) >= 2:
            (ax, ay), (bx, by) = out[-2], out[-1]
            cross = (bx - ax) * (p[1] - by) - (by - ay) * (p[0] - bx)
            if cross != 0 or (bx - ax) * (p[0] - bx) + (by - ay) * (p[1] - by) <= 0:
                break
            out.pop()
        out.append(p)
    return out


class _Repair:
    """Per-span repair levels of one planned curve and the chain they produce."""

    def __init__(self, plan: SpanPlan) -> None:
        self.plan = plan
        self.levels = [LEVEL_FIT] * len(plan.spans)
        self.path = _CrackPath(plan.curve)
        self._placed: dict[int, tuple[Point, float]] = {}
        self._sizes = list(plan.span_sizes)
        self._memo: dict[tuple[int, int, bool, bool], tuple[Segment, ...]] = {}

    def span_of(self, segment: int) -> int:
        return int(np.searchsorted(np.cumsum(self._sizes), segment, side="right"))

    def escalate(self, span: int) -> bool:
        if self.levels[span] >= LEVEL_RAW:
            return False
        self.levels[span] += 1
        return True

    def escalate_all(self) -> bool:
        changed = any(lv < LEVEL_RAW for lv in self.levels)
        self.levels = [LEVEL_RAW] * len(self.levels)
        return changed

    # -- break placement

    def _dense(self, position: int) -> int:
        return self.plan.offset + position

    def _fixed(self, k: int) -> bool:
        """Junction endpoints of an open curve never move."""
        return not self.plan.curve.closed and k in (0, len(self.plan.breaks) - 1)

    def _raw_break(self, k: int) -> tuple[Point, float]:
        """Break k on the crack path: a detected corner goes to the nearest crack vertex along
        the path within reach, anything else to its crack-edge midpoint. Cached, so a closed
        curve's first and last break stay the same point.

        A corner may only take a vertex from its own cell [halfway from the previous break,
        halfway to the next), so neighbouring corners can never collapse onto one vertex and
        placed breaks stay in path order."""
        plan, path = self.plan, self.path
        n_breaks = len(plan.breaks)
        key = 0 if plan.curve.closed and k == n_breaks - 1 else k
        if key in self._placed:
            p, s = self._placed[key]
            return (p, s + path.total) if k != key else (p, s)
        dense = self._dense(plan.breaks[key])
        s = path.position(dense)
        placed = (path.sample_at(dense), s)
        if key in plan.corners:
            before = path.position(self._dense(plan.breaks[key - 1])) if key > 0 else (
                path.position(self._dense(plan.breaks[-2])) - path.total
            )
            after = path.position(self._dense(plan.breaks[key + 1]))
            lo, hi = (before + s) / 2, (s + after) / 2
            hit = path.nearest_vertex(s, CORNER_REACH_PX * plan.scale, lo, hi)
            placed = hit or placed
        self._placed[key] = placed
        return self._raw_break(k)

    def _break_moves(self, k: int) -> bool:
        """A break leaves its fitted placement once a span on either side is repaired."""
        if self._fixed(k):
            return False
        n = len(self.levels)
        adjacent = (n - 1, 0) if self.plan.curve.closed and k in (0, n) else (k - 1, k)
        return any(0 <= i < n and self.levels[i] >= LEVEL_CORNERS for i in adjacent)

    # -- spans

    def _span(self, k: int) -> tuple[Segment, ...]:
        key = (k, self.levels[k], self._break_moves(k), self._break_moves(k + 1))
        if key not in self._memo:
            self._memo[key] = self._fit(*key)
        return self._memo[key]

    def _fit(self, k: int, level: int, move_a: bool, move_b: bool) -> tuple[Segment, ...]:
        plan = self.plan
        a, b = plan.breaks[k], plan.breaks[k + 1]
        if level == LEVEL_FIT and not (move_a or move_b):
            return plan.spans[k]
        start = self._raw_break(k) if move_a else (plan.points[a], None)
        end = self._raw_break(k + 1) if move_b else (plan.points[b], None)
        s0 = start[1] if start[1] is not None else self.path.position(self._dense(a))
        s1 = end[1] if end[1] is not None else self.path.position(self._dense(b))
        if level == LEVEL_RAW:  # both ends are on the crack path: moved, or junction ends
            pts = _merge_collinear([start[0], *self.path.midpoints_between(s0, s1), end[0]])
            return tuple(Line(p, q) for p, q in zip(pts, pts[1:]))
        # Samples whose crack edge lies past a moved end would make the span double back.
        inner = [
            plan.points[i]
            for i in range(a + 1, b)
            if s0 < self.path.position(self._dense(i)) < s1
        ]
        pts = (start[0], *inner, end[0])
        params = plan.params
        if level == LEVEL_REFIT:
            params = replace(params, tolerance_px=params.tolerance_px * REFIT_TOLERANCE_FACTOR)
        return fit_span(pts, params)

    def build(self) -> FittedCurve:
        spans = [self._span(k) for k in range(len(self.levels))]
        self._sizes = [len(s) for s in spans]
        chain = [s for span in spans for s in span]
        bounds = np.cumsum([0, *self._sizes])
        locked = frozenset(
            i
            for k, level in enumerate(self.levels)
            if level >= LEVEL_CORNERS
            for i in range(bounds[k], bounds[k + 1])
        )
        return FittedCurve(self.plan.curve.id, snap_axis(chain, self.plan.params, locked), None)


# ---------------------------------------------------------------- guard


def _raw_tracks(curves: Sequence[BoundaryCurve]) -> list[Track]:
    return [
        Track(c.id, RAW, np.asarray(c.points, np.float64), np.zeros(len(c.points) - 1, np.int64))
        for c in curves
        if len(c.points) >= 2
    ]


def _fitted_tracks(f: FittedCurve, planned: bool) -> list[Track]:
    out = []
    if planned:  # unplanned curves are exact polylines: their fit is their raw path
        pts, tags = sample_segments(f.segments)
        out.append(Track(f.curve_id, FITTED, pts, tags))
    if f.primitive is not None:
        pts = sample_shape(f.primitive)
        out.append(Track(f.curve_id, PRIMITIVE, pts, np.zeros(len(pts) - 1, np.int64)))
    return out


def _tracks(
    curves: Sequence[BoundaryCurve], fitted: Mapping[int, FittedCurve], planned: set[int]
) -> list[Track]:
    """RAW tracks of every curve plus FITTED / PRIMITIVE tracks of the fitted curves."""
    out = _raw_tracks(curves)
    for cid, f in fitted.items():
        out.extend(_fitted_tracks(f, cid in planned))
    return out


def guard_crossings(
    curves: Sequence[BoundaryCurve],
    fitted: Sequence[FittedCurve],
    plans: Mapping[int, SpanPlan],
    scale: float,
) -> tuple[FittedCurve, ...]:
    """`fitted` with every crossing between neighbouring curves repaired (see module docstring).

    `plans` holds a `SpanPlan` for each smoothly fitted curve; curves without one are exact
    polylines (they can only be crossed, never cross) unless they carry a primitive.

    Each round escalates every offending span by exactly one level, however many of its
    segments cross. A crossing with a primitive only drops that primitive; the other side is
    re-checked next round against the segments that replace it.
    """
    by_id = {f.curve_id: f for f in fitted}
    ends = {c.id: (c.points[0], c.points[-1]) for c in curves if not c.closed and c.points}
    clearance = JUNCTION_CLEARANCE_PX * scale
    repairs: dict[int, _Repair] = {}
    raw = grid_tracks(_raw_tracks(curves))
    sampled = {cid: _fitted_tracks(f, cid in plans) for cid, f in by_id.items()}
    crossings: list[Crossing] = []
    for round_no in range(MAX_GUARD_ROUNDS + 1):
        tracks = [t for ts in sampled.values() for t in ts]
        crossings = find_crossings(tracks, ends, clearance, raw)
        if not crossings or round_no == MAX_GUARD_ROUNDS:
            break
        primitives = {t[0] for x in crossings for t in (x.a, x.b) if t[1] == PRIMITIVE}
        spans: set[tuple[int, int]] = set()
        for x in crossings:
            if PRIMITIVE in (x.a[1], x.b[1]):
                continue
            for cid, kind, tag in (x.a, x.b):
                if kind == FITTED:
                    if cid not in repairs:
                        repairs[cid] = _Repair(plans[cid])
                    spans.add((cid, repairs[cid].span_of(tag)))
        changed = set(primitives)
        for cid in primitives:
            by_id[cid] = replace(by_id[cid], primitive=None)
        whole = round_no >= MAX_REPAIR_ROUNDS
        for cid, span in sorted(spans):
            rep = repairs[cid]
            if rep.escalate_all() if whole else rep.escalate(span):
                changed.add(cid)
        for cid in changed:
            if cid in repairs:
                by_id[cid] = replace(repairs[cid].build(), primitive=by_id[cid].primitive)
            sampled[cid] = _fitted_tracks(by_id[cid], cid in plans)
        if not changed:  # pragma: no cover - raw paths never cross, so escalation always helps
            break
    if crossings:
        logger.warning("crossing guard: %d fitted curve crossings left unrepaired", len(crossings))
    return tuple(by_id[f.curve_id] for f in fitted)
