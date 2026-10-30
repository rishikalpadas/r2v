"""Shared-boundary tracing: turn a RegionMap into curves that neighboring regions share.

Every boundary between two labels is traced exactly once into a `BoundaryCurve`; each region's
loops then reference those curves (possibly reversed), so adjacent regions are guaranteed to use
identical geometry along their common border and the output can never have gaps or overlaps.

Conventions
-----------
* Coordinates are pixel-corner grid points in processing-pixel units (the RegionMap's grid,
  i.e. original pixels x `RasterImage.scale`), x right, y down: pixel (x, y) covers
  [x, x+1] x [y, y+1]. Export maps them back to original pixels.
* Orientation: "left" of a walking direction (dx, dy) is (-dy, dx), with coordinates used as-is
  (no y flip). Every curve keeps its larger label on the left, so `left` is always a region id
  and `right` may be a region id, `TRANSPARENT` or `OUTSIDE`. A region's loops are walked with the
  region on the left, so with the plain shoelace formula on (x, y) an outer loop has positive
  signed area and a hole has negative signed area.
* Curve points keep only the vertices where the direction changes plus the endpoints (collinear
  runs are collapsed). Closed curves repeat their first point as their last; their first point
  is always a direction-change vertex.
* A grid corner is a junction when >= 3 labels touch it or 2 labels touch it diagonally
  (checkerboard). Curves start and end only at junctions; boundaries with no junction become
  `closed=True` curves. At a diagonal corner a region's loop turns left, i.e. it hugs the same
  pixel, matching the 4-connectivity of regions.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from engine.types import OUTSIDE, BoundaryCurve, BoundaryGraph, LoopRef, RegionMap


class _Edges(NamedTuple):
    """Directed unit crack edges; corners are flat indices y * stride + x."""

    start: np.ndarray
    end: np.ndarray
    left: np.ndarray
    right: np.ndarray


class _Chains(NamedTuple):
    """Edges grouped into chains: `chain[i]` is the curve id of edge i, `rank[i]` its position."""

    chain: np.ndarray
    rank: np.ndarray
    closed: np.ndarray  # per chain


class _HalfCurves(NamedTuple):
    """One side of a curve as seen by the region on that side, walked with that region on the left.

    Corners are flat indices; directions are unit (dx, dy) of the first and last segments.
    """

    region: np.ndarray
    curve: np.ndarray
    reversed: np.ndarray
    start: np.ndarray
    end: np.ndarray
    first_dx: np.ndarray
    first_dy: np.ndarray
    last_dx: np.ndarray
    last_dy: np.ndarray
    area2: np.ndarray  # doubled signed shoelace area contribution


def build_boundary_graph(regions: RegionMap) -> BoundaryGraph:
    """Trace every label boundary once and assemble each region's outer loop and holes."""
    labels = np.asarray(regions.labels)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"labels must be a 2-D integer array, got {labels.dtype} {labels.shape}")
    h, w = labels.shape
    stride = w + 1
    padded = np.pad(labels.astype(np.int64), 1, constant_values=OUTSIDE)
    edges = _crack_edges(padded, h, w)
    if len(edges.start) == 0:
        return BoundaryGraph(curves=(), region_loops={})
    chains = _chain_edges(edges, _junction_corners(padded).ravel())
    curves, halves = _build_curves(edges, chains, stride)
    return BoundaryGraph(curves=curves, region_loops=_assemble_loops(halves))


def _junction_corners(padded: np.ndarray) -> np.ndarray:
    """(h+1, w+1) bool: corners touched by >= 3 labels or by 2 labels in a diagonal pattern."""
    a, b = padded[:-1, :-1], padded[:-1, 1:]  # top-left, top-right
    c, d = padded[1:, :-1], padded[1:, 1:]  # bottom-left, bottom-right
    distinct = (
        1
        + (b != a)
        + ((c != a) & (c != b))
        + ((d != a) & (d != b) & (d != c))
    )
    diagonal = (a == d) & (b == c) & (a != b)
    return (distinct >= 3) | diagonal


def _crack_edges(padded: np.ndarray, h: int, w: int) -> _Edges:
    """All unit edges between differently-labelled pixels, oriented with the larger label left.

    Edges whose both sides are non-region (TRANSPARENT / OUTSIDE) are dropped.
    """
    stride = w + 1
    # Vertical edges x in [0, w], y in [0, h): left pixel a, right pixel b.
    a, b = padded[1 : h + 1, :-1], padded[1 : h + 1, 1:]
    ys, xs = np.nonzero(a != b)
    la, lb = a[ys, xs], b[ys, xs]
    top, bottom = ys * stride + xs, (ys + 1) * stride + xs
    down = la > lb  # walking +y keeps the left pixel (a) on the left
    v = (np.where(down, top, bottom), np.where(down, bottom, top), la, lb)
    # Horizontal edges y in [0, h], x in [0, w): pixel above a, pixel below b.
    a, b = padded[:-1, 1 : w + 1], padded[1:, 1 : w + 1]
    ys, xs = np.nonzero(a != b)
    la, lb = a[ys, xs], b[ys, xs]
    west, east = ys * stride + xs, ys * stride + xs + 1
    forward = lb > la  # walking +x keeps the pixel below (b) on the left
    hz = (np.where(forward, west, east), np.where(forward, east, west), la, lb)

    start = np.concatenate([v[0], hz[0]])
    end = np.concatenate([v[1], hz[1]])
    la = np.concatenate([v[2], hz[2]])
    lb = np.concatenate([v[3], hz[3]])
    left, right = np.maximum(la, lb), np.minimum(la, lb)
    keep = left >= 0
    return _Edges(start[keep], end[keep], left[keep], right[keep])


def _chain_edges(edges: _Edges, junction: np.ndarray) -> _Chains:
    """Link edges through non-junction corners into chains and rank edges along each chain."""
    n = len(edges.start)
    idx = np.arange(n)
    out_edge = np.full(len(junction), -1, np.int64)
    through = ~junction[edges.start]
    out_edge[edges.start[through]] = idx[through]
    nxt = np.where(junction[edges.end], -1, out_edge[edges.end])
    linked = nxt >= 0
    prev = np.full(n, -1, np.int64)
    prev[nxt[linked]] = idx[linked]

    graph = coo_matrix((np.ones(linked.sum()), (idx[linked], nxt[linked])), shape=(n, n))
    n_chains, chain = connected_components(graph, directed=True, connection="weak")
    has_head = np.zeros(n_chains, bool)
    has_head[chain[prev < 0]] = True

    # Cut each cycle at its lowest-index edge that starts with a direction change.
    direction = edges.end - edges.start
    turning = (prev < 0) | (direction != direction[prev])
    cut = np.full(n_chains, n, np.int64)
    np.minimum.at(cut, chain[turning], idx[turning])
    heads = cut[~has_head]
    prev[heads] = -1
    return _Chains(chain=chain, rank=_list_rank(prev), closed=~has_head)


def _list_rank(prev: np.ndarray) -> np.ndarray:
    """Distance of every node to the head of its list (Wyllie pointer jumping)."""
    rank = (prev >= 0).astype(np.int64)
    ptr = prev.copy()
    active = np.flatnonzero(ptr >= 0)
    while active.size:
        hop = ptr[active]
        rank[active] += rank[hop]
        ptr[active] = ptr[hop]
        active = active[ptr[active] >= 0]
    return rank


def _build_curves(
    edges: _Edges, chains: _Chains, stride: int
) -> tuple[tuple[BoundaryCurve, ...], _HalfCurves]:
    """Turn ranked chains into curves with collinear runs collapsed."""
    order = np.lexsort((chains.rank, chains.chain))
    start, end = edges.start[order], edges.end[order]
    chain, rank = chains.chain[order], chains.rank[order]
    direction = end - start
    keep = (rank == 0) | (direction != np.roll(direction, 1))
    is_last = np.append(chain[1:] != chain[:-1], True)

    keys = np.concatenate([2 * np.flatnonzero(keep), 2 * np.flatnonzero(is_last) + 1])
    corners = np.concatenate([start[keep], end[is_last]])
    owner = np.concatenate([chain[keep], chain[is_last]])
    by_key = np.argsort(keys, kind="stable")
    corners, owner = corners[by_key], owner[by_key]
    xs, ys = corners % stride, corners // stride

    n_chains = len(chains.closed)
    offsets = np.concatenate([[0], np.cumsum(np.bincount(owner, minlength=n_chains))])
    first = np.flatnonzero(rank == 0)  # one per chain, in chain order
    left, right = edges.left[order][first], edges.right[order][first]
    sx, sy, ex, ey = start % stride, start // stride, end % stride, end // stride
    area2 = np.rint(np.bincount(chain, weights=sx * ey - ex * sy, minlength=n_chains))

    pts = list(zip(xs.astype(np.float64).tolist(), ys.astype(np.float64).tolist()))
    off = offsets.tolist()
    curves = tuple(
        BoundaryCurve(id=i, points=tuple(pts[off[i] : off[i + 1]]), left=l, right=r, closed=c)
        for i, (l, r, c) in enumerate(zip(left.tolist(), right.tolist(), chains.closed.tolist()))
    )
    lo, hi = offsets[:-1], offsets[1:] - 1
    curve_ends = (
        corners[lo],
        corners[hi],
        np.sign(xs[lo + 1] - xs[lo]),
        np.sign(ys[lo + 1] - ys[lo]),
        np.sign(xs[hi] - xs[hi - 1]),
        np.sign(ys[hi] - ys[hi - 1]),
    )
    return curves, _half_curves(left, right, curve_ends, area2.astype(np.int64))


def _half_curves(
    left: np.ndarray, right: np.ndarray, curve_ends: tuple[np.ndarray, ...], area2: np.ndarray
) -> _HalfCurves:
    """Left sides walk curves forward; region right sides walk them reversed."""
    start, end, fdx, fdy, ldx, ldy = curve_ends
    ids = np.arange(len(left))
    rev = right >= 0
    r = ids[rev]
    cat = np.concatenate
    return _HalfCurves(
        region=cat([left, right[rev]]),
        curve=cat([ids, r]),
        reversed=cat([np.zeros(len(ids), bool), np.ones(len(r), bool)]),
        start=cat([start, end[rev]]),
        end=cat([end, start[rev]]),
        first_dx=cat([fdx, -ldx[rev]]),
        first_dy=cat([fdy, -ldy[rev]]),
        last_dx=cat([ldx, -fdx[rev]]),
        last_dy=cat([ldy, -fdy[rev]]),
        area2=cat([area2, -area2[rev]]),
    )


def _successors(halves: _HalfCurves) -> np.ndarray:
    """Index of the half-curve that follows each half-curve around its region.

    Candidates start where the half ends, on the same region. Only a diagonal junction offers
    two; then the left turn is taken so the loop hugs the same pixel (4-connectivity).
    """
    span = int(max(halves.start.max(), halves.end.max())) + 1
    key_out = halves.region * span + halves.start
    key_in = halves.region * span + halves.end
    by_out = np.argsort(key_out, kind="stable")
    sorted_out = key_out[by_out]
    pos = np.searchsorted(sorted_out, key_in)
    if pos.max() >= len(sorted_out) or (sorted_out[pos] != key_in).any():
        raise RuntimeError("a boundary curve has no continuation for its region")
    first = by_out[pos]
    nxt_pos = np.minimum(pos + 1, len(sorted_out) - 1)
    has_second = (pos + 1 < len(sorted_out)) & (sorted_out[nxt_pos] == key_in)
    turn = halves.last_dx * halves.first_dy[first] - halves.last_dy * halves.first_dx[first]
    return np.where(has_second & (turn <= 0), by_out[nxt_pos], first)


def _assemble_loops(halves: _HalfCurves) -> dict[int, tuple[LoopRef, ...]]:
    """Follow each region's half-curves end to end into loops; outer loop (largest |area|) first."""
    n = len(halves.region)
    idx = np.arange(n)
    succ = _successors(halves)
    if (np.bincount(succ, minlength=n) != 1).any():
        raise RuntimeError("region boundary loops do not close")
    graph = coo_matrix((np.ones(n), (idx, succ)), shape=(n, n))
    n_loops, loop = connected_components(graph, directed=True, connection="weak")
    prev = np.empty(n, np.int64)
    prev[succ] = idx
    head = np.full(n_loops, n, np.int64)
    np.minimum.at(head, loop, idx)
    prev[head] = -1
    rank = _list_rank(prev)

    region = halves.region[head]
    area = np.abs(np.bincount(loop, weights=halves.area2, minlength=n_loops))
    loop_order = np.lexsort((np.arange(n_loops), -area, region))
    position = np.empty(n_loops, np.int64)
    position[loop_order] = np.arange(n_loops)
    walk = np.lexsort((rank, position[loop]))

    refs = list(zip(halves.curve[walk].tolist(), halves.reversed[walk].tolist()))
    offsets = np.concatenate([[0], np.cumsum(np.bincount(loop, minlength=n_loops)[loop_order])])
    off = offsets.tolist()
    result: dict[int, list[LoopRef]] = {}
    for i, r in enumerate(region[loop_order].tolist()):
        result.setdefault(r, []).append(tuple(refs[off[i] : off[i + 1]]))
    return {r: tuple(loops) for r, loops in result.items()}
