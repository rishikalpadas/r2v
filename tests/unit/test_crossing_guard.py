import contextlib
import math

import numpy as np
import pytest

from engine.geometry.bezier import sample
from engine.geometry.crossing_guard import (
    FITTED,
    LEVEL_CORNERS,
    LEVEL_FIT,
    LEVEL_RAW,
    PRIMITIVE,
    RAW,
    Track,
    find_crossings,
    guard_crossings,
    plan_spans,
    sample_segments,
    sample_shape,
)
from engine.presets import FitParams
from engine.types import (
    Arc,
    BoundaryCurve,
    Circle,
    CubicBezier,
    Ellipse,
    FittedCurve,
    Line,
    Polygon,
    Rect,
)


def _track(curve_id, kind, points):
    pts = np.asarray(points, np.float64)
    return Track(curve_id, kind, pts, np.arange(len(pts) - 1))


# ---------------------------------------------------------------- detector


def test_crossing_is_found_with_its_point_and_tags():
    a = _track(1, FITTED, [(0, 0), (10, 10)])
    b = _track(2, RAW, [(0, 10), (10, 0)])
    (hit,) = find_crossings([a, b], ends={}, clearance=1.0)
    assert {hit.a, hit.b} == {(1, FITTED, 0), (2, RAW, 0)}
    assert hit.point == pytest.approx((5.0, 5.0))


def test_disjoint_and_parallel_tracks_do_not_cross():
    a = _track(1, FITTED, [(0, 0), (10, 0)])
    b = _track(2, RAW, [(0, 1), (10, 1)])
    c = _track(3, FITTED, [(20, 20), (30, 25)])
    assert find_crossings([a, b, c], ends={}, clearance=1.0) == []


def test_touching_without_crossing_is_not_a_crossing():
    # a touches b's vertex and turns back; then runs along b (collinear overlap)
    a = _track(1, FITTED, [(0, -5), (5, 0), (10, -5)])
    b = _track(2, RAW, [(0, 0), (5, 0), (10, 0)])
    c = _track(3, FITTED, [(2, 0), (8, 0)])
    assert find_crossings([a, b, c], ends={}, clearance=1.0) == []


def test_crossing_counts_once_per_pair_of_edges():
    a = _track(1, FITTED, [(5, -5), (5, -0.5), (5.2, 0.5), (5, 5)])
    b = _track(2, RAW, [(0, 0), (4, 0), (10, 0)])
    (hit,) = find_crossings([a, b], ends={}, clearance=1.0)
    assert hit.a == (1, FITTED, 1) and hit.b == (2, RAW, 1)


def test_crossings_near_a_shared_junction_are_excluded():
    # both curves end at the junction (0, 0); a crosses b at distance ~0.7 and ~4.2
    ends = {1: ((0.0, 0.0), (10.0, 0.0)), 2: ((0.0, 0.0), (0.0, 10.0))}
    near = _track(1, FITTED, [(0, 0), (-0.5, 0.5), (10, 0)])
    assert find_crossings([near, _track(2, RAW, [(0, 0), (0, 10)])], ends, clearance=1.0) == []
    far = _track(1, FITTED, [(0, 0), (1, 1), (-3, 3), (10, 0)])
    hits = find_crossings([far, _track(2, RAW, [(0, 0), (0, 10)])], ends, clearance=1.0)
    assert len(hits) == 2 and all(h.point[1] > 1.0 for h in hits)


def test_unshared_endpoints_are_not_excluded():
    ends = {1: ((0.0, 0.0), (10.0, 0.0)), 2: ((0.5, -5.0), (0.5, 5.0))}
    a = _track(1, FITTED, [(0, 0), (10, 0)])
    b = _track(2, RAW, [(0.5, -5), (0.5, 5)])
    assert len(find_crossings([a, b], ends, clearance=1.0)) == 1


def test_raw_pairs_and_same_curve_pairs_are_ignored():
    a_raw = _track(1, RAW, [(0, 0), (10, 10)])
    b_raw = _track(2, RAW, [(0, 10), (10, 0)])
    a_fit = _track(1, FITTED, [(0, 10), (10, 0)])  # crosses a's own raw, overlaps b's raw
    assert find_crossings([a_raw, b_raw, a_fit], ends={}, clearance=1.0) == []


def test_long_segments_across_many_cells_report_one_crossing():
    a = _track(1, FITTED, [(0, 0), (400, 300)])
    b = _track(2, RAW, [(0, 300), (400, 0)])
    hits = find_crossings([a, b], ends={}, clearance=1.0)
    assert len(hits) == 1 and hits[0].point == pytest.approx((200.0, 150.0))


def test_fitted_against_fitted_and_primitive_are_checked():
    a = _track(1, FITTED, [(0, 0), (10, 10)])
    b = _track(2, FITTED, [(0, 10), (10, 0)])
    c = _track(3, PRIMITIVE, [(0, 5), (10, 5)])
    kinds = sorted(tuple(sorted((h.a[1], h.b[1]))) for h in find_crossings([a, b, c], {}, 1.0))
    assert kinds == sorted([(FITTED, FITTED), (FITTED, PRIMITIVE), (FITTED, PRIMITIVE)])


def test_empty_input():
    assert find_crossings([], ends={}, clearance=1.0) == []


# ---------------------------------------------------------------- sampling


def test_sample_segments_keeps_lines_exact_and_tags_segments():
    segs = (Line((0, 0), (10, 0)), Line((10, 0), (10, 5)))
    pts, tags = sample_segments(segs, max_error=0.25)
    assert pts.tolist() == [[0, 0], [10, 0], [10, 5]]
    assert tags.tolist() == [0, 1]


def test_sample_segments_bezier_within_chord_error():
    seg = CubicBezier((0, 0), (0, 40), (60, 40), (60, 0))
    pts, tags = sample_segments((seg,), max_error=0.25)
    assert tuple(pts[0]) == seg.p0 and tuple(pts[-1]) == seg.p1
    dense = sample(seg, np.linspace(0, 1, 4001))
    # distance of every true curve point to the polyline
    a, b = pts[:-1], pts[1:]
    ab = b - a
    t = np.clip(((dense[:, None, :] - a) * ab).sum(2) / (ab * ab).sum(1), 0, 1)
    d = np.hypot(*(dense[:, None, :] - (a + t[..., None] * ab)).transpose(2, 0, 1)).min(1)
    assert d.max() <= 0.25
    assert set(tags.tolist()) == {0}


@pytest.mark.parametrize("sweep, large", [(True, False), (False, True)])  # centre fixes large
def test_sample_segments_arc_points_on_circle(sweep, large):
    # circle centre (5, 5) radius 5, p0 (0, 5) -> p1 (5, 0): a quarter turn one way, 3/4 the other
    centre = (5.0, 5.0)
    arc = Arc((0.0, 5.0), (5.0, 0.0), centre, 5.0, large_arc=large, sweep=sweep)
    pts, _ = sample_segments((arc,), max_error=0.05)
    assert tuple(pts[0]) == arc.p0 and tuple(pts[-1]) == arc.p1
    r = np.hypot(pts[:, 0] - 5, pts[:, 1] - 5)
    assert np.allclose(r, 5.0)
    ang = np.unwrap(np.arctan2(pts[:, 1] - 5, pts[:, 0] - 5))
    travel = ang[-1] - ang[0]
    assert (travel > 0) == sweep
    assert (abs(travel) > math.pi) == large
    mids = (pts[:-1] + pts[1:]) / 2
    assert (5.0 - np.hypot(mids[:, 0] - 5, mids[:, 1] - 5)).max() <= 0.05


@pytest.mark.parametrize(
    "shape",
    [
        Circle(10, 10, 5),
        Ellipse(10, 10, 6, 3, 30.0),
        Rect(2, 3, 8, 4, 15.0),
        Polygon(((0, 0), (4, 0), (2, 3))),
    ],
)
def test_sample_shape_is_closed(shape):
    pts = sample_shape(shape, max_error=0.25)
    assert len(pts) >= 4 and tuple(pts[0]) == pytest.approx(tuple(pts[-1]))


def test_sample_shape_rect_rotates_about_top_left():
    pts = sample_shape(Rect(1, 1, 4, 2, 90.0), max_error=0.25)
    assert np.allclose(pts, [[1, 1], [1, 5], [-1, 5], [-1, 1], [1, 1]])


# ---------------------------------------------------------------- repair escalation

# Curve A: open, junctions (0, 0) and (10, 10), one pixel corner at (0, 10).
_A = BoundaryCurve(
    id=0, points=((0.0, 0.0), (0.0, 10.0), (10.0, 10.0)), left=1, right=2, closed=False
)
_PARAMS = FitParams(tolerance_px=0.75)


def _dense_a():
    from engine.geometry.corners import dense_samples

    return list(dense_samples(_A.points, False))


def _plan(points, breaks, corners):
    return plan_spans(_A, tuple(points), tuple(breaks), frozenset(corners), 0, _PARAMS, 1.0)


@contextlib.contextmanager
def _max_levels():
    """Collect the highest span level of every repaired curve, per build."""
    import engine.geometry.crossing_guard as cg

    seen: list[int] = []
    build = cg._Repair.build

    def spy(self):
        seen.append(max(self.levels))
        return build(self)

    cg._Repair.build = spy
    try:
        yield seen
    finally:
        cg._Repair.build = build


def _crossings(curves, fitted):
    tracks = [_track(c.id, RAW, c.points) for c in curves]
    for f in fitted:
        pts, tags = sample_segments(f.segments, 0.25)
        tracks.append(Track(f.curve_id, FITTED, pts, tags))
    ends = {c.id: (c.points[0], c.points[-1]) for c in curves if not c.closed}
    return find_crossings(tracks, ends, clearance=1.0)


def test_repair_level_one_puts_the_corner_on_its_pixel_vertex():
    dense = _dense_a()  # 22 samples; the corner (0, 10) sits between dense[10] and dense[11]
    dense[10] = (-1.5, 11.5)  # sub-pixel extrapolation overshooting the corner
    plan, segs = _plan(dense, (0, 10, len(dense) - 1), {1})
    # B runs diagonally just outside A's pixel corner: never crosses A's crack path
    b = BoundaryCurve(id=1, points=((-3.0, 9.0), (1.0, 13.0)), left=3, right=1, closed=False)
    a_fit = FittedCurve(0, segs, None)
    assert _crossings([_A, b], [a_fit])  # today's fit crosses B
    b_fit = FittedCurve(1, (Line(*b.points),), None)
    with _max_levels() as levels:
        out = guard_crossings([_A, b], [a_fit, b_fit], {0: plan}, 1.0)
    fixed = out[0]
    assert not _crossings([_A, b], [fixed])
    assert levels == [LEVEL_CORNERS]  # the first repair level was enough: never went RAW
    assert fixed.segments[0].p0 == _A.points[0] and fixed.segments[-1].p1 == _A.points[-1]
    corner_pts = {s.p1 for s in fixed.segments[:-1]}
    assert (0.0, 10.0) in corner_pts  # moved onto the pixel vertex, not the crack midpoint
    assert out[1] is b_fit  # an exact neighbour is never touched


def test_span_escalates_once_per_round_however_many_segments_cross():
    # span 0 is fitted as a polyline through a shallow bulge: two of its segments cross B
    dense = _dense_a()
    for i in range(6, 10):
        dense[i] = (-0.4, dense[i][1])
    polyline = lambda pts, params: tuple(Line(p, q) for p, q in zip(pts, pts[1:]))  # noqa: E731
    plan, segs = plan_spans(_A, tuple(dense), (0, 10, 21), frozenset({1}), 0, _PARAMS, 1.0, polyline)
    b = BoundaryCurve(id=1, points=((-0.3, 2.0), (-0.3, 9.8)), left=3, right=1, closed=False)
    a_fit = FittedCurve(0, segs, None)
    assert len({h.a[2] for h in _crossings([_A, b], [a_fit])}) >= 2  # several crossing segments
    with _max_levels() as levels:
        out = guard_crossings([_A, b], [a_fit], {0: plan}, 1.0)
    assert not _crossings([_A, b], out[:1])
    assert levels == [LEVEL_CORNERS]  # a refit at full tolerance suffices: no level skipped


def test_primitive_crossing_drops_the_primitive_and_spares_the_neighbour():
    from engine.geometry.corners import dense_samples

    sq = BoundaryCurve(
        id=0,
        points=((0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0), (0.0, 0.0)),
        left=1, right=2, closed=True,
    )
    segs = tuple(Line(p, q) for p, q in zip(sq.points, sq.points[1:]))
    b = BoundaryCurve(id=1, points=((-3.0, 9.0), (1.0, 13.0)), left=3, right=2, closed=False)
    b_dense = dense_samples(b.points, False)
    b_plan, b_segs = plan_spans(b, b_dense, (0, len(b_dense) - 1), frozenset(), 0, _PARAMS, 1.0)
    b_fit = FittedCurve(1, b_segs, None)
    out = guard_crossings([sq, b], [FittedCurve(0, segs, Circle(5.0, 5.0, 9.0)), b_fit], {1: b_plan}, 1.0)
    assert out[0].primitive is None and out[0].segments == segs
    assert out[1] is b_fit


def test_corner_breaks_never_collapse_onto_one_vertex():
    import engine.geometry.crossing_guard as cg

    # corners at dense 10 (s=9.5) and 11 (s=10.5) straddle the one pixel vertex (0, 10)
    plan, _ = _plan(_dense_a(), (0, 10, 11, 21), {1, 2})
    for level in (LEVEL_CORNERS, LEVEL_RAW):
        rep = cg._Repair(plan)
        rep.levels = [level] * 3
        segments = rep.build().segments
        assert all(s.p0 != s.p1 for s in segments)
        assert all(n >= 1 for n in rep._sizes)
        assert segments[0].p0 == (0.0, 0.0) and segments[-1].p1 == (10.0, 10.0)


def test_interior_points_past_a_moved_corner_are_dropped():
    import engine.geometry.crossing_guard as cg

    # the corner sample at dense 12 (s=11.5) moves back to the vertex (0, 10) at s=10; dense 11
    # (s=10.5) now lies past the span's end and must not make it double back
    tight = FitParams(tolerance_px=0.3)  # (0.5, 10) is 0.5 px off the true edge
    plan, _ = plan_spans(_A, tuple(_dense_a()), (0, 12, 21), frozenset({1}), 0, tight, 1.0)
    rep = cg._Repair(plan)
    rep.levels = [LEVEL_CORNERS, LEVEL_FIT]
    assert rep.build().segments[0] == Line((0.0, 0.0), (0.0, 10.0))


def test_repair_loop_is_capped_and_warns(monkeypatch, caplog):
    import engine.geometry.crossing_guard as cg

    monkeypatch.setattr(cg, "MAX_GUARD_ROUNDS", 0)
    dense = _dense_a()
    dense[10] = (-1.5, 11.5)
    plan, segs = _plan(dense, (0, 10, len(dense) - 1), {1})
    b = BoundaryCurve(id=1, points=((-3.0, 9.0), (1.0, 13.0)), left=3, right=1, closed=False)
    fitted = FittedCurve(0, segs, None)
    with caplog.at_level("WARNING", logger="engine.geometry.crossing_guard"):
        out = guard_crossings([_A, b], [fitted], {0: plan}, 1.0)
    assert out[0] is fitted
    assert "crossing" in caplog.text


def test_repair_level_two_falls_back_to_the_crack_path():
    dense = _dense_a()
    for i in range(3, 9):  # a bulge on the vertical span that no refit can avoid
        dense[i] = (-1.6, dense[i][1])
    plan, segs = _plan(dense, (0, 10, len(dense) - 1), {1})
    b = BoundaryCurve(id=1, points=((-1.0, -5.0), (-1.0, 15.0)), left=3, right=1, closed=False)
    a_fit = FittedCurve(0, segs, None)
    assert _crossings([_A, b], [a_fit])
    out = guard_crossings([_A, b], [a_fit], {0: plan}, 1.0)
    assert not _crossings([_A, b], out[:1])
    ends = [s.p1 for s in out[0].segments]
    first_span = out[0].segments[: ends.index((0.0, 10.0)) + 1]  # up to the pixel corner
    assert first_span == (Line((0.0, 0.0), (0.0, 10.0)),)
    assert out[0].segments[0].p0 == (0.0, 0.0) and out[0].segments[-1].p1 == (10.0, 10.0)


def test_repair_keeps_closed_curves_closed():
    # closed square ring of A-side pixels 0..10; one corner overshoots past B outside it
    from engine.geometry.corners import dense_samples

    sq = BoundaryCurve(
        id=0,
        points=((0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0), (0.0, 0.0)),
        left=1, right=2, closed=True,
    )
    ring = list(dense_samples(sq.points, True))  # 40 samples, corners near 0, 10, 20, 30
    breaks = [0, 10, 20, 30, 40]
    seq = ring + [ring[0]]
    seq[10] = (-1.5, 11.5)
    plan, segs = plan_spans(sq, tuple(seq), tuple(breaks), frozenset(range(5)), 0, _PARAMS, 1.0)
    b = BoundaryCurve(id=1, points=((-3.0, 9.0), (1.0, 13.0)), left=3, right=2, closed=False)
    fitted = FittedCurve(0, segs, None)
    assert _crossings([sq, b], [fitted])
    out = guard_crossings([sq, b], [fitted], {0: plan}, 1.0)
    assert not _crossings([sq, b], out[:1])
    assert out[0].segments[0].p0 == out[0].segments[-1].p1


def test_closed_curve_going_raw_follows_its_crack_path_and_stays_closed(monkeypatch):
    import engine.geometry.crossing_guard as cg
    from engine.geometry.corners import dense_samples

    monkeypatch.setattr(cg, "MAX_REPAIR_ROUNDS", 0)
    sq = BoundaryCurve(
        id=0,
        points=((0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0), (0.0, 0.0)),
        left=1, right=2, closed=True,
    )
    ring = list(dense_samples(sq.points, True))
    seq = ring[5:] + ring[:5] + [ring[5]]  # rotated to start mid-side (offset 5)
    seq[5] = (-1.5, 11.5)  # the corner at (0, 10), now at sequence position 5
    breaks = (0, 5, 15, 25, 35, 40)
    plan, segs = plan_spans(sq, tuple(seq), breaks, frozenset({1, 2, 3, 4}), 5, _PARAMS, 1.0)
    b = BoundaryCurve(id=1, points=((-3.0, 9.0), (1.0, 13.0)), left=3, right=2, closed=False)
    out = guard_crossings([sq, b], [FittedCurve(0, segs, None)], {0: plan}, 1.0)
    pts = [s.p0 for s in out[0].segments]
    assert out[0].segments[-1].p1 == pts[0] == (0.0, 5.5)  # the crack-edge midpoint start
    assert set(pts) == {(0.0, 5.5), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0), (0.0, 0.0)}


def test_sample_segments_of_nothing_and_of_a_tiny_circle():
    pts, tags = sample_segments((), max_error=0.25)
    assert pts.shape == (0, 2) and tags.shape == (0,)
    tiny = sample_shape(Circle(0.0, 0.0, 0.1), max_error=0.25)
    assert len(tiny) >= 5 and np.allclose(np.hypot(*tiny.T), 0.1)


def test_crossing_primitive_is_dropped():
    sq = BoundaryCurve(
        id=0,
        points=((0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0), (0.0, 0.0)),
        left=1, right=2, closed=True,
    )
    b = BoundaryCurve(id=1, points=((-3.0, 9.0), (1.0, 13.0)), left=3, right=2, closed=False)
    segs = tuple(Line(p, q) for p, q in zip(sq.points, sq.points[1:]))
    big = FittedCurve(0, segs, Circle(5.0, 5.0, 9.0))
    out = guard_crossings([sq, b], [big], {}, 1.0)
    assert out[0].primitive is None and out[0].segments == segs
    small = FittedCurve(0, segs, Circle(5.0, 5.0, 4.0))
    assert guard_crossings([sq, b], [small], {}, 1.0)[0] is small


def test_no_crossings_returns_the_input_unchanged():
    plan, segs = _plan(_dense_a(), (0, 10, 21), {1})
    fitted = (FittedCurve(0, segs, None),)
    out = guard_crossings([_A], fitted, {0: plan}, 1.0)
    assert out[0] is fitted[0]


def test_plan_spans_matches_fit_spans_layout():
    dense = _dense_a()
    plan, segs = _plan(dense, (0, 10, len(dense) - 1), {1})
    assert sum(plan.span_sizes) == len(segs)
    assert segs[0].p0 == (0.0, 0.0) and segs[-1].p1 == (10.0, 10.0)


def test_whole_curves_go_raw_after_the_round_cap(monkeypatch):
    import engine.geometry.crossing_guard as cg

    monkeypatch.setattr(cg, "MAX_REPAIR_ROUNDS", 0)  # skip span escalation entirely
    dense = _dense_a()
    dense[10] = (-1.5, 11.5)
    plan, segs = _plan(dense, (0, 10, len(dense) - 1), {1})
    b = BoundaryCurve(id=1, points=((-3.0, 9.0), (1.0, 13.0)), left=3, right=1, closed=False)
    out = guard_crossings([_A, b], [FittedCurve(0, segs, None)], {0: plan}, 1.0)
    assert out[0].segments == (Line((0.0, 0.0), (0.0, 10.0)), Line((0.0, 10.0), (10.0, 10.0)))


def test_snap_axis_leaves_locked_segments_alone():
    from engine.geometry.segment_fit import snap_axis

    chain = [Line((0, 0), (5, 5)), Line((5, 5), (15, 5.2)), Line((15, 5.2), (20, 0))]
    params = FitParams(tolerance_px=0.75)
    assert snap_axis(chain, params)[1].p0[1] == snap_axis(chain, params)[1].p1[1]
    assert snap_axis(chain, params, locked=frozenset({0})) == tuple(chain)


# ---------------------------------------------------------------- pipeline regression

# A fill wedge whose apex tapers toward a thin dark outline over a blue ground, drawn
# anti-aliased at processing resolution (as enhance hands it to segment). Near the apex the band
# is under one original pixel: band pixels blend fill and ground, the sub-pixel edge reads pushed
# outward, and the extrapolated corner (like the L in happiness.jpg) lands across the outline.
_SIZE = 48
_BG, _BLACK = (110, 150, 210, 255), (41, 40, 40, 255)
_PINK, _WHITE = (230, 190, 200, 255), (250, 250, 250, 255)


def _wedge_image(band, edge_deg, side1_deg, side2_deg, scale):
    from eval.synthetic import solid, supersample
    from engine.types import RasterImage

    p = np.array([24.0, 30.0])
    phi = math.radians(edge_deg)
    normal = np.array([math.sin(phi), -math.cos(phi)])  # toward the outline's dark side
    apex = p + band * normal
    a1, a2 = phi - math.radians(side1_deg), phi + math.pi + math.radians(side2_deg)

    def draw(n):
        k = n / _SIZE
        ys, xs = np.mgrid[0:n, 0:n]
        x, y = (xs + 0.5) / k, (ys + 0.5) / k
        px = solid(n, _BG)
        dark = ((x - p[0]) * normal[0] + (y - p[1]) * normal[1] >= 0) & (x > 3) & (x < 45) & (y > 3)
        px[dark] = _BLACK
        qx, qy = x - apex[0], y - apex[1]
        fill = (qx * -math.sin(a1) + qy * math.cos(a1) <= 0) & (
            qx * -math.sin(a2) + qy * math.cos(a2) >= 0
        )
        fill &= (np.hypot(qx, qy) <= 16) & dark & (x > 5.5) & (x < 42.5) & (y > 5.5)
        px[fill] = _PINK
        px[fill & (np.hypot(qx, qy) > 10)] = _WHITE  # pink meets white: its outline curve is open
        return px

    pixels = supersample(draw, _SIZE * scale, 8)
    return RasterImage(pixels=pixels, original_size=(_SIZE, _SIZE), scale=float(scale))


def _all_crossings(graph, fitted, scale):
    from engine.geometry.crossing_guard import JUNCTION_CLEARANCE_PX, _tracks

    tracks = _tracks(graph.curves, {f.curve_id: f for f in fitted}, {c.id for c in graph.curves})
    ends = {c.id: (c.points[0], c.points[-1]) for c in graph.curves if not c.closed}
    return find_crossings(tracks, ends, JUNCTION_CLEARANCE_PX * scale)


def _vectorize_stages(image, monkeypatch, guarded):
    from engine.geometry.subpixel import make_edge_context
    from engine.presets import resolve_preset
    from engine.stages import fit_geometry as fg
    from engine.stages.export import build_doc, to_svg
    from engine.stages.segment import segment
    from engine.stages.trace import trace

    preset = resolve_preset("auto")
    regions = segment(image, preset.segment)
    graph = trace(regions)
    with monkeypatch.context() as m:
        if not guarded:  # raising=False: also runs on the code before the guard existed
            m.setattr(fg, "guard_crossings", lambda c, f, p, s: tuple(f), raising=False)
        fitted = fg.fit_geometry(graph, preset.fit, image.scale, make_edge_context(image, regions))
    doc = build_doc(regions, graph, fitted, preset.export, image.scale, image.original_size)
    return regions, graph, fitted, to_svg(doc)


@pytest.mark.parametrize(
    "wedge, scale",
    [((0.62, 48.7, 5.66, 46.6), 2), ((0.367, 44.43, 23.56, 81.03), 4)],
    ids=["scale2", "scale4"],
)
def test_fill_corner_never_crosses_a_thin_outline(wedge, scale, monkeypatch):
    from eval.metrics import render_svg, ssim

    image = _wedge_image(*wedge, scale)
    _, graph, unguarded, svg_before = _vectorize_stages(image, monkeypatch, guarded=False)
    assert _all_crossings(graph, unguarded, scale), "scene no longer reproduces a crossing"
    regions, graph, fitted, svg_after = _vectorize_stages(image, monkeypatch, guarded=True)
    colours = [tuple(int(v) for v in c) for c in regions.palette_rgb]
    pink, ground = colours.index(_PINK[:3]), colours.index(_BG[:3])
    assert (min(pink, ground), max(pink, ground)) not in regions.adjacency  # the band is intact
    assert _all_crossings(graph, fitted, scale) == []
    for curve, fit in zip(graph.curves, fitted):
        if curve.closed:
            assert fit.segments[0].p0 == fit.segments[-1].p1
        else:
            assert fit.segments[0].p0 == curve.points[0] and fit.segments[-1].p1 == curve.points[-1]
    size = (_SIZE * scale, _SIZE * scale)
    before = ssim(image.pixels, render_svg(svg_before, *size))
    after = ssim(image.pixels, render_svg(svg_after, *size))
    assert after >= before - 0.002


# sha256(to_svg(...))[:16] with enhance off, recorded on master dd1c9dd (before the guard). These
# scenes have no crossings, so the guard must leave their output byte-identical.
NO_CROSSING_DIGESTS = [
    ("aa_shapes(0)", 2, "c87c4281c46b4c14"),
    # Re-recorded for Q2 (segment_regions): the anti-aliasing ribbons along the thin orange and
    # teal shapes are now split between their two sides instead of staying as 1-px regions;
    # checked by eye: same shapes, nodes 513 -> 78, the sub-pixel orange tail tip is ~2 px shorter
    # (SSIM 0.9664 -> 0.9649, edge F1 0.937 -> 0.916). Still a scene without crossings.
    ("aa_shapes(3)", 4, "d4fd84507df7ef46"),
    ("aa_disc()", 2, "5afe4b735dfc8a81"),
    ("ring_in_ring()", 4, "506a12e1486e5799"),
]


@pytest.mark.parametrize("make, scale, digest", NO_CROSSING_DIGESTS)
def test_scenes_without_crossings_are_unchanged(make, scale, digest):
    import hashlib

    from engine import to_svg
    from engine.pipeline import run_pipeline
    from engine.presets import apply_overrides, resolve_preset
    from engine.types import RasterImage
    from eval import synthetic

    px = eval(f"synthetic.{make}")  # noqa: S307 - fixed test table
    big = np.repeat(np.repeat(px, scale, 0), scale, 1)
    image = RasterImage(pixels=big, original_size=(px.shape[1], px.shape[0]), scale=float(scale))
    preset = apply_overrides(resolve_preset("auto"), {"enhance.enabled": False})
    svg = to_svg(run_pipeline(image, preset).doc)
    assert hashlib.sha256(svg.encode()).hexdigest()[:16] == digest
