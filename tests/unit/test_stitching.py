import io
import time

import numpy as np
import pytest
from PIL import Image
from scipy import ndimage

from engine.errors import StageError
from engine.geometry.stitching import build_boundary_graph
from engine.presets import SegmentParams
from engine.stages.segment import segment
from engine.stages.trace import trace
from engine.types import OUTSIDE, TRANSPARENT, BoundaryGraph, RasterImage, RegionMap
from eval.synthetic import flat_shapes, ring_in_ring, solid, transparent_logo


def _img(pixels: np.ndarray) -> RasterImage:
    h, w = pixels.shape[:2]
    return RasterImage(pixels=pixels, original_size=(w, h))


def _regions(pixels: np.ndarray) -> RegionMap:
    return segment(_img(pixels), SegmentParams())


def _label_map(labels: list[list[int]]) -> RegionMap:
    arr = np.array(labels, np.int32)
    k = int(arr.max()) + 1
    return RegionMap(
        labels=arr,
        palette_rgb=np.zeros((k, 3), np.uint8),
        palette_lab=np.zeros((k, 3)),
        adjacency=frozenset(),
    )


def _loop_points(graph: BoundaryGraph, loop) -> list[list[tuple[float, float]]]:
    out = []
    for cid, rev in loop:
        pts = list(graph.curves[cid].points)
        out.append(pts[::-1] if rev else pts)
    return out


def _signed_area(graph: BoundaryGraph, loop) -> float:
    total = 0.0
    for pts in _loop_points(graph, loop):
        p = np.asarray(pts, np.float64)
        total += float(np.sum(p[:-1, 0] * p[1:, 1] - p[1:, 0] * p[:-1, 1]))
    return total / 2.0


def _assert_loops_close(graph: BoundaryGraph) -> None:
    for loops in graph.region_loops.values():
        for loop in loops:
            chains = _loop_points(graph, loop)
            for cur, nxt in zip(chains, chains[1:] + chains[:1]):
                assert cur[-1] == nxt[0]


def _assert_area_matches(graph: BoundaryGraph, labels: np.ndarray) -> None:
    counts = np.bincount(labels[labels >= 0].ravel())
    assert set(graph.region_loops) == {int(r) for r in np.flatnonzero(counts)}
    for r, loops in graph.region_loops.items():
        areas = [_signed_area(graph, loop) for loop in loops]
        assert areas[0] > 0, "outer loop must have positive signed area"
        assert all(a < 0 for a in areas[1:]), "holes must have negative signed area"
        assert abs(areas[0]) - sum(abs(a) for a in areas[1:]) == counts[r]


def test_every_curve_shared():
    rm = _regions(flat_shapes(7))
    graph = build_boundary_graph(rm)
    users: dict[int, list[int]] = {c.id: [] for c in graph.curves}
    for r, loops in graph.region_loops.items():
        for loop in loops:
            for cid, _ in loop:
                users[cid].append(r)
    for curve in graph.curves:
        expected = sorted(s for s in (curve.left, curve.right) if s >= 0)
        assert sorted(users[curve.id]) == expected
        assert curve.left >= 0 or curve.right >= 0


def test_curve_ids_match_positions():
    graph = build_boundary_graph(_regions(flat_shapes(7)))
    assert [c.id for c in graph.curves] == list(range(len(graph.curves)))


def test_loops_close():
    graph = build_boundary_graph(_regions(flat_shapes(7)))
    _assert_loops_close(graph)
    for curve in graph.curves:
        if curve.closed:
            assert curve.points[0] == curve.points[-1]


def test_ring_in_ring_holes():
    rm = _regions(ring_in_ring())
    graph = build_boundary_graph(rm)
    c = rm.labels.shape[0] // 2
    middle = int(rm.labels[c, c - int(rm.labels.shape[0] * 0.25)])  # inside the 0.32 ring only
    assert len(graph.region_loops[middle]) == 2


def test_border_region_uses_outside():
    rm = _regions(solid(8, (10, 20, 30, 255)))
    graph = build_boundary_graph(rm)
    assert len(graph.curves) == 1
    curve = graph.curves[0]
    assert curve.closed
    assert OUTSIDE in (curve.left, curve.right)
    assert set(curve.points) == {(0.0, 0.0), (8.0, 0.0), (8.0, 8.0), (0.0, 8.0)}
    assert len(curve.points) == 5  # four corners, first repeated last
    assert graph.region_loops[0] == (((0, False),) if curve.left == 0 else ((0, True),),)


def test_checkerboard_junction():
    rm = _label_map([[0, 1], [2, 3]])  # two colors in a 2x2 checkerboard -> 4 regions
    graph = build_boundary_graph(rm)
    _assert_loops_close(graph)
    _assert_area_matches(graph, rm.labels)


def test_region_touching_itself_diagonally():
    # Region 1 is 4-connected and wraps around pixel (2,1) (region 2), touching itself
    # diagonally at corner (2,2) where region 0 also meets.
    rm = _label_map(
        [
            [0, 1, 1, 1, 0],
            [0, 1, 2, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    graph = build_boundary_graph(rm)
    _assert_loops_close(graph)
    _assert_area_matches(graph, rm.labels)


def test_area_matches_pixels():
    for px in (flat_shapes(7), flat_shapes(11, n_colors=6), ring_in_ring()):
        rm = _regions(px)
        _assert_area_matches(build_boundary_graph(rm), rm.labels)


def test_transparent_neighbors_kept_and_non_region_curves_dropped():
    rm = _regions(transparent_logo())
    graph = build_boundary_graph(rm)
    assert all(c.left >= 0 or c.right >= 0 for c in graph.curves)
    assert any(TRANSPARENT in (c.left, c.right) for c in graph.curves)
    _assert_loops_close(graph)
    _assert_area_matches(graph, rm.labels)


def test_fully_transparent_has_no_curves():
    rm = _regions(solid(8, (0, 0, 0, 0)))
    graph = build_boundary_graph(rm)
    assert graph.curves == () and dict(graph.region_loops) == {}


def test_trace_wraps_errors_in_stage_error():
    bad = RegionMap(
        labels=np.zeros((4, 4, 2), np.int32),
        palette_rgb=np.zeros((1, 3), np.uint8),
        palette_lab=np.zeros((1, 3)),
        adjacency=frozenset(),
    )
    with pytest.raises(StageError) as info:
        trace(bad)
    assert info.value.stage == "trace"


def test_trace_matches_build():
    rm = _regions(flat_shapes(7))
    assert trace(rm) == build_boundary_graph(rm)


@pytest.mark.slow
def test_large_noisy_image_traces_fast():
    base = np.kron(flat_shapes(4, n_colors=5), np.ones((16, 16, 1), np.uint8))
    buf = io.BytesIO()
    Image.fromarray(base[..., :3], "RGB").save(buf, format="JPEG", quality=85)
    rgb = np.array(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
    px = np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])
    rm = _regions(px)
    t0 = time.perf_counter()
    graph = trace(rm)
    elapsed = time.perf_counter() - t0
    print(f"trace 2048x2048 noisy: {elapsed:.2f}s, {len(graph.curves)} curves")
    _assert_area_matches(graph, rm.labels)
    assert elapsed < 15


@pytest.mark.slow
def test_many_small_regions_trace_fast():
    # Worst-case-ish topology: 4x4-pixel random blocks in 6 colors -> ~170k regions at 2048x2048.
    rng = np.random.default_rng(0)
    colors = np.kron(rng.integers(0, 6, (512, 512)), np.ones((4, 4), np.int64))
    labels = np.empty(colors.shape, np.int32)
    k = 0
    for c in range(6):
        comp, n = ndimage.label(colors == c)
        labels[comp > 0] = comp[comp > 0] - 1 + k
        k += n
    labels[:100, :100] = TRANSPARENT
    rm = RegionMap(labels, np.zeros((k, 3), np.uint8), np.zeros((k, 3)), frozenset())
    t0 = time.perf_counter()
    graph = trace(rm)
    elapsed = time.perf_counter() - t0
    print(f"trace 2048x2048 {k} regions: {elapsed:.2f}s, {len(graph.curves)} curves")
    assert elapsed < 15
    counts = np.bincount(labels[labels >= 0], minlength=k)
    for r in rng.choice(np.flatnonzero(counts), 500, replace=False).tolist():
        loops = graph.region_loops[r]
        areas = [_signed_area(graph, loop) for loop in loops]
        assert areas[0] - sum(abs(a) for a in areas[1:]) == counts[r]


def test_trace_error_reason_is_user_safe(monkeypatch):
    def boom(regions):
        raise ValueError("secret detail")

    monkeypatch.setattr("engine.stages.trace.build_boundary_graph", boom)
    with pytest.raises(StageError) as info:
        trace(None)
    assert info.value.reason == "internal error (ValueError)"
    assert isinstance(info.value.__cause__, ValueError)
