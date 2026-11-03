"""Defect detectors on small synthetic cases with known answers, plus a CLI smoke test."""
import csv
import math
import re
from pathlib import Path

import numpy as np
import pytest

from eval.defects import (
    CurveGeom,
    ShapeGeom,
    detect_crossings,
    detect_junction_spikes,
    detect_specks,
    detect_straightness,
    detect_thin_bands,
    detect_wobble,
    detect_worst_patches,
    geometry_from_state,
    shapes_from_doc,
)
from eval.synthetic import circle, flat_shapes, to_png_bytes
from engine.types import CubicBezier, Line, Path as VPath, SolidFill, VectorDoc

# ---------------------------------------------------------------- helpers


def _ring(radius_fn, n=720, cx=100.0, cy=100.0) -> np.ndarray:
    t = np.linspace(0.0, 2 * math.pi, n + 1)
    r = radius_fn(t)
    pts = np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1)
    pts[-1] = pts[0]
    return pts


def _closed(cid: int, pts: np.ndarray) -> CurveGeom:
    return CurveGeom(id=cid, points=np.asarray(pts, float), left=1, right=0, closed=True)


def _open(cid: int, pts, left=1, right=0) -> CurveGeom:
    return CurveGeom(id=cid, points=np.asarray(pts, float), left=left, right=right, closed=False)


def _inside(box, x, y, pad=0.0) -> bool:
    x0, y0, x1, y1 = box
    return x0 - pad <= x <= x1 + pad and y0 - pad <= y <= y1 + pad


# ---------------------------------------------------------------- wobble


def _flat_line(lead: float, wave_from: float, wave_to: float, amp: float, k: float) -> CurveGeom:
    x = np.linspace(-lead, 200, int((200 + lead) * 4) + 1)
    y = np.where((x > wave_from) & (x < wave_to), 50 + amp * np.sin((x - wave_from) * k), 50.0)
    return _open(0, np.stack([x, y], axis=1))


def test_clean_circle_has_low_wobble():
    res = detect_wobble([_closed(0, _ring(lambda t: np.full_like(t, 60.0)))])
    assert res.score < 0.05
    assert res.spots == ()


def test_jittered_circle_has_high_wobble():
    jitter = _ring(lambda t: 60.0 + 0.8 * np.sin(40 * t))  # ~9 px wavelength, 0.8 px amplitude
    res = detect_wobble([_closed(0, jitter)])
    assert res.score > 5
    assert len(res.spots) >= 1


def test_jitter_scores_far_above_clean():
    clean = detect_wobble([_closed(0, _ring(lambda t: np.full_like(t, 60.0)))])
    rng = np.random.default_rng(0)
    noise = np.convolve(rng.normal(0, 0.6, 721), np.ones(5) / 5, mode="same")
    noisy = detect_wobble([_closed(0, _ring(lambda t: 60.0 + noise))])
    assert noisy.score > clean.score + 2


def test_true_corners_of_both_signs_are_not_wobble():
    # an L-shaped outline: five convex corners and one reflex corner
    shape = np.array([(20, 20), (120, 20), (120, 60), (60, 60), (60, 120), (20, 120), (20, 20)],
                     float)
    res = detect_wobble([_closed(0, shape)])
    assert res.score == 0.0
    assert res.spots == ()
    res = detect_wobble([_closed(0, shape[::-1])])  # walked the other way: signs swap
    assert res.score == 0.0


_X = np.linspace(0, 200, 800)
_T = np.linspace(0, 2 * math.pi, 2001)


@pytest.mark.parametrize("pts", [
    np.stack([_X, 100 + 30 * np.sin(_X / 200 * 2 * math.pi)], 1),
    np.stack([_X, 100 + 15 * np.sin(_X / 100 * 2 * math.pi)], 1),
    np.stack([_X, 100 + 5 * np.sin(_X / 60 * 2 * math.pi)], 1),
    _ring(lambda t: np.full_like(t, 15.0)),
    _ring(lambda t: np.full_like(t, 200.0), n=4000),
    np.stack([100 + 80 * np.cos(_T), 100 + 20 * np.sin(_T)], 1),
], ids=["s_curve", "s_amp15_wl100", "s_amp5_wl60", "circle_r15", "circle_r200", "ellipse_80x20"])
def test_clean_curves_have_no_wobble(pts):
    closed = bool(np.allclose(pts[0], pts[-1]))
    curve = _closed(0, pts) if closed else _open(0, pts)
    res = detect_wobble([curve])
    assert res.score == 0.0
    assert res.spots == ()


def test_wobble_spot_is_on_the_jittery_part():
    x = np.linspace(0, 200, 801)
    y = np.where(x > 120, 50 + 0.7 * np.sin(x * 0.7), 50.0)
    res = detect_wobble([_open(0, np.stack([x, y], axis=1))])
    assert res.spots
    assert all(s.box[2] > 115 for s in res.spots)


def test_wobble_score_does_not_jump_with_lead_in():
    end = 80 + 9 * math.pi / 0.5  # nine half waves: the wave ends at zero
    scores = [detect_wobble([_flat_line(lead, 80, end, 0.5, 0.5)]).score for lead in (0, 6, 21)]
    assert min(scores) > 0
    assert max(scores) < 1.1 * min(scores)


def test_invisible_wiggle_is_gated_by_amplitude():
    end = 80 + 9 * math.pi / 0.5
    assert detect_wobble([_flat_line(0, 80, end, 0.06, 0.5)]).score == 0.0


def test_long_low_amplitude_bend_is_wobble():
    # a near-straight edge with a 0.5 px, 40 px-wavelength wave: too long for turning excess at
    # the fairing scale, still a visibly wavy edge
    res = detect_wobble([_open(0, np.stack([_X, 50 + 0.5 * np.sin(_X / 40 * 2 * math.pi)], 1))])
    assert res.score > 0
    assert any("bend" in s.note for s in res.spots)


# ---------------------------------------------------------------- straightness


def _half_plane(size=100, edge_x=50):
    px = np.full((size, size, 4), 255, np.uint8)
    px[:, :edge_x, :3] = 20
    return px


def test_straight_source_edge_followed_straight_is_clean():
    y = np.linspace(5, 95, 361)
    curve = _open(0, np.stack([np.full_like(y, 50.0), y], 1))
    assert detect_straightness([curve], _half_plane()).score == 0.0


def test_bowed_fit_of_straight_source_edge_is_flagged():
    y = np.linspace(5, 95, 361)
    x = 50.0 + 0.9 * np.sin((y - 5) / 90 * 2 * math.pi)
    res = detect_straightness([_open(0, np.stack([x, y], 1))], _half_plane())
    assert res.score > 0
    assert res.spots


def test_curved_source_edge_is_not_a_straightness_defect():
    # the fit follows a real circle: its windows turn too much to count as straight edges
    px = np.full((200, 200, 4), 255, np.uint8)
    yy, xx = np.mgrid[0:200, 0:200] + 0.5
    px[(xx - 100) ** 2 + (yy - 100) ** 2 <= 60**2, :3] = 20
    res = detect_straightness([_closed(0, _ring(lambda t: np.full_like(t, 60.0)))], px)
    assert res.score == 0.0


# ---------------------------------------------------------------- specks


def test_tiny_speck_is_found():
    shapes = [ShapeGeom(area=2.5, box=(10, 10, 12, 12)), ShapeGeom(area=900, box=(0, 0, 30, 30))]
    res = detect_specks(shapes)
    assert res.score > 0
    assert len(res.spots) == 1
    assert res.spots[0].box == (10, 10, 12, 12)


def test_shapes_from_doc_measures_path_area():
    square = (Line((0, 0), (10, 0)), Line((10, 0), (10, 10)), Line((10, 10), (0, 10)),
              Line((0, 10), (0, 0)))
    dot = (Line((20, 20), (21, 20)), Line((21, 20), (21, 21)), Line((21, 21), (20, 20)))
    doc = VectorDoc(width=30, height=30, layer_mode="stacked", primitives=(),
                    paths=(VPath((square,), SolidFill((0, 0, 0)), 0),
                           VPath((dot,), SolidFill((9, 9, 9)), 1)))
    areas = sorted(s.area for s in shapes_from_doc(doc))
    assert areas == pytest.approx([0.5, 100.0])
    res = detect_specks(shapes_from_doc(doc))
    assert len(res.spots) == 1 and _inside(res.spots[0].box, 20.5, 20.5)


# ---------------------------------------------------------------- thin bands


def _band(top_y, bottom_y_fn):
    x = np.linspace(0, 120, 241)
    top = _open(0, np.stack([x, np.full_like(x, top_y)], axis=1), left=1, right=0)
    bottom = _open(1, np.stack([x, bottom_y_fn(x)], axis=1), left=2, right=1)
    return [top, bottom]


def test_even_band_scores_lower_than_uneven_band():
    even = detect_thin_bands(_band(50.0, lambda x: np.full_like(x, 53.0)))
    uneven = detect_thin_bands(_band(50.0, lambda x: 53.0 + 0.9 * np.sin(x / 3.0)))
    assert even.score < 0.05
    assert even.spots == ()
    assert uneven.score > even.score + 0.5
    assert uneven.spots


def test_band_score_does_not_jump_with_lead_in():
    def band(lead):
        x = np.linspace(-lead, 120, int((120 + lead) * 2) + 1)
        bottom = np.where(x > 30, 53.0 + 0.9 * np.sin((x - 30) / 3.0), 53.0)
        top = _open(0, np.stack([x, np.full_like(x, 50.0)], 1), left=1, right=0)
        return detect_thin_bands([top, _open(1, np.stack([x, bottom], 1), left=2, right=1)]).score

    scores = [band(lead) for lead in (0, 6, 21)]
    assert min(scores) > 0
    assert max(scores) < 1.1 * min(scores)


def test_wide_region_is_not_a_band():
    wide = detect_thin_bands(_band(50.0, lambda x: 70.0 + 3 * np.sin(x / 3.0)))
    assert wide.score == 0.0


# ---------------------------------------------------------------- crossings


def test_two_crossing_curves_are_found():
    a = _open(0, [(0, 0), (20, 20)])
    b = _open(1, [(0, 20), (20, 0)])
    res = detect_crossings([a, b])
    assert res.score == 1
    assert _inside(res.spots[0].box, 10, 10)


def test_exact_touch_is_not_a_crossing_at_scale_3():
    """e (6).jpeg at scale 3: b's vertex (1177, 1396) lies exactly on a's segment and b stays on
    one side (a touch). Divided by 3 the rounding made it two proper crossings; tested in
    processing px, as the engine's guard does, it is none."""
    a_px = np.array([(1182, 1398), (1181, 1404), (1175, 1392), (1178, 1391)], float)
    b_px = np.array([(1178, 1391), (1177, 1396), (1182, 1398)], float)
    a = CurveGeom(0, a_px / 3, 1, 2, False, points_px=a_px)
    b = CurveGeom(1, b_px / 3, 1, 3, False, points_px=b_px)
    assert detect_crossings([a, b], scale=3.0).score == 0
    unscaled = [CurveGeom(0, a_px / 3, 1, 2, False), CurveGeom(1, b_px / 3, 1, 3, False)]
    assert detect_crossings(unscaled).score == 2  # the old behaviour this guards against


def test_crossing_spots_are_in_original_px_when_tested_in_processing_px():
    a_px = np.array([(0, 0), (60, 60)], float)
    b_px = np.array([(0, 60), (60, 0)], float)
    curves = [CurveGeom(0, a_px / 3, 1, 2, False, points_px=a_px),
              CurveGeom(1, b_px / 3, 1, 3, False, points_px=b_px)]
    res = detect_crossings(curves, scale=3.0)
    assert res.score == 1 and _inside(res.spots[0].box, 10, 10)


def test_crossing_within_junction_clearance_is_ignored():
    # b leaves the shared junction (10, 10) and overshoots a by 0.5 px before turning away
    a = _open(0, [(0, 10), (10, 10)])
    b = _open(1, [(10, 10), (9.5, 10.5), (9.5, 0)])
    c = _open(2, [(10, 10), (20, 10)])
    assert detect_crossings([a, b, c]).score == 0
    assert detect_crossings([a, b, c], clearance=0.0).score == 1


# ---------------------------------------------------------------- junction spikes


def _hook_pts(j, angle, hook: bool):
    d = np.array([math.cos(angle), math.sin(angle)])
    n = np.array([-d[1], d[0]])
    local = [(0, 0), (1, -1.5), (2, 0), (10, 0), (40, 0)] if hook else [(0, 0), (10, 0), (40, 0)]
    return np.array([np.asarray(j) + u * d + v * n for u, v in local])


def _stair(j, angle, length, step):
    """Crack-grid path (grid `step`) from `j` along `angle`: what the source pixels give."""
    t = np.linspace(0, length, int(length / step * 8) + 1)
    q = np.round((np.asarray(j) + np.outer(t, [math.cos(angle), math.sin(angle)])) / step) * step
    q = q[np.r_[True, np.any(np.diff(q, axis=0) != 0, axis=1)]]
    out = [q[0]]
    for p in q[1:]:
        prev = out[-1]
        if prev[0] != p[0] and prev[1] != p[1]:
            out.append(np.array([p[0], prev[1]]))
        out.append(p)
    return np.array(out)


def _star(hook: bool, step: float | None = None, angle: float = math.radians(20)):
    j = (50.0, 50.0)
    raw = _stair(j, angle, 40, step) if step else None
    east = CurveGeom(0, _hook_pts(j, angle, hook), 1, 2, False, raw=raw)
    # the other two leave the junction along gentle arcs (radius 40), not straight lines
    t = np.linspace(0, 0.8, 60)
    arc1 = np.stack([-40 * np.sin(t), -40 * (1 - np.cos(t))], 1)
    arc2 = np.stack([-40 * np.sin(t), 40 * (1 - np.cos(t))], 1)
    rot = np.array([[math.cos(0.6), -math.sin(0.6)], [math.sin(0.6), math.cos(0.6)]])
    arc1 = arc1 @ rot.T + np.array(j)
    arc2 = arc2 @ rot + np.array(j)
    return [east, _open(1, arc1, 2, 3), _open(2, arc2, 3, 1)]


def test_spike_at_a_junction_is_found():
    res = detect_junction_spikes(_star(hook=True))
    assert res.score > 0
    assert any(_inside(s.box, 50, 50, pad=1) for s in res.spots)


def test_clean_junction_has_no_spike():
    assert detect_junction_spikes(_star(hook=False)).score == 0


@pytest.mark.parametrize("scale", [1, 2, 4])
def test_hook_is_found_at_every_processing_scale(scale):
    # the crack path is a staircase with steps of 1/scale original px
    assert detect_junction_spikes(_star(hook=True, step=1.0 / scale)).score > 0
    assert detect_junction_spikes(_star(hook=False, step=1.0 / scale)).score == 0


# ---------------------------------------------------------------- worst patches


def test_worst_patch_is_where_the_render_differs():
    src = np.full((96, 96, 4), 255, np.uint8)
    out = src.copy()
    out[60:70, 20:30, :3] = (200, 30, 30)
    res = detect_worst_patches(src, out)
    assert res.score > 0
    assert _inside(res.spots[0].box, 25, 65)


# ---------------------------------------------------------------- CLI smoke


def test_geometry_from_state_at_scale_2_matches_export():
    from engine.pipeline import run_stages
    from engine.presets import apply_overrides, resolve_preset
    from engine.types import Circle, RasterImage

    size = 64
    yy, xx = np.mgrid[0 : 2 * size, 0 : 2 * size] + 0.5
    px = np.full((2 * size, 2 * size, 4), 255, np.uint8)
    px[(xx - 64) ** 2 + (yy - 64) ** 2 <= 40**2] = (220, 40, 40, 255)  # red disc ...
    px[(xx - 64) ** 2 + (yy - 64) ** 2 <= 16**2] = (30, 60, 200, 255)  # ... with a blue hole
    image = RasterImage(pixels=px, original_size=(size, size), scale=2.0)
    preset = apply_overrides(resolve_preset("auto"), {"enhance.enabled": False})
    state, _ = run_stages(image, preset)
    geom = geometry_from_state(state)

    assert geom.scale == 2.0
    pts = np.vstack([c.points for c in geom.curves])
    assert pts.min() >= -1e-6 and pts.max() <= size + 1e-6  # original px
    prims = [c for c in geom.curves if c.primitive is not None]
    assert len(prims) == len(state.doc.primitives)
    # the red region has a hole bordering the blue disc, yet stacked export draws it as a circle
    radii = sorted(round(p.shape.r) for p in state.doc.primitives if isinstance(p.shape, Circle))
    assert 20 in radii
    ring = [c for c in prims if np.ptp(c.primitive[:, 0]) > 30]
    assert ring and np.allclose(ring[0].points, ring[0].primitive)  # drawn outline is the circle


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_cli_smoke(tmp_path):
    from eval.run_defects import main

    images = tmp_path / "imgs"
    images.mkdir()
    (images / "a_circle.png").write_bytes(to_png_bytes(circle(96, 48, 48, 30)))
    (images / "b_flat.png").write_bytes(to_png_bytes(flat_shapes(3, size=96)))
    (images / "c_broken.png").write_bytes(b"not an image")
    out = tmp_path / "out"
    main(["--images", str(images), "--out", str(out), "--enhance", "off"])
    rows = {r["image"]: r for r in _rows(out / "defects.csv")}
    assert set(rows) == {"a_circle.png", "b_flat.png", "c_broken.png"}
    assert rows["c_broken.png"]["status"] == "error"
    assert rows["c_broken.png"]["error"]
    for name in ("a_circle.png", "b_flat.png"):
        assert rows[name]["status"] == "ok"
        assert float(rows[name]["ssim"]) > 0.9
        assert rows[name]["enhance"] == "off"
        assert float(rows[name]["scale"]) == 1.0
        assert "git_sha" in rows[name]
        assert (out / "svg" / f"{name}.svg").is_file()  # the extension stays in output names
    report = (out / "report.html").read_text(encoding="utf-8")
    assert "a_circle.png" in report and "c_broken.png" in report

    out2 = tmp_path / "out2"
    main(["--images", str(images), "--out", str(out2), "--enhance", "off",
          "--compare", str(out / "defects.csv")])
    report2 = (out2 / "report.html").read_text(encoding="utf-8")
    assert re.search(r'<span class="d[0pn]">\(Δ [^)]+\)</span>', report2)
    assert "mean (compare, 2 shared)" in report2
    assert "warning" not in report2.lower()


def test_cli_compare_warns_on_no_shared_rows_and_different_settings(tmp_path):
    from eval.run_defects import main

    images = tmp_path / "imgs"
    images.mkdir()
    (images / "a_circle.png").write_bytes(to_png_bytes(circle(64, 32, 32, 20)))
    other = tmp_path / "other.csv"
    other.write_text("image,status,enhance\nelsewhere/a_circle.png,ok,on\n", encoding="utf-8")
    out = tmp_path / "out"
    main(["--images", str(images), "--out", str(out), "--enhance", "off", "--compare", str(other)])
    report = (out / "report.html").read_text(encoding="utf-8").lower()
    assert "0 shared" in report and "warning" in report and "enhance" in report


def test_cli_rejects_missing_images_dir(tmp_path):
    from eval.run_defects import main

    with pytest.raises(SystemExit):
        main(["--images", str(tmp_path / "nope"), "--out", str(tmp_path / "out")])
