"""Defect detectors on small synthetic cases with known answers, plus a CLI smoke test."""
import csv
import math
from pathlib import Path

import numpy as np
import pytest

from eval.defects import (
    WOBBLE_FLAG_RAD,
    CurveGeom,
    ShapeGeom,
    detect_crossings,
    detect_junction_spikes,
    detect_specks,
    detect_thin_bands,
    detect_wobble,
    detect_worst_patches,
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


def test_clean_circle_has_low_wobble():
    res = detect_wobble([_closed(0, _ring(lambda t: np.full_like(t, 60.0)))])
    assert res.score < 0.1
    assert res.spots == ()


def test_jittered_circle_has_high_wobble():
    jitter = _ring(lambda t: 60.0 + 0.8 * np.sin(40 * t))  # ~9 px wavelength, 0.8 px amplitude
    res = detect_wobble([_closed(0, jitter)])
    assert res.score > 10 * WOBBLE_FLAG_RAD
    assert len(res.spots) >= 1


def test_jitter_scores_far_above_clean():
    clean = detect_wobble([_closed(0, _ring(lambda t: np.full_like(t, 60.0)))])
    rng = np.random.default_rng(0)
    noise = np.convolve(rng.normal(0, 0.6, 721), np.ones(5) / 5, mode="same")
    noisy = detect_wobble([_closed(0, _ring(lambda t: 60.0 + noise))])
    assert noisy.score > clean.score + 5 * WOBBLE_FLAG_RAD


def test_square_true_corners_are_not_wobble():
    sq = np.array([(20, 20), (120, 20), (120, 120), (20, 120), (20, 20)], float)
    res = detect_wobble([_closed(0, sq)])
    assert res.score == 0.0
    assert res.spots == ()


def test_clean_s_curve_has_low_wobble():
    x = np.linspace(0, 200, 400)
    s = np.stack([x, 100 + 30 * np.sin(x / 200 * 2 * math.pi)], axis=1)
    res = detect_wobble([_open(0, s)])
    assert res.score < 0.1
    assert res.spots == ()


def test_wobble_spot_is_on_the_jittery_part():
    x = np.linspace(0, 200, 801)
    y = np.where(x > 120, 50 + 0.7 * np.sin(x * 0.7), 50.0)
    res = detect_wobble([_open(0, np.stack([x, y], axis=1))])
    assert res.spots
    assert all(s.box[2] > 115 for s in res.spots)


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


def test_curves_meeting_at_a_junction_do_not_cross():
    a = _open(0, [(0, 0), (10, 10)])
    b = _open(1, [(10, 10), (20, 0)])
    c = _open(2, [(10, 10), (10, 30)])
    assert detect_crossings([a, b, c]).score == 0


# ---------------------------------------------------------------- junction spikes


def _star(hook: bool):
    j = (50.0, 50.0)
    east = [j, (52.0, 47.0), (53.5, 50.0), (60, 50), (90, 50)] if hook else [j, (60, 50), (90, 50)]
    return [
        _open(0, east, 1, 2),
        _open(1, [j, (50 - 20, 50 - 34.6)], 2, 3),
        _open(2, [j, (50 - 20, 50 + 34.6)], 3, 1),
    ]


def test_spike_at_a_junction_is_found():
    res = detect_junction_spikes(_star(hook=True))
    assert res.score > 0
    assert any(_inside(s.box, 50, 50, pad=1) for s in res.spots)


def test_clean_junction_has_no_spike():
    assert detect_junction_spikes(_star(hook=False)).score == 0


# ---------------------------------------------------------------- worst patches


def test_worst_patch_is_where_the_render_differs():
    src = np.full((96, 96, 4), 255, np.uint8)
    out = src.copy()
    out[60:70, 20:30, :3] = (200, 30, 30)
    res = detect_worst_patches(src, out)
    assert res.score > 0
    assert _inside(res.spots[0].box, 25, 65)


# ---------------------------------------------------------------- CLI smoke


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
        assert (out / "svg" / f"{Path(name).stem}.svg").is_file()
    report = (out / "report.html").read_text(encoding="utf-8")
    assert "a_circle.png" in report and "c_broken.png" in report

    out2 = tmp_path / "out2"
    main(["--images", str(images), "--out", str(out2), "--enhance", "off",
          "--compare", str(out / "defects.csv")])
    report2 = (out2 / "report.html").read_text(encoding="utf-8")
    assert "Δ" in report2
