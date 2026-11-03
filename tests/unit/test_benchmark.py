import csv
import sys
import types
from pathlib import Path

import pytest

from eval.report import write_report
from eval.run_benchmark import RUNNERS, run_benchmark
from eval.synthetic import circle, flat_shapes, ring_in_ring, to_png_bytes

HEADER = ["category", "image", "engine", "status", "ssim", "delta_e",
          "edge_f1", "nodes", "paths", "runtime_s", "reference"]


def _svg_for(pixels) -> str:
    h, w = pixels.shape[:2]
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">'
            f'<circle cx="64" cy="64" r="40" fill="#000"/></svg>')


@pytest.fixture
def dataset(tmp_path) -> Path:
    cat = tmp_path / "ds" / "logos"
    (cat / "reference").mkdir(parents=True)
    imgs = {"a": circle(), "b": ring_in_ring()}
    for name, px in imgs.items():
        (cat / f"{name}.png").write_bytes(to_png_bytes(px))
    (cat / "reference" / "a.svg").write_text(_svg_for(imgs["a"]))
    return tmp_path / "ds"


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_results_csv_columns_and_rows(dataset, tmp_path, monkeypatch):
    monkeypatch.setitem(RUNNERS, "boom", lambda p: 1 / 0)
    csv_path = run_benchmark(dataset, tmp_path / "out", ["vtracer", "reference", "boom"])
    with csv_path.open(newline="", encoding="utf-8") as f:
        assert next(csv.reader(f)) == HEADER
    assert len(_rows(csv_path)) == 6


def test_failing_runner_recorded_as_error(dataset, tmp_path, monkeypatch):
    monkeypatch.setitem(RUNNERS, "boom", lambda p: 1 / 0)
    rows = _rows(run_benchmark(dataset, tmp_path / "out", ["vtracer", "boom"]))
    assert {r["status"] for r in rows if r["engine"] == "boom"} == {"error"}
    assert {r["status"] for r in rows if r["engine"] == "vtracer"} == {"ok"}


def test_missing_reference_is_skipped(dataset, tmp_path):
    rows = _rows(run_benchmark(dataset, tmp_path / "out", ["reference"]))
    status = {r["image"]: r["status"] for r in rows}
    assert status == {"a.png": "ok", "b.png": "skipped"}


def test_small_image_is_error_not_crash(tmp_path):
    cat = tmp_path / "ds" / "logos"
    cat.mkdir(parents=True)
    (cat / "tiny.png").write_bytes(to_png_bytes(flat_shapes(1, size=4)))
    rows = _rows(run_benchmark(tmp_path / "ds", tmp_path / "out", ["vtracer"]))
    assert rows[0]["status"] == "error"


def test_report_escapes_names(tmp_path):
    # '<' is illegal in Windows file names, so feed the report a hand-written CSV.
    csv_path = tmp_path / "results.csv"
    lines = [
        ",".join(HEADER),
        "logos,a<b>.png,vtracer,ok,0.9,1.0,0.8,10,2,0.1,input",
        'logos,a<b>.png,"x<i>",error,,,,,,,input',
    ]
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    html = write_report(csv_path).read_text(encoding="utf-8")
    assert "a&lt;b&gt;" in html
    assert "<b>" not in html
    assert "<i>" not in html


def test_ours_skipped_only_when_engine_missing(dataset, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "engine", None)
    rows = _rows(run_benchmark(dataset, tmp_path / "out", ["ours"]))
    assert {r["status"] for r in rows} == {"skipped"}


def test_ours_broken_engine_is_error(dataset, tmp_path, monkeypatch):
    broken = types.ModuleType("engine")  # importable but lacks vectorize/to_svg
    monkeypatch.setitem(sys.modules, "engine", broken)
    rows = _rows(run_benchmark(dataset, tmp_path / "out", ["ours"]))
    assert {r["status"] for r in rows} == {"error"}


def test_ours_runs_real_engine(dataset, tmp_path):
    rows = _rows(run_benchmark(dataset, tmp_path / "out", ["ours"]))
    assert {r["status"] for r in rows} == {"ok"}


def _black_square_svg(side: int) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{side}" height="{side}">'
            f'<rect x="0" y="0" width="{side}" height="{side}" fill="#000"/></svg>')


@pytest.fixture
def degraded_dataset(tmp_path) -> Path:
    import numpy as np
    from PIL import Image

    cat = tmp_path / "ds" / "degraded"
    (cat / "clean").mkdir(parents=True)
    # input: a white 32 px JPEG; clean target: a black 64 px PNG (deliberately different)
    Image.new("RGB", (32, 32), (255, 255, 255)).save(cat / "x.jpg", format="JPEG", quality=75)
    black = np.zeros((64, 64, 4), dtype=np.uint8)
    black[..., 3] = 255
    (cat / "clean" / "x.png").write_bytes(to_png_bytes(black))
    Image.new("RGB", (32, 32), (255, 255, 255)).save(cat / "y.jpg", format="JPEG", quality=75)
    return tmp_path / "ds"


def test_clean_target_is_scoring_reference(degraded_dataset, tmp_path, monkeypatch):
    from PIL import Image

    monkeypatch.setitem(RUNNERS, "black", lambda p: _black_square_svg(32))
    out = tmp_path / "out"
    rows = {r["image"]: r for r in _rows(run_benchmark(degraded_dataset, out, ["black"]))}
    assert set(rows) == {"x.jpg", "y.jpg"}  # nothing from clean/ is benchmarked as an input
    x, y = rows["x.jpg"], rows["y.jpg"]
    assert x["status"] == y["status"] == "ok"
    assert x["reference"] == "clean"
    assert float(x["ssim"]) > 0.99 and float(x["delta_e"]) < 0.5  # black vs black target
    with Image.open(out / "renders" / "black" / "degraded" / "x.png") as render:
        assert render.size == (64, 64)  # rendered at the clean target's size
    assert y["reference"] == "input"
    assert float(y["delta_e"]) > 50  # black render vs white input
    with Image.open(out / "renders" / "black" / "degraded" / "y.png") as render:
        assert render.size == (32, 32)


def test_clean_target_shown_in_report(degraded_dataset, tmp_path, monkeypatch):
    monkeypatch.setitem(RUNNERS, "black", lambda p: _black_square_svg(32))
    out = tmp_path / "out"
    html = write_report(run_benchmark(degraded_dataset, out, ["black"])).read_text("utf-8")
    assert "renders/clean/degraded/x.png" in html
    assert (out / "renders" / "clean" / "degraded" / "x.png").is_file()
    assert "<th>reference</th>" in html


def test_ours_noenhance_disables_enhance(dataset, monkeypatch):
    calls = []
    fake = types.ModuleType("engine")
    fake.vectorize = lambda data, **kw: calls.append(kw) or types.SimpleNamespace(doc="d")
    fake.to_svg = lambda doc: "<svg/>"
    monkeypatch.setitem(sys.modules, "engine", fake)
    assert RUNNERS["ours_noenhance"](dataset / "logos" / "a.png") == "<svg/>"
    assert calls == [{"overrides": {"enhance.enabled": False}}]
