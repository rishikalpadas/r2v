import csv
from pathlib import Path

import pytest

from eval.report import write_report
from eval.run_benchmark import RUNNERS, run_benchmark
from eval.synthetic import circle, flat_shapes, ring_in_ring, to_png_bytes

HEADER = ["category", "image", "engine", "status", "ssim", "delta_e",
          "edge_f1", "nodes", "paths", "runtime_s"]


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
        "logos,a<b>.png,vtracer,ok,0.9,1.0,0.8,10,2,0.1",
        'logos,a<b>.png,"x<i>",error,,,,,,',
    ]
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    html = write_report(csv_path).read_text(encoding="utf-8")
    assert "a&lt;b&gt;" in html
    assert "<b>" not in html
    assert "<i>" not in html
