"""Benchmark runner: dataset x engines -> results.csv and rendered PNGs."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from PIL import Image

from eval.baselines import RUNNERS
from eval.metrics import count_nodes, edge_f1, mean_delta_e, render_svg, ssim

CATEGORIES = ("logos", "aa_logos", "line_art", "scans", "photos", "degraded")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".gif"}
COLUMNS = ["category", "image", "engine", "status", "ssim", "delta_e",
           "edge_f1", "nodes", "paths", "runtime_s"]

__all__ = ["RUNNERS", "run_benchmark", "main"]


def _images(dataset: Path) -> list[tuple[str, Path]]:
    found = []
    for category in CATEGORIES:
        cat_dir = dataset / category
        if not cat_dir.is_dir():
            continue
        for p in sorted(cat_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
                found.append((category, p))
    return found


def _row(category: str, path: Path, engine: str, status: str, **extra) -> dict:
    row = dict.fromkeys(COLUMNS, "")
    row.update(category=category, image=path.name, engine=engine, status=status)
    row.update(extra)
    return row


def _evaluate(category: str, path: Path, engine: str, original: np.ndarray, out: Path) -> dict:
    start = time.perf_counter()
    try:
        svg = RUNNERS[engine](path)
    except NotImplementedError:
        return _row(category, path, engine, "skipped")
    except Exception:
        return _row(category, path, engine, "error")
    runtime = time.perf_counter() - start
    if svg is None:
        return _row(category, path, engine, "skipped")
    try:
        h, w = original.shape[:2]
        render = render_svg(svg, w, h)
        nodes, paths = count_nodes(svg)
        metrics = {
            "ssim": ssim(original, render),
            "delta_e": mean_delta_e(original, render),
            "edge_f1": edge_f1(original, render),
        }
    except Exception:
        return _row(category, path, engine, "error", runtime_s=f"{runtime:.4f}")
    dest = out / "renders" / engine / category / f"{path.stem}.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(render, "RGBA").save(dest)
    return _row(category, path, engine, "ok", nodes=nodes, paths=paths,
                runtime_s=f"{runtime:.4f}", **{k: f"{v:.6f}" for k, v in metrics.items()})


def run_benchmark(dataset: Path, out: Path, engines: list[str]) -> Path:
    dataset, out = Path(dataset), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for category, path in _images(dataset):
        try:
            original = np.array(Image.open(path).convert("RGBA"), dtype=np.uint8)
        except Exception:
            rows.extend(_row(category, path, e, "error") for e in engines)
            continue
        orig_dest = out / "renders" / "original" / category / f"{path.stem}.png"
        orig_dest.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(original, "RGBA").save(orig_dest)
        rows.extend(_evaluate(category, path, e, original, out) for e in engines)
    csv_path = out / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("eval/dataset"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--engines", default="ours,vtracer,potrace,reference")
    args = parser.parse_args(argv)
    engines = [e for e in args.engines.split(",") if e]
    unknown = [e for e in engines if e not in RUNNERS]
    if unknown:
        parser.error(f"unknown engines: {', '.join(unknown)}")
    csv_path = run_benchmark(args.dataset, args.out, engines)
    from eval.report import write_report

    print(f"results: {csv_path}\nreport: {write_report(csv_path)}")


if __name__ == "__main__":
    main()
