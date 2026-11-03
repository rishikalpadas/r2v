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
           "edge_f1", "nodes", "paths", "runtime_s", "reference"]
# <category>/clean/<stem>.png: clean scoring target for a degraded input. Clean targets and
# renders are keyed by stem, so keep one input per stem in each category (no a.png + a.jpg).
CLEAN_DIR = "clean"
ASPECT_TOLERANCE_PX = 1  # input scaled to the clean target may differ by rounding only

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


def _clean_target(path: Path) -> Path | None:
    target = path.parent / CLEAN_DIR / f"{path.stem}.png"
    return target if target.is_file() else None


def _same_aspect(a: np.ndarray, b: np.ndarray) -> bool:
    """True when `a` scaled to `b`'s width (or height) matches `b` within rounding."""
    (ha, wa), (hb, wb) = a.shape[:2], b.shape[:2]
    return (abs(ha * wb / wa - hb) <= ASPECT_TOLERANCE_PX
            and abs(wa * hb / ha - wb) <= ASPECT_TOLERANCE_PX)


def _row(category: str, path: Path, engine: str, status: str, reference: str,
         **extra) -> dict:
    row = dict.fromkeys(COLUMNS, "")
    row.update(category=category, image=path.name, engine=engine, status=status,
               reference=reference)
    row.update(extra)
    return row


def _evaluate(category: str, path: Path, engine: str, original: np.ndarray, out: Path,
              reference: str) -> dict:
    def row(status: str, **extra) -> dict:
        return _row(category, path, engine, status, reference, **extra)

    start = time.perf_counter()
    try:
        svg = RUNNERS[engine](path)
    except NotImplementedError:
        return row("skipped")
    except Exception:
        return row("error")
    runtime = time.perf_counter() - start
    if svg is None:
        return row("skipped")
    try:
        h, w = original.shape[:2]
        render = render_svg(svg, w, h)
        if render.shape[:2] != (h, w):
            # resvg keeps the SVG's aspect ratio; a 1 px rounding gap to the clean target
            # would otherwise make the metrics fail on mismatched shapes
            render = np.array(Image.fromarray(render, "RGBA").resize(
                (w, h), Image.Resampling.LANCZOS), dtype=np.uint8)
        nodes, paths = count_nodes(svg)
        metrics = {
            "ssim": ssim(original, render),
            "delta_e": mean_delta_e(original, render),
            "edge_f1": edge_f1(original, render),
        }
    except Exception:
        return row("error", runtime_s=f"{runtime:.4f}")
    _save(render, out / "renders" / engine / category / f"{path.stem}.png")
    return row("ok", nodes=nodes, paths=paths, runtime_s=f"{runtime:.4f}",
               **{k: f"{v:.6f}" for k, v in metrics.items()})


def _save(pixels: np.ndarray, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, "RGBA").save(dest)


def run_benchmark(dataset: Path, out: Path, engines: list[str]) -> Path:
    dataset, out = Path(dataset), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for category, path in _images(dataset):
        clean = _clean_target(path)
        reference = "clean" if clean else "input"
        try:
            original = np.array(Image.open(path).convert("RGBA"), dtype=np.uint8)
            target = (np.array(Image.open(clean).convert("RGBA"), dtype=np.uint8)
                      if clean else original)
        except Exception:
            rows.extend(_row(category, path, e, "error", reference) for e in engines)
            continue
        if not _same_aspect(original, target):  # clean target is not this input's source
            rows.extend(_row(category, path, e, "error", reference) for e in engines)
            continue
        _save(original, out / "renders" / "original" / category / f"{path.stem}.png")
        if clean:
            _save(target, out / "renders" / CLEAN_DIR / category / f"{path.stem}.png")
        # each render is scored against `target` (clean source if any), at the target's size
        rows.extend(_evaluate(category, path, e, target, out, reference) for e in engines)
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
