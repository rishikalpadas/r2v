"""Run evaluation across all 35 logos in eval/dataset/logos.
Reuses the 17 already-computed results from defects_after_gradient,
computes the 18 new images, and writes unified report.html and composites.
"""
from __future__ import annotations

import csv
import shutil
import sys
import time
import traceback
from pathlib import Path

from eval.defects import DETECTORS
from eval.metrics import count_nodes, render_svg
from eval.run_defects import (
    COLUMNS,
    TOP_N,
    _images,
    _image_section,
    _load_rgba,
    _ranking,
    _read_csv,
    _safe,
    _totals,
    analyse_image,
    compare_warnings,
    git_sha,
    write_report,
)


def main():
    images_dir = Path("eval/dataset/logos")
    out_dir = Path("eval/reports/defects_35_baseline")
    prior_dir = Path("eval/reports/defects_after_gradient")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "svg").mkdir(parents=True, exist_ok=True)
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)
    
    # Read prior rows if available
    prior_rows = {}
    prior_csv = prior_dir / "defects.csv"
    if prior_csv.is_file():
        prior_rows = _read_csv(prior_csv)
            
    # Also copy prior svgs and crops
    for svg_file in (prior_dir / "svg").glob("*.svg"):
        dest = out_dir / "svg" / svg_file.name
        if not dest.is_file():
            shutil.copy2(svg_file, dest)
            
    for crop_folder in (prior_dir / "crops").iterdir():
        if crop_folder.is_dir():
            dest_folder = out_dir / "crops" / crop_folder.name
            if not dest_folder.is_dir():
                shutil.copytree(crop_folder, dest_folder)

    sha = git_sha()
    all_images = _images(images_dir)
    print(f"Total images in dataset: {len(all_images)}", flush=True)
    
    rows = []
    csv_path = out_dir / "defects.csv"
    
    # We will write out defects.csv progressively
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        
        for name, path in all_images:
            if name in prior_rows and (out_dir / "svg" / f"{_safe(name)}.svg").is_file():
                row = prior_rows[name]
                print(f"[REUSED 17] {name}: combined={row.get('combined', '')}", flush=True)
            else:
                print(f"[COMPUTING NEW] {name}...", flush=True)
                t0 = time.perf_counter()
                try:
                    row = analyse_image(name, path, out_dir, enhance=True, top=TOP_N, sha=sha)
                    dt = time.perf_counter() - t0
                    print(f"  -> ok in {dt:.1f}s | combined={row.get('combined', '')} | nodes={row.get('nodes', '')}", flush=True)
                except Exception as exc:
                    row = dict.fromkeys(COLUMNS, "")
                    row.update(
                        image=name,
                        status="error",
                        enhance="on",
                        git_sha=sha,
                        error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}",
                    )
                    print(f"  -> ERROR: {exc}", flush=True)
            writer.writerow(row)
            f.flush()
            rows.append(row)
            
    # Ensure every row has _svg and _spots for report generation
    for r in rows:
        stem = _safe(r["image"])
        r["_svg"] = f"svg/{stem}.svg"
        if "_spots" not in r:
            shown = {}
            crop_dir = out_dir / "crops" / stem
            if crop_dir.is_dir():
                for d in DETECTORS:
                    items = []
                    for k in range(TOP_N):
                        crop_file = crop_dir / f"{d}_{k}.png"
                        if crop_file.is_file():
                            items.append((None, f"crops/{stem}/{d}_{k}.png"))
                    if items:
                        shown[d] = items
            r["_spots"] = shown
            
    report_path = write_report(rows, out_dir, compare=None)
    print(f"\nAll done! defects: {csv_path}\nreport: {report_path}", flush=True)


if __name__ == "__main__":
    main()
