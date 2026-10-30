# Evaluation harness

Benchmarks the raster-to-vector engine (`ours`) against baselines (`vtracer`, `potrace`).

## Dataset layout

```
<dataset>/
  logos/  line_art/  scans/  photos/  degraded/
    <name>.png            # input raster
    reference/<name>.svg  # optional Vectorizer.AI output, collected manually
```

Real images live in `eval/dataset/` (git-ignored; target about 200 per category).

**Licensing rule:** only add images whose license permits this use (benchmarking and
analysis). Record the source and license of each image outside the repo. Vectorizer.AI
outputs are collected by hand and placed in `<category>/reference/<name>.svg`.

## Synthetic dataset

```
python -m eval.make_synthetic_dataset            # -> eval/dataset_synthetic/logos/ (25 PNGs, 256 px)
```

Deterministic: 20 `flat_shapes` (seeds 0-19), 3 circles, `ring_in_ring`, `transparent_logo`.

## Running

Re-run the benchmark on every pipeline change and compare against the baseline below.

```
python -m eval.run_benchmark --dataset eval/dataset_synthetic --out eval/reports/synthetic \
    --engines ours,vtracer,potrace
```

Real images (once `eval/dataset/` is populated):

```
python -m eval.run_benchmark --dataset eval/dataset --out eval/reports/real \
    --engines ours,vtracer,potrace
```

Writes `results.csv` (columns `category,image,engine,status,ssim,delta_e,edge_f1,nodes,paths,runtime_s`),
rendered PNGs, and `report.html`. Output under `eval/reports/` is git-ignored.

## Reading the report

- `ssim`: structural similarity of the render vs the original (higher is better).
- `delta_e`: mean CIEDE2000 colour difference (lower is better).
- `edge_f1`: edge-map F1 between render and original (higher is better).
- `nodes` / `paths`: SVG complexity (fewer nodes at equal fidelity is better).
- `runtime_s`: wall time per image.
- `status`: `ok`, or `skipped` (engine unavailable) / error.

## Known limitations

- potrace is optional; the CLI is not installed here, so its rows are `skipped`.
- vtracer runs through the file-based `convert_image_to_svg_py`; its raw-bytes API
  segfaults on this platform.
- The current engine handles flat art only. Enhance, refine and photo modes arrive in
  later milestones, so photos/scans/degraded results are not yet meaningful.

## Synthetic baseline (2026-10-30)

25 synthetic logos at 256 px; potrace skipped (25 rows). Means over `ok` rows.

| engine  | ok | SSIM   | delta_e | edge_f1 | nodes | paths | runtime_s |
|---------|----|--------|---------|---------|-------|-------|-----------|
| ours    | 25 | 0.9893 | 0.1843  | 0.9962  | 45.52 | 7.32  | 0.3205    |
| vtracer | 25 | 0.9784 | 0.4361  | 0.9828  | 79.44 | 4.72  | 0.0258    |

`ours` beats vtracer on SSIM, delta_e and edge F1 with about 43% fewer nodes, but emits
more paths than vtracer (7.32 vs 4.72 per image) and is roughly 12x slower. This is synthetic flat art only; the real-image baseline is pending
a licensed dataset in `eval/dataset/`.
