# Evaluation harness

Benchmarks the raster-to-vector engine (`ours`) against baselines (`vtracer`, `potrace`).

## Dataset layout

```
<dataset>/
  logos/  aa_logos/  line_art/  scans/  photos/  degraded/
    <name>.png            # input raster
    reference/<name>.svg  # optional Vectorizer.AI output, collected manually
```

Real images live in `eval/dataset/` (git-ignored; target about 200 per category).

**Licensing rule:** only add images whose license permits this use (benchmarking and
analysis). Record the source and license of each image outside the repo. Vectorizer.AI
outputs are collected by hand and placed in `<category>/reference/<name>.svg`.

## Synthetic dataset

```
python -m eval.make_synthetic_dataset            # -> eval/dataset_synthetic/{logos,aa_logos}/ (256 px)
```

Deterministic: `logos/` holds 20 `flat_shapes` (seeds 0-19), 3 circles, `ring_in_ring`,
`transparent_logo` (no anti-aliasing). `aa_logos/` holds anti-aliased (box-filtered
supersampled) images: 4 `aa_shapes`, an `aa_disc` and an `aa_disc` on a transparent background.

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

## GPU setup

The Enhance stage (M4) runs ML models through PyTorch and spandrel, installed with the
optional `gpu` extra. The flat-art pipeline still works without them. Commands used on
the dev laptop (RTX 4050 Laptop GPU, driver 616.64, CUDA 13.4), in order:

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e .[dev]
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu134
.venv\Scripts\python -m pip install torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cu134
.venv\Scripts\python -m pip install -e .[dev,gpu]
.venv\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.venv\Scripts\python -m engine.ml.fetch_weights --all
```

- Install torch and torchvision from the CUDA index *before* the `gpu` extra. Otherwise
  pip resolves `torch` from PyPI and installs the CPU-only build.
- Pick the `cuXYZ` index that matches the CUDA version `nvidia-smi` reports (here 13.4, so `cu134`).
- The check command should print `True NVIDIA GeForce RTX 4050 Laptop GPU`.
- Tests marked `gpu` are skipped automatically when torch is missing or CUDA is unavailable.

Weights download only through `python -m engine.ml.fetch_weights` (`--all` or names), never at
import or inside `vectorize()`. Each file is sha256-checked against the manifest in
`engine/ml/weights.py`. They are cached in `%LOCALAPPDATA%\raster2vector\weights` on Windows and
`~/.cache/raster2vector/weights` elsewhere; set `R2V_WEIGHTS_DIR` to override.

| name                         | kind    | size   | license      | source repo          |
|------------------------------|---------|--------|--------------|----------------------|
| `realesrgan_x4plus`          | sr (4x) | 67 MB  | BSD-3-Clause | xinntao/Real-ESRGAN  |
| `realesrgan_x4plus_anime_6b` | sr (4x) | 18 MB  | BSD-3-Clause | xinntao/Real-ESRGAN  |
| `fbcnn_color`                | deblock | 288 MB | Apache-2.0   | jiaxi-jiang/FBCNN    |
| `scunet_color_real_psnr`     | denoise | 72 MB  | Apache-2.0   | cszn/SCUNet (file hosted in the cszn/KAIR release) |

## Trying the engine on your own images

- **Preview page:** double-click `preview.bat` (or run `.venv\Scripts\python -m app.server`) and open http://127.0.0.1:8765. Drop an image to see the original next to the SVG, with similarity score, node count, timing, an outlines view and a download button.
- **Command line:** `.venv\Scripts\python -m app.cli path\to\image.png` writes `image.svg` next to it. Options: `--max-colors N`, `--layer-mode stacked|cutout`, `--no-fit`, `--tolerance PX`, `-o out.svg`.
- Sample inputs live in `samples/`.
