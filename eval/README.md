# Evaluation harness

Benchmarks the raster-to-vector engine (`ours`) against baselines (`vtracer`, `potrace`).

## Dataset layout

```
<dataset>/
  logos/  aa_logos/  line_art/  scans/  photos/  degraded/
    <name>.png            # input raster
    reference/<name>.svg  # optional Vectorizer.AI output, collected manually
    clean/<name>.png      # optional clean source of a degraded input (scoring target)
```

When `<category>/clean/<stem>.png` exists, the input `<stem>.*` is treated as a degraded
copy of it: the SVG is rendered at the clean image's size and ssim/delta_e/edge_f1 are
computed against the clean image, not against the degraded input. Scoring a vectorized
JPEG against the JPEG itself rewards reproducing JPEG noise (in a scratch run, bicubic
upscaling scored best against the degraded input and worst against the clean source).
Files in `clean/` (and `reference/`) are never benchmarked as inputs.

Real images live in `eval/dataset/` (git-ignored; target about 200 per category).

**Licensing rule:** only add images whose license permits this use (benchmarking and
analysis). Record the source and license of each image outside the repo. Vectorizer.AI
outputs are collected by hand and placed in `<category>/reference/<name>.svg`.

## Synthetic dataset

```
python -m eval.make_synthetic_dataset   # -> eval/dataset_synthetic/{logos,aa_logos,degraded}/
```

Deterministic: `logos/` holds 20 `flat_shapes` (seeds 0-19), 3 circles, `ring_in_ring`,
`transparent_logo` (256 px, no anti-aliasing). `aa_logos/` holds anti-aliased (box-filtered
supersampled) images: 4 `aa_shapes`, an `aa_disc` and an `aa_disc` on a transparent background.
`degraded/` holds each logo composited over white, downscaled 2x with Lanczos (128 px) and
saved as JPEG q75, plus the full-size clean logo in `degraded/clean/<stem>.png` as its
scoring target.

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

Writes `results.csv` (columns
`category,image,engine,status,ssim,delta_e,edge_f1,nodes,paths,runtime_s,reference`),
rendered PNGs, and `report.html`. Output under `eval/reports/` is git-ignored.

Engines: `ours` (default preset, Enhance on), `ours_noenhance` (the same preset with
the override `enhance.enabled=False`), `vtracer`, `potrace`, `reference` (Vectorizer.AI SVGs).
To compare Enhance on and off in one run:

```
python -m eval.run_benchmark --dataset eval/dataset_synthetic --out eval/reports/m4 \
    --engines ours,ours_noenhance,vtracer
```

## Reading the report

- `ssim`: structural similarity of the render vs the scoring reference (higher is better).
- `delta_e`: mean CIEDE2000 colour difference (lower is better).
- `edge_f1`: edge-map F1 between render and scoring reference (higher is better).
- `reference`: what the metrics compare against: `input` (the image itself) or `clean`
  (its clean source in `clean/`). The report shows the clean target next to the input.
- `nodes` / `paths`: SVG complexity (fewer nodes at equal fidelity is better).
- `runtime_s`: wall time per image.
- `status`: `ok`, or `skipped` (engine unavailable) / error.

## Known limitations

- potrace is optional; the CLI is not installed here, so its rows are `skipped`.
- vtracer runs through the file-based `convert_image_to_svg_py`; its raw-bytes API
  segfaults on this platform.
- The engine targets flat art. Since M4 the Enhance stage (ML upscale, JPEG deblock,
  optional denoise) handles low-resolution and compressed logos, so `degraded/` results
  are meaningful. Refine and photo modes arrive in later milestones, so photos/scans
  results are not yet meaningful.
- The synthetic `degraded/` set is one degradation (2x Lanczos + JPEG q75) of synthetic
  flat art; it says little about real scans, phone photos or heavier compression.

## Synthetic baseline (2026-10-30)

25 synthetic logos at 256 px; potrace skipped (25 rows). Means over `ok` rows.

| engine  | ok | SSIM   | delta_e | edge_f1 | nodes | paths | runtime_s |
|---------|----|--------|---------|---------|-------|-------|-----------|
| ours    | 25 | 0.9893 | 0.1843  | 0.9962  | 45.52 | 7.32  | 0.3205    |
| vtracer | 25 | 0.9784 | 0.4361  | 0.9828  | 79.44 | 4.72  | 0.0258    |

`ours` beats vtracer on SSIM, delta_e and edge F1 with about 43% fewer nodes, but emits
more paths than vtracer (7.32 vs 4.72 per image) and is roughly 12x slower. This is synthetic flat art only; the real-image baseline is pending
a licensed dataset in `eval/dataset/`.

## M4 baseline (2026-11-03)

Enhance stage on (default preset) vs off (`ours_noenhance`) vs vtracer, on the dev laptop
(RTX 4050 Laptop GPU, driver 616.64, weights installed; `device_used=cuda`). potrace not
installed, so not run. Means over `ok` rows; no rows failed. Command:
`python -m eval.run_benchmark --dataset eval/dataset_synthetic --out eval/reports/m4 --engines ours,ours_noenhance,vtracer`.
`degraded/` is scored against the clean 256 px source (`reference=clean`); the other
categories against their input.

**logos** (25, 256 px, exact colours)

| engine         | ok | SSIM   | delta_e | edge_f1 | nodes | paths | runtime_s |
|----------------|----|--------|---------|---------|-------|-------|-----------|
| ours           | 25 | 0.9893 | 0.1843  | 0.9962  | 45.52 | 7.32  | 0.1788    |
| ours_noenhance | 25 | 0.9893 | 0.1843  | 0.9962  | 45.52 | 7.32  | 0.1330    |
| vtracer        | 25 | 0.9784 | 0.4361  | 0.9828  | 79.44 | 4.72  | 0.0199    |

**aa_logos** (6, 256 px, anti-aliased)

| engine         | ok | SSIM   | delta_e | edge_f1 | nodes | paths | runtime_s |
|----------------|----|--------|---------|---------|-------|-------|-----------|
| ours           | 6  | 0.9951 | 0.1268  | 0.9947  | 56.83 | 7.50  | 2.3971    |
| ours_noenhance | 6  | 0.9951 | 0.0943  | 0.9859  | 23.83 | 4.00  | 0.1822    |
| vtracer        | 6  | 0.9819 | 0.5294  | 0.9622  | 91.83 | 5.17  | 0.0208    |

**degraded** (25, 128 px JPEG q75, scored against the clean 256 px logo)

| engine         | ok | SSIM   | delta_e | edge_f1 | nodes  | paths | runtime_s |
|----------------|----|--------|---------|---------|--------|-------|-----------|
| ours           | 25 | 0.9760 | 0.6197  | 0.9863  | 62.72  | 7.80  | 0.9938    |
| ours_noenhance | 25 | 0.9373 | 1.3167  | 0.8808  | 213.64 | 30.44 | 0.2092    |
| vtracer        | 25 | 0.8969 | 2.0163  | 0.8842  | 511.92 | 50.64 | 0.0273    |

GPU runtime per image with Enhance on: logos 0.18 s mean (0.13 s median; the first image
pays about 1 s of one-time startup), aa_logos 2.40 s mean (1.59 s median), degraded 0.99 s
mean (0.95 s median). Enhance passes exact-colour input through unchanged, so `logos` is
identical with Enhance on and off.

**Real image** (`eval/dataset/logos/happiness.jpg`, 1600x1411 JPEG, scored against itself;
the only real image so far, so this is one data point, not a baseline):

| engine         | SSIM   | delta_e | edge_f1 | nodes | paths | runtime_s |
|----------------|--------|---------|---------|-------|-------|-----------|
| ours           | 0.9781 | 0.8689  | 0.9934  | 3800  | 339   | 51.24     |
| ours_noenhance | 0.9771 | 0.9095  | 0.9894  | 3525  | 322   | 8.60      |
| vtracer        | 0.9600 | 1.2359  | 0.9956  | 17470 | 1420  | 0.88      |

With Enhance on, this image runs FBCNN deblock + `realesrgan_x4plus` at an effective 2x
(capped by `max_output_pixels`). Stage times in a separate run: enhance 33.3 s, segment 4.3 s,
fit 12.1 s.

**What changed since 2026-10-30:** the synthetic set gained `aa_logos` and `degraded`;
`results.csv` gained the `reference` column, and degraded inputs are scored against their
clean source. `logos` fidelity and node counts are unchanged (Enhance is a pass-through
there). `ours` runtime on logos reads 0.18 s now vs 0.32 s then, but the two runs were in
different sessions, so treat that as timing noise rather than a speed-up.

**Reading these numbers:**

- On `degraded`, Enhance is a clear win: SSIM 0.976 vs 0.937 (off) and 0.897 (vtracer),
  with about 3.4x fewer nodes than Enhance off and 8x fewer than vtracer.
- On `aa_logos`, Enhance on matches SSIM and improves edge F1, but delta_e is worse
  (0.127 vs 0.094) and nodes and paths are more than 2x higher (56.8 vs 23.8 nodes).
  The run is 13x slower. Anti-aliased but clean 256 px art does not seem to benefit
  from 4x SR.
- On happiness.jpg, Enhance costs 6x the runtime (51 s vs 8.6 s) for a small fidelity
  gain (SSIM +0.001, delta_e -0.04) and 8% more nodes.
- Scoring against the clean source matters. Against the degraded JPEG itself, a method
  that copies JPEG noise looks good. In a scratch run, bicubic upscaling scored best
  against the degraded input but worst against the clean source.

## GPU setup

The Enhance stage (M4) runs ML models through PyTorch and spandrel, installed with the
optional `gpu` extra. The flat-art pipeline still works without them. Commands used on
the dev laptop (RTX 4050 Laptop GPU, driver 616.64, CUDA 13.4), in order:

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e .[dev]
.venv\Scripts\python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu134
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
