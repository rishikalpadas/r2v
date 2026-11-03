# Quality roadmap: clean, sharp, smooth output on any image

Goal: output comparable to vectorizer.ai on logos, lettering and flat art. That means clean shapes,
smooth curves, crisp corners, even stroke widths, no specks, few nodes, on any input image.

## Principle

Hand inspection of single images does not scale. Every defect class gets an automatic detector.
Every fix is judged on the whole test set by those detectors, plus the existing metrics:
SSIM, ΔE, edge F1, nodes and time.

## Q1: defect report (first)

`eval/defects.py` plus `eval/run_defects.py`. The runner converts every image in a folder and runs each detector.
It writes a CSV, plus an HTML report sorted worst-first with automatic 5× source-vs-output crops of the worst spots.

| Defect | Detector |
|---|---|
| Crossing curves | Proper crossings between neighbouring fitted curves |
| Specks | Output shapes below a minimum area (original px) |
| Wobble | Curvature sign changes and total turning versus a faired version of the same curve |
| Uneven thin bands | Width variance along bands ≤ a few px wide |
| Junction spikes | Sharp turns close to curve endpoints at shared junctions |
| Worst patches | Local ΔE/SSIM windows between the render and the source |

## Q2+: fixes, in the order the report ranks them

Expected candidates:

- **Speck cleanup:** merge regions below a minimum area into their best neighbour.
- **Curve fairing:** fit long gentle spans with fewer, better-placed nodes and a curvature penalty.
- **Corner handling:** keep corners crisp and avoid false corners.
- **Thin-band handling:** fit both sides of a thin band as one centreline plus a width, so the band stays parallel.
- **Junction cleanup:** remove small spikes and bumps where curves meet at a junction.

Gate for each fix: its target detector improves on the test set, no other detector gets worse by more than noise,
and SSIM/ΔE do not drop. All goldens and digests stay green, or are re-recorded with a reason.

## Later

- **M5 Refine** (differentiable rendering) runs on top of the cleaned geometry, under the same gate.
- **Learned components** (for example, a segmentation model) are added only if the report shows classical fixes plateau.
