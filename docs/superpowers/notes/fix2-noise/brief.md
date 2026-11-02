# Fix 2 — Noise-adaptive fitting (smoothing, corners, tolerance, MDL)

## Problem (user-reported, after fix 1)
On real JPEG art (user's "LOOKS"/HAPPINESS image) outlines still wobble and show small kinks. Cause:
fitting parameters are fixed and tuned for clean synthetic edges — `smooth_sigma=1` sample,
`tolerance_px=0.75`, `CORNER_WINDOW=2`, `corner_angle_deg=35`, `complexity_weight=2`. JPEG noise
(~0.3–0.8 px of edge jitter even after sub-pixel placement) produces fake corners every few px;
spans between fake corners are fitted separately, so joins have tangent breaks (kinks), and a tight
tolerance makes Béziers follow the jitter (many short segments).

## Goal
Fit cleanly through noise without rounding off real corners or small real features:
fewer fake corners, fewer segments, smooth (G1) joins where the shape is smooth, same or better
SSIM. Clean synthetic inputs must not get worse (all existing goldens pass unchanged).

## Design (binding unless you measure a problem — then report it with numbers)

1. **Per-curve noise estimate** — new `engine/geometry/noise.py`:
   `estimate_noise(dense: Sequence[Point], closed: bool) -> float` = robust std (1.4826·MAD) of the
   normal-direction residuals of the raw (unsmoothed, sub-pixel) samples against a strongly smoothed
   copy (sigma ≈ 4 samples), excluding ±3 samples around detected strong corners. Floor
   `NOISE_FLOOR_PX = 0.15` (processing px). Curves shorter than ~16 samples → floor.

2. **Scale-space corners** — replace single-scale `find_corners` use in the fit stage with
   `find_corners_multiscale(raw_dense, closed, angle_deg, scales=(1, 2, 4)·noise-aware)` in
   `corners.py`: compute turning angles on copies smoothed at increasing sigmas; a corner is accepted
   only if it is a local maximum above threshold at a coarse scale (sigma ≈ max(2, 3·noise) samples,
   window scaled with sigma) AND can be tracked down to the finest scale within ±2 samples; its
   index is the fine-scale location. Real corners persist across scale; noise bumps vanish.
   Keep `find_corners` unchanged for other callers.

3. **Adaptive smoothing** — sigma for the fitting samples = `max(params.smooth_sigma·scale,
   SMOOTH_PER_NOISE · noise)` with `SMOOTH_PER_NOISE ≈ 2.0`, applied between corners only (corner
   samples stay pinned, so smoothing never rounds a real corner). Implement as smoothing each
   corner-to-corner run separately with endpoints fixed.

4. **Adaptive tolerance** — effective `tolerance_px = max(params.tolerance_px·scale,
   TOL_PER_NOISE · noise)` with `TOL_PER_NOISE ≈ 2.5`.

5. **Noise-normalised MDL** — in `fit_span` selection, cost = `sse / σ² + complexity_weight · n_params`
   with `σ = max(noise, NOISE_FLOOR_PX)` (pass σ in; keep the old formula when σ is not given so
   direct callers/tests are unchanged). This also fixes the known scale³ drift of the old cost
   (sse grows with scale, penalty didn't). Re-tune `complexity_weight` default if needed and report
   the value and why.

6. **Span merge pass** — after fitting a curve, try merging each pair of adjacent spans across an
   interior corner whose turning angle at the coarse scale is < `MERGE_MAX_ANGLE ≈ 60°`: refit the
   union; accept if max error ≤ tolerance and MDL cost is lower. Then enforce G1 at remaining
   non-corner joins (Bézier handles collinear, as `snap_axis` already does).

Junction endpoints stay bitwise exact; closed chains close exactly; shared curves stay fitted once
(no change to the stitching contract). Name every constant with a one-line rationale. Keep
`fit_geometry.py` from growing: put corner/merge helpers in `engine/geometry/` modules.
`FitParams` gets `adaptive: bool = True` (validated); `adaptive=False` reproduces today exactly.

## Tests (RED first; record failing output)
Use `eval/synthetic.py` AA generators (from fix 1) and add `add_noise(px, sigma, seed)` +
JPEG round-trip helpers if missing.
1. AA disc r=40, JPEG q50 + Gaussian pixel noise σ=6 (8-bit): fitted curve has **0 interior corners**,
   ≤ 8 segments, Circle primitive detected (centre/radius within 0.4 px). Record before-fix numbers.
2. AA rect (rotated 15°), same noise: **exactly 4 corners**, each within 0.5 px of truth.
3. AA shape with a real 100° corner and a real small notch (e.g. a heart-like cusp ~6 px deep) under
   the same noise: both corners retained (within 1 px).
4. Smooth blob (superellipse/sine-perturbed circle), same noise: 0 corners; max tangent break at
   any join ≤ 5°.
5. Noise estimate: on synthetic edges with known jitter (sub-pixel samples + Gaussian normal noise
   σ ∈ {0, 0.3, 0.8}) `estimate_noise` within ±30% (or ≤ floor for σ=0).
6. `adaptive=False` → output identical to before for `flat_shapes(0..9)`, `aa_shapes(0..3)`, `aa_disc`.
7. All existing tests and goldens pass with thresholds untouched; coverage ≥ 80% on engine/.

## Real-image check (report only; do not commit the image)
The user's HAPPINESS image (1600×1411 JPEG, not in the repo — ask the user for it; pass its path via
the `R2V_REAL_IMAGE` environment variable to `scripts/fix2_eval.py`). Compare adaptive off vs on (current master = off baseline: nodes 1901, SSIM
0.9755, ΔE 1.963): nodes, paths, SSIM, ΔE, fit-stage time, number of interior corners, and
count of joins with tangent break > 10° that are not at corners. Target: nodes down ≥ 25%, SSIM not
below 0.9735, kink count down sharply. Save `looks_side_by_side.png` (source | off | on, crop
x 130–720, y 700–990) and `looks_zoom3x.png` plus both SVGs into `.superpowers/fix2-noise/`
(git-ignored scratch) in the repo.

Out of scope: colour segmentation (fix 3). Do not change segment.py.
