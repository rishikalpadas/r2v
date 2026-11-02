# raster2vector Engine M4 (Enhance: ML Upscale + Deblock + Denoise) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the Enhance stage (spec §5.2): ML 4× super-resolution, JPEG deblocking and denoising before segmentation, so tracing runs on clean upscaled pixels and edges land at sub-pixel accuracy in original coordinates.

**Architecture:** New `engine/ml/` package owns device selection, a weights manifest with verified downloads, and tiled inference with OOM back-off. `engine/stages/enhance.py` turns a `RasterImage` (scale 1) into a cleaned, upscaled `RasterImage` (scale 4, same `original_size`). The pipeline already maps geometry back by `scale` in `build_doc` and multiplies length thresholds by `scale` (area thresholds by `scale²`), so downstream stages need no new coordinate logic. Every ML step has a non-ML fallback (spec §6).

**Tech Stack:** PyTorch (CUDA build), spandrel (MIT; loads ESRGAN/SwinIR/SCUNet/FBCNN-family weights without basicsr), existing numpy/Pillow/scipy stack.

**Spec:** `docs/superpowers/specs/2026-10-30-raster2vector-engine-design.md`

**Runs on:** the user's laptop with an RTX 4050 Laptop GPU (6 GB VRAM). Every test that needs CUDA is marked `@pytest.mark.gpu` and skipped when `torch.cuda.is_available()` is false; everything else must pass on CPU.

**Starting point:** branch with milestones 1–3 merged (flat-art pipeline, eval harness, scale mapping in `build_doc`). Read `eval/README.md` and `engine/pipeline.py` first.

## Global Constraints

- Upscale factor 4 (spec §5.2). Output `RasterImage.scale == 4.0`, `original_size` unchanged.
- Fallbacks (spec §6): upscaler fails → bicubic upscale; any ML failure is logged in the `StageReport` params, never raised. Engine raises only `InputError` / `StageError`.
- GPU out-of-memory → retry with tile size halved, down to a floor of 64 px; below the floor → bicubic fallback.
- Model weights: only licenses permitting commercial use (spec §1). Each weight in the manifest records name, URL, sha256, license. Weights download only via an explicit command, never at import or inside `vectorize()`.
- Weights cache dir: `%LOCALAPPDATA%/raster2vector/weights` on Windows, `~/.cache/raster2vector/weights` elsewhere; overridable by env var `R2V_WEIGHTS_DIR`.
- `engine/` must still not import `eval/`. Torch and spandrel are imported lazily inside `engine/ml/` so the CPU-only flat-art pipeline keeps working without them (optional extra `[gpu]`).
- All engine types frozen; stages pure. Coverage ≥ 80% on `engine/` measured on CPU (GPU-only lines may be excluded with `# pragma: no cover` only inside `engine/ml/device.py`).
- Golden thresholds from M1–M3 must not be loosened.

## Review Focus

1. **RGBA input** → models are RGB-only; alpha must be upscaled separately (bicubic then re-threshold at 128) and transparent areas stay transparent after enhance. Test in Task 4.
2. **Tiny images** (1×1 … 16×16) → no crash; skip ML below 16 px on the short side and use bicubic. Test in Task 4.
3. **Large images** (4096 px long side → 16384 px after 4×) → memory: enhance output must be capped; if `long_side * 4 > 8192`, reduce the factor to fit 8192 (minimum 1) and record it in `scale`. Test in Task 4.
4. **No CUDA / missing weights** → falls back to bicubic with a report note, pipeline still succeeds. Test in Task 4.
5. **Seams between tiles** → overlap + feathered blending; a flat-color tile boundary must not show a step > 2 levels. Test in Task 2.

---

## File Structure

```
engine/ml/__init__.py        lazy-import helpers, `ml_available() -> bool`
engine/ml/device.py          pick_device(pref: str) -> str ("cuda" | "cpu")
engine/ml/weights.py         MANIFEST, weights_dir(), weight_path(name), fetch(name), verify(name)
engine/ml/fetch_weights.py   CLI: python -m engine.ml.fetch_weights [--all | name ...]
engine/ml/tiling.py          run_tiled(model_fn, rgb, scale, tile, overlap) with OOM back-off
engine/ml/models.py          load_model(name, device) (spandrel), cached per (name, device)
engine/stages/enhance.py     EnhanceParams usage, enhance(image, params) -> RasterImage
engine/presets.py            + EnhanceParams; logo preset gains "enhance" stage
engine/pipeline.py           + "enhance" in STAGES (runs before segment)
tests/unit/test_weights.py, test_tiling.py, test_enhance.py, tests/gpu/test_enhance_gpu.py
```

---

### Task 1: GPU environment, device selection, weights manifest

**Files:** Create `engine/ml/__init__.py`, `engine/ml/device.py`, `engine/ml/weights.py`, `engine/ml/fetch_weights.py`; Modify `pyproject.toml` (extra `gpu = ["torch", "spandrel"]`, register pytest marker `gpu`); Test `tests/unit/test_weights.py`

**Interfaces — Produces:**
- `ml_available() -> bool` — True when torch and spandrel import
- `pick_device(pref: Literal["auto","cuda","cpu"] = "auto") -> str`
- `WeightSpec(name: str, url: str, sha256: str, license: str, scale: int, kind: Literal["sr","deblock","denoise"])`
- `MANIFEST: dict[str, WeightSpec]` with at least: `realesrgan_x4plus` (general SR), `realesrgan_x4plus_anime_6b` (flat-art SR), one JPEG-deblock model (FBCNN family), one denoise model (SCUNet or NAFNet family)
- `weights_dir() -> Path`, `weight_path(name) -> Path`, `verify(name) -> bool` (sha256 match), `fetch(name) -> Path` (download to temp file, verify, atomic rename; mismatch → delete + raise)

- [ ] **Step 1: Laptop setup (record exact commands in `eval/README.md` → new "GPU setup" section).** Create venv (`py -3.12 -m venv .venv` if torch has no wheel for the default Python), install CUDA torch from the official index (`pip install torch --index-url https://download.pytorch.org/whl/cu12x` matching the installed driver), then `pip install -e .[dev,gpu]`. Verify: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` → `True NVIDIA GeForce RTX 4050 Laptop GPU`.
- [ ] **Step 2: Fill the manifest.** For each weight, take the URL from the model's official GitHub release, confirm its license permits commercial use (record the license string), download once, compute sha256, record it. Any weight without a clear commercial-use license is left out and noted in the report.
- [ ] **Step 3: Write failing tests:** `test_weights_dir_env_override` (R2V_WEIGHTS_DIR respected); `test_fetch_rejects_bad_hash` (monkeypatched downloader serving wrong bytes → raises, no file left behind); `test_fetch_atomic` (existing valid file untouched on re-fetch); `test_manifest_entries_complete` (every field non-empty, sha256 is 64 hex chars, license non-empty); `test_pick_device_cpu_when_no_cuda` (monkeypatch); `test_engine_imports_without_torch` (simulate missing torch via `sys.modules` patch → `import engine; engine.vectorize(png)` still works).
- [ ] **Step 4:** Run → FAIL. **Step 5:** Implement. **Step 6:** Run → PASS.
- [ ] **Step 7:** `python -m engine.ml.fetch_weights --all` on the laptop → all verified.
- [ ] **Step 8:** Commit `feat: ml device selection and verified weights manifest`.

---

### Task 2: Tiled inference with OOM back-off

**Files:** Create `engine/ml/tiling.py`; Test `tests/unit/test_tiling.py`

**Interfaces — Produces:** `run_tiled(model_fn: Callable[[np.ndarray], np.ndarray], rgb: np.ndarray /*HxWx3 float32 0..1*/, scale: int, tile: int = 512, overlap: int = 32, min_tile: int = 64) -> np.ndarray /*(H*scale)x(W*scale)x3*/`

- Tiles of `tile` px with `overlap` px context on each side; outputs cropped and blended with a linear feather across the overlap.
- When `model_fn` raises an exception whose type name is `OutOfMemoryError` (torch.cuda.OutOfMemoryError) or message contains `"out of memory"`: free cache if torch present, halve `tile`, restart; below `min_tile` → re-raise as `MemoryError`.

- [ ] **Step 1: Write failing tests** (CPU, `model_fn` = nearest-neighbour upscale via `np.repeat`):
  - `test_identity_matches_whole_image`: tiled result equals whole-image `np.repeat` exactly for a 300×200 random image with `tile=64`.
  - `test_no_seam_on_flat_color`: flat color image, a model that adds a tile-dependent ±1 offset → max abs step across tile boundaries ≤ 2/255.
  - `test_oom_backoff`: model raising a fake `OutOfMemoryError` when input > 100 px → succeeds with smaller tiles; records final tile size (return it via a second return value or attribute; pick one and document).
  - `test_oom_floor_raises_memoryerror`.
  - `test_image_smaller_than_tile`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat: tiled inference with feathered blending and OOM back-off`.

---

### Task 3: Model loading

**Files:** Create `engine/ml/models.py`; Test `tests/unit/test_models.py`, `tests/gpu/test_models_gpu.py`

**Interfaces — Produces:** `load_model(name: str, device: str) -> Callable[[np.ndarray], np.ndarray]` — loads `weight_path(name)` with `spandrel.ModelLoader`, moves to device, eval mode, fp16 on CUDA when the architecture supports it (`model.supports_half`), wraps as numpy-in/numpy-out (HxWx3 float32 0..1). Cached with `functools.lru_cache` on `(name, device)`. Missing weight file → `FileNotFoundError` naming `python -m engine.ml.fetch_weights <name>`.

- [ ] **Step 1: Failing tests:** CPU: `test_missing_weight_message` (message contains the fetch command); `test_wrapper_shapes` using a tiny spandrel-compatible model saved in a tmp dir (or monkeypatched loader returning a torch `nn.Upsample(scale_factor=4)`), checks output shape and dtype. GPU (`@pytest.mark.gpu`): `test_realesrgan_runs` on a 64×64 image → 256×256 output, values in [0,1], runtime printed.
- [ ] **Step 2–4:** FAIL → implement → PASS (CPU here; GPU tests on laptop).
- [ ] **Step 5:** Commit `feat: spandrel model loading with device and fp16 handling`.

---

### Task 4: Enhance stage

**Files:** Create `engine/stages/enhance.py`; Modify `engine/presets.py`, `engine/pipeline.py` (merge the enhance report dict into that stage's `StageReport.params`), `engine/types.py` + `engine/loader.py` (`RasterImage.source_format`); Test `tests/unit/test_enhance.py`, `tests/gpu/test_enhance_gpu.py`

**Interfaces:**
- Consumes: Tasks 1–3, `RasterImage`, `StageReport`
- Produces:
  - `EnhanceParams(enabled: bool = True, scale: int = 4, sr_model_flat: str = "realesrgan_x4plus_anime_6b", sr_model_general: str = "realesrgan_x4plus", deblock: bool = True, denoise: bool = False, tile: int = 512, device: Literal["auto","cuda","cpu"] = "auto", max_output_side: int = 8192)` — validated like the other params (types, ranges)
  - `enhance(image: RasterImage, params: EnhanceParams) -> tuple[RasterImage, dict[str, object]]` — second value goes into the stage report (`model_used`, `fallback_reason`, `final_tile`, `effective_scale`, `device`)
  - Model choice: `sr_model_flat` when the image has ≤ 64 unique colors after 5-bit-per-channel quantization, else `sr_model_general` (stage 5.1 Analyze replaces this heuristic in M6)
  - Order: deblock (only if source format was JPEG — pass a `source_format` field; add `source_format: str | None = None` to `RasterImage`, set by the loader) → denoise (if enabled) → SR
  - Alpha: upscale alpha with bicubic, re-threshold at 128; RGB of fully transparent pixels is ignored
  - Effective scale = min(params.scale, floor(max_output_side / long_side)), at least 1; scale 1 → return input unchanged with report note
  - Fallback: ML unavailable, weights missing, or any model exception → bicubic resize by effective scale, `fallback_reason` set
  - Pipeline: `"enhance"` added to `STAGES`; `PRESETS["logo"].stages` becomes `("enhance","segment","trace","fit","export")`; `"enhance.enabled": False` override restores M3 behavior exactly

- [ ] **Step 1: Failing CPU tests** (monkeypatch `load_model` to a nearest-upscale fake):
  - `test_scale_and_original_size`: 64×48 input → output pixels 256×192, `scale == 4.0`, `original_size == (64, 48)`.
  - `test_alpha_preserved`: `transparent_logo()` → alpha agreement with nearest-upscaled source alpha ≥ 0.98.
  - `test_tiny_image_bicubic`: 8×8 → bicubic path, report `fallback_reason == "too small"`.
  - `test_large_image_capped`: 4096×100 → effective scale 2, output long side 8192.
  - `test_fallback_when_ml_unavailable`: `ml_available` patched False → bicubic, pipeline succeeds.
  - `test_fallback_on_model_exception`.
  - `test_enhance_disabled_is_m3_identical`: with `"enhance.enabled": False`, SVG output equals the pre-M4 output byte-for-byte on `flat_shapes(0)`.
  - `test_jpeg_only_deblock`: deblock invoked only for JPEG sources (spy).
  - `test_pipeline_end_to_end_cpu_fake`: `vectorize(png)` with fake model → doc width/height = original size, SSIM vs source ≥ 0.95.
- [ ] **Step 2: GPU tests** (`tests/gpu/`, `@pytest.mark.gpu`): real models on `flat_shapes(0..4)` downscaled 2× with Lanczos and saved as JPEG q75 ("degraded logos"); `vectorize` with enhance on vs off → mean SSIM (vs the clean full-size source rendered at the degraded image's size) improves, and mean nodes do not increase by more than 25%. Print a table.
- [ ] **Step 3:** Run → FAIL. **Step 4:** Implement. **Step 5:** Run CPU suite + coverage gate → PASS; run `pytest -m gpu` on laptop → PASS.
- [ ] **Step 6:** Commit `feat: enhance stage with ML upscale, deblock, denoise and fallbacks`.

---

### Task 5: Calibrate sample-count thresholds at scale 4

Carried from the M1–M3 final review: sample-count thresholds (`CORNER_WINDOW`, `CORNER_TRIM`, `MIN_SIDE_POINTS`, `MIN_LOOP_POINTS`, segment 3×3 cleanup window) are not scaled, so at 4× corner detection becomes 4× more local. Also `FitParams.complexity_weight` is not scale-converted: SSE grows ~scale³ (scale× more samples, scale² larger squared distances), so the MDL penalty effectively shrinks — calibrate it here too (candidate: `w * scale**3`). Also validate `RasterImage.scale` (finite, > 0) in `__post_init__`.

**Files:** Modify `engine/stages/fit_geometry.py`, `engine/geometry/primitives.py`, `engine/stages/segment.py` as measured; Test extend `tests/golden/test_golden.py` + `tests/gpu/`

- [ ] **Step 1: Write failing golden** (CPU, deterministic): for `flat_shapes(0..9)` and `circle()`, build `RasterImage(scale=4)` by nearest-upscaling the source 4× and run `run_pipeline` with enhance disabled → render vs scale-1 render SSIM ≥ 0.98, node count ≤ 1.3× the scale-1 node count, `circle()` still emits `<circle>`.
- [ ] **Step 2:** Run → record which assertions fail and by how much.
- [ ] **Step 3:** For each failing threshold, make it scale-aware (`round(value * scale)` for window-like counts, or convert to a length in original px), one at a time, re-running the golden after each; keep a table of before/after in the report.
- [ ] **Step 4:** Run full suite → PASS (M1–M3 goldens untouched).
- [ ] **Step 5:** Commit `fix: scale sample-count thresholds with image scale`.

---

### Task 6: Degraded benchmark + docs

**Files:** Modify `eval/make_synthetic_dataset.py` (add `degraded/` category: the logos downscaled 2× Lanczos + JPEG q75), `eval/README.md`

- [ ] **Step 1:** Regenerate synthetic dataset; run `python -m eval.run_benchmark --dataset eval/dataset_synthetic --out eval/reports/m4 --engines ours,vtracer` on the laptop GPU.
- [ ] **Step 2:** Record per-category numbers (logos + degraded) and GPU runtime per image in `eval/README.md` under "M4 baseline (date)". Honest numbers, no tuning in this task. If `ours` does not beat vtracer on degraded SSIM, report it as a concern.
- [ ] **Step 3:** If real images exist in `eval/dataset/`, run them too and record.
- [ ] **Step 4:** Commit `docs: M4 enhance benchmark baseline`.

---

### Task 7: Re-evaluate parked fix 2 (noise-adaptive fitting) on enhanced input

Branch `fix/noise-adaptive-fit` (head 3b9b354, rebased on master 2f293c9 with fixes 1+3; 502 tests pass; `FitParams.adaptive=True`, `False` = master's fit byte-identical) was parked on 2026-11-02. On 40 degraded images at 1× it gave nodes −8.0% summed (−11.3% mean per image), SSIM +0.0004, ΔE −0.005, kinks 831→157, fit time +13%; clean 1.jpg −8.7% nodes, SSIM −0.0003, fit +19%. Keep rule (≥10% fewer nodes summed, SSIM not lower, ΔE not higher) missed. Its constants were tuned for raw 1× JPEG edges.

- [ ] **Step 1:** Rebase `fix/noise-adaptive-fit` onto master after Tasks 1–6; full suite green.
- [ ] **Step 2:** Run the fix 2 evaluation (degraded set built as in `.superpowers/fix2-noise/brief.md` §real-image check + the degradations listed in its eval notes: JPEG q50/q30, small web, scan, phone) with enhance ON, adaptive off vs on. Report nodes (summed), SSIM, ΔE, fit time, kinks, inter-curve crossings, lost ≥90° corners.
- [ ] **Step 3:** Keep rule as above. If met → independent review, then merge. If not → delete the branch and record why in the rulings note.
