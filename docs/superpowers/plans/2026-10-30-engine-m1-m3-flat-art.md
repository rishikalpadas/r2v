# raster2vector Engine M1–M3 (Harness + Flat-Art Pipeline + Geometry) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the eval harness and a CPU-only flat-art vectorizer (segment → shared-boundary trace → geometry fitting → SVG export) behind the `vectorize()` API, with benchmark numbers against VTracer/Potrace.

**Architecture:** Immutable `PipelineState` flows through pure stage functions selected by a `Preset`. Region boundaries are traced once into a shared `BoundaryGraph`, so every fitted curve is used by both neighboring regions and gaps are impossible by construction. The eval harness renders any SVG with resvg and scores it against the source raster.

**Tech Stack:** Python ≥3.11 (dev machine has 3.14), numpy, scipy, scikit-image, scikit-learn, Pillow, resvg-py, svgelements, vtracer (baseline only), pytest, pytest-cov, hypothesis.

**Spec:** `docs/superpowers/specs/2026-10-30-raster2vector-engine-design.md`

**Covers spec milestones 1–3.** Out of this plan, in later plans: Enhance (5.2), Refine (5.6), Analyze classifier (5.1), photo/line-art segmentation, gradient fills, GPU OOM tiling. Symmetry snapping (5.5) is also deferred to the Refine plan, where it can be a loss term; this plan does axis snapping and G1 continuity only.

## Global Constraints

- Accepted input formats: PNG, JPEG, WebP, BMP, TIFF, GIF (first frame only).
- Max input 50 MB (`50 * 1024 * 1024` bytes); max long side 8192 px ("8K").
- Output format: SVG only.
- Potrace (GPL) is invoked only as an external CLI from `eval/`, never imported by `engine/`. `engine/` must not import `vtracer` either.
- All engine data types are frozen dataclasses; stages return new objects (`dataclasses.replace`), never mutate inputs.
- Engine raises only `InputError` (bad input, user-safe message) or `StageError(stage, reason)`.
- Coverage ≥ 80% on `engine/` (`--cov=engine --cov-fail-under=80`).
- Coordinates are in original-image pixel units, origin top-left, pixel `(x, y)` covers `[x, x+1] × [y, y+1]`.
- Label constants: `TRANSPARENT = -1` (alpha < 128), `OUTSIDE = -2` (beyond canvas). Neither is ever emitted as a shape.

## Review Focus

1. **Transparent PNG input** → transparent areas emit no shapes; rendered alpha matches source alpha (≥ 0.98 agreement). Test in Task 5 and golden in Task 7.
2. **Single-color or blank image** (fewer unique colors than `max_colors`) → valid SVG with one shape (or zero if fully transparent), no KMeans crash. Test in Task 5 and Task 7.
3. **Tiny images (1×1, 2×2) and 1-px-wide lines** → valid SVG, line region survives min-area cleanup when it is the only pixels of its color. Test in Task 5 and Task 7.
4. **Non-RGB modes** (L, LA, P with transparency, CMYK, 16-bit I;16) → loaded as correct RGBA. Test in Task 2.
5. **Nested holes and border-touching regions** (ring in ring, shape touching edge) → holes preserved in cutout mode, border curves use `OUTSIDE` neighbor. Test in Task 6 and Task 7.

---

## File Structure

```
pyproject.toml                 deps, pytest + coverage config
engine/__init__.py             re-exports vectorize, to_svg
engine/types.py                all data types (Task 1, extended in Tasks 6–7)
engine/errors.py               InputError, StageError
engine/loader.py               bytes → RasterImage with validation
engine/presets.py              Preset, params dataclasses, apply_overrides
engine/pipeline.py             PipelineState, vectorize()
engine/stages/segment.py       flat-art color segmentation
engine/stages/trace.py         stage wrapper over geometry/stitching
engine/stages/fit_geometry.py  stage wrapper over geometry/*
engine/stages/export.py        build_doc(), to_svg()
engine/geometry/stitching.py   crack-edge boundary graph
engine/geometry/corners.py     smoothing + corner detection
engine/geometry/segment_fit.py line/arc fit + MDL choice + axis snap
engine/geometry/bezier.py      Schneider cubic fitting
engine/geometry/primitives.py  circle/ellipse/rect/polygon detection
eval/synthetic.py              deterministic test-image generator
eval/metrics.py                render + SSIM/ΔE/edge-F1/node count
eval/baselines.py              ours / vtracer / potrace / reference runners
eval/run_benchmark.py          CLI → results.csv
eval/report.py                 results.csv → report.html
tests/unit/...                 one test file per module
tests/golden/test_golden.py    end-to-end thresholds
```

---

### Task 1: Project scaffold, types, errors

**Files:**
- Create: `pyproject.toml`, `engine/__init__.py`, `engine/types.py`, `engine/errors.py`, `engine/stages/__init__.py`, `engine/geometry/__init__.py`, `eval/__init__.py`, `.gitignore`
- Test: `tests/unit/test_types.py`

**Interfaces — Produces** (all `@dataclass(frozen=True)`; `Point = tuple[float, float]`):
- `RasterImage(pixels: np.ndarray /*HxWx4 uint8*/, original_size: tuple[int,int] /*(w,h)*/, scale: float = 1.0, color_space: str = "sRGB")`
- `ImageProfile(kind: Literal["logo","line_art","photo","mixed"], color_count: int, noise_level: float, preset: str)`
- `RegionMap(labels: np.ndarray /*HxW int32*/, palette_rgb: np.ndarray /*Kx3 uint8*/, palette_lab: np.ndarray /*Kx3 float64*/, adjacency: frozenset[tuple[int,int]], gradient_regions: frozenset[int] = frozenset())`
- Segments: `Line(p0, p1)`, `Arc(p0, p1, center: Point, radius: float, large_arc: bool, sweep: bool)`, `CubicBezier(p0, c1, c2, p1)`; `Segment = Line | Arc | CubicBezier`
- `SolidFill(rgb: tuple[int,int,int])`; `Fill = SolidFill`
- `Path(contours: tuple[tuple[Segment, ...], ...], fill: Fill, z_order: int)` — `contours[0]` outer, rest holes
- Primitive geometry: `Circle(cx, cy, r)`, `Ellipse(cx, cy, rx, ry, angle_deg)`, `Rect(x, y, w, h, angle_deg)`, `Polygon(points: tuple[Point, ...])`; `PrimitiveShape = Circle | Ellipse | Rect | Polygon`
- `Primitive(shape: PrimitiveShape, fill: Fill, z_order: int)`
- `VectorDoc(width: int, height: int, paths: tuple[Path, ...], primitives: tuple[Primitive, ...], layer_mode: Literal["stacked","cutout"])`
- `StageReport(stage: str, duration_s: float, params: Mapping[str, object], debug_images: Mapping[str, np.ndarray])`
- `VectorizeResult(doc: VectorDoc, reports: tuple[StageReport, ...])`
- Constants `TRANSPARENT = -1`, `OUTSIDE = -2`
- `errors.InputError(Exception)`; `errors.StageError(Exception)` with attributes `stage: str`, `reason: str`, `str()` = `"{stage}: {reason}"`

- [ ] **Step 1:** Write `pyproject.toml`: project `raster2vector`, `requires-python = ">=3.11"`, deps `numpy scipy scikit-image scikit-learn pillow resvg-py svgelements`, optional `[dev]`: `pytest pytest-cov hypothesis vtracer`. Pytest `testpaths = ["tests"]`. `.gitignore`: `.venv/ __pycache__/ eval/reports/ eval/dataset/ .coverage`. Create venv and install: `python -m venv .venv && .venv/Scripts/python -m pip install -e .[dev]`. Expected: install succeeds. If a dep lacks a 3.14 wheel, recreate the venv with `py -3.12`.
- [ ] **Step 2: Write failing tests** `test_types.py`:
  - `test_dataclasses_are_frozen`: assigning `Line((0,0),(1,1)).p0 = (2,2)` raises `dataclasses.FrozenInstanceError`.
  - `test_stage_error_message`: `str(StageError("segment", "boom")) == "segment: boom"` and `.stage == "segment"`.
  - `test_label_constants`: `TRANSPARENT == -1 and OUTSIDE == -2`.
- [ ] **Step 3:** Run `.venv/Scripts/python -m pytest tests/unit/test_types.py -v` → FAIL (import error).
- [ ] **Step 4:** Implement `engine/types.py`, `engine/errors.py` per Interfaces.
- [ ] **Step 5:** Run again → PASS.
- [ ] **Step 6:** Commit `feat: scaffold project with core types and errors`.

---

### Task 2: Loader with validation

**Files:** Create `engine/loader.py`; Test `tests/unit/test_loader.py`

**Interfaces:**
- Consumes: `RasterImage`, `InputError`
- Produces: `load_image(data: bytes) -> RasterImage`; constants `MAX_BYTES = 50 * 1024 * 1024`, `MAX_SIDE = 8192`, `ALLOWED_FORMATS = frozenset({"PNG","JPEG","WEBP","BMP","TIFF","GIF"})`

- [ ] **Step 1: Write failing tests** (build inputs in-memory with Pillow):
  - `test_rejects_oversize_bytes`: `b"\0" * (MAX_BYTES + 1)` → `InputError` with message containing `"50 MB"`. Check size before decoding.
  - `test_rejects_long_side_over_8192`: 8193×1 PNG → `InputError`.
  - `test_rejects_undecodable`: `b"not an image"` → `InputError`.
  - `test_rejects_unsupported_format`: ICO bytes → `InputError`.
  - `test_gif_uses_first_frame`: 2-frame GIF, red then blue → pixel `(0,0)` RGB is red.
  - `test_mode_conversion` (parametrized over `L`, `LA`, `P` with transparency index, `CMYK`, `I;16`): result `pixels.shape == (h, w, 4)`, `dtype == uint8`; `P` transparent index → alpha 0; `I;16` value 65535 → 255.
  - `test_original_size_and_scale`: 30×20 PNG → `original_size == (30, 20)`, `scale == 1.0`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement `load_image`. Use `Image.open(BytesIO)`, `img.verify()` on a first handle then reopen; check `img.format in ALLOWED_FORMATS`; `img.seek(0)`; `I;16` → scale by `>> 8` before converting to `L`; then `.convert("RGBA")`. Wrap every Pillow exception in `InputError("could not decode image")` (no internal detail in message).
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat: add validated image loader`.

---

### Task 3: Synthetic images + eval metrics

**Files:** Create `eval/synthetic.py`, `eval/metrics.py`; Test `tests/unit/test_metrics.py`

**Interfaces:**
- Produces `eval/synthetic.py` (all deterministic, no anti-aliasing, return `np.ndarray` HxWx4 uint8):
  - `flat_shapes(seed: int, size: int = 128, n_colors: int = 4) -> np.ndarray` — background + random rects/ellipses/triangles in `n_colors` colors
  - `circle(size: int = 128, cx: float = 64, cy: float = 64, r: float = 40) -> np.ndarray`
  - `ring_in_ring(size: int = 128) -> np.ndarray` — 3 concentric colored rings on background
  - `transparent_logo(size: int = 128) -> np.ndarray` — opaque shape on alpha-0 background
  - `solid(size: int, rgba: tuple[int,int,int,int]) -> np.ndarray`
  - `to_png_bytes(pixels: np.ndarray) -> bytes`
- Produces `eval/metrics.py`:
  - `render_svg(svg: str, width: int, height: int) -> np.ndarray` (HxWx4 uint8, via resvg-py; confirm the call name against the installed resvg-py version, e.g. `resvg_py.svg_to_bytes`, and decode the PNG with Pillow)
  - `ssim(a: np.ndarray, b: np.ndarray) -> float` — both RGBA composited over white, `skimage.metrics.structural_similarity(channel_axis=2, data_range=255)`
  - `mean_delta_e(a, b) -> float` — composite over white, `skimage.color.rgb2lab` + `deltaE_ciede2000`, mean
  - `edge_f1(a, b, tolerance_px: int = 2) -> float` — Canny (`sigma=1`) on grayscale of each; precision/recall by matching within tolerance using `scipy.ndimage.distance_transform_edt`; returns 1.0 when both edge maps are empty
  - `count_nodes(svg: str) -> tuple[int, int]` — `(node_count, path_count)` via `svgelements.SVG.parse`; node = each path segment except `Move`/`Close`; `<circle>/<ellipse>/<rect>` count 1 node each; `<polygon>` counts its points; path_count = number of shape elements
  - `alpha_agreement(a, b) -> float` — fraction of pixels where `(alpha >= 128)` matches

- [ ] **Step 1: Write failing tests:**
  - `test_ssim_identical_is_one`: `ssim(x, x) == pytest.approx(1.0)`.
  - `test_delta_e_identical_is_zero`.
  - `test_edge_f1_identical_is_one` and `test_edge_f1_both_empty_is_one` (two `solid` images).
  - `test_render_svg_rect`: `<svg width="10" height="10"><rect width="10" height="10" fill="#ff0000"/></svg>` → every pixel `(255,0,0,255)`.
  - `test_count_nodes`: SVG with one triangle path `M0 0 L10 0 L5 8 Z` + one `<circle>` → `(4, 2)` (3 lines + 1 circle).
  - `test_synthetic_deterministic`: `flat_shapes(1)` equals `flat_shapes(1)`; unique colors in `flat_shapes(1, n_colors=4)` ≤ 4.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement both modules.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat: add synthetic images and eval metrics`.

---

### Task 4: Baselines, benchmark runner, HTML report

**Files:** Create `eval/baselines.py`, `eval/run_benchmark.py`, `eval/report.py`; Test `tests/unit/test_benchmark.py`

**Interfaces:**
- Consumes: Task 3 metrics. `ours` runner imports `engine.vectorize`/`to_svg` lazily inside the function (exists from Task 7; before that the runner raises `NotImplementedError`, which the benchmark records as skipped).
- Produces:
  - `Runner = Callable[[Path], str | None]` (image path → SVG text, `None` = skipped)
  - `RUNNERS: dict[str, Runner]` keys `"ours"`, `"vtracer"`, `"potrace"`, `"reference"`
    - `vtracer`: `vtracer.convert_raw_image_to_svg(bytes, img_format=...)`, default color mode
    - `potrace`: skip (`None`) if `shutil.which("potrace")` is None; else grayscale threshold 128 → PBM in a temp dir → `subprocess.run(["potrace", "-s", "-o", out, in], timeout=60, check=True)`
    - `reference`: read `<category>/reference/<stem>.svg` if it exists, else `None`
  - Dataset layout: `eval/dataset/<category>/<name>.{png,jpg,...}` with optional `eval/dataset/<category>/reference/<name>.svg`; categories `logos, line_art, scans, photos, degraded`.
  - `run_benchmark(dataset: Path, out: Path, engines: list[str]) -> Path` writes `out/results.csv` (columns `category,image,engine,status,ssim,delta_e,edge_f1,nodes,paths,runtime_s`; `status` ∈ `ok|skipped|error`), saves renders to `out/renders/<engine>/<category>/<name>.png`, returns CSV path. One image/engine failure is recorded as `error` and the run continues.
  - CLI: `python -m eval.run_benchmark --dataset eval/dataset --out eval/reports/<name> --engines ours,vtracer,potrace,reference`
  - `write_report(results_csv: Path) -> Path` → `report.html` next to the CSV: per-category table of mean metrics per engine (ok rows only, with counts), then per-image rows of original + each engine render. All file names and text go through `html.escape`.

- [ ] **Step 1: Write failing tests** (temp dataset built from `eval.synthetic`, engines `["vtracer", "reference"]`, plus a fake runner injected via `monkeypatch.setitem(RUNNERS, "boom", lambda p: 1/0)`):
  - `test_results_csv_columns_and_rows`: 2 images × 3 engines → 6 rows, header exact.
  - `test_failing_runner_recorded_as_error`: `boom` rows have `status == "error"`, others `ok`.
  - `test_missing_reference_is_skipped`.
  - `test_report_escapes_names`: image named `a<b>.png` → HTML contains `a&lt;b&gt;` and not `<b>`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement the three modules.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat: add benchmark runner, baselines and HTML report`.

---

### Task 5: Segment stage (flat art)

**Files:** Create `engine/presets.py` (params only, preset table comes in Task 7), `engine/stages/segment.py`; Test `tests/unit/test_segment.py`

**Interfaces:**
- Consumes: `RasterImage`, `RegionMap`, `TRANSPARENT`
- Produces:
  - `SegmentParams(max_colors: int = 16, merge_delta_e: float = 8.0, min_region_px: int = 4, sample_px: int = 200_000, seed: int = 0)` (frozen, in `presets.py`)
  - `segment(image: RasterImage, params: SegmentParams) -> RegionMap`

**Algorithm (decisions the signature does not fix):**
1. Opaque mask = alpha ≥ 128; transparent pixels get label `TRANSPARENT`.
2. Unique opaque colors ≤ `max_colors` → each unique color is a cluster (no KMeans). Else KMeans in Lab with `k = max_colors`, `n_init=4`, `random_state=seed`, fit on ≤ `sample_px` random opaque pixels, predict all.
3. Merge clusters whose centroid ΔE2000 < `merge_delta_e` (union-find, palette color = pixel-weighted mean).
4. Anti-alias cleanup: one pass of 3×3 majority filter over opaque labels, applied only to pixels whose 4-neighbors are not all the same label.
5. Connected components per label (4-connectivity); each component becomes its own region id. Components smaller than `min_region_px` are merged into the neighbor region sharing the longest border — unless the component is the only component of its color, which is kept (thin 1-px lines must survive).
6. `adjacency` = unordered pairs of region ids sharing a 4-neighbor edge.

- [ ] **Step 1: Write failing tests:**
  - `test_exact_colors_recovered`: `flat_shapes(3, n_colors=4)` → for every region, its palette RGB equals the source color of all its pixels.
  - `test_single_color_image`: `solid(16, (10,20,30,255))` → one region, no exception.
  - `test_fully_transparent`: `solid(16, (0,0,0,0))` → all labels `TRANSPARENT`, zero regions.
  - `test_transparent_background`: `transparent_logo()` → labels are `TRANSPARENT` exactly where alpha < 128.
  - `test_tiny_images` (parametrized 1×1, 2×2) → valid `RegionMap`, shape matches.
  - `test_one_px_line_survives`: 32×32 white with a 1-px black horizontal line → a black region exists covering the line pixels.
  - `test_jpeg_noise_merged`: `flat_shapes(5, n_colors=3)` round-tripped through JPEG quality 85 → number of distinct palette colors ≤ 4.
  - `test_does_not_mutate_input`: input `pixels` unchanged after call.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat: add flat-art segmentation stage`.

---

### Task 6: Shared-boundary trace

**Files:** Create `engine/geometry/stitching.py`, `engine/stages/trace.py`; Test `tests/unit/test_stitching.py`

**Interfaces:**
- Consumes: `RegionMap`, `OUTSIDE`, `TRANSPARENT`
- Produces (add to `engine/types.py`):
  - `BoundaryCurve(id: int, points: tuple[Point, ...], left: int, right: int, closed: bool)` — polyline on pixel-corner grid; `left`/`right` = region ids (or `TRANSPARENT`/`OUTSIDE`) on each side when walking `points` in order
  - `LoopRef = tuple[tuple[int, bool], ...]` — `(curve_id, reversed)` in walking order
  - `BoundaryGraph(curves: tuple[BoundaryCurve, ...], region_loops: Mapping[int, tuple[LoopRef, ...]])` — per region, `loops[0]` is the outer boundary (largest absolute signed area), others are holes
  - `build_boundary_graph(regions: RegionMap) -> BoundaryGraph` in `stitching.py`
  - `trace(regions: RegionMap) -> BoundaryGraph` in `stages/trace.py` (wraps, converts unexpected exceptions to `StageError("trace", ...)`)

**Algorithm:**
1. Pad `labels` with `OUTSIDE` by one pixel. Crack edges = every unit edge between two pixels with different labels (vectorized with numpy over horizontal and vertical neighbor pairs).
2. Grid corner is a **junction** if ≥ 3 distinct labels touch it, or 2 labels touch it in a diagonal (checkerboard) pattern.
3. Walk chains of crack edges between junctions; each chain with a constant `(left, right)` pair is one `BoundaryCurve`. Chains with no junction form `closed=True` curves.
4. Drop curves where both sides are non-region (`TRANSPARENT`/`OUTSIDE`).
5. For each region, assemble loops by following its curves end to end (reversing as needed so the region stays on the left); classify outer vs holes by signed area.

- [ ] **Step 1: Write failing tests:**
  - `test_every_curve_shared`: on `flat_shapes(7)` each curve id appears in loops of exactly the regions named by its `left` and `right` (regions only), and in no other region's loops.
  - `test_loops_close`: for every loop, end point of each curve equals start of the next (respecting `reversed`), last connects to first.
  - `test_ring_in_ring_holes`: `ring_in_ring()` → the middle ring region has 2 loops (outer + 1 hole).
  - `test_border_region_uses_outside`: `solid(8, opaque)` → one closed curve, `right == OUTSIDE` or `left == OUTSIDE`, points form the 8×8 square perimeter.
  - `test_checkerboard_junction`: 2×2 checkerboard of two colors → builds without error, every loop closes.
  - `test_area_matches_pixels`: for each region, `|signed area of outer| - sum(|holes|)` equals its pixel count.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat: trace shared region boundaries into a boundary graph`.

---

### Task 7: Export, presets, pipeline, `vectorize()` — flat art end to end (Milestone 2)

**Files:** Create `engine/stages/export.py`, `engine/pipeline.py`; Modify `engine/presets.py`, `engine/__init__.py`; Test `tests/unit/test_export.py`, `tests/unit/test_pipeline.py`, `tests/golden/test_golden.py`

**Interfaces:**
- Consumes: Tasks 1–6.
- Produces:
  - Add `FittedCurve(curve_id: int, segments: tuple[Segment, ...], primitive: PrimitiveShape | None)` to `types.py`
  - `ExportParams(layer_mode: Literal["stacked","cutout"] = "stacked", precision: int = 2)`
  - `FitParams(enabled: bool = True, tolerance_px: float = 0.75, complexity_weight: float = 2.0, axis_snap_deg: float = 3.0, smooth_sigma: float = 1.0, corner_angle_deg: float = 35.0)` — defined here so presets are complete; consumed from Task 8 on
  - `Preset(name: str, stages: tuple[str, ...], segment: SegmentParams, fit: FitParams, export: ExportParams)`; `PRESETS: dict[str, Preset]` with `"logo"` = stages `("segment","trace","fit","export")`; `"auto"` resolves to `"logo"` in this plan
  - Stage `"fit"` dispatches to `polyline_fit(graph)` in this task; Task 11 switches it to `fit_geometry(graph, preset.fit)`
  - `apply_overrides(preset: Preset, overrides: Mapping[str, object]) -> Preset` — keys `"<group>.<field>"`, e.g. `"segment.max_colors"`; unknown key → `InputError`
  - `polyline_fit(graph: BoundaryGraph) -> tuple[FittedCurve, ...]` — each curve's points as `Line` segments with collinear runs merged (export's "drop redundant nodes")
  - `build_doc(regions: RegionMap, graph: BoundaryGraph, fitted: tuple[FittedCurve, ...], params: ExportParams) -> VectorDoc`
    - stacked: z-order by region area descending; each region emits only its outer loop
    - cutout: each region emits all loops; SVG uses `fill-rule="evenodd"`
    - region whose single loop is one curve with `primitive` set → `Primitive`, only if the region has no holes (cutout) or always (stacked)
  - `to_svg(doc: VectorDoc) -> str` — `<svg xmlns=... width height viewBox>`, numbers rounded to `precision`, `fill="#rrggbb"`; arcs as SVG `A` commands
  - `PipelineState(image: RasterImage, preset: Preset, regions: RegionMap | None = None, graph: BoundaryGraph | None = None, fitted: tuple[FittedCurve, ...] | None = None, doc: VectorDoc | None = None)`
  - `vectorize(image_bytes: bytes, preset: str = "auto", overrides: Mapping[str, object] | None = None) -> VectorizeResult` — runs `preset.stages` in order, one `StageReport` per stage; non-`InputError` exceptions inside a stage become `StageError(stage, reason)`
  - `engine/__init__.py` exports `vectorize`, `to_svg`, `InputError`, `StageError`

- [ ] **Step 1: Write failing unit tests:**
  - `test_to_svg_parses`: output parses with `xml.etree.ElementTree`; root tag ends with `svg`; `viewBox == "0 0 W H"`.
  - `test_precision_rounding`: a point `(1.23456, 2)` emits `1.23` with default params.
  - `test_cutout_uses_evenodd_and_holes`: `ring_in_ring` in cutout mode → middle-ring path has 2 `M` commands and `fill-rule="evenodd"`.
  - `test_apply_overrides`: `"segment.max_colors": 8` → new preset with 8, original unchanged; `"segment.nope"` → `InputError`.
  - `test_vectorize_invalid_bytes_raises_input_error`.
  - `test_stage_failure_wrapped`: monkeypatch `segment` to raise `ValueError` → `StageError` with `.stage == "segment"`.
  - `test_reports_per_stage`: result has one report per stage, names match `preset.stages`.
- [ ] **Step 2: Write failing golden tests** (`tests/golden/test_golden.py`, render with `eval.metrics.render_svg`):
  - `test_flat_shapes_fidelity` (seeds 0–9): `ssim ≥ 0.95`, `mean_delta_e ≤ 3.0`.
  - `test_stacked_no_gaps`: for opaque `flat_shapes(0..9)`, rendered stacked output has alpha == 255 on every pixel.
  - `test_transparent_logo_alpha`: `alpha_agreement ≥ 0.98`.
  - `test_degenerate_inputs` (1×1, 2×2, solid opaque, solid transparent, 1-px line): `vectorize` returns a doc whose SVG parses; fully transparent → zero shapes.
  - `test_property_random_rects` (hypothesis, `max_examples=25`): 1–5 random opaque axis-aligned rects on 64×64 → `ssim ≥ 0.95`.
- [ ] **Step 3:** Run both → FAIL.
- [ ] **Step 4:** Implement export, presets, pipeline.
- [ ] **Step 5:** Run `python -m pytest -v` → PASS.
- [ ] **Step 6:** Run `python -m pytest --cov=engine --cov-fail-under=80` → PASS.
- [ ] **Step 7:** Commit `feat: end-to-end flat-art vectorize pipeline`.

---

### Task 8: Smoothing and corner detection

**Files:** Create `engine/geometry/corners.py`; Test `tests/unit/test_corners.py`

**Interfaces:**
- Produces:
  - `smooth_polyline(points: tuple[Point, ...], closed: bool, sigma: float = 1.0) -> tuple[Point, ...]` — resample crack polyline at edge midpoints, Gaussian-smooth coordinates (wrap mode if closed); for open curves the two endpoints stay exactly fixed (they are shared junctions)
  - `find_corners(points: tuple[Point, ...], closed: bool, angle_deg: float = 35.0, window: int = 3) -> tuple[int, ...]` — indices where the turning angle between vectors to `±window` neighbors exceeds `angle_deg` and is a local maximum; open curves always include `0` and `len-1`

- [ ] **Step 1: Write failing tests:**
  - `test_square_has_four_corners`: crack perimeter of a 20×20 square (closed) → 4 corners, each within 1.5 px of a true corner.
  - `test_circle_has_no_corners`: perimeter of `circle(r=40)` region → 0 corners.
  - `test_open_endpoints_fixed`: smoothing an open polyline keeps `points[0]` and `points[-1]` identical.
  - `test_staircase_line_no_corners`: crack edges of a 1:3 slope line → 0 interior corners.
- [ ] **Step 2:** Run → FAIL.  **Step 3:** Implement.  **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat: add polyline smoothing and corner detection`.

---

### Task 9: Line, arc, Bézier fitting with MDL choice and axis snap

**Files:** Create `engine/geometry/segment_fit.py`, `engine/geometry/bezier.py`; Test `tests/unit/test_segment_fit.py`, `tests/unit/test_bezier.py`

**Interfaces:**
- Consumes: `Line`, `Arc`, `CubicBezier`, Task 8 outputs, `FitParams` (Task 7)
- Produces:
  - `bezier.fit_cubic(points: Sequence[Point], tolerance_px: float) -> tuple[CubicBezier, ...]` — Schneider (Graphics Gems 1990): chord-length parameterization, least-squares control points with end tangents, Newton–Raphson reparameterization (max 4 iterations), split at max-error point recursively; interior joins are G1 (shared tangent)
  - `segment_fit.fit_line(points) -> tuple[Line, float]`, `fit_arc(points) -> tuple[Arc, float] | None` (Kåsa circle fit; `None` if radius > 10× chord length), both returning `(segment, sum_squared_error)`; segment endpoints are exactly `points[0]` and `points[-1]`
  - `segment_fit.fit_span(points, params: FitParams) -> tuple[Segment, ...]` — candidates line (4 params), arc (7), Bézier chain (8 per segment); choose minimum `sse + complexity_weight * n_params` among candidates whose max error ≤ `tolerance_px`; Bézier chain always qualifies
  - `segment_fit.snap_axis(segments, params) -> tuple[Segment, ...]` — a `Line` within `axis_snap_deg` of horizontal/vertical whose endpoints are both interior corners (not curve endpoints) gets its endpoints moved to the shared average coordinate; the adjoining segments' endpoints move with it

- [ ] **Step 1: Write failing tests:**
  - `test_fit_line_exact`: collinear points → `Line`, sse ≈ 0.
  - `test_fit_arc_recovers_circle`: 30 points on a quarter circle r=20 → center within 0.1, radius within 0.1.
  - `test_fit_cubic_within_tolerance`: points sampled from a known S-curve → every point within 0.75 px of the result; ≤ 2 segments.
  - `test_fit_cubic_g1_joins`: forced split (tight tolerance on a long wiggle) → at each join, tangent directions differ by < 1°.
  - `test_mdl_prefers_line`: noisy-but-straight points (±0.2 px) → `fit_span` returns a single `Line`.
  - `test_endpoints_preserved`: for each candidate type, first/last points equal input endpoints exactly.
  - `test_axis_snap`: near-horizontal line at 2° between two interior corners → both endpoints share y after snap; neighbors still connected.
- [ ] **Step 2:** Run → FAIL.  **Step 3:** Implement.  **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat: fit lines, arcs and bezier chains with MDL selection`.

---

### Task 10: Whole-shape primitive detection

**Files:** Create `engine/geometry/primitives.py`; Test `tests/unit/test_primitives.py`

**Interfaces:**
- Consumes: smoothed closed polyline (Task 8), corner indices
- Produces: `detect_primitive(points: tuple[Point, ...], corners: tuple[int, ...]) -> PrimitiveShape | None`
  - Rect: exactly 4 corners, all interior angles within 5° of 90°, each side's max deviation from its line ≤ 1.0 px → `Rect` with `angle_deg` (0 when axis-aligned within 3°)
  - Circle: 0 corners, Kåsa fit, max radial deviation ≤ `max(1.0, 0.03 * r)`
  - Ellipse: 0 corners, `skimage.measure.EllipseModel`, max deviation ≤ 1.0 px, and not already a circle
  - Polygon: 3–12 corners, every side straight within 1.0 px
  - Checked in order rect → circle → ellipse → polygon; first match wins

- [ ] **Step 1: Write failing tests:**
  - `test_detects_circle`: region outline of `circle(cx=64, cy=64, r=40)` → `Circle` with center and radius each within 1.0 px.
  - `test_detects_rect_and_rotated_rect` (0° and 30°).
  - `test_detects_ellipse`: rx=40, ry=20 → `Ellipse`, axes within 1.0 px.
  - `test_detects_triangle_polygon`: 3 points within 1.5 px of true vertices.
  - `test_blob_returns_none`: irregular smooth blob → `None`.
- [ ] **Step 2:** Run → FAIL.  **Step 3:** Implement.  **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat: detect circle, ellipse, rect and polygon primitives`.

---

### Task 11: `fit_geometry` stage + logo preset (Milestone 3)

**Files:** Create `engine/stages/fit_geometry.py`; Modify `engine/pipeline.py` (stage `"fit"` calls `fit_geometry`); Test `tests/unit/test_fit_geometry.py`, extend `tests/golden/test_golden.py`

**Interfaces:**
- Consumes: `BoundaryGraph`, Tasks 8–10
- Produces: `fit_geometry(graph: BoundaryGraph, params: FitParams) -> tuple[FittedCurve, ...]` — per curve: smooth → corners → `fit_span` between consecutive corners → `snap_axis`; for `closed` curves also `detect_primitive`, stored in `FittedCurve.primitive` (segments still filled so a neighbor's hole can use them). Curve endpoints are never moved, so neighbors stay stitched.

- [ ] **Step 1: Write failing tests:**
  - `test_endpoints_unchanged`: for every curve, first/last fitted point equals `curve.points[0]`/`[-1]`.
  - `test_same_curve_both_sides`: `build_doc` in cutout mode on `ring_in_ring` → the hole contour of the outer ring and the outer contour of the inner ring are built from the same `FittedCurve` segments (reversed).
  - Golden `test_circle_emitted_as_primitive`: `vectorize(circle())` SVG contains a `<circle>` with `cx`,`cy` within 1.0 px of 64 and `r` within 1.0 px of 40.
  - Golden `test_fit_reduces_nodes`: on `flat_shapes(0..9)`, mean `count_nodes` with the logo preset ≤ 50% of the same run with override `{"fit.enabled": False}` (when disabled, `fit_geometry` returns `polyline_fit(graph)`).
  - Golden: all Task 7 golden tests still pass with the fitted pipeline.
- [ ] **Step 2:** Run → FAIL.  **Step 3:** Implement.  **Step 4:** Run full suite + coverage gate → PASS.
- [ ] **Step 5:** Commit `feat: geometry fitting stage in logo preset`.

---

### Task 12: Baseline benchmark run

**Files:** Create `eval/README.md` (dataset layout, licensing rule, how to run), `eval/make_synthetic_dataset.py`; add `eval/dataset_synthetic/` to `.gitignore`; no engine code.

- [ ] **Step 1:** Run benchmark on a synthetic dataset to prove the flow: write `eval/make_synthetic_dataset.py` producing `eval/dataset_synthetic/logos/*.png` (20 `flat_shapes` + 5 `circle`/`ring_in_ring`/`transparent_logo`), then `python -m eval.run_benchmark --dataset eval/dataset_synthetic --out eval/reports/synthetic --engines ours,vtracer,potrace` → `results.csv` and `report.html` exist; `ours` rows all `ok`.
- [ ] **Step 2:** Check: on synthetic logos, `ours` mean SSIM ≥ vtracer mean SSIM and `ours` mean nodes < vtracer mean nodes. If not, record numbers in `eval/README.md` and stop for review rather than tuning silently.
- [ ] **Step 3: User action:** place licensed real images in `eval/dataset/<category>/` (target ~200 per category per spec §7) and Vectorizer.AI outputs in `reference/`. Real-image benchmark gets run once this lands; the result is the baseline for the next plan (Enhance + Refine).
- [ ] **Step 4:** Commit `docs: benchmark usage and synthetic baseline`.
