# raster2vector — Vectorization Engine + Eval Harness Design

- **Date:** 2026-10-30
- **Status:** Approved in brainstorming, pending written-spec review
- **Scope:** Sub-project 1 (engine) + sub-project 2 (eval harness). SaaS platform (API, UI, auth, billing) is a separate later spec.

## 1. Goal

Convert raster images (PNG, JPG, WebP, BMP, TIFF, GIF) into high-quality SVG vectors, at a quality level comparable to Vectorizer.AI, for use as the core of a commercial SaaS product.

### Requirements (stated by user)

- Input mix: logos/icons/flat art, line art/scans/sketches, photos/complex illustrations.
- Commercial SaaS product.
- Quality target: Vectorizer.AI level.
- Best possible results; compute cost and GPU usage are not constraints.

### Assumptions (open to correction)

- Primary output format: SVG. PDF/EPS/DXF are later additions.
- Inputs up to 8K px on the long side and 50 MB.
- Processing runs on a CUDA GPU server. Latency target is not a primary constraint; quality wins over speed.
- Engine is Python + PyTorch.

### Success criteria

- On the benchmark set, pipeline beats Potrace and raw VTracer on every category for SSIM and mean ΔE2000.
- On logos and flat art, pipeline matches or beats collected Vectorizer.AI reference outputs on SSIM and edge F-score with equal or lower node count.
- Adjacent regions share exact boundaries: zero gaps/overlaps in output.
- Geometric primitives (circle, ellipse, rect) in input logos are emitted as true SVG primitives.

### Licensing constraints

- Prefer permissive (MIT/BSD/Apache) dependencies. VTracer (MIT) is acceptable as a reference/base tracer.
- Potrace (GPL) is used only as an eval baseline, never linked into the engine.
- ML model weights must have licenses permitting commercial use; verify each before adoption.

## 2. Approach

Hybrid pipeline combining two ideas:

1. **Structure first (classical + ML):** enhance, segment into regions, trace shared boundaries, fit clean geometry (corners, primitives, Béziers).
2. **Fidelity second (differentiable rendering):** refine control points and fills by gradient descent against the original image, with regularizers that preserve the clean structure from step 1.

Rejected alternatives:

- **Classical only:** caps quality below target; no sub-pixel fidelity refinement.
- **Differentiable only (DiffVG from scratch):** messy topology, unstable on complex images.
- **End-to-end neural (image → SVG tokens):** research-grade, unreliable on complex images.

## 3. Architecture

```
image ─► 1 Analyze ─► 2 Enhance ─► 3 Segment ─► 4 Trace ─► 5 Fit geometry ─► 6 Refine ─► 7 Export ─► SVG
                                                                                   │
                         Eval harness: render SVG (resvg) → compare to original ◄──┘
```

Every stage is a pure function `StageInput → StageOutput` over immutable typed data. Stages are swappable; the harness can score the pipeline with any stage replaced, which is how improvements are validated.

Preset selection: stage 1 picks a preset (stage configuration) per image type. Callers can override any preset parameter.

## 4. Data model

All types are immutable dataclasses.

| Type | Fields |
|---|---|
| `RasterImage` | RGBA tensor, original size, current scale factor, color space |
| `ImageProfile` | detected type (`logo`, `line_art`, `photo`, `mixed`), estimated color count, noise level, chosen preset name |
| `RegionMap` | per-pixel label mask, palette in Lab, region adjacency graph, gradient-region flags |
| `Segment` | one of `Line`, `Arc`, `CubicBezier` |
| `Path` | closed contour as list of `Segment`, fill (solid or gradient), z-order, hole references |
| `Primitive` | one of `Circle`, `Ellipse`, `Rect`, `Polygon`, with fill and z-order |
| `VectorDoc` | paths, primitives, canvas size, layer mode (`stacked` or `cutout`) |
| `StageReport` | stage name, duration, parameters used, debug images |
| `VectorizeResult` | final `VectorDoc` plus list of `StageReport` |

## 5. Stages

### 5.1 Analyze

- Small CNN classifier fine-tuned on a labeled set, combined with heuristics: unique-color histogram, edge density, noise estimate.
- Output: `ImageProfile` and preset name.

### 5.2 Enhance

- ML super-resolution 4× (Real-ESRGAN or SwinIR family; art-tuned model for flat art, general model for photos).
- JPEG deblocking and edge-preserving denoise.
- Rationale: tracing on an upscaled clean image yields sub-pixel-accurate edges in original coordinates.

### 5.3 Segment

- **Flat art:** k-means in Lab color space, palette merge by ΔE threshold, reassign anti-aliasing pixels to nearest neighbor region.
- **Photos / complex:** SAM-family masks, color quantization inside each mask, flag smooth-gradient regions for gradient fills.
- **Line art:** adaptive binarization, then skeletonize (centerline mode) or keep outlines (outline mode); preset decides.

### 5.4 Trace

- Build a boundary graph of all region edges. Trace each edge exactly once (marching squares at the enhanced resolution).
- Neighboring regions reference the same edge, guaranteeing zero gaps and overlaps.

### 5.5 Fit geometry

- Curvature-based corner detection splits edges into segments.
- For each segment fit candidates: line, arc, cubic Bézier. For each closed region also test whole-shape primitives: circle, ellipse, rect, polygon.
- Pick candidate by MDL-style score: fit error plus complexity penalty.
- Snapping: near-horizontal/vertical lines, symmetry, tangent continuity at smooth joins.

### 5.6 Refine

- Differentiable rasterizer (DiffVG or a newer Bézier-splatting-style renderer; choose by benchmark).
- Optimize control points and fill colors, multi-scale, a few hundred iterations on GPU.
- Loss: pixel L2 + SSIM + edge loss, plus regularizers for node count, curvature smoothness, deviation from fitted primitives, and shared-edge consistency.
- Acceptance: the refined result replaces the stage 5 result only if the eval score improves.

### 5.7 Export

- SVG in stacked or cutout layer mode.
- Linear/radial gradients for flagged gradient regions.
- Coordinate precision rounding and SVG cleanup (merge identical styles, drop redundant nodes).

### 5.8 Engine API

```python
result = vectorize(image_bytes, preset="auto", overrides={...})  # -> VectorizeResult
svg_text = to_svg(result.doc)
```

## 6. Error handling

- **Boundary validation:** accepted formats, max 50 MB, max 8K px long side, successful decode. Failure raises `InputError` with user-safe message.
- **Stage failure:** each stage raises `StageError(stage, reason)`.
- **ML fallbacks:**
  - Upscaler fails → bicubic upscale.
  - Segmentation model fails → color-only segmentation.
  - Refine diverges or does not improve score → keep stage 5 output.
- **Resources:** per-stage timeout; on GPU out-of-memory, retry with smaller tiles.

## 7. Eval harness

- **Dataset:** about 200 images per category: logos, line art, scans, photos, low-res/JPEG-degraded inputs. Only images with licenses permitting this use.
- **Metrics:** SSIM, mean ΔE2000, edge F-score, node count, path count, runtime. SVG rendered with resvg at original resolution.
- **Baselines:** Potrace, raw VTracer, and Vectorizer.AI reference outputs collected manually.
- **Report:** HTML table per category plus side-by-side visuals per image. Run on every pipeline change.

## 8. Testing

- **Unit:** geometry fitting on synthetic shapes (known circle/rect/Bézier must be recovered within tolerance); boundary stitching (no gaps); SVG export validity.
- **Golden:** small fixed image set with metric thresholds that must not regress.
- **Property-based:** for random synthetic flat-color images, rendered output is close to input.
- **Coverage:** 80%+ on non-ML code.

## 9. Project layout

```
raster2vector/
  engine/
    types.py
    pipeline.py
    presets.py
    errors.py
    stages/
      analyze.py  enhance.py  segment.py  trace.py
      fit_geometry.py  refine.py  export.py
    geometry/
      corners.py  primitives.py  bezier.py  stitching.py
  eval/
    dataset/
    metrics.py
    run_benchmark.py
    report.py
  tests/
    unit/
    golden/
  docs/superpowers/specs/
```

## 10. Milestones

1. Skeleton, types, eval harness, baselines — measure before building.
2. Segment + trace + export — flat art works end to end.
3. Geometry fitting — largest quality gain for logos.
4. Enhance — ML upscale and denoise.
5. Refine — differentiable rendering.
6. Analyze + presets; photo and line-art modes.
7. Separate spec: SaaS platform (API, web UI with preview, auth, billing, job queue, storage).

## 11. Out of scope for this spec

- Web UI, REST API, authentication, billing, storage, job queue.
- PDF/EPS/DXF export.
- Training custom models from scratch (fine-tuning the stage 1 classifier is in scope).
