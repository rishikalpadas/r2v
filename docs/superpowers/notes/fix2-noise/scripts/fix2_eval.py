"""Fix 2 evaluation: adaptive off vs on over degraded inputs. Run from repo root.

Two phases, so the GPU is used by one process only and both fits see the same enhanced pixels:

    python fix2_eval.py enhance            # degrade sources, run the enhance stage once per image
                                           # (one process; the GPU if available), cache the output
    python fix2_eval.py fit --enhance on   # segment/trace/fit/export the cached enhanced images,
                                           # adaptive off and on, in CPU worker processes
    python fix2_eval.py fit --enhance off  # same from the degraded images, enhance disabled

`fit` options: --adaptive off,on (which fits to run), --tag NAME (results_<enhance>_<tag>.csv,
default "base"), --workers N (default 4), --only SUBSTR (jobs whose stem contains it).
Workers get CUDA_VISIBLE_DEVICES=-1: they never touch the GPU.

`enhance` fails loudly when the ML upscaler silently fell back to bicubic (the enhance
StageReport's `model_used` / `fallback_reason`), and checks once that
segment->export on the cached image reproduces `vectorize` exactly.

Sources: `eval.synthetic` AA scenes plus, when R2V_REAL_IMAGE is set, the user's HAPPINESS
JPEG (git-ignored; never commit it), each under five degradations; HAPPINESS is also run clean
(kind "clean", not part of the degraded set). Outputs go to .superpowers/fix2-noise/eval/.

Per run (one row per source x kind x adaptive): nodes, paths, SSIM and mean Delta E of the render
against the (degraded) input at original size, fit-stage seconds, total seconds of the
non-enhance stages, median per-curve noise estimate, kinks, inter-curve crossings, the count of
>= 90 deg corners, and the enhance model that ran. The fitted curves are pickled so
`fix2_summary.py` can compare off and on fits (lost corners).

Metric definitions (all on the fitted segments in processing pixels, i.e. original px x scale):
* kinks: joins between consecutive segments of a non-primitive curve whose tangent break is
  10-30 deg (wobble-sized breaks that are neither smooth nor a real corner).
* inter-curve crossings: number of unordered pairs of different fitted curves whose fitted
  geometry intersects away from their shared endpoints. Segments are flattened to pieces of at
  most FLATTEN_PX processing px; two pieces of different curves count when they intersect
  (closed orientation test, so touching and collinear-overlapping pieces count too: crack-grid
  geometry often meets exactly at piece vertices), and the intersection point is farther than
  SHARED_END_RADIUS_PX x scale from every endpoint the two curves share (junctions are bitwise
  exact, so curves that only meet there do not count).
* >= 90 deg corners: interior joins of a curve (between consecutive segments, plus the wrap join
  of a closed chain) whose tangent turn is >= 90 deg.
* lost >= 90 deg corners (computed by fix2_summary.py): >= 90 deg corners of the adaptive-off fit
  with no segment join (any segment endpoint of any curve) of the adaptive-on fit within
  LOST_CORNER_RADIUS_PX x scale processing px, i.e. 1.5 original px.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

REPO = Path(__file__).resolve().parents[5]  # docs/superpowers/notes/fix2-noise/scripts -> repo root
sys.path.insert(0, str(REPO))
OUT = REPO / ".superpowers" / "fix2-noise" / "eval"
# Optional real test image (the user's HAPPINESS JPEG); skipped when unset.
USER_IMG = os.environ.get("R2V_REAL_IMAGE")
USER_NAME = "happiness"

FLATTEN_PX = 1.0  # processing px: max piece length when flattening segments for crossings
SHARED_END_RADIUS_PX = 1.0  # original px around a shared endpoint where crossings are ignored
LOST_CORNER_RADIUS_PX = 1.5  # original px: an on-fit join this close keeps an off-fit corner
CORNER_DEG = 90.0
# Enhance fallbacks that are legitimate (not a silent ML failure).
ALLOWED_FALLBACKS = ("too small",)


def jpeg_bytes(rgb, q):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=q)
    return buf.getvalue()


def decode(data):
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB")).copy()


def noise(rgb, sigma, seed):
    rng = np.random.default_rng(seed)
    return np.clip(np.rint(rgb + rng.normal(0, sigma, rgb.shape)), 0, 255).astype(np.uint8)


def blur(rgb, s):
    return np.asarray(Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(s)))


def gradient(rgb, amp=14.0):
    h, w = rgb.shape[:2]
    g = (np.linspace(-1, 1, w)[None, :] * 0.6 + np.linspace(-1, 1, h)[:, None] * 0.4) * amp
    return np.clip(rgb + g[..., None], 0, 255).astype(np.uint8)


def degrade(kind, rgb, seed):
    """JPEG file bytes of the degraded image (every degradation ends in a JPEG save, so the
    loader reports source_format JPEG and enhance deblocks it, as for a real JPEG upload)."""
    if kind == "jpeg_q50":
        return jpeg_bytes(rgb, 50)
    if kind == "jpeg_q30":
        return jpeg_bytes(rgb, 30)
    if kind == "small_web":
        h, w = rgb.shape[:2]
        small = Image.fromarray(rgb).resize((w // 2, h // 2), Image.LANCZOS)
        return jpeg_bytes(np.asarray(small), 75)
    if kind == "scan":
        return jpeg_bytes(noise(blur(rgb, 0.8), 6, seed), 80)
    if kind == "phone":
        return jpeg_bytes(gradient(noise(blur(rgb, 1.2), 10, seed)), 70)
    raise ValueError(kind)


KINDS = ("jpeg_q50", "jpeg_q30", "small_web", "scan", "phone")


def sources():
    from eval.synthetic import aa_disc, aa_shapes
    src = {USER_NAME: np.asarray(Image.open(USER_IMG).convert("RGB"))} if USER_IMG else {}
    for s in range(6):
        src[f"aa_shapes_{s}"] = aa_shapes(s, size=256)[..., :3].copy()
    src["aa_disc"] = aa_disc(size=256, cx=128.3, cy=127.7, r=80.8)[..., :3].copy()
    return src


def _input_path(stem):
    """The job's input file: degraded JPEG bytes, or the user's original file for "clean"."""
    return OUT / f"{stem}.jpg"


# ---------------------------------------------------------------- geometry metrics


def _tan(seg, end):
    from engine.types import Arc, CubicBezier, Line
    if isinstance(seg, Line):
        a, b = seg.p0, seg.p1
    elif isinstance(seg, CubicBezier):
        pts = (seg.p0, seg.c1, seg.c2, seg.p1)
        if end:
            a = next((p for p in pts[2::-1] if p != seg.p1), seg.p0); b = seg.p1
        else:
            a = seg.p0; b = next((p for p in pts[1:] if p != seg.p0), seg.p1)
    elif isinstance(seg, Arc):
        p = seg.p1 if end else seg.p0
        rx, ry = p[0] - seg.center[0], p[1] - seg.center[1]
        d = (-ry, rx) if seg.sweep else (ry, -rx)
        return math.atan2(d[1], d[0])
    else:
        return None
    return math.atan2(b[1] - a[1], b[0] - a[0])


def _joins(fc):
    """(point, turn deg) of every interior join of a fitted curve (wrap join if closed)."""
    segs = fc.segments
    pairs = list(zip(segs, segs[1:]))
    if len(segs) > 1 and segs[-1].p1 == segs[0].p0:
        pairs.append((segs[-1], segs[0]))
    out = []
    for a, b in pairs:
        t0, t1 = _tan(a, True), _tan(b, False)
        if t0 is None or t1 is None:
            continue
        out.append((a.p1, abs(math.degrees((t1 - t0 + math.pi) % (2 * math.pi) - math.pi))))
    return out


def kinks(fitted, lo=10.0, hi=30.0):
    return sum(lo <= d <= hi for fc in fitted if fc.primitive is None for _, d in _joins(fc))


def corners(fitted, deg=CORNER_DEG):
    """Points of interior joins turning >= deg."""
    return [p for fc in fitted for p, d in _joins(fc) if d >= deg]


def join_points(fitted):
    """Every segment endpoint of every curve (joins and junctions)."""
    return [p for fc in fitted for s in fc.segments for p in (s.p0, s.p1)]


def lost_corners(off, on, scale):
    """>= 90 deg corners of `off` with no join of `on` within LOST_CORNER_RADIUS_PX x scale."""
    from scipy.spatial import cKDTree
    pts = corners(off)
    if not pts:
        return 0, 0
    tree = cKDTree(np.asarray(join_points(on), np.float64))
    d, _ = tree.query(np.asarray(pts, np.float64))
    return int((d > LOST_CORNER_RADIUS_PX * scale).sum()), len(pts)


def _flatten(seg):
    """Polyline points of one segment, pieces at most ~FLATTEN_PX long."""
    from engine.types import Arc, CubicBezier, Line
    if isinstance(seg, Line):
        p0, p1 = np.asarray(seg.p0, float), np.asarray(seg.p1, float)
        k = max(1, math.ceil(np.hypot(*(p1 - p0)) / FLATTEN_PX))
        t = np.linspace(0, 1, k + 1)[:, None]
        return p0 + t * (p1 - p0)
    if isinstance(seg, CubicBezier):
        P = np.asarray((seg.p0, seg.c1, seg.c2, seg.p1), float)
        poly = sum(np.hypot(*(P[i + 1] - P[i])) for i in range(3))
        k = min(4000, max(1, math.ceil(poly / FLATTEN_PX)))
        t = np.linspace(0, 1, k + 1)[:, None]
        u = 1 - t
        return u**3 * P[0] + 3 * u**2 * t * P[1] + 3 * u * t**2 * P[2] + t**3 * P[3]
    if isinstance(seg, Arc):
        c = np.asarray(seg.center, float)
        a0 = math.atan2(seg.p0[1] - c[1], seg.p0[0] - c[0])
        a1 = math.atan2(seg.p1[1] - c[1], seg.p1[0] - c[0])
        delta = (a1 - a0) % (2 * math.pi) if seg.sweep else -((a0 - a1) % (2 * math.pi))
        k = min(4000, max(1, math.ceil(abs(delta) * seg.radius / FLATTEN_PX)))
        a = a0 + np.linspace(0, delta, k + 1)
        pts = c + seg.radius * np.column_stack([np.cos(a), np.sin(a)])
        pts[0], pts[-1] = seg.p0, seg.p1
        return pts
    raise TypeError(type(seg))


def crossings(fitted, scale):
    """Unordered pairs of different curves that cross away from shared endpoints (see top)."""
    from scipy.spatial import cKDTree
    starts, ends, owner = [], [], []
    endpoints = {}
    for fc in fitted:
        if not fc.segments:
            continue
        endpoints[fc.curve_id] = {fc.segments[0].p0, fc.segments[-1].p1}
        for seg in fc.segments:
            pts = _flatten(seg)
            starts.append(pts[:-1]); ends.append(pts[1:])
            owner.append(np.full(len(pts) - 1, fc.curve_id))
    if not starts:
        return 0
    a, b, own = np.concatenate(starts), np.concatenate(ends), np.concatenate(owner)
    mid = (a + b) / 2
    reach = float(np.hypot(*(b - a).T).max())
    pairs = cKDTree(mid).query_pairs(reach + 1e-9, output_type="ndarray")
    if len(pairs) == 0:
        return 0
    i, j = pairs[:, 0], pairs[:, 1]
    keep = own[i] != own[j]
    i, j = i[keep], j[keep]

    def orient(p, q, r):
        return (q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1]) - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0])

    d1, d2 = orient(a[i], b[i], a[j]), orient(a[i], b[i], b[j])
    d3, d4 = orient(a[j], b[j], a[i]), orient(a[j], b[j], b[i])
    lo_i, hi_i = np.minimum(a[i], b[i]), np.maximum(a[i], b[i])
    lo_j, hi_j = np.minimum(a[j], b[j]), np.maximum(a[j], b[j])
    boxes = np.all((lo_i <= hi_j) & (lo_j <= hi_i), axis=1)  # decides the collinear case
    hit = (d1 * d2 <= 0) & (d3 * d4 <= 0) & boxes
    i, j, d1, d2 = i[hit], j[hit], d1[hit], d2[hit]
    den = d1 - d2
    t = np.where(den != 0, d1 / np.where(den != 0, den, 1.0), 0.5)[:, None]
    x = a[j] + t * (b[j] - a[j])  # collinear overlap: piece j's midpoint stands in
    radius = SHARED_END_RADIUS_PX * scale
    found = set()
    for k in range(len(i)):
        ci, cj = int(own[i[k]]), int(own[j[k]])
        pair = (min(ci, cj), max(ci, cj))
        if pair in found:
            continue
        shared = endpoints[ci] & endpoints[cj]
        if any(math.hypot(x[k, 0] - p[0], x[k, 1] - p[1]) <= radius for p in shared):
            continue
        found.add(pair)
    return len(found)


# ---------------------------------------------------------------- phase 1: enhance (GPU)


def _jobs():
    jobs = []
    for name, rgb in sources().items():
        for i, kind in enumerate(KINDS):
            jobs.append((name, kind, degrade(kind, rgb, seed=i)))
        if name == USER_NAME:
            jobs.append((name, "clean", None))
    return jobs


def _save_input(stem, kind, data):
    """Write the job's input file and its decoded pixels (the metric reference) as PNG."""
    p = _input_path(stem)
    p.write_bytes(Path(USER_IMG).read_bytes() if kind == "clean" else data)
    Image.fromarray(decode(p.read_bytes())).save(OUT / f"{stem}.png")
    return p


def phase_enhance():
    from engine.loader import load_image
    from engine.presets import resolve_preset
    from engine.stages.enhance import enhance
    OUT.mkdir(parents=True, exist_ok=True)
    preset = resolve_preset("auto")
    meta = {}
    for name, kind, degraded in _jobs():
        stem = f"{name}__{kind}"
        data = _save_input(stem, kind, degraded).read_bytes()
        image = load_image(data)
        t = time.perf_counter()
        out, report = enhance(image, preset.enhance, preset.segment.max_colors)
        report = {k: v for k, v in report.items()}
        report["enhance_s"] = time.perf_counter() - t
        if report.get("model_used") == "bicubic" and report.get("fallback_reason") not in ALLOWED_FALLBACKS:
            raise SystemExit(f"{stem}: enhance fell back to bicubic ({report.get('fallback_reason')})")
        np.savez(OUT / f"{stem}__enh.npz", pixels=out.pixels, scale=out.scale,
                 original_size=np.asarray(out.original_size), source_format=str(out.source_format),
                 color_space=out.color_space)
        meta[stem] = report
        print(stem, report["model_used"], report["fallback_reason"], report["skipped_reason"],
              f"x{report['effective_scale']}", f"{report['enhance_s']:.1f}s", flush=True)
    (OUT / "enhance_meta.json").write_text(json.dumps(meta, indent=1))
    _check_cache_matches_vectorize(meta)


def _svg_digest(svg):
    return hashlib.sha256(svg.encode()).hexdigest()


def _check_cache_matches_vectorize(meta):
    """segment..export on the cached enhanced image must equal vectorize() on the input bytes."""
    from engine import to_svg
    from engine.pipeline import vectorize
    stem = next(s for s in meta if s.startswith("aa_shapes_0"))
    data = _input_path(stem).read_bytes()
    for adaptive in (False, True):
        direct = to_svg(vectorize(data, overrides={"fit.adaptive": adaptive}).doc)
        cached = to_svg(_run_cached(stem, "on", adaptive)[0].doc)
        if _svg_digest(direct) != _svg_digest(cached):
            raise SystemExit(f"{stem}: cached-enhance pipeline differs from vectorize (adaptive={adaptive})")
    print("cache check: cached-enhance pipeline == vectorize", flush=True)


# ---------------------------------------------------------------- phase 2: fit (CPU workers)


def _run_cached(stem, enhance_mode, adaptive, fit_hook=None):
    import engine.pipeline as pl
    from engine.loader import load_image
    from engine.presets import apply_overrides, resolve_preset
    from engine.types import RasterImage
    preset = apply_overrides(resolve_preset("auto"),
                             {"enhance.enabled": False, "fit.adaptive": adaptive})
    if enhance_mode == "on":
        z = np.load(OUT / f"{stem}__enh.npz")
        fmt = str(z["source_format"])
        image = RasterImage(z["pixels"], tuple(int(v) for v in z["original_size"]),
                            scale=float(z["scale"]), color_space=str(z["color_space"]),
                            source_format=None if fmt == "None" else fmt)
    else:
        image = load_image(_input_path(stem).read_bytes())
    t = time.perf_counter()
    res = pl.run_pipeline(image, preset)
    return res, time.perf_counter() - t, image.scale


def run_file_stem(stem, enhance_mode, tag, adaptive):
    """Per-run output name (the .svg and .fit.pkl files)."""
    return f"{stem}__enh{enhance_mode}__{tag}__{'on' if adaptive else 'off'}"


def run_one(job):
    stem, enhance_mode, adaptives, run_tag = job
    import engine.geometry.adaptive_fit as af
    import engine.pipeline as pl
    from engine import to_svg
    from eval.metrics import count_nodes, mean_delta_e, render_svg, ssim

    noises, captured = [], []
    orig_est, orig_fit = af.estimate_noise, pl.fit_geometry

    def est(*a, **k):
        v = orig_est(*a, **k); noises.append(v); return v

    def fit(*a, **k):
        r = orig_fit(*a, **k); captured.append(r); return r

    af.estimate_noise, pl.fit_geometry = est, fit
    px = np.asarray(Image.open(OUT / f"{stem}.png").convert("RGBA"))
    h, w = px.shape[:2]
    meta = json.loads((OUT / "enhance_meta.json").read_text()).get(stem, {}) if enhance_mode == "on" else {}
    name, kind = stem.split("__")
    rows = []
    for adaptive in adaptives:
        noises.clear(); captured.clear()
        res, total, scale = _run_cached(stem, enhance_mode, adaptive)
        svg = to_svg(res.doc)
        tag = run_file_stem(stem, enhance_mode, run_tag, adaptive)
        (OUT / f"{tag}.svg").write_text(svg)
        fitted = captured[-1]
        with open(OUT / f"{tag}.fit.pkl", "wb") as fh:
            pickle.dump((scale, fitted), fh)
        render = render_svg(svg, w, h)
        nodes, paths = count_nodes(svg)
        fit_s = sum(r.duration_s for r in res.reports if r.stage == "fit")
        rows.append(dict(
            source=name, kind=kind, enhance=enhance_mode, adaptive=int(adaptive),
            scale=scale, model_used=meta.get("model_used") or "", fallback=meta.get("fallback_reason") or "",
            skipped=meta.get("skipped_reason") or "", nodes=nodes, paths=paths,
            ssim=ssim(px, render), delta_e=mean_delta_e(px, render), fit_s=fit_s, total_s=total,
            noise_med=float(np.median(noises)) if noises else float("nan"),
            kinks_10_30=kinks(fitted), crossings=crossings(fitted, scale),
            corners90=len(corners(fitted)), svg_sha=_svg_digest(svg)[:16]))
    return rows


def phase_fit(args):
    adaptives = tuple({"off": False, "on": True}[a] for a in args.adaptive.split(","))
    meta_path = OUT / "enhance_meta.json"
    if not meta_path.exists():
        raise SystemExit("run the enhance phase first")
    stems = list(json.loads(meta_path.read_text()))
    if args.only:
        stems = [s for s in stems if args.only in s]
    stems.sort(key=lambda s: not s.startswith(USER_NAME))  # big ones first
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # workers inherit it: no GPU in the fit phase
    jobs = [(s, args.enhance, adaptives, args.tag) for s in stems]
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for r in ex.map(run_one, jobs):
            rows += r
            print(r[0]["source"], r[0]["kind"], *(x["nodes"] for x in r), flush=True)
    path = OUT / f"results_{args.enhance}_{args.tag}.csv"
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("enhance", "fit"))
    ap.add_argument("--enhance", choices=("on", "off"), default="on")
    ap.add_argument("--adaptive", default="off,on")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    if args.phase == "enhance":
        phase_enhance()
    else:
        phase_fit(args)


if __name__ == "__main__":
    main()
