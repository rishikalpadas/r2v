"""Fix 2 evaluation: adaptive off vs on over degraded inputs. Run from repo root."""
import csv
import io
import math
import os
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


def jpeg(rgb, q):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=q)
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB")).copy()


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
    if kind == "jpeg_q50":
        return jpeg(rgb, 50)
    if kind == "jpeg_q30":
        return jpeg(rgb, 30)
    if kind == "small_web":
        h, w = rgb.shape[:2]
        small = Image.fromarray(rgb).resize((w // 2, h // 2), Image.LANCZOS)
        return jpeg(np.asarray(small), 75)
    if kind == "scan":
        return jpeg(noise(blur(rgb, 0.8), 6, seed), 80)
    if kind == "phone":
        return jpeg(gradient(noise(blur(rgb, 1.2), 10, seed)), 70)
    raise ValueError(kind)


KINDS = ("jpeg_q50", "jpeg_q30", "small_web", "scan", "phone")


def sources():
    from eval.synthetic import aa_disc, aa_shapes
    src = {"1jpg": np.asarray(Image.open(USER_IMG).convert("RGB"))} if USER_IMG else {}
    for s in range(6):
        src[f"aa_shapes_{s}"] = aa_shapes(s, size=256)[..., :3].copy()
    src["aa_disc"] = aa_disc(size=256, cx=128.3, cy=127.7, r=80.8)[..., :3].copy()
    return src


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


def kinks(fitted, lo=10.0, hi=30.0):
    n = 0
    for fc in fitted:
        segs = fc.segments
        if fc.primitive is not None or len(segs) < 2:
            continue
        pairs = list(zip(segs, segs[1:]))
        if segs[-1].p1 == segs[0].p0:
            pairs.append((segs[-1], segs[0]))
        for a, b in pairs:
            t0, t1 = _tan(a, True), _tan(b, False)
            if t0 is None or t1 is None:
                continue
            d = abs(math.degrees((t1 - t0 + math.pi) % (2 * math.pi) - math.pi))
            n += lo <= d <= hi
    return n


def run_one(job):
    name, kind, path = job
    import engine.geometry.adaptive_fit as af
    import engine.pipeline as pl
    from engine import to_svg
    from engine.pipeline import vectorize
    from eval.metrics import count_nodes, mean_delta_e, render_svg, ssim

    noises, captured = [], []
    orig_est, orig_fit = af.estimate_noise, pl.fit_geometry

    def est(*a, **k):
        v = orig_est(*a, **k); noises.append(v); return v

    def fit(*a, **k):
        r = orig_fit(*a, **k); captured.append(r); return r

    af.estimate_noise, pl.fit_geometry = est, fit
    data = Path(path).read_bytes()
    px = np.asarray(Image.open(path).convert("RGBA"))
    h, w = px.shape[:2]
    rows = []
    for adaptive in (False, True):
        noises.clear(); captured.clear()
        t = time.perf_counter()
        res = vectorize(data, overrides={"fit.adaptive": adaptive})
        total = time.perf_counter() - t
        svg = to_svg(res.doc)
        Path(path).with_name(f"{name}__{kind}__{'on' if adaptive else 'off'}.svg").write_text(svg)
        render = render_svg(svg, w, h)
        nodes, paths = count_nodes(svg)
        fit_s = sum(r.duration_s for r in res.reports if "fit" in r.stage)
        rows.append(dict(source=name, kind=kind, adaptive=int(adaptive), nodes=nodes, paths=paths,
                         ssim=ssim(px, render), delta_e=mean_delta_e(px, render), fit_s=fit_s,
                         total_s=total,
                         noise_med=float(np.median(noises)) if noises else float("nan"),
                         kinks_10_30=kinks(captured[-1]) if captured else -1))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name, rgb in sources().items():
        for i, kind in enumerate(KINDS):
            p = OUT / f"{name}__{kind}.png"
            Image.fromarray(degrade(kind, rgb, seed=i)).save(p)
            jobs.append((name, kind, str(p)))
    jobs.sort(key=lambda j: j[0] != "1jpg")  # big ones first
    rows = []
    with ProcessPoolExecutor(4) as ex:
        for r in ex.map(run_one, jobs):
            rows += r
            print(r[0]["source"], r[0]["kind"], r[0]["nodes"], r[1]["nodes"], flush=True)
    with open(OUT / "results.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)


if __name__ == "__main__":
    main()
