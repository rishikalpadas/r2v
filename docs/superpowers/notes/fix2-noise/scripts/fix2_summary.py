import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[5]  # docs/superpowers/notes/fix2-noise/scripts -> repo root
sys.path.insert(0, str(REPO))
from eval.metrics import render_svg  # noqa: E402

OUT = REPO / ".superpowers" / "fix2-noise" / "eval"
rows = list(csv.DictReader(open(OUT / "results.csv")))
by = {}
for r in rows:
    by.setdefault((r["source"], r["kind"]), {})[int(r["adaptive"])] = r
d = []
for (src, kind), p in by.items():
    a, b = p[0], p[1]

    def f(k, x):
        return float(x[k])

    d.append(dict(source=src, kind=kind, nodes_off=int(a["nodes"]), nodes_on=int(b["nodes"]),
                  dnodes_pct=(f("nodes", b) / f("nodes", a) - 1) * 100,
                  ssim_off=f("ssim", a), dssim=f("ssim", b) - f("ssim", a),
                  dE_off=f("delta_e", a), ddE=f("delta_e", b) - f("delta_e", a),
                  fit_off=f("fit_s", a), fit_on=f("fit_s", b),
                  dfit_pct=(f("fit_s", b) / f("fit_s", a) - 1) * 100,
                  noise=f("noise_med", b), kinks_off=int(a["kinks_10_30"]),
                  kinks_on=int(b["kinks_10_30"])))
with open(OUT / "paired.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(d[0]))
    w.writeheader()
    w.writerows(d)


def agg(g):
    no, nn = sum(x["nodes_off"] for x in g), sum(x["nodes_on"] for x in g)

    def m(k):
        return float(np.mean([x[k] for x in g]))

    return (f"n={len(g):2d} noise={np.median([x['noise'] for x in g]):.3f} nodes {no}->{nn} "
            f"tot {100*(nn/no-1):+.1f}% mean {m('dnodes_pct'):+.1f}% | dSSIM mean {m('dssim'):+.4f} "
            f"min {min(x['dssim'] for x in g):+.4f} | ddE mean {m('ddE'):+.3f} "
            f"max {max(x['ddE'] for x in g):+.3f} | fit "
            f"{100*(sum(x['fit_on'] for x in g)/sum(x['fit_off'] for x in g)-1):+.0f}% | kinks "
            f"{sum(x['kinks_off'] for x in g)}->{sum(x['kinks_on'] for x in g)}")


KINDS = ("jpeg_q50", "jpeg_q30", "small_web", "scan", "phone")
for label, sel in (("all", lambda x: True), ("synthetic", lambda x: x["source"] != "1jpg")):
    print(f"== per degradation ({label}) ==")
    for k in KINDS:
        print(f"{k:10s}", agg([x for x in d if x["kind"] == k and sel(x)]))
    print(f"{'TOTAL':10s}", agg([x for x in d if sel(x)]))
print("== 1.jpg ==")
for x in d:
    if x["source"] == "1jpg":
        print(f"{x['kind']:10s} noise={x['noise']:.3f} nodes {x['nodes_off']}->{x['nodes_on']} "
              f"({x['dnodes_pct']:+.1f}%) SSIM {x['ssim_off']:.4f} {x['dssim']:+.4f} "
              f"dE {x['dE_off']:.3f} {x['ddE']:+.3f} fit {x['fit_off']:.1f}s {x['dfit_pct']:+.0f}% "
              f"kinks {x['kinks_off']}->{x['kinks_on']}")

# crops for the variants where on and off differ most (relative node change)
for row in [SimpleNamespace(**x) for x in sorted(d, key=lambda x: -abs(x["dnodes_pct"]))[:2]]:
    stem = f"{row.source}__{row.kind}"
    src = np.asarray(Image.open(OUT / f"{stem}.png").convert("RGBA"))
    h, w = src.shape[:2]
    ren = {m: render_svg((OUT / f"{stem}__{m}.svg").read_text(), w, h) for m in ("off", "on")}
    diff = np.abs(ren["off"].astype(int) - ren["on"].astype(int)).sum(axis=2)
    ys, xs = np.nonzero(diff > 30)
    cy, cx = (int(np.median(ys)), int(np.median(xs))) if len(ys) else (h // 2, w // 2)
    s = min(160, h, w) // 2
    y0, x0 = max(0, min(h - 2 * s, cy - s)), max(0, min(w - 2 * s, cx - s))
    scale = 3
    tiles = [Image.fromarray(a[y0:y0 + 2 * s, x0:x0 + 2 * s]).convert("RGB").resize(
        (2 * s * scale, 2 * s * scale), Image.NEAREST) for a in (src, ren["off"], ren["on"])]
    W = tiles[0].width
    canvas = Image.new("RGB", (3 * W + 20, W + 24), "white")
    dr = ImageDraw.Draw(canvas)
    labels = ("input", f"off {row.nodes_off} nodes", f"on {row.nodes_on} nodes")
    for i, (t, lab) in enumerate(zip(tiles, labels)):
        canvas.paste(t, (i * (W + 10), 24))
        dr.text((i * (W + 10) + 4, 6), lab, fill="black")
    canvas.save(OUT / f"crop_{stem}.png")
    print("crop", OUT / f"crop_{stem}.png")
