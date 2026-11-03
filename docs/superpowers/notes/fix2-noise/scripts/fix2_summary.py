"""Summarise fix2_eval.py results: keep-rule tables, per-kind tables, HAPPINESS, crops.

    python fix2_summary.py --enhance on --tag B            # off and on rows from results_on_B.csv
    python fix2_summary.py --enhance on --tag A --off-tag B  # on rows from A, off rows from B

Keep rule (M4 plan Task 7), on the degraded set (every kind except "clean"):
>= 10% fewer nodes summed, mean SSIM not lower, mean Delta E not higher. Lost >= 90 deg corners
are computed here from the pickled fits (definition in fix2_eval.py). With --crops, the
HAPPINESS clean LOOKS crop and a 3x zoom are written to .superpowers/fix2-noise/.
"""
import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from fix2_eval import OUT, USER_NAME, lost_corners, run_file_stem  # noqa: E402

KINDS = ("jpeg_q50", "jpeg_q30", "small_web", "scan", "phone")
LOOKS_CROP = (130, 700, 730, 1000)  # x0, y0, x1, y1 in original px
ZOOM_BOX = (225, 720, 365, 860)
ZOOM = 3


def _rows(enhance, tag):
    return list(csv.DictReader(open(OUT / f"results_{enhance}_{tag}.csv")))


def _load_fit(stem, enhance, tag, adaptive):
    with open(OUT / f"{run_file_stem(stem, enhance, tag, adaptive)}.fit.pkl", "rb") as fh:
        return pickle.load(fh)


def paired(enhance, tag, off_tag):
    off = {(r["source"], r["kind"]): r for r in _rows(enhance, off_tag) if r["adaptive"] == "0"}
    on = {(r["source"], r["kind"]): r for r in _rows(enhance, tag) if r["adaptive"] == "1"}
    out = []
    for key in sorted(set(off) & set(on)):
        a, b = off[key], on[key]
        stem = "__".join(key)
        scale, fit_off = _load_fit(stem, enhance, off_tag, False)
        _, fit_on = _load_fit(stem, enhance, tag, True)
        lost, total = lost_corners(fit_off, fit_on, scale)
        f = float
        out.append(dict(
            source=key[0], kind=key[1], scale=f(a["scale"]), model=a.get("model_used", ""),
            fallback=a.get("fallback", ""), skipped=a.get("skipped", ""),
            nodes_off=int(a["nodes"]), nodes_on=int(b["nodes"]),
            paths_off=int(a["paths"]), paths_on=int(b["paths"]),
            ssim_off=f(a["ssim"]), ssim_on=f(b["ssim"]),
            dE_off=f(a["delta_e"]), dE_on=f(b["delta_e"]),
            fit_off=f(a["fit_s"]), fit_on=f(b["fit_s"]),
            noise=f(b["noise_med"]), kinks_off=int(a["kinks_10_30"]), kinks_on=int(b["kinks_10_30"]),
            cross_off=int(a["crossings"]), cross_on=int(b["crossings"]),
            corners90_off=total, lost90=lost))
    return out


def agg(g):
    no, nn = sum(x["nodes_off"] for x in g), sum(x["nodes_on"] for x in g)
    m = lambda k: float(np.mean([x[k] for x in g]))  # noqa: E731
    s = lambda k: sum(x[k] for x in g)  # noqa: E731
    return dict(
        n=len(g), nodes_off=no, nodes_on=nn, dnodes=100 * (nn / no - 1),
        dnodes_mean=float(np.mean([100 * (x["nodes_on"] / x["nodes_off"] - 1) for x in g])),
        ssim_off=m("ssim_off"), ssim_on=m("ssim_on"), dssim=m("ssim_on") - m("ssim_off"),
        dssim_min=min(x["ssim_on"] - x["ssim_off"] for x in g),
        dE_off=m("dE_off"), dE_on=m("dE_on"), ddE=m("dE_on") - m("dE_off"),
        ddE_max=max(x["dE_on"] - x["dE_off"] for x in g),
        fit_off=s("fit_off"), fit_on=s("fit_on"), dfit=100 * (s("fit_on") / s("fit_off") - 1),
        kinks_off=s("kinks_off"), kinks_on=s("kinks_on"),
        cross_off=s("cross_off"), cross_on=s("cross_on"),
        corners90=s("corners90_off"), lost90=s("lost90"),
        noise=float(np.median([x["noise"] for x in g])))


def table(groups):
    head = ("| set | n | nodes off->on | sum % | mean % | SSIM off -> on (d, min d) | dE off -> on "
            "(d, max d) | fit s off->on (%) | kinks | crossings | lost >=90 / corners | noise med |")
    lines = [head, "|" + "---|" * 12]
    for label, g in groups:
        if not g:
            continue
        a = agg(g)
        lines.append(
            f"| {label} | {a['n']} | {a['nodes_off']}->{a['nodes_on']} | {a['dnodes']:+.1f}% | "
            f"{a['dnodes_mean']:+.1f}% | {a['ssim_off']:.4f} -> {a['ssim_on']:.4f} ({a['dssim']:+.4f}, "
            f"{a['dssim_min']:+.4f}) | {a['dE_off']:.3f} -> {a['dE_on']:.3f} ({a['ddE']:+.3f}, "
            f"{a['ddE_max']:+.3f}) | {a['fit_off']:.1f}->{a['fit_on']:.1f} ({a['dfit']:+.0f}%) | "
            f"{a['kinks_off']}->{a['kinks_on']} | {a['cross_off']}->{a['cross_on']} | "
            f"{a['lost90']}/{a['corners90']} | {a['noise']:.3f} |")
    return "\n".join(lines)


def keep_rule(g):
    a = agg(g)
    checks = (("nodes summed <= -10%", a["dnodes"] <= -10.0, f"{a['dnodes']:+.1f}%"),
              ("mean SSIM not lower", a["dssim"] >= 0.0, f"{a['dssim']:+.4f}"),
              ("mean dE not higher", a["ddE"] <= 0.0, f"{a['ddE']:+.3f}"))
    lines = [f"- {name}: {value} -> {'PASS' if ok else 'FAIL'}" for name, ok, value in checks]
    lines.append(f"- keep rule: {'MET' if all(ok for _, ok, _ in checks) else 'NOT MET'}")
    return "\n".join(lines)


def happiness(d):
    lines = ["| kind | x | model | nodes off->on | paths off->on | SSIM off->on | dE off->on | "
             "fit s off->on | kinks | crossings | lost >=90 / corners |", "|" + "---|" * 11]
    for x in d:
        if x["source"] != USER_NAME:
            continue
        lines.append(
            f"| {x['kind']} | {x['scale']:g} | {x['model'] or x['skipped'] or '-'} | "
            f"{x['nodes_off']}->{x['nodes_on']} ({100 * (x['nodes_on'] / x['nodes_off'] - 1):+.1f}%) | "
            f"{x['paths_off']}->{x['paths_on']} | {x['ssim_off']:.4f}->{x['ssim_on']:.4f} | "
            f"{x['dE_off']:.3f}->{x['dE_on']:.3f} | {x['fit_off']:.1f}->{x['fit_on']:.1f} | "
            f"{x['kinks_off']}->{x['kinks_on']} | {x['cross_off']}->{x['cross_on']} | "
            f"{x['lost90']}/{x['corners90_off']} |")
    return "\n".join(lines)


def crops(enhance, tag, off_tag):
    from eval.metrics import render_svg
    stem = f"{USER_NAME}__clean"
    src = np.asarray(Image.open(OUT / f"{stem}.png").convert("RGBA"))
    h, w = src.shape[:2]
    ren = {}
    for adaptive, t in ((False, off_tag), (True, tag)):
        svg = (OUT / f"{run_file_stem(stem, enhance, t, adaptive)}.svg").read_text()
        ren[adaptive] = render_svg(svg, w, h)
    dest = OUT.parent
    for (x0, y0, x1, y1), zoom, name in ((LOOKS_CROP, 1, "looks_side_by_side"),
                                         (ZOOM_BOX, ZOOM, "looks_zoom3x")):
        tiles = [Image.fromarray(a[y0:y1, x0:x1]).convert("RGB") for a in (src, ren[False], ren[True])]
        tiles = [t.resize((t.width * zoom, t.height * zoom), Image.NEAREST) for t in tiles]
        W, H = tiles[0].size
        canvas = Image.new("RGB", (3 * W + 20, H + 24), "white")
        dr = ImageDraw.Draw(canvas)
        for i, (t, lab) in enumerate(zip(tiles, ("source", "adaptive off", "adaptive on"))):
            canvas.paste(t, (i * (W + 10), 24))
            dr.text((i * (W + 10) + 4, 6), lab, fill="black")
        path = dest / f"{name}_m4_enh{enhance}_{tag}.png"
        canvas.save(path)
        print("crop", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enhance", choices=("on", "off"), default="on")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--off-tag", default=None)
    ap.add_argument("--crops", action="store_true")
    args = ap.parse_args()
    off_tag = args.off_tag or args.tag
    d = paired(args.enhance, args.tag, off_tag)
    with open(OUT / f"paired_{args.enhance}_{args.tag}.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(d[0]))
        wr.writeheader(); wr.writerows(d)
    degraded = [x for x in d if x["kind"] != "clean"]
    print(f"## enhance {args.enhance}, on={args.tag}, off={off_tag}")
    models = sorted({(x["model"] or x["skipped"] or "-", x["fallback"] or "") for x in d})
    print("enhance models:", models)
    print("\n### keep rule (degraded set)\n" + keep_rule(degraded))
    print("\n### per degradation\n" + table(
        [(k, [x for x in degraded if x["kind"] == k]) for k in KINDS]
        + [("synthetic", [x for x in degraded if x["source"] != USER_NAME]),
           ("TOTAL", degraded)]))
    print("\n### HAPPINESS\n" + happiness(d))
    if args.crops and any(x["source"] == USER_NAME and x["kind"] == "clean" for x in d):
        crops(args.enhance, args.tag, off_tag)


if __name__ == "__main__":
    main()
