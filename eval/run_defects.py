"""Defect report: convert every image in a folder and rank them by automatic defect detectors.

    python -m eval.run_defects --images eval/dataset/logos --out eval/reports/defects
    python -m eval.run_defects --out eval/reports/after --compare eval/reports/defects/defects.csv

Writes `defects.csv` (one row per image), `svg/<image>.svg`, `crops/<image>/<detector>_<k>.png`
(5x source | output | output outlines, around each top spot) and `report.html`, images sorted
worst-first by the combined score. See `eval.defects` for what each detector measures.
"""
from __future__ import annotations

import argparse
import csv
import html
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageDraw

from eval.defects import DETECTORS, DetectorResult, Spot, detect_all, geometry_from_state
from eval.metrics import count_nodes, edge_f1, mean_delta_e, render_svg, ssim
from eval.run_benchmark import CLEAN_DIR, IMAGE_SUFFIXES

# `image` is the path relative to the --images root (posix), the key --compare matches on.
COLUMNS = ["image", "status", "error", "combined", *DETECTORS,
           "ssim", "delta_e", "edge_f1", "nodes", "paths", "runtime_s", "reference",
           "enhance", "scale", "git_sha"]
METRICS = ("ssim", "delta_e", "edge_f1", "nodes", "runtime_s")
SKIP_DIRS = {CLEAN_DIR, "reference"}

# Combined score = sum over detectors of score / NORM: each NORM is the score that counts as one
# unit of visible badness, so no single detector's units dominate the ranking.
NORM = {
    "crossings": 1.0,  # every crossing is a visible spike through a band
    "specks": 5.0,  # a handful of specks is one noticeable blemish
    "wobble": 2.0,  # e.g. two window lengths at twice the flag line
    "thin_bands": 2.0,  # same units as wobble
    "junction_spikes": 3.0,  # ~3 clear spikes
    "straightness": 2.0,  # same units as wobble
    "worst_patches": 3.0,  # worst windows at ~3 units (e.g. dE 6, or SSIM 0.7)
}
TOP_N = 3  # spots shown per detector per image
CROP_ZOOM = 5
CROP_MIN_PX = 40  # crop side in reference px, at least (enough context around a speck)
CROP_MAX_PX = 160  # and at most (a larger spot is shown around its centre)
CROP_PAD_PX = 6
OUTLINE_STYLE = "<style>*{{fill-opacity:.45;stroke:#e0007a;stroke-width:{w}}}</style>"


# ---------------------------------------------------------------- per image


def _images(folder: Path) -> list[tuple[str, Path]]:
    """Images directly in `folder`, plus one level of sub-folders (e.g. a dataset root)."""
    found = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            found.append((p.name, p))
        elif p.is_dir() and p.name not in SKIP_DIRS:
            found.extend((f"{p.name}/{q.name}", q) for q in sorted(p.iterdir())
                         if q.is_file() and q.suffix.lower() in IMAGE_SUFFIXES)
    return found


def _safe(name: str) -> str:
    """File-safe output name; keeps the extension so a.png and a.jpg do not collide."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).as_posix())


def _scaled(result: DetectorResult, k: tuple[float, float]) -> DetectorResult:
    kx, ky = k
    if (kx, ky) == (1.0, 1.0):
        return result
    spots = tuple(Spot((s.box[0] * kx, s.box[1] * ky, s.box[2] * kx, s.box[3] * ky),
                       s.severity, s.note) for s in result.spots)
    return DetectorResult(result.score, spots)


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=10, cwd=Path(__file__).resolve().parent)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _crop_box(box, w: int, h: int) -> tuple[int, int, int, int]:
    """Crop around `box`, padded, each side clamped to [CROP_MIN_PX, CROP_MAX_PX], inside the image."""
    x0, y0, x1, y1 = box
    out = []
    for lo, hi, size in ((x0, x1, w), (y0, y1, h)):
        side = min(int(np.clip(hi - lo + 2 * CROP_PAD_PX, CROP_MIN_PX, CROP_MAX_PX)), size)
        start = int(np.clip(round((lo + hi - side) / 2), 0, size - side))
        out.append((start, start + side))
    (left, right), (top, bottom) = out
    return left, top, right, bottom


def _svg_view(svg: str, view: tuple[float, float, float, float], size: tuple[int, int],
              extra: str = "") -> str:
    vx, vy, vw, vh = view
    root = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size[0]}" height="{size[1]}" '
            f'viewBox="{vx} {vy} {vw} {vh}">{extra}')
    return re.sub(r"<svg[^>]*>", lambda _: root, svg, count=1)


def _render_view(svg: str, view, size, extra: str = "") -> Image.Image:
    px = render_svg(_svg_view(svg, view, size, extra), *size)
    tile = Image.new("RGBA", px.shape[1::-1], (255, 255, 255, 255))
    tile.alpha_composite(Image.fromarray(px, "RGBA"))
    return tile.convert("RGB")


def _write_crop(dest: Path, target: np.ndarray, svg: str, k: tuple[float, float],
                spot: Spot) -> None:
    h, w = target.shape[:2]
    x0, y0, x1, y1 = _crop_box(spot.box, w, h)
    size = ((x1 - x0) * CROP_ZOOM, (y1 - y0) * CROP_ZOOM)
    src = Image.fromarray(target[y0:y1, x0:x1], "RGBA")
    src_tile = Image.new("RGBA", src.size, (255, 255, 255, 255))
    src_tile.alpha_composite(src)
    src_tile = src_tile.convert("RGB").resize(size, Image.Resampling.NEAREST)
    kx, ky = k
    view = (x0 / kx, y0 / ky, (x1 - x0) / kx, (y1 - y0) / ky)
    out = _render_view(svg, view, size)
    lines = _render_view(svg, view, size, OUTLINE_STYLE.format(w=1.0 / (CROP_ZOOM * min(k))))
    gap = 6
    sheet = Image.new("RGB", (size[0] * 3 + gap * 2, size[1]), (128, 128, 128))
    bx = [(v - o) * CROP_ZOOM for v, o in zip(spot.box, (x0, y0, x0, y0))]
    for i, tile in enumerate((src_tile, out, lines)):
        ImageDraw.Draw(tile).rectangle(bx, outline=(0, 160, 255), width=2)
        sheet.paste(tile, (i * (size[0] + gap), 0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dest)


def _load_rgba(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGBA"), dtype=np.uint8)


def analyse_image(name: str, path: Path, out: Path, enhance: bool, top: int,
                  sha: str = "") -> dict:
    """Convert one image, run every detector, write its SVG and crops; returns its CSV row
    plus `_spots` (detector -> spot list with crop paths) for the report."""
    from engine.loader import load_image
    from engine.pipeline import run_stages
    from engine.presets import apply_overrides, resolve_preset
    from engine.stages.export import to_svg

    preset = resolve_preset("auto")
    if not enhance:
        preset = apply_overrides(preset, {"enhance.enabled": False})
    image = load_image(path.read_bytes())
    start = time.perf_counter()
    state, _ = run_stages(image, preset)
    runtime = time.perf_counter() - start
    svg = to_svg(state.doc)
    stem = _safe(name)
    (out / "svg").mkdir(parents=True, exist_ok=True)
    (out / "svg" / f"{stem}.svg").write_text(svg, encoding="utf-8")

    clean = path.parent / CLEAN_DIR / f"{path.stem}.png"
    target = _load_rgba(clean) if clean.is_file() else image.pixels
    h, w = target.shape[:2]
    k = (w / image.original_size[0], h / image.original_size[1])  # reference px per input px
    render = render_svg(svg, w, h)
    if render.shape[:2] != (h, w):
        render = np.array(Image.fromarray(render, "RGBA").resize((w, h), Image.Resampling.LANCZOS))
    geometry = geometry_from_state(state)
    results = {d: (r if d == "worst_patches" else _scaled(r, k))
               for d, r in detect_all(geometry, target, render, image.pixels).items()}
    nodes, paths = count_nodes(svg)
    row = {"image": name, "status": "ok", "error": "",
           "ssim": f"{ssim(target, render):.6f}", "delta_e": f"{mean_delta_e(target, render):.6f}",
           "edge_f1": f"{edge_f1(target, render):.6f}", "nodes": nodes, "paths": paths,
           "runtime_s": f"{runtime:.3f}", "reference": "clean" if clean.is_file() else "input",
           "enhance": "on" if enhance else "off", "scale": f"{state.image.scale:g}",
           "git_sha": sha}
    for d, r in results.items():
        row[d] = f"{r.score:.4f}"
    row["combined"] = f"{combined(row):.4f}"
    shown = {}
    for d, r in results.items():
        items = []
        for i, spot in enumerate(r.spots[:top]):
            rel = f"crops/{stem}/{d}_{i}.png"
            _write_crop(out / rel, target, svg, k, spot)
            items.append((spot, rel))
        shown[d] = items
    row["_spots"] = shown
    row["_svg"] = f"svg/{stem}.svg"
    return row


def combined(row: dict) -> float:
    return sum(float(row[d] or 0) / NORM[d] for d in DETECTORS)


# ---------------------------------------------------------------- report


def _read_csv(path: Path) -> dict[str, dict]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return {r["image"]: r for r in csv.DictReader(f)}


def _delta(value, before) -> str:
    try:
        d = float(value) - float(before)
    except (TypeError, ValueError):
        return ""
    if abs(d) < 5e-5:
        return ' <span class="d0">(Δ 0)</span>'
    return f' <span class="{"dp" if d > 0 else "dn"}">(Δ {d:+.3g})</span>'


def _cell(row: dict, key: str, before: dict | None) -> str:
    value = row.get(key, "")
    text = html.escape(str(value))
    if before is not None and key in before:
        text += _delta(value, before[key])
    return f"<td>{text}</td>"


def _ranking(rows: list[dict], compare: dict | None) -> str:
    keys = ["combined", *DETECTORS, *METRICS]
    head = "".join(f"<th>{html.escape(k)}</th>" for k in ["image", "status", *keys])
    body = []
    for r in rows:
        before = compare.get(r["image"]) if compare is not None else None
        cells = "".join(_cell(r, k, before) for k in keys)
        anchor = html.escape(quote(r["image"]), quote=True)
        body.append(f'<tr><td class="l"><a href="#{anchor}">{html.escape(r["image"])}</a></td>'
                    f"<td>{html.escape(r['status'])}</td>{cells}</tr>")
    return f"<table><tr>{head}</tr>{''.join(body)}</table>"


def compare_warnings(rows: list[dict], compare: dict | None) -> list[str]:
    """Reasons a --compare is not like for like: no shared images, or different settings."""
    if compare is None:
        return []
    out = []
    shared = [r for r in rows if r["image"] in compare]
    if not shared:
        out.append("0 shared images with the compare run (rows are keyed by the path relative "
                   "to --images; were both runs given the same root?)")
    mine = {r.get("enhance", "") for r in rows} - {""}
    theirs = {r.get("enhance", "") for r in compare.values()} - {""}
    if mine and theirs and mine != theirs:  # a run-level setting: check it even with 0 shared
        out.append(f"enhance differs: this run {'/'.join(sorted(mine))}, "
                   f"compare run {'/'.join(sorted(theirs))}")
    for key in ("scale",):
        differ = [r["image"] for r in shared
                  if r["status"] == "ok" and compare[r["image"]].get("status") == "ok"
                  and str(r.get(key, "")) != str(compare[r["image"]].get(key, ""))]
        if differ:
            out.append(f"{key} differs from the compare run for {len(differ)} image(s), "
                       f"e.g. {differ[0]}")
    return out


def _totals(rows: list[dict], compare: dict | None) -> str:
    ok = [r for r in rows if r["status"] == "ok"]
    if not ok:
        return ""
    keys = ["combined", *DETECTORS, *METRICS]

    def mean(rs, k):
        vals = [float(r[k]) for r in rs if r.get(k) not in ("", None)]
        return sum(vals) / len(vals) if vals else float("nan")

    cells = "".join(f"<td>{mean(ok, k):.4g}</td>" for k in keys)
    out = (f"<table><tr><th></th>{''.join(f'<th>{html.escape(k)}</th>' for k in keys)}</tr>"
           f'<tr><td class="l">mean (this run)</td>{cells}</tr>')
    if compare is not None:
        both = [r for r in ok if compare.get(r["image"], {}).get("status") == "ok"]
        if both:
            before = [compare[r["image"]] for r in both]
            prev = "".join(f"<td>{mean(before, k):.4g}</td>" for k in keys)
            diff = "".join(f"<td>{mean(both, k) - mean(before, k):+.4g}</td>" for k in keys)
            out += (f'<tr><td class="l">mean (compare, {len(both)} shared)</td>{prev}</tr>'
                    f'<tr><td class="l">Δ this − compare</td>{diff}</tr>')
    return out + "</table>"


def _image_section(r: dict) -> str:
    anchor = html.escape(quote(r["image"]), quote=True)
    title = f'<h3 id="{anchor}">{html.escape(r["image"])}</h3>'
    if r["status"] != "ok":
        return f'<section>{title}<pre class="err">{html.escape(r["error"])}</pre></section>'
    metrics = ", ".join(f"{k} {html.escape(str(r[k]))}" for k in ("combined", *METRICS))
    parts = [title, f'<p>{metrics} · <a href="{html.escape(r["_svg"], quote=True)}">SVG</a></p>']
    for d in DETECTORS:
        items = r["_spots"].get(d, [])
        if not items:
            continue
        figs = "".join(
            f'<figure><img src="{html.escape(quote(rel), quote=True)}" loading="lazy" '
            f'alt="{html.escape(d, quote=True)}"><figcaption>#{i + 1} severity {s.severity:.3g} · '
            f"box ({s.box[0]:.0f}, {s.box[1]:.0f})–({s.box[2]:.0f}, {s.box[3]:.0f}) · "
            f"{html.escape(s.note)}</figcaption></figure>"
            for i, (s, rel) in enumerate(items)
        )
        parts.append(f"<h4>{html.escape(d)} (score {html.escape(r[d])})</h4>"
                     f'<div class="row">{figs}</div>')
    return f"<section>{''.join(parts)}</section>"


_STYLE = ("body{font-family:sans-serif;margin:1.5rem}table{border-collapse:collapse}"
          "td,th{border:1px solid #bbb;padding:.2rem .45rem;text-align:right;font-size:.85rem}"
          "td.l{text-align:left}.row{display:flex;flex-wrap:wrap;gap:1rem}figure{margin:0}"
          "figcaption{font-size:.8rem;max-width:44rem}img{max-width:100%}"
          ".dp{color:#b00}.dn{color:#070}.d0{color:#888}.err{color:#b00;white-space:pre-wrap}"
          "section{border-top:2px solid #ddd;margin-top:1.5rem}")
_NOTE = ("<p>Images are sorted worst-first by <em>combined</em> = Σ detector score / NORM "
         "(see <code>eval/run_defects.py</code>). Crops are 5×: source (pixels) | output | output "
         "with path outlines; the blue box is the spot. Spot boxes are in reference px. "
         "Δ is this run minus the compare run; for every detector and for ΔE higher is worse, "
         "for SSIM and edge F1 lower is worse.</p>")


def write_report(rows: list[dict], out: Path, compare: dict | None) -> Path:
    order = sorted(rows, key=lambda r: (r["status"] != "ok", -float(r["combined"] or 0)))
    title = "Defect report"
    warn = "".join(f'<p class="err">Warning: {html.escape(w)}</p>'
                   for w in compare_warnings(order, compare))
    doc = (f'<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>'
           f"<style>{_STYLE}</style></head><body><h1>{title}</h1>{_NOTE}{warn}"
           f"{_totals(order, compare)}<h2>Ranking</h2>{_ranking(order, compare)}"
           f"<h2>Per image (worst first)</h2>{''.join(_image_section(r) for r in order)}"
           "</body></html>")
    dest = out / "report.html"
    dest.write_text(doc, encoding="utf-8")
    return dest


# ---------------------------------------------------------------- run


def run_defects(images: Path, out: Path, enhance: bool = True, top: int = TOP_N,
                compare: Path | None = None) -> tuple[Path, Path]:
    images, out = Path(images), Path(out)
    if not images.is_dir():
        raise NotADirectoryError(f"--images is not a directory: {images}")
    out.mkdir(parents=True, exist_ok=True)
    sha = git_sha()
    rows = []
    csv_path = out / "defects.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for name, path in _images(images):
            try:
                row = analyse_image(name, path, out, enhance, top, sha)
            except Exception as exc:  # noqa: BLE001 - one bad image must not stop the run
                row = dict.fromkeys(COLUMNS, "")
                row.update(image=name, status="error", enhance="on" if enhance else "off",
                           git_sha=sha,
                           error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}")
            writer.writerow(row)  # written as each image finishes, so a long run keeps its rows
            f.flush()
            rows.append(row)
            print(f"{name}: {row['status']} combined={row.get('combined', '')}", flush=True)
    before = _read_csv(compare) if compare else None
    for warning in compare_warnings(rows, before):
        print(f"warning: {warning}", file=sys.stderr)
    return csv_path, write_report(rows, out, before)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, default=Path("eval/dataset/logos"))
    parser.add_argument("--out", type=Path, default=Path("eval/reports/defects"))
    parser.add_argument("--enhance", choices=("on", "off"), default="on")
    parser.add_argument("--top", type=int, default=TOP_N, help="spots shown per detector")
    parser.add_argument("--compare", type=Path, help="another run's defects.csv")
    args = parser.parse_args(argv)
    if not args.images.is_dir():
        parser.error(f"--images is not a directory: {args.images}")
    if args.compare is not None and not args.compare.is_file():
        parser.error(f"--compare is not a file: {args.compare}")
    csv_path, report = run_defects(args.images, args.out, args.enhance == "on", args.top,
                                   args.compare)
    print(f"defects: {csv_path}\nreport: {report}")


if __name__ == "__main__":
    main()
