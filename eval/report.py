"""Render results.csv into a static report.html (all text HTML-escaped)."""
from __future__ import annotations

import csv
import html
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

METRICS = ("ssim", "delta_e", "edge_f1", "nodes", "paths", "runtime_s")


def _mean(values: list[float]) -> str:
    return f"{sum(values) / len(values):.4f}" if values else "-"


def _summary_table(rows: list[dict]) -> str:
    groups: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["status"] == "ok":
            groups[r["category"]][r["engine"]].append(r)
    head = "".join(f"<th>{html.escape(m)}</th>" for m in METRICS)
    out = []
    for category in sorted(groups):
        body = []
        for engine in sorted(groups[category]):
            ok = groups[category][engine]
            cells = "".join(f"<td>{_mean([float(r[m]) for r in ok])}</td>" for m in METRICS)
            ref = "/".join(sorted({r.get("reference") or "input" for r in ok}))
            body.append(f"<tr><td>{html.escape(engine)}</td><td>{len(ok)}</td>{cells}"
                        f"<td>{html.escape(ref)}</td></tr>")
        out.append(f"<h2>{html.escape(category)}</h2><table><tr><th>engine</th><th>n</th>"
                   f"{head}<th>reference</th></tr>{''.join(body)}</table>")
    return "".join(out) or "<p>No successful results.</p>"


def _img(rel: str, label: str) -> str:
    src = html.escape(quote(rel, safe="/"), quote=True)
    alt = html.escape(label, quote=True)
    return (f'<figure><img src="{src}" alt="{alt}" width="160">'
            f"<figcaption>{html.escape(label)}</figcaption></figure>")


def _per_image(rows: list[dict]) -> str:
    by_image: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_image[(r["category"], r["image"])].append(r)
    out = []
    for (category, image), group in sorted(by_image.items()):
        stem = Path(image).stem
        figs = [_img(f"renders/original/{category}/{stem}.png", "original")]
        if any(r.get("reference") == "clean" for r in group):
            figs.append(_img(f"renders/clean/{category}/{stem}.png", "clean target"))
        for r in group:
            if r["status"] == "ok":
                figs.append(_img(f"renders/{r['engine']}/{category}/{stem}.png", r["engine"]))
            else:
                figs.append(f"<figure><figcaption>{html.escape(r['engine'])}: "
                            f"{html.escape(r['status'])}</figcaption></figure>")
        out.append(f"<section><h3>{html.escape(category)} / {html.escape(image)}</h3>"
                   f'<div class="row">{"".join(figs)}</div></section>')
    return "".join(out)


_NOTE = ("<p>Metrics compare each render with its <em>reference</em>: the input image, or the "
         "clean source (<code>&lt;category&gt;/clean/&lt;stem&gt;.png</code>, rendered at its "
         "size) when the input is a degraded copy of it.</p>")


def write_report(results_csv: Path) -> Path:
    results_csv = Path(results_csv)
    with results_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    doc = ('<!doctype html><html><head><meta charset="utf-8"><title>Benchmark report</title>'
           "<style>body{font-family:sans-serif;margin:1.5rem}table{border-collapse:collapse}"
           "td,th{border:1px solid #bbb;padding:.2rem .5rem;text-align:right}"
           ".row{display:flex;flex-wrap:wrap;gap:1rem}figure{margin:0}</style></head><body>"
           f"<h1>Benchmark report</h1>{_NOTE}{_summary_table(rows)}<h2>Per image</h2>"
           f"{_per_image(rows)}</body></html>")
    dest = results_csv.parent / "report.html"
    dest.write_text(doc, encoding="utf-8")
    return dest
