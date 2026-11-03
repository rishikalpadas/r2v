"""Baseline and reference SVG producers. Each runner maps an image path to SVG text.

A runner returns None when it cannot produce output (tool missing, no reference).
Potrace is only ever used through its external CLI.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from PIL import Image

Runner = Callable[[Path], "str | None"]

SUBPROCESS_TIMEOUT_S = 60
POTRACE_THRESHOLD = 128


def run_ours(path: Path, overrides: dict | None = None) -> str | None:
    try:
        from engine import to_svg, vectorize
    except ModuleNotFoundError as exc:
        if exc.name != "engine":  # a broken engine is an error; only a missing one is a skip
            raise
        raise NotImplementedError("engine is not installed") from exc
    if overrides:
        return to_svg(vectorize(path.read_bytes(), overrides=overrides).doc)
    return to_svg(vectorize(path.read_bytes()).doc)


def run_ours_noenhance(path: Path) -> str | None:
    """`ours` with the Enhance stage switched off, for enhance on/off comparisons."""
    return run_ours(path, overrides={"enhance.enabled": False})


def run_vtracer(path: Path) -> str | None:
    # convert_raw_image_to_svg segfaults in the installed Windows wheel (vtracer 0.6.x,
    # Python 3.14), so go through the file-based API with a normalised PNG.
    import vtracer

    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.png", Path(tmp) / "out.svg"
        Image.open(path).convert("RGBA").save(src, format="PNG")
        vtracer.convert_image_to_svg_py(str(src), str(dst))
        return dst.read_text(encoding="utf-8")


def run_potrace(path: Path) -> str | None:
    exe = shutil.which("potrace")
    if exe is None:
        return None
    img = Image.open(path).convert("RGBA")
    white = Image.new("RGBA", img.size, (255, 255, 255, 255))
    gray = Image.alpha_composite(white, img).convert("L")
    bilevel = gray.point(lambda v: 255 if v >= POTRACE_THRESHOLD else 0, mode="1")
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.pbm", Path(tmp) / "out.svg"
        bilevel.save(src, format="PPM")  # PIL writes mode "1" as PBM
        subprocess.run([exe, "-s", "-o", str(dst), str(src)],
                       timeout=SUBPROCESS_TIMEOUT_S, check=True)
        return dst.read_text(encoding="utf-8")


def run_reference(path: Path) -> str | None:
    ref = path.parent / "reference" / f"{path.stem}.svg"
    return ref.read_text(encoding="utf-8") if ref.is_file() else None


RUNNERS: dict[str, Runner] = {
    "ours": run_ours,
    "ours_noenhance": run_ours_noenhance,
    "vtracer": run_vtracer,
    "potrace": run_potrace,
    "reference": run_reference,
}
