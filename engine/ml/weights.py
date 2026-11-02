"""Model weights manifest, cache location, and sha256-verified downloads."""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_CHUNK = 1 << 20


@dataclass(frozen=True)
class WeightSpec:
    name: str
    url: str
    sha256: str
    license: str
    scale: int
    kind: Literal["sr", "deblock", "denoise"]


# Licenses checked via the GitHub license API on 2026-11-03: xinntao/Real-ESRGAN is
# BSD-3-Clause, jiaxi-jiang/FBCNN is Apache-2.0, cszn/SCUNet is Apache-2.0 (its weight
# file is hosted in the cszn/KAIR release; KAIR itself is MIT).
MANIFEST: dict[str, WeightSpec] = {
    spec.name: spec
    for spec in (
        WeightSpec(
            "realesrgan_x4plus",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
            "BSD-3-Clause", 4, "sr",
        ),
        WeightSpec(
            "realesrgan_x4plus_anime_6b",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
            "f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da",
            "BSD-3-Clause", 4, "sr",
        ),
        WeightSpec(
            "fbcnn_color",
            "https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_color.pth",
            "8b0e4ef23d59cf7ac934a342cb31a17619e4fa4a0b3374a9d78c5174312387e8",
            "Apache-2.0", 1, "deblock",
        ),
        WeightSpec(
            "scunet_color_real_psnr",
            "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth",
            "fa78899ba2caec9d235a900e91d96c689da71c42029230c2028b00f09f809c2e",
            "Apache-2.0", 1, "denoise",
        ),
    )
}


def weights_dir() -> Path:
    """Weights cache directory (not created here)."""
    override = os.environ.get("R2V_WEIGHTS_DIR")
    if override:
        return Path(override)
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(local) / "raster2vector" / "weights"
    return Path.home() / ".cache" / "raster2vector" / "weights"


def _spec(name: str) -> WeightSpec:
    try:
        return MANIFEST[name]
    except KeyError:
        raise KeyError(f"unknown weight {name!r}; known: {', '.join(sorted(MANIFEST))}") from None


def weight_path(name: str) -> Path:
    return weights_dir() / f"{_spec(name).name}.pth"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def verify(name: str) -> bool:
    """True when the cached file exists and its sha256 matches the manifest."""
    path = weight_path(name)
    return path.is_file() and _sha256(path) == _spec(name).sha256


def _download(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out, _CHUNK)


def fetch(name: str) -> Path:
    """Download a weight if needed; verify sha256 and move it into place atomically."""
    spec = _spec(name)
    path = weight_path(name)
    if verify(name):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{spec.name}.", suffix=".part", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        _download(spec.url, tmp)
        actual = _sha256(tmp)
        if actual != spec.sha256:
            raise ValueError(f"sha256 mismatch for {spec.name}: expected {spec.sha256}, got {actual}")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path
