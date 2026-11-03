"""Real-weight smoke tests on CUDA. Run with `pytest -m gpu -s` to see the runtimes."""
import time
from pathlib import Path

import numpy as np
import pytest

from engine.ml import models, weights

pytestmark = pytest.mark.gpu

REPO_ROOT = Path(__file__).resolve().parents[2]
LOGO = REPO_ROOT / "eval" / "dataset" / "logos" / "happiness.jpg"  # git-ignored

# name -> (scale, tiling hint)
MODELS = {
    "realesrgan_x4plus": (4, "SUPPORTED"),
    "realesrgan_x4plus_anime_6b": (4, "SUPPORTED"),
    "fbcnn_color": (1, "SUPPORTED"),
    "scunet_color_real_psnr": (1, "DISCOURAGED"),
}


def _load(name, device="cuda"):
    if not weights.weight_path(name).is_file():
        pytest.skip(f"weight {name} missing; run python -m engine.ml.fetch_weights {name}")
    return models.load_model(name, device)


def _image(h, w):
    # smooth gradients plus a hard edge, so the models have something to work on
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    rgb = np.stack([x / max(w - 1, 1), y / max(h - 1, 1), ((x + y) % 16 < 8).astype(np.float32)], axis=-1)
    return np.ascontiguousarray(rgb, dtype=np.float32)


def _logo_crop(size=128):
    if not LOGO.is_file():
        pytest.skip(f"{LOGO} missing (git-ignored eval dataset)")
    from PIL import Image

    with Image.open(LOGO) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    h, w = rgb.shape[:2]
    if min(h, w) < size:
        pytest.skip(f"{LOGO.name} is smaller than {size} px")
    y, x = (h - size) // 2, (w - size) // 2
    return np.ascontiguousarray(rgb[y:y + size, x:x + size])


def _timed(name, fn, rgb):
    import torch

    fn(rgb)  # warm-up: cuDNN autotune and kernel loading
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(rgb)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    print(f"\n{name}: {rgb.shape[1]}x{rgb.shape[0]} -> {out.shape[1]}x{out.shape[0]} in {ms:.1f} ms")
    return out


def _check(out, shape):
    assert out.shape == shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0


@pytest.mark.parametrize("name", list(MODELS))
def test_model_runs(name):
    scale, tiling = MODELS[name]
    fn = _load(name)
    assert fn.scale == scale
    assert fn.tiling == tiling
    side = 32 if name == "scunet_color_real_psnr" else 64  # 32 < SCUNet minimum 40: exercises padding
    _check(_timed(name, fn, _image(side, side)), (side * scale, side * scale, 3))


def test_realesrgan_odd_size():
    fn = _load("realesrgan_x4plus")
    _check(_timed("realesrgan_x4plus odd", fn, _image(37, 53)), (148, 212, 3))


@pytest.mark.parametrize("name", list(MODELS))
def test_fp16_matches_fp32(name):
    rgb = _logo_crop()
    fast, ref = _load(name, "cuda"), _load(name, "cpu")  # cuda runs fp16, cpu is the fp32 copy
    diff = float(np.abs(fast(rgb) - ref(rgb)).max())
    print(f"\n{name}: fp16 vs fp32 max abs diff {diff:.4f}")
    assert diff < 0.03
