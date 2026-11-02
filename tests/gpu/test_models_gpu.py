"""Real-weight smoke tests on CUDA. Run with `pytest -m gpu -s` to see the runtimes."""
import time

import numpy as np
import pytest

from engine.ml import models, weights

pytestmark = pytest.mark.gpu


def _load(name):
    if not weights.weight_path(name).is_file():
        pytest.skip(f"weight {name} missing; run python -m engine.ml.fetch_weights {name}")
    return models.load_model(name, "cuda")


def _image(h, w):
    # smooth gradients plus a hard edge, so the models have something to work on
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    rgb = np.stack([x / max(w - 1, 1), y / max(h - 1, 1), ((x + y) % 16 < 8).astype(np.float32)], axis=-1)
    return np.ascontiguousarray(rgb, dtype=np.float32)


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


def test_realesrgan_runs():
    fn = _load("realesrgan_x4plus")
    assert fn.scale == 4
    _check(_timed("realesrgan_x4plus", fn, _image(64, 64)), (256, 256, 3))


def test_realesrgan_anime_runs():
    fn = _load("realesrgan_x4plus_anime_6b")
    assert fn.scale == 4
    _check(_timed("realesrgan_x4plus_anime_6b", fn, _image(64, 64)), (256, 256, 3))


def test_fbcnn_runs():
    fn = _load("fbcnn_color")
    assert fn.scale == 1
    _check(_timed("fbcnn_color", fn, _image(64, 64)), (64, 64, 3))


def test_scunet_pads_below_minimum():
    fn = _load("scunet_color_real_psnr")
    assert fn.scale == 1
    _check(_timed("scunet_color_real_psnr", fn, _image(32, 32)), (32, 32, 3))
