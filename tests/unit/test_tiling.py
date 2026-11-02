from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from engine.ml.tiling import TiledResult, run_tiled


class OutOfMemoryError(RuntimeError):
    """Stand-in for torch.cuda.OutOfMemoryError; only the type name matters."""


def _nearest(scale):
    def model(x):
        return np.repeat(np.repeat(x, scale, axis=0), scale, axis=1)
    return model


def _rand(h, w, seed=0):
    return np.random.default_rng(seed).random((h, w, 3), dtype=np.float32)


def test_identity_matches_whole_image():
    rgb = _rand(300, 200)
    res = run_tiled(_nearest(2), rgb, 2, tile=64)
    assert isinstance(res, TiledResult)
    assert res.image.dtype == np.float32
    assert res.image.shape == (600, 400, 3)
    np.testing.assert_array_equal(res.image, _nearest(2)(rgb))
    assert res.final_tile == 64


def test_no_seam_on_flat_color():
    rgb = np.full((300, 200, 3), 0.5, dtype=np.float32)
    calls = []

    def model(x):
        calls.append(x.shape)
        offset = (1 if len(calls) % 2 else -1) / 255
        return _nearest(2)(x) + np.float32(offset)

    out = run_tiled(model, rgb, 2, tile=64, overlap=16).image
    assert len(calls) > 4
    step = max(np.abs(np.diff(out, axis=0)).max(), np.abs(np.diff(out, axis=1)).max())
    assert step <= 2 / 255
    assert step < 0.5 / 255  # feathered, not a hard 2-level step


def test_oom_backoff():
    rgb = _rand(300, 200)
    sizes = []

    def model(x):
        sizes.append(x.shape[:2])
        if max(x.shape[:2]) > 100:
            raise OutOfMemoryError("CUDA out of whack")
        return _nearest(4)(x)

    res = run_tiled(model, rgb, 4, tile=512)
    assert res.final_tile == 64
    assert all(max(s) <= 64 for s in sizes[-10:])
    np.testing.assert_array_equal(res.image, _nearest(4)(rgb))


def test_oom_message_variant():
    rgb = _rand(150, 150)

    def model(x):
        if max(x.shape[:2]) > 100:
            raise RuntimeError("CUDA error: out of memory")
        return _nearest(1)(x)

    res = run_tiled(model, rgb, 1, tile=256, overlap=8)
    assert res.final_tile == 64
    np.testing.assert_array_equal(res.image, rgb)


def test_oom_empties_cuda_cache_when_torch_loaded(monkeypatch):
    emptied = []
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda: emptied.append(1)))
    monkeypatch.setitem(sys.modules, "torch", fake)

    def model(x):
        if max(x.shape[:2]) > 64:
            raise OutOfMemoryError("boom")
        return x

    res = run_tiled(model, _rand(100, 100), 1, tile=128)
    assert res.final_tile == 64
    assert emptied == [1]


def test_oom_survives_empty_cache_failure(monkeypatch):
    def broken():
        raise RuntimeError("CUDA driver gone")

    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=broken))
    monkeypatch.setitem(sys.modules, "torch", fake)

    def model(x):
        if max(x.shape[:2]) > 64:
            raise OutOfMemoryError("boom")
        return x

    assert run_tiled(model, _rand(100, 100), 1, tile=128).final_tile == 64


def test_oom_without_torch_does_not_import_it(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    def model(x):
        if max(x.shape[:2]) > 64:
            raise OutOfMemoryError("boom")
        return x

    run_tiled(model, _rand(100, 100), 1, tile=128)
    assert "torch" not in sys.modules


def test_oom_floor_raises_memoryerror():
    def model(x):
        raise OutOfMemoryError("always")

    with pytest.raises(MemoryError) as info:
        run_tiled(model, _rand(200, 200), 2, tile=256, min_tile=64)
    assert isinstance(info.value.__cause__, OutOfMemoryError)


def test_non_oom_exception_propagates_unchanged():
    err = KeyError("nope")

    def model(x):
        raise err

    with pytest.raises(KeyError) as info:
        run_tiled(model, _rand(50, 50), 2)
    assert info.value is err


def test_image_smaller_than_tile():
    rgb = _rand(40, 30)
    calls = []

    def model(x):
        calls.append(x.shape)
        return _nearest(4)(x)

    res = run_tiled(model, rgb, 4, tile=512)
    assert calls == [(40, 30, 3)]
    np.testing.assert_array_equal(res.image, _nearest(4)(rgb))
    assert res.final_tile == 512


def test_tiles_are_full_size_when_image_allows():
    calls = []

    def model(x):
        calls.append(x.shape[:2])
        return x

    run_tiled(model, _rand(130, 65), 1, tile=64, overlap=8)
    assert set(calls) == {(64, 64)}


@pytest.mark.parametrize(
    ("h", "w", "tile", "overlap"),
    [(1, 1, 64, 32), (65, 130, 64, 32), (65, 130, 64, 40), (64, 64, 64, 32), (200, 3, 64, 8)],
)
def test_odd_sizes(h, w, tile, overlap):
    rgb = _rand(h, w)
    res = run_tiled(_nearest(3), rgb, 3, tile=tile, overlap=overlap)
    np.testing.assert_array_equal(res.image, _nearest(3)(rgb))


@pytest.mark.parametrize(
    "rgb",
    [np.zeros((10, 10), np.float32), np.zeros((10, 10, 4), np.float32), np.zeros((0, 10, 3), np.float32)],
)
def test_rejects_bad_input_shape(rgb):
    with pytest.raises(ValueError):
        run_tiled(_nearest(1), rgb, 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scale": 0},
        {"scale": 1.5},
        {"scale": True},
        {"scale": 2, "tile": 0},
        {"scale": 2, "overlap": -1},
        {"scale": 2, "min_tile": 0},
        {"scale": 2, "tile": 32, "min_tile": 64},
    ],
)
def test_rejects_bad_params(kwargs):
    with pytest.raises(ValueError):
        run_tiled(_nearest(2), _rand(10, 10), **kwargs)


def test_rejects_bad_model_output_shape():
    with pytest.raises(ValueError):
        run_tiled(_nearest(1), _rand(10, 10), 2)
