"""Enhance stage on CPU: fake models (nearest-neighbour upscale), fallbacks, pipeline plumbing."""
import hashlib
import io
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from engine import InputError, pipeline, to_svg, vectorize
from engine.loader import load_image
from engine.pipeline import STAGES
from engine.presets import PRESETS, EnhanceParams
from engine.stages import enhance as enh
from engine.stages.enhance import enhance
from engine.types import RasterImage
from eval.metrics import alpha_agreement, render_svg, ssim
from eval.synthetic import aa_shapes, flat_shapes, supersample, to_png_bytes, transparent_logo

MAX_COLORS = 16


def _fake_model(scale, tiling="SUPPORTED", calls=None, name=None):
    def run(rgb):
        if calls is not None:
            calls.append((name, rgb.copy()))
        return np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1).astype(np.float32)

    run.scale = scale
    run.tiling = tiling
    return run


@pytest.fixture
def fake_models(monkeypatch):
    """Patch load_model with nearest-upscale fakes; returns the list of loaded names."""
    loaded = []
    calls = []
    kinds = {"fbcnn_color": (1, "SUPPORTED"), "scunet_color_real_psnr": (1, "DISCOURAGED")}

    def load(name, device):
        loaded.append(name)
        scale, tiling = kinds.get(name, (4, "SUPPORTED"))
        return _fake_model(scale, tiling, calls, name)

    monkeypatch.setattr(enh, "ml_available", lambda: True)
    monkeypatch.setattr(enh, "load_model", load)
    monkeypatch.setattr(enh, "pick_device", lambda pref: "cpu")
    load.calls = calls
    return loaded


def _noisy(h, w, seed=0, alpha=255):
    rng = np.random.default_rng(seed)
    px = np.empty((h, w, 4), np.uint8)
    px[..., :3] = rng.integers(0, 256, (h, w, 3))
    px[..., 3] = alpha
    return px


def _img(px, source_format=None):
    h, w = px.shape[:2]
    return RasterImage(px, (w, h), source_format=source_format)


def _encode(px, fmt, **kwargs):
    buf = io.BytesIO()
    Image.fromarray(px).convert("RGB" if fmt == "JPEG" else "RGBA").save(buf, format=fmt, **kwargs)
    return buf.getvalue()


# ------------------------------------------------------------------ plan Step 1


def test_scale_and_original_size(fake_models):
    out, report = enhance(_img(_noisy(48, 64)), EnhanceParams(), MAX_COLORS)
    assert out.pixels.shape == (192, 256, 4)
    assert out.pixels.dtype == np.uint8
    assert out.scale == 4.0
    assert out.original_size == (64, 48)
    assert report["effective_scale"] == 4
    assert report["fallback_reason"] is None
    assert report["model_used"] == "realesrgan_x4plus"
    assert report["device_used"] == "cpu"
    assert report["final_tile"] == 64  # clamped to the long side by run_tiled


def test_alpha_preserved(fake_models):
    px = transparent_logo()
    out, report = enhance(_img(px), EnhanceParams())  # no max_colors: never skipped
    assert report["fallback_reason"] is None
    nearest = np.repeat(np.repeat(px, 4, axis=0), 4, axis=1)
    assert alpha_agreement(nearest, out.pixels) >= 0.98
    assert set(np.unique(out.pixels[..., 3])) <= {0, 255}
    assert (out.pixels[:16, :16, 3] == 0).all()  # transparent corner stays transparent


def test_transparent_rgb_filled_before_ml(fake_models):
    # opaque blue disc; fully transparent pixels carry garbage green that must not reach the model
    px = transparent_logo(64)
    px[px[..., 3] == 0, :3] = (0, 255, 0)
    px[0, 0, 3] = 1  # one barely-visible pixel keeps its own colour
    out, _ = enhance(_img(px), EnhanceParams())
    (_, fed), = enh.load_model.calls
    transparent = px[..., 3] == 0
    blue = np.array([30, 90, 200]) / 255
    assert np.allclose(fed[transparent & (np.arange(64)[None, :] > 40)], blue, atol=1e-6)
    assert np.allclose(fed[0, 0], [0, 1, 0])
    # no green bleeds into the visible output
    vis = out.pixels[..., 3] == 255
    assert out.pixels[vis][:, 1].max() <= 90


def test_tiny_image_bicubic(fake_models):
    out, report = enhance(_img(_noisy(8, 8)), EnhanceParams(), MAX_COLORS)
    assert report["fallback_reason"] == "too small"
    assert report["model_used"] == "bicubic"
    assert out.pixels.shape == (32, 32, 4)
    assert out.scale == 4.0
    assert fake_models == []


@pytest.mark.parametrize("h, w", [(1, 1), (1, 40), (15, 300), (16, 15)])
def test_tiny_and_thin_images_never_crash(fake_models, h, w):
    out, report = enhance(_img(_noisy(h, w)), EnhanceParams(), None)  # 1x1 would be exact colour
    assert out.pixels.shape == (h * 4, w * 4, 4)
    assert report["fallback_reason"] == "too small"


def test_large_image_capped(fake_models):
    out, report = enhance(_img(_noisy(100, 4096)), EnhanceParams(), MAX_COLORS)
    assert report["effective_scale"] == 2
    assert report["cap_reason"] == "max_output_side"
    assert out.scale == 2.0
    assert out.pixels.shape == (200, 8192, 4)
    assert out.original_size == (4096, 100)
    assert report["fallback_reason"] is None


def test_capped_scale_downsamples_model_output(fake_models):
    px = _noisy(40, 60)
    px[:20] = (200, 30, 30, 255)
    params = EnhanceParams(max_output_side=180)  # 180 / 60 = 3
    out, report = enhance(_img(px), params, MAX_COLORS)
    assert report["effective_scale"] == 3
    assert out.pixels.shape == (120, 180, 4)
    assert (np.abs(out.pixels[:50, :, :3].astype(int) - (200, 30, 30)) <= 1).all()


def test_fallback_when_ml_unavailable(fake_models, monkeypatch):
    monkeypatch.setattr(enh, "ml_available", lambda: False)
    out, report = enhance(_img(_noisy(32, 32)), EnhanceParams(), MAX_COLORS)
    assert report["fallback_reason"] == "ml unavailable"
    assert report["model_used"] == "bicubic"
    assert report["device_used"] is None
    assert out.pixels.shape == (128, 128, 4)
    assert fake_models == []
    result = vectorize(to_png_bytes(aa_shapes(0)))
    assert result.doc.width == 128


@pytest.mark.parametrize(
    "exc, reason",
    [
        (RuntimeError("cuda kernel failed"), "model error (RuntimeError)"),
        (MemoryError("out of memory at tile 64"), "model error (MemoryError)"),
        (FileNotFoundError("nope"), "weights missing"),
        (ValueError("bad hash"), "model error (ValueError)"),
        (KeyError("unknown"), "model error (KeyError)"),
    ],
)
def test_fallback_on_model_exception(fake_models, monkeypatch, exc, reason):
    def boom(name, device):
        raise exc

    monkeypatch.setattr(enh, "load_model", boom)
    out, report = enhance(_img(_noisy(32, 32)), EnhanceParams(), MAX_COLORS)
    assert report["fallback_reason"].startswith(reason)
    assert report["model_used"] == "bicubic"
    assert out.pixels.shape == (128, 128, 4)


def test_fallback_on_exception_inside_model(fake_models, monkeypatch):
    def load(name, device):
        def run(rgb):
            raise RuntimeError("boom")

        run.scale, run.tiling = 4, "SUPPORTED"
        return run

    monkeypatch.setattr(enh, "load_model", load)
    out, report = enhance(_img(_noisy(32, 32)), EnhanceParams(), MAX_COLORS)
    assert report["fallback_reason"] == "model error (RuntimeError)"
    assert out.scale == 4.0


def test_missing_weights_fall_back_deterministically():
    # conftest points R2V_WEIGHTS_DIR at an empty dir for non-gpu tests
    out, report = enhance(_img(_noisy(32, 32)), EnhanceParams(device="cpu"), MAX_COLORS)
    if enh.ml_available():
        assert report["fallback_reason"].startswith("weights missing")
    else:
        assert report["fallback_reason"] == "ml unavailable"
    assert out.pixels.shape == (128, 128, 4)


# Pre-M4 digests: sha256(to_svg(vectorize(to_png_bytes(px)).doc)) on the code before Task 4.
PRE_M4 = [
    (lambda: flat_shapes(0), "dfbd4f6332566d6ff0af6499ba4a570b6bb1fc228dd0d24d89cfb610b4ba7830"),
    (lambda: aa_shapes(0, 256, 4), "c971f9d6bd5d9e940c2d791d9ab9d7bc2c7649959a4c23ec3f9dbb440a9f0269"),
]


@pytest.mark.parametrize("make, digest", PRE_M4, ids=["flat_shapes0", "aa_shapes0"])
def test_enhance_disabled_is_m3_identical(make, digest):
    svg = to_svg(vectorize(to_png_bytes(make()), overrides={"enhance.enabled": False}).doc)
    assert hashlib.sha256(svg.encode()).hexdigest() == digest


def test_disabled_returns_input(fake_models):
    image = _img(_noisy(32, 32))
    out, report = enhance(image, EnhanceParams(enabled=False), MAX_COLORS)
    assert out is image
    assert report["skipped_reason"] == "disabled"
    assert report["effective_scale"] == 1
    assert fake_models == []


def test_jpeg_only_deblock(fake_models):
    enhance(_img(_noisy(32, 32), "PNG"), EnhanceParams(), MAX_COLORS)
    assert fake_models == ["realesrgan_x4plus"]
    fake_models.clear()
    _, report = enhance(_img(_noisy(32, 32), "JPEG"), EnhanceParams(), MAX_COLORS)
    assert fake_models == ["fbcnn_color", "realesrgan_x4plus"]
    assert report["model_used"] == "fbcnn_color+realesrgan_x4plus"
    fake_models.clear()
    enhance(_img(_noisy(32, 32), "JPEG"), EnhanceParams(deblock=False), MAX_COLORS)
    assert fake_models == ["realesrgan_x4plus"]


def test_order_deblock_denoise_sr(fake_models):
    _, report = enhance(_img(_noisy(32, 32), "JPEG"), EnhanceParams(denoise=True), MAX_COLORS)
    assert fake_models == ["fbcnn_color", "scunet_color_real_psnr", "realesrgan_x4plus"]
    assert [name for name, _ in enh.load_model.calls] == fake_models


def test_discouraged_tiling_gets_larger_tile(fake_models, monkeypatch):
    tiles = []
    real = enh.run_tiled

    def spy(model_fn, rgb, scale, tile, **kwargs):
        tiles.append((model_fn.tiling, tile))
        return real(model_fn, rgb, scale, tile, **kwargs)

    monkeypatch.setattr(enh, "run_tiled", spy)
    enhance(_img(_noisy(32, 32)), EnhanceParams(denoise=True, tile=256), MAX_COLORS)
    assert tiles == [("DISCOURAGED", 512), ("SUPPORTED", 256)]


def test_model_choice_flat_vs_general(fake_models):
    flat = aa_shapes(1, 64, 2)  # a few blend colours, still <= 64 after 5-bit quantization
    _, report = enhance(_img(flat), EnhanceParams(), None)
    assert report["model_used"] == "realesrgan_x4plus_anime_6b"
    _, report = enhance(_img(_noisy(64, 64)), EnhanceParams(), None)
    assert report["model_used"] == "realesrgan_x4plus"


def test_pick_device_receives_preference(fake_models, monkeypatch):
    seen = []
    monkeypatch.setattr(enh, "pick_device", lambda pref: seen.append(pref) or "cpu")
    _, report = enhance(_img(_noisy(32, 32)), EnhanceParams(device="cuda"), MAX_COLORS)
    assert seen == ["cuda"]
    assert report["device_used"] == "cpu"


def test_pipeline_end_to_end_cpu_fake(fake_models):
    px = aa_shapes(0)
    result = vectorize(to_png_bytes(px))
    assert (result.doc.width, result.doc.height) == (128, 128)
    report = result.reports[0]
    assert report.stage == "enhance"
    assert report.params["effective_scale"] == 4
    assert report.params["fallback_reason"] is None
    svg = to_svg(result.doc)
    assert ssim(px, render_svg(svg, 128, 128)) >= 0.95


# ------------------------------------------------------------------ exact colour, scale 1


def test_exact_colour_input_is_skipped(fake_models):
    image = _img(flat_shapes(0))
    out, report = enhance(image, EnhanceParams(), MAX_COLORS)
    assert out is image
    assert report["skipped_reason"] == "exact colours"
    assert report["effective_scale"] == 1
    assert report["model_used"] is None
    assert fake_models == []


def test_exact_colour_skip_follows_max_colors(fake_models):
    image = _img(flat_shapes(0, n_colors=4))
    _, report = enhance(image, EnhanceParams(), 3)
    assert report["skipped_reason"] is None
    assert report["effective_scale"] == 4


def test_partial_alpha_is_not_exact(fake_models):
    px = supersample(lambda n: transparent_logo(n), 64, 4)  # AA alpha edge, one RGB colour
    _, report = enhance(_img(px), EnhanceParams(), MAX_COLORS)
    assert report["skipped_reason"] is None
    assert report["effective_scale"] == 4


def test_fully_transparent_input(fake_models):
    px = np.zeros((32, 32, 4), np.uint8)
    out, report = enhance(_img(px), EnhanceParams(), MAX_COLORS)
    assert report["skipped_reason"] == "exact colours"
    out, report = enhance(_img(px), EnhanceParams(), None)
    assert out.pixels.shape == (128, 128, 4)
    assert (out.pixels[..., 3] == 0).all()


@pytest.mark.parametrize("params", [EnhanceParams(scale=1), EnhanceParams(max_output_side=60)])
def test_effective_scale_one_returns_input(fake_models, params):
    image = _img(_noisy(32, 40))
    out, report = enhance(image, params, MAX_COLORS)
    assert out is image
    assert report["effective_scale"] == 1
    assert report["skipped_reason"] == "scale 1"
    assert fake_models == []


# ------------------------------------------------------------------ params, presets, pipeline


def test_logo_preset_runs_enhance_first():
    assert PRESETS["logo"].stages == ("enhance", "segment", "trace", "fit", "export")
    assert PRESETS["logo"].enhance == EnhanceParams()
    assert "enhance" in STAGES


def test_enhance_params_defaults():
    p = EnhanceParams()
    assert (p.enabled, p.scale, p.deblock, p.denoise, p.tile, p.device, p.max_output_side) == (
        True, 4, True, False, 512, "auto", 8192)
    assert p.max_output_pixels == 2**24
    assert (p.sr_model_flat, p.sr_model_general) == ("realesrgan_x4plus_anime_6b", "realesrgan_x4plus")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enabled": 1},
        {"deblock": "yes"},
        {"denoise": None},
        {"scale": 0},
        {"scale": 5},
        {"scale": 2.0},
        {"scale": True},
        {"tile": 32},
        {"tile": 512.0},
        {"device": "gpu"},
        {"max_output_side": 0},
        {"max_output_side": 1.5},
        {"max_output_pixels": 0},
        {"max_output_pixels": 2.0**24},
        {"sr_model_flat": ""},
        {"sr_model_general": 3},
    ],
)
def test_enhance_params_validation(kwargs):
    with pytest.raises(ValueError):
        EnhanceParams(**kwargs)


def test_enhance_override_plumbing_and_report_merge(fake_models):
    result = vectorize(to_png_bytes(aa_shapes(0, 64)), overrides={"enhance.tile": 128})
    report = result.reports[0]
    assert report.stage == "enhance"
    assert report.params["tile"] == 128  # EnhanceParams fields
    for key in ("model_used", "fallback_reason", "final_tile", "effective_scale", "device_used",
                "skipped_reason", "cap_reason"):
        assert key in report.params
    assert report.params["final_tile"] == 64
    assert report.params["device"] == "auto"  # the EnhanceParams field survives the merge
    assert report.params["device_used"] == "cpu"
    assert "model_used" not in result.reports[1].params  # merged into enhance only


@pytest.mark.parametrize("overrides", [{"enhance.tile": 1}, {"enhance.nope": 1}, {"enhance.device": "tpu"}])
def test_bad_enhance_overrides_raise_input_error(overrides):
    with pytest.raises(InputError):
        vectorize(to_png_bytes(flat_shapes(0, 32)), overrides=overrides)


def test_exact_colour_skip_through_pipeline(fake_models):
    result = vectorize(to_png_bytes(flat_shapes(0)))
    assert result.reports[0].params["skipped_reason"] == "exact colours"
    assert fake_models == []


def test_source_format_from_loader():
    px = aa_shapes(0, 32)
    assert load_image(_encode(px, "PNG")).source_format == "PNG"
    assert load_image(_encode(px, "JPEG", quality=90)).source_format == "JPEG"
    assert RasterImage(px, (32, 32)).source_format is None


def test_enhance_keeps_source_format(fake_models):
    out, _ = enhance(_img(_noisy(32, 32), "JPEG"), EnhanceParams(), MAX_COLORS)
    assert out.source_format == "JPEG"


def test_enhanced_pixels_feed_segment(fake_models, monkeypatch):
    # the enhanced image is what segment sees: 4x the original size, scale 4
    seen = []
    real = pipeline.segment

    def spy(image, params):
        seen.append((image.pixels.shape, image.scale))
        return real(image, params)

    monkeypatch.setattr(pipeline, "segment", spy)
    vectorize(to_png_bytes(aa_shapes(0, 64)))
    assert seen == [((256, 256, 4), 4.0)]


# ------------------------------------------------------------------ review fixes


@pytest.mark.parametrize(
    "w, h, params, scale, cap",
    [
        (64, 48, EnhanceParams(), 4, None),
        (1600, 1411, EnhanceParams(), 2, "max_output_pixels"),  # happiness.jpg
        (2048, 2048, EnhanceParams(), 2, "max_output_pixels"),
        (2049, 2048, EnhanceParams(), 1, "max_output_pixels"),
        (4096, 100, EnhanceParams(), 2, "max_output_side"),
        (100, 100, EnhanceParams(max_output_pixels=100 * 100 * 9), 3, "max_output_pixels"),
        (100, 100, EnhanceParams(max_output_pixels=100 * 100 * 16), 4, None),
        (100, 100, EnhanceParams(max_output_pixels=100 * 100 * 4 - 1), 1, "max_output_pixels"),
        (100, 100, EnhanceParams(scale=2, max_output_pixels=100), 1, "max_output_pixels"),
        (100, 100, EnhanceParams(scale=2), 2, None),
    ],
)
def test_effective_scale_caps(w, h, params, scale, cap):
    assert enh.effective_scale(params, w, h) == (scale, cap)


def test_area_cap_reported(fake_models):
    params = EnhanceParams(max_output_pixels=40 * 40 * 9)
    out, report = enhance(_img(_noisy(40, 40)), params, MAX_COLORS)
    assert report["effective_scale"] == 3
    assert report["cap_reason"] == "max_output_pixels"
    assert out.pixels.shape == (120, 120, 4)


@pytest.mark.parametrize("scale", [2, 3])
def test_pipeline_end_to_end_capped_scales(fake_models, scale):
    px = aa_shapes(0)
    result = vectorize(to_png_bytes(px), overrides={"enhance.scale": scale})
    assert (result.doc.width, result.doc.height) == (128, 128)
    assert result.reports[0].params["effective_scale"] == scale
    assert ssim(px, render_svg(to_svg(result.doc), 128, 128)) >= 0.95


def test_real_oom_floor_falls_back(fake_models, monkeypatch):
    def load(name, device):
        def run(rgb):
            raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")

        run.scale, run.tiling = 4, "SUPPORTED"
        return run

    monkeypatch.setattr(enh, "load_model", load)  # the real run_tiled halves down to its floor
    out, report = enhance(_img(_noisy(200, 200)), EnhanceParams(), MAX_COLORS)
    assert report["fallback_reason"] == "model error (MemoryError)"
    assert report["model_used"] == "bicubic"
    assert out.pixels.shape == (800, 800, 4)


@pytest.mark.parametrize("where", ["ml_available", "pick_device"])
def test_device_probe_errors_fall_back(fake_models, monkeypatch, where):
    def boom(*args):
        raise OSError("broken driver")

    monkeypatch.setattr(enh, where, boom)
    out, report = enhance(_img(_noisy(32, 32)), EnhanceParams(), MAX_COLORS)
    assert report["fallback_reason"] == "model error (OSError)"
    assert out.pixels.shape == (128, 128, 4)


def test_read_only_input_unchanged(fake_models):
    px = transparent_logo(64)
    px[px[..., 3] == 0, :3] = (0, 255, 0)
    px[20:40, 20:40, 3] = 200  # partial alpha: not exact colour
    before = px.copy()
    px.flags.writeable = False
    image = _img(px)
    for source in ("PNG", "JPEG"):
        enhance(replace(image, source_format=source), EnhanceParams(denoise=True), MAX_COLORS)
    enhance(image, EnhanceParams(), MAX_COLORS)
    assert np.array_equal(px, before)


def test_cached_model_not_mutated(fake_models, monkeypatch):
    model = _fake_model(4)
    model.tiling = "SUPPORTED"
    attrs = dict(vars(model))
    monkeypatch.setattr(enh, "load_model", lambda name, device: model)
    enhance(_img(_noisy(40, 60)), EnhanceParams(max_output_side=180), MAX_COLORS)  # scale 3 wrapper
    enhance(_img(_noisy(40, 60)), EnhanceParams(), MAX_COLORS)
    assert vars(model) == attrs


def test_final_tile_is_sr_tile(fake_models):
    _, report = enhance(_img(_noisy(32, 32), "JPEG"), EnhanceParams(denoise=True), MAX_COLORS)
    assert report["final_tile"] == 64  # SR model's tile (the restoration steps used 64 too)


def test_colour_correction_restores_low_frequency_colour(fake_models, monkeypatch):
    # an SR model that shifts every colour by +0.1: the correction must pull it back
    def load(name, device):
        def run(rgb):
            up = np.repeat(np.repeat(rgb, 4, axis=0), 4, axis=1)
            return np.clip(up + 0.1, 0, 1).astype(np.float32)

        run.scale, run.tiling = 4, "SUPPORTED"
        return run

    monkeypatch.setattr(enh, "load_model", load)
    px = aa_shapes(2, 64)
    out, _ = enhance(_img(px), EnhanceParams(), MAX_COLORS)
    nearest = np.repeat(np.repeat(px[..., :3], 4, axis=0), 4, axis=1).astype(int)
    interior = (slice(40, 216), slice(40, 216))
    assert np.abs(out.pixels[..., :3].astype(int) - nearest)[interior].mean() < 6  # 0.1 = 25.5
