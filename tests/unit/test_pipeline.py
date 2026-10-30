from dataclasses import replace

import pytest

from engine import InputError, StageError, presets, to_svg, vectorize
from engine.presets import PRESETS
from eval.synthetic import circle, to_png_bytes


def test_vectorize_invalid_bytes_raises_input_error():
    with pytest.raises(InputError):
        vectorize(b"not an image")


def test_unknown_preset_raises_input_error():
    with pytest.raises(InputError):
        vectorize(to_png_bytes(circle()), preset="nope")


def test_invalid_override_raises_input_error():
    with pytest.raises(InputError):
        vectorize(to_png_bytes(circle()), overrides={"segment.nope": 1})


def test_stage_failure_wrapped(monkeypatch):
    def boom(*args, **kwargs):
        raise ValueError("secret detail")

    monkeypatch.setattr("engine.pipeline.segment", boom)
    with pytest.raises(StageError) as info:
        vectorize(to_png_bytes(circle()))
    assert info.value.stage == "segment"
    assert "secret detail" not in str(info.value)
    assert isinstance(info.value.__cause__, ValueError)


def test_stage_error_passes_through(monkeypatch):
    def boom(regions):
        raise StageError("trace", "own reason")

    monkeypatch.setattr("engine.pipeline.trace", boom)
    with pytest.raises(StageError) as info:
        vectorize(to_png_bytes(circle()))
    assert (info.value.stage, info.value.reason) == ("trace", "own reason")


def test_reports_per_stage():
    result = vectorize(to_png_bytes(circle()))
    assert [r.stage for r in result.reports] == list(PRESETS["logo"].stages)
    assert all(r.duration_s >= 0 for r in result.reports)
    assert result.reports[0].params["max_colors"] == 16


def test_auto_equals_logo():
    data = to_png_bytes(circle())
    assert to_svg(vectorize(data).doc) == to_svg(vectorize(data, preset="logo").doc)


def test_invalid_stage_order_is_stage_error(monkeypatch):
    bad = replace(PRESETS["logo"], stages=("trace", "export"))
    monkeypatch.setitem(presets.PRESETS, "bad", bad)
    with pytest.raises(StageError) as info:
        vectorize(to_png_bytes(circle()), preset="bad")
    assert info.value.stage == "trace"


def test_unknown_stage_is_stage_error(monkeypatch):
    bad = replace(PRESETS["logo"], stages=("segment", "warp"))
    monkeypatch.setitem(presets.PRESETS, "bad", bad)
    with pytest.raises(StageError) as info:
        vectorize(to_png_bytes(circle()), preset="bad")
    assert info.value.stage == "warp"
