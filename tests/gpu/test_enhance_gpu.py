"""Enhance stage with the real models on degraded logos. Run with `pytest -m gpu -s` for the table."""
import io
import time

import numpy as np
import pytest
from PIL import Image

from engine import to_svg, vectorize
from engine.ml import weights
from eval.metrics import count_nodes, mean_delta_e, render_svg, ssim
from eval.synthetic import flat_shapes

pytestmark = pytest.mark.gpu

SIZE = 256  # clean source; the degraded input is half that
SEEDS = range(5)
MAX_NODE_GROWTH = 1.25


def _degraded(clean: np.ndarray) -> bytes:
    """Downscale 2x with Lanczos and save as JPEG q75."""
    img = Image.fromarray(clean[..., :3]).resize((SIZE // 2, SIZE // 2), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return buf.getvalue()


def _run(data: bytes, clean: np.ndarray, enabled: bool) -> dict:
    start = time.perf_counter()
    result = vectorize(data, overrides={"enhance.enabled": enabled})
    seconds = time.perf_counter() - start
    svg = to_svg(result.doc)
    render = render_svg(svg, SIZE, SIZE)
    return {
        "ssim": ssim(clean, render),
        "de": mean_delta_e(clean, render),
        "nodes": count_nodes(svg)[0],
        "time": seconds,
        "report": result.reports[0].params,
    }


@pytest.fixture(scope="module")
def rows():
    for name in ("realesrgan_x4plus", "realesrgan_x4plus_anime_6b", "fbcnn_color"):
        if not weights.weight_path(name).is_file():
            pytest.skip(f"weight {name} missing; run python -m engine.ml.fetch_weights {name}")
    out = []
    for seed in SEEDS:
        clean = flat_shapes(seed, size=SIZE)
        data = _degraded(clean)
        out.append((seed, _run(data, clean, False), _run(data, clean, True)))
    print("\nseed  ssim off/on      dE off/on      nodes off/on   time off/on (s)  model")
    for seed, off, on in out:
        print(
            f"{seed:>4}  {off['ssim']:.3f} / {on['ssim']:.3f}  "
            f"{off['de']:5.2f} / {on['de']:5.2f}  {off['nodes']:5d} / {on['nodes']:5d}  "
            f"{off['time']:6.2f} / {on['time']:6.2f}  {on['report']['model_used']}"
        )
    return out


def test_enhance_ran_real_models(rows):
    for _, _, on in rows:
        report = on["report"]
        assert report["fallback_reason"] is None, report
        assert report["device_used"] == "cuda"
        assert report["effective_scale"] == 4
        assert report["model_used"].startswith("fbcnn_color+")  # JPEG input is deblocked


def test_enhance_improves_fidelity_on_degraded_logos(rows):
    off = np.mean([r[1]["ssim"] for r in rows])
    on = np.mean([r[2]["ssim"] for r in rows])
    print(f"\nmean ssim off {off:.4f} on {on:.4f}")
    assert on > off


def test_enhance_node_growth_bounded(rows):
    off = np.mean([r[1]["nodes"] for r in rows])
    on = np.mean([r[2]["nodes"] for r in rows])
    print(f"\nmean nodes off {off:.1f} on {on:.1f} ({on / off:.2f}x)")
    assert on <= MAX_NODE_GROWTH * off


def test_enhance_does_not_shift_colours(rows):
    # the GAN shifts flat colours; the low-frequency colour correction must undo that
    off = np.mean([r[1]["de"] for r in rows])
    on = np.mean([r[2]["de"] for r in rows])
    print(f"\nmean dE off {off:.3f} on {on:.3f}")
    assert on <= off
