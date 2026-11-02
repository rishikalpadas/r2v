"""Exact-colour (<= max_colors) input must segment and vectorize exactly as on master d1f3181.

Fix 3 (colour segmentation) targets noisy JPEG/AA input only. Digests were recorded from master:
sha256 of segment labels (<i4) + palette bytes, and sha256 of the exported SVG (first 16 hex).
The SVG digest pins `fit.adaptive=False`: fix 2 (noise-adaptive fitting) deliberately changes the
fit, and `adaptive=False` is its reproduce-the-old-fit switch, so this still isolates segmentation.
"""
import hashlib

import pytest

from engine import to_svg
from engine.pipeline import vectorize
from engine.presets import SegmentParams
from engine.stages.segment import segment
from engine.types import RasterImage
from eval.synthetic import circle, flat_shapes, ring_in_ring, to_png_bytes, transparent_logo

MASTER_DIGESTS = [
    ("flat_shapes(0, n_colors=3)", "9459e7e08d9a82e2", "b5d71bc6da9bded6"),
    ("flat_shapes(0, n_colors=4)", "9555384438e56c4a", "dfbd4f6332566d6f"),
    ("flat_shapes(0, n_colors=6)", "0a4f0d75c0ae65e2", "651a9b24064ad452"),
    ("flat_shapes(1, n_colors=3)", "a32ede4d789ebff1", "dd62de50d24b350c"),
    ("flat_shapes(1, n_colors=4)", "9c14343eaed36b42", "9db4943ef606baca"),
    ("flat_shapes(1, n_colors=6)", "7eba6bf0cd5b08fb", "225ea106061c6359"),
    ("flat_shapes(2, n_colors=3)", "4e9d49182ae2a7b4", "1b6c21c04d0ba1d2"),
    ("flat_shapes(2, n_colors=4)", "8fd153a84aa3dec3", "76ee71fe3fbf21ec"),
    ("flat_shapes(2, n_colors=6)", "724d2b05b78454a1", "b46a1475f8a5840d"),
    ("flat_shapes(3, n_colors=3)", "7bb870a19c3dee11", "0e7975a1678d3d33"),
    ("flat_shapes(3, n_colors=4)", "5125233da61cec0e", "6907def49c22838b"),
    ("flat_shapes(3, n_colors=6)", "f4d7d43f1f6a43a2", "82cfb19ef10dc116"),
    ("flat_shapes(4, n_colors=3)", "784709e4e84c2a56", "ed1f275fcfd1fb36"),
    ("flat_shapes(4, n_colors=4)", "abd331483954a0a5", "5cbe32723af4093e"),
    ("flat_shapes(4, n_colors=6)", "957fce26c086c8be", "c9d4058eca253b03"),
    ("flat_shapes(5, n_colors=3)", "e104bd43c9a3d919", "98e8e58bb304399d"),
    ("flat_shapes(5, n_colors=4)", "8263fb80e682fa68", "0c63d0344cf030a7"),
    ("flat_shapes(5, n_colors=6)", "4312ac4584663fda", "483ddd2da0ba22f7"),
    ("flat_shapes(6, n_colors=3)", "4e458a8d17ac11c7", "79988eb7338f56c8"),
    ("flat_shapes(6, n_colors=4)", "d2b403b7225e6db6", "762c0c4d3f82ff00"),
    ("flat_shapes(6, n_colors=6)", "5323305c39683be0", "cb0bc6b4712f3ffe"),
    ("flat_shapes(7, n_colors=3)", "d5a3b331508fbb06", "13a3d61bc39db1ff"),
    ("flat_shapes(7, n_colors=4)", "9c5d067d5ba28af8", "fb8a1fdf59113f4b"),
    ("flat_shapes(7, n_colors=6)", "a9aa814c19ad7265", "12652cfe5fb1a51c"),
    ("flat_shapes(8, n_colors=3)", "75b4f658b1fc35e4", "70a1c37d1754ec3f"),
    ("flat_shapes(8, n_colors=4)", "ad8b1d70f150ac69", "5e08ab289659f95e"),
    ("flat_shapes(8, n_colors=6)", "dfa5cb0668e96bbf", "8ea32a503698cca9"),
    ("flat_shapes(9, n_colors=3)", "12108ef81e2c0d7f", "77eaa3fd1618d067"),
    ("flat_shapes(9, n_colors=4)", "8bb134428f109111", "4cc45fefaac8d116"),
    ("flat_shapes(9, n_colors=6)", "285312669c8f907f", "908f72946fcee560"),
    ("ring_in_ring()", "f1bf62ae630387ab", "ba8a9132d31a77f2"),
    ("transparent_logo()", "4b2e05f76587c803", "59fb8141bd2e3755"),
    ("circle()", "ce4ee9bae8c4e7d4", "7c4d1e496e0d21f0"),
]
CASES = {f"flat_shapes({s}, n_colors={n})": (flat_shapes, (s,), {"n_colors": n})
         for s in range(10) for n in (3, 4, 6)}
CASES.update({f"{fn.__name__}()": (fn, (), {}) for fn in (ring_in_ring, transparent_logo, circle)})


def _build(call: str):
    fn, args, kwargs = CASES[call]
    return fn(*args, **kwargs)


@pytest.mark.parametrize(("call", "seg_digest", "svg_digest"), MASTER_DIGESTS)
def test_exact_colour_output_identical_to_master(call, seg_digest, svg_digest):
    px = _build(call)
    rm = segment(RasterImage(pixels=px, original_size=px.shape[1::-1]), SegmentParams())
    seg = hashlib.sha256(rm.labels.astype("<i4").tobytes() + rm.palette_rgb.tobytes())
    assert seg.hexdigest()[:16] == seg_digest
    svg = to_svg(vectorize(to_png_bytes(px), overrides={"fit.adaptive": False}).doc)
    assert hashlib.sha256(svg.encode()).hexdigest()[:16] == svg_digest
