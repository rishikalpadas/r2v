import hashlib
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from engine.ml import models
from engine.ml.weights import MANIFEST, WeightSpec

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = b"fake model weights"


class _Req:
    def __init__(self, minimum=0, multiple_of=1):
        self.minimum = minimum
        self.multiple_of = multiple_of


class _FakeDescriptor:
    """Just enough of a spandrel ImageModelDescriptor for the wrapper."""

    def __init__(self, scale=4, minimum=0, multiple_of=1, overshoot=False, supports_half=True):
        self.scale = scale
        self.supports_half = supports_half
        self.size_requirements = _Req(minimum, multiple_of)
        self.model = torch.nn.Upsample(scale_factor=scale, mode="nearest")
        self.overshoot = overshoot
        self.inputs = []
        self.calls = []

    def to(self, *args, **kwargs):
        self.calls.append(("to", args, kwargs))
        return self

    def eval(self):
        self.calls.append(("eval",))
        return self

    def half(self):
        self.calls.append(("half",))
        return self

    def float(self):
        self.calls.append(("float",))
        return self

    def __call__(self, image):
        self.inputs.append(tuple(image.shape))
        req = self.size_requirements
        h, w = image.shape[-2:]
        if h < req.minimum or w < req.minimum or h % req.multiple_of or w % req.multiple_of:
            raise RuntimeError(f"input {h}x{w} violates size requirements")
        out = self.model(image)
        return out * 3 - 1 if self.overshoot else out


@pytest.fixture(autouse=True)
def _clear_cache():
    models.load_model.cache_clear()
    yield
    models.load_model.cache_clear()


@pytest.fixture
def wdir(tmp_path, monkeypatch):
    d = tmp_path / "weights"
    monkeypatch.setenv("R2V_WEIGHTS_DIR", str(d))
    return d


@pytest.fixture
def fake_spec(monkeypatch):
    spec = WeightSpec(
        name="fake",
        url="https://example.invalid/fake.pth",
        sha256=hashlib.sha256(PAYLOAD).hexdigest(),
        license="MIT",
        scale=4,
        kind="sr",
    )
    monkeypatch.setitem(MANIFEST, "fake", spec)
    return spec


@pytest.fixture
def fake_file(wdir, fake_spec):
    wdir.mkdir(parents=True)
    path = wdir / "fake.pth"
    path.write_bytes(PAYLOAD)
    return path


def _patch_loader(monkeypatch, descriptor):
    loaded = []

    def load(path):
        loaded.append(Path(path))
        return descriptor

    monkeypatch.setattr(models, "_load_descriptor", load)
    return loaded


def _rand(h, w, seed=0):
    return np.random.default_rng(seed).random((h, w, 3), dtype=np.float32)


def test_missing_weight_message(wdir, fake_spec):
    with pytest.raises(FileNotFoundError) as info:
        models.load_model("fake", "cpu")
    assert "python -m engine.ml.fetch_weights fake" in str(info.value)


def test_unknown_name_raises_keyerror(wdir):
    with pytest.raises(KeyError, match="unknown weight"):
        models.load_model("no_such_model", "cpu")


def test_hash_mismatch_tells_user_to_refetch(wdir, fake_spec, monkeypatch):
    wdir.mkdir(parents=True)
    (wdir / "fake.pth").write_bytes(b"corrupted")
    loaded = _patch_loader(monkeypatch, _FakeDescriptor())
    with pytest.raises(ValueError) as info:
        models.load_model("fake", "cpu")
    assert "sha256" in str(info.value)
    assert "python -m engine.ml.fetch_weights fake" in str(info.value)
    assert loaded == []


def test_wrapper_shapes(fake_file, monkeypatch):
    desc = _FakeDescriptor(scale=4, overshoot=True)
    loaded = _patch_loader(monkeypatch, desc)
    fn = models.load_model("fake", "cpu")
    assert loaded == [fake_file]
    assert fn.scale == 4
    out = fn(_rand(9, 13))
    assert isinstance(out, np.ndarray)
    assert out.shape == (36, 52, 3)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert out.min() == 0.0 and out.max() == 1.0  # overshoot really was clamped
    assert desc.inputs == [(1, 3, 9, 13)]


def test_wrapper_values_follow_model(fake_file, monkeypatch):
    _patch_loader(monkeypatch, _FakeDescriptor(scale=2))
    rgb = _rand(5, 6)
    out = models.load_model("fake", "cpu")(rgb)
    np.testing.assert_allclose(out, np.repeat(np.repeat(rgb, 2, axis=0), 2, axis=1), atol=1e-6)


def test_cpu_stays_fp32_and_eval(fake_file, monkeypatch):
    desc = _FakeDescriptor(supports_half=True)
    _patch_loader(monkeypatch, desc)
    models.load_model("fake", "cpu")
    names = [c[0] for c in desc.calls]
    assert "eval" in names
    assert "half" not in names
    assert ("to", ("cpu",), {}) in desc.calls


@pytest.mark.parametrize("supports_half, expect", [(True, "half"), (False, "float")])
def test_cuda_precision_choice(fake_file, monkeypatch, supports_half, expect):
    # loading only; the fake descriptor never touches CUDA, so this runs on CPU
    desc = _FakeDescriptor(supports_half=supports_half)
    _patch_loader(monkeypatch, desc)
    models.load_model("fake", "cuda")
    names = [c[0] for c in desc.calls]
    assert expect in names
    assert ({"half", "float"} - {expect}).isdisjoint(names)
    assert ("to", ("cuda",), {}) in desc.calls


def test_padding_for_minimum_size(fake_file, monkeypatch):
    desc = _FakeDescriptor(scale=1, minimum=40)
    _patch_loader(monkeypatch, desc)
    rgb = _rand(10, 7)
    out = models.load_model("fake", "cpu")(rgb)
    assert desc.inputs == [(1, 3, 40, 40)]
    assert out.shape == (10, 7, 3)
    np.testing.assert_allclose(out, rgb, atol=1e-6)  # padding cropped off exactly


def test_padding_for_multiple_of_with_scale(fake_file, monkeypatch):
    desc = _FakeDescriptor(scale=4, multiple_of=8)
    _patch_loader(monkeypatch, desc)
    rgb = _rand(10, 17)
    out = models.load_model("fake", "cpu")(rgb)
    assert desc.inputs == [(1, 3, 16, 24)]
    assert out.shape == (40, 68, 3)
    np.testing.assert_allclose(out, np.repeat(np.repeat(rgb, 4, axis=0), 4, axis=1), atol=1e-6)


def test_padding_one_pixel_image_uses_replicate(fake_file, monkeypatch):
    desc = _FakeDescriptor(scale=1, minimum=40)
    _patch_loader(monkeypatch, desc)
    rgb = np.full((1, 1, 3), 0.25, dtype=np.float32)
    out = models.load_model("fake", "cpu")(rgb)
    assert desc.inputs == [(1, 3, 40, 40)]
    assert out.shape == (1, 1, 3)
    np.testing.assert_allclose(out, rgb, atol=1e-6)


def test_cache_returns_same_object(fake_file, monkeypatch):
    loaded = _patch_loader(monkeypatch, _FakeDescriptor())
    a = models.load_model("fake", "cpu")
    b = models.load_model("fake", "cpu")
    assert a is b
    assert len(loaded) == 1


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((4, 4), dtype=np.float32),
        np.zeros((4, 4, 4), dtype=np.float32),
        np.zeros((0, 4, 3), dtype=np.float32),
        np.zeros((4, 4, 3), dtype=np.float64),
        np.zeros((4, 4, 3), dtype=np.uint8),
        [[[0.0, 0.0, 0.0]]],
    ],
    ids=["2d", "4ch", "empty", "float64", "uint8", "list"],
)
def test_invalid_input_raises_valueerror(fake_file, monkeypatch, bad):
    _patch_loader(monkeypatch, _FakeDescriptor())
    fn = models.load_model("fake", "cpu")
    with pytest.raises(ValueError):
        fn(bad)


def test_oom_propagates_unchanged(fake_file, monkeypatch):
    class OutOfMemoryError(RuntimeError):
        pass

    class _Oom(_FakeDescriptor):
        def __call__(self, image):
            raise OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

    _patch_loader(monkeypatch, _Oom())
    fn = models.load_model("fake", "cpu")
    with pytest.raises(OutOfMemoryError):
        fn(_rand(4, 4))


def test_load_descriptor_uses_weights_only(tmp_path, monkeypatch):
    """The real loader reads the file with torch.load(weights_only=True), not spandrel's unpickler."""
    seen = {}
    real_load = torch.load

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    path = tmp_path / "junk.pth"
    torch.save({"not_a_model.weight": torch.zeros(1)}, path)
    with pytest.raises(Exception):  # spandrel: unsupported architecture
        models._load_descriptor(path)
    assert seen.get("weights_only") is True


def _run_python(code):
    env = {k: v for k, v in os.environ.items() if k != "R2V_WEIGHTS_DIR"}
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120,
    )


def test_no_torch_import_at_module_import():
    proc = _run_python(
        """
        import sys
        import engine.ml.models
        assert "torch" not in sys.modules, "torch imported eagerly"
        assert "spandrel" not in sys.modules, "spandrel imported eagerly"
        print("ok")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"
