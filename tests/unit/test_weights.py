import hashlib
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from engine.ml import device, fetch_weights, weights
from engine.ml.weights import MANIFEST, WeightSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = b"fake weights payload"


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
        scale=1,
        kind="denoise",
    )
    monkeypatch.setitem(MANIFEST, "fake", spec)
    return spec


def _serve(payload, calls=None):
    def download(url, dest):
        if calls is not None:
            calls.append(url)
        Path(dest).write_bytes(payload)

    return download


def _run_python(code):
    env = {k: v for k, v in os.environ.items() if k != "R2V_WEIGHTS_DIR"}
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120,
    )


def test_weights_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("R2V_WEIGHTS_DIR", str(tmp_path / "w"))
    assert weights.weights_dir() == tmp_path / "w"
    assert weights.weight_path("realesrgan_x4plus") == tmp_path / "w" / "realesrgan_x4plus.pth"
    assert not (tmp_path / "w").exists()


def test_weights_dir_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("R2V_WEIGHTS_DIR", "")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setattr(weights.sys, "platform", "win32")
    assert weights.weights_dir() == tmp_path / "local" / "raster2vector" / "weights"
    monkeypatch.setattr(weights.sys, "platform", "linux")
    monkeypatch.setattr(weights.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    assert weights.weights_dir() == tmp_path / "home" / ".cache" / "raster2vector" / "weights"


def test_weight_path_unknown_name(wdir):
    with pytest.raises(KeyError, match="realesrgan_x4plus"):
        weights.weight_path("nope")


def test_manifest_entries_complete():
    assert set(MANIFEST) == {
        "realesrgan_x4plus", "realesrgan_x4plus_anime_6b", "fbcnn_color", "scunet_color_real_psnr",
    }
    kinds = {spec.kind for spec in MANIFEST.values()}
    assert {"sr", "deblock", "denoise"} <= kinds
    for name, spec in MANIFEST.items():
        assert spec.name == name
        assert spec.url.startswith("https://") and spec.url.endswith(".pth")
        assert re.fullmatch(r"[0-9a-f]{64}", spec.sha256)
        assert spec.license
        assert spec.kind in ("sr", "deblock", "denoise")
        assert isinstance(spec.scale, int) and spec.scale >= 1


def test_weight_spec_frozen():
    with pytest.raises(AttributeError):
        MANIFEST["fbcnn_color"].scale = 2


def test_verify_missing_file(wdir, fake_spec):
    assert weights.verify("fake") is False


def test_verify_wrong_content(wdir, fake_spec):
    wdir.mkdir()
    weights.weight_path("fake").write_bytes(b"other")
    assert weights.verify("fake") is False


def test_fetch_success(wdir, fake_spec, monkeypatch):
    calls = []
    monkeypatch.setattr(weights, "_download", _serve(PAYLOAD, calls))
    path = weights.fetch("fake")
    assert path == wdir / "fake.pth"
    assert path.read_bytes() == PAYLOAD
    assert calls == [fake_spec.url]
    assert weights.verify("fake")
    assert [p.name for p in wdir.iterdir()] == ["fake.pth"]


def test_fetch_rejects_bad_hash(wdir, fake_spec, monkeypatch):
    monkeypatch.setattr(weights, "_download", _serve(b"tampered"))
    with pytest.raises(ValueError) as info:
        weights.fetch("fake")
    msg = str(info.value)
    assert fake_spec.sha256 in msg
    assert hashlib.sha256(b"tampered").hexdigest() in msg
    assert list(wdir.iterdir()) == []


def test_fetch_replaces_corrupt_file(wdir, fake_spec, monkeypatch):
    wdir.mkdir()
    (wdir / "fake.pth").write_bytes(b"corrupt")
    monkeypatch.setattr(weights, "_download", _serve(PAYLOAD))
    assert weights.fetch("fake").read_bytes() == PAYLOAD
    assert [p.name for p in wdir.iterdir()] == ["fake.pth"]


def test_fetch_download_error_leaves_no_temp(wdir, fake_spec, monkeypatch):
    def broken(url, dest):
        Path(dest).write_bytes(b"partial")
        raise OSError("connection reset")

    monkeypatch.setattr(weights, "_download", broken)
    with pytest.raises(OSError, match="connection reset"):
        weights.fetch("fake")
    assert list(wdir.iterdir()) == []


def test_fetch_atomic(wdir, fake_spec, monkeypatch):
    wdir.mkdir()
    path = wdir / "fake.pth"
    path.write_bytes(PAYLOAD)
    os.utime(path, (1_000_000_000, 1_000_000_000))
    before = path.stat().st_mtime_ns
    calls = []
    monkeypatch.setattr(weights, "_download", _serve(b"new", calls))
    assert weights.fetch("fake") == path
    assert calls == []
    assert path.stat().st_mtime_ns == before
    assert path.read_bytes() == PAYLOAD


def test_download_writes_dest(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(PAYLOAD)
    dest = tmp_path / "dest.bin"
    weights._download(src.as_uri(), dest)
    assert dest.read_bytes() == PAYLOAD


def test_pick_device_cpu_when_no_cuda(monkeypatch):
    monkeypatch.setattr(device, "_cuda_available", lambda: False)
    assert device.pick_device() == "cpu"
    assert device.pick_device("auto") == "cpu"
    assert device.pick_device("cuda") == "cpu"
    assert device.pick_device("cpu") == "cpu"


def test_pick_device_cuda_when_available(monkeypatch):
    monkeypatch.setattr(device, "_cuda_available", lambda: True)
    assert device.pick_device("auto") == "cuda"
    assert device.pick_device("cuda") == "cuda"
    assert device.pick_device("cpu") == "cpu"


def test_pick_device_invalid():
    with pytest.raises(ValueError, match="gpu"):
        device.pick_device("gpu")


def test_cuda_available_false_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert device._cuda_available() is False


def test_ml_available_false_without_spandrel(monkeypatch):
    from engine import ml

    monkeypatch.setitem(sys.modules, "spandrel", None)
    assert ml.ml_available() is False


def test_ml_available_true_when_both_import(monkeypatch):
    from engine import ml

    imported = []
    monkeypatch.setattr(ml.importlib, "import_module", lambda name: imported.append(name))
    assert ml.ml_available() is True
    assert imported == ["torch", "spandrel"]


def test_cli_no_args_is_usage_error(capsys):
    with pytest.raises(SystemExit) as info:
        fetch_weights.main([])
    assert info.value.code == 2
    assert "usage" in capsys.readouterr().err.lower()


def test_cli_all_and_names_conflict(capsys):
    with pytest.raises(SystemExit) as info:
        fetch_weights.main(["--all", "fbcnn_color"])
    assert info.value.code == 2


def test_cli_verified_and_downloaded(wdir, fake_spec, monkeypatch, capsys):
    monkeypatch.setattr(weights, "_download", _serve(PAYLOAD))
    assert fetch_weights.main(["fake"]) == 0
    out = capsys.readouterr().out
    assert "fake" in out and "downloaded" in out and "MIT" in out
    assert fetch_weights.main(["fake"]) == 0
    out = capsys.readouterr().out
    assert "verified" in out and "MIT" in out


def test_cli_failure_exit_code(wdir, fake_spec, monkeypatch, capsys):
    monkeypatch.setattr(weights, "_download", _serve(b"tampered"))
    assert fetch_weights.main(["fake", "nope"]) == 1
    out = capsys.readouterr().out
    lines = out.strip().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("fake") and "failed" in lines[0] and "sha256" in lines[0]
    assert lines[1].startswith("nope") and "failed" in lines[1]


def test_cli_all_covers_manifest(wdir, monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(fetch_weights, "fetch", lambda name: seen.append(name) or wdir / name)
    monkeypatch.setattr(fetch_weights, "verify", lambda name: True)
    assert fetch_weights.main(["--all"]) == 0
    assert seen == list(MANIFEST)
    assert len(capsys.readouterr().out.strip().splitlines()) == len(MANIFEST)


def test_engine_import_does_not_load_torch():
    result = _run_python("""
        import sys
        import engine, engine.ml, engine.ml.device, engine.ml.weights
        assert "torch" not in sys.modules, "torch imported"
        assert "spandrel" not in sys.modules, "spandrel imported"
        print("ok")
    """)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_engine_imports_without_torch():
    result = _run_python("""
        import io, sys
        # Block the imports the way an uninstalled package fails. A sys.modules[...] = None
        # entry is not equivalent: scipy's array_api_compat treats any "torch" key as a
        # loaded module and crashes on getattr(None, "Tensor").
        class _Missing:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in ("torch", "spandrel"):
                    raise ModuleNotFoundError(f"No module named {name!r}", name=name)
                return None
        sys.meta_path.insert(0, _Missing())
        for mod in [m for m in sys.modules if m.split(".")[0] in ("torch", "spandrel")]:
            del sys.modules[mod]
        import numpy as np
        from PIL import Image
        import engine
        from engine.ml import ml_available
        from engine.ml.device import pick_device
        arr = np.zeros((32, 32, 3), np.uint8)
        arr[8:24, 8:24] = (200, 30, 30)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        result = engine.vectorize(buf.getvalue())
        assert "<svg" in engine.to_svg(result.doc)
        assert ml_available() is False
        assert pick_device("auto") == "cpu"
        assert "torch" not in sys.modules
        print("ok")
    """)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
