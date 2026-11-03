from eval.make_synthetic_dataset import make_dataset


def test_writes_expected_logo_files(tmp_path):
    root = make_dataset(tmp_path / "ds")
    files = sorted(p.name for p in (root / "logos").glob("*.png"))
    assert len(files) == 25
    assert "flat_00.png" in files and "flat_19.png" in files
    assert {"circle_0.png", "ring_in_ring.png", "transparent_logo.png"} <= set(files)


def test_is_deterministic(tmp_path):
    a = make_dataset(tmp_path / "a")
    b = make_dataset(tmp_path / "b")
    for p in (a / "logos").glob("*.png"):
        assert p.read_bytes() == (b / "logos" / p.name).read_bytes()


def test_writes_degraded_logos_with_clean_targets(tmp_path):
    from PIL import Image

    root = make_dataset(tmp_path / "ds", size=64)
    logos = sorted(p.stem for p in (root / "logos").glob("*.png"))
    degraded = sorted(p.stem for p in (root / "degraded").glob("*.jpg"))
    assert degraded == logos
    for stem in logos:
        with Image.open(root / "degraded" / f"{stem}.jpg") as img:
            assert img.format == "JPEG"
            assert img.size == (32, 32)
        clean = root / "degraded" / "clean" / f"{stem}.png"
        assert clean.read_bytes() == (root / "logos" / f"{stem}.png").read_bytes()


def test_degraded_is_deterministic(tmp_path):
    a = make_dataset(tmp_path / "a", size=64)
    b = make_dataset(tmp_path / "b", size=64)
    for p in (a / "degraded").glob("*.jpg"):
        assert p.read_bytes() == (b / "degraded" / p.name).read_bytes()
