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
