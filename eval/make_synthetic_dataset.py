"""Write a small deterministic synthetic logo dataset for benchmark smoke runs."""
from __future__ import annotations

import argparse
from pathlib import Path

from eval.synthetic import circle, flat_shapes, ring_in_ring, to_png_bytes, transparent_logo

SIZE = 256
N_FLAT = 20


def make_dataset(root: Path, size: int = SIZE) -> Path:
    logos = Path(root) / "logos"
    logos.mkdir(parents=True, exist_ok=True)
    images = {f"flat_{seed:02d}.png": flat_shapes(seed, size=size) for seed in range(N_FLAT)}
    images["circle_0.png"] = circle(size, size / 2, size / 2, size * 0.3)
    images["circle_1.png"] = circle(size, size * 0.4, size * 0.55, size * 0.2)
    images["circle_2.png"] = circle(size, size / 2, size / 2, size * 0.42)
    images["ring_in_ring.png"] = ring_in_ring(size)
    images["transparent_logo.png"] = transparent_logo(size)
    for name, pixels in images.items():
        (logos / name).write_bytes(to_png_bytes(pixels))
    return Path(root)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("eval/dataset_synthetic"))
    args = ap.parse_args(argv)
    print(f"wrote {make_dataset(args.out)}")


if __name__ == "__main__":
    main()
