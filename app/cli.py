"""Command line: python -m app.cli input.png [-o output.svg] [options]."""
from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

from app.convert import convert
from engine import InputError, StageError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.cli", description="Convert a raster image to SVG or PDF.")
    parser.add_argument("input", type=Path, help="PNG, JPEG, WebP, BMP, TIFF or GIF file")
    parser.add_argument("-o", "--output", type=Path, help="output SVG or PDF path (default: input name with .svg)")
    parser.add_argument("--max-colors", type=int, help="maximum colors to keep (default 16)")
    parser.add_argument("--layer-mode", choices=["stacked", "cutout"], help="SVG layering (default stacked)")
    parser.add_argument("--no-fit", action="store_true", help="skip curve fitting (raw pixel outlines)")
    parser.add_argument("--tolerance", type=float, help="curve fit tolerance in pixels (default 0.75)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output or args.input.with_suffix(".svg")
    try:
        data = args.input.read_bytes()
    except OSError as exc:
        print(f"error: cannot read {args.input}: {exc.strerror}", file=sys.stderr)
        return 2

    options = {
        "max_colors": args.max_colors,
        "layer_mode": args.layer_mode,
        "fit": False if args.no_fit else None,
        "tolerance": args.tolerance,
    }
    try:
        result = convert(data, options)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except StageError as exc:
        print(f"error: conversion failed ({exc})", file=sys.stderr)
        return 1

    if output.suffix.lower() == ".pdf":
        output.write_bytes(base64.b64decode(result.pdf_base64))
    else:
        output.write_text(result.svg, encoding="utf-8")
    quality = f"SSIM {result.ssim:.4f}, mean dE {result.delta_e:.2f}" if result.ssim is not None else "quality n/a (image too small)"
    print(f"wrote {output}  ({result.width}x{result.height}, {result.paths} shapes, {result.nodes} nodes, "
          f"{result.seconds:.2f}s, {quality})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
