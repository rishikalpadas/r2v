"""Validated image loading: bytes in, normalised RGBA RasterImage out."""
from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from engine.errors import InputError
from engine.types import RasterImage

MAX_BYTES = 50 * 1024 * 1024
MAX_SIDE = 8192
ALLOWED_FORMATS = frozenset({"PNG", "JPEG", "WEBP", "BMP", "TIFF", "GIF"})

_DECODE_MSG = "could not decode image"


def _to_rgba(img: Image.Image) -> Image.Image:
    if img.mode == "I;16":
        arr = np.asarray(img, dtype=np.uint16) >> 8
        img = Image.fromarray(arr.astype(np.uint8), mode="L")
    return img.convert("RGBA")


def load_image(data: bytes) -> RasterImage:
    if len(data) > MAX_BYTES:
        raise InputError("image is larger than the 50 MB limit")
    try:
        with Image.open(BytesIO(data)) as probe:
            probe.verify()
        with Image.open(BytesIO(data)) as img:
            if img.format not in ALLOWED_FORMATS:
                raise InputError(
                    "unsupported image format; use PNG, JPEG, WebP, BMP, TIFF or GIF"
                )
            width, height = img.size
            if max(width, height) > MAX_SIDE:
                raise InputError(
                    f"image is larger than the {MAX_SIDE}px limit on its long side"
                )
            img.seek(0)
            pixels = np.array(_to_rgba(img), dtype=np.uint8)
    except InputError:
        raise
    except Exception as exc:  # Pillow raises many types; never leak details
        raise InputError(_DECODE_MSG) from exc
    return RasterImage(pixels=pixels, original_size=(width, height))
