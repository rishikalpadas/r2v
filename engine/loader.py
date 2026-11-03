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
    """Convert to RGBA. Integer modes ("I", "I;16", "I;16B", ...) hold 16-bit samples (Pillow
    decodes 16-bit PNG/TIFF to them), so they are clipped to 0..65535 and scaled to 8 bits
    instead of Pillow's default clipping to 0..255."""
    if img.mode.startswith("I"):
        arr = np.clip(np.asarray(img).astype(np.int64), 0, 0xFFFF) >> 8
        img = Image.fromarray(arr.astype(np.uint8), mode="L")
    return img.convert("RGBA")


def load_image(data: bytes | bytearray | memoryview) -> RasterImage:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise InputError("image data must be bytes")
    try:
        size = memoryview(data).nbytes
    except ValueError:  # a released memoryview
        raise InputError(_DECODE_MSG) from None
    if size > MAX_BYTES:
        raise InputError("image is larger than the 50 MB limit")
    try:
        data = bytes(data)
        with Image.open(BytesIO(data)) as probe:
            probe.verify()
        with Image.open(BytesIO(data)) as img:
            if img.format not in ALLOWED_FORMATS:
                raise InputError(
                    "unsupported image format; use PNG, JPEG, WebP, BMP, TIFF or GIF"
                )
            source_format = img.format
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
    return RasterImage(pixels=pixels, original_size=(width, height), source_format=source_format)
