from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from engine.errors import InputError
from engine.loader import ALLOWED_FORMATS, MAX_BYTES, MAX_SIDE, load_image


def _encode(img: Image.Image, fmt: str = "PNG", **kwargs) -> bytes:
    buf = BytesIO()
    img.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


def test_constants():
    assert MAX_BYTES == 50 * 1024 * 1024
    assert MAX_SIDE == 8192
    assert ALLOWED_FORMATS == frozenset({"PNG", "JPEG", "WEBP", "BMP", "TIFF", "GIF"})


def test_rejects_oversize_bytes():
    with pytest.raises(InputError, match="50 MB"):
        load_image(b"\0" * (MAX_BYTES + 1))


def test_rejects_long_side_over_8192():
    with pytest.raises(InputError):
        load_image(_encode(Image.new("RGB", (MAX_SIDE + 1, 1))))


def test_rejects_undecodable():
    with pytest.raises(InputError, match="could not decode image"):
        load_image(b"not an image")


def test_rejects_unsupported_format():
    data = _encode(Image.new("RGBA", (16, 16), (1, 2, 3, 255)), "ICO")
    with pytest.raises(InputError):
        load_image(data)


def test_gif_uses_first_frame():
    red = Image.new("RGB", (4, 4), (255, 0, 0))
    blue = Image.new("RGB", (4, 4), (0, 0, 255))
    data = _encode(red, "GIF", save_all=True, append_images=[blue])
    out = load_image(data)
    assert tuple(out.pixels[0, 0, :3]) == (255, 0, 0)


def _make(mode: str) -> Image.Image:
    if mode == "L":
        return Image.new("L", (5, 3), 128)
    if mode == "LA":
        return Image.new("LA", (5, 3), (128, 200))
    if mode == "P":
        img = Image.new("P", (5, 3), 0)
        img.putpalette([255, 0, 0] + [0, 255, 0] + [0] * 762)
        img.putpixel((1, 0), 1)
        img.info["transparency"] = 0
        return img
    if mode == "CMYK":
        return Image.new("CMYK", (5, 3), (0, 0, 0, 0))
    img = Image.new("I;16", (5, 3), 65535)
    return img


@pytest.mark.parametrize("mode", ["L", "LA", "P", "CMYK", "I;16"])
def test_mode_conversion(mode):
    img = _make(mode)
    fmt = "TIFF" if mode in ("CMYK", "I;16") else "PNG"
    kwargs = {"transparency": 0} if mode == "P" else {}
    out = load_image(_encode(img, fmt, **kwargs))
    assert out.pixels.shape == (3, 5, 4)
    assert out.pixels.dtype == np.uint8
    if mode == "P":
        assert out.pixels[0, 0, 3] == 0
        assert out.pixels[0, 1, 3] == 255
    if mode == "I;16":
        assert tuple(out.pixels[0, 0]) == (255, 255, 255, 255)


def test_original_size_and_scale():
    out = load_image(_encode(Image.new("RGB", (30, 20))))
    assert out.original_size == (30, 20)
    assert out.scale == 1.0


def test_released_memoryview_is_input_error():
    view = memoryview(b"abc")
    view.release()
    with pytest.raises(InputError):
        load_image(view)
