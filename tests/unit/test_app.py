import json
import threading
import urllib.error
import urllib.request

import pytest

from app import cli, server
from app.convert import convert, to_overrides
from eval.synthetic import circle, flat_shapes, to_png_bytes


def test_to_overrides_maps_and_drops_unset():
    assert to_overrides({"max_colors": 8, "fit": None, "bogus": 1}) == {"segment.max_colors": 8}


def test_convert_scores_result():
    result = convert(to_png_bytes(flat_shapes(1)))
    assert result.svg.startswith("<svg")
    assert result.ssim is not None and result.ssim >= 0.95
    assert result.nodes > 0 and result.paths > 0
    assert set(result.stage_seconds) == {"segment", "trace", "fit", "export"}


def test_cli_writes_svg(tmp_path, capsys):
    src = tmp_path / "logo.png"
    src.write_bytes(to_png_bytes(circle()))
    assert cli.main([str(src)]) == 0
    assert "<circle" in (tmp_path / "logo.svg").read_text(encoding="utf-8")
    assert "wrote" in capsys.readouterr().out


def test_cli_bad_image_exit_code(tmp_path, capsys):
    src = tmp_path / "bad.png"
    src.write_bytes(b"not an image")
    assert cli.main([str(src)]) == 2
    assert "error:" in capsys.readouterr().err


def test_cli_missing_file(tmp_path):
    assert cli.main([str(tmp_path / "nope.png")]) == 2


@pytest.fixture()
def live_server():
    httpd = server.make_server(port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _post(url, body):
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/octet-stream"})
    return urllib.request.urlopen(req, timeout=60)


def test_server_serves_page(live_server):
    with urllib.request.urlopen(live_server + "/", timeout=10) as resp:
        assert resp.status == 200
        assert b"<html" in resp.read()


def test_server_vectorizes(live_server):
    with _post(live_server + "/api/vectorize?max_colors=8&layer_mode=cutout", to_png_bytes(flat_shapes(2))) as resp:
        payload = json.loads(resp.read())
    assert payload["svg"].startswith("<svg")
    assert payload["width"] == 128 and payload["ssim"] >= 0.95


def test_server_rejects_bad_image(live_server):
    with pytest.raises(urllib.error.HTTPError) as err:
        _post(live_server + "/api/vectorize", b"garbage")
    assert err.value.code == 400
    assert "error" in json.loads(err.value.read())


def test_server_rejects_bad_option(live_server):
    with pytest.raises(urllib.error.HTTPError) as err:
        _post(live_server + "/api/vectorize?max_colors=abc", to_png_bytes(circle()))
    assert err.value.code == 400


def test_server_unknown_path_404(live_server):
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(live_server + "/nope", timeout=10)
    assert err.value.code == 404
