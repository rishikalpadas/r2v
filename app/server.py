"""Local preview server: python -m app.server [--port 8765].

Serves one page where you drop an image and compare it with its SVG.
Binds to 127.0.0.1 only; it is a local testing tool, not a production API.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from app.convert import convert
from engine import InputError, StageError
from engine.loader import MAX_BYTES

PAGE = Path(__file__).with_name("static") / "index.html"
DEFAULT_PORT = 8765
LAYER_MODES = ("stacked", "cutout")


class OptionError(ValueError):
    pass


def parse_options(query: str) -> dict[str, object]:
    """Parse ?max_colors=&layer_mode=&fit=&tolerance= into convert() options."""
    params = {k: v[-1] for k, v in parse_qs(query).items()}
    options: dict[str, object] = {}
    try:
        if "max_colors" in params:
            options["max_colors"] = int(params["max_colors"])
        if "tolerance" in params:
            options["tolerance"] = float(params["tolerance"])
    except ValueError as exc:
        raise OptionError("max_colors must be an integer and tolerance a number") from exc
    if "layer_mode" in params:
        if params["layer_mode"] not in LAYER_MODES:
            raise OptionError("layer_mode must be stacked or cutout")
        options["layer_mode"] = params["layer_mode"]
    if "fit" in params:
        if params["fit"] not in ("0", "1"):
            raise OptionError("fit must be 0 or 1")
        options["fit"] = params["fit"] == "1"
    return options


class Handler(BaseHTTPRequestHandler):
    server_version = "raster2vector-preview"

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write(f"[preview] {fmt % args}\n")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict[str, object]) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        if urlparse(self.path).path in ("/", "/index.html"):
            self._send(HTTPStatus.OK, PAGE.read_bytes(), "text/html; charset=utf-8")
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        if url.path != "/api/vectorize":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._json(HTTPStatus.LENGTH_REQUIRED, {"error": "Content-Length required"})
            return
        if length <= 0 or length > MAX_BYTES:
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "image must be between 1 byte and 50 MB"})
            return
        try:
            options = parse_options(url.query)
            result = convert(self.rfile.read(length), options)
        except (OptionError, InputError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except StageError as exc:
            self.log_message("stage failure: %s", exc)
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"conversion failed in {exc.stage}"})
            return
        self._json(HTTPStatus.OK, asdict(result))


def make_server(port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    httpd = make_server(args.port)
    print(f"raster2vector preview running at http://127.0.0.1:{httpd.server_address[1]}  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
