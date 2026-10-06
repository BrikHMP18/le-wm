"""Local HTTP server for browsing recordings: ``uv run dataviz <file-or-dir>``."""

import argparse
import io
import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from urllib.parse import parse_qs, urlparse

from PIL import Image

from .reader import Recording, discover

STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}


class Library:
    """Lazily opened recordings, keyed by display name."""

    def __init__(self, target):
        self.paths = discover(target)
        self.open = {}
        self.lock = threading.Lock()

    def get(self, name):
        if name not in self.paths:
            raise KeyError(f"Unknown file: {name}")
        with self.lock:
            if name not in self.open:
                self.open[name] = Recording(self.paths[name])
            return self.open[name]

    def close(self):
        for recording in self.open.values():
            recording.close()


def encode_jpeg(image, quality):
    if image.shape[-1] == 1:
        pil = Image.fromarray(image[..., 0], mode="L")
    elif image.shape[-1] == 4:
        pil = Image.fromarray(image, mode="RGBA").convert("RGB")
    else:
        pil = Image.fromarray(image, mode="RGB")
    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def make_handler(library, quality):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def send(self, status, body, content_type, cache=False):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            if cache:
                self.send_header("Cache-Control", "max-age=3600")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, payload, status=HTTPStatus.OK):
            body = json.dumps(payload, allow_nan=False).encode()
            self.send(status, body, "application/json")

        def do_GET(self):
            url = urlparse(self.path)
            query = {k: v[-1] for k, v in parse_qs(url.query).items()}
            try:
                if url.path in STATIC:
                    filename, content_type = STATIC[url.path]
                    body = resources.files("dataviz.static").joinpath(filename).read_bytes()
                    return self.send(HTTPStatus.OK, body, content_type)
                if url.path == "/api/files":
                    return self.send_json(list(library.paths))
                recording = library.get(query.get("file", ""))
                if url.path == "/api/info":
                    return self.send_json(recording.info())
                if url.path == "/api/episodes":
                    return self.send_json(
                        {
                            "lengths": recording.lengths.tolist(),
                            "offsets": recording.offsets.tolist(),
                            "attributes": recording.episode_attributes(),
                        }
                    )
                if url.path == "/api/episode":
                    return self.send_json(recording.episode(int(query["ep"])))
                if url.path == "/api/frame":
                    image = recording.frame(query["col"], int(query["ep"]), int(query["t"]))
                    body = encode_jpeg(image, quality)
                    return self.send(HTTPStatus.OK, body, "image/jpeg", cache=True)
                self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except (KeyError, IndexError, ValueError) as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    return Handler


def serve(target, host="127.0.0.1", port=8765, quality=90):
    library = Library(target)
    server = ThreadingHTTPServer((host, port), make_handler(library, quality))
    server.daemon_threads = True
    server.library = library
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", help="HDF5 file, or a directory searched for *.h5 / *.hdf5")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--quality", type=int, default=90, help="JPEG quality for frames")
    parser.add_argument("--open", action="store_true", help="Open a browser tab")
    args = parser.parse_args(argv)

    server = serve(args.target, args.host, args.port, args.quality)
    url = f"http://{args.host}:{server.server_address[1]}/"
    print(f"Serving {len(server.library.paths)} file(s) from {args.target} at {url}")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.library.close()


if __name__ == "__main__":
    main()
