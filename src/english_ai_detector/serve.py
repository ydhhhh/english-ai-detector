"""Private localhost HTTP entry point for a locally trained English bundle."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from urllib.parse import unquote, urlsplit

from english_ai_detector.documents import DocumentError, doc_converter, extract_document

WEB = Path(__file__).resolve().parent / "web" / "index.html"


@dataclass
class Limits:
    max_request_bytes: int = 8 * 1024 * 1024
    max_characters: int = 200_000
    requests_per_minute: int = 6
    socket_timeout_seconds: int = 30


class DetectorServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, detector, limits=None):
        self.detector = detector
        self.limits = limits or Limits()
        self.inference_lock = threading.Lock()
        self.rate_lock = threading.Lock()
        self.rate = {}
        super().__init__(address, Handler)

    def request_allowed(self, client):
        now = time.monotonic()
        with self.rate_lock:
            if len(self.rate) > 128:
                self.rate = {key: value for key,value in self.rate.items() if value and now-value[-1] < 60}
            bucket = self.rate.setdefault(client, deque())
            while bucket and now-bucket[0] >= 60:
                bucket.popleft()
            if len(bucket) >= self.limits.requests_per_minute:
                return False
            bucket.append(now)
            return True


class Handler(BaseHTTPRequestHandler):
    server_version = "EnglishDetector/1"
    protocol_version = "HTTP/1.0"

    def setup(self):
        super().setup()
        self.connection.settimeout(self.server.limits.socket_timeout_seconds)

    def log_message(self, fmt, *args):
        # Never log request text, document names, full URLs or model output.
        pass

    def origin_allowed(self):
        origin = self.headers.get("Origin")
        if origin in (None, "null"):
            return True
        parsed = urlsplit(origin)
        return parsed.scheme in ("http", "https") and parsed.hostname in ("localhost", "127.0.0.1", "::1")

    def headers_common(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if self.origin_allowed() and self.headers.get("Origin"):
            self.send_header("Access-Control-Allow-Origin", self.headers["Origin"])
            self.send_header("Vary", "Origin")

    def send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status); self.headers_common()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        if status == 429:
            self.send_header("Retry-After", "60")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_OPTIONS(self):
        if not self.origin_allowed():
            self.send_json(403, {"error": "Only local browser origins are allowed."}); return
        self.send_response(204); self.headers_common()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Filename")
        self.send_header("Content-Length", "0"); self.end_headers()

    def do_GET(self):
        route = urlsplit(self.path).path
        if route == "/health":
            self.send_json(200, {"ready": True, "model_id": self.server.detector.model_id, "language": "en", "limits": {
                "max_request_bytes": self.server.limits.max_request_bytes,
                "max_characters": self.server.limits.max_characters,
                "requests_per_minute": self.server.limits.requests_per_minute,
                "concurrent_detections": 1}, "doc_parser": doc_converter()[0],
                "text_storage": "not stored by HTTP service", "window_policy": "full coverage; oversized input rejected"})
        elif route in ("/", "/index.html"):
            body = WEB.read_bytes(); self.send_response(200); self.headers_common()
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self' http://127.0.0.1:* http://localhost:*")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        else:
            self.send_json(404, {"error": "Endpoint not found."})

    def do_POST(self):
        if not self.origin_allowed():
            self.send_json(403, {"error": "Only local browser origins are allowed."}); return
        route = urlsplit(self.path).path
        if route not in ("/detect", "/detect-file"):
            self.send_json(404, {"error": "Endpoint not found."}); return
        if self.headers.get("Transfer-Encoding"):
            self.send_json(400, {"error": "Use a request with Content-Length."}); return
        try:
            count = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            count = -1
        if count < 0:
            self.send_json(411, {"error": "Content-Length is required."}); return
        if count > self.server.limits.max_request_bytes:
            self.send_json(413, {"error": "Request exceeds the upload limit; no truncation was performed."}); return
        if not self.server.request_allowed(self.client_address[0]):
            self.send_json(429, {"error": "Too many requests; retry after one minute."}); return
        if not self.server.inference_lock.acquire(blocking=False):
            self.send_json(429, {"error": "Another detection is running; retry shortly."}); return
        try:
            body = self.rfile.read(count)
            if len(body) != count:
                self.send_json(400, {"error": "Request body was not fully received."}); return
            if route == "/detect":
                if self.headers.get("Content-Type", "").split(";")[0].strip().casefold() != "application/json":
                    self.send_json(415, {"error": "Text requests require application/json."}); return
                try:
                    payload = json.loads(body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self.send_json(400, {"error": "Request is not valid UTF-8 JSON."}); return
                text = payload.get("text") if isinstance(payload, dict) else None
                if not isinstance(text, str) or not text.strip():
                    self.send_json(400, {"error": "Provide nonempty English body text."}); return
                if len(text) > self.server.limits.max_characters:
                    self.send_json(413, {"error": f"Text exceeds the {self.server.limits.max_characters:,}-character limit; no truncation was performed."}); return
            else:
                filename = unquote(self.headers.get("X-Filename", ""))
                try:
                    text = extract_document(body, filename, max_bytes=self.server.limits.max_request_bytes,
                                            max_characters=self.server.limits.max_characters)
                except DocumentError as exc:
                    self.send_json(400, {"error": str(exc)}); return
            # The detector owns model calibration, language gates and full windows.
            result = self.server.detector.predict(text)
            self.send_json(200, result)
        except (TimeoutError, ConnectionResetError):
            self.send_json(408, {"error": "Request read timed out; retry."})
        except Exception:
            self.send_json(500, {"error": "Detection failed; reduce input size or check the local bundle configuration. Text was not truncated."})
        finally:
            self.server.inference_lock.release()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--max-characters", type=int, default=200_000)
    parser.add_argument("--max-upload-mib", type=int, default=8)
    parser.add_argument("--requests-per-minute", type=int, default=6)
    args = parser.parse_args(argv)
    if min(args.max_characters, args.max_upload_mib, args.requests_per_minute) < 1:
        parser.error("Resource limits must be positive")
    from english_ai_detector import EnglishDetector
    detector = EnglishDetector.load(args.bundle, device=args.device)
    server = DetectorServer((args.host, args.port), detector, Limits(args.max_upload_mib*1024*1024, args.max_characters, args.requests_per_minute))
    print(f"English detector: http://{args.host}:{server.server_port} (local text, no request-body logging)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
