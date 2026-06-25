"""Vercel serverless entrypoint for the demo dashboard.

Wraps the existing dashboard server logic (dashboard/server.py) but forces
include_live=False so the demo never makes outbound yfinance/network calls
on every request — it only serves the bundled snapshot in storage/.
"""

import json
import mimetypes
import sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard.server import snapshot, STATIC_DIR  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/snapshot":
            qs = parse_qs(parsed.query)
            run_id = (qs.get("run") or [None])[0]
            self._json(snapshot(run_id=run_id, include_live=False))
            return
        if parsed.path == "/api/history":
            self._json({"dates": [], "series": [], "note": "live history disabled in demo"})
            return
        self._static(parsed.path)

    def _json(self, payload):
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _static(self, path):
        if path in ("", "/"):
            path = "/index.html"
        target = (STATIC_DIR / path.lstrip("/")).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.exists() or not target.is_file():
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
