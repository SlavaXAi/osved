"""Loopback-only HTTP bridge for the optional OSVED browser extension."""

from __future__ import annotations

import hmac
import ipaddress
import json
import re
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from app.collectors.browser_queue import BrowserTaskError, BrowserTaskQueue

MAX_BODY_BYTES = 512 * 1024


class BrowserBridgeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        queue: BrowserTaskQueue,
        *,
        extension_id: str,
        pairing_token: str | None = None,
        port: int = 8769,
    ) -> None:
        if not re.fullmatch(r"[a-p]{32}", extension_id):
            raise ValueError("A Chromium extension ID of 32 lowercase a-p characters is required")
        token = pairing_token or secrets.token_urlsafe(32)
        if not re.fullmatch(r"[A-Za-z0-9_-]{32,128}", token):
            raise ValueError("Pairing token must be 32-128 URL-safe characters")
        if not 0 <= port <= 65535:
            raise ValueError("Invalid port")
        self.queue = queue
        self.pairing_token = token
        self.allowed_origin = f"chrome-extension://{extension_id}"
        super().__init__(("127.0.0.1", port), _BridgeHandler)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}"


class _BridgeHandler(BaseHTTPRequestHandler):
    server: BrowserBridgeServer
    server_version = "OSVED-Local-Sensor/1"
    sys_version = ""

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(5)

    def log_message(self, format: str, *args: Any) -> None:
        # Headers, bodies and URLs must never accidentally reach application logs.
        return

    def _reply(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        if self.headers.get("Origin") == self.server.allowed_origin:
            self.send_header("Access-Control-Allow-Origin", self.server.allowed_origin)
            self.send_header("Vary", "Origin")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _allow_request(self, *, require_token: bool = True) -> bool:
        peer = ipaddress.ip_address(self.client_address[0])
        hosts = self.headers.get_all("Host") or []
        origins = self.headers.get_all("Origin") or []
        if not peer.is_loopback or hosts != [f"127.0.0.1:{self.server.server_port}"]:
            self._reply(403, {"error": "host_not_allowed"})
            return False
        if origins != [self.server.allowed_origin]:
            self._reply(403, {"error": "origin_not_allowed"})
            return False
        if require_token:
            values = self.headers.get_all("Authorization") or []
            expected = f"Bearer {self.server.pairing_token}"
            if len(values) != 1 or not hmac.compare_digest(values[0].encode(), expected.encode()):
                self._reply(401, {"error": "unauthorized"})
                return False
        return True

    def do_OPTIONS(self) -> None:
        if not self._allow_request(require_token=False):
            return
        if self.headers.get("Access-Control-Request-Method") not in {"GET", "POST"}:
            self._reply(405, {"error": "method_not_allowed"})
            return
        headers = {
            header.strip().lower()
            for header in self.headers.get("Access-Control-Request-Headers", "").split(",")
            if header.strip()
        }
        if headers - {"authorization", "content-type"}:
            self._reply(403, {"error": "headers_not_allowed"})
            return
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", self.server.allowed_origin)
        self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Max-Age", "300")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        if not self._allow_request():
            return
        if self.path != "/v1/health":
            self._reply(404, {"error": "unknown_endpoint"})
            return
        self._reply(200, {"status": "ok", "capability": "x_visible_for_you_read_only"})

    def do_POST(self) -> None:
        if not self._allow_request():
            return
        if self.path not in {"/v1/tasks/claim", "/v1/tasks/result", "/v1/tasks/fail"}:
            self._reply(404, {"error": "unknown_endpoint"})
            return
        lengths = self.headers.get_all("Content-Length") or []
        if self.headers.get("Transfer-Encoding") or len(lengths) != 1:
            self._reply(411, {"error": "content_length_required"})
            return
        try:
            length = int(lengths[0])
        except ValueError:
            self._reply(400, {"error": "invalid_content_length"})
            return
        if length < 0 or length > MAX_BODY_BYTES:
            self._reply(413, {"error": "body_too_large"})
            return
        if self.headers.get_content_type() != "application/json":
            self._reply(415, {"error": "json_required"})
            return
        try:
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("incomplete body")
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("object required")
        except (ValueError, UnicodeDecodeError, TimeoutError):
            self._reply(400, {"error": "invalid_json"})
            return
        try:
            if self.path == "/v1/tasks/claim":
                if data:
                    raise BrowserTaskError("unexpected_claim_fields")
                self._reply(200, {"task": self.server.queue.claim()})
            elif self.path == "/v1/tasks/result":
                if set(data) != {"task_id", "claim_token", "result"}:
                    raise BrowserTaskError("unexpected_result_fields")
                if not isinstance(data["task_id"], str) or not isinstance(data["claim_token"], str):
                    raise BrowserTaskError("invalid_claim")
                created = self.server.queue.stage(
                    data["task_id"], data["claim_token"], data["result"],
                )
                self._reply(200, {"status": "staged", "replayed": not created})
            else:
                if set(data) != {"task_id", "claim_token", "error_code"}:
                    raise BrowserTaskError("unexpected_failure_fields")
                if not all(isinstance(value, str) for value in data.values()):
                    raise BrowserTaskError("invalid_failure_fields")
                self.server.queue.fail(data["task_id"], data["claim_token"], data["error_code"])
                self._reply(200, {"status": "failed"})
        except BrowserTaskError as error:
            self._reply(409, {"error": str(error)})

    def do_PUT(self) -> None:
        self._reply(405, {"error": "method_not_allowed"})

    do_DELETE = do_PUT
    do_PATCH = do_PUT
