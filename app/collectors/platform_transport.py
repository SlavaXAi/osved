"""Bounded, injectable transports for public feeds and paid Monid sensors."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from app.collectors.x_monid import monid_cli_path

MAX_RESPONSE_BYTES = 4_000_000


class PlatformError(RuntimeError):
    def __init__(self, message: str, *, status: str = "failed", cost_usd: float = 0) -> None:
        super().__init__(message)
        self.status = status
        self.cost_usd = cost_usd


@dataclass(frozen=True)
class MonidResponse:
    payload: Any
    cost_usd: float
    run_id: str | None = None


def fetch_public(url: str) -> bytes:
    parsed = urlsplit(url)
    allowed = {"www.reddit.com", "www.youtube.com", "www.v2ex.com"}
    if parsed.scheme != "https" or parsed.hostname not in allowed or parsed.username:
        raise PlatformError("Public sensor URL is outside its platform allowlist")
    request = Request(
        url,
        headers={
            "User-Agent": "OSVED/0.1 (personal awareness reader; low-frequency read-only)",
            "Accept": "application/atom+xml, application/rss+xml, application/json, text/xml",
        },
    )
    try:
        with urlopen(request, timeout=25) as response:
            final = urlsplit(response.geturl())
            if final.scheme != "https" or final.hostname not in allowed:
                raise PlatformError("Platform redirected outside its allowlist")
            if int(response.headers.get("Content-Length") or 0) > MAX_RESPONSE_BYTES:
                raise PlatformError("Platform response is too large", status="schema_changed")
            payload = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        status = (
            "rate_limited"
            if exc.code == 429
            else ("login_required" if exc.code in {401, 403} else "failed")
        )
        raise PlatformError(f"Platform HTTP {exc.code}", status=status) from exc
    except (URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise PlatformError(
            f"Platform network failure: {type(exc).__name__}: {_safe_error(str(reason))}"
        ) from exc
    if len(payload) > MAX_RESPONSE_BYTES:
        raise PlatformError("Platform response is too large", status="schema_changed")
    return payload


def _safe_error(value: str) -> str:
    value = re.sub(r"monid_(?:live|test)_[A-Za-z0-9_-]+", "[REDACTED]", value)
    value = re.sub(r"(?i)(authorization|cookie|api[-_ ]?key)\s*[:=]\s*\S+", r"\1=[REDACTED]", value)
    return value[-600:]


def monid_fetch(
    endpoint: str, query: dict[str, Any], output_dir: Path, *, estimated_cost: float = 0.0015
) -> MonidResponse:
    """One paid request at most; never silently restart an unfinished paid run."""
    executable = monid_cli_path()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"sensor-{uuid.uuid4().hex}.json"
    command = [
        executable,
        "run",
        "-p",
        "tikhub",
        "-e",
        endpoint,
        "--query",
        json.dumps(query, ensure_ascii=False),
        "-w",
        "60",
        "-o",
        str(output_path),
        "-j",
    ]
    environment = {**os.environ, "NO_COLOR": "1"}
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=85, check=False, env=environment
        )
    except subprocess.TimeoutExpired as exc:
        raise PlatformError(
            "Monid run timed out; inspect existing run before retrying",
            status="partial",
            cost_usd=estimated_cost,
        ) from exc
    try:
        receipt = json.loads(result.stdout)
    except (ValueError, TypeError):
        receipt = {}
    if not isinstance(receipt, dict):
        receipt = {}
    run_status = str(receipt.get("status", "")).upper()
    cost_value = receipt.get("cost")
    cost = estimated_cost
    if isinstance(cost_value, dict):
        reported = cost_value.get("value")
        if (
            isinstance(reported, (int, float))
            and not isinstance(reported, bool)
            and math.isfinite(reported)
        ):
            cost = max(0.0, float(reported))
    if run_status == "BLOCKED":
        raise PlatformError(
            "Monid workspace budget/run control blocked the request", status="disabled", cost_usd=0
        )
    if result.returncode:
        detail = _safe_error(result.stderr or result.stdout)
        raise PlatformError(f"Monid failed: {detail}", cost_usd=cost)
    run_id = receipt.get("id") or receipt.get("runId")
    if run_status in {"FAILED", "STOPPED", "TIME_OUT"}:
        raise PlatformError(f"Monid run ended with {run_status}", cost_usd=cost)
    if run_status in {"READY", "RUNNING"}:
        raise PlatformError(
            f"Monid run {run_id or '(unknown ID)'} is still active; inspect it before retrying",
            status="partial",
            cost_usd=cost,
        )
    if not output_path.is_file():
        raise PlatformError(
            "Monid has not produced a completed result; do not restart blindly",
            status="partial",
            cost_usd=cost,
        )
    if output_path.stat().st_size > MAX_RESPONSE_BYTES:
        raise PlatformError("Monid response is too large", status="schema_changed", cost_usd=cost)
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise PlatformError(
            "Monid wrote invalid JSON", status="schema_changed", cost_usd=cost
        ) from exc
    return MonidResponse(payload, cost, run_id)


def unwrap_data(payload: Any) -> Any:
    """Remove transport envelopes, retaining platform result containers."""
    for _ in range(5):
        if not isinstance(payload, dict):
            break
        status = str(payload.get("status", "")).lower()
        code = payload.get("code")
        if code in {401, 403, 429}:
            raise PlatformError(
                f"Platform returned {code}",
                status="rate_limited" if code == 429 else "login_required",
            )
        if status in {"error", "failed"} or (isinstance(code, int) and code >= 400):
            raise PlatformError("Platform returned an error envelope")
        if "output" in payload and len(payload) <= 12:
            payload = payload["output"]
        elif "data" in payload and (
            "code" in payload
            or "request_id" in payload
            or set(payload) <= {"data", "status", "message"}
        ):
            payload = payload["data"]
        else:
            break
    return payload
