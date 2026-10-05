"""Same grounded report contract, but one standalone Telegram paragraph, no PDF."""

from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path

from app.video.report_format import MODEL, PRICES, SECRET, request_payload

PROMPT = Path(__file__).with_name("tldr_prompt.txt").read_text(encoding="utf-8").rstrip() + "\n"
VERSION = hashlib.sha256((PROMPT + MODEL + "tldr-v1").encode()).hexdigest()[:16]


def extract_tldr(original: str, *, full_report: bool = False) -> str:
    if not isinstance(original, str) or not original.startswith("## TL;DR"):
        raise ValueError("invalid_output")
    body = original.removeprefix("## TL;DR").strip()
    if full_report:
        parts = re.split(r"(?m)^## Подробно\s*$", body)
        if len(parts) != 2 or not parts[1].strip():
            raise ValueError("invalid_output")
        body = parts[0].strip()
    if (
        not body
        or len(re.split(r"\n\s*\n", body)) != 1
        or re.search(r"(?m)^\s*(?:#|[-*•]\s|\d+[.)]\s)", body)
        or re.search(r"https?://|file://|javascript:|\[\[|\bE\d{4}\b", body, re.I)
        or SECRET.search(body)
    ):
        raise ValueError("invalid_output")
    body = " ".join(body.splitlines())
    # Telegram's actual text limit, not a target word count. Never crop the answer.
    if len(body.encode("utf-16-le")) // 2 > 3900:
        raise ValueError("invalid_output")
    return body


def telegram_tldr(text: str) -> str:
    return "<b>TL;DR</b>\n\n" + html.escape(text)


def tldr_request(source: dict) -> tuple[dict, float]:
    payload, _ = request_payload(source)
    payload["messages"][0]["content"] = PROMPT
    payload["max_tokens"] = 4096
    prompt_bytes = len(json.dumps(payload, ensure_ascii=False).encode()) + 4096
    if prompt_bytes + payload["max_tokens"] > 1_000_000:
        raise ValueError("insufficient_context")
    reserve = (prompt_bytes * PRICES["prompt"] + payload["max_tokens"] * PRICES["completion"]) / 1e6
    return payload, reserve
