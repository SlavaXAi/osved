"""Grounded video synopsis in readable, self-contained Telegram prose."""

from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path

from app.video.report_format import MODEL, PRICES, SECRET, request_payload
from app.video.transcripts import youtube_video_id

PROMPT = Path(__file__).with_name("tldr_prompt.txt").read_text(encoding="utf-8").rstrip() + "\n"
VERSION = hashlib.sha256((PROMPT + MODEL + "tldr-v2").encode()).hexdigest()[:16]
# Formatting changes must not make a paid, usable synopsis disappear from the cache.
LEGACY_VERSIONS = ("c1b14664e329d27f",)
TELEGRAM_TEXT_LIMIT = 4096
TLDR_TEXT_LIMIT = 3900
GENERATED_TLDR_TEXT_LIMIT = 3500
_SENTENCE_END = re.compile(r"[.!?…][\"'»”’)\]]*$")
_ABBREVIATIONS = {
    "г",
    "гг",
    "т",
    "п",
    "д",
    "ул",
    "им",
    "в",
    "рис",
    "см",
    "стр",
    "руб",
    "млн",
    "млрд",
    "тыс",
    "др",
    "пр",
    "т.д",
    "т.п",
    "т.е",
}


def _text_units(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def extract_tldr(
    original: str, *, full_report: bool = False, require_finished_sentences: bool = False
) -> str:
    if not isinstance(original, str) or not re.match(r"\A## TL;DR(?:\s|$)", original):
        raise ValueError("invalid_output")
    body = original.removeprefix("## TL;DR").strip()
    if full_report:
        parts = re.split(r"(?m)^## Подробно\s*$", body)
        if len(parts) != 2 or not parts[1].strip():
            raise ValueError("invalid_output")
        body = parts[0].strip()
    paragraphs = [" ".join(part.split()) for part in re.split(r"\n\s*\n", body)]
    if (
        not body
        or not 1 <= len(paragraphs) <= 4
        or re.search(r"(?m)^\s*(?:#|[-*•]\s|\d+[.)]\s)", body)
        or re.search(r"https?://|file://|javascript:|\[\[|\bE\d{4}\b", body, re.I)
        or (require_finished_sentences and re.search(r"</?[A-Za-z][^>]*>|```|\*\*", body))
        or SECRET.search(body)
        or (require_finished_sentences and any(not _SENTENCE_END.search(p) for p in paragraphs))
    ):
        raise ValueError("invalid_output")
    body = "\n\n".join(paragraphs)
    # New output leaves room for the title/footer; cached prose retains its old limit.
    limit = GENERATED_TLDR_TEXT_LIMIT if require_finished_sentences else TLDR_TEXT_LIMIT
    if _text_units(body) > limit:
        raise ValueError("invalid_output")
    return body


def format_legacy_paragraphs(text: str) -> str:
    """Add paragraph breaks to long cached prose, without adding or cutting words."""
    if "\n\n" in text or len(text) < 700:
        return text
    boundaries = []
    for match in re.finditer(r"[.!?…]+[\"'»”’)\]]*(\s+)(?=[\"«“(]*[A-ZА-ЯЁ0-9])", text):
        punctuation = text[match.start() : match.start(1)]
        token = re.search(r"([\w.]+)\.$", text[: match.start(1)])
        if punctuation == "." and token:
            word = token[1].rstrip(".")
            if word.casefold() in _ABBREVIATIONS or (len(word) == 1 and word.isalpha()):
                continue
        boundaries.append((match.start(1), match.end(1)))
    # A few long sentences still benefit from a break. Do not create a tiny tail.
    if len(boundaries) < 5:
        balanced = [
            boundary
            for boundary in boundaries
            if boundary[0] >= 180 and len(text) - boundary[1] >= 180
        ]
        if not balanced:
            return text
        split = min(balanced, key=lambda boundary: abs(boundary[0] - len(text) / 2))
        return text[: split[0]] + "\n\n" + text[split[1] :]
    # Three paragraphs need enough real sentence boundaries to avoid tiny fragments.
    first_index = min(
        range(1, len(boundaries) - 2), key=lambda i: abs(boundaries[i][0] - len(text) / 3)
    )
    second_index = min(
        range(first_index + 2, len(boundaries)),
        key=lambda i: abs(boundaries[i][0] - len(text) * 2 / 3),
    )
    first, second = boundaries[first_index], boundaries[second_index]
    return text[: first[0]] + "\n\n" + text[first[1] : second[0]] + "\n\n" + text[second[1] :]


def telegram_tldr(text: str, *, title: str | None = None, video_id: str | None = None) -> str:
    """Escape prose and source metadata, then validate the complete displayed text."""
    text = extract_tldr("## TL;DR\n\n" + text)
    text = format_legacy_paragraphs(text)
    if title is not None and not isinstance(title, str):
        raise ValueError("invalid_metadata")
    heading = " ".join((title or "TL;DR").split()) or "TL;DR"
    if _text_units(heading) > 500 or SECRET.search(heading):
        raise ValueError("invalid_metadata")
    rendered = "<b>" + html.escape(heading) + "</b>\n\n" + html.escape(text)
    visible = heading + "\n\n" + text
    if video_id is not None:
        video = youtube_video_id(video_id)
        rendered += f'\n\n<a href="https://www.youtube.com/watch?v={video}">Смотреть видео</a>'
        visible += "\n\nСмотреть видео"
    if _text_units(visible) > TELEGRAM_TEXT_LIMIT:
        raise ValueError("invalid_output")
    return rendered


def tldr_request(source: dict) -> tuple[dict, float]:
    payload, _ = request_payload(source)
    payload["messages"][0]["content"] = PROMPT
    payload["max_tokens"] = 4096
    prompt_bytes = len(json.dumps(payload, ensure_ascii=False).encode()) + 4096
    if prompt_bytes + payload["max_tokens"] > 1_000_000:
        raise ValueError("insufficient_context")
    reserve = (prompt_bytes * PRICES["prompt"] + payload["max_tokens"] * PRICES["completion"]) / 1e6
    return payload, reserve
