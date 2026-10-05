"""The approved PDF report contract. No network, generation or editorial rewriting."""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
from pathlib import Path

MODEL = "google/gemini-3.8-flash"
PROVIDER = "google-ai-studio"
MAX_OUTPUT = 32768
PRICES = {"prompt": 0.75, "completion": 3.75}
PROMPT = Path(__file__).with_name("report_prompt.txt").read_text(encoding="utf-8").rstrip() + "\n"
VERSION = hashlib.sha256((PROMPT + MODEL + "pdf-v1").encode()).hexdigest()[:16]
SECRET = re.compile(r"(?:sk-or-v1-|monid_live_|Bearer\s+)[A-Za-z0-9_.-]{12,}", re.I)
DASHES = "\u2010\u2011\u2012\u2013\u2014\u2015"


def clock(seconds: float) -> str:
    hour, rest = divmod(int(seconds), 3600)
    minute, second = divmod(rest, 60)
    return f"{hour}:{minute:02}:{second:02}" if hour else f"{minute:02}:{second:02}"


def source_payload(raw: object, video_id: str) -> dict:
    if isinstance(raw, dict) and "output" in raw:
        raw = raw["output"]
    if not isinstance(raw, list) or len(raw) != 1 or not isinstance(raw[0], dict):
        raise ValueError("unavailable")
    item = raw[0]
    if item.get("video_id") not in (None, video_id):
        raise ValueError("source_mismatch")
    segments = item.get("transcript")
    if not isinstance(segments, list) or not segments:
        raise ValueError("unavailable")
    groups, buffer, size, joined = [], [], 0, []
    last_start = -1.0
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("invalid_transcript")
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        start = float(segment["start"])
        end = float(segment.get("end", start + float(segment.get("duration", 0))))
        if not all(math.isfinite(v) for v in (start, end)) or start < last_start or end < start:
            raise ValueError("invalid_timestamps")
        last_start = start
        if buffer and size >= 1000:
            groups.append(buffer)
            buffer, size = [], 0
        buffer.append({"start": start, "end": end, "text": text})
        joined.append(text)
        size += len(text) + 1
    if buffer:
        groups.append(buffer)
    if not groups or len(" ".join(joined)) > 600_000:
        raise ValueError("insufficient_context")
    evidence = {
        f"E{i:04}": {
            "start": g[0]["start"],
            "end": g[-1]["end"],
            "text": " ".join(s["text"] for s in g),
        }
        for i, g in enumerate(groups, 1)
    }
    assert " ".join(joined) == " ".join(e["text"] for e in evidence.values())
    keys = (
        "title",
        "description",
        "channel_name",
        "published_at",
        "duration_seconds",
        "language",
        "is_auto_generated",
    )
    metadata = {k: item.get(k) for k in keys}
    metadata["video_id"] = video_id
    duration = float(metadata.get("duration_seconds") or 0)
    if not math.isfinite(duration) or duration <= 0 or groups[-1][-1]["end"] < duration * 0.90:
        raise ValueError("incomplete_transcript")
    if len(str(metadata.get("title") or "")) > 500:
        raise ValueError("invalid_metadata")
    blocks = "\n\n".join(
        f"[{eid} @ {clock(e['start'])}] {e['text']}" for eid, e in evidence.items()
    )
    user = (
        "МЕТАДАННЫЕ ВИДЕО:\n"
        + json.dumps(metadata, ensure_ascii=False)
        + "\n\nПОЛНАЯ ДОСТУПНАЯ РАСШИФРОВКА:\n"
        + blocks
    )
    return {
        "metadata": metadata,
        "evidence": evidence,
        "user": user,
        "source_segment_count": len(segments),
        "included_nonempty_segments": len(joined),
        "input_sha256": hashlib.sha256(user.encode()).hexdigest(),
    }


def request_payload(source: dict) -> tuple[dict, float]:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": source["user"]},
        ],
        "temperature": 0.2,
        "max_tokens": MAX_OUTPUT,
        "stream": False,
        "reasoning": {"effort": "low", "exclude": True},
        "provider": {
            "only": [PROVIDER],
            "order": [PROVIDER],
            "allow_fallbacks": False,
            "require_parameters": True,
            "max_price": PRICES,
        },
    }
    # Deliberately conservative: bytes upper-bound tokens; never silently truncate input.
    input_envelope = len(json.dumps(payload, ensure_ascii=False).encode()) + 4096
    if input_envelope + MAX_OUTPUT > 1_000_000:
        raise ValueError("insufficient_context")
    reserve = (input_envelope * PRICES["prompt"] + MAX_OUTPUT * PRICES["completion"]) / 1e6
    return payload, reserve


def render_citations(original: str, source: dict) -> str:
    if not original.startswith("## TL;DR") or SECRET.search(original):
        raise ValueError("invalid_output")
    parts = re.split(r"(?m)^## Подробно\s*$", original)
    if len(parts) != 2 or not parts[1].strip():
        raise ValueError("invalid_output")
    lead = parts[0].removeprefix("## TL;DR").strip()
    if not lead or len(re.split(r"\n\s*\n", lead)) != 1:
        raise ValueError("invalid_output")
    # Untrusted text cannot introduce external links or executable PDF content.
    if re.search(r"https?://|file://|javascript:", original, re.I):
        raise ValueError("unexpected_url")

    def replace(match):
        inner = match[1]
        if not re.fullmatch(r"[\[\]\s,;–—\-E0-9]+", inner):
            return match[0]
        identifiers = []
        # Gemini sometimes groups IDs or ranges. Use the same deterministic expansion
        # as the approved report renderer, without touching any surrounding prose.
        for item in re.finditer(r"(E\d{4})(?:\]?\s*[–—-]\s*\[?(E\d{4}))?", inner):
            first, last = item.groups()
            if last:
                a, b = int(first[1:]), int(last[1:])
                if b < a or b - a > 1000:
                    raise ValueError("invalid_evidence_range")
                identifiers.extend(f"E{i:04}" for i in range(a, b + 1))
            else:
                identifiers.append(first)
        if not identifiers:
            return match[0]
        anchors = []
        for eid in dict.fromkeys(identifiers):
            if eid not in source["evidence"]:
                raise ValueError("unknown_evidence")
            start = int(source["evidence"][eid]["start"])
            anchors.append(
                f"[{clock(start)}](https://www.youtube.com/watch?v={source['metadata']['video_id']}&t={start}s)"
            )
        return " ".join(anchors)

    rendered = re.sub(r"\[\[([^\n]*?)\]\]", replace, original)
    if re.search(r"\bE\d{4}\b", rendered):
        raise ValueError("invalid_evidence_marker")
    return rendered


def typography(text: str) -> str:
    return text.translate({ord(c): "-" for c in DASHES})


def markdown_blocks(text: str) -> list[tuple[str, str]]:
    """Keep adjacent headings and list items separate, even without blank lines."""
    blocks, pending = [], []

    def flush():
        if pending:
            blocks.append(("Body", " ".join(pending)))
            pending.clear()

    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            flush()
        elif heading := re.fullmatch(r"(#{2,3}) (.+)", line):
            flush()
            blocks.append((f"H{len(heading[1])}", heading[2]))
        elif re.match(r"^(?:[-*+] |\d+[.)] )", line):
            flush()
            pending.append(re.sub(r"^[-*+] ", "- ", line))
        else:
            pending.append(line)
    flush()
    return blocks


def create_pdf(path: Path, original: str, source: dict) -> dict:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    rendered = render_citations(original, source)
    candidates = [
        (
            Path("/usr/share/fonts/truetype/liberation2"),
            "LiberationSans-Regular.ttf",
            "LiberationSans-Bold.ttf",
            "LiberationSans-Italic.ttf",
            "LiberationSans-BoldItalic.ttf",
        ),
        (
            Path("/usr/share/fonts/truetype/liberation"),
            "LiberationSans-Regular.ttf",
            "LiberationSans-Bold.ttf",
            "LiberationSans-Italic.ttf",
            "LiberationSans-BoldItalic.ttf",
        ),
        (
            Path("/System/Library/Fonts/Supplemental"),
            "Arial.ttf",
            "Arial Bold.ttf",
            "Arial Italic.ttf",
            "Arial Bold Italic.ttf",
        ),
    ]
    fonts = next(
        (entry for entry in candidates if all((entry[0] / f).is_file() for f in entry[1:])), None
    )
    if fonts is None:
        raise ValueError("missing_cyrillic_fonts")
    for name, file in zip(
        ("OSVEDReport", "OSVEDReport-Bold", "OSVEDReport-Italic", "OSVEDReport-BoldItalic"),
        fonts[1:],
        strict=True,
    ):
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(fonts[0] / file)))
    pdfmetrics.registerFontFamily(
        "OSVEDReport",
        normal="OSVEDReport",
        bold="OSVEDReport-Bold",
        italic="OSVEDReport-Italic",
        boldItalic="OSVEDReport-BoldItalic",
    )
    body = ParagraphStyle(
        "Body",
        fontName="OSVEDReport",
        fontSize=10.5,
        leading=14.5,
        spaceAfter=6,
        allowWidows=0,
        allowOrphans=0,
    )
    styles = {
        "Body": body,
        "Title": ParagraphStyle(
            "Title",
            parent=body,
            fontName="OSVEDReport-Bold",
            fontSize=21,
            leading=25,
            spaceAfter=11,
        ),
        "H2": ParagraphStyle(
            "H2",
            parent=body,
            fontName="OSVEDReport-Bold",
            fontSize=13,
            leading=17,
            spaceBefore=11,
            keepWithNext=True,
        ),
        "H3": ParagraphStyle(
            "H3",
            parent=body,
            fontName="OSVEDReport-Bold",
            fontSize=11.4,
            leading=15,
            spaceBefore=8,
            keepWithNext=True,
        ),
        "Meta": ParagraphStyle(
            "Meta", parent=body, fontSize=9, leading=12.3, textColor=colors.HexColor("#505050")
        ),
    }

    def inline(text):
        safe = html.escape(typography(text))
        safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
        safe = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", safe)
        safe = re.sub(r"`([^`]+)`", r"\1", safe)
        return re.sub(
            r"\[([^\]]+)\]\((https://www.youtube.com/watch\?[^)]+)\)",
            r'<link href="\2" color="#233E58"><u>\1</u></link>',
            safe,
        )

    def para(text, style="Body"):
        return Paragraph(inline(text), styles[style])

    meta = source["metadata"]
    url = f"https://www.youtube.com/watch?v={meta['video_id']}"
    title = str(meta.get("title") or "Резюме видео")
    channel = str(meta.get("channel_name") or "YouTube")
    metadata = (
        f"{channel} · {str(meta.get('published_at') or '')[:10]} · "
        f"{clock(float(meta['duration_seconds']))} · Gemini 3.8 Flash"
    )
    story = [
        Paragraph(html.escape(typography(title)), styles["Title"]),
        Paragraph(html.escape(typography(metadata)), styles["Meta"]),
        para(f"[Оригинал]({url})", "Meta"),
    ]
    for style, text in markdown_blocks(rendered):
        story.append(para(text, style))
    note = (
        "Текст Gemini 3.8 Flash по полной расшифровке видео. Промпт B с расширенным TL;DR. "
        "Без редакторских исправлений. Таймкоды добавлены автоматически."
    )
    story += [Spacer(1, 8), para(note, "Meta")]

    def frame(canvas, doc):
        canvas.saveState()
        canvas.setFont("OSVEDReport", 8)
        canvas.setFillColor(colors.HexColor("#505050"))
        if doc.page > 1:
            canvas.drawString(19 * mm, A4[1] - 11 * mm, "ОСВЕД / Резюме видео")
        canvas.drawString(19 * mm, 10 * mm, "Gemini 3.8 Flash / Без редактуры")
        canvas.drawRightString(A4[0] - 19 * mm, 10 * mm, str(doc.page))
        canvas.restoreState()

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".partial.pdf")
    doc = SimpleDocTemplate(
        str(temp),
        pagesize=A4,
        leftMargin=19 * mm,
        rightMargin=19 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=title,
        author="Gemini 3.8 Flash / unedited model response",
    )
    doc.build(story, onFirstPage=frame, onLaterPages=frame)
    if not temp.read_bytes().startswith(b"%PDF-") or temp.stat().st_size > 40_000_000:
        raise ValueError("invalid_pdf")
    temp.replace(path)
    return {
        "body_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "pdf_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "font": fonts[1],
        "editorial_revision": False,
        "model": MODEL,
        "prompt_version": VERSION,
    }
