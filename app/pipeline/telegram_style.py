"""Presentation policy for news: structured prose in, Telegram HTML out.

Cleanup is cosmetic only: never shorten paths, reactivate defanged domains,
change numbers, or invent hyperlinks. Editorial selection belongs to the writer.
"""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from urllib.parse import urlsplit

NEWS_STYLE_INSTRUCTION = (
    "Оформление для Telegram: итог отправляется с parse_mode=HTML, но HTML-теги "
    "расставляет программа, не модель. В JSON-полях возвращай обычный русский текст "
    "без Markdown и HTML: никаких **звёздочек**, обратных кавычек, решёток-заголовков, "
    "подчёркиваний для курсива, fenced code blocks, таблиц и декоративных разделителей. "
    "Не вставляй хештеги без отдельного запроса. Один абзац — одна смысловая мысль, "
    "обычно 1–3 предложения. Заголовок будет жирным, абзацы разделены пустой строкой, "
    "ссылки встроены в осмысленные фразы по source_anchors. Не дублируй это оформление "
    "в тексте и не добавляй строку «Источник». Highlight — максимум одна самостоятельная "
    "важная мысль или атрибутированная цитата, только если она действительно нужна; "
    "иначе null. Не превращай каждый абзац в список, цитату или подзаголовок. "
    "Это новости для любознательного читателя, не журнал технических логов: полные пути "
    "файлов, ключи реестра, хеши, адреса управляющих серверов, IOC, команды и длинные "
    "идентификаторы не переносить в карточку ради технического вида. Объясни их роль "
    "обычными словами, сохранив подтверждённый смысл; краткое имя файла или уязвимости "
    "оставляй лишь если оно необходимо для понимания события. Существенные показатели, "
    "названия продуктов и фактические ограничения сохраняй. Limitations — только "
    "реальное ограничение результата, продукта или исследования, а не последняя порция "
    "технических подробностей. Не пиши шаблонный хвост «Что важно учитывать» и "
    "не повторяй оговорки ради формальности. "
)


def clean_news_text(value: str) -> str:
    """Remove accidental formatting delimiters, not their factual contents."""
    value = re.sub(r"(?m)^\s{0,3}```[^\n]*$", "", value)
    value = re.sub(r"(?m)^[ \t]{0,3}#{1,6}[ \t]+", "", value)
    # Keep identifiers (snake_case), arithmetic and ordinary punctuation intact.
    for marker in ("***", "**", "__", "~~", "*", "_"):
        token = re.escape(marker)
        character = re.escape(marker[0])
        value = re.sub(
            rf"(?<![\w{character}]){token}(?!{character})(?=\S)"
            rf"([^\n]+?)(?<=\S)(?<!{character}){token}(?![\w{character}])",
            lambda match, marker=marker: match[0] if marker == "__" and re.fullmatch(
                r"[A-Za-z][A-Za-z0-9_]*", match[1]
            ) else match[1],
            value,
        )
    value = value.replace("`", "")
    # Models occasionally include these despite the plain-text JSON contract.
    value = re.sub(r"</?(?:b|strong|i|em|u|s|code|pre)>", "", value, flags=re.I)
    return value


class _NewsHTML(HTMLParser):
    """Keep renderer-owned entities and links when cleaning old cached cards."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.stack: list[tuple[str, str | None]] = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        emitted = None
        if tag == "a":
            url = attrs.get("href") or ""
            try:
                parts = urlsplit(url)
                valid = parts.scheme in {"http", "https"} and bool(parts.netloc)
            except ValueError:
                valid = False
            if valid and not any(t == "a" for _, t in self.stack):
                emitted = "a"
                self.parts.append(f'<a href="{html.escape(url, quote=True)}">')
        elif tag in {"b", "strong", "code", "pre", "blockquote"}:
            emitted = "b" if tag == "strong" else tag
            self.parts.append(f"<{emitted}>")
        # Legacy full-paragraph italic footers become ordinary prose. Inline
        # emphasis is deliberately restrained; the title/labels remain bold.
        if tag not in {"br", "hr", "img"}:
            self.stack.append((tag, emitted))
        elif tag == "br":
            self.parts.append("\n")

    def handle_endtag(self, tag):
        indices = [i for i, (original, _) in enumerate(self.stack) if original == tag]
        if not indices:
            return
        while len(self.stack) > indices[-1]:
            _, emitted = self.stack.pop()
            if emitted:
                self.parts.append(f"</{emitted}>")

    def handle_data(self, data):
        if not any(emitted in {"code", "pre"} for _, emitted in self.stack):
            data = clean_news_text(data)
            data = re.sub(r"^Что важно учитывать:\s*", "", data)
        self.parts.append(html.escape(data))

    def finish(self) -> str:
        self.close()
        while self.stack:
            _, emitted = self.stack.pop()
            if emitted:
                self.parts.append(f"</{emitted}>")
        return "".join(self.parts).strip()


def clean_news_html(value: str) -> str:
    """Apply the same formatting at generation and send time, without paid rewrites."""
    parser = _NewsHTML()
    parser.feed(value)
    return parser.finish()
