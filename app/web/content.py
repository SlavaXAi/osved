"""Convert stored Telegram markup to a deliberately small safe browser vocabulary."""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit

MEDIA_ASSET = re.compile(
    r"\.(?:jpe?g|png|svg|gif|webp|avif|bmp|ico|tiff?|mp4|webm|mov|mp3|wav)$",
    re.I,
)


def public_url(value: object) -> str | None:
    if not isinstance(value, str) or len(value) > 4096:
        return None
    try:
        parts = urlsplit(value.strip())
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            return None
        if parts.username or parts.password:
            return None
        return parts.geturl()
    except ValueError:
        return None


def evidence_url(value: object) -> str | None:
    """Presentation filter: an image/video asset is not a research article.

    Keep URLs and complete provenance in storage unchanged. PDFs and normal
    article pages remain valid evidence; no guesses from hostnames or query text.
    """
    url = public_url(value)
    if url and not MEDIA_ASSET.search(unquote(urlsplit(url).path)):
        return url
    return None


class SafeHTML(HTMLParser):
    tags = {"b", "strong", "i", "em", "u", "s", "del", "code", "pre", "blockquote", "p", "br"}
    suppressed = {"script", "style", "iframe", "object", "svg", "math", "template"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.output: list[str] = []
        self.stack: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.suppressed:
            self.skip += 1
        if self.skip:
            return
        if tag == "a":
            url = public_url(dict(attrs).get("href"))
            if url:
                self.output.append(
                    f'<a href="{html.escape(url, quote=True)}" target="_blank" '
                    'rel="noopener noreferrer">'
                )
                self.stack.append(tag)
        elif tag in self.tags:
            self.output.append(f"<{tag}>")
            if tag != "br":
                self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in self.suppressed and self.skip:
            self.skip -= 1
            return
        if not self.skip and tag in self.stack:
            while self.stack:
                closing = self.stack.pop()
                self.output.append(f"</{closing}>")
                if closing == tag:
                    break

    def handle_data(self, data: str) -> None:
        if not self.skip:
            self.output.append(html.escape(data))

    def result(self) -> str:
        return "".join(self.output) + "".join(f"</{tag}>" for tag in reversed(self.stack))


def safe_html(value: object) -> str:
    parser = SafeHTML()
    parser.feed(str(value or "")[:100_000])
    return parser.result()


class PlainText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def plain_text(value: object) -> str:
    parser = PlainText()
    parser.feed(safe_html(value))
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()
