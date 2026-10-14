"""Pure paragraph pagination of renderer-owned Telegram HTML; no text cropping."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urlsplit

TELEGRAM_MESSAGE_LIMIT = 4096
_STYLE_TAGS = {"b", "strong", "i", "em", "u", "ins", "s", "strike", "del", "pre"}
_NAMED_ENTITIES = {"lt", "gt", "amp", "quot"}


class TelegramPagingError(ValueError):
    pass


def utf16_units(value: str) -> int:
    try:
        return len(value.encode("utf-16-le")) // 2
    except UnicodeError as exc:
        raise TelegramPagingError("Telegram text contains invalid Unicode") from exc


@dataclass(frozen=True)
class _Paragraph:
    rendered: str
    visible: str
    headline: bool


class _ParagraphParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.paragraphs: list[_Paragraph] = []
        self.stack: list[str] = []
        self.rendered: list[str] = []
        self.visible: list[str] = []
        self.roots: list[str] = []
        self.outside_text = False

    def _flush(self):
        rendered = "".join(self.rendered).strip()
        visible = "".join(self.visible)
        if rendered and visible.strip():
            self.paragraphs.append(
                _Paragraph(
                    rendered, visible, self.roots in (["b"], ["strong"]) and not self.outside_text
                )
            )
        self.rendered, self.visible, self.roots = [], [], []
        self.outside_text = False

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if len(attributes) != len(attrs):
            raise TelegramPagingError("Telegram HTML contains duplicate attributes")
        rendered = f"<{tag}>"
        if tag == "a":
            url = attributes.get("href") or ""
            try:
                parts = urlsplit(url)
                valid = parts.scheme in {"http", "https"} and bool(parts.netloc)
            except ValueError:
                valid = False
            if (
                not valid
                or set(attributes) != {"href"}
                or any(ord(character) < 32 for character in url)
                or "a" in self.stack
            ):
                raise TelegramPagingError("Telegram HTML contains an invalid source link")
            rendered = f'<a href="{html.escape(url, quote=True)}">'
        elif tag == "blockquote":
            if attributes and set(attributes) != {"expandable"}:
                raise TelegramPagingError(
                    "Telegram HTML contains unsupported blockquote attributes"
                )
            rendered = "<blockquote expandable>" if attributes else "<blockquote>"
        elif tag == "code":
            language = attributes.get("class") or ""
            if attributes and not (
                set(attributes) == {"class"} and re.fullmatch(r"language-[A-Za-z0-9_+-]+", language)
            ):
                raise TelegramPagingError("Telegram HTML contains unsupported code attributes")
            rendered = f'<code class="{language}">' if attributes else "<code>"
        elif tag not in _STYLE_TAGS or attributes:
            raise TelegramPagingError("Telegram HTML contains an unsupported tag or attributes")
        if not self.stack:
            self.roots.append(tag)
        self.stack.append(tag)
        self.rendered.append(rendered)

    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1] != tag:
            raise TelegramPagingError("Telegram HTML tags are not balanced")
        self.stack.pop()
        self.rendered.append(f"</{tag}>")

    def handle_startendtag(self, tag, attrs):
        raise TelegramPagingError("Telegram HTML contains an unsupported self-closing tag")

    def handle_data(self, data):
        if any(character in data for character in "<>&"):
            raise TelegramPagingError(
                "Telegram HTML text must escape angle brackets and ampersands"
            )
        parts = [data] if self.stack else re.split(r"(\n[ \t\r]*\n+)", data)
        for part in parts:
            if not self.stack and re.fullmatch(r"\n[ \t\r]*\n+", part):
                self._flush()
            else:
                self.rendered.append(part)
                self.visible.append(part)
                if not self.stack and part.strip():
                    self.outside_text = True

    def handle_entityref(self, name):
        if name not in _NAMED_ENTITIES:
            raise TelegramPagingError("Telegram HTML contains an unsupported named entity")
        self.rendered.append(f"&{name};")
        self.visible.append(html.unescape(f"&{name};"))
        if not self.stack:
            self.outside_text = True

    def handle_charref(self, name):
        try:
            value = int(name[1:], 16) if name[:1].casefold() == "x" else int(name)
        except ValueError as exc:
            raise TelegramPagingError(
                "Telegram HTML contains an invalid character reference"
            ) from exc
        if not 0 < value <= 0x10FFFF or 0xD800 <= value <= 0xDFFF:
            raise TelegramPagingError("Telegram HTML contains an invalid character reference")
        self.rendered.append(f"&#{value};")
        self.visible.append(chr(value))
        if not self.stack:
            self.outside_text = True

    def handle_comment(self, data):
        raise TelegramPagingError("Telegram HTML comments are not supported")

    def handle_decl(self, decl):
        raise TelegramPagingError("Telegram HTML declarations are not supported")

    def handle_pi(self, data):
        raise TelegramPagingError("Telegram HTML processing instructions are not supported")

    def finish(self):
        self.close()
        if self.stack:
            raise TelegramPagingError("Telegram HTML tags are not balanced")
        self._flush()
        if not self.paragraphs:
            raise TelegramPagingError("Telegram HTML has no visible text")
        return self.paragraphs


def _paragraphs(value: str) -> list[_Paragraph]:
    if not isinstance(value, str):
        raise TelegramPagingError("Telegram HTML must be text")
    parser = _ParagraphParser()
    parser.feed(value)
    return parser.finish()


def telegram_html_visible_text(value: str) -> str:
    return "\n\n".join(paragraph.visible for paragraph in _paragraphs(value))


def telegram_html_visible_units(value: str) -> int:
    return utf16_units(telegram_html_visible_text(value))


def paginate_telegram_html(
    full_html: str, *, repeat_header: bool = True, page_limit: int = TELEGRAM_MESSAGE_LIMIT
) -> list[str]:
    """Split only between complete HTML paragraphs; repeat a bold first headline.

    A paragraph containing formatting or links is indivisible. If it cannot fit
    with the repeated headline, fail explicitly instead of cutting text or tags.
    """
    if (
        isinstance(page_limit, bool)
        or not isinstance(page_limit, int)
        or not 1 <= page_limit <= 4096
    ):
        raise TelegramPagingError("Telegram page limit must be between 1 and 4096")
    paragraphs = _paragraphs(full_html)
    header = paragraphs[0] if repeat_header and paragraphs[0].headline else None
    body = paragraphs[1:] if header else paragraphs
    if header and utf16_units(header.visible) > page_limit:
        raise TelegramPagingError("Telegram headline exceeds the page limit")
    if not body:
        return [paragraphs[0].rendered]
    pages: list[str] = []
    current = [header] if header else []
    for paragraph in body:
        candidate = [*current, paragraph]
        units = utf16_units("\n\n".join(part.visible for part in candidate))
        if units > page_limit:
            minimum = [header, paragraph] if header else [paragraph]
            if utf16_units("\n\n".join(part.visible for part in minimum)) > page_limit:
                raise TelegramPagingError("A complete Telegram paragraph exceeds the page limit")
            if current:
                pages.append("\n\n".join(part.rendered for part in current))
            current = minimum
        else:
            current = candidate
    if current:
        pages.append("\n\n".join(part.rendered for part in current))
    return pages
