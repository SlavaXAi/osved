"""Owner-requested, zero-network views over OSVED's stored factual corpus."""

from __future__ import annotations

import asyncio
import html
import logging
from html.parser import HTMLParser
from typing import Any, Literal
from urllib.parse import urlsplit

from aiogram.filters import Command
from aiogram.types import LinkPreviewOptions

from app.collectors.ingest import redact_text
from app.config import Settings
from app.db.repository import Repository
from app.intelligence.service import IntelligenceService

logger = logging.getLogger(__name__)
ProductKind = Literal["weekly_map", "weekly_personas"]
MAX_VISIBLE_UNITS = 3800


def _units(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


class _BoundedHTML(HTMLParser):
    """Keep the existing render, limiting visible UTF-16 length and balancing tags."""

    def __init__(self, limit: int) -> None:
        super().__init__(convert_charrefs=True)
        self.remaining = limit
        self.parts: list[str] = []
        self.stack: list[str] = []
        self.truncated = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.truncated or tag not in {"b", "i", "a", "blockquote"}:
            return
        attribute = ""
        if tag == "a":
            href = str(dict(attrs).get("href") or "")
            try:
                parsed = urlsplit(href)
                if (
                    parsed.scheme not in {"http", "https"}
                    or not parsed.hostname
                    or parsed.username
                    or len(href) > 2048
                ):
                    return
            except ValueError:
                return
            attribute = f' href="{html.escape(href, quote=True)}"'
        self.parts.append(f"<{tag}{attribute}>")
        self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag not in self.stack:
            return
        while self.stack:
            closing = self.stack.pop()
            self.parts.append(f"</{closing}>")
            if closing == tag:
                break

    def handle_data(self, data: str) -> None:
        if self.truncated:
            return
        size = _units(data)
        if size > self.remaining:
            data = data.encode("utf-16-le")[: self.remaining * 2].decode(
                "utf-16-le", errors="ignore"
            )
            self.truncated = True
        self.parts.append(html.escape(data))
        self.remaining -= _units(data)

    def rendered(self) -> str:
        self.close()
        while self.stack:
            self.parts.append(f"</{self.stack.pop()}>")
        return "".join(self.parts)


def intelligence_message(product: dict[str, Any], kind: ProductKind) -> str:
    notice = (
        "Только сохранённые проверенные события OSVED. "
        "Это не полный обзор мира и не полная история действий людей за неделю. "
        "Новый поиск не запускался."
    )
    status = product.get("validation_status")
    if status == "insufficient_data":
        subject = "по людям и институтам" if kind == "weekly_personas" else "для недельной карты"
        return (
            f"<b>Пока недостаточно данных {subject}</b>\n\n"
            "В сохранённом проверенном корпусе не хватает существенных сигналов. "
            "Это не означает, что за неделю ничего не произошло.\n\n"
            f"<i>{notice}</i>"
        )
    if status != "valid" or not str(product.get("rendered_html") or "").strip():
        return (
            "<b>Карта пока не готова</b>\n\n"
            "Сохранённый результат не прошёл проверку. Непроверенную сводку не показываю.\n\n"
            f"<i>{notice}</i>"
        )
    parser = _BoundedHTML(MAX_VISIBLE_UNITS - _units(notice) - 100)
    parser.feed(str(product["rendered_html"]))
    result = parser.rendered()
    shortened = "\n\n<i>Показана сокращённая карта.</i>" if parser.truncated else ""
    return f"<i>{notice}</i>\n\n{result}{shortened}"


def register_intelligence_handlers(router, settings: Settings, repository: Repository) -> None:
    from app.telegram_bot.bot import is_allowed

    async def send_product(message, kind: ProductKind) -> None:
        user_id = message.from_user.id if message.from_user else None
        # An owner issuing a command in a group must not disclose the private corpus there.
        if not is_allowed(settings, repository, user_id) or message.chat.id != user_id:
            return
        try:
            product = await asyncio.to_thread(IntelligenceService(settings, repository).build, kind)
        except Exception as exc:
            logger.warning("Stored intelligence view failed: %s", redact_text(str(exc))[:400])
            await message.answer(
                "Не получилось собрать карту из базы. Новые поиски не запускались."
            )
            return
        await message.answer(
            intelligence_message(product, kind),
            parse_mode="HTML",
            link_preview_options=LinkPreviewOptions(is_disabled=True),
        )

    @router.message(Command("week"))
    async def week_handler(message):
        await send_product(message, "weekly_map")

    @router.message(Command("people"))
    async def people_handler(message):
        await send_product(message, "weekly_personas")
