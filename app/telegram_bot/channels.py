"""Owner-only YouTube channel inbox. Video links still go to the PDF worker."""

from __future__ import annotations

import asyncio
import html
import logging
import re

from aiogram.dispatcher.event.bases import SkipHandler
from aiogram.filters import BaseFilter, Command
from aiogram.filters.callback_data import CallbackData
from aiogram.types import LinkPreviewOptions
from aiogram.utils.keyboard import InlineKeyboardBuilder

from app.video.channels import (
    ChannelError,
    ChannelService,
    change_channel,
    channel_rows,
    channel_url,
)

logger = logging.getLogger(__name__)


class ChannelControl(CallbackData, prefix="ych"):
    action: str
    source_id: str = ""
    page: int = 0


def links(message) -> list[str]:
    text = message.text or message.caption or ""
    values = re.findall(r"(?:https?://|(?:www\.|m\.)?youtube\.com/)[^\s<>]+", text)
    values += [
        e.url
        for e in (message.entities or message.caption_entities or [])
        if e.type == "text_link" and e.url
    ]
    found = []
    for value in values:
        try:
            url = channel_url(value.rstrip(".,!?)\"'>]"))
        except ValueError:
            continue
        if url not in found:
            found.append(url)
    return found


def command(message) -> str:
    text = message.text or message.caption or ""
    return text.strip().split(maxsplit=1)[0].split("@")[0].lower() if text.strip() else ""


class ChannelFilter(BaseFilter):
    async def __call__(self, message):
        cmd = command(message)
        if cmd.startswith("/") and cmd != "/channel":
            return False
        return cmd == "/channel" or bool(links(message))


def menu(settings, repository, page=0):
    rows = channel_rows(settings, repository)
    pages = max(1, (len(rows) + 7) // 8)
    page = max(0, min(page, pages - 1))
    builder = InlineKeyboardBuilder()
    for row in rows[page * 8 : page * 8 + 8]:
        builder.button(
            text=("🟢 " if row["enabled"] else "⏸ ") + row["title"].removeprefix("YouTube — ")[:48],
            callback_data=ChannelControl(action="view", source_id=row["source_id"], page=page),
        )
    builder.adjust(1)
    nav = InlineKeyboardBuilder()
    for label, target in (("←", page - 1), ("→", page + 1)):
        if 0 <= target < pages:
            nav.button(text=label, callback_data=ChannelControl(action="list", page=target))
    nav.adjust(2)
    builder.attach(nav)
    return (
        f"<b>YouTube-каналы · {len(rows)}</b>\n\n"
        "Пришли ссылку на канал — добавлю его в радар. "
        "Нажми на название для паузы или удаления.\n\n"
        "Отслеживаем новые выпуски, без загрузки архива "
        "и без изменения твоих подписок YouTube. "
        "Проверка новых каналов — раз в 4 часа в рабочие часы радара.\n"
        "/videos — найденные рекомендации\n/report — PDF по ссылке на отдельный ролик\n"
        f"Страница {page + 1}/{pages}",
        builder.as_markup(),
    )


def register_channel_handlers(router, settings, repository):
    from app.telegram_bot.bot import is_allowed

    service = ChannelService(settings, repository)
    # No pile-up of resolver threads when several links arrive at once.
    pending = asyncio.Semaphore(1)

    @router.message(Command("channels"))
    async def channels_handler(message):
        if (
            not message.from_user
            or not is_allowed(settings, repository, message.from_user.id)
            or message.chat.type != "private"
        ):
            return
        text, keyboard = menu(settings, repository)
        await message.answer(
            text, reply_markup=keyboard, link_preview_options=LinkPreviewOptions(is_disabled=True)
        )

    @router.message(ChannelFilter())
    async def channel_handler(message):
        if (
            not message.from_user
            or not is_allowed(settings, repository, message.from_user.id)
            or message.chat.type != "private"
        ):
            return
        session = repository.feedback_session_for_user(message.from_user.id)
        if session and session["state"] == "awaiting_comment" and command(message) != "/channel":
            raise SkipHandler()
        urls = links(message)
        if len(urls) != 1:
            await message.answer(
                "Пришли одну ссылку на канал, например /channel https://youtube.com/@ycombinator"
            )
            return
        if pending.locked():
            await message.answer(
                "Ещё обрабатываю предыдущий канал. Пришли следующую ссылку после подтверждения."
            )
            return
        async with pending:
            await message.answer("Проверяю канал и его публичную ленту…")
            try:
                result = await asyncio.to_thread(service.add, urls[0], message.from_user.id)
            except ChannelError as exc:
                await message.answer(html.escape(str(exc)))
                return
            except Exception:
                logger.exception("Channel addition failed")
                await message.answer(
                    "Не удалось завершить добавление. Проверь /channels перед повторной ссылкой."
                )
                return
        title = html.escape(result["title"])
        heading = "Канал уже в радаре" if result["existing"] else "Канал добавлен в радар"
        builder = InlineKeyboardBuilder()
        builder.button(
            text="Управлять каналом",
            callback_data=ChannelControl(action="view", source_id=result["source_id"]),
        )
        await message.answer(
            f"<b>{heading}: {title}</b>\n\n"
            "Новые выпуски пришлю карточками в часы доставки, без суточной квоты. "
            "Архив не загружаю, личную ленту и историю YouTube не читаю. "
            "Под карточкой: «Интересно» → краткая сводка текстом или подробный PDF. "
            "Отчёт создаётся только по твоему запросу.\n\n"
            "/channels — список и настройки каналов",
            reply_markup=builder.as_markup(),
            link_preview_options=LinkPreviewOptions(is_disabled=True),
        )

    @router.callback_query(ChannelControl.filter())
    async def channel_callback(query, callback_data: ChannelControl):
        if (
            not is_allowed(settings, repository, query.from_user.id)
            or not query.message
            or query.message.chat.type != "private"
        ):
            await query.answer("Недоступно", show_alert=True)
            return
        action, source_id, page = callback_data.action, callback_data.source_id, callback_data.page
        if action in {"pause", "resume", "remove"}:
            try:
                change_channel(settings, repository, source_id, action, query.from_user.id)
            except ChannelError as exc:
                await query.answer(str(exc), show_alert=True)
                return
            await query.answer(
                {
                    "pause": "Канал на паузе",
                    "resume": "Канал включён",
                    "remove": "Канал удалён из радара",
                }[action]
            )
            action = "list" if action == "remove" else "view"
        else:
            await query.answer()
        if action in {"view", "confirm"}:
            row = next(
                (r for r in channel_rows(settings, repository) if r["source_id"] == source_id), None
            )
            if row is None:
                action = "list"
            else:
                builder = InlineKeyboardBuilder()
                if action == "confirm":
                    text = (
                        f"Удалить <b>{html.escape(row['title'])}</b> из радара?\n"
                        "Сохранённые публикации и реакции останутся."
                    )
                    builder.button(
                        text="Да, удалить",
                        callback_data=ChannelControl(
                            action="remove", source_id=source_id, page=page
                        ),
                    )
                else:
                    text = f"<b>{html.escape(row['title'])}</b>\n\n" + (
                        "🟢 Сбор включён" if row["enabled"] else "⏸ На паузе"
                    )
                    builder.button(
                        text="Пауза" if row["enabled"] else "Возобновить",
                        callback_data=ChannelControl(
                            action="pause" if row["enabled"] else "resume",
                            source_id=source_id,
                            page=page,
                        ),
                    )
                    builder.button(
                        text="Удалить",
                        callback_data=ChannelControl(
                            action="confirm", source_id=source_id, page=page
                        ),
                    )
                    builder.button(
                        text="Открыть канал",
                        url="https://www.youtube.com/channel/" + row["channel_id"],
                    )
                builder.button(
                    text="← Все каналы", callback_data=ChannelControl(action="list", page=page)
                )
                builder.adjust(2, 1, 1)
                keyboard = builder.as_markup()
        if action not in {"view", "confirm"}:
            text, keyboard = menu(settings, repository, page)
        await query.message.edit_text(
            text, reply_markup=keyboard, link_preview_options=LinkPreviewOptions(is_disabled=True)
        )
