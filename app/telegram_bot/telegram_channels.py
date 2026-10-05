"""Telegram links and forwarded channel posts become explicit radar subscriptions."""

from __future__ import annotations

import asyncio
import html
import re

from aiogram.dispatcher.event.bases import SkipHandler
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import BaseFilter, Command
from aiogram.filters.callback_data import CallbackData
from aiogram.types import LinkPreviewOptions
from aiogram.utils.keyboard import InlineKeyboardBuilder

from app.radar.telegram_channels import (
    TelegramChannelError,
    add_channel,
    change_channel,
    channel_identity,
    channel_rows,
    channel_username,
)

LINK = re.compile(r"(?<![\w./])(?:https?://)?(?:www\.)?(?:t\.me|telegram\.me)/[^\s<>]+", re.I)
PREVIEW = LinkPreviewOptions(is_disabled=True)


class TelegramChannelControl(CallbackData, prefix="tgr"):
    action: str
    key: str = ""
    page: int = 0


def command(message):
    text = (message.text or message.caption or "").strip()
    return text.split(maxsplit=1)[0].split("@")[0].lower() if text else ""


def channel_references(message) -> list[str]:
    origin = getattr(message, "forward_origin", None)
    if origin and origin.type == "channel":
        # The origin is authoritative; mentions in the post belong to its contents.
        return ["@" + origin.chat.username] if origin.chat.username else []
    text = message.text or message.caption or ""
    values = LINK.findall(text)
    values += [
        e.url
        for e in (message.entities or message.caption_entities or [])
        if e.type == "text_link" and e.url and LINK.fullmatch(e.url)
    ]
    remainder = text.split(maxsplit=1)[-1].strip() if command(message) == "/tg" else text.strip()
    if remainder.startswith("@") and not re.search(r"\s", remainder):
        values.append(remainder)
    return list(dict.fromkeys(v.rstrip(".,!?)]}\"'»") for v in values))


class TelegramChannelFilter(BaseFilter):
    async def __call__(self, message):
        cmd = command(message)
        if cmd.startswith("/") and cmd != "/tg":
            return False
        origin = getattr(message, "forward_origin", None)
        return (
            cmd == "/tg"
            or bool(channel_references(message))
            or bool(origin and origin.type == "channel")
        )


def menu(settings, repository, page=0):
    rows = channel_rows(settings, repository)
    pages = max(1, (len(rows) + 7) // 8)
    page = max(0, min(page, pages - 1))
    builder = InlineKeyboardBuilder()
    for row in rows[page * 8 : page * 8 + 8]:
        builder.button(
            text=("🟢 " if row["enabled"] else "⏸ ") + row["title"][:48],
            callback_data=TelegramChannelControl(action="view", key=row["source_id"], page=page),
        )
    builder.adjust(1)
    nav = InlineKeyboardBuilder()
    for label, target in (("←", page - 1), ("→", page + 1)):
        if 0 <= target < pages:
            nav.button(text=label, callback_data=TelegramChannelControl(action="list", page=target))
    nav.adjust(2)
    builder.attach(nav)
    return (
        f"<b>Telegram-каналы · {len(rows)}</b>\n\n"
        "Пришли ссылку на канал, @имя или перешли его пост. "
        "Покажу название и кнопку «Добавить в радар».\n\n"
        "Нажми на канал, чтобы поставить его на паузу или удалить. "
        "Публикации проходят общий отбор радара.\n"
        f"Страница {page + 1}/{pages}",
        builder.as_markup(),
    )


def register_telegram_channel_handlers(router, settings, repository):
    from app.telegram_bot.bot import is_allowed

    def allowed(message):
        return (
            message.from_user
            and is_allowed(settings, repository, message.from_user.id)
            and message.chat.type == "private"
        )

    @router.message(Command("tgchannels"))
    async def telegram_channels_handler(message):
        if not allowed(message):
            return
        text, keyboard = menu(settings, repository)
        await message.answer(text, reply_markup=keyboard, link_preview_options=PREVIEW)

    @router.message(TelegramChannelFilter())
    async def telegram_channel_handler(message):
        if not allowed(message):
            return
        session = repository.feedback_session_for_user(message.from_user.id)
        if session and session["state"] == "awaiting_comment" and command(message) != "/tg":
            raise SkipHandler()
        refs = channel_references(message)
        if not refs and (message.text or "").strip().split("@")[0] == "/tg":
            text, keyboard = menu(settings, repository)
            await message.answer(text, reply_markup=keyboard, link_preview_options=PREVIEW)
            return
        if len(refs) != 1:
            await message.answer("Пришли одну публичную ссылку t.me/имя или @имя канала.")
            return
        try:
            username = channel_username(refs[0])
            chat = await message.bot.get_chat("@" + username)
            _, username, title = channel_identity(chat)
        except TelegramChannelError as exc:
            await message.answer(html.escape(str(exc)))
            return
        except TelegramAPIError:
            await message.answer(
                "Не получилось проверить канал. Проверь публичную ссылку и пришли её снова."
            )
            return
        row = next(
            (r for r in channel_rows(settings, repository) if r["username"].casefold() == username),
            None,
        )
        builder = InlineKeyboardBuilder()
        if row:
            builder.button(
                text="Управлять каналом",
                callback_data=TelegramChannelControl(action="view", key=row["source_id"]),
            )
            text = f"<b>{html.escape(title)}</b> уже в радаре" + (
                "." if row["enabled"] else " и сейчас на паузе."
            )
        else:
            builder.button(
                text="Добавить в радар",
                callback_data=TelegramChannelControl(action="add", key=username),
            )
            builder.button(text="Отмена", callback_data=TelegramChannelControl(action="cancel"))
            text = (
                f"Добавить <b>{html.escape(title)}</b> в радар?\n"
                f"https://t.me/{username}\n\n"
                "Начну со свежих публикаций за последние сутки, затем буду следить за новыми."
            )
        builder.adjust(2)
        await message.answer(text, reply_markup=builder.as_markup(), link_preview_options=PREVIEW)

    @router.callback_query(TelegramChannelControl.filter())
    async def telegram_channel_callback(query, callback_data: TelegramChannelControl):
        if (
            not is_allowed(settings, repository, query.from_user.id)
            or not query.message
            or query.message.chat.type != "private"
        ):
            await query.answer("Недоступно", show_alert=True)
            return
        await query.answer()
        action, key, page = callback_data.action, callback_data.key, callback_data.page
        if action == "cancel":
            await query.message.edit_text("Добавление отменено.")
            return
        try:
            if action == "add":
                username = channel_username("@" + key)
                chat = await query.bot.get_chat("@" + username)
                result = await asyncio.to_thread(
                    add_channel, settings, repository, chat, query.from_user.id
                )
                builder = InlineKeyboardBuilder()
                builder.button(
                    text="Мои Telegram-каналы", callback_data=TelegramChannelControl(action="list")
                )
                heading = "Канал уже в радаре" if result["existing"] else "Канал добавлен в радар"
                status = (
                    "Свежие посты будут учитываться при следующем сборе радара."
                    if result["enabled"]
                    else "Канал остаётся на паузе. Включить его можно в списке."
                )
                await query.message.edit_text(
                    f"<b>{heading}: {html.escape(result['title'])}</b>\n\n"
                    f"{status}\n"
                    "/tgchannels — список, пауза и удаление.",
                    reply_markup=builder.as_markup(),
                    link_preview_options=PREVIEW,
                )
                return
            if action in {"pause", "resume", "remove"}:
                await asyncio.to_thread(
                    change_channel, settings, repository, key, action, query.from_user.id
                )
                action = "list" if action == "remove" else "view"
            if action in {"view", "confirm"}:
                row = next(
                    (r for r in channel_rows(settings, repository) if r["source_id"] == key), None
                )
                if row is None:
                    raise TelegramChannelError(
                        "Канал уже удалён. Пришли ссылку, чтобы добавить его снова."
                    )
                builder = InlineKeyboardBuilder()
                if action == "confirm":
                    text = f"Удалить <b>{html.escape(row['title'])}</b> из радара?"
                    builder.button(
                        text="Да, удалить",
                        callback_data=TelegramChannelControl(action="remove", key=key, page=page),
                    )
                else:
                    text = f"<b>{html.escape(row['title'])}</b>\n\n" + (
                        "🟢 Сбор включён" if row["enabled"] else "⏸ Канал на паузе"
                    )
                    builder.button(
                        text="Пауза" if row["enabled"] else "Возобновить",
                        callback_data=TelegramChannelControl(
                            action="pause" if row["enabled"] else "resume", key=key, page=page
                        ),
                    )
                    builder.button(
                        text="Удалить",
                        callback_data=TelegramChannelControl(action="confirm", key=key, page=page),
                    )
                    builder.button(text="Открыть канал", url="https://t.me/" + row["username"])
                builder.button(
                    text="← Все каналы",
                    callback_data=TelegramChannelControl(action="list", page=page),
                )
                builder.adjust(2, 1, 1)
                keyboard = builder.as_markup()
            else:
                text, keyboard = menu(settings, repository, page)
            await query.message.edit_text(text, reply_markup=keyboard, link_preview_options=PREVIEW)
        except TelegramChannelError as exc:
            await query.message.answer(html.escape(str(exc)))
        except TelegramAPIError:
            await query.message.answer(
                "Telegram временно не ответил. Проверь /tgchannels перед повтором."
            )
