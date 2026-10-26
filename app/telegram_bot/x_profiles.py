"""Profile links subscribe to the radar; post links retain their media route."""

import html
import re
from urllib.parse import urlsplit

from aiogram.dispatcher.event.bases import SkipHandler
from aiogram.filters import BaseFilter
from aiogram.types import LinkPreviewOptions

from app.radar.x_profiles import HOSTS, add_profile, profile_handle

LINK = re.compile(r"(?<![\w./])(?:https?://)?(?:www\.|mobile\.)?(?:x|twitter)\.com/[^\s<>]+", re.I)


def command(message):
    text = (message.text or message.caption or "").strip()
    return text.split(maxsplit=1)[0].split("@")[0].lower() if text else ""


def references(message):
    text = message.text or message.caption or ""
    values = LINK.findall(text)
    values += [
        e.url
        for e in (message.entities or message.caption_entities or [])
        if e.type == "text_link" and e.url
    ]
    result = []
    for value in values:
        value = value.rstrip(".,!?)]}\"'»")
        try:
            parsed = urlsplit(value if "://" in value else "https://" + value)
        except ValueError:
            continue
        # Posts, including /i/status links, belong to the downloader.
        if (parsed.hostname or "").lower() not in HOSTS or "/status/" in parsed.path:
            continue
        if value not in result:
            result.append(value)
    if command(message) == "/x":
        rest = text.strip().split(maxsplit=1)
        if len(rest) == 2 and not result:
            result.append(rest[1])
    return result


class XProfileFilter(BaseFilter):
    async def __call__(self, message):
        cmd = command(message)
        if cmd.startswith("/"):
            return cmd == "/x"
        return bool(references(message))


def register_x_profile_handlers(router, settings, repository):
    from app.telegram_bot.bot import is_allowed

    @router.message(XProfileFilter())
    async def x_profile_handler(message):
        user = message.from_user.id if message.from_user else None
        if not is_allowed(settings, repository, user) or message.chat.type != "private":
            return
        explicit = command(message) == "/x"
        feedback = repository.feedback_session_for_user(user)
        if not explicit and feedback and feedback["state"] == "awaiting_comment":
            raise SkipHandler()
        refs = references(message)
        if len(refs) != 1:
            await message.answer("Пришли одну ссылку на профиль X: /x https://x.com/имя")
            return
        try:
            handle = profile_handle(refs[0])
        except ValueError as exc:
            await message.answer(html.escape(str(exc)))
            return
        row = add_profile(settings, repository, refs[0], user)
        label = "уже в радаре" if row["existing"] else "добавлен в радар"
        if not row["enabled"]:
            label += " и сейчас на паузе. Включить можно в /sources"
        await message.answer(
            f"<b>@{handle}</b> {label}.\nhttps://x.com/{handle}\n\n"
            "Доступность публикаций проверю при плановом сборе. "
            "В ленту отберу существенное по твоим интересам. Управление — /sources.",
            parse_mode="HTML",
            link_preview_options=LinkPreviewOptions(is_disabled=True),
        )
