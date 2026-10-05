"""Small, isolated UX examples. They never rate news or start paid research."""

from __future__ import annotations

import json
from typing import Literal

from aiogram.exceptions import TelegramBadRequest
from aiogram.filters.callback_data import CallbackData
from aiogram.types import CallbackQuery, InlineKeyboardMarkup, LinkPreviewOptions
from aiogram.utils.keyboard import InlineKeyboardBuilder

from app.db.repository import Repository, utc_now

DemoAction = Literal["short", "context", "article", "choose"]


class NewsDemoCallback(CallbackData, prefix="nx"):
    demo_id: str
    action: DemoAction


def demo_key(demo_id: str) -> str:
    return f"news_ux_demo:{demo_id}"


def demo_keyboard(demo: dict, level: str) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    labels = demo["labels"]

    def button(action: DemoAction, label: str):
        builder.button(
            text=label,
            callback_data=NewsDemoCallback(
                demo_id=demo["id"],
                action=action,
            ),
        )

    if demo["navigation"] == "tabs":
        for action in ("short", "context", "article"):
            button(action, ("· " if action == level else "") + labels[action])
        builder.adjust(3)
    else:
        if level == "short":
            button("context", labels["context"])
        elif level == "context":
            button("article", labels["article"])
            button("short", labels["collapse"])
        else:
            button("context", labels["back"])
            button("short", labels["collapse"])
        builder.adjust(2)
    builder.row()
    builder.button(
        text=f"Выбираю {demo['variant']}",
        callback_data=NewsDemoCallback(
            demo_id=demo["id"],
            action="choose",
        ),
    )
    # Explicit row creation keeps the choice out of the navigation row.
    markup = builder.as_markup()
    buttons = [button for row in markup.inline_keyboard for button in row]
    markup.inline_keyboard = [buttons[:-1], buttons[-1:]]
    return markup


async def handle_news_demo(
    query: CallbackQuery,
    data: NewsDemoCallback,
    repository: Repository,
    *,
    allowed: bool,
) -> None:
    if not allowed:
        await query.answer("Недоступно", show_alert=True)
        return
    try:
        demo = json.loads(repository.get_runtime_setting(demo_key(data.demo_id)) or "null")
    except (TypeError, ValueError):
        demo = None
    if (
        not isinstance(demo, dict)
        or query.message is None
        or demo.get("chat_id") != query.message.chat.id
        or demo.get("message_id") != query.message.message_id
    ):
        await query.answer("Пример не найден для этого сообщения", show_alert=True)
        return
    if data.action == "choose":
        repository.set_runtime_setting(
            f"news_ux_vote:{demo['group']}:{query.from_user.id}",
            json.dumps(
                {"variant": demo["variant"], "demo_id": data.demo_id, "selected_at": utc_now()},
                ensure_ascii=False,
            ),
        )
        await query.answer(f"Выбран {demo['variant']}. Можно сравнить остальные и передумать.")
        return
    await query.answer()
    try:
        await query.message.edit_text(
            demo["levels"][data.action],
            parse_mode="HTML",
            reply_markup=demo_keyboard(demo, data.action),
            # Identical text-first examples; a varying link preview must not bias the test.
            link_preview_options=LinkPreviewOptions(is_disabled=True),
        )
    except TelegramBadRequest as exc:
        if "message is not modified" not in str(exc).lower():
            raise


async def publish_news_demos(bot, repository: Repository, *, chat_id: int, demos: list[dict]):
    """Explicit one-off publication only; never called by the schedule or news queue."""
    sent = []
    for demo in demos:
        key = demo_key(demo["id"])
        saved = repository.get_runtime_setting(key)
        if saved:
            existing = json.loads(saved)
            if existing.get("message_id"):
                sent.append(existing["message_id"])
                continue
            raise RuntimeError("Previous demo send is uncertain; inspect before retrying")
        item = {**demo, "chat_id": chat_id, "message_id": None, "created_at": utc_now()}
        repository.set_runtime_setting(key, json.dumps(item, ensure_ascii=False))
        message = await bot.send_message(
            chat_id=chat_id,
            text=demo["levels"]["short"],
            parse_mode="HTML",
            reply_markup=demo_keyboard(demo, "short"),
            link_preview_options=LinkPreviewOptions(is_disabled=True),
            disable_notification=True,
        )
        item["message_id"] = message.message_id
        repository.set_runtime_setting(key, json.dumps(item, ensure_ascii=False))
        sent.append(message.message_id)
    return sent
