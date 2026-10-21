from __future__ import annotations

import json

from aiogram.filters import Command
from aiogram.filters.callback_data import CallbackData
from aiogram.types import ForceReply, InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder

from app.biography.storage import BiographyStore
from app.telegram_bot.feedback import Reaction
from app.telegram_bot.keyboards import REACTION_LABELS, REACTION_ORDER


class BiographyCallback(CallbackData, prefix="bio"):
    delivery_id: int
    action: str


def biography_keyboard(delivery_id: int, feedback: dict | None = None) -> InlineKeyboardMarkup:
    feedback = feedback or {}
    builder = InlineKeyboardBuilder()
    for axis, positive, negative in (
        ("hero", "Интересный герой", "Неинтересный герой"),
        ("text", "Хороший текст", "Плохой текст"),
    ):
        for value, label in ((1, positive), (-1, negative)):
            builder.button(
                text=("✅ " if feedback.get(axis) == value else "") + label,
                callback_data=BiographyCallback(delivery_id=delivery_id, action=f"{axis}{value}"),
            )
    for reaction in REACTION_ORDER:
        builder.button(
            text=("✅ " if feedback.get("reaction") == reaction.value else "")
            + REACTION_LABELS[reaction],
            callback_data=BiographyCallback(delivery_id=delivery_id, action=reaction.value),
        )
    builder.button(
        text="💬 Комментарий",
        callback_data=BiographyCallback(delivery_id=delivery_id, action="comment"),
    )
    builder.adjust(2, 2, 4, 1)
    return builder.as_markup()


def controls(enabled: bool) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.button(
        text="⏸ Пауза" if enabled else "▶️ Возобновить",
        callback_data=BiographyCallback(delivery_id=0, action="pause" if enabled else "resume"),
    )
    return builder.as_markup()


def register_biography_handlers(router, settings, repository) -> None:
    from app.telegram_bot.bot import is_allowed

    store = BiographyStore(repository)

    def private_owner(message, user_id):
        return (
            is_allowed(settings, repository, user_id)
            and message.chat.type == "private"
            and message.chat.id == user_id
        )

    async def show_status(message):
        store.initialize()
        status = store.status()
        enabled = store.enabled(settings.biography_enabled)
        text = (
            "<b>Биографии</b>\n\n"
            + ("Каждый день в 09:00 по Москве." if enabled else "Ежедневная выдача на паузе.")
            + f"\nГотово в запасе: {status['ready']} из 3."
            + f"\nРасходы биографий за месяц: ${status['month_cost_usd']:.4f}."
            + "\n\nПод постом можно отдельно оценить героя и текст или оставить комментарий."
        )
        if status["unsettled_receipts"]:
            text += "\nЕсть запрос с неопределённым результатом; автоматический повтор остановлен."
        await message.answer(text, reply_markup=controls(enabled))

    @router.message(Command("bio"))
    async def biography_status(message):
        owner = message.from_user.id if message.from_user else None
        if not private_owner(message, owner):
            return
        argument = (message.text or "").split(maxsplit=1)
        if len(argument) > 1 and argument[1].strip() in {"pause", "resume"}:
            store.set_enabled(argument[1].strip() == "resume")
        await show_status(message)

    @router.callback_query(BiographyCallback.filter())
    async def biography_callback(query, callback_data: BiographyCallback):
        owner = query.from_user.id
        if not query.message or not private_owner(query.message, owner):
            await query.answer("Недоступно", show_alert=True)
            return
        action = callback_data.action
        if callback_data.delivery_id == 0 and action in {"pause", "resume"}:
            store.set_enabled(action == "resume")
            await query.answer("Биографии на паузе" if action == "pause" else "Биографии включены")
            await show_status(query.message)
            return
        delivery = store.delivery(callback_data.delivery_id)
        if (
            not delivery
            or delivery["status"] != "sent"
            or delivery["owner_id"] != owner
            or delivery["telegram_message_id"] != query.message.message_id
        ):
            await query.answer("Этот пост недоступен", show_alert=True)
            return
        if action == "comment":
            await query.answer()
            prompt = await query.message.answer(
                "Напиши, что понравилось или что стоит изменить в этой биографии.",
                reply_markup=ForceReply(selective=True),
            )
            repository.set_runtime_setting(
                f"biography_comment:{owner}:{prompt.message_id}",
                json.dumps({"delivery_id": delivery["id"], "prompt_id": prompt.message_id}),
            )
            return
        if action in {"hero1", "hero-1", "text1", "text-1"}:
            axis = "hero" if action.startswith("hero") else "text"
            value = -1 if action.endswith("-1") else 1
        elif action in {r.value for r in Reaction}:
            axis, value = "reaction", action
        else:
            await query.answer("Неизвестная оценка", show_alert=True)
            return
        store.set_feedback(delivery["id"], owner, axis, value)
        await query.answer("Оценка сохранена")
        await query.message.edit_reply_markup(
            reply_markup=biography_keyboard(delivery["id"], store.feedback(delivery["id"], owner))
        )

    def comment_delivery(message):
        if message.text and message.text.startswith("/"):
            return None
        owner = message.from_user.id if message.from_user else None
        if not message.text or not private_owner(message, owner) or not message.reply_to_message:
            return None
        reply_id = message.reply_to_message.message_id
        saved = repository.get_runtime_setting(f"biography_comment:{owner}:{reply_id}")
        if saved:
            context = json.loads(saved)
            if context.get("prompt_id") == reply_id:
                return context["delivery_id"]
        with repository.connection() as connection:
            row = connection.execute(
                "SELECT id FROM biography_deliveries WHERE owner_id=? AND "
                "telegram_message_id=? AND status='sent'",
                (owner, reply_id),
            ).fetchone()
        return row["id"] if row else None

    async def is_biography_comment(message):
        return comment_delivery(message) is not None

    @router.message(is_biography_comment)
    async def biography_comment(message):
        delivery_id = comment_delivery(message)
        store.set_feedback(delivery_id, message.from_user.id, "comment", message.text)
        await message.answer("Комментарий к биографии сохранён.")
