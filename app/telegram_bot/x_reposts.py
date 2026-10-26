from __future__ import annotations

import json

from aiogram.dispatcher.event.bases import SkipHandler
from aiogram.filters import BaseFilter
from aiogram.filters.callback_data import CallbackData
from aiogram.types import ReplyParameters
from aiogram.utils.keyboard import InlineKeyboardBuilder

from app.media.downloads import message_links
from app.reposts import media, store


class RepostAction(CallbackData, prefix="xr"):
    repost_id: int
    action: str


class RepostFilter(BaseFilter):
    async def __call__(self, message):
        text = message.text or message.caption or ""
        if text.startswith("/"):
            return text.split(maxsplit=1)[0].split("@")[0] == "/repost"
        return any(link[0] == "x" for link in message_links(message))


def error_keyboard(row):
    b = InlineKeyboardBuilder()
    b.button(
        text="Открыть сохранённое / повторить",
        callback_data=RepostAction(repost_id=row["id"], action="retry"),
    )
    b.button(text="Оригинал", url=row["source_url"])
    b.adjust(1)
    return b.as_markup()


def feedback_buttons(repository, panel):
    row = store.for_delivery(repository, panel["delivery_id"])
    b = InlineKeyboardBuilder()
    if not row:
        return b
    with repository.connection() as db:
        current = db.execute(
            "SELECT vote FROM x_curation_samples WHERE chat_id=? AND tweet_id=?",
            (row["chat_id"], row["tweet_id"]),
        ).fetchone()
    for value, label in [(1, "👍 Полезно"), (-1, "👎 Не полезно")]:
        b.button(
            text=("✓ " if current and current[0] == value else "") + label,
            callback_data=RepostAction(
                repost_id=row["id"], action="like" if value == 1 else "dislike"
            ),
        )
    b.adjust(2)
    translation = json.loads(row["translation_json"] or "{}")
    if translation.get("thesis"):
        b.row()
        b.button(
            text="Разобрать тезис", callback_data=RepostAction(repost_id=row["id"], action="thesis")
        )
    if row.get("error"):
        b.row()
        b.button(
            text="Медиа недоступно · повторить загрузку",
            callback_data=RepostAction(repost_id=row["id"], action="retry"),
        )
    return b


def register_repost_handlers(router, settings, repository):
    from app.telegram_bot.bot import is_allowed

    @router.message(RepostFilter())
    async def x_repost_incoming(message):
        owner = message.from_user.id if message.from_user else None
        if not is_allowed(settings, repository, owner) or message.chat.type != "private":
            return
        if not settings.x_reposts_enabled:
            raise SkipHandler()
        session = repository.feedback_session_for_user(owner)
        if (
            session
            and session["state"] == "awaiting_comment"
            and not (message.text or "").startswith("/repost")
        ):
            raise SkipHandler()
        links = message_links(message)
        if len(links) != 1 or links[0][0] != "x":
            await message.answer("Пришли одну ссылку на пост X.")
            return
        try:
            row, created = store.enqueue(
                repository, owner=owner, message_id=message.message_id, link=links[0]
            )
        except ValueError as exc:
            await message.answer(str(exc))
            return
        if not created:
            kwargs = (
                {"reply_parameters": ReplyParameters(message_id=row["panel_message_id"])}
                if row["panel_message_id"]
                else {}
            )
            await message.answer(
                "Эта ссылка уже сохранена. Открой её карточку — повторной генерации не будет.",
                **kwargs,
            )
            return
        try:
            panel = await message.bot.send_rich_message(
                chat_id=owner,
                rich_message=media.rich_content("Готовлю перевод и медиа…", []),
                reply_parameters=ReplyParameters(message_id=message.message_id),
            )
        except Exception:
            with repository.connection() as db:
                db.execute(
                    "UPDATE x_reposts SET status='delivery_uncertain',"
                    "error='initial_panel_unconfirmed' WHERE id=?",
                    (row["id"],),
                )
            raise
        with repository.connection() as db:
            db.execute(
                "UPDATE x_reposts SET panel_message_id=? WHERE id=?", (panel.message_id, row["id"])
            )

    @router.callback_query(RepostAction.filter())
    async def x_repost_action(query, callback_data):
        row = store.get(repository, callback_data.repost_id)
        if (
            not row
            or not query.message
            or not is_allowed(settings, repository, query.from_user.id)
            or query.message.chat.type != "private"
            or query.message.chat.id != row["chat_id"]
            or query.from_user.id != row["chat_id"]
            or query.message.message_id != row["panel_message_id"]
        ):
            await query.answer("Эта карточка недоступна", show_alert=True)
            return
        if not settings.x_reposts_enabled:
            await query.answer("Обработка X-публикаций выключена", show_alert=True)
            return
        from app.telegram_bot.news_panel import ensure_news_panel, refresh_markup

        action = callback_data.action
        if action in {"like", "dislike"}:
            store.vote(repository, row, 1 if action == "like" else -1)
            await query.answer("Учту общую полезность результата")
            await refresh_markup(query.bot, repository, row["delivery_id"])
        elif action == "retry":
            try:
                store.retry(repository, row)
            except ValueError as exc:
                await query.answer(str(exc), show_alert=True)
                return
            await query.answer("Проверю сохранённые данные")
        elif action == "thesis":
            from app.db.news_questions import claim_input, open_session, queue_question
            from app.telegram_bot.news_panel import set_loading, show_result

            if not settings.details_enabled or row["status"] != "ready":
                await query.answer("Разбор пока недоступен")
                return
            thesis = json.loads(row["translation_json"] or "{}").get("thesis")
            if not thesis:
                await query.answer("В этом посте нет отдельного тезиса")
                return
            panel = ensure_news_panel(repository, row["delivery_id"])
            # Synthetic negative IDs are reserved for this one explicit button input.
            input_id = -row["id"]
            try:
                previous, request = store.saved_thesis(repository, panel, input_id)
                if previous:
                    if request and request.get("article_id"):
                        await query.answer("Открываю сохранённый разбор")
                        if await show_result(
                            query.bot,
                            repository,
                            panel,
                            request,
                            repository.get_detailed_article(request["article_id"]),
                        ):
                            repository.mark_detail_request_sent(request["request_id"])
                    else:
                        await query.answer("Разбор уже запрошен; его состояние видно в карточке")
                    return
                open_session(repository, panel)
                session = {"chat_id": panel["chat_id"], "delivery_id": panel["delivery_id"]}
                if not claim_input(
                    repository, session, input_id, "text", max_daily=settings.details_max_daily_new
                ):
                    await query.answer("Запрос уже принят")
                    return
                request = queue_question(
                    repository,
                    panel,
                    input_id,
                    "Разбери конкретный тезис из поста: «"
                    + thesis
                    + "». Объясни смысл, основания и ограничения; "
                    "найди первоисточник, если автор пересказывает чужие слова.",
                )
                await query.answer("Разбираю тезис")
                if request["status"] == "ready":
                    if await show_result(
                        query.bot,
                        repository,
                        panel,
                        request,
                        repository.get_detailed_article(request["article_id"]),
                    ):
                        repository.mark_detail_request_sent(request["request_id"])
                else:
                    await set_loading(query.bot, repository, panel, request)
            except ValueError as exc:
                with repository.connection() as db:
                    db.execute(
                        "DELETE FROM news_question_inputs WHERE chat_id=? AND message_id=? "
                        "AND status='receiving' AND request_id IS NULL",
                        (row["chat_id"], input_id),
                    )
                await query.answer(str(exc), show_alert=True)
