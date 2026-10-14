"""News-reader navigation: explicit research forward, cached navigation backward."""

from __future__ import annotations

import asyncio
import json

from app.telegram_bot.news_panel import (
    artifact_pages,
    cached_back_target,
    ensure_news_panel,
    set_loading,
    show_base,
    show_error,
    show_result,
)

LOCKS: dict[int, asyncio.Lock] = {}


def terminal_text(request):
    status = request["status"]
    error = str(request.get("last_error") or "").casefold()
    if "uncertain" in error or "not confirmed" in error or "preparing" in error:
        return "Результат предыдущего шага не подтверждён. Повторная оплата заблокирована."
    if "402" in error or "insufficient_credits" in error:
        return "Сервис генерации отклонил запрос: недостаточно средств."
    if status == "budget_blocked":
        return "Разбор остановлен лимитом расходов. Сохранённые материалы доступны для возврата."
    if status == "insufficient_evidence":
        return (
            "Не хватило новых прочитанных источников по этой новости для содержательного разбора."
        )
    return "Не удалось завершить разбор. Исходная новость и собранные материалы сохранены."


def pending_for_delivery(repository, delivery_id):
    with repository.connection() as db:
        return db.execute(
            "SELECT id FROM detail_requests WHERE delivery_id=? AND depth!='legacy' "
            "AND status IN('queued','running') LIMIT 1",
            (delivery_id,),
        ).fetchone()


async def handle_news_action(
    query, repository, settings, delivery_id, action, *, page=0, revision=-1
):
    context = repository.get_delivery_context(delivery_id)
    if (
        not context
        or not query.message
        or context["chat_id"] != query.message.chat.id
        or context["chat_id"] != query.from_user.id
        or context["telegram_message_id"] != query.message.message_id
    ):
        await query.answer("Кнопка не относится к этой новости", show_alert=True)
        return
    if action not in {"base", "back", "expanded", "article", "prev", "next", "retry"}:
        await query.answer("Неизвестное действие", show_alert=True)
        return
    event = repository.event_context(context["event_id"])
    if not event or event.get("event_type") == "system_test":
        await query.answer("Для тестовой карточки исследование не запускается")
        return
    lock = LOCKS.setdefault(delivery_id, asyncio.Lock())
    async with lock:
        panel = ensure_news_panel(repository, delivery_id)
        if revision >= 0 and revision != panel["revision"]:
            await query.answer("Карточка обновилась. Нажми кнопку в текущем меню.")
            return
        if action == "base":
            await query.answer()
            await show_base(query.bot, repository, panel)
            return
        if action == "back":
            await query.answer()
            depth, request, artifact = cached_back_target(repository, panel)
            if depth == "base":
                await show_base(query.bot, repository, panel)
            else:
                saved = repository.get_runtime_setting(
                    f"news_reader_position:{delivery_id}:{depth}"
                )
                try:
                    saved_page = int(saved or 0)
                except (ValueError, TypeError):
                    saved_page = 0
                saved_page = max(0, min(saved_page, len(artifact_pages(repository, artifact)) - 1))
                await show_result(
                    query.bot,
                    repository,
                    panel,
                    request,
                    artifact,
                    page=saved_page,
                )
            return
        if action in {"prev", "next"}:
            request = repository.get_detail_request(panel.get("displayed_request_id") or -1)
            if not request or panel["view"] != request["depth"]:
                await query.answer("Сначала открой готовый результат")
                return
            artifact = repository.get_detailed_article(request["article_id"])
            await query.answer()
            await show_result(
                query.bot,
                repository,
                panel,
                request,
                artifact,
                page=panel["page"] + (-1 if action == "prev" else 1),
            )
            return
        if not settings.details_enabled:
            await query.answer("Подробные разборы выключены", show_alert=True)
            return
        if action == "retry":
            request = repository.get_detail_request(
                panel.get("active_request_id") or panel.get("displayed_request_id") or -1
            )
            if not request or request["status"] not in {
                "failed",
                "insufficient_evidence",
                "budget_blocked",
            }:
                await query.answer("Повтор сейчас не требуется")
                return
            reader = request["reader_context"]
            with repository.connection() as db:
                uncertain = db.execute(
                    "SELECT 1 FROM details_provider_calls WHERE event_id=? AND depth=? "
                    "AND baseline_fingerprint=? AND provider='openrouter' "
                    "AND status IN('preparing','uncertain') LIMIT 1",
                    (request["event_id"], request["depth"], reader["context_fingerprint"]),
                ).fetchone()
            if uncertain:
                await query.answer(
                    "Предыдущий платный шаг не подтверждён; повтор заблокирован.", show_alert=True
                )
                return
            request = repository.retry_detail_request(request["request_id"])
        else:
            request = repository.detail_request_for_delivery(delivery_id, depth=action)
            if request and request["status"] in {"ready", "sent"}:
                artifact = repository.get_detailed_article(request["article_id"])
                await query.answer()
                if await show_result(query.bot, repository, panel, request, artifact, page=0):
                    repository.mark_detail_request_sent(request["request_id"])
                return
            if request and request["status"] in {
                "failed",
                "insufficient_evidence",
                "budget_blocked",
            }:
                await query.answer()
                await show_error(
                    query.bot, repository, panel, request, terminal_text(request), from_worker=False
                )
                return
            if pending_for_delivery(repository, delivery_id) and not request:
                await query.answer("Сначала дождись текущего разбора; можно вернуться к новости.")
                return
            request = repository.enqueue_detail_request(
                delivery_id,
                query.from_user.id,
                depth=action,
                reader_context=panel["reader_context"],
                max_daily_new=settings.details_max_daily_new,
            )
        if request["status"] == "daily_limit":
            await query.answer("Сегодня достигнут лимит новых разборов", show_alert=True)
            return
        await query.answer("Собираю дополнительные источники")
        if request["status"] == "ready":
            artifact = repository.get_detailed_article(request["article_id"])
            if await show_result(query.bot, repository, panel, request, artifact):
                repository.mark_detail_request_sent(request["request_id"])
        else:
            await set_loading(query.bot, repository, panel, request)


def record_news_result_feedback(repository, panel, request, action):
    artifact = repository.get_detailed_article(request.get("article_id") or -1)
    if (
        not artifact
        or panel["view"] != request["depth"]
        or panel["displayed_request_id"] != request["request_id"]
        or artifact["depth"] != request["depth"]
        or artifact["context_fingerprint"] != panel["reader_context"]["context_fingerprint"]
    ):
        raise PermissionError("Сначала открой этот результат")
    with repository.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        old = db.execute(
            "SELECT action FROM news_result_feedback WHERE article_id=? AND chat_id=?",
            (artifact["id"], panel["chat_id"]),
        ).fetchone()
        if old and old[0] == action:
            return False
        cursor = db.execute(
            "INSERT INTO feedback_events(delivery_id,event_id,card_id,action,metadata_json) "
            "VALUES(?,?,?,?,?)",
            (
                panel["delivery_id"],
                request["event_id"],
                request["card_id"],
                "detail_useful" if action == "useful" else "detail_not_worth_it",
                json.dumps(
                    {
                        "article_id": artifact["id"],
                        "depth": request["depth"],
                        "meaning": "overall_result_usefulness",
                        "previous_action": old[0] if old else None,
                    }
                ),
            ),
        )
        db.execute(
            "INSERT INTO news_result_feedback VALUES(?,?,?,?,CURRENT_TIMESTAMP) "
            "ON CONFLICT(article_id,chat_id) DO UPDATE SET action=excluded.action,"
            "feedback_event_id=excluded.feedback_event_id,updated_at=excluded.updated_at",
            (artifact["id"], panel["chat_id"], action, cursor.lastrowid),
        )
    return True
