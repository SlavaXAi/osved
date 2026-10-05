"""Product health, separate from process liveness and video activity."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from app.db.repository import Repository


def feed_health(repository: Repository, *, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    since = (now - timedelta(hours=24)).isoformat()
    with repository.connection() as db:
        row = db.execute(
            """SELECT MAX(delivered_at) AS last_news_at,
               SUM(CASE WHEN julianday(delivered_at)>=julianday(?) THEN 1 ELSE 0 END) AS news_24h
               FROM deliveries WHERE delivery_status='sent'""",
            (since,),
        ).fetchone()
        slots = db.execute(
            """SELECT job_type,status,metrics_json FROM schedule_slots
               WHERE mode='live' AND julianday(slot_at)>=julianday(?)
               ORDER BY julianday(slot_at) DESC""",
            (since,),
        ).fetchall()
    latest_search = "not_run"
    failed_searches = 0
    research_attempts = cards_created = 0
    for slot in slots:
        # A running slot has no final metrics yet. Do not turn this normal
        # observation into a failed search, or erase the last completed result.
        if slot["status"] in {"planned", "running", "skipped"}:
            continue
        try:
            metrics = json.loads(slot["metrics_json"] or "{}")
        except (ValueError, TypeError):
            metrics = {}
        if slot["job_type"] == "radar":
            search = metrics.get("search") or {}
            if not isinstance(search, dict) or search.get("status") == "already_attempted":
                continue
            state = search.get("status") or (
                "ok" if search.get("lenses_succeeded", 0) > 0 else "failed"
            )
            if latest_search == "not_run":
                latest_search = state
            if state == "failed":
                failed_searches += 1
        elif slot["job_type"] == "research":
            research_attempts += int(metrics.get("events_seen") or 0)
            cards_created += int(metrics.get("cards_created") or 0)
    enabled = repository.get_runtime_setting("schedule_enabled") == "true"
    dry = repository.get_runtime_setting("schedule_dry_run") != "false"
    count = int(row["news_24h"] or 0)
    from app.telegram_bot.news_queue import rating_gate_enabled

    waiting = False
    if rating_gate_enabled(repository):
        with repository.connection() as db:
            current = db.execute("""SELECT d.id FROM runtime_settings r
                JOIN deliveries d ON d.id=CAST(r.value AS INTEGER)
                WHERE r.key LIKE 'news_queue_current:%' AND d.delivery_status='sent'""").fetchall()
        waiting = any(not repository.current_reaction(r["id"]) for r in current)
    issues = []
    if enabled and not dry:
        if latest_search == "failed":
            issues.append("search_failed")
        if not count and not waiting and (research_attempts or failed_searches):
            issues.append("news_stalled")
    return {
        "status": "paused"
        if not enabled or dry
        else "degraded"
        if issues
        else "waiting_for_rating"
        if waiting
        else "ok",
        "waiting_for_rating": waiting,
        "news_24h": count,
        "last_news_at": row["last_news_at"],
        "latest_search": latest_search,
        "failed_searches_24h": failed_searches,
        "research_attempts_24h": research_attempts,
        "cards_created_24h": cards_created,
        "issues": issues,
    }


def feed_health_text(snapshot: dict[str, Any]) -> str:
    state = {
        "ok": "работает",
        "degraded": "требует внимания",
        "paused": "приостановлено",
        "waiting_for_rating": "ждёт оценки предыдущей новости",
    }
    search = {
        "ok": "работает",
        "failed": "ошибка",
        "disabled": "выключен",
        "budget_blocked": "остановлен по бюджету",
        "not_run": "ещё не запускался",
    }
    return (
        f"Состояние новостей: {state.get(snapshot['status'], snapshot['status'])}\n"
        f"Обычных новостей за 24 часа: {snapshot['news_24h']}\n"
        f"Последняя новость (UTC): {snapshot['last_news_at'] or 'ещё не было'}\n"
        f"Веб-поиск: {search.get(snapshot['latest_search'], snapshot['latest_search'])}\n"
        f"Проверено кандидатов / подготовлено карточек за 24 ч: "
        f"{snapshot['research_attempts_24h']} / {snapshot['cards_created_24h']}\n"
        "YouTube и PDF в счётчик новостей не входят."
    )
