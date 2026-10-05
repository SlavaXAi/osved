import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.db.repository import Repository
from app.ops.feed_health import feed_health
from app.ops.schedule import ScheduledJobExecutor, configure_schedule
from app.telegram_bot.bot import build_router, delivery_keyboard
from app.telegram_bot.feedback import FeedbackAction
from app.telegram_bot.keyboards import FeedbackCallback
from app.telegram_bot.news_queue import current_news, send_next_news
from tests.test_schedule import insert_card, insert_event, setup_schedule


def fixture(tmp_path, monkeypatch, count=4):
    settings, repo = setup_schedule(tmp_path)
    settings.allowed_user_id = 123
    repo.set_runtime_setting("news_delivery_mode", "on_rating")
    for _ in range(count):
        insert_card(repo, insert_event(repo))
    stamp = datetime.now(UTC).isoformat()
    with repo.connection() as db:
        db.execute("UPDATE source_items SET published_at=?", (stamp,))
        db.execute("UPDATE events SET event_at=?,latest_source_at=?", (stamp, stamp))
    send = AsyncMock(return_value=SimpleNamespace(message_id=100))
    monkeypatch.setattr("app.telegram_bot.bot.send_formatted_card", send)
    return settings, repo, send


def test_queue_restart_and_old_rating_do_not_send_extra(tmp_path, monkeypatch):
    s, r, send = fixture(tmp_path, monkeypatch)
    first = asyncio.run(send_next_news(s, r, bot=object()))
    assert first["sent"] == 1
    reopened = Repository(r.database_path)
    assert asyncio.run(send_next_news(s, reopened, bot=object()))["status"] == "waiting_for_rating"
    r.finalize_feedback(first["delivery_id"], action="reaction", reaction="like")
    second = asyncio.run(send_next_news(s, r, bot=object(), after_delivery_id=first["delivery_id"]))
    assert second["sent"] == 1
    assert second["delivery_id"] != first["delivery_id"]
    old = asyncio.run(send_next_news(s, r, bot=object(), after_delivery_id=first["delivery_id"]))
    assert old["status"] == "already_advanced"
    assert send.await_count == 2


def test_pending_superreaction_does_not_advance(tmp_path, monkeypatch):
    s, r, send = fixture(tmp_path, monkeypatch)
    first = asyncio.run(send_next_news(s, r, bot=object()))["delivery_id"]
    r.begin_feedback_session(first, 123, "superlike")
    assert asyncio.run(send_next_news(s, r, bot=object()))["status"] == "waiting_for_rating"
    assert any(
        "Подробнее" in b.text
        for row in delivery_keyboard(r, first, user_id=123).inline_keyboard
        for b in row
    )
    r.finalize_feedback(
        first, action="superreaction_comment", reaction="superlike", comment="Useful"
    )
    assert asyncio.run(send_next_news(s, r, bot=object(), after_delivery_id=first))["sent"] == 1
    assert send.await_count == 2


def test_concurrent_next_and_failed_delivery_are_not_automatically_retried(tmp_path, monkeypatch):
    s, r, send = fixture(tmp_path, monkeypatch)

    async def failure(*args, **kwargs):
        await asyncio.sleep(0.01)
        raise TimeoutError("delivery acknowledgement lost")

    send.side_effect = failure

    async def two():
        return await asyncio.gather(
            send_next_news(s, r, bot=object()), send_next_news(s, r, bot=object())
        )

    results = asyncio.run(two())
    assert {x["status"] for x in results} == {"sending", "delivery_confirmation_needed"}
    assert send.await_count == 1
    assert (
        asyncio.run(send_next_news(s, r, bot=object()))["status"] == "delivery_confirmation_needed"
    )
    send.side_effect = None
    assert asyncio.run(send_next_news(s, r, bot=object(), retry=True))["sent"] == 1
    assert send.await_count == 2


def test_live_health_distinguishes_reading_pause_from_outage(tmp_path, monkeypatch):
    s, r, _ = fixture(tmp_path, monkeypatch)
    configure_schedule(r, enabled=True, dry_run=False)
    asyncio.run(send_next_news(s, r, bot=object()))
    now = datetime.now(UTC)
    with r.connection() as db:
        db.execute("UPDATE deliveries SET delivered_at=?", ((now - timedelta(days=2)).isoformat(),))
    slot = r.claim_schedule_slot("research", now.isoformat(), "live")
    r.update_schedule_slot(slot, status="completed", metrics={"events_seen": 5, "cards_created": 2})
    result = feed_health(r, now=now)
    assert result["news_24h"] == 0
    assert result["status"] == "waiting_for_rating"
    assert "news_stalled" not in result["issues"]


def test_real_feedback_handlers_advance_only_on_final_decision(tmp_path, monkeypatch):
    s, r, send = fixture(tmp_path, monkeypatch)
    asyncio.run(send_next_news(s, r, bot=object()))
    router = build_router(s, r)
    handler = next(
        x.callback
        for x in router.callback_query.handlers
        if x.callback.__name__ == "feedback_callback_handler"
    )
    query = SimpleNamespace(
        from_user=SimpleNamespace(id=123),
        bot=object(),
        answer=AsyncMock(),
        message=SimpleNamespace(edit_reply_markup=AsyncMock()),
    )
    first = current_news(r, 123)["id"]
    asyncio.run(
        handler(query, FeedbackCallback(delivery_id=first, action=FeedbackAction.SUPERLIKE))
    )
    assert send.await_count == 1
    asyncio.run(handler(query, FeedbackCallback(delivery_id=first, action=FeedbackAction.SILENT)))
    assert send.await_count == 2
    second = current_news(r, 123)["id"]
    asyncio.run(handler(query, FeedbackCallback(delivery_id=second, action=FeedbackAction.DISLIKE)))
    assert send.await_count == 3
    asyncio.run(handler(query, FeedbackCallback(delivery_id=first, action=FeedbackAction.LIKE)))
    assert send.await_count == 3


def test_ready_backlog_avoids_paid_generation(tmp_path, monkeypatch):
    s, r, _ = fixture(tmp_path, monkeypatch, count=18)

    def forbidden(*args):
        raise AssertionError("Must not pay while sufficient fresh cards are ready")

    monkeypatch.setattr(ScheduledJobExecutor, "research_event", forbidden)
    result = ScheduledJobExecutor(s, r).research()
    assert result["status"] == "backlog_ready"
    assert result["ready_cards"] == 18


def test_refilled_queue_resumes_only_after_rating_and_not_when_paused(tmp_path, monkeypatch):
    from app.ops.service import resume_waiting_reader

    s, r, send = fixture(tmp_path, monkeypatch)
    bot = SimpleNamespace(session=SimpleNamespace(close=AsyncMock()))
    monkeypatch.setattr("app.telegram_bot.bot.build_bot", lambda settings: bot)
    configure_schedule(r, enabled=True, dry_run=False)
    # The idle scheduler must not start a stream by itself outside delivery slots.
    asyncio.run(resume_waiting_reader(s, r))
    assert send.await_count == 0
    first = asyncio.run(send_next_news(s, r, bot=bot))["delivery_id"]
    asyncio.run(resume_waiting_reader(s, r))
    assert send.await_count == 1
    r.finalize_feedback(first, action="reaction", reaction="like")
    configure_schedule(r, enabled=False, dry_run=False)
    asyncio.run(resume_waiting_reader(s, r))
    assert send.await_count == 1
    configure_schedule(r, enabled=True, dry_run=False)
    asyncio.run(resume_waiting_reader(s, r))
    assert send.await_count == 2
    asyncio.run(resume_waiting_reader(s, r))
    assert send.await_count == 2
