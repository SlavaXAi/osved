import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.ops.feed_health import feed_health
from app.ops.schedule import ScheduledJobExecutor, configure_schedule
from app.telegram_bot.news_queue import current_news, send_next_news
from tests.test_news_queue import fixture


def setup(tmp_path, monkeypatch, count=20):
    settings, repo, send = fixture(tmp_path, monkeypatch, count=count)
    repo.set_runtime_setting("news_delivery_mode", "hybrid")
    bot = SimpleNamespace(session=SimpleNamespace(close=AsyncMock()))
    monkeypatch.setattr("app.telegram_bot.bot.build_bot", lambda settings: bot)
    return settings, repo, send, bot


def test_confirmed_external_send_does_not_repeat_or_leak_to_other_readers(tmp_path, monkeypatch):
    from app.ops.schedule import fresh_delivery_card_ids

    _, repo, _, _ = setup(tmp_path, monkeypatch, count=2)
    with repo.connection() as db:
        ids = [r[0] for r in db.execute("SELECT id FROM cards ORDER BY id")]
        db.execute(
            "UPDATE cards SET source_urls_json=? WHERE id=?",
            (json.dumps(["https://example.com/already-sent/"]), ids[0]),
        )
        db.execute(
            "UPDATE cards SET source_urls_json=? WHERE id=?",
            (json.dumps(["https://example.com/new"]), ids[1]),
        )
    repo.set_runtime_setting(
        "news_external_receipts:123",
        json.dumps([{"source_urls": ["https://example.com/already-sent"], "message_id": 352}]),
    )
    assert fresh_delivery_card_ids(repo, owner_id=123, now=datetime.now(UTC), limit=5) == [ids[1]]
    assert set(fresh_delivery_card_ids(repo, owner_id=456, now=datetime.now(UTC), limit=5)) == set(
        ids
    )


def test_three_baseline_batches_arrive_without_any_rating(tmp_path, monkeypatch):
    settings, repo, send, bot = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(
        ScheduledJobExecutor, "daily_video_recommendation", lambda self, owner_id: {"sent": 0}
    )
    job = ScheduledJobExecutor(settings, repo, now=datetime.now(UTC))
    outcomes = [job.delivery() for _ in range(3)]
    assert [x["news_sent"] for x in outcomes] == [5, 5, 5]
    ids = [d for x in outcomes for d in x["delivery_ids"]]
    assert len(set(ids)) == 15
    assert send.await_count == 15
    assert current_news(repo, 123)["id"] == ids[-1]
    assert repo.current_reaction(ids[-1]) is None


def test_only_rating_last_card_unlocks_an_extra_batch(tmp_path, monkeypatch):
    settings, repo, send, bot = setup(tmp_path, monkeypatch)
    first = asyncio.run(send_next_news(settings, repo, bot=bot, limit=5))
    earlier, last = first["delivery_ids"][0], first["delivery_ids"][-1]
    repo.finalize_feedback(earlier, action="reaction", reaction="like")
    assert (
        asyncio.run(send_next_news(settings, repo, bot=bot, after_delivery_id=earlier, limit=3))[
            "status"
        ]
        == "already_advanced"
    )
    assert (
        asyncio.run(send_next_news(settings, repo, bot=bot, after_delivery_id=last, limit=3))[
            "status"
        ]
        == "waiting_for_rating"
    )
    repo.begin_feedback_session(last, 123, "superlike")
    assert (
        asyncio.run(send_next_news(settings, repo, bot=bot, after_delivery_id=last, limit=3))[
            "sent"
        ]
        == 0
    )
    repo.finalize_feedback(last, action="superreaction_silent", reaction="superlike")
    more = asyncio.run(send_next_news(settings, repo, bot=bot, after_delivery_id=last, limit=3))
    assert more["sent"] == 3 and send.await_count == 8
    assert (
        asyncio.run(send_next_news(settings, repo, bot=bot, after_delivery_id=last, limit=3))[
            "status"
        ]
        == "already_advanced"
    )
    # A later baseline does not need a rating of the last extra card either.
    assert asyncio.run(send_next_news(settings, repo, bot=bot, limit=5))["sent"] == 5


def test_schedule_and_rating_cannot_interleave_two_batches(tmp_path, monkeypatch):
    settings, repo, send, bot = setup(tmp_path, monkeypatch)
    first = asyncio.run(send_next_news(settings, repo, bot=bot, limit=5))
    last = first["delivery_id"]
    repo.finalize_feedback(last, action="reaction", reaction="dislike")

    async def slow_send(*args, **kwargs):
        await asyncio.sleep(0.01)
        return SimpleNamespace(message_id=200)

    send.side_effect = slow_send

    async def race():
        return await asyncio.gather(
            send_next_news(settings, repo, bot=bot, limit=5),
            send_next_news(settings, repo, bot=bot, after_delivery_id=last, limit=3),
        )

    outcomes = asyncio.run(race())
    assert sum(x["sent"] for x in outcomes) == 5
    assert {x["status"] for x in outcomes} == {"sent", "sending"}
    assert send.await_count == 10


def test_partial_delivery_never_auto_retries_uncertain_card(tmp_path, monkeypatch):
    settings, repo, send, bot = setup(tmp_path, monkeypatch)
    send.side_effect = [
        SimpleNamespace(message_id=101),
        SimpleNamespace(message_id=102),
        TimeoutError("unknown"),
    ]
    failed = asyncio.run(send_next_news(settings, repo, bot=bot, limit=5))
    assert failed["sent"] == 2 and failed["status"] == "delivery_confirmation_needed"
    uncertain = current_news(repo, 123)["id"]
    assert asyncio.run(send_next_news(settings, repo, bot=bot, limit=5))["sent"] == 0
    assert send.await_count == 3
    send.side_effect = None
    retried = asyncio.run(send_next_news(settings, repo, bot=bot, retry=True, limit=5))
    assert retried["sent"] == 1 and retried["delivery_id"] == uncertain
    assert send.await_count == 4


def test_hybrid_health_does_not_excuse_stalled_feed_as_waiting_for_rating(tmp_path, monkeypatch):
    settings, repo, _, bot = setup(tmp_path, monkeypatch)
    configure_schedule(repo, enabled=True, dry_run=False)
    asyncio.run(send_next_news(settings, repo, bot=bot, limit=5))
    now = datetime.now(UTC)
    with repo.connection() as db:
        db.execute("UPDATE deliveries SET delivered_at=?", ((now - timedelta(days=2)).isoformat(),))
    slot = repo.claim_schedule_slot("research", now.isoformat(), "live")
    repo.update_schedule_slot(slot, status="completed", metrics={"events_seen": 5})
    result = feed_health(repo, now=now)
    assert not result["waiting_for_rating"]
    assert "news_stalled" in result["issues"]


def test_refill_resumes_requested_extra_batch_but_never_invents_a_request(tmp_path, monkeypatch):
    from app.ops.service import resume_waiting_reader

    settings, repo, send, bot = setup(tmp_path, monkeypatch)
    configure_schedule(repo, enabled=True, dry_run=False)
    first = asyncio.run(send_next_news(settings, repo, bot=bot, limit=5))
    asyncio.run(resume_waiting_reader(settings, repo))
    assert send.await_count == 5
    repo.finalize_feedback(first["delivery_id"], action="reaction", reaction="like")
    asyncio.run(resume_waiting_reader(settings, repo))
    assert send.await_count == 8
    asyncio.run(resume_waiting_reader(settings, repo))
    assert send.await_count == 8


def test_rating_confirms_uncertain_delivery_and_prevents_duplicate(tmp_path, monkeypatch):
    settings, repo, send, bot = setup(tmp_path, monkeypatch)
    send.side_effect = TimeoutError("acknowledgement lost after send")
    assert asyncio.run(send_next_news(settings, repo, bot=bot, limit=5))["errors"] == 1
    uncertain = current_news(repo, 123)["id"]
    repo.finalize_feedback(uncertain, action="reaction", reaction="like")
    send.side_effect = None
    result = asyncio.run(
        send_next_news(settings, repo, bot=bot, after_delivery_id=uncertain, limit=3)
    )
    assert result["sent"] == 3
    assert uncertain not in result["delivery_ids"]


def test_actual_rating_callback_requests_three_not_one(tmp_path, monkeypatch):
    from app.telegram_bot.bot import build_router
    from app.telegram_bot.feedback import FeedbackAction
    from app.telegram_bot.keyboards import FeedbackCallback

    settings, repo, send, bot = setup(tmp_path, monkeypatch)
    first = asyncio.run(send_next_news(settings, repo, bot=bot, limit=5))
    router = build_router(settings, repo)
    handler = next(
        x.callback
        for x in router.callback_query.handlers
        if x.callback.__name__ == "feedback_callback_handler"
    )
    query = SimpleNamespace(
        from_user=SimpleNamespace(id=123),
        bot=bot,
        answer=AsyncMock(),
        message=SimpleNamespace(edit_reply_markup=AsyncMock()),
    )
    asyncio.run(
        handler(
            query,
            FeedbackCallback(
                delivery_id=first["delivery_id"],
                action=FeedbackAction.DISLIKE,
            ),
        )
    )
    assert send.await_count == 8
    asyncio.run(
        handler(
            query,
            FeedbackCallback(
                delivery_id=first["delivery_ids"][0],
                action=FeedbackAction.LIKE,
            ),
        )
    )
    assert send.await_count == 8
