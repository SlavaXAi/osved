import asyncio
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.ops.schedule import fresh_delivery_card_ids, fresh_research_events
from app.personalization.daily import DailyContext, engagement_examples
from app.pipeline.ranking import rank_event_candidates
from app.telegram_bot.news_queue import send_next_news
from tests.test_daily_personalization import NOW, POLICY, event
from tests.test_daily_personalization import daily_env as daily_env
from tests.test_menu_batches import card


@pytest.mark.parametrize("iso", [False, True])
def test_video_history_uses_real_job_timestamps(daily_env, iso):
    _, repo = daily_env
    for n, days, status, owner in [
        (1, 1, "sent", 123),
        (2, 61, "sent", 123),
        (3, -1, "sent", 123),
        (4, 1, "failed", 123),
        (5, 1, "sent", 456),
    ]:
        card(repo, "youtube")
        stamp = NOW - timedelta(days=days)
        value = stamp.isoformat() if iso else stamp.timestamp()
        with repo.connection() as db:
            db.execute(
                "UPDATE source_items SET external_id=? WHERE id=(SELECT max(id) FROM source_items)",
                (f"video{n}",),
            )
            db.execute(
                "INSERT INTO video_pdf_jobs(video_id,user_id,chat_id,request_message_id,"
                "output_kind,status,created_at,updated_at) VALUES(?,?,?,?,'tldr',?,?,?)",
                (f"video{n}", owner, owner, n, status, value, value),
            )
    with repo.connection() as db:
        db.execute(
            "INSERT INTO video_pdf_jobs(video_id,user_id,chat_id,request_message_id,"
            "output_kind,status,created_at,updated_at) SELECT video_id,user_id,chat_id,99,"
            "'pdf',status,created_at,updated_at FROM video_pdf_jobs WHERE video_id='video1'"
        )
    assert [r["key"] for r in engagement_examples(repo, 123, now=NOW)] == ["video:video1"]
    with repo.connection() as db:
        job = db.execute("SELECT min(id) FROM video_pdf_jobs WHERE video_id='video1'").fetchone()[0]
        db.execute(
            "INSERT INTO video_result_feedback VALUES(?,123,'dislike',?,?)",
            (job, NOW.isoformat(), NOW.isoformat()),
        )
    assert not engagement_examples(repo, 123, now=NOW)


def learned_context(strength):
    return DailyContext(
        POLICY,
        [
            dict(
                topic_weight=strength,
                created_at=NOW.isoformat(),
                traits=["ai_coding"],
                event_id=i,
                feedback_id=i,
            )
            for i in range(3)
        ],
        NOW,
    )


def test_intensity_and_headroom_affect_core_priority():
    item = event(1, "Claude Code open source harness on GitHub")
    values = [learned_context(v).adjustment(item) for v in (0, 0.5, 1, 2)]
    assert [x["feedback"] for x in values] == [0, 0.2, 0.3, 0.4]
    assert [x["adjustment"] for x in values] == [1.4, 1.6, 1.7, 1.8]
    assert all(abs(x["adjustment"]) <= POLICY.max_adjustment for x in values)


def test_personal_priority_cannot_unlock_exceptional_fatigue_relief():
    rows = [event(1, "Claude Code agent harness", 8), event(2, "International sanctions", 8)]
    result = rank_event_candidates(rows, personalization=learned_context(2), limit=2)
    for row in result:
        assert row["score_features"]["saturation"]["exceptional_penalty_factor"] == 1


@pytest.mark.parametrize("paused", ["runtime", "registry", "history"])
def test_paused_source_excluded_before_limit_and_research(daily_env, paused):
    settings, repo = daily_env
    cid, eid = card(repo)
    other, _ = card(repo, "web")
    if paused == "runtime":
        repo.set_runtime_setting("source_enabled:rss", "false")
    else:
        with repo.connection() as db:
            db.execute(
                "UPDATE sources SET "
                + ("enabled=0" if paused == "registry" else "admission_mode='history'")
                + " WHERE id='rss'"
            )
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    assert fresh_delivery_card_ids(repo, owner_id=123, now=now, limit=1, settings=settings) == [
        other
    ]
    assert not repo.card_source_is_available(cid)
    assert repo.deliverable_cards_by_ids(123, [cid, other])[0]["card_id"] == other
    assert [r["card_id"] for r in repo.deliverable_cards(123, 1)] == [other]
    with repo.connection() as db:
        db.execute("DELETE FROM cards WHERE id=?", (cid,))
        db.execute("UPDATE events SET status='candidate' WHERE id=?", (eid,))
    assert eid not in [r["id"] for r in fresh_research_events(repo, now=now)]


@pytest.mark.parametrize("retry", [False, True])
def test_queue_rechecks_source_and_preserves_frontier(daily_env, retry):
    settings, repo = daily_env
    cid, _ = card(repo)
    repo.set_runtime_setting("news_delivery_mode", "hybrid")
    if retry:
        did = repo.ensure_delivery(cid, 123)
        repo.mark_delivery_error(did, "unknown network outcome")
        repo.set_runtime_setting("news_queue_current:123", str(did))
        repo.set_runtime_setting("source_enabled:rss", "false")
    else:
        original = repo.deliverable_cards_by_ids

        def paused_after_selection(*args):
            rows = original(*args)
            repo.set_runtime_setting("source_enabled:rss", "false")
            return rows

        repo.deliverable_cards_by_ids = paused_after_selection
    before = repo.get_runtime_setting("news_queue_current:123")
    with patch(
        "app.telegram_bot.bot.send_formatted_card",
        AsyncMock(return_value=SimpleNamespace(message_id=100)),
    ) as wire:
        result = asyncio.run(send_next_news(settings, repo, bot=object(), retry=retry))
    assert result["sent"] == 0
    wire.assert_not_called()
    assert repo.get_runtime_setting("news_queue_current:123") == before


@pytest.mark.parametrize("reason", ["removed", "before_subscription"])
def test_youtube_lifecycle_blocks_old_cards(daily_env, reason):
    _, repo = daily_env
    cid, _ = card(repo, "youtube")
    with repo.connection() as db:
        published = db.execute("SELECT published_at FROM source_items").fetchone()[0]
        db.execute(
            "INSERT INTO youtube_channels VALUES('channel','youtube','Example',123,?,?,?)",
            (
                published,
                (NOW + timedelta(days=100)).isoformat()
                if reason == "before_subscription"
                else published,
                NOW.isoformat() if reason == "removed" else None,
            ),
        )
    assert not repo.card_source_is_available(cid)
    assert not repo.deliverable_cards_by_ids(123, [cid])


@pytest.mark.parametrize("explicit", [False, True])
def test_legacy_dispatch_rechecks_after_selection(daily_env, explicit):
    from app.telegram_bot.bot import send_cards_by_ids, send_pending_cards

    settings, repo = daily_env
    cid, _ = card(repo)
    method = "deliverable_cards_by_ids" if explicit else "deliverable_cards"
    original = getattr(repo, method)

    def pause(*args):
        rows = original(*args)
        repo.set_runtime_setting("source_enabled:rss", "false")
        return rows

    setattr(repo, method, pause)
    with (
        patch("app.telegram_bot.bot.build_bot", return_value=AsyncMock()),
        patch("app.telegram_bot.bot.send_formatted_card", AsyncMock()) as wire,
    ):
        result = asyncio.run(
            send_cards_by_ids(settings, repo, [cid])
            if explicit
            else send_pending_cards(settings, repo)
        )
    assert result["sent"] == 0
    wire.assert_not_called()


def test_runtime_enable_overrides_registry_pause(daily_env):
    _, repo = daily_env
    cid, _ = card(repo)
    with repo.connection() as db:
        db.execute("UPDATE sources SET enabled=0 WHERE id='rss'")
    repo.set_runtime_setting("source_enabled:rss", "true")
    assert repo.card_source_is_available(cid)
