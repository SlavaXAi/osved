import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.config import Settings, SourceConfig, SourceRegistry
from app.db.repository import Repository
from app.details.models import DetailArticleBuildResult
from app.details.worker import DetailWorker
from app.research.detail_policy import detail_policy_versions
from app.telegram_bot.bot import (
    build_router,
    send_ready_detail_articles,
    send_terminal_detail_results,
)
from app.telegram_bot.keyboards import (
    DetailFeedbackAction,
    DetailFeedbackCallback,
    NewsDepthCallback,
)
from app.telegram_bot.news_panel import NewsPanelControl, get_news_panel, update_loading


def handler(router, name):
    return next(h.callback for h in router.callback_query.handlers if h.callback.__name__ == name)


def query(bot, message_id=9001, user_id=123):
    return SimpleNamespace(
        bot=bot,
        from_user=SimpleNamespace(id=user_id),
        answer=AsyncMock(),
        message=SimpleNamespace(
            message_id=message_id,
            chat=SimpleNamespace(id=user_id, type="private"),
            edit_reply_markup=AsyncMock(),
        ),
    )


@pytest.fixture
def env(tmp_path):
    settings = Settings(_env_file=None, app_env="test", data_dir=tmp_path, allowed_user_id=123)
    repo = Repository(settings.database_path)
    repo.migrate()
    repo.sync_sources(
        SourceRegistry(
            version=1,
            sources=[
                SourceConfig(
                    id="news",
                    name="Fixture",
                    kind="rss",
                    locator="https://example.org/feed",
                    role="radar",
                    region="global",
                    language="en",
                )
            ],
        )
    )
    delivery = repo.create_smoke_delivery(123, "https://example.org/story")
    repo.mark_delivery_sent(delivery["delivery_id"], 9001)
    with repo.connection() as db:
        db.execute(
            "UPDATE events SET event_type='company_event' WHERE id=?", (delivery["event_id"],)
        )
    bot = SimpleNamespace(
        edit_message_text=AsyncMock(),
        send_message=AsyncMock(),
        edit_message_reply_markup=AsyncMock(),
        delete_message=AsyncMock(),
    )
    return settings, repo, delivery, bot, build_router(settings, repo)


def article_result(request, paragraphs=3):
    depth = request["depth"]
    schema, prompt = detail_policy_versions(depth)
    body = "\n\n".join(
        (
            f"Смысловая линия {i}. "
            + "Автор объясняет устройство подхода и конкретные условия применения. " * 9
        )
        for i in range(paragraphs)
    )
    target = 3 if depth == "expanded" else 6
    context = request["reader_context"]
    return DetailArticleBuildResult(
        event_id=request["event_id"],
        generation_key=f"reader-flow:{depth}:{context['context_fingerprint']}",
        schema_version=schema,
        prompt_version=prompt,
        model="offline/model",
        depth=depth,
        reader_context=context,
        provider_response_metadata={
            "actual_additional_sources": target,
            "target_additional_sources": target,
        },
        research_plan={},
        fact_snapshot={"facts": [{"text": "Verified fixture fact."}]},
        document_ids=list(range(1, target + 2)),
        structured_output={"headline": "Объяснение подхода"},
        rendered_html='<b><a href="https://example.org/story">Объяснение подхода</a></b>\n\n'
        + body,
        source_urls=[f"https://source{i}.example/article" for i in range(target)],
        validation_status="valid",
        validation_errors=[],
        cost_usd=0.01,
        expires_at=(datetime.now(UTC) + timedelta(days=7)).isoformat(),
    )


def run_builder(env, depth="expanded", paragraphs=3):
    settings, repo, delivery, _, _ = env
    request = repo.detail_request_for_delivery(delivery["delivery_id"], depth=depth)
    calls = []

    def builder(_s, _r, event_id, **kwargs):
        calls.append((event_id, kwargs["depth"]))
        assert (
            kwargs["reader_context"]["context_fingerprint"]
            == request["reader_context"]["context_fingerprint"]
        )
        return article_result(request, paragraphs)

    result = DetailWorker(settings, repo, builder=builder).run_once()
    assert result.status == "ready", result.error
    assert calls == [(delivery["event_id"], depth)]
    return repo.get_detail_request(request["id"])


def open_more(env):
    _, _, delivery, bot, router = env
    asyncio.run(
        handler(router, "news_depth_handler")(
            query(bot), NewsDepthCallback(delivery_id=delivery["delivery_id"], expanded=True)
        )
    )


def action(env, value, *, revision=-1, message_id=9001):
    _, _, delivery, bot, router = env
    q = query(bot, message_id)
    asyncio.run(
        handler(router, "news_panel_handler")(
            q,
            NewsPanelControl(delivery_id=delivery["delivery_id"], action=value, revision=revision),
        )
    )
    return q


def test_more_research_updates_original_and_back_reopens_free(env):
    _, repo, delivery, bot, _ = env
    open_more(env)
    panel = get_news_panel(repo, delivery["delivery_id"])
    assert panel["view"] == "loading"
    assert "Запрос в очереди" in bot.edit_message_text.call_args.kwargs["text"]
    request = run_builder(env)
    assert asyncio.run(send_ready_detail_articles(bot, repo))["sent"] == 1
    assert get_news_panel(repo, delivery["delivery_id"])["view"] == "expanded"
    assert bot.edit_message_text.call_args.kwargs["message_id"] == 9001
    labels = [
        b.text
        for row in bot.edit_message_text.call_args.kwargs["reply_markup"].inline_keyboard
        for b in row
    ]
    assert any("Полезно" in x for x in labels) and any("Статья" in x for x in labels)
    action(env, "base")
    assert get_news_panel(repo, delivery["delivery_id"])["view"] == "base"
    action(env, "expanded")
    assert repo.get_detail_request(request["id"])["attempt_count"] == 1
    assert get_news_panel(repo, delivery["delivery_id"])["view"] == "expanded"
    bot.send_message.assert_not_awaited()


def test_article_paginates_in_same_message_and_returns_to_cached_more(env):
    _, repo, delivery, bot, _ = env
    open_more(env)
    run_builder(env)
    asyncio.run(send_ready_detail_articles(bot, repo))
    action(env, "article")
    request = run_builder(env, "article", 9)
    asyncio.run(send_ready_detail_articles(bot, repo))
    assert "Страница 1/" in bot.edit_message_text.call_args.kwargs["text"]
    action(env, "next")
    assert get_news_panel(repo, delivery["delivery_id"])["page"] == 1
    assert bot.edit_message_text.call_args.kwargs["message_id"] == 9001
    action(env, "back")
    assert get_news_panel(repo, delivery["delivery_id"])["view"] == "expanded"
    assert repo.get_detail_request(request["id"])["attempt_count"] == 1
    bot.send_message.assert_not_awaited()


def test_worker_completion_and_progress_do_not_reopen_card_after_back(env):
    _, repo, delivery, bot, _ = env
    open_more(env)
    request = repo.detail_request_for_delivery(delivery["delivery_id"], depth="expanded")
    action(env, "base")
    calls = bot.edit_message_text.await_count
    assert not asyncio.run(
        update_loading(bot, repo, delivery["delivery_id"], request["id"], "Loading")
    )
    run_builder(env)
    assert asyncio.run(send_ready_detail_articles(bot, repo))["ready"] == 0
    assert bot.edit_message_text.await_count == calls
    action(env, "expanded")
    assert get_news_panel(repo, delivery["delivery_id"])["view"] == "expanded"
    bot.send_message.assert_not_awaited()


def test_rating_idempotent_and_stale_or_forwarded_callbacks_rejected(env):
    _, repo, delivery, bot, router = env
    open_more(env)
    request = run_builder(env)
    asyncio.run(send_ready_detail_articles(bot, repo))
    vote = DetailFeedbackCallback(
        delivery_id=delivery["delivery_id"],
        article_id=request["article_id"],
        action=DetailFeedbackAction.USEFUL,
    )
    fn = handler(router, "detail_feedback_callback_handler")
    for _ in range(2):
        asyncio.run(fn(query(bot), vote))
    with repo.connection() as db:
        assert db.execute("SELECT count(*) FROM news_result_feedback").fetchone()[0] == 1
        assert (
            db.execute(
                "SELECT count(*) FROM feedback_events WHERE action='detail_useful'"
            ).fetchone()[0]
            == 1
        )
    action(env, "base")
    q = query(bot)
    asyncio.run(fn(q, vote))
    assert q.answer.call_args.kwargs.get("show_alert") is True
    action(env, "expanded", message_id=9002)
    with repo.connection() as db:
        assert (
            db.execute("SELECT count(*) FROM detail_requests WHERE depth!='legacy'").fetchone()[0]
            == 1
        )


def test_failure_keeps_original_context_and_retry_is_explicit(env):
    _, repo, delivery, bot, _ = env
    open_more(env)
    request = repo.claim_next_detail_request()
    repo.fail_detail_request(request["id"], error="invalid prose")
    assert asyncio.run(send_terminal_detail_results(bot, repo))["sent"] == 1
    panel = get_news_panel(repo, delivery["delivery_id"])
    assert (
        panel["view"] == "error"
        and panel["baseline_html"] in bot.edit_message_text.call_args.kwargs["text"]
    )
    action(env, "expanded")
    assert repo.get_detail_request(request["id"])["status"] == "failed"
    action(env, "retry")
    retry = repo.get_detail_request(request["id"])
    assert retry["status"] == "queued" and retry["reader_context"]["retry_authorization"]
    bot.send_message.assert_not_awaited()
