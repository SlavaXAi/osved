from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import EditMessageText

from app.telegram_bot.bot import delivery_keyboard
from app.telegram_bot.feedback import Reaction
from app.telegram_bot.keyboards import DetailsCallback, NewsDepthCallback
from app.telegram_bot.news_depth import brief_card_html, depth_key
from app.telegram_bot.news_panel import NewsPanelControl, get_news_panel
from tests.test_details_telegram import (
    FakeDetailsRepository,
    _router_handler,
)
from tests.test_details_telegram import (
    real_reader_env as real_reader_env,
)

CARD = {
    "structured_output_json": json.dumps(
        {
            "headline": "Компания открыла новый инструмент",
            "lead": "Инструмент доступен через API. Он выполняет задачи <быстрее> & дешевле.",
        }
    ),
    "rendered_html": "<b>Компания открыла новый инструмент</b>\n\nКонтекст и подробности.",
    "primary_preview_url": "https://example.com/release",
}


class DepthRepository(FakeDetailsRepository):
    def __init__(self):
        super().__init__()
        self.runtime = {}

    def get_runtime_setting(self, key):
        return self.runtime.get(key)

    def set_runtime_setting(self, key, value):
        self.runtime[key] = value

    def get_delivery_context(self, delivery_id):
        context = super().get_delivery_context(delivery_id)
        return {**context, **CARD} if context else None


def test_brief_preserves_complete_lead_and_source_without_markdown():
    text = brief_card_html(CARD)
    assert text.startswith('<b><a href="https://example.com/release">')
    assert "&lt;быстрее&gt; &amp; дешевле." in text
    assert "Контекст и подробности" not in text
    assert "`" not in text


@pytest.mark.parametrize("payload", ["{}", "[]", "invalid", "null", '{"headline":""}'])
def test_legacy_card_falls_back_without_losing_information(payload):
    assert brief_card_html({**CARD, "structured_output_json": payload}) == CARD["rendered_html"]


def test_expansion_enqueues_once_and_back_keeps_original_message_and_reaction(real_reader_env):
    _, repo, delivery, query, router = real_reader_env
    handler = _router_handler(router, "news_depth_handler")
    repo.finalize_feedback(
        delivery["delivery_id"], action="reaction", reaction=Reaction.SUPERLIKE.value
    )

    async def exercise():
        await handler(query, NewsDepthCallback(delivery_id=delivery["delivery_id"], expanded=True))
        await handler(query, NewsDepthCallback(delivery_id=delivery["delivery_id"], expanded=True))
        await handler(query, NewsDepthCallback(delivery_id=delivery["delivery_id"], expanded=False))

    asyncio.run(exercise())
    assert query.calls == ["answer", "edit_text"] * 3
    request = repo.detail_request_for_delivery(delivery["delivery_id"], depth="expanded")
    assert request["status"] == "queued" and request["depth"] == "expanded"
    assert repo.current_reaction(delivery["delivery_id"]) == Reaction.SUPERLIKE.value
    panel = get_news_panel(repo, delivery["delivery_id"])
    assert panel["view"] == "base"
    assert query.message.message_id == 700
    edits = query.bot.edited_text
    assert all(edit["message_id"] == 700 and edit["parse_mode"] == "HTML" for edit in edits)
    assert all(edit["link_preview_options"].is_disabled for edit in edits)
    assert edits[-1]["text"] == panel["baseline_html"]
    final = edits[-1]["reply_markup"].inline_keyboard
    assert [button.text for button in final[0]] == ["⛔", "👎", "👍", "✅ 🔥"]
    assert [button.text for button in final[1]] == ["Подробнее", "Запросить разбор"]
    with repo.connection() as db:
        assert db.execute("SELECT count(*) FROM detail_requests").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM feedback_events").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM usage_ledger").fetchone()[0] == 0


@pytest.mark.parametrize("depth,multiplier", [("expanded", 2), ("article", 4)])
def test_depth_buttons_queue_the_correct_tier_with_frozen_visible_baseline(
    real_reader_env, depth, multiplier
):
    from app.details.research import reader_target_units

    _, repo, delivery, query, router = real_reader_env
    if depth == "expanded":
        handler = _router_handler(router, "news_depth_handler")
        callback = NewsDepthCallback(delivery_id=delivery["delivery_id"], expanded=True)
    else:
        handler = _router_handler(router, "details_callback_handler")
        callback = DetailsCallback(delivery_id=delivery["delivery_id"])
    asyncio.run(handler(query, callback))
    request = repo.detail_request_for_delivery(delivery["delivery_id"], depth=depth)
    reader = request["reader_context"]
    assert request["depth"] == depth and request["placeholder_message_id"] == 700
    assert reader["baseline_card_id"] == delivery["card_id"]
    assert reader_target_units(depth, reader) == multiplier * reader["baseline_units"]
    assert all(edit["message_id"] == 700 for edit in query.bot.edited_text)
    with repo.connection() as db:
        assert db.execute("SELECT count(*) FROM usage_ledger").fetchone()[0] == 0


def test_pending_comment_state_and_article_navigation_survive_expansion(real_reader_env):
    _, repo, delivery, query, router = real_reader_env
    repo.begin_feedback_session(delivery["delivery_id"], 123, "superlike")
    repo.set_feedback_session_state(
        delivery["delivery_id"], "awaiting_comment", prompt_message_id=701
    )
    asyncio.run(
        _router_handler(router, "news_depth_handler")(
            query, NewsDepthCallback(delivery_id=delivery["delivery_id"], expanded=True)
        )
    )
    assert repo.feedback_session_for_user(123)["state"] == "awaiting_comment"
    labels = [
        button.text
        for row in query.bot.edited_text[-1]["reply_markup"].inline_keyboard
        for button in row
    ]
    assert "Запросить разбор" in labels and "← Назад" in labels
    assert any(label.startswith("Готовлю") for label in labels)


@pytest.mark.parametrize("field", ["user", "chat", "message", "delivery"])
def test_depth_checks_owner_and_exact_message(real_reader_env, field):
    _, repo, delivery, query, router = real_reader_env
    handler = _router_handler(router, "news_depth_handler")
    if field == "user":
        query.from_user.id = 999
    if field == "chat":
        query.message.chat.id = 999
    if field == "message":
        query.message.message_id = 999
    asyncio.run(
        handler(
            query,
            NewsDepthCallback(
                delivery_id=999 if field == "delivery" else delivery["delivery_id"],
                expanded=True,
            ),
        )
    )
    assert query.calls == ["answer"]
    assert query.answers[0][1] is True
    with repo.connection() as db:
        assert db.execute("SELECT count(*) FROM news_panels").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM detail_requests").fetchone()[0] == 0


def test_same_text_telegram_response_is_not_an_error_or_paid_retry(real_reader_env):
    _, repo, delivery, query, router = real_reader_env
    handler = _router_handler(router, "news_depth_handler")

    async def unchanged(*args, **kwargs):
        raise TelegramBadRequest(
            method=EditMessageText(text="text"),
            message="Bad Request: message is not modified",
        )

    query.bot.edit_message_text = unchanged
    asyncio.run(
        handler(query, NewsDepthCallback(delivery_id=delivery["delivery_id"], expanded=True))
    )
    asyncio.run(
        handler(query, NewsDepthCallback(delivery_id=delivery["delivery_id"], expanded=True))
    )
    assert get_news_panel(repo, delivery["delivery_id"])["view"] == "loading"
    assert query.calls == ["answer", "answer"]
    with repo.connection() as db:
        assert db.execute("SELECT count(*) FROM detail_requests").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM usage_ledger").fetchone()[0] == 0


def test_legacy_article_status_updates_do_not_reopen_a_collapsed_card(monkeypatch):
    monkeypatch.setattr("app.telegram_bot.bot.get_news_panel", lambda *args: None)
    repo = DepthRepository()
    repo.request = {"status": "ready"}
    repo.set_runtime_setting(depth_key(7), "brief")
    assert delivery_keyboard(repo, 7).inline_keyboard[1][0].text == "Подробнее"
    repo.set_runtime_setting(depth_key(7), "expanded")
    assert delivery_keyboard(repo, 7).inline_keyboard[1][0].text == "✅ Разбор готов"
    assert all(
        len(button.callback_data.encode()) <= 64
        for row in delivery_keyboard(repo, 7).inline_keyboard
        for button in row
    )


def test_more_cached_sources_still_require_one_explicit_reader_retry(real_reader_env):
    from tests.test_details_research import document

    _, repo, delivery, query, router = real_reader_env
    handler = _router_handler(router, "details_callback_handler")
    asyncio.run(handler(query, DetailsCallback(delivery_id=delivery["delivery_id"])))
    request = repo.claim_next_detail_request()
    repo.fail_detail_request(
        request["id"],
        status="insufficient_evidence",
        error="detail research requires an independent full-text source",
    )
    documents = [
        document(1, "https://magentic.com/stories/raise", "primary", "Official story. " * 40),
        document(2, "https://thenextweb.com/news/magentic", "corroborating", "TNW report. " * 40),
    ]
    now = datetime.now(UTC)
    for item in documents:
        identifier = repo.upsert_research_document(
            canonical_url=item["url"],
            url=item["url"],
            title=item["title"],
            publisher=item["publisher"],
            source_class=item["source_class"],
            retrieved_at=now.isoformat(),
            expires_at=(now + timedelta(days=1)).isoformat(),
            content_text=item["content_text"],
            metadata={"content_kind": "scraped_fulltext"},
        )
        repo.attach_research_document(
            event_id=delivery["event_id"],
            research_document_id=identifier,
            evidence_role=item["source_class"],
            relevance=1,
        )
    asyncio.run(handler(query, DetailsCallback(delivery_id=delivery["delivery_id"])))
    assert repo.get_detail_request(request["id"])["status"] == "insufficient_evidence"
    retry = next(
        button
        for row in query.bot.edited_text[-1]["reply_markup"].inline_keyboard
        for button in row
        if button.text == "↻ Повторить"
    )
    asyncio.run(
        _router_handler(router, "news_panel_handler")(
            query, NewsPanelControl.unpack(retry.callback_data)
        )
    )
    assert repo.get_detail_request(request["id"])["status"] == "queued"
    with repo.connection() as db:
        assert db.execute("SELECT count(*) FROM detail_requests").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM usage_ledger").fetchone()[0] == 0
