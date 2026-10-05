from __future__ import annotations

import asyncio
import json

import pytest
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import EditMessageText

from app.config import Settings
from app.telegram_bot.bot import build_router, delivery_keyboard
from app.telegram_bot.feedback import Reaction
from app.telegram_bot.keyboards import DetailsCallback, NewsDepthCallback
from app.telegram_bot.news_depth import brief_card_html, depth_key
from tests.test_details_telegram import (
    FakeCallbackMessage,
    FakeCallbackQuery,
    FakeDetailsRepository,
    _router_handler,
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


class DepthMessage(FakeCallbackMessage):
    def __init__(self, calls):
        super().__init__(calls)
        self.edits = []

    async def edit_text(self, text, **kwargs):
        self.calls.append("edit_text")
        self.reply_markup = kwargs["reply_markup"]
        self.edits.append({"text": text, **kwargs})


def context():
    repo = DepthRepository()
    router = build_router(Settings(allowed_user_id=123, _env_file=None), repo)
    query = FakeCallbackQuery(repo.calls)
    query.message = DepthMessage(repo.calls)
    return repo, query, _router_handler(router, "news_depth_handler")


def test_brief_preserves_complete_lead_and_source_without_markdown():
    text = brief_card_html(CARD)
    assert text.startswith('<b><a href="https://example.com/release">')
    assert "&lt;быстрее&gt; &amp; дешевле." in text
    assert "Контекст и подробности" not in text
    assert "`" not in text


@pytest.mark.parametrize("payload", ["{}", "[]", "invalid", "null", '{"headline":""}'])
def test_legacy_card_falls_back_without_losing_information(payload):
    assert brief_card_html({**CARD, "structured_output_json": payload}) == CARD["rendered_html"]


def test_expand_collapse_is_cached_idempotent_and_does_not_generate_or_rate():
    repo, query, handler = context()
    repo.current = Reaction.SUPERLIKE.value

    async def exercise():
        await handler(query, NewsDepthCallback(delivery_id=7, expanded=True))
        await handler(query, NewsDepthCallback(delivery_id=7, expanded=True))
        await handler(query, NewsDepthCallback(delivery_id=7, expanded=False))

    asyncio.run(exercise())
    assert repo.calls == ["answer", "edit_text"] * 3
    assert repo.request is None
    assert repo.current == Reaction.SUPERLIKE.value
    assert repo.runtime[depth_key(7)] == "brief"
    assert query.message.message_id == 700
    assert [edit["text"] for edit in query.message.edits] == [
        CARD["rendered_html"],
        CARD["rendered_html"],
        brief_card_html(CARD),
    ]
    expanded = query.message.edits[0]["reply_markup"].inline_keyboard
    assert [button.text for button in expanded[0]] == ["⛔", "👎", "👍", "✅ 🔥"]
    assert [button.text for button in expanded[1]] == ["Статья · глубже", "Свернуть"]
    assert DetailsCallback.unpack(expanded[1][0].callback_data).delivery_id == 7
    assert query.message.reply_markup.inline_keyboard[1][0].text == "Подробнее"
    assert all(edit["parse_mode"] == "HTML" for edit in query.message.edits)
    assert all(
        edit["link_preview_options"].url == CARD["primary_preview_url"]
        for edit in query.message.edits
    )


def test_superreaction_comment_controls_and_article_survive_expansion():
    repo, query, handler = context()
    repo.session = {"delivery_id": 7, "reaction": "superlike", "state": "awaiting_comment"}
    asyncio.run(handler(query, NewsDepthCallback(delivery_id=7, expanded=True)))
    assert repo.session["state"] == "awaiting_comment"
    assert [row[0].text for row in query.message.reply_markup.inline_keyboard] == [
        "💬 Прокомментировать",
        "✅ Отправить молча",
        "↩️ Назад",
        "Статья · глубже",
    ]
    assert "enqueue" not in repo.calls


@pytest.mark.parametrize("field", ["user", "chat", "message", "delivery"])
def test_depth_checks_owner_and_exact_message(field):
    repo, query, handler = context()
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
                delivery_id=999 if field == "delivery" else 7,
                expanded=True,
            ),
        )
    )
    assert repo.calls == ["answer"]
    assert query.answers[0][1] is True
    assert not repo.runtime


def test_same_text_telegram_response_is_not_an_error_or_paid_retry():
    repo, query, handler = context()

    async def unchanged(*args, **kwargs):
        raise TelegramBadRequest(
            method=EditMessageText(text="text"),
            message="Bad Request: message is not modified",
        )

    query.message.edit_text = unchanged
    asyncio.run(handler(query, NewsDepthCallback(delivery_id=7, expanded=True)))
    assert repo.runtime[depth_key(7)] == "expanded"
    assert repo.calls == ["answer"]


def test_article_status_updates_do_not_reopen_a_collapsed_card():
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


def test_repaired_cached_sources_allow_one_explicit_retry_without_a_search():
    from tests.test_details_research import document

    repo = DepthRepository()
    repo.request = {
        "request_id": 1,
        "event_id": 17,
        "status": "insufficient_evidence",
        "last_error": "detail research requires an independent full-text source",
    }
    repo.event_research_documents = lambda _: [
        document(1, "https://magentic.com/stories/raise", "primary", "Official story. " * 40),
        document(2, "https://thenextweb.com/news/magentic", "corroborating", "TNW report. " * 40),
    ]
    router = build_router(Settings(allowed_user_id=123, _env_file=None), repo)
    handler = _router_handler(router, "details_callback_handler")
    query = FakeCallbackQuery(repo.calls)
    asyncio.run(handler(query, DetailsCallback(delivery_id=7)))
    assert repo.calls == ["retry", "answer", "edit"]
    assert repo.request["status"] == "queued"
