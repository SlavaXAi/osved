from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from app.config import Settings
from app.telegram_bot.bot import (
    build_router,
    delivery_keyboard,
    details_acknowledgement,
    details_button_state,
    send_detail_placeholders,
    send_ready_detail_articles,
    send_terminal_detail_results,
    sync_detail_keyboards,
)
from app.telegram_bot.feedback import FeedbackAction, Reaction
from app.telegram_bot.keyboards import (
    DetailFeedbackAction,
    DetailFeedbackCallback,
    DetailsButtonState,
    DetailsCallback,
    FeedbackCallback,
)


class FakeDetailsRepository:
    def get_runtime_setting(self, key):
        return None

    def __init__(self) -> None:
        self.request: dict[str, Any] | None = None
        self.session: dict[str, Any] | None = None
        self.current: str | None = None
        self.calls: list[str] = []
        self.detail_feedback: list[tuple[int, int, str]] = []

    def get_delivery_context(self, delivery_id: int) -> dict[str, Any] | None:
        if delivery_id != 7:
            return None
        return {
            "delivery_id": 7,
            "chat_id": 123,
            "telegram_message_id": 700,
            "card_id": 70,
            "event_id": 17,
        }

    def event_context(self, event_id: int) -> dict[str, Any] | None:
        if event_id != 17:
            return None
        return {"id": 17, "event_type": "release"}

    def detail_request_for_delivery(self, delivery_id: int) -> dict[str, Any] | None:
        assert delivery_id == 7
        return self.request

    def enqueue_detail_request(
        self,
        delivery_id: int,
        requested_by_user_id: int,
        *,
        max_daily_new: int,
    ) -> dict[str, Any]:
        assert delivery_id == 7
        assert requested_by_user_id == 123
        assert max_daily_new == 5
        self.calls.append("enqueue")
        if self.request is None:
            self.request = {
                "request_id": 1,
                "delivery_id": 7,
                "event_id": 17,
                "status": "queued",
                "stage": "queued",
                "article_id": None,
                "created": True,
                "cache_hit": False,
                "generation_key": "event:17:detail_v1",
            }
        return self.request

    def retry_detail_request(self, request_id: int) -> dict[str, Any]:
        assert request_id == 1
        assert self.request is not None
        self.calls.append("retry")
        self.request = {**self.request, "status": "queued", "stage": "queued"}
        return self.request

    def mark_detail_keyboard_synced(self, request_id: int, display_status: str) -> bool:
        assert request_id == 1
        assert display_status in {
            "pending",
            "ready",
            "sent",
            "failed",
            "insufficient_evidence",
            "budget_blocked",
        }
        return True

    def feedback_session_for_user(self, user_id: int) -> dict[str, Any] | None:
        assert user_id == 123
        return self.session

    def current_reaction(self, delivery_id: int) -> str | None:
        assert delivery_id == 7
        return self.current

    def begin_feedback_session(self, delivery_id: int, user_id: int, reaction: str) -> None:
        assert delivery_id == 7
        assert user_id == 123
        self.session = {
            "delivery_id": delivery_id,
            "reaction": reaction,
            "state": "choose_comment",
        }

    def clear_feedback_session(self, delivery_id: int) -> None:
        assert delivery_id == 7
        self.session = None

    def set_feedback_session_state(
        self,
        delivery_id: int,
        state: str,
        *,
        prompt_message_id: int,
    ) -> None:
        assert delivery_id == 7
        assert self.session is not None
        self.session = {
            **self.session,
            "state": state,
            "prompt_message_id": prompt_message_id,
        }

    def finalize_feedback(
        self,
        delivery_id: int,
        *,
        action: str,
        reaction: str,
        comment: str | None = None,
    ) -> int:
        assert delivery_id == 7
        assert action in {"reaction", "superreaction_silent", "superreaction_comment"}
        if action == "superreaction_comment":
            assert comment
        self.current = reaction
        self.session = None
        return 1

    def record_detail_feedback(
        self,
        delivery_id: int,
        article_id: int,
        *,
        action: str,
    ) -> int:
        self.detail_feedback.append((delivery_id, article_id, action))
        return 1


class FakeCallbackMessage:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.reply_markup = None
        self.chat = SimpleNamespace(id=123)
        self.message_id = 700

    async def edit_reply_markup(self, *, reply_markup) -> None:
        self.calls.append("edit")
        self.reply_markup = reply_markup

    async def reply(self, *_args, **_kwargs):
        self.calls.append("reply")
        return SimpleNamespace(message_id=701)


class FakeCallbackQuery:
    def __init__(self, calls: list[str]) -> None:
        self.from_user = SimpleNamespace(id=123)
        self.bot = object()
        self.message = FakeCallbackMessage(calls)
        self.calls = calls
        self.answers: list[tuple[str | None, bool]] = []

    async def answer(self, text: str | None = None, *, show_alert: bool = False) -> None:
        self.calls.append("answer")
        self.answers.append((text, show_alert))


def _router_handler(router, name: str):
    return next(
        handler.callback
        for handler in router.callback_query.handlers
        if handler.callback.__name__ == name
    )


def _message_handler(router, name: str):
    return next(
        handler.callback for handler in router.message.handlers if handler.callback.__name__ == name
    )


def test_details_callback_enqueues_and_answers_before_telegram_markup_edit() -> None:
    repository = FakeDetailsRepository()
    settings = Settings(allowed_user_id=123, details_enabled=True, _env_file=None)
    router = build_router(settings, repository)  # type: ignore[arg-type]
    handler = _router_handler(router, "details_callback_handler")
    query = FakeCallbackQuery(repository.calls)

    asyncio.run(handler(query, DetailsCallback(delivery_id=7)))

    assert repository.calls == ["enqueue", "answer", "edit"]
    assert query.answers == [("Собираю подробный разбор", False)]
    assert query.message.reply_markup.inline_keyboard[1][0].text == "Разбор готовится…"


def test_details_callback_rejects_a_foreign_or_stale_message() -> None:
    repository = FakeDetailsRepository()
    settings = Settings(allowed_user_id=123, details_enabled=True, _env_file=None)
    handler = _router_handler(
        build_router(settings, repository),  # type: ignore[arg-type]
        "details_callback_handler",
    )

    wrong_chat = FakeCallbackQuery(repository.calls)
    wrong_chat.message.chat.id = 999
    asyncio.run(handler(wrong_chat, DetailsCallback(delivery_id=7)))

    wrong_message = FakeCallbackQuery(repository.calls)
    wrong_message.message.message_id = 701
    asyncio.run(handler(wrong_message, DetailsCallback(delivery_id=7)))

    assert repository.calls == ["answer", "answer"]
    assert wrong_chat.answers == [("Кнопка не относится к этому сообщению", True)]
    assert wrong_message.answers == [("Кнопка не относится к этому сообщению", True)]


def test_details_callback_preserves_pending_superreaction_controls() -> None:
    repository = FakeDetailsRepository()
    repository.session = {
        "delivery_id": 7,
        "reaction": Reaction.SUPERLIKE.value,
        "state": "awaiting_comment",
    }
    settings = Settings(allowed_user_id=123, details_enabled=True, _env_file=None)
    handler = _router_handler(
        build_router(settings, repository),  # type: ignore[arg-type]
        "details_callback_handler",
    )
    query = FakeCallbackQuery(repository.calls)

    asyncio.run(handler(query, DetailsCallback(delivery_id=7)))

    assert [row[0].text for row in query.message.reply_markup.inline_keyboard] == [
        "💬 Прокомментировать",
        "✅ Отправить молча",
        "↩️ Назад",
        "Разбор готовится…",
    ]


def test_failed_details_require_an_explicit_retry_click() -> None:
    repository = FakeDetailsRepository()
    repository.request = {
        "request_id": 1,
        "delivery_id": 7,
        "event_id": 17,
        "status": "failed",
        "stage": "search",
        "created": False,
        "cache_hit": False,
    }
    settings = Settings(allowed_user_id=123, details_enabled=True, _env_file=None)
    handler = _router_handler(
        build_router(settings, repository),  # type: ignore[arg-type]
        "details_callback_handler",
    )
    query = FakeCallbackQuery(repository.calls)

    asyncio.run(handler(query, DetailsCallback(delivery_id=7)))

    assert repository.calls == ["retry", "answer", "edit"]
    assert query.message.reply_markup.inline_keyboard[1][0].text == "Разбор готовится…"


def test_detail_article_feedback_is_separate_from_news_reaction() -> None:
    repository = FakeDetailsRepository()
    repository.current = Reaction.SUPERLIKE.value
    settings = Settings(allowed_user_id=123, details_enabled=True, _env_file=None)
    handler = _router_handler(
        build_router(settings, repository),  # type: ignore[arg-type]
        "detail_feedback_callback_handler",
    )
    query = FakeCallbackQuery(repository.calls)

    asyncio.run(
        handler(
            query,
            DetailFeedbackCallback(
                delivery_id=7,
                article_id=91,
                action=DetailFeedbackAction.USEFUL,
            ),
        )
    )

    assert repository.detail_feedback == [(7, 91, "detail_useful")]
    assert repository.current == Reaction.SUPERLIKE.value
    assert query.answers == [("Оценка разбора сохранена", False)]
    assert [button.text for button in query.message.reply_markup.inline_keyboard[0]] == [
        "✅ 👍 Полезно",
        "👎 Не окупилось",
    ]


def test_delivery_keyboard_preserves_reaction_after_details_state_changes() -> None:
    repository = FakeDetailsRepository()
    repository.current = Reaction.SUPERDISLIKE.value
    repository.request = {"status": "ready"}

    keyboard = delivery_keyboard(repository, 7, user_id=123)  # type: ignore[arg-type]

    assert [button.text for button in keyboard.inline_keyboard[0]] == [
        "✅ ⛔",
        "👎",
        "👍",
        "🔥",
    ]
    assert keyboard.inline_keyboard[1][0].text == "✅ Разбор готов"


def test_details_statuses_have_explicit_labels_and_acknowledgements() -> None:
    assert details_button_state(None) == DetailsButtonState.REQUEST
    assert details_button_state({"status": "running"}) == DetailsButtonState.RUNNING
    assert details_button_state({"status": "sent"}) == DetailsButtonState.READY
    assert details_button_state({"status": "failed"}) == DetailsButtonState.RETRY
    assert (
        details_button_state({"status": "insufficient_evidence"}) == DetailsButtonState.INSUFFICIENT
    )
    assert details_button_state({"status": "budget_blocked"}) == (DetailsButtonState.BUDGET_BLOCKED)
    assert details_acknowledgement({"status": "daily_limit"})[1]
    assert details_acknowledgement({"status": "insufficient_evidence"})[1]


def test_superreaction_back_silent_and_comment_all_preserve_details_button() -> None:
    repository = FakeDetailsRepository()
    settings = Settings(allowed_user_id=123, details_enabled=True, _env_file=None)
    router = build_router(settings, repository)  # type: ignore[arg-type]
    feedback_handler = _router_handler(router, "feedback_callback_handler")

    choose = FakeCallbackQuery(repository.calls)
    asyncio.run(
        feedback_handler(
            choose,
            FeedbackCallback(delivery_id=7, action=FeedbackAction.SUPERDISLIKE),
        )
    )
    assert [row[0].text for row in choose.message.reply_markup.inline_keyboard][-1] == ("Подробнее")

    back = FakeCallbackQuery(repository.calls)
    asyncio.run(
        feedback_handler(
            back,
            FeedbackCallback(delivery_id=7, action=FeedbackAction.BACK),
        )
    )
    assert back.message.reply_markup.inline_keyboard[1][0].text == "Подробнее"

    choose_silent = FakeCallbackQuery(repository.calls)
    asyncio.run(
        feedback_handler(
            choose_silent,
            FeedbackCallback(delivery_id=7, action=FeedbackAction.SUPERLIKE),
        )
    )
    silent = FakeCallbackQuery(repository.calls)
    asyncio.run(
        feedback_handler(
            silent,
            FeedbackCallback(delivery_id=7, action=FeedbackAction.SILENT),
        )
    )
    assert [button.text for button in silent.message.reply_markup.inline_keyboard[0]] == [
        "⛔",
        "👎",
        "👍",
        "✅ 🔥",
    ]
    assert silent.message.reply_markup.inline_keyboard[1][0].text == "Подробнее"

    choose_comment = FakeCallbackQuery(repository.calls)
    asyncio.run(
        feedback_handler(
            choose_comment,
            FeedbackCallback(delivery_id=7, action=FeedbackAction.SUPERDISLIKE),
        )
    )
    request_comment = FakeCallbackQuery(repository.calls)
    asyncio.run(
        feedback_handler(
            request_comment,
            FeedbackCallback(delivery_id=7, action=FeedbackAction.COMMENT),
        )
    )
    assert repository.session is not None
    assert repository.session["state"] == "awaiting_comment"

    edited: dict[str, Any] = {}

    class FakeBot:
        async def edit_message_reply_markup(self, **kwargs) -> None:
            edited.update(kwargs)

    class FakeCommentMessage:
        from_user = SimpleNamespace(id=123)
        text = "Это принципиально важный сигнал."
        bot = FakeBot()

        async def answer(self, _text: str) -> None:
            return None

    comment_handler = _message_handler(router, "comment_handler")
    asyncio.run(comment_handler(FakeCommentMessage()))

    assert repository.current == Reaction.SUPERDISLIKE.value
    assert edited["reply_markup"].inline_keyboard[1][0].text == "Подробнее"


def test_placeholder_and_ready_delivery_edit_one_reply_in_place() -> None:
    repository = FakeDetailsRepository()
    repository.request = {
        "id": 1,
        "request_id": 1,
        "delivery_id": 7,
        "event_id": 17,
        "chat_id": 123,
        "user_id": 123,
        "status": "queued",
        "stage": "queued",
        "article_id": 91,
        "placeholder_message_id": None,
    }
    placeholder_claimed = False
    ready_claimed = False
    keyboard_synced: list[str] = []

    def claim_detail_placeholder_request():
        nonlocal placeholder_claimed
        if placeholder_claimed or repository.request["placeholder_message_id"] is not None:
            return None
        placeholder_claimed = True
        return {**repository.request, "parent_message_id": 700}

    def set_detail_placeholder_message_id(request_id: int, message_id: int) -> bool:
        assert request_id == 1
        repository.request["placeholder_message_id"] = message_id
        return True

    def release_detail_placeholder_claim(_request_id: int, _error: str | None = None) -> bool:
        return True

    def claim_ready_detail_request_for_delivery():
        nonlocal ready_claimed
        if ready_claimed or repository.request["status"] != "ready":
            return None
        ready_claimed = True
        return {
            **repository.request,
            "rendered_html": "<b>Подробный разбор</b>\n\nПроверенные факты.",
            "source_urls": ["https://example.com/evidence"],
            "parent_message_id": 700,
        }

    def mark_detail_request_sent(request_id: int) -> bool:
        assert request_id == 1
        repository.request["status"] = "sent"
        return True

    def mark_detail_delivery_error(_request_id: int, _error: str) -> bool:
        raise AssertionError("delivery should not fail")

    def mark_detail_keyboard_synced(request_id: int, display_status: str) -> bool:
        assert request_id == 1
        keyboard_synced.append(display_status)
        return True

    repository.claim_detail_placeholder_request = claim_detail_placeholder_request  # type: ignore[attr-defined]
    repository.set_detail_placeholder_message_id = (  # type: ignore[attr-defined]
        set_detail_placeholder_message_id
    )
    repository.release_detail_placeholder_claim = (  # type: ignore[attr-defined]
        release_detail_placeholder_claim
    )
    repository.claim_ready_detail_request_for_delivery = (  # type: ignore[attr-defined]
        claim_ready_detail_request_for_delivery
    )
    repository.mark_detail_request_sent = mark_detail_request_sent  # type: ignore[attr-defined]
    repository.mark_detail_delivery_error = mark_detail_delivery_error  # type: ignore[attr-defined]
    repository.mark_detail_keyboard_synced = mark_detail_keyboard_synced  # type: ignore[method-assign]

    class FakeBot:
        def __init__(self) -> None:
            self.sent: list[dict[str, Any]] = []
            self.edited_text: list[dict[str, Any]] = []
            self.edited_markup: list[dict[str, Any]] = []

        async def send_message(self, **kwargs):
            self.sent.append(kwargs)
            return SimpleNamespace(message_id=800)

        async def edit_message_text(self, **kwargs) -> None:
            self.edited_text.append(kwargs)

        async def edit_message_reply_markup(self, **kwargs) -> None:
            self.edited_markup.append(kwargs)

    bot = FakeBot()
    placeholder_result = asyncio.run(
        send_detail_placeholders(bot, repository)  # type: ignore[arg-type]
    )
    assert placeholder_result == {"claimed": 1, "sent": 1, "errors": 0}
    assert len(bot.sent) == 1
    assert bot.sent[0]["reply_parameters"].message_id == 700
    assert repository.request["placeholder_message_id"] == 800

    repository.request["status"] = "ready"
    ready_result = asyncio.run(
        send_ready_detail_articles(bot, repository)  # type: ignore[arg-type]
    )
    assert ready_result == {"ready": 1, "sent": 1, "errors": 0}
    assert len(bot.sent) == 1
    assert bot.edited_text[0]["message_id"] == 800
    assert bot.edited_text[0]["text"].startswith("<b>Подробный разбор</b>")
    assert [button.text for button in bot.edited_text[0]["reply_markup"].inline_keyboard[0]] == [
        "👍 Полезно",
        "👎 Не окупилось",
    ]
    assert bot.edited_markup[0]["reply_markup"].inline_keyboard[1][0].text == ("✅ Разбор готов")
    assert keyboard_synced == ["sent"]


def test_terminal_worker_state_is_synced_without_erasing_reactions() -> None:
    repository = FakeDetailsRepository()
    repository.current = Reaction.LIKE.value
    repository.request = {
        "id": 1,
        "request_id": 1,
        "delivery_id": 7,
        "event_id": 17,
        "chat_id": 123,
        "user_id": 123,
        "status": "insufficient_evidence",
        "display_status": "insufficient_evidence",
        "telegram_message_id": 700,
    }
    synced: list[tuple[int, str]] = []
    repository.detail_requests_needing_keyboard_sync = (  # type: ignore[attr-defined]
        lambda *, limit: [repository.request]
    )
    repository.mark_detail_keyboard_synced = (  # type: ignore[method-assign]
        lambda request_id, display_status: not synced.append((request_id, display_status))
    )

    class FakeBot:
        def __init__(self) -> None:
            self.markup = None

        async def edit_message_reply_markup(self, **kwargs) -> None:
            self.markup = kwargs["reply_markup"]

    bot = FakeBot()
    result = asyncio.run(sync_detail_keyboards(bot, repository))  # type: ignore[arg-type]

    assert result == {"seen": 1, "synced": 1, "errors": 0}
    assert synced == [(1, "insufficient_evidence")]
    assert [button.text for button in bot.markup.inline_keyboard[0]] == [
        "⛔",
        "👎",
        "✅ 👍",
        "🔥",
    ]
    assert bot.markup.inline_keyboard[1][0].text == "Недостаточно источников"


def test_terminal_worker_state_replaces_placeholder_with_honest_result() -> None:
    repository = FakeDetailsRepository()
    repository.request = {
        "id": 1,
        "request_id": 1,
        "delivery_id": 7,
        "event_id": 17,
        "chat_id": 123,
        "user_id": 123,
        "status": "insufficient_evidence",
        "stage": "delivery",
        "placeholder_message_id": 800,
        "placeholder_lease_until": "2026-08-25T08:02:00+00:00",
    }
    claimed = False

    def claim_terminal_detail_placeholder():
        nonlocal claimed
        if claimed:
            return None
        claimed = True
        return dict(repository.request)

    def mark_terminal_detail_placeholder_sent(
        request_id: int,
        *,
        status: str,
        claim_lease_until: str,
    ) -> bool:
        assert request_id == 1
        assert status == "insufficient_evidence"
        assert claim_lease_until == "2026-08-25T08:02:00+00:00"
        repository.request["stage"] = "complete"
        return True

    repository.claim_terminal_detail_placeholder = (  # type: ignore[attr-defined]
        claim_terminal_detail_placeholder
    )
    repository.mark_terminal_detail_placeholder_sent = (  # type: ignore[attr-defined]
        mark_terminal_detail_placeholder_sent
    )
    repository.release_terminal_detail_placeholder_claim = (  # type: ignore[attr-defined]
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("delivery should not fail"))
    )

    class FakeBot:
        def __init__(self) -> None:
            self.edited_text: list[dict[str, Any]] = []

        async def edit_message_text(self, **kwargs) -> None:
            self.edited_text.append(kwargs)

    bot = FakeBot()
    result = asyncio.run(
        send_terminal_detail_results(bot, repository)  # type: ignore[arg-type]
    )

    assert result == {"claimed": 1, "sent": 1, "errors": 0}
    assert bot.edited_text[0]["message_id"] == 800
    assert "не собран" in bot.edited_text[0]["text"]
    assert "reply_markup" not in bot.edited_text[0]
    assert repository.request["stage"] == "complete"
