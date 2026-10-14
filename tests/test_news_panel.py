from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.methods import EditMessageText

from app.config import Settings
from app.db.repository import Repository
from app.details.paging import telegram_html_visible_units
from app.research.detail_policy import detail_policy_versions
from app.telegram_bot.bot import build_router
from app.telegram_bot.feedback import FeedbackAction
from app.telegram_bot.keyboards import (
    DetailFeedbackAction,
    DetailFeedbackCallback,
    FeedbackCallback,
)
from app.telegram_bot.news_panel import (
    NewsPanelControl,
    NewsPanelProgressBot,
    artifact_pages,
    cached_back_target,
    ensure_news_panel,
    get_news_panel,
    panel_keyboard,
    refresh_markup,
    set_loading,
    show_base,
    show_error,
    show_result,
    update_loading,
)
from tests.test_details_repository import sent_delivery


@pytest.fixture
def panel_env(tmp_path):
    repo = Repository(tmp_path / "panels.sqlite")
    repo.migrate()
    delivery = sent_delivery(repo)
    draft = {
        "headline": "Компания объяснила новый механизм автоматизации",
        "lead": "Изменение связывает инструменты и проверки внутри одного рабочего процесса.",
        "headline_evidence_ids": [0],
        "lead_evidence_ids": [1],
        "sections": [],
    }
    with repo.connection() as db:
        db.execute(
            "UPDATE cards SET structured_output_json=? WHERE id=?",
            (json.dumps(draft, ensure_ascii=False), delivery["card_id"]),
        )
    panel = ensure_news_panel(repo, delivery["delivery_id"])
    bot = SimpleNamespace(
        edit_message_text=AsyncMock(),
        edit_message_reply_markup=AsyncMock(),
        send_message=AsyncMock(),
    )
    return repo, panel, bot


def request_for(repo, panel, depth="expanded"):
    return repo.enqueue_detail_request(
        panel["delivery_id"], panel["chat_id"], depth=depth, reader_context=panel["reader_context"]
    )


def completed_result(repo, panel, depth="expanded", *, long=False, metadata=None):
    request = request_for(repo, panel, depth)
    schema, prompt = detail_policy_versions(depth)
    text = '<b><a href="https://example.com/release">Связный разбор события</a></b>\n\n'
    text += "\n\n".join(
        f"Абзац {index}: "
        + "Существенные условия и механизм работы. " * 7
        + "Вывод автора сохранён."
        for index in range(20 if long else 2)
    )
    with repo.connection() as db:
        cursor = db.execute(
            "INSERT INTO detailed_articles(event_id,generation_key,schema_version,prompt_version,"
            "model,fact_snapshot_json,structured_output_json,rendered_html,source_urls_json,"
            "validation_status,expires_at,depth,reader_context_json,context_fingerprint,"
            "provider_response_metadata_json) VALUES(?,?,?,?,?,'{}','{}',?,?,'valid',?,?,?,?,?)",
            (
                request["event_id"],
                request["generation_key"],
                schema,
                prompt,
                "offline",
                text,
                json.dumps(["https://example.com/release", "https://independent.test/report"]),
                "2099-01-01T00:00:00+00:00",
                depth,
                json.dumps(panel["reader_context"]),
                panel["reader_context"]["context_fingerprint"],
                json.dumps(metadata or {}),
            ),
        )
        db.execute(
            "UPDATE detail_requests SET article_id=?,status='ready' WHERE id=?",
            (cursor.lastrowid, request["request_id"]),
        )
    request = repo.get_detail_request(request["request_id"])
    return request, repo.get_detailed_article(request["article_id"])


def labels(markup):
    return [button.text for row in markup.inline_keyboard for button in row]


def test_baseline_is_frozen_and_base_menu_keeps_four_news_reactions(panel_env):
    repo, panel, _ = panel_env
    context = panel["reader_context"]
    assert context["baseline_html"] == panel["baseline_html"]
    assert context["baseline_units"] == telegram_html_visible_units(panel["baseline_html"])
    assert len(context["context_fingerprint"]) == 64
    with repo.connection() as db:
        db.execute(
            "UPDATE cards SET rendered_html='Changed',structured_output_json='{}' WHERE id=?",
            (context["baseline_card_id"],),
        )
    assert ensure_news_panel(repo, panel["delivery_id"])["reader_context"] == context
    markup = panel_keyboard(repo, panel)
    assert [button.text for button in markup.inline_keyboard[0]] == ["⛔", "👎", "👍", "🔥"]
    assert "Подробнее" in labels(markup) and "Статья" in labels(markup)
    controls = [
        button
        for row in markup.inline_keyboard
        for button in row
        if button.callback_data and button.callback_data.startswith("np:")
    ]
    assert all(NewsPanelControl.unpack(button.callback_data).revision == 0 for button in controls)


@pytest.mark.parametrize("state", ["choose_comment", "awaiting_comment"])
def test_pending_superreaction_survives_reading_and_returns_to_base_menu(panel_env, state):
    repo, panel, bot = panel_env
    repo.begin_feedback_session(panel["delivery_id"], panel["chat_id"], "superlike", state=state)
    session_before = repo.feedback_session_for_user(panel["chat_id"])
    markup = panel_keyboard(repo, panel)
    assert labels(markup)[:3] == ["💬 Прокомментировать", "✅ Отправить молча", "↩️ Назад"]
    controls = [
        FeedbackCallback.unpack(row[0].callback_data).action for row in markup.inline_keyboard[:3]
    ]
    assert controls == [FeedbackAction.COMMENT, FeedbackAction.SILENT, FeedbackAction.BACK]
    assert "Подробнее" in labels(markup) and "Статья" in labels(markup)
    request, artifact = completed_result(repo, panel)
    assert asyncio.run(show_result(bot, repo, panel, request, artifact))
    result_labels = labels(bot.edit_message_text.call_args.kwargs["reply_markup"])
    assert "👍 Полезно" in result_labels and "👎 Не полезно" in result_labels
    assert "💬 Прокомментировать" not in result_labels
    assert asyncio.run(show_base(bot, repo, panel))
    assert labels(bot.edit_message_text.call_args.kwargs["reply_markup"])[:3] == labels(markup)[:3]
    assert repo.feedback_session_for_user(panel["chat_id"]) == session_before
    bot.send_message.assert_not_awaited()
    with repo.connection() as db:
        assert db.execute("SELECT count(*) FROM usage_ledger").fetchone()[0] == 0


def test_fire_callback_after_panel_creation_preserves_comment_controls(panel_env):
    repo, panel, bot = panel_env
    router = build_router(Settings(_env_file=None, allowed_user_id=panel["chat_id"]), repo)
    handler = next(
        item.callback
        for item in router.callback_query.handlers
        if item.callback.__name__ == "feedback_callback_handler"
    )
    message = SimpleNamespace(
        chat=SimpleNamespace(id=panel["chat_id"], type="private"),
        message_id=panel["message_id"],
        edit_reply_markup=AsyncMock(),
        reply=AsyncMock(),
    )
    query = SimpleNamespace(
        from_user=SimpleNamespace(id=panel["chat_id"]), message=message, bot=bot, answer=AsyncMock()
    )
    asyncio.run(
        handler(
            query,
            FeedbackCallback(delivery_id=panel["delivery_id"], action=FeedbackAction.SUPERLIKE),
        )
    )
    markup = bot.edit_message_reply_markup.call_args.kwargs["reply_markup"]
    assert labels(markup)[:3] == ["💬 Прокомментировать", "✅ Отправить молча", "↩️ Назад"]
    assert repo.feedback_session_for_user(panel["chat_id"])["state"] == "choose_comment"
    assert get_news_panel(repo, panel["delivery_id"])["view"] == "base"
    session = repo.feedback_session_for_user(panel["chat_id"])
    request, artifact = completed_result(repo, panel)
    assert asyncio.run(show_result(bot, repo, panel, request, artifact))
    with repo.connection() as db:
        db.execute("UPDATE events SET event_type='release' WHERE id=?", (request["event_id"],))
    back_handler = next(
        item.callback
        for item in router.callback_query.handlers
        if item.callback.__name__ == "news_panel_handler"
    )
    revision = get_news_panel(repo, panel["delivery_id"])["revision"]
    asyncio.run(
        back_handler(
            query,
            NewsPanelControl(
                delivery_id=panel["delivery_id"],
                action="back",
                revision=revision,
            ),
        )
    )
    assert labels(bot.edit_message_text.call_args.kwargs["reply_markup"])[:3] == [
        "💬 Прокомментировать",
        "✅ Отправить молча",
        "↩️ Назад",
    ]
    assert repo.feedback_session_for_user(panel["chat_id"]) == session
    assert get_news_panel(repo, panel["delivery_id"])["view"] == "base"
    message.reply.assert_not_awaited()
    bot.send_message.assert_not_awaited()
    with repo.connection() as db:
        assert db.execute("SELECT count(*) FROM usage_ledger").fetchone()[0] == 0


def test_session_from_another_news_card_does_not_replace_this_card_reactions(panel_env):
    repo, panel, _ = panel_env
    other = sent_delivery(repo)
    repo.begin_feedback_session(other["delivery_id"], panel["chat_id"], "superlike")
    markup = panel_keyboard(repo, panel)
    assert [button.text for button in markup.inline_keyboard[0]] == ["⛔", "👎", "👍", "🔥"]


def test_baseline_documents_follow_used_facts_not_all_event_urls(panel_env):
    repo, panel, _ = panel_env
    with repo.connection() as db:
        db.execute("DELETE FROM news_panels WHERE delivery_id=?", (panel["delivery_id"],))
        ids = []
        for index in range(3):
            url = f"https://source{index}.test/report"
            cursor = db.execute(
                "INSERT INTO research_documents(canonical_url,url,source_class,retrieved_at,"
                "expires_at) VALUES(?,?,'primary','2010-01-01','2099-01-01')",
                (url, url),
            )
            ids.append(cursor.lastrowid)
            db.execute(
                "INSERT INTO event_research_documents(event_id,research_document_id,evidence_role) "
                "VALUES(?,?,'primary')",
                (panel["reader_context"]["baseline_event_id"], cursor.lastrowid),
            )
        facts = [
            {"text": f"Проверенный факт {index} о механизме.", "document_ids": [document_id]}
            for index, document_id in enumerate(ids)
        ]
        db.execute(
            "INSERT INTO event_fact_sheets(event_id,fact_sheet_version,schema_version,facts_json,"
            "coverage_json,validation_status,expires_at,created_at) "
            "VALUES(?,1,'test',?,'{}','valid','2099-01-01','2010-01-01')",
            (panel["reader_context"]["baseline_event_id"], json.dumps(facts)),
        )
        db.execute(
            "UPDATE cards SET source_urls_json=? WHERE id=?",
            (
                json.dumps([f"https://source{index}.test/report" for index in range(3)]),
                panel["reader_context"]["baseline_card_id"],
            ),
        )
    reader = ensure_news_panel(repo, panel["delivery_id"])["reader_context"]
    assert reader["baseline_document_ids"] == ids[:2]
    assert "https://source2.test/report" not in reader["baseline_source_urls"]
    assert reader["baseline_source_urls"][0] == "https://example.com/release"


@pytest.mark.parametrize("structured", ["Not JSON", "[]"])
def test_legacy_card_without_usable_structure_keeps_complete_baseline(panel_env, structured):
    repo, panel, _ = panel_env
    old = '<b><a href="https://example.com/release">Старое сообщение</a></b>\n\nПолный текст.'
    with repo.connection() as db:
        db.execute("DELETE FROM news_panels WHERE delivery_id=?", (panel["delivery_id"],))
        db.execute(
            "UPDATE cards SET structured_output_json=?,rendered_html=? WHERE id=?",
            (structured, old, panel["reader_context"]["baseline_card_id"]),
        )
    current = ensure_news_panel(repo, panel["delivery_id"])
    assert current["baseline_html"] == old
    assert current["reader_context"]["baseline_document_ids"] == []


def test_result_rejects_request_with_changed_frozen_fingerprint(panel_env):
    repo, panel, bot = panel_env
    request, artifact = completed_result(repo, panel)
    with repo.connection() as db:
        reader = {**panel["reader_context"], "context_fingerprint": "f" * 64}
        db.execute(
            "UPDATE detail_requests SET reader_context_json=? WHERE id=?",
            (json.dumps(reader), request["request_id"]),
        )
    assert not asyncio.run(show_result(bot, repo, panel, request, artifact))
    bot.edit_message_text.assert_not_awaited()


def test_result_in_original_message_has_usefulness_navigation_sources_and_free_pages(panel_env):
    repo, panel, bot = panel_env
    request, artifact = completed_result(repo, panel, long=True)
    assert asyncio.run(set_loading(bot, repo, panel, request))
    assert asyncio.run(show_result(bot, repo, panel, request, artifact, from_worker=True))
    current = get_news_panel(repo, panel["delivery_id"])
    assert current["view"] == "expanded" and current["active_request_id"] is None
    kwargs = bot.edit_message_text.call_args.kwargs
    assert kwargs["message_id"] == panel["message_id"] == 9001
    assert "Источники:" in kwargs["text"] and "Страница 1/" in kwargs["text"]
    assert "👍 Полезно" in labels(kwargs["reply_markup"])
    assert "👎 Не полезно" in labels(kwargs["reply_markup"])
    assert "Следующая →" in labels(kwargs["reply_markup"])
    assert len(artifact_pages(repo, artifact)) > 1
    assert asyncio.run(
        show_result(
            bot, repo, current, request, artifact, page=1, expected_revision=current["revision"]
        )
    )
    assert get_news_panel(repo, panel["delivery_id"])["page"] == 1
    assert "👍 Полезно" in labels(panel_keyboard(repo, get_news_panel(repo, panel["delivery_id"])))
    bot.send_message.assert_not_awaited()
    with repo.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM usage_ledger").fetchone()[0] == 0


def test_back_during_loading_blocks_late_progress_and_worker_result(panel_env):
    repo, panel, bot = panel_env
    request, artifact = completed_result(repo, panel)
    assert asyncio.run(set_loading(bot, repo, panel, request))
    assert asyncio.run(show_base(bot, repo, panel))
    count = bot.edit_message_text.await_count
    assert not asyncio.run(
        update_loading(bot, repo, panel["delivery_id"], request["request_id"], "Поздний этап")
    )
    assert not asyncio.run(show_result(bot, repo, panel, request, artifact, from_worker=True))
    assert bot.edit_message_text.await_count == count
    assert get_news_panel(repo, panel["delivery_id"])["view"] == "base"


def test_progress_and_navigation_share_lock_across_network_await(panel_env):
    repo, panel, bot = panel_env
    request = request_for(repo, panel)

    async def run():
        assert await set_loading(bot, repo, panel, request)
        entered, release = asyncio.Event(), asyncio.Event()

        async def edit(**kwargs):
            if "Замедленный этап" in kwargs["text"]:
                entered.set()
                await release.wait()

        bot.edit_message_text.side_effect = edit
        pulse = asyncio.create_task(
            update_loading(
                bot, repo, panel["delivery_id"], request["request_id"], "Замедленный этап"
            )
        )
        await entered.wait()
        back = asyncio.create_task(show_base(bot, repo, panel))
        await asyncio.sleep(0)
        assert not back.done()
        release.set()
        assert await pulse and await back

    asyncio.run(run())
    assert bot.edit_message_text.call_args.kwargs["text"] == panel["baseline_html"]
    assert get_news_panel(repo, panel["delivery_id"])["view"] == "base"


def test_loading_and_error_preserve_previously_read_content_with_retry_back(panel_env):
    repo, panel, bot = panel_env
    expanded, artifact = completed_result(repo, panel)
    assert asyncio.run(show_result(bot, repo, panel, expanded, artifact))
    current = get_news_panel(repo, panel["delivery_id"])
    article = request_for(repo, panel, "article")
    assert asyncio.run(set_loading(bot, repo, current, article))
    prior = artifact_pages(repo, artifact)[0]
    assert bot.edit_message_text.call_args.kwargs["text"].startswith(prior)
    with repo.connection() as db:
        db.execute(
            "UPDATE detail_requests SET status='insufficient_evidence' WHERE id=?",
            (article["request_id"],),
        )
    assert asyncio.run(
        show_error(bot, repo, current, article, "Не хватило качественных источников.")
    )
    current = get_news_panel(repo, panel["delivery_id"])
    assert bot.edit_message_text.call_args.kwargs["text"].startswith(prior)
    assert "↻ Повторить" in labels(bot.edit_message_text.call_args.kwargs["reply_markup"])
    assert cached_back_target(repo, current)[0] == "expanded"


def test_article_back_uses_only_already_existing_expanded_result(panel_env):
    repo, panel, bot = panel_env
    article, artifact = completed_result(repo, panel, "article")
    assert asyncio.run(show_result(bot, repo, panel, article, artifact))
    current = get_news_panel(repo, panel["delivery_id"])
    assert cached_back_target(repo, current) == ("base", None, None)
    completed_result(repo, panel, "expanded")
    assert cached_back_target(repo, current)[0] == "expanded"


@pytest.mark.parametrize(
    "field,value",
    [
        ("context_fingerprint", "foreign"),
        ("depth", "article"),
        ("validation_status", "invalid"),
        ("event_id", 999),
    ],
)
def test_result_rejects_foreign_invalid_or_wrong_depth_artifacts(panel_env, field, value):
    repo, panel, bot = panel_env
    request, artifact = completed_result(repo, panel)
    if field == "event_id":
        foreign = repo.create_smoke_delivery(456, "https://foreign.test/release")
        value = foreign["event_id"]
        assert value != request["event_id"]
    with repo.connection() as db:
        db.execute(f"UPDATE detailed_articles SET {field}=? WHERE id=?", (value, artifact["id"]))
    assert not asyncio.run(show_result(bot, repo, panel, request, artifact))
    bot.edit_message_text.assert_not_awaited()


def test_old_revision_cannot_change_current_page(panel_env):
    repo, panel, bot = panel_env
    request, artifact = completed_result(repo, panel, long=True)
    assert asyncio.run(show_result(bot, repo, panel, request, artifact))
    count = bot.edit_message_text.await_count
    assert not asyncio.run(
        show_result(bot, repo, panel, request, artifact, page=1, expected_revision=0)
    )
    assert bot.edit_message_text.await_count == count


def test_existing_paid_archive_reopens_free_without_restamping(panel_env):
    repo, panel, bot = panel_env
    request, artifact = completed_result(repo, panel)
    with repo.connection() as db:
        db.execute(
            "UPDATE detailed_articles SET expires_at='2000-01-01',prompt_version='old-policy' "
            "WHERE id=?",
            (artifact["id"],),
        )
    assert asyncio.run(show_result(bot, repo, panel, request, artifact))
    assert "Сохранённый разбор от" in bot.edit_message_text.call_args.kwargs["text"]
    assert repo.get_detailed_article(artifact["id"])["prompt_version"] == "old-policy"


def test_only_real_underfilled_source_target_is_disclosed(panel_env):
    repo, panel, _ = panel_env
    _, artifact = completed_result(
        repo, panel, metadata={"actual_additional_sources": 2, "target_additional_sources": 3}
    )
    assert all(
        "Дополнительных источников: 2 из 3" in page for page in artifact_pages(repo, artifact)
    )
    with repo.connection() as db:
        db.execute(
            "UPDATE detailed_articles SET provider_response_metadata_json=? WHERE id=?",
            (
                json.dumps({"actual_additional_sources": 3, "target_additional_sources": 3}),
                artifact["id"],
            ),
        )
    assert all(
        "Дополнительных источников:" not in page
        for page in artifact_pages(repo, repo.get_detailed_article(artifact["id"]))
    )


@pytest.mark.parametrize("not_modified", [False, True])
def test_flood_wait_keeps_state_and_not_modified_commits_safe_state(panel_env, not_modified):
    repo, panel, bot = panel_env
    request = request_for(repo, panel)
    method = EditMessageText(chat_id=123, message_id=9001, text="Текст")
    bot.edit_message_text.side_effect = (
        TelegramBadRequest(method=method, message="Bad Request: message is not modified")
        if not_modified
        else TelegramRetryAfter(method=method, message="Flood wait", retry_after=1)
    )
    assert asyncio.run(set_loading(bot, repo, panel, request)) is not_modified
    assert get_news_panel(repo, panel["delivery_id"])["view"] == (
        "loading" if not_modified else "base"
    )


def test_progress_adapter_rejects_wrong_message_and_stale_request(panel_env):
    repo, panel, bot = panel_env
    request = request_for(repo, panel)
    adapter = NewsPanelProgressBot(bot, repo, panel["delivery_id"], request["request_id"])
    assert not asyncio.run(adapter.edit_message_text(chat_id=123, message_id=999, text="Этап"))
    assert not asyncio.run(adapter.edit_message_text(chat_id=123, message_id=9001, text="Этап"))
    bot.edit_message_text.assert_not_awaited()


def test_late_thumbs_markup_refresh_uses_current_base_after_back(panel_env):
    repo, panel, bot = panel_env
    request, artifact = completed_result(repo, panel)
    router = build_router(Settings(_env_file=None, allowed_user_id=panel["chat_id"]), repo)
    handler = next(
        item.callback
        for item in router.callback_query.handlers
        if item.callback.__name__ == "detail_feedback_callback_handler"
    )
    message = SimpleNamespace(
        chat=SimpleNamespace(id=panel["chat_id"]),
        message_id=panel["message_id"],
        edit_reply_markup=AsyncMock(),
    )

    async def run():
        assert await show_result(bot, repo, panel, request, artifact)

        async def back_during_acknowledgement(*args, **kwargs):
            assert await show_base(bot, repo, get_news_panel(repo, panel["delivery_id"]))

        query = SimpleNamespace(
            from_user=SimpleNamespace(id=panel["chat_id"]),
            message=message,
            bot=bot,
            answer=AsyncMock(side_effect=back_during_acknowledgement),
        )
        await handler(
            query,
            DetailFeedbackCallback(
                delivery_id=panel["delivery_id"],
                article_id=artifact["id"],
                action=DetailFeedbackAction.USEFUL,
            ),
        )

    asyncio.run(run())
    current = get_news_panel(repo, panel["delivery_id"])
    assert current["view"] == "base"
    final = bot.edit_message_reply_markup.call_args.kwargs
    assert final["message_id"] == panel["message_id"]
    assert [button.text for button in final["reply_markup"].inline_keyboard[0]] == [
        "⛔",
        "👎",
        "👍",
        "🔥",
    ]
    for row in final["reply_markup"].inline_keyboard:
        for button in row:
            if button.callback_data and button.callback_data.startswith("np:"):
                assert NewsPanelControl.unpack(button.callback_data).revision == current["revision"]
    message.edit_reply_markup.assert_not_awaited()
    with repo.connection() as db:
        assert db.execute("SELECT action FROM news_result_feedback").fetchone()[0] == "useful"
        assert db.execute("SELECT count(*) FROM usage_ledger").fetchone()[0] == 0


def test_markup_refresh_and_navigation_share_same_network_edit_lock(panel_env):
    repo, panel, bot = panel_env
    request, artifact = completed_result(repo, panel)

    async def run():
        assert await show_result(bot, repo, panel, request, artifact)
        entered, release = asyncio.Event(), asyncio.Event()
        edits = []

        async def markup(**kwargs):
            entered.set()
            await release.wait()
            edits.append("markup")

        async def text(**kwargs):
            edits.append("text")

        bot.edit_message_reply_markup.side_effect = markup
        bot.edit_message_text.side_effect = text
        refresh = asyncio.create_task(refresh_markup(bot, repo, panel["delivery_id"]))
        await entered.wait()
        back = asyncio.create_task(show_base(bot, repo, panel))
        await asyncio.sleep(0)
        assert not back.done()
        release.set()
        assert await refresh and await back
        assert edits == ["markup", "text"]

    asyncio.run(run())
    assert get_news_panel(repo, panel["delivery_id"])["view"] == "base"
    assert bot.edit_message_text.call_args.kwargs["text"] == panel["baseline_html"]
