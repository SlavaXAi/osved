from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from aiogram.types import CallbackQuery, Chat, Message, User
from pydantic import SecretStr

from app.collectors.browser_queue import BrowserTaskError, BrowserTaskQueue
from app.config import Settings, SourceConfig, SourceRegistry
from app.db.repository import Repository
from app.telegram_bot.bot import build_router
from app.telegram_bot.keyboards import DetailsCallback
from app.telegram_bot.sensory import (
    SourceControl,
    SourcePage,
    VideoControl,
    radar_status_text,
    send_video_recommendations,
    source_menu,
    video_description,
)

OWNER = 123
VIDEO_URL = "https://www.youtube.com/watch?v=abcdefghijk"


class FakeBot:
    def __init__(self):
        self.sent = []

    async def send_message(self, **kwargs):
        self.sent.append(kwargs)
        return SimpleNamespace(message_id=700 + len(self.sent))


class FakeMessage:
    def __init__(self, user_id=OWNER, *, message_id=701, chat_id=None):
        self.from_user = SimpleNamespace(id=user_id)
        self.chat = SimpleNamespace(id=chat_id if chat_id is not None else user_id)
        self.message_id = message_id
        self.bot = FakeBot()
        self.answers = []
        self.edits = []
        self.reply_markup = None

    async def answer(self, text, **kwargs):
        self.answers.append((text, kwargs))
        return SimpleNamespace(message_id=800 + len(self.answers))

    async def edit_text(self, text, **kwargs):
        self.edits.append((text, kwargs))
        self.reply_markup = kwargs.get("reply_markup")

    async def edit_reply_markup(self, *, reply_markup):
        self.reply_markup = reply_markup
        self.edits.append(("markup", reply_markup))


class FakeCallback:
    def __init__(self, user_id=OWNER, *, message_id=701, chat_id=None):
        self.from_user = SimpleNamespace(id=user_id)
        self.message = FakeMessage(user_id, message_id=message_id, chat_id=chat_id)
        self.bot = self.message.bot
        self.answers = []

    async def answer(self, text=None, **kwargs):
        self.answers.append((text, kwargs))


def handler(router, name, *, message=False):
    observer = router.message if message else router.callback_query
    return next(item.callback for item in observer.handlers if item.callback.__name__ == name)


def buttons(markup):
    return [button for row in markup.inline_keyboard for button in row]


def make_source(index=0, **updates):
    return SourceConfig(
        **{
            "id": f"source_{index}",
            "name": f"Источник {index}",
            "kind": "rss",
            "locator": f"https://example.com/feed/{index}",
            "role": "radar",
            "region": "global",
            "language": "en",
            **updates,
        }
    )


@pytest.fixture
def environment(tmp_path, monkeypatch):
    settings = Settings(
        app_env="test",
        data_dir=tmp_path,
        allowed_user_id=OWNER,
        _env_file=None,
        openrouter_api_key=None,
    )
    repository = Repository(settings.database_path)
    repository.migrate()
    sources = [make_source(index) for index in range(10)]
    sources += [
        make_source(
            20,
            id="youtube_test",
            name="Tech podcast",
            kind="youtube",
            locator="channel",
            admission_mode="trigger_only",
            enabled=True,
        )
    ]
    repository.sync_sources(SourceRegistry(version=1, sources=sources))

    def no_network(*args, **kwargs):
        raise AssertionError("A network transport must not be called by this fixture")

    monkeypatch.setattr("app.telegram_bot.sensory.http_transport", no_network)
    return settings, repository, build_router(settings, repository)


def insert_video(repository, *, chat_id=OWNER, recommendation=False):
    item_id = repository.insert_source_item_with_id(
        source_id="youtube_test",
        external_id="abcdefghijk",
        source_url=VIDEO_URL,
        original_text="Researcher discusses agent reliability and deployment failures.",
        original_language="en",
        published_at=datetime.now(UTC).isoformat(),
        outbound_urls=[VIDEO_URL],
        media_metadata={},
        payload_hash="video-fixture",
        raw_payload={"video_id": "abcdefghijk"},
    )
    if not recommendation:
        return item_id
    with repository.connection() as db:
        cursor = db.execute(
            """INSERT INTO video_recommendations
            (source_item_id,chat_id,telegram_message_id,description_html,status,created_at)
            VALUES(?,?,701,'Test description','sent',?)""",
            (item_id, chat_id, datetime.now(UTC).isoformat()),
        )
        return int(cursor.lastrowid)


def database_snapshot(repository):
    with repository.connection() as db:
        return list(db.iterdump())


def test_source_menu_paginates_and_clamps_page(environment):
    _, repository, _ = environment
    first, keyboard = source_menu(repository, 0)
    assert "1/2" in first
    assert len([b for b in buttons(keyboard) if b.callback_data.startswith("src:")]) == 8
    assert any(b.callback_data == SourcePage(page=1).pack() for b in buttons(keyboard))
    last, keyboard = source_menu(repository, 999)
    assert "2/2" in last
    assert len([b for b in buttons(keyboard) if b.callback_data.startswith("src:")]) == 3
    assert any(b.callback_data == SourcePage(page=0).pack() for b in buttons(keyboard))
    assert "1/2" in source_menu(repository, -5)[0]
    assert "только на будущий сбор" in first
    assert "Уже найденные события остаются" in first
    assert "веб-поиск управляется отдельно" in first


def test_radar_shows_collection_time_and_three_recent_safe_source_labels(environment):
    _, repository, router = environment
    with repository.connection() as db:
        for index, status in enumerate(("error", "timeout", "schema_changed", "rate_limited")):
            stamp = f"2026-09-07T0{index}:30:00+00:00"
            db.execute(
                "INSERT INTO source_health(source_id,status,last_attempt_at,last_error) "
                "VALUES(?,?,?,?)",
                (f"source_{index}", status, stamp, "SECRET_RAW_ERROR_AND_COOKIE"),
            )
        db.execute("UPDATE sources SET name='<b>Тест & канал</b>' WHERE id='source_3'")
        for run_id, strategy, stamp in (
            ("live-run", "feed", "2026-09-07T03:30:00+00:00"),
            ("replayed", "snapshot_replay", "2026-09-07T05:30:00+00:00"),
        ):
            db.execute(
                "INSERT INTO source_runs(id,source_id,adapter_id,strategy,status,started_at,"
                "lease_until) VALUES(?,'source_0','fixture',?,'ok',?,?)",
                (run_id, strategy, stamp, stamp),
            )
    before = database_snapshot(repository)
    message = FakeMessage()
    asyncio.run(handler(router, "radar_status_handler", message=True)(message))
    text = message.answers[0][0]
    assert "07.09.2026 06:30 МСК" in text
    assert "08:30" not in text  # A replay is not a newer network collection.
    assert "&lt;b&gt;Тест &amp; канал&lt;/b&gt;" in text
    assert "временное ограничение запросов" in text
    assert "изменился формат данных" in text
    assert "источник не ответил вовремя" in text
    assert "Источник 0" not in text
    assert "Ещё источников с проблемами: 1" in text
    assert text.count("• ") == 3
    assert "SECRET_RAW" not in text
    assert "Расход OSVED за месяц: $0.0000" in text
    assert len(text) < 4096
    assert database_snapshot(repository) == before


def test_radar_empty_and_disabled_sources_do_not_become_false_problems(environment):
    _, repository, _ = environment
    with repository.connection() as db:
        db.execute("INSERT INTO source_health(source_id,status) VALUES('source_0','error')")
        db.execute(
            "INSERT INTO source_health(source_id,status) VALUES('source_1','ready_from_replay')"
        )
    repository.set_runtime_setting("source_enabled:source_0", "false")
    text = radar_status_text(repository)
    assert "Последняя попытка сбора: ещё не запускался" in text
    assert "Источников с проблемами: 0" in text
    assert "Без ошибок по последнему состоянию: 1" in text
    assert "• " not in text


def test_radar_unknown_status_never_exposes_provider_error(environment):
    _, repository, _ = environment
    with repository.connection() as db:
        db.execute(
            "INSERT INTO source_health(source_id,status,last_error) VALUES(?,?,?)",
            ("source_0", "unknown SECRET_AUTH_STATUS", "SECRET_AUTH_ERROR"),
        )
    text = radar_status_text(repository)
    assert "Источник 0 — ошибка сбора" in text
    assert "SECRET_AUTH" not in text


def test_non_owner_source_commands_and_callbacks_make_no_writes(environment):
    _, repository, router = environment
    before = database_snapshot(repository)
    query = FakeCallback(999)
    asyncio.run(
        handler(router, "source_toggle_handler")(
            query,
            SourceControl(source_id="source_0", page=0),
        )
    )
    asyncio.run(handler(router, "source_page_handler")(query, SourcePage(page=1)))
    for name in (
        "sources_handler",
        "radar_status_handler",
        "videos_handler",
        "preferences_handler",
    ):
        message = FakeMessage(999)
        asyncio.run(handler(router, name, message=True)(message))
        assert message.answers == []
        assert message.bot.sent == []
    assert query.message.edits == []
    assert database_snapshot(repository) == before


def test_owner_can_toggle_and_paginate_source_menu(environment):
    _, repository, router = environment
    query = FakeCallback()
    toggle = handler(router, "source_toggle_handler")
    asyncio.run(toggle(query, SourceControl(source_id="source_0", page=0)))
    assert repository.get_runtime_setting("source_enabled:source_0") == "false"
    assert query.answers[-1][0] == "Выключен"
    asyncio.run(toggle(query, SourceControl(source_id="source_0", page=0)))
    assert repository.get_runtime_setting("source_enabled:source_0") == "true"
    asyncio.run(handler(router, "source_page_handler")(query, SourcePage(page=1)))
    assert "2/2" in query.message.edits[-1][0]
    asyncio.run(toggle(query, SourceControl(source_id="not_configured", page=0)))
    assert repository.get_runtime_setting("source_enabled:not_configured") is None


def test_source_off_cancels_pending_browser_work_and_invalidates_lease(environment):
    _, repository, router = environment
    queue = BrowserTaskQueue(repository.database_path)
    queue.initialize()
    task_id = queue.enqueue("source_0", task_key="turn-off-regression")
    task = queue.claim()
    asyncio.run(
        handler(router, "source_toggle_handler")(
            FakeCallback(), SourceControl(source_id="source_0", page=0)
        )
    )
    assert queue.claim() is None
    assert queue.task_status(task_id)["status"] == "failed"
    with pytest.raises(BrowserTaskError):
        queue.stage(task_id, task["claim_token"], {"posts": []})


def test_first_video_card_has_no_url_or_preview_and_is_not_sent_twice(environment):
    settings, repository, _ = environment
    insert_video(repository)
    bot = FakeBot()
    result = asyncio.run(send_video_recommendations(bot, settings, repository, chat_id=OWNER))
    assert result["sent"] == 1
    sent = bot.sent[0]
    assert "http" not in sent["text"]
    assert sent["link_preview_options"].is_disabled
    assert all(button.url is None for button in buttons(sent["reply_markup"]))
    actions = [
        VideoControl.unpack(button.callback_data).action for button in buttons(sent["reply_markup"])
    ]
    assert actions == ["interest", "dismiss"]
    assert (
        asyncio.run(
            send_video_recommendations(
                bot,
                settings,
                repository,
                chat_id=OWNER,
            )
        )["sent"]
        == 0
    )
    assert len(bot.sent) == 1


def test_all_uploads_over_old_30_and_3_limits_then_new_upload_same_day(environment, monkeypatch):
    from unittest.mock import AsyncMock

    settings, repository, _ = environment
    monkeypatch.setattr("app.telegram_bot.sensory.asyncio.sleep", AsyncMock())
    insert_video(repository)
    with repository.connection() as db:
        for index in range(35):
            db.execute(
                "INSERT INTO source_items(source_id,external_id,source_url,original_text,"
                "original_language,published_at,payload_hash,raw_payload_json) "
                "SELECT source_id,?,source_url,original_text,original_language,published_at,"
                "?,raw_payload_json FROM source_items WHERE external_id='abcdefghijk'",
                (f"new{index}", f"hash{index}"),
            )
    repository.set_runtime_setting("scheduled_video:123:2026-09-12", "completed")
    bot = FakeBot()
    assert (
        asyncio.run(send_video_recommendations(bot, settings, repository, chat_id=OWNER))["sent"]
        == 36
    )
    assert (
        asyncio.run(send_video_recommendations(bot, settings, repository, chat_id=OWNER))["sent"]
        == 0
    )
    with repository.connection() as db:
        db.execute(
            "INSERT INTO source_items(source_id,external_id,source_url,original_text,"
            "original_language,published_at,payload_hash,raw_payload_json) "
            "SELECT source_id,'later',source_url,original_text,original_language,published_at,"
            "'laterhash',raw_payload_json FROM source_items WHERE external_id='abcdefghijk'"
        )
    assert (
        asyncio.run(send_video_recommendations(bot, settings, repository, chat_id=OWNER))["sent"]
        == 1
    )
    assert len(bot.sent) == 37


@pytest.mark.parametrize("mode", ["history", "personal", "before_follow", "paused"])
def test_channel_delivery_excludes_private_history_and_prefollow_uploads(environment, mode):
    settings, repository, _ = environment
    insert_video(repository)
    with repository.connection() as db:
        if mode == "history":
            db.execute("UPDATE sources SET admission_mode='history' WHERE id='youtube_test'")
        elif mode == "personal":
            db.execute("UPDATE sources SET strategy='for_you' WHERE id='youtube_test'")
        elif mode == "before_follow":
            db.execute(
                "INSERT INTO youtube_channels(channel_id,source_id,title,added_by,added_at,"
                "not_before) VALUES('UCtest','youtube_test','Test',123,datetime('now'),"
                "datetime('now','+1 day'))"
            )
        else:
            db.execute("UPDATE sources SET enabled=0 WHERE id='youtube_test'")
    assert (
        asyncio.run(send_video_recommendations(FakeBot(), settings, repository, chat_id=OWNER))[
            "sent"
        ]
        == 0
    )


def test_video_interest_reveals_short_long_and_original_url(environment):
    _, repository, router = environment
    recommendation_id = insert_video(repository, recommendation=True)
    query = FakeCallback()
    asyncio.run(
        handler(router, "video_handler")(
            query,
            VideoControl(recommendation_id=recommendation_id, action="interest"),
        )
    )
    keyboard = buttons(query.message.reply_markup)
    actions = [
        VideoControl.unpack(button.callback_data).action
        for button in keyboard
        if button.callback_data
    ]
    assert actions == ["short", "long", "dismiss"]
    assert [button.url for button in keyboard if button.url] == [VIDEO_URL]
    with repository.connection() as db:
        assert db.execute("SELECT action FROM video_feedback_events").fetchone()[0] == "interest"


@pytest.mark.parametrize(
    "user_id,chat_id,action",
    [
        (999, OWNER, "interest"),
        (OWNER, 999, "interest"),
        (OWNER, OWNER, "invented_action"),
    ],
)
def test_video_callback_rejects_non_owner_foreign_recommendation_and_action(
    environment,
    user_id,
    chat_id,
    action,
):
    _, repository, router = environment
    recommendation_id = insert_video(repository, chat_id=chat_id, recommendation=True)
    before = database_snapshot(repository)
    query = FakeCallback(user_id)
    asyncio.run(
        handler(router, "video_handler")(
            query,
            VideoControl(recommendation_id=recommendation_id, action=action),
        )
    )
    assert query.message.edits == []
    assert query.message.answers == []
    assert database_snapshot(repository) == before


@pytest.mark.parametrize("mode", ["short", "long"])
def test_video_summary_callback_queues_requested_mode_without_network(
    environment, monkeypatch, mode
):
    _, repository, router = environment
    recommendation_id = insert_video(repository, recommendation=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("Old summary pipeline must not be called")

    monkeypatch.setattr("app.video.summary.build_video_summary", forbidden)
    query = FakeCallback()
    asyncio.run(
        handler(router, "video_handler")(
            query,
            VideoControl(recommendation_id=recommendation_id, action=mode),
        )
    )
    with repository.connection() as db:
        job = dict(db.execute("SELECT * FROM video_pdf_jobs").fetchone())
    assert job["video_id"] == "abcdefghijk"
    assert job["output_kind"] == ("tldr" if mode == "short" else "pdf")
    assert job["status"] == "queued"
    assert query.message.answers == []
    assert query.answers[0][0] == (
        "Готовлю TL;DR текстом" if mode == "short" else "Готовлю подробный PDF-отчёт"
    )


def test_sensory_routes_registered_before_catchall_and_do_not_consume_details(environment):
    _, _, router = environment
    names = [entry.callback.__name__ for entry in router.message.handlers]
    catchall = next(
        index for index, entry in enumerate(router.message.handlers) if not entry.filters
    )
    assert names.index("sources_handler") < catchall
    assert names.index("videos_handler") < catchall
    query = CallbackQuery(
        id="callback",
        from_user=User(id=OWNER, is_bot=False, first_name="Owner"),
        chat_instance="instance",
        data=DetailsCallback(delivery_id=7).pack(),
        message=Message(
            message_id=701, date=datetime.now(UTC), chat=Chat(id=OWNER, type="private")
        ),
    )

    async def first_match():
        for entry in router.callback_query.handlers:
            accepted, _ = await entry.check(query)
            if accepted:
                return entry.callback.__name__
        return None

    assert asyncio.run(first_match()) == "details_callback_handler"


def test_runtime_enabled_youtube_source_can_deliver(environment):
    settings, repository, _ = environment
    insert_video(repository)
    with repository.connection() as db:
        db.execute("UPDATE sources SET enabled=0 WHERE id='youtube_test'")
    repository.set_runtime_setting("source_enabled:youtube_test", "true")
    result = asyncio.run(
        send_video_recommendations(
            FakeBot(),
            settings,
            repository,
            chat_id=OWNER,
        )
    )
    assert result["sent"] == 1


def test_metadata_description_does_not_leak_llm_supplied_urls(environment, monkeypatch):
    settings, repository, _ = environment
    settings = settings.model_copy(
        update={
            "openrouter_api_key": SecretStr("fake-key"),
            "openrouter_video_model": "video/stable",
            "openrouter_card_model": "news/editor",
        }
    )
    payload = {
        "title": "Разговор об агентах",
        "description": (
            "Надёжность агентов. Смотреть https://www.youtube.com/watch?v=abcdefghijk "
            "и https://example.com/ref?campaign=1 для деталей."
        ),
    }
    calls = []

    def transport(request, key):
        calls.append(request)
        return {
            "choices": [{"message": {"content": json.dumps(payload)}}],
            "usage": {"cost": 0.001},
        }

    monkeypatch.setattr("app.telegram_bot.sensory.http_transport", transport)
    description = video_description(
        settings,
        repository,
        {
            "id": 1,
            "source_name": "Channel",
            "original_text": "Known metadata",
        },
    )
    assert "https://" not in description
    assert "www.youtube.com" not in description
    assert calls[0]["model"] == "video/stable"


@pytest.mark.parametrize("model", ["openai/gpt-5.6-luna", "openai/gpt-4.1-mini"])
@pytest.mark.parametrize("invalid_json", [False, True])
def test_metadata_model_parameters_and_byok_receipt_are_preserved(
    environment, monkeypatch, model, invalid_json
):
    settings, repository, _ = environment
    settings = settings.model_copy(
        update={"openrouter_api_key": SecretStr("fake-key"), "openrouter_video_model": model}
    )
    calls = []

    def transport(request, key):
        calls.append(request)
        return {
            "id": "metadata-response",
            "model": model,
            "choices": [{"message": {"content": "invalid" if invalid_json else "{}"}}],
            "usage": {
                "cost": 0,
                "is_byok": True,
                "prompt_tokens": 1000,
                "completion_tokens": 200,
                "cost_details": {"upstream_inference_cost": 0.002},
            },
        }

    monkeypatch.setattr("app.telegram_bot.sensory.http_transport", transport)
    item = {"id": 42, "source_name": "Channel", "original_text": "Known metadata"}
    if invalid_json:
        with pytest.raises(json.JSONDecodeError):
            video_description(settings, repository, item)
    else:
        video_description(settings, repository, item)
    request = calls[0]
    assert request["provider"] == {
        "max_price": {"prompt": 0.5, "completion": 2.0},
        "require_parameters": True,
    }
    if "gpt-5" in model:
        assert "temperature" not in request
        assert request["reasoning"] == {"effort": "low", "exclude": True}
        assert request["max_tokens"] == 3200
    else:
        assert "reasoning" not in request
        assert request["temperature"] == 0.2
        assert request["max_tokens"] == 400
    with repository.connection() as db:
        rows = db.execute("SELECT * FROM usage_ledger").fetchall()
    assert len(rows) == 1
    assert rows[0]["cost_usd"] == 0.002
    assert rows[0]["input_units"] == 1000 and rows[0]["output_units"] == 200
    metadata = json.loads(rows[0]["metadata_json"])
    assert metadata["source_item_id"] == 42
    assert metadata["model"] == model
    assert metadata["response_id"] == "metadata-response"
    assert metadata["billing_basis"] == "openrouter_plus_byok_upstream"
    assert metadata["cost_estimated"] is False


@pytest.mark.parametrize(
    "model,text,blocked",
    [
        ("openai/gpt-5.6-luna", "Short metadata", True),
        ("openai/gpt-4.1-mini", "Short metadata", False),
        ("openai/gpt-4.1-mini", "Я" * 6000, True),
    ],
)
def test_metadata_reserves_input_bytes_and_model_output_allowance_before_call(
    environment, monkeypatch, model, text, blocked
):
    settings, repository, _ = environment
    settings = settings.model_copy(
        update={
            "openrouter_api_key": SecretStr("fake-key"),
            "openrouter_video_model": model,
            "budget_hard_stop_usd": 0.006,
        }
    )
    calls = []

    def transport(request, key):
        calls.append(request)
        return {"choices": [{"message": {"content": "{}"}}], "usage": {"cost": 0.001}}

    monkeypatch.setattr("app.telegram_bot.sensory.http_transport", transport)
    result = video_description(
        settings, repository, {"id": 42, "source_name": "Channel", "original_text": text}
    )
    assert ("отложено" in result) is blocked
    assert len(calls) == (0 if blocked else 1)
    assert repository.current_month_cost() == (0 if blocked else 0.001)


def test_metadata_missing_cost_with_unknown_model_uses_explicit_price_cap(environment, monkeypatch):
    settings, repository, _ = environment
    settings = settings.model_copy(
        update={
            "openrouter_api_key": SecretStr("fake-key"),
            "openrouter_video_model": "custom/unknown",
        }
    )
    monkeypatch.setattr(
        "app.telegram_bot.sensory.http_transport",
        lambda *_: {"choices": [{"message": {"content": "{}"}}]},
    )
    video_description(settings, repository, {"id": 42, "source_name": "Channel"})
    with repository.connection() as db:
        row = db.execute("SELECT * FROM usage_ledger").fetchone()
    metadata = json.loads(row["metadata_json"])
    assert row["cost_usd"] == metadata["reservation_usd"] > 0
    assert metadata["cost_estimated"] is True
    assert metadata["billing_basis"] == "request_reservation_estimate"
    assert metadata["source_item_id"] == 42


def test_metadata_timeout_accounts_reserved_cost_without_retry(environment, monkeypatch):
    settings, repository, _ = environment
    settings = settings.model_copy(update={"openrouter_api_key": SecretStr("fake-key")})
    calls = []

    def transport(*args):
        calls.append(args)
        raise TimeoutError("provider timeout")

    monkeypatch.setattr("app.telegram_bot.sensory.http_transport", transport)
    with pytest.raises(TimeoutError):
        video_description(settings, repository, {"id": 42, "source_name": "Channel"})
    with repository.connection() as db:
        row = db.execute("SELECT * FROM usage_ledger").fetchone()
    metadata = json.loads(row["metadata_json"])
    assert len(calls) == 1
    assert row["operation"] == "video.metadata_description.transport_error"
    assert row["cost_usd"] == metadata["reservation_usd"] > 0
    assert metadata["source_item_id"] == 42
    assert metadata["cost_estimated"] is True


@pytest.mark.parametrize("message_id,chat_id", [(999, OWNER), (701, 999)])
def test_owner_video_callback_rejects_stale_or_foreign_message(environment, message_id, chat_id):
    _, repository, router = environment
    recommendation_id = insert_video(repository, recommendation=True)
    query = FakeCallback(message_id=message_id, chat_id=chat_id)
    before = database_snapshot(repository)
    asyncio.run(
        handler(router, "video_handler")(
            query,
            VideoControl(recommendation_id=recommendation_id, action="interest"),
        )
    )
    assert query.message.edits == []
    assert database_snapshot(repository) == before
