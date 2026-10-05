from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.dispatcher.event.bases import SkipHandler
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import GetChat

from app.collectors.base import CollectedItem, CollectionBatch, SourceCapabilities, SourceHealth
from app.collectors.orchestrator import collect_due_sources
from app.config import Settings, runtime_source_registry
from app.db.repository import Repository
from app.radar.telegram_channels import (
    TelegramChannelError,
    add_channel,
    change_channel,
    channel_rows,
    channel_username,
)
from app.telegram_bot.bot import build_router
from app.telegram_bot.sensory import source_menu
from app.telegram_bot.telegram_channels import (
    TelegramChannelControl,
    TelegramChannelFilter,
    channel_references,
    menu,
)


def chat(username="test_channel", chat_id=-1001234567890, kind="channel"):
    return SimpleNamespace(id=chat_id, username=username, title="Test & Channel", type=kind)


def message(text="https://t.me/test_channel", user=123, kind="private", origin=None):
    return SimpleNamespace(
        text=text,
        caption=None,
        entities=[],
        caption_entities=[],
        forward_origin=origin,
        from_user=SimpleNamespace(id=user),
        chat=SimpleNamespace(type=kind),
        answer=AsyncMock(),
        edit_text=AsyncMock(),
        bot=SimpleNamespace(get_chat=AsyncMock(return_value=chat())),
    )


@pytest.fixture
def env(tmp_path):
    settings = Settings(_env_file=None, app_env="test", data_dir=tmp_path, allowed_user_id=123)
    repo = Repository(settings.database_path)
    repo.migrate()
    return settings, repo


def handler(router, name, callback=False):
    observer = router.callback_query if callback else router.message
    return next(h.callback for h in observer.handlers if h.callback.__name__ == name)


@pytest.mark.parametrize(
    "value",
    [
        "@Test_Channel",
        "t.me/test_channel",
        "https://t.me/Test_Channel/123?single",
        "https://t.me/s/test_channel/123",
        "https://telegram.me/test_channel/",
    ],
)
def test_public_channel_and_post_links(value):
    assert channel_username(value) == "test_channel"


@pytest.mark.parametrize(
    "value",
    [
        "https://t.me.evil.test/test_channel",
        "https://x@t.me/test_channel",
        "https://t.me:444/test_channel",
        "https://t.me/+invite",
        "https://t.me/joinchat/private",
        "https://t.me/c/123/456",
        "https://t.me/share/url?url=https://example.com",
        "https://t.me/test_channel/abc",
        "https://t.me/test_channel/1/2",
        "file:///etc/passwd",
        "https://127.0.0.1/test_channel",
        "https://t.me/test%2fchannel",
        "@invalid-name",
        "https://t.me/s/",
    ],
)
def test_reject_private_links_and_non_channel_urls(value):
    with pytest.raises(TelegramChannelError):
        channel_username(value)


def test_forward_origin_wins_over_mentions_and_captions():
    m = message(
        "Читайте https://t.me/other_channel/1", origin=SimpleNamespace(type="channel", chat=chat())
    )
    assert channel_references(m) == ["@test_channel"]
    assert asyncio.run(TelegramChannelFilter()(m))
    m.forward_origin.chat.username = None
    assert channel_references(m) == []
    assert asyncio.run(TelegramChannelFilter()(m))
    m = message(None)
    m.caption = "Смотри t.me/test_channel/1"
    assert channel_references(m) == ["t.me/test_channel/1"]


def test_text_link_and_feedback_filters():
    m = message("Вот канал")
    m.entities = [SimpleNamespace(type="text_link", url="https://t.me/test_channel")]
    assert channel_references(m) == ["https://t.me/test_channel"]
    assert not asyncio.run(TelegramChannelFilter()(message("/report https://t.me/test_channel")))
    assert not asyncio.run(TelegramChannelFilter()(message("Обычный комментарий @test_channel")))


def test_add_dedup_pause_restart_remove_restore(env):
    settings, repo = env
    added = add_channel(settings, repo, chat(), 123)
    key = added["source_id"]
    assert not added["existing"]
    assert add_channel(settings, repo, chat(), 123)["existing"]
    change_channel(settings, repo, key, "pause", 123)
    assert not add_channel(settings, repo, chat(), 123)["enabled"]
    restarted = Repository(settings.database_path)
    assert not next(r for r in channel_rows(settings, restarted) if r["source_id"] == key)[
        "enabled"
    ]
    change_channel(settings, repo, key, "resume", 123)
    assert next(r for r in channel_rows(settings, repo) if r["source_id"] == key)["enabled"]
    change_channel(settings, repo, key, "remove", 123)
    assert not any(r["source_id"] == key for r in channel_rows(settings, restarted))
    text, keyboard = source_menu(repo)
    assert key not in str(keyboard)
    removed = next(s for s in runtime_source_registry(settings).sources if s.id == key)
    assert not removed.enabled and removed.options["removed"]
    assert not add_channel(settings, repo, chat(), 123)["existing"]
    with repo.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM telegram_channels").fetchone()[0] == 1


def test_stable_identity_after_username_change_and_takeover(env):
    settings, repo = env
    key = add_channel(settings, repo, chat(), 123)["source_id"]
    assert add_channel(settings, repo, chat("new_channel"), 123)["source_id"] == key
    assert (
        next(s for s in runtime_source_registry(settings).sources if s.id == key).locator
        == "new_channel"
    )
    with pytest.raises(TelegramChannelError, match="владелец"):
        add_channel(settings, repo, chat("new_channel", -100999), 123)


@pytest.mark.parametrize("kind", ["private", "supergroup", "group"])
def test_accounts_and_groups_are_not_channels(env, kind):
    with pytest.raises(TelegramChannelError):
        add_channel(*env, chat(kind=kind), 123)


def test_static_channel_no_duplicate_and_history_role_protected(env):
    settings, repo = env
    existing = next(
        s for s in runtime_source_registry(settings).sources if s.id == "telegram_ai_newz"
    )
    result = add_channel(settings, repo, chat(existing.locator), 123)
    assert result["existing"] and result["source_id"] == existing.id
    change_channel(settings, repo, existing.id, "remove", 123)
    assert not add_channel(settings, repo, chat(existing.locator), 123)["existing"]
    history = next(
        s for s in runtime_source_registry(settings).sources if s.role.value == "own_history"
    )
    with pytest.raises(TelegramChannelError, match="архив"):
        add_channel(settings, repo, chat(history.locator, -100999), 123)


def test_preview_then_confirmation_persists_and_makes_source_collectable(env):
    settings, repo = env
    router = build_router(settings, repo)
    m = message()
    asyncio.run(handler(router, "telegram_channel_handler")(m))
    with repo.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM telegram_channels").fetchone()[0] == 0
    assert "Test &amp; Channel" in m.answer.call_args.args[0]
    buttons = m.answer.call_args.kwargs["reply_markup"].inline_keyboard
    data = TelegramChannelControl.unpack(buttons[0][0].callback_data)
    q = SimpleNamespace(from_user=m.from_user, message=m, bot=m.bot, answer=AsyncMock())
    asyncio.run(handler(router, "telegram_channel_callback", True)(q, data))
    assert "Канал добавлен" in m.edit_text.call_args.args[0]
    source = next(
        s for s in runtime_source_registry(settings).sources if s.locator == "test_channel"
    )
    assert source.admission_mode == "event_candidate" and source.role.value == "radar"
    now = datetime.now(UTC)

    class Adapter:
        def capabilities(self):
            return SourceCapabilities()

        def healthcheck(self, source):
            return SourceHealth()

        def collect(self, request):
            return CollectionBatch(
                [
                    CollectedItem(
                        "1",
                        "https://t.me/test_channel/1",
                        "Old post",
                        (now - timedelta(days=3)).isoformat(),
                    ),
                    CollectedItem(
                        "2",
                        "https://t.me/test_channel/2",
                        "Fresh post",
                        (now - timedelta(hours=1)).isoformat(),
                    ),
                ],
                next_cursor="2",
            )

    result = collect_due_sources(
        settings, repo, source_ids=[source.id], adapters={"monid_telegram": Adapter()}, now=now
    )
    assert result["items_inserted"] == 1
    again = collect_due_sources(
        settings,
        repo,
        source_ids=[source.id],
        adapters={"monid_telegram": Adapter()},
        now=now,
        force=True,
    )
    assert again["items_inserted"] == 0
    change_channel(settings, repo, source.id, "pause", 123)
    assert (
        collect_due_sources(
            settings,
            repo,
            source_ids=[source.id],
            adapters={"monid_telegram": Adapter()},
            force=True,
        )["sources_attempted"]
        == 0
    )


@pytest.mark.parametrize("user,kind", [(777, "private"), (123, "group")])
def test_owner_private_only_messages_and_callbacks(env, user, kind):
    settings, repo = env
    router = build_router(settings, repo)
    m = message(user=user, kind=kind)
    asyncio.run(handler(router, "telegram_channel_handler")(m))
    q = SimpleNamespace(from_user=m.from_user, message=m, bot=m.bot, answer=AsyncMock())
    asyncio.run(
        handler(router, "telegram_channel_callback", True)(
            q, TelegramChannelControl(action="add", key="test_channel")
        )
    )
    assert not m.bot.get_chat.called and not m.answer.called


def test_comment_priority_and_no_add_on_api_failure(env, monkeypatch):
    settings, repo = env
    router = build_router(settings, repo)
    h = handler(router, "telegram_channel_handler")
    monkeypatch.setattr(repo, "feedback_session_for_user", lambda u: {"state": "awaiting_comment"})
    with pytest.raises(SkipHandler):
        asyncio.run(h(message()))
    m = message("/tg @test_channel")
    m.bot.get_chat.side_effect = TelegramBadRequest(
        method=GetChat(chat_id="@test_channel"), message="not found"
    )
    asyncio.run(h(m))
    assert "Не получилось" in m.answer.call_args.args[0]
    with repo.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM telegram_channels").fetchone()[0] == 0


def test_management_controls_and_telegram_callback_limit(env):
    settings, repo = env
    key = add_channel(settings, repo, chat(), 123)["source_id"]
    router = build_router(settings, repo)
    m = message()
    q = SimpleNamespace(from_user=m.from_user, message=m, bot=m.bot, answer=AsyncMock())
    cb = handler(router, "telegram_channel_callback", True)
    for action in ["view", "pause", "resume", "confirm", "remove", "list", "cancel"]:
        asyncio.run(cb(q, TelegramChannelControl(action=action, key=key)))
    for rows in [menu(settings, repo)[1].inline_keyboard, m.answer.call_args_list]:
        if rows and isinstance(rows[0], list):
            assert all(
                len(b.callback_data.encode()) <= 64 for row in rows for b in row if b.callback_data
            )


def test_config_read_without_database_does_not_create_it(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path / "absent")
    runtime_source_registry(settings)
    assert not settings.database_path.exists()
