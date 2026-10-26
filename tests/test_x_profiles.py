import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.dispatcher.event.bases import SkipHandler

from app.collectors.base import CollectedItem, CollectionBatch, SourceCapabilities, SourceHealth
from app.collectors.orchestrator import collect_due_sources
from app.config import Settings, runtime_source_registry
from app.db.repository import Repository
from app.radar.x_profiles import add_profile, profile_handle
from app.telegram_bot.bot import build_router
from app.telegram_bot.media_downloads import MediaFilter
from app.telegram_bot.sensory import source_menu
from app.telegram_bot.x_profiles import XProfileFilter, references


@pytest.fixture
def env(tmp_path):
    empty = tmp_path / "sources.yaml"
    empty.write_text("version: 1\nsources: []\n")
    authors = tmp_path / "authors.yaml"
    authors.write_text("version: 1\nauthors: []\n")
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        sources_file=empty,
        sensory_sources_file=empty,
        x_authors_file=authors,
        allowed_user_id=123,
    )
    repo = Repository(settings.database_path)
    repo.migrate()
    return settings, repo


def message(text="https://x.com/Polymarket", user=123, kind="private"):
    return SimpleNamespace(
        text=text,
        caption=None,
        entities=[],
        caption_entities=[],
        from_user=SimpleNamespace(id=user),
        chat=SimpleNamespace(type=kind),
        answer=AsyncMock(),
    )


def handler(settings, repo):
    return next(
        h.callback
        for h in build_router(settings, repo).message.handlers
        if h.callback.__name__ == "x_profile_handler"
    )


@pytest.mark.parametrize(
    "value",
    [
        "https://x.com/Polymarket",
        "https://twitter.com/POLYMARKET/",
        "x.com/Polymarket?s=21",
        "https://www.x.com/Polymarket",
        "@Polymarket",
        "https://mobile.twitter.com/Polymarket/with_replies",
    ],
)
def test_profile_variants(value):
    assert profile_handle(value) == "polymarket"


@pytest.mark.parametrize(
    "value",
    [
        "https://x.com/a/status/123",
        "https://x.com/i/status/123",
        "https://x.com/home",
        "https://x.com/search?q=x",
        "https://x.com/intent/user?screen_name=x",
        "https://x.com.evil.test/Polymarket",
        "https://evil@x.com/Polymarket",
        "https://x.com:443/Polymarket",
        "https://x.com/way-too-long-or-invalid",
        "https://x.com/a%2fb",
        "file:///etc/passwd",
        "https://127.0.0.1/Polymarket",
    ],
)
def test_reject_non_profiles(value):
    with pytest.raises(ValueError):
        profile_handle(value)


def test_persistence_case_insensitive_duplicates_and_pause(env):
    settings, repo = env
    row = add_profile(settings, repo, "https://x.com/Polymarket", 123)
    assert not row["existing"] and row["enabled"]
    with repo.connection() as db:
        before = dict(db.execute("SELECT * FROM x_profile_subscriptions").fetchone())
    repo.set_runtime_setting("source_enabled:" + row["source_id"], "false")
    repo = Repository(settings.database_path)  # Simulate a process restart.
    result = add_profile(settings, repo, "https://twitter.com/POLYMARKET", 123)
    assert result["existing"] and not result["enabled"]
    sources = runtime_source_registry(settings).sources
    assert len(sources) == 1 and sources[0].id == row["source_id"]
    assert sources[0].adapter_id == "monid_x_author" and not sources[0].candidate_enabled
    with repo.connection() as db:
        assert dict(db.execute("SELECT * FROM x_profile_subscriptions").fetchone()) == before
        assert db.execute("SELECT count(*) FROM sources").fetchone()[0] == 1
    assert "X — @polymarket" in str(source_menu(repo))
    assert repo.current_month_cost() == 0


def test_configured_author_preserves_tier_and_pause(env):
    settings, repo = env
    settings.x_authors_file.write_text(
        "version: 1\nauthors:\n  - handle: Polymarket\n"
        "    name: Polymarket\n    initial_tier: sparse\n    enabled: false\n"
    )
    before = runtime_source_registry(settings).sources[0]
    result = add_profile(settings, repo, "https://x.com/polymarket", 123)
    assert result["existing"] and not result["enabled"]
    after = runtime_source_registry(settings).sources
    assert len(after) == 1 and before == after[0]


@pytest.mark.parametrize("text", ["https://x.com/test/status/123", "https://x.com/i/status/123"])
def test_post_links_keep_media_route(text):
    m = message(text)
    assert not asyncio.run(XProfileFilter()(m))
    assert asyncio.run(MediaFilter()(m))


def test_handler_adds_and_confirms_and_explicit_command(env):
    settings, repo = env
    callback = handler(settings, repo)
    for text in ["https://x.com/Polymarket", "/x @Polymarket"]:
        m = message(text)
        assert asyncio.run(XProfileFilter()(m))
        asyncio.run(callback(m))
        assert "радар" in m.answer.call_args.args[0]
    with repo.connection() as db:
        assert db.execute("SELECT count(*) FROM x_profile_subscriptions").fetchone()[0] == 1


@pytest.mark.parametrize(("user", "kind"), [(456, "private"), (123, "group")])
def test_handler_is_owner_private_only(env, user, kind):
    settings, repo = env
    m = message(user=user, kind=kind)
    asyncio.run(handler(settings, repo)(m))
    m.answer.assert_not_called()
    assert not runtime_source_registry(settings).sources


def test_feedback_and_multiple_profiles(env):
    settings, repo = env
    repo.feedback_session_for_user = lambda _: {"state": "awaiting_comment"}
    with pytest.raises(SkipHandler):
        asyncio.run(handler(settings, repo)(message()))
    repo.feedback_session_for_user = lambda _: None
    m = message("https://x.com/Polymarket https://x.com/Other")
    asyncio.run(handler(settings, repo)(m))
    assert "одну ссылку" in m.answer.call_args.args[0]
    assert not runtime_source_registry(settings).sources
    m = message("/x https://x.com/home")
    asyncio.run(handler(settings, repo)(m))
    assert "Нужна ссылка" in m.answer.call_args.args[0]


def test_hidden_links_and_explicit_command(env):
    m = message("Подписаться")
    m.entities = [SimpleNamespace(type="text_link", url="https://x.com/Polymarket")]
    assert references(m) == ["https://x.com/Polymarket"]
    assert asyncio.run(XProfileFilter()(m))
    m = message("/download https://x.com/test/status/123")
    assert not asyncio.run(XProfileFilter()(m))


def test_future_only_polling_and_pause_with_mock_adapter(env):
    settings, repo = env
    row = add_profile(settings, repo, "https://x.com/Polymarket", 123)
    now = datetime.now(UTC) + timedelta(seconds=1)

    class Adapter:
        def capabilities(self):
            return SourceCapabilities()

        def healthcheck(self, source):
            return SourceHealth()

        def collect(self, request):
            return CollectionBatch(
                items=[
                    CollectedItem(
                        "old",
                        "https://x.com/Polymarket/status/1",
                        "Old post",
                        (now - timedelta(days=2)).isoformat(),
                    ),
                    CollectedItem(
                        "new", "https://x.com/Polymarket/status/2", "New post", now.isoformat()
                    ),
                ]
            )

    adapter = Adapter()
    result = collect_due_sources(
        settings, repo, source_ids=[row["source_id"]], adapters={"monid_x_author": adapter}, now=now
    )
    assert result["items_inserted"] == 1
    repo.set_runtime_setting("source_enabled:" + row["source_id"], "false")
    assert (
        collect_due_sources(
            settings, repo, adapters={"monid_x_author": adapter}, now=now + timedelta(days=1)
        )["sources_attempted"]
        == 0
    )
