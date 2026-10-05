from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.dispatcher.event.bases import SkipHandler
from pydantic import SecretStr

from app.collectors.base import CollectedItem, CollectionBatch
from app.collectors.orchestrator import collect_due_sources
from app.collectors.platform_transport import MonidResponse
from app.collectors.platforms import YouTubeRSSAdapter
from app.config import Settings, runtime_source_registry
from app.db.repository import Repository
from app.telegram_bot.bot import build_router
from app.telegram_bot.channels import ChannelControl, ChannelFilter, links, menu
from app.telegram_bot.video_reports import VideoReportFilter
from app.video.channels import (
    PRICE,
    ChannelError,
    ChannelService,
    change_channel,
    channel_rows,
    channel_url,
    feed_title,
)

ID = "UC" + "a" * 22
URL = "https://www.youtube.com/@testing"
DIRECT = "https://www.youtube.com/channel/" + ID
YC = "UCcefcZRL2oaA_uBNeo5UOWg"


def xml(channel=ID, title="Test &amp; Channel"):
    return (
        f'<feed xmlns="http://www.w3.org/2005/Atom" '
        f'xmlns:yt="http://www.youtube.com/xml/schemas/2015">'
        f"<yt:channelId>{channel}</yt:channelId><title>{title}</title></feed>"
    ).encode()


@pytest.fixture
def env(tmp_path):
    settings = Settings(
        _env_file=None,
        app_env="test",
        data_dir=tmp_path,
        allowed_user_id=123,
        monid_api_key=SecretStr("fake"),
        sensory_sources_file=Path(__file__).parent / "fixtures/config/sensory-sources.yaml",
    )
    repo = Repository(settings.database_path)
    repo.migrate()
    calls = []

    def paid(url):
        calls.append(url)
        return MonidResponse({"channel_id": ID}, PRICE, "known-run")

    service = ChannelService(settings, repo, fetcher=lambda u: xml(), paid=paid)
    return settings, repo, service, calls


@pytest.mark.parametrize(
    "value",
    [
        URL,
        "https://youtube.com/@Testing?si=tracking",
        "http://m.youtube.com/@Testing/videos",
        "youtube.com/@testing/streams",
        "https://www.youtube.com/@Testing/about/",
    ],
)
def test_canonical_handle(value):
    assert channel_url(value) == URL


@pytest.mark.parametrize(
    "value",
    [
        "https://youtube.com.evil.test/@x",
        "https://youtube.com:123/@x",
        "https://name@youtube.com/@x",
        "file:///etc/passwd",
        "https://127.0.0.1/@x",
        "https://youtube.com/watch?v=abcdefghijk",
        "https://youtu.be/abcdefghijk",
        "https://youtube.com/@foo%2fbar",
        "https://youtube.com/channel/wrong",
        "https://youtube.com/@foo%0abar",
        "https://youtube.com/@foo/video/xyz",
    ],
)
def test_invalid_urls(value):
    with pytest.raises(ValueError):
        channel_url(value)


def test_legacy_and_unicode():
    assert channel_url("https://youtube.com/c/OldName/videos").endswith("/c/OldName")
    assert channel_url("https://youtube.com/user/OldName").endswith("/user/OldName")
    assert "%" in channel_url("https://youtube.com/@новостикосмоса")


def test_feed_validation():
    assert feed_title(xml(), ID) == "Test & Channel"
    assert feed_title(xml(ID[2:]), ID) == "Test & Channel"
    for bad in (xml(YC), b"<html/>", b"<!DOCTYPE feed><feed/>", b"broken", xml(title="")):
        with pytest.raises(ChannelError):
            feed_title(bad, ID)


def test_add_persistent_dedup_no_paid_repeat(env):
    settings, repo, service, calls = env
    result = service.add(URL, 123)
    assert result["title"] == "Test & Channel"
    assert len(calls) == 1
    assert service.add(URL + "?si=x", 123)["existing"]
    assert service.add(DIRECT, 123)["existing"]
    assert len(calls) == 1
    assert repo.current_month_cost() == PRICE
    # A new Repository plus registry sync simulates process restart.
    reopened = Repository(settings.database_path)
    reopened.sync_sources(runtime_source_registry(settings))
    source = next(
        s for s in runtime_source_registry(settings).sources if s.id == result["source_id"]
    )
    assert source.locator == ID and source.options["not_before"]
    assert source.admission_mode == "trigger_only" and source.poll_interval_minutes == 240
    assert len([r for r in channel_rows(settings, reopened) if r["channel_id"] == ID]) == 1


def test_direct_needs_no_provider(env):
    settings, repo, service, calls = env
    settings.monid_api_key = None
    assert service.add(DIRECT, 123)["channel_id"] == ID
    assert not calls
    with pytest.raises(ChannelError):
        service.resolve(URL)


def test_pause_resume_remove_restore_and_keep_records(env):
    settings, repo, service, calls = env
    added = service.add(URL, 123)
    source = added["source_id"]
    change_channel(settings, repo, source, "pause", 123)
    assert not next(r for r in channel_rows(settings, repo) if r["source_id"] == source)["enabled"]
    assert service.add(URL, 123)["existing"]  # A repeated link doesn't undo an intentional pause.
    assert repo.get_runtime_setting("source_enabled:" + source) == "false"
    change_channel(settings, repo, source, "resume", 123)
    assert next(r for r in channel_rows(settings, repo) if r["source_id"] == source)["enabled"]
    change_channel(settings, repo, source, "remove", 123)
    assert all(r["source_id"] != source for r in channel_rows(settings, repo))
    assert next(s for s in runtime_source_registry(settings).sources if s.id == source).options[
        "removed"
    ]
    with repo.connection() as db:
        assert db.execute("SELECT 1 FROM sources WHERE id=?", (source,)).fetchone()
    assert service.add(DIRECT, 123)["source_id"] == source
    assert len(calls) == 1


def test_static_channel_deleted_without_yaml_resurrection(env):
    settings, repo, service, calls = env
    change_channel(settings, repo, "youtube_ycombinator", "remove", 123)
    registry = runtime_source_registry(settings)
    repo.sync_sources(registry)
    seed = [s for s in registry.sources if s.locator == YC]
    assert len(seed) == 1 and not seed[0].enabled
    assert all(r["channel_id"] != YC for r in channel_rows(settings, repo))


def test_canonical_id_reuses_static_source(env):
    settings, repo, _, _ = env
    service = ChannelService(settings, repo, fetcher=lambda u: xml(YC, "Y Combinator"))
    result = service.add("https://youtube.com/channel/" + YC, 123)
    assert result["source_id"] == "youtube_ycombinator"
    assert len([s for s in runtime_source_registry(settings).sources if s.locator == YC]) == 1


def test_rss_failure_resolves_once_and_can_retry_free(env):
    settings, repo, service, calls = env
    service.fetcher = lambda u: b"<html/>"
    with pytest.raises(ChannelError):
        service.add(URL, 123)
    service.fetcher = lambda u: xml()
    service.add(URL, 123)
    assert len(calls) == 1


def test_uncertain_paid_call_never_replayed(env):
    settings, repo, service, calls = env

    def fail(url):
        calls.append(url)
        raise TimeoutError()

    service.paid = fail
    for _ in range(2):
        with pytest.raises(ChannelError):
            service.add(URL, 123)
    assert len(calls) == 1 and repo.current_month_cost() == PRICE


def test_budget_prevents_paid_call(env):
    settings, repo, service, calls = env
    repo.record_usage("test", "test", cost_usd=settings.budget_hard_stop_usd)
    with pytest.raises(ChannelError):
        service.add(URL, 123)
    assert not calls


def test_collection_fresh_only_and_disabled(env):
    settings, repo, service, calls = env
    added = service.add(DIRECT, 123)
    source = next(
        s for s in runtime_source_registry(settings).sources if s.id == added["source_id"]
    )
    cutoff = datetime.fromisoformat(source.options["not_before"])

    def item(video, stamp):
        return CollectedItem(
            external_id=video,
            canonical_url="https://www.youtube.com/watch?v=" + video,
            text="Useful new podcast description",
            published_at=stamp.isoformat(),
            content_type="video",
            discovery_mode="channel_uploads",
        )

    adapter = YouTubeRSSAdapter()
    adapter.collect = lambda request: CollectionBatch(
        status="ok",
        items=[
            item("aaaaaaaaaaa", cutoff - timedelta(days=1)),
            item("bbbbbbbbbbb", cutoff + timedelta(minutes=1)),
        ],
    )
    kwargs = dict(
        source_ids=[source.id], adapters={"youtube_rss": adapter}, now=cutoff + timedelta(minutes=2)
    )
    result = collect_due_sources(settings, repo, **kwargs)
    assert result["items_inserted"] == 1, result
    assert collect_due_sources(settings, repo, force=True, **kwargs)["items_inserted"] == 0
    change_channel(settings, repo, source.id, "pause", 123)
    assert collect_due_sources(settings, repo, force=True, **kwargs)["sources_attempted"] == 0
    change_channel(settings, repo, source.id, "remove", 123)
    repo.set_runtime_setting("source_enabled:" + source.id, "true")  # stale generic toggle
    assert collect_due_sources(settings, repo, force=True, **kwargs)["sources_attempted"] == 0


def message(text=URL, *, user_id=123, chat_type="private"):
    return SimpleNamespace(
        text=text,
        caption=None,
        entities=[],
        caption_entities=[],
        from_user=SimpleNamespace(id=user_id),
        chat=SimpleNamespace(type=chat_type, id=user_id),
        answer=AsyncMock(),
    )


def test_filters_do_not_steal_pdf_or_other_commands():
    video = message("https://youtu.be/pAoGicZqfHs")
    assert not asyncio.run(ChannelFilter()(video))
    assert asyncio.run(VideoReportFilter()(video))
    assert asyncio.run(ChannelFilter()(message()))
    assert not asyncio.run(VideoReportFilter()(message()))
    assert not asyncio.run(ChannelFilter()(message("/report " + URL)))
    assert asyncio.run(ChannelFilter()(message("/channel")))
    assert links(message("Вот " + URL)) == [URL]


def get_handler(router, name, *, callback=False):
    observer = router.callback_query if callback else router.message
    return next(x.callback for x in observer.handlers if x.callback.__name__ == name)


@pytest.mark.parametrize("user_id,chat_type", [(777, "private"), (123, "group")])
def test_owner_private_only(env, user_id, chat_type, monkeypatch):
    settings, repo, service, calls = env
    monkeypatch.setattr("app.telegram_bot.channels.ChannelService", lambda *a: service)
    router = build_router(settings, repo)
    m = message(user_id=user_id, chat_type=chat_type)
    asyncio.run(get_handler(router, "channel_handler")(m))
    assert not calls and not m.answer.called


def test_comment_priority_and_explicit_override(env, monkeypatch):
    settings, repo, service, calls = env
    monkeypatch.setattr("app.telegram_bot.channels.ChannelService", lambda *a: service)
    monkeypatch.setattr(repo, "feedback_session_for_user", lambda u: {"state": "awaiting_comment"})
    router = build_router(settings, repo)
    handler = get_handler(router, "channel_handler")
    with pytest.raises(SkipHandler):
        asyncio.run(handler(message()))
    m = message("/channel " + URL)
    asyncio.run(handler(m))
    assert len(calls) == 1
    assert "Test &amp; Channel" in m.answer.call_args.args[0]


def test_menu_callback_size_and_mutations(env, monkeypatch):
    settings, repo, service, calls = env
    source = service.add(DIRECT, 123)["source_id"]
    text, keyboard = menu(settings, repo)
    for row in keyboard.inline_keyboard:
        for b in row:
            assert len(b.callback_data.encode()) <= 64
    router = build_router(settings, repo)
    q = SimpleNamespace(from_user=SimpleNamespace(id=123), answer=AsyncMock(), message=message())
    q.message.edit_text = AsyncMock()
    callback = get_handler(router, "channel_callback", callback=True)
    for action in ("view", "pause", "resume", "confirm", "remove", "list"):
        asyncio.run(callback(q, ChannelControl(action=action, source_id=source)))
    assert all(r["source_id"] != source for r in channel_rows(settings, repo))


def test_config_does_not_create_database(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path / "absent")
    runtime_source_registry(settings)
    assert not settings.database_path.exists()
