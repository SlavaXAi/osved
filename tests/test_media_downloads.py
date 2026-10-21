from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.db.repository import Repository
from app.media.downloads import (
    Downloader,
    MediaError,
    choices_for,
    enqueue,
    get_request,
    media_link,
    normalize_info,
    request,
)
from app.telegram_bot.media_downloads import MediaAction, MediaFilter, MediaWorker, keyboard


@pytest.fixture
def env(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path, allowed_user_id=123)
    repo = Repository(settings.database_path)
    repo.migrate()
    return settings, repo


def ready(repo, message_id=10):
    req = request(
        repo, chat_id=123, message_id=message_id, link=media_link("https://youtu.be/abcdefghijk")
    )
    manifest = {
        "title": "Example",
        "items": [
            {
                "title": "Example",
                "duration": 10,
                "choices": {"360": {"selector": "18", "label": "360p", "size": 100}},
            }
        ],
    }
    with repo.connection() as db:
        db.execute(
            "UPDATE media_requests SET status='ready',manifest_json=?,panel_message_id=42 "
            "WHERE id=?",
            (json.dumps(manifest), req["id"]),
        )
    return get_request(repo, req["id"])


@pytest.mark.parametrize(
    "url,platform,ident",
    [
        ("https://youtu.be/abcdefghijk?si=tracking", "youtube", "abcdefghijk"),
        ("https://www.youtube.com/shorts/abcdefghijk", "youtube", "abcdefghijk"),
        ("https://m.youtube.com/watch?v=abcdefghijk&list=ignored", "youtube", "abcdefghijk"),
        ("https://twitter.com/name/status/123456/video/1", "x", "123456"),
        ("https://x.com/i/web/status/123456", "x", "123456"),
    ],
)
def test_canonical_media_urls(url, platform, ident):
    assert media_link(url)[:2] == (platform, ident)


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/private",
        "http://youtube.com/watch?v=abcdefghijk",
        "https://youtube.com.evil.example/watch?v=abcdefghijk",
        "https://user:pass@youtube.com/watch?v=abcdefghijk",
        "https://youtube.com:444/watch?v=abcdefghijk",
        "https://x.com/user",
        "https://youtube.com/playlist?list=123",
    ],
)
def test_unsupported_urls_never_reach_downloader(url):
    with pytest.raises(ValueError):
        media_link(url)


def test_available_formats_include_sound_and_do_not_invent_quality():
    info = {
        "platform": "youtube",
        "duration": 20,
        "formats": [
            {"format_id": "137", "height": 1080, "vcodec": "avc1", "acodec": "none", "ext": "mp4"},
            {"format_id": "140", "vcodec": "none", "acodec": "mp4a", "ext": "m4a", "abr": 128},
            {"format_id": "18", "height": 360, "vcodec": "avc1", "acodec": "mp4a", "ext": "mp4"},
        ],
    }
    choices = choices_for(info)
    assert set(choices) == {"1080", "360", "mp3"}
    assert choices["1080"]["selector"] == "137+140"
    assert choices["mp3"]["size"] == 320000
    info["formats"] = info["formats"][:1]
    assert choices_for(info) == {}  # A silent YouTube file is not a full video download.


def test_playlist_rejects_mass_download():
    with pytest.raises(MediaError, match="too_many_items"):
        normalize_info({"_type": "playlist", "entries": [{}] * 11}, "x")


def test_multiple_videos_keep_their_playlist_positions():
    entry = {
        "title": "Clip",
        "formats": [
            {
                "format_id": "http-256",
                "height": 360,
                "vcodec": "avc1",
                "acodec": "mp4a",
                "ext": "mp4",
            }
        ],
    }
    manifest = normalize_info({"_type": "playlist", "entries": [entry, entry]}, "x")
    assert [item["playlist_index"] for item in manifest["items"]] == [1, 2]
    assert all(item["multiple"] for item in manifest["items"])


def test_repeated_link_reuses_saved_format_metadata(env):
    _, repo = env
    first = ready(repo)
    second = request(
        repo, chat_id=123, message_id=11, link=media_link("https://youtu.be/abcdefghijk?si=other")
    )
    assert second["id"] != first["id"]
    assert second["status"] == "ready"
    assert second["manifest_json"] == first["manifest_json"]


def test_duplicate_request_and_double_click_are_idempotent(env):
    _, repo = env
    req = ready(repo)
    again = request(
        repo, chat_id=123, message_id=10, link=media_link("https://youtu.be/abcdefghijk")
    )
    assert req["id"] == again["id"]
    assert enqueue(repo, req, 0, "360")["id"] == enqueue(repo, req, 0, "360")["id"]
    with pytest.raises(MediaError, match="invalid_choice"):
        enqueue(repo, req, 0, "1080")


def test_keyboard_excludes_oversized_quality(env):
    _, repo = env
    req = ready(repo)
    kb = keyboard(req, 50)
    assert not any("360p" in b.text for row in kb.inline_keyboard for b in row)
    assert all(
        len(b.callback_data.encode()) <= 64
        for row in kb.inline_keyboard
        for b in row
        if b.callback_data
    )
    assert MediaAction(request_id=req["id"], action="get", choice="360").pack()


class FakeBot:
    id = 101

    def __init__(self, *, fail=False):
        self.sent = []
        self.fail = fail

    async def edit_message_text(self, **kwargs):
        return None

    async def send_video(self, **kwargs):
        self.sent.append(kwargs)
        if self.fail:
            raise TimeoutError("ambiguous send")
        return SimpleNamespace(message_id=99, video=SimpleNamespace(file_id="telegram-cache"))


class FakeDownloader:
    def __init__(self, settings):
        self.calls = 0
        self.path = settings.data_dir / "fixture.mp4"
        self.path.write_bytes(b"video")

    async def download(self, req, job, **kwargs):
        self.calls += 1
        return dict(
            path=str(self.path),
            kind="video",
            size=5,
            title="Example",
            duration=10,
            width=640,
            height=360,
        )

    def remove_job_files(self, job_id):
        pass


def test_repeated_delivery_uses_telegram_file_id_without_download(env):
    settings, repo = env
    req = ready(repo)
    bot = FakeBot()
    downloader = FakeDownloader(settings)
    worker = MediaWorker(bot, settings, repo, downloader=downloader)
    enqueue(repo, req, 0, "360")
    asyncio.run(worker.run_once())
    enqueue(repo, req, 0, "360")
    asyncio.run(worker.run_once())
    assert downloader.calls == 1
    assert bot.sent[1]["video"] == "telegram-cache"
    assert "https://www.youtube.com/watch?v=abcdefghijk" in bot.sent[1]["caption"]


def test_ambiguous_delivery_is_not_automatically_replayed(env):
    settings, repo = env
    req = ready(repo)
    bot = FakeBot(fail=True)
    worker = MediaWorker(bot, settings, repo, downloader=FakeDownloader(settings))
    job = enqueue(repo, req, 0, "360")
    asyncio.run(worker.run_once())
    worker.recover()
    assert asyncio.run(worker.run_once()) is False
    with repo.connection() as db:
        assert (
            db.execute(
                "SELECT status FROM media_download_jobs WHERE id=?", (job["id"],)
            ).fetchone()[0]
            == "uncertain"
        )
    assert len(bot.sent) == 1


def test_large_files_require_configured_local_bot_api(env):
    settings, _ = env
    assert Downloader(settings).max_bytes == 49_000_000
    assert (
        Downloader(
            settings.model_copy(update={"telegram_api_base_url": "http://telegram-bot-api:8081"})
        ).max_bytes
        == 1_900_000_000
    )


def test_plain_youtube_keeps_existing_summary_flow():
    msg = SimpleNamespace(
        text="https://youtu.be/abcdefghijk", caption=None, entities=None, caption_entities=None
    )
    assert asyncio.run(MediaFilter()(msg)) is False
    msg.text = "/download https://youtu.be/abcdefghijk"
    assert asyncio.run(MediaFilter()(msg)) is True


def test_cancelled_command_terminates_child(env):
    settings, _ = env
    import sys

    with pytest.raises(MediaError, match="cancelled"):
        asyncio.run(
            Downloader(settings).command(
                [sys.executable, "-c", "import time;time.sleep(20)"],
                timeout=30,
                cancelled=lambda: True,
            )
        )


def test_foreign_or_detached_callback_cannot_enqueue(env):
    from unittest.mock import AsyncMock

    from aiogram import Router

    from app.telegram_bot.media_downloads import register_media_handlers

    settings, repo = env
    req = ready(repo)
    router = Router()
    register_media_handlers(router, settings, repo)
    handler = router.callback_query.handlers[0].callback
    for user, message_id in [(999, 42), (123, 41)]:
        callback = SimpleNamespace(
            from_user=SimpleNamespace(id=user),
            message=SimpleNamespace(
                chat=SimpleNamespace(id=123, type="private"), message_id=message_id
            ),
            answer=AsyncMock(),
        )
        asyncio.run(
            handler(callback, MediaAction(request_id=req["id"], action="get", choice="360"))
        )
        assert callback.answer.call_args.kwargs["show_alert"] is True
    with repo.connection() as db:
        assert db.execute("SELECT count(*) FROM media_download_jobs").fetchone()[0] == 0
