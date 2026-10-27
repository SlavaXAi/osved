import asyncio
import json
import shutil
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from test_media_downloads import env as env  # noqa: F401

from app.media import yandex_music as music
from app.media.downloads import Downloader, MediaError, enqueue, get_request, media_link, request
from app.telegram_bot.media_downloads import MediaFilter, MediaWorker

URL = "https://music.yandex.com/album/123/track/456?utm_source=web"
MANIFEST = {
    "title": "Artist — Song",
    "items": [
        {
            "title": "Song",
            "performer": "Artist",
            "album": "Album",
            "duration": 2,
            "choices": {"mp3": {"label": "MP3", "size": 0, "selector": "mp3"}},
        }
    ],
}


def ready_music(repo, message=10):
    row = request(repo, chat_id=123, message_id=message, link=media_link(URL))
    with repo.connection() as db:
        db.execute(
            "UPDATE media_requests SET status='ready',manifest_json=?,panel_message_id=42 "
            "WHERE id=?",
            (json.dumps(MANIFEST), row["id"]),
        )
    return get_request(repo, row["id"])


@pytest.mark.parametrize(
    "url",
    [URL, "https://music.yandex.ru/track/456", "https://music.yandex.kz/album/789/track/456/"],
)
def test_one_track_urls_have_one_identity(url):
    assert media_link(url)[:2] == ("yandex_music", "456")


@pytest.mark.parametrize(
    "url",
    [
        "https://music.yandex.ru/album/123",
        "https://music.yandex.ru/playlists/123",
        "https://music.yandex.ru.evil.test/track/456",
        "https://music.yandex.ru/track/0",
        "https://music.yandex.ru/album/1/track/2/video/3",
    ],
)
def test_albums_playlists_and_unrelated_hosts_are_rejected(url):
    with pytest.raises(ValueError):
        media_link(url)


def test_music_metadata_is_bound_to_requested_track(env):
    s, _ = env
    track = SimpleNamespace(
        id=456,
        title="Song",
        artists=[SimpleNamespace(name="Artist")],
        albums=[SimpleNamespace(title="Album")],
        duration_ms=2000,
    )
    api = SimpleNamespace(tracks=Mock(return_value=[track]))
    req = {"external_id": "456"}
    assert music.describe(s, req, client_factory=lambda _: api) == MANIFEST
    track.id = 999
    with pytest.raises(MediaError):
        music.describe(s, req, client_factory=lambda _: api)
    with pytest.raises(MediaError, match="music_login_required"):
        music.token(s)


@pytest.mark.parametrize(
    "url",
    [
        "https://localhost/a",
        "https://storage.yandex.net.evil.test/a",
        "https://user:pass@storage.yandex.net/a",
        "https://storage.yandex.net:8080/a",
    ],
)
def test_music_media_never_fetches_arbitrary_urls(url):
    with pytest.raises(MediaError):
        music.media_url(url)


def test_preview_is_not_a_full_song():
    with pytest.raises(MediaError, match="music_preview_only"):
        music.check_full_duration(30, 192)
    music.check_full_duration(191.9, 192)


def test_yandex_streaming_redirect_is_allowed_but_unrelated_hosts_are_not():
    url = "https://strm-rad-24.strm.yandex.net/audio.mp3?ticket=example"
    assert music.media_url(url) == url
    with pytest.raises(MediaError):
        music.media_url("https://strm.yandex.net.evil.test/audio.mp3")


def test_real_mp3_contains_cover_and_tags_and_reuses_completed_file(env, monkeypatch):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is required for the media acceptance test")
    settings, repo = env
    row = ready_music(repo)
    job = enqueue(repo, row, 0, "mp3", once=True)
    downloader = Downloader(settings)
    acquired = []

    async def prepare():
        root = settings.data_dir
        audio, cover = root / "fixture.mp3", root / "fixture.jpg"
        await downloader.command(
            [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=2",
                "-c:a",
                "libmp3lame",
                str(audio),
            ],
            timeout=15,
        )
        await downloader.command(
            [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=purple:s=600x600",
                "-frames:v",
                "1",
                "-update",
                "1",
                str(cover),
            ],
            timeout=15,
        )

        def acquire(_s, _r, folder, _cap, **kwargs):
            acquired.append(True)
            shutil.copyfile(audio, folder / "source.mp3")
            shutil.copyfile(cover, folder / "cover-source")

        monkeypatch.setattr(music, "acquire", acquire)
        result = await downloader.download(row, job)
        assert (await downloader.download(row, job)) == result and len(acquired) == 1
        probe = json.loads(
            await downloader.command(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_streams",
                    "-show_format",
                    "-of",
                    "json",
                    result["path"],
                ],
                timeout=15,
            )
        )
        assert probe["format"]["tags"]["title"] == "Song"
        assert probe["format"]["tags"]["artist"] == "Artist"
        assert any(x.get("disposition", {}).get("attached_pic") == 1 for x in probe["streams"])
        assert not any("http" in str(v) for v in probe["format"]["tags"].values())

    asyncio.run(prepare())


def test_worker_sends_clean_audio_and_uses_telegram_cache(env):
    settings, repo = env
    row = ready_music(repo)
    first = enqueue(repo, row, 0, "mp3", once=True)
    assert enqueue(repo, row, 0, "mp3", once=True)["id"] == first["id"]
    downloader = SimpleNamespace(
        max_bytes=49_000_000,
        download=AsyncMock(
            return_value={
                "path": "/tmp/song.mp3",
                "thumbnail_path": "/tmp/cover.jpg",
                "kind": "audio",
                "title": "Song",
                "performer": "Artist",
                "duration": 2,
                "size": 1000,
            }
        ),
        remove_job_files=Mock(),
    )
    bot = SimpleNamespace(
        id=12,
        edit_message_text=AsyncMock(),
        delete_message=AsyncMock(),
        send_audio=AsyncMock(
            return_value=SimpleNamespace(message_id=70, audio=SimpleNamespace(file_id="saved"))
        ),
    )
    worker = MediaWorker(bot, settings, repo, downloader=downloader)
    assert asyncio.run(worker.run_once())
    sent = bot.send_audio.call_args.kwargs
    assert sent["title"] == "Song" and sent["performer"] == "Artist" and "thumbnail" in sent
    assert "caption" not in sent and "reply_markup" not in sent
    # A new user message may request the same song; its file is reused without fetching audio.
    second = request(repo, chat_id=123, message_id=11, link=media_link(URL))
    enqueue(repo, second, 0, "mp3", once=True)
    assert asyncio.run(worker.run_once())
    assert downloader.download.call_count == 1
    assert bot.send_audio.call_args.kwargs["audio"] == "saved"
    assert "thumbnail" not in bot.send_audio.call_args.kwargs
    with repo.connection() as db:
        metadata = json.loads(
            db.execute("SELECT metadata_json FROM media_file_cache").fetchone()[0]
        )
        assert "path" not in metadata and "thumbnail_path" not in metadata


def test_restart_recovers_ready_music_without_replaying_failed_job(env):
    settings, repo = env
    ready_music(repo)
    worker = MediaWorker(SimpleNamespace(), settings, repo)
    worker.recover()
    with repo.connection() as db:
        assert db.execute("SELECT count(*) FROM media_download_jobs").fetchone()[0] == 1
        db.execute("UPDATE media_download_jobs SET status='failed'")
    worker.recover()
    with repo.connection() as db:
        assert db.execute("SELECT status FROM media_download_jobs").fetchone()[0] == "failed"
    msg = SimpleNamespace(text=URL, caption=None, entities=None, caption_entities=None)
    assert asyncio.run(MediaFilter()(msg)) is True
