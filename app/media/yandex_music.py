"""One requested track, using the owner's music access, with embedded cover art."""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from app.media.downloads import MediaError


def token(settings):
    if settings.yandex_music_token:
        return settings.yandex_music_token.get_secret_value()
    path = settings.data_dir / "yandex-music-token.json"
    if path.is_file():
        value = json.loads(path.read_text())
        result = value.get("access_token")
        if isinstance(result, str) and result.strip():
            return result.strip()
    raise MediaError("music_login_required")


def client(settings):
    from yandex_music import Client
    from yandex_music.exceptions import UnauthorizedError
    from yandex_music.utils.request import Request as MusicRequest

    try:
        return Client(token(settings), request=MusicRequest(timeout=25)).init()
    except UnauthorizedError:
        raise MediaError("music_login_required") from None


def _track(api, ident):
    tracks = api.tracks(ident)
    if len(tracks) != 1 or str(tracks[0].id) != ident:
        raise MediaError("unavailable")
    return tracks[0]


def describe(settings, req, *, client_factory=client):
    track = _track(client_factory(settings), req["external_id"])
    title = str(track.title or "").strip()[:300]
    artists = ", ".join(a.name for a in track.artists if a.name)[:300]
    duration = float(track.duration_ms or 0) / 1000
    if not title or not artists or not 0 < duration <= 7200:
        raise MediaError("invalid_response")
    return {
        "title": f"{artists} — {title}",
        "items": [
            {
                "title": title,
                "performer": artists,
                "duration": duration,
                "album": str(track.albums[0].title or "")[:300] if track.albums else "",
                "choices": {"mp3": {"label": "MP3", "selector": "mp3", "size": 0}},
            }
        ],
    }


def media_url(value):
    """Only vendor media hosts; never forward an OAuth header to a media host."""
    try:
        p = urlsplit(str(value))
        host = (p.hostname or "").lower()
        if (
            p.scheme not in {"https", "http"}
            or p.username
            or p.password
            or p.port not in {None, 443, 80}
        ):
            raise ValueError()
        allowed = any(
            host == d or host.endswith("." + d)
            for d in (
                "storage.yandex.net",
                "storage.mds.yandex.net",
            "strm.yandex.net",
                "music.yandex.net",
                "avatars.yandex.net",
                "avatars.mds.yandex.net",
                "music.yandex.ru",
            )
        )
        if not allowed or any(ord(c) < 32 for c in str(value)):
            raise ValueError()
        return urlunsplit(("https", host, p.path, p.query, ""))
    except ValueError:
        raise MediaError("invalid_response") from None


class MediaRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return super().redirect_request(req, fp, code, msg, headers, media_url(newurl))


def fetch_file(url, path, *, limit, cancelled=lambda: False, opener=None):
    partial = path.with_suffix(path.suffix + ".part")
    deadline = time.monotonic() + 180
    try:
        with (opener or build_opener(MediaRedirect())).open(
            Request(media_url(url), headers={"User-Agent": "OSVED music"}), timeout=25
        ) as response:
            if int(response.headers.get("Content-Length") or 0) > limit:
                raise MediaError("too_large")
            size = 0
            with partial.open("wb") as output:
                while chunk := response.read(256 * 1024):
                    if cancelled():
                        raise MediaError("cancelled")
                    if time.monotonic() > deadline:
                        raise MediaError("timeout")
                    size += len(chunk)
                    if size > limit:
                        raise MediaError("too_large")
                    output.write(chunk)
            if not size:
                raise MediaError("invalid_response")
        partial.replace(path)
        path.chmod(0o600)
    finally:
        partial.unlink(missing_ok=True)


def acquire(settings, req, directory, cap, *, cancelled=lambda: False, client_factory=client):
    api = client_factory(settings)
    track = _track(api, req["external_id"])
    variants = [
        v
        for v in api.tracks_download_info(req["external_id"])
        if v.codec == "mp3" and not getattr(v, "preview", False)
    ]
    if not variants:
        raise MediaError("music_full_track_unavailable")
    chosen = max(variants, key=lambda v: v.bitrate_in_kbps or 0)
    # Validate the SDK's intermediate URL before it is fetched too.
    chosen.download_info_url = media_url(chosen.download_info_url)
    audio_url = media_url(chosen.get_direct_link(timeout=25))
    if not track.cover_uri:
        raise MediaError("music_cover_unavailable")
    cover_url = media_url(track.get_cover_url("600x600"))
    fetch_file(audio_url, directory / "source.mp3", limit=cap, cancelled=cancelled)
    fetch_file(cover_url, directory / "cover-source", limit=5_000_000, cancelled=cancelled)


def check_full_duration(actual, expected):
    if expected <= 0 or actual < expected - max(2, expected * 0.02):
        raise MediaError("music_preview_only")
    if abs(actual - expected) > max(3, expected * 0.03):
        raise MediaError("invalid_response")


async def download(downloader, req, job, *, cancelled=lambda: False):
    directory = downloader.directory(job["id"])
    directory.mkdir(exist_ok=True)
    marker = directory / "ready.json"
    if marker.is_file():
        saved = json.loads(marker.read_text())
        path = Path(saved["path"])
        thumbnail = Path(saved.get("thumbnail_path") or "")
        if (
            path.parent == directory
            and path.is_file()
            and path.stat().st_size <= downloader.max_bytes
            and thumbnail.is_file()
        ):
            return saved
    downloader.check_space()
    item = json.loads(req["manifest_json"])["items"][0]
    await asyncio.to_thread(
        acquire, downloader.settings, req, directory, downloader.max_bytes, cancelled=cancelled
    )
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
                str(directory / "source.mp3"),
            ],
            timeout=30,
            cancelled=cancelled,
        )
    )
    actual = float(probe.get("format", {}).get("duration") or 0)
    if not any(s.get("codec_name") == "mp3" for s in probe.get("streams", [])):
        raise MediaError("invalid_response")
    check_full_duration(actual, item["duration"])
    for name, size in [("cover.jpg", 600), ("thumbnail.jpg", 320)]:
        await downloader.command(
            [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-v",
                "error",
                "-i",
                str(directory / "cover-source"),
                "-vf",
                f"scale={size}:{size}:force_original_aspect_ratio=decrease",
                "-frames:v",
                "1",
                "-q:v",
                "4",
                "-update",
                "1",
                str(directory / name),
            ],
            timeout=30,
            cancelled=cancelled,
            directory=directory,
        )
    if (directory / "thumbnail.jpg").stat().st_size >= 200_000:
        raise MediaError("music_cover_unavailable")
    name = re.sub(r'[\x00-\x1f/\\:*?"<>|]', "", f"{item['performer']} — {item['title']}")[:170]
    target = directory / (name.strip(" .") + ".mp3")
    await downloader.command(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-i",
            str(directory / "source.mp3"),
            "-i",
            str(directory / "cover.jpg"),
            "-map",
            "0:a:0",
            "-map",
            "1:v:0",
            "-map_metadata",
            "-1",
            "-c:a",
            "copy",
            "-c:v",
            "mjpeg",
            "-disposition:v",
            "attached_pic",
            "-id3v2_version",
            "3",
            "-metadata",
            "title=" + item["title"],
            "-metadata",
            "artist=" + item["performer"],
            "-metadata",
            "album=" + item["album"],
            "-metadata:s:v",
            "title=Album cover",
            "-metadata:s:v",
            "comment=Cover (front)",
            str(target),
        ],
        timeout=60,
        cancelled=cancelled,
        directory=directory,
    )
    if not 0 < target.stat().st_size <= downloader.max_bytes:
        raise MediaError("too_large")
    result = {
        "path": str(target),
        "thumbnail_path": str(directory / "thumbnail.jpg"),
        "kind": "audio",
        "size": target.stat().st_size,
        "title": item["title"],
        "performer": item["performer"],
        "duration": round(actual),
    }
    marker.write_text(json.dumps(result))
    marker.chmod(0o600)
    return result
