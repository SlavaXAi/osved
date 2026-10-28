from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import signal
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


class MediaError(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def media_link(value: str) -> tuple[str, str, str]:
    p = urlsplit(value.strip())
    if p.scheme != "https" or p.username or p.password or p.port not in (None, 443):
        raise ValueError("Unsupported media URL")
    host = (p.hostname or "").lower()
    if host in {
        "music.yandex.ru",
        "music.yandex.com",
        "music.yandex.by",
        "music.yandex.kz",
        "music.yandex.uz",
    }:
        match = re.fullmatch(r"/(?:album/([1-9]\d{0,14})/)?track/([1-9]\d{0,14})/?", p.path)
        if match:
            album, track = match.groups()
            path = f"/album/{album}/track/{track}" if album else "/track/" + track
            return "yandex_music", track, "https://music.yandex.ru" + path
    if host in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}:
        parts = p.path.strip("/").split("/")
        ident = (
            parts[0]
            if host == "youtu.be"
            else (
                parse_qs(p.query).get("v", [""])[0]
                if p.path == "/watch"
                else parts[1]
                if len(parts) == 2 and parts[0] in {"shorts", "live"}
                else ""
            )
        )
        if re.fullmatch(r"[A-Za-z0-9_-]{11}", ident):
            return "youtube", ident, "https://www.youtube.com/watch?v=" + ident
    if host in {"x.com", "www.x.com", "twitter.com", "www.twitter.com", "mobile.twitter.com"}:
        m = re.fullmatch(
            r"/(?:[A-Za-z0-9_]+/status|i/(?:web/)?status)/(\d+)(?:/(?:video|photo)/\d+)?/?", p.path
        )
        if m:
            return "x", m[1], "https://x.com/i/status/" + m[1]
    raise ValueError("Unsupported media URL")


def message_links(message) -> list[tuple[str, str, str]]:
    text = message.text or message.caption or ""
    values = re.findall(r"https://[^\s<>]+", text)
    values += [
        e.url
        for e in (message.entities or message.caption_entities or [])
        if e.type == "text_link" and e.url
    ]
    found = []
    for value in values:
        try:
            link = media_link(value.rstrip(".,!?\")']>"))
        except ValueError:
            continue
        if link not in found:
            found.append(link)
    return found


def request(repository, *, chat_id, message_id, link, video_panel_id=None):
    now = time.time()
    with repository.connection() as db:
        old = db.execute(
            "SELECT manifest_json FROM media_requests WHERE chat_id=? AND platform=? "
            "AND external_id=? AND manifest_json IS NOT NULL AND updated_at>? "
            "ORDER BY id DESC LIMIT 1",
            (chat_id, link[0], link[1], now - 21600),
        ).fetchone()
        db.execute(
            "INSERT OR IGNORE INTO media_requests "
            "(chat_id,request_message_id,platform,external_id,source_url,video_panel_id,"
            "created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
            (chat_id, message_id, *link, video_panel_id, now, now),
        )
        if old:
            db.execute(
                "UPDATE media_requests SET manifest_json=?,status='ready' "
                "WHERE chat_id=? AND request_message_id=? AND manifest_json IS NULL",
                (old[0], chat_id, message_id),
            )
        if video_panel_id:
            db.execute(
                "UPDATE media_requests SET video_panel_id=? "
                "WHERE chat_id=? AND request_message_id=?",
                (video_panel_id, chat_id, message_id),
            )
        return dict(
            db.execute(
                "SELECT * FROM media_requests WHERE chat_id=? AND request_message_id=?",
                (chat_id, message_id),
            ).fetchone()
        )


def get_request(repository, ident):
    with repository.connection() as db:
        r = db.execute("SELECT * FROM media_requests WHERE id=?", (ident,)).fetchone()
    return dict(r) if r else None


def cache_key(req, index, choice):
    return hashlib.sha256(
        f"{req['platform']}:{req['external_id']}:{index}:{choice}".encode()
    ).hexdigest()


def enqueue(repository, req, index, choice, *, once=False):
    manifest = json.loads(req["manifest_json"] or "{}")
    items = manifest.get("items", [])
    if not 0 <= index < len(items) or choice not in items[index]["choices"]:
        raise MediaError("invalid_choice")
    with repository.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        # A previously delivered result is deliberately reusable by an explicit click.
        states = "" if once else "AND status IN ('queued','running','sending')"
        r = db.execute(
            "SELECT * FROM media_download_jobs WHERE request_id=? "
            "AND item_index=? AND choice=? " + states + " ORDER BY id DESC LIMIT 1",
            (req["id"], index, choice),
        ).fetchone()
        if r:
            return dict(r)
        active = db.execute(
            "SELECT count(*) FROM media_download_jobs WHERE status IN "
            "('queued','running','sending')"
        ).fetchone()[0]
        if active >= 5:
            raise MediaError("queue_full")
        now = time.time()
        ident = db.execute(
            "INSERT INTO media_download_jobs "
            "(request_id,item_index,choice,created_at,updated_at) VALUES(?,?,?,?,?)",
            (req["id"], index, choice, now, now),
        ).lastrowid
        return dict(db.execute("SELECT * FROM media_download_jobs WHERE id=?", (ident,)).fetchone())


def choices_for(info):
    formats = [f for f in info.get("formats", []) if isinstance(f, dict)]
    audio = [f for f in formats if f.get("acodec") not in (None, "none")]
    only_audio = [f for f in audio if f.get("vcodec") == "none"]
    best_audio = max(
        only_audio, key=lambda f: (f.get("ext") == "m4a", f.get("abr") or 0), default=None
    )
    choices = {}
    for f in formats:
        ident = str(f.get("format_id") or "")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", ident):
            continue
        h = int(f.get("height") or 0)
        if not h or f.get("vcodec") in (None, "none"):
            continue
        if f.get("acodec") == "none" and not best_audio and info.get("platform") != "x":
            continue
        selector = ident
        size = f.get("filesize") or f.get("filesize_approx") or 0
        if f.get("acodec") == "none" and best_audio:
            selector += "+" + str(best_audio["format_id"])
            size += best_audio.get("filesize") or best_audio.get("filesize_approx") or 0
        key = str(h)
        rank = (
            str(f.get("vcodec", "")).startswith(("avc", "h264")),
            f.get("ext") == "mp4",
            f.get("tbr") or 0,
        )
        old = choices.get(key)
        if not old or rank > tuple(old["_rank"]):
            choices[key] = {
                "label": f"{h}p",
                "selector": selector,
                "size": int(size),
                "_rank": list(rank),
            }
    if audio:
        choices["mp3"] = {
            "label": "MP3 · звук",
            "selector": "bestaudio/best",
            "size": int((info.get("duration") or 0) * 16000),
        }
    for choice in choices.values():
        choice.pop("_rank", None)
    return choices


def normalize_info(raw, platform):
    entries = raw.get("entries") if raw.get("_type") in {"playlist", "multi_video"} else [raw]
    if not isinstance(entries, list) or len(entries) > 10:
        raise MediaError("too_many_items")
    items = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        entry = {**entry, "platform": platform}
        choices = choices_for(entry)
        if choices:
            items.append(
                {
                    "title": str(entry.get("title") or raw.get("title") or "Видео")[:300],
                    "duration": int(entry.get("duration") or 0),
                    "choices": choices,
                    "playlist_index": i + 1,
                    "multiple": len(entries) > 1,
                }
            )
    if not items:
        raise MediaError("no_video")
    return {"items": items, "title": str(raw.get("title") or items[0]["title"])[:300]}


class Downloader:
    def __init__(self, settings):
        self.settings = settings
        self.root = settings.data_dir / "media-downloads"
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def max_bytes(self):
        cap = 1_900_000_000 if self.settings.telegram_api_base_url else 49_000_000
        return min(self.settings.media_max_file_bytes, cap)

    def directory(self, job_id):
        return self.root / f"job-{int(job_id)}"

    def check_space(self, directory=None):
        used = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        if used > self.settings.media_max_storage_bytes:
            raise MediaError("disk_full")
        if shutil.disk_usage(self.root).free < self.settings.media_min_free_bytes:
            raise MediaError("disk_full")
        if (
            directory
            and sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
            > self.max_bytes * 2 + 32_000_000
        ):
            raise MediaError("too_large")

    async def command(self, args, *, timeout, cancelled=lambda: False, directory=None):
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )

        async def read(stream, limit):
            data = bytearray()
            while part := await stream.read(65536):
                data.extend(part)
                if len(data) > limit:
                    raise MediaError("invalid_response")
            return bytes(data)

        out = asyncio.create_task(read(proc.stdout, 8_000_000))
        err = asyncio.create_task(read(proc.stderr, 500_000))
        start = time.monotonic()
        try:
            while proc.returncode is None:
                if cancelled():
                    raise MediaError("cancelled")
                if time.monotonic() - start > timeout:
                    raise MediaError("timeout")
                if directory:
                    self.check_space(directory)
                for reader in (out, err):
                    if reader.done() and reader.exception():
                        raise reader.exception()
                await asyncio.sleep(0.25)
            stdout, stderr = await asyncio.gather(out, err)
            if proc.returncode:
                detail = stderr.decode(errors="replace").lower()
                if "not a bot" in detail or "confirm you’re not" in detail:
                    raise MediaError("youtube_blocked")
                if any(s in detail for s in ["sign in", "login", "private video", "not a bot"]):
                    raise MediaError("login_required")
                if "429" in detail:
                    raise MediaError("rate_limited")
                raise MediaError("unavailable")
            return stdout
        finally:
            if proc.returncode is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(proc.wait(), 3)
                except TimeoutError:
                    os.killpg(proc.pid, signal.SIGKILL)
                    await proc.wait()
            for reader in (out, err):
                if not reader.done():
                    reader.cancel()
            await asyncio.gather(out, err, return_exceptions=True)

    def base_args(self):
        return [
            sys.executable,
            "-m",
            "yt_dlp",
            "--ignore-config",
            "--js-runtimes",
            "node",
            "--extractor-args",
            "youtube:player_client=android_vr",
            "--no-warnings",
            "--socket-timeout",
            "20",
            "--retries",
            "0",
            "--fragment-retries",
            "0",
            "--playlist-end",
            "10",
            "--no-check-formats",
        ]

    async def inspect(self, req, *, cancelled=lambda: False):
        if req["platform"] == "yandex_music":
            from app.media.yandex_music import describe

            return await asyncio.to_thread(describe, self.settings, req)
        try:
            output = await self.command(
                self.base_args() + ["--dump-single-json", "--skip-download", req["source_url"]],
                timeout=90,
                cancelled=cancelled,
            )
        except MediaError as exc:
            if self.use_youtube_fallback(req, exc):
                from app.media.youtube_monid import manifest, source

                return manifest(await source(self, req, cancelled=cancelled))
            raise
        try:
            raw = json.loads(output)
        except ValueError as exc:
            raise MediaError("invalid_response") from exc
        if not isinstance(raw, dict):
            raise MediaError("invalid_response")
        return normalize_info(raw, req["platform"])

    def use_youtube_fallback(self, req, error):
        return (
            req["platform"] == "youtube"
            and self.settings.media_youtube_monid_enabled
            and error.code in {"youtube_blocked", "rate_limited", "unavailable", "timeout"}
        )

    async def download(self, req, job, *, cancelled=lambda: False):
        if req["platform"] == "yandex_music":
            from app.media.yandex_music import download

            return await download(self, req, job, cancelled=cancelled)
        item = json.loads(req["manifest_json"])["items"][job["item_index"]]
        choice = item["choices"][job["choice"]]
        if choice.get("size", 0) > self.max_bytes:
            raise MediaError("too_large")
        self.check_space()
        directory = self.directory(job["id"])
        directory.mkdir(exist_ok=True)
        # A successfully prepared file is reused after restart; never re-download it.
        marker = directory / "ready.json"
        if marker.is_file():
            result = json.loads(marker.read_text())
            path = Path(result["path"])
            if (
                path.parent == directory
                and path.is_file()
                and path.stat().st_size <= self.max_bytes
            ):
                return result
        if (
            req["platform"] == "youtube"
            and json.loads(req["manifest_json"]).get("backend") == "monid_youtube"
        ):
            from app.media.youtube_monid import download

            return await download(self, req, job, cancelled=cancelled)
        args = self.base_args() + [
            "--no-progress",
            "--max-filesize",
            str(self.max_bytes),
            "--format",
            choice["selector"],
            "--output",
            str(directory / "source.%(ext)s"),
        ]
        if item["multiple"]:
            args += ["--playlist-items", str(item["playlist_index"])]
        else:
            args += ["--no-playlist"]
        if job["choice"] == "mp3":
            args += ["--extract-audio", "--audio-format", "mp3", "--audio-quality", "128K"]
        else:
            args += ["--merge-output-format", "mkv"]
        try:
            await self.command(
                args + [req["source_url"]],
                timeout=self.settings.media_timeout_seconds,
                cancelled=cancelled,
                directory=directory,
            )
        except MediaError as exc:
            if self.use_youtube_fallback(req, exc):
                from app.media.youtube_monid import download

                return await download(self, req, job, cancelled=cancelled)
            raise
        paths = [
            p
            for p in directory.glob("source.*")
            if p.suffix.lower() in {".mp4", ".webm", ".mkv", ".mov", ".mp3", ".m4a"}
        ]
        if len(paths) != 1:
            raise MediaError("too_large" if not paths else "invalid_response")
        source = paths[0]
        probe = json.loads(
            await self.command(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_streams",
                    "-show_format",
                    "-of",
                    "json",
                    str(source),
                ],
                timeout=30,
                cancelled=cancelled,
            )
        )
        streams = probe.get("streams", [])
        video = next((s for s in streams if s.get("codec_type") == "video"), {})
        audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
        if job["choice"] != "mp3":
            target = directory / "video.mp4"
            codec = (
                ["-c:v", "copy"]
                if video.get("codec_name") == "h264"
                else [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "23",
                    "-threads",
                    "1",
                    "-pix_fmt",
                    "yuv420p",
                ]
            )
            codec += (
                ["-c:a", "copy"] if audio.get("codec_name") in (None, "aac") else ["-c:a", "aac"]
            )
            await self.command(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(source),
                    *codec,
                    "-movflags",
                    "+faststart",
                    str(target),
                ],
                timeout=self.settings.media_timeout_seconds,
                cancelled=cancelled,
                directory=directory,
            )
            source.unlink()
            source = target
        size = source.stat().st_size
        if not 0 < size <= self.max_bytes:
            raise MediaError("too_large")
        result = {
            "path": str(source),
            "kind": "audio" if job["choice"] == "mp3" else "video",
            "size": size,
            "title": item["title"],
            "width": video.get("width", 0),
            "height": video.get("height", 0),
            "duration": int(float(probe.get("format", {}).get("duration") or 0)),
        }
        marker.write_text(json.dumps(result))
        return result

    def remove_job_files(self, job_id):
        shutil.rmtree(self.directory(job_id), ignore_errors=True)

    def cleanup(self, active_ids):
        cutoff = time.time() - 24 * 3600
        for path in self.root.glob("job-*"):
            if (
                path.is_dir()
                and path.name[4:].isdigit()
                and int(path.name[4:]) not in active_ids
                and path.stat().st_mtime < cutoff
            ):
                shutil.rmtree(path)
        for path in (self.root / "youtube").glob("*"):
            if path.is_dir() and path.stat().st_mtime < cutoff:
                shutil.rmtree(path)
