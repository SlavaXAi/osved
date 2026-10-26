"""Bounded downloads from X's media hosts, and one forwardable rich publication."""

from __future__ import annotations

import hashlib
import html
import json
import shutil
import time
from pathlib import Path
from urllib.request import HTTPRedirectHandler, Request, build_opener

from aiogram.types import (
    FSInputFile,
    InputMediaPhoto,
    InputMediaVideo,
    InputRichMessage,
    InputRichMessageMedia,
)

from app.reposts.provider import safe_media_url


class SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not safe_media_url(newurl):
            raise ValueError("Media redirected outside X")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_file(url, path, *, cap, opener=None, timeout=180):
    if not safe_media_url(url):
        raise ValueError("Unsupported media host")
    if path.exists() and 0 < path.stat().st_size <= cap:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    deadline = time.monotonic() + timeout
    try:
        with (opener or build_opener(SafeRedirect())).open(
            Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=45
        ) as response:
            if int(response.headers.get("Content-Length") or 0) > cap:
                raise ValueError("too_large")
            total = 0
            with partial.open("wb") as output:
                while chunk := response.read(256 * 1024):
                    if time.monotonic() > deadline:
                        raise TimeoutError("Media download exceeded its time limit")
                    total += len(chunk)
                    if total > cap:
                        raise ValueError("too_large")
                    output.write(chunk)
            if total == 0:
                raise ValueError("empty_media")
        partial.replace(path)
        path.chmod(0o600)
        return path
    finally:
        partial.unlink(missing_ok=True)


def acquire(settings, row, post):
    previous = json.loads(row.get("media_json") or "[]")
    if previous and all(m.get("file_id") or Path(m.get("path", "")).is_file() for m in previous):
        return previous
    root = settings.data_dir / "x-reposts" / "media" / str(row["id"])
    root.mkdir(parents=True, exist_ok=True)
    used = sum(p.stat().st_size for p in root.parent.rglob("*") if p.is_file())
    required = len(post["media"]) * min(settings.media_max_file_bytes, 49_000_000)
    if used + required > settings.media_max_storage_bytes:
        raise ValueError("Временное хранилище медиа заполнено")
    if shutil.disk_usage(root).free < settings.media_min_free_bytes + 490_000_000:
        raise ValueError("Недостаточно места для скачивания медиа")
    files = []
    cap = min(settings.media_max_file_bytes, 49_000_000)
    for index, item in enumerate(post["media"]):
        candidates = (
            [item["url"]] if item["kind"] == "photo" else [v["url"] for v in item["variants"]]
        )
        error = None
        for url in candidates:
            name = hashlib.sha256(url.encode()).hexdigest()[:24]
            path = root / (name + (".jpg" if item["kind"] == "photo" else ".mp4"))
            try:
                download_file(
                    url, path, cap=min(cap, 10_000_000) if item["kind"] == "photo" else cap
                )
                files.append(
                    {
                        "kind": item["kind"],
                        "path": str(path.resolve()),
                        "index": index,
                        "size": path.stat().st_size,
                    }
                )
                break
            except ValueError as exc:
                error = exc
        else:
            if error and str(error) == "too_large":
                raise ValueError(
                    "Цельный файл больше 49 МБ даже в доступном меньшем качестве"
                ) from error
            raise error or ValueError("Не удалось скачать медиа")
    return files


def body_html(post, translation, source_url):
    paragraphs = [html.escape(p) for p in translation["paragraphs"]]
    if not paragraphs:
        paragraphs = ["Публикация без текста"]
    author = ("@" + post["author"]) if post.get("author") else "X"
    return (
        "\n\n".join(paragraphs)
        + f'\n\n<a href="{html.escape(source_url, quote=True)}">'
        + f"{html.escape(author)} · Оригинал</a>"
    )


def rich_content(text, files):
    # HTML in text is generated/escaped by our renderers, never provider HTML.
    paragraphs = "".join("<p>" + p.replace("\n", "<br>") + "</p>" for p in text.split("\n\n") if p)
    attachments = []
    embeds = []
    for index, item in enumerate(files):
        ident = "media_" + str(index)
        value = item.get("file_id") or FSInputFile(item["path"])
        if item["kind"] == "video":
            media = InputMediaVideo(media=value, supports_streaming=True)
            embeds.append(f'<video src="tg://video?id={ident}"></video>')
        else:
            media = InputMediaPhoto(media=value)
            embeds.append(f'<img src="tg://photo?id={ident}">')
        attachments.append(InputRichMessageMedia(id=ident, media=media))
    return InputRichMessage(
        html=paragraphs + "".join(embeds), media=attachments or None, skip_entity_detection=True
    )


def remember_file_ids(message, files):
    rich = getattr(message, "rich_message", None)
    if not rich:
        return files
    media = []

    def walk(blocks):
        for block in blocks:
            kind = str(block.type)
            if kind == "video":
                media.append(("video", block.video.file_id))
            elif kind == "photo":
                media.append(("photo", max(block.photo, key=lambda p: p.width * p.height).file_id))
            elif hasattr(block, "blocks"):
                walk(block.blocks)

    walk(rich.blocks)
    if len(media) == len(files) and all(
        kind == f["kind"] for (kind, _), f in zip(media, files, strict=True)
    ):
        return [{**f, "file_id": ident} for (_, ident), f in zip(media, files, strict=True)]
    return files


def cleanup(settings, repository):
    """Retain only undelivered local files; successful Telegram file IDs are durable."""
    with repository.connection() as db:
        rows = db.execute("SELECT id,media_json FROM x_reposts WHERE status='ready'").fetchall()
    for row in rows:
        files = json.loads(row["media_json"] or "[]")
        if all(f.get("file_id") for f in files):
            shutil.rmtree(
                settings.data_dir / "x-reposts" / "media" / str(row["id"]), ignore_errors=True
            )
    with repository.connection() as db:
        stale = db.execute(
            "SELECT id FROM x_reposts WHERE status NOT IN ('running','queued') "
            "AND datetime(updated_at)<datetime('now','-1 day')"
        ).fetchall()
    for row in stale:
        shutil.rmtree(
            settings.data_dir / "x-reposts" / "media" / str(row["id"]), ignore_errors=True
        )
