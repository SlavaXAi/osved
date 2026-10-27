"""Ordinary Telegram posts; exact edit targets and durable companion sends."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from aiogram.exceptions import TelegramBadRequest
from aiogram.types import FSInputFile, InputMediaPhoto, InputMediaVideo, LinkPreviewOptions

from app.details.paging import (
    paginate_native_html,
    telegram_html_visible_text,
    telegram_html_visible_units,
    utf16_units,
)
from app.reposts import media, store


class UnconfirmedPublication(RuntimeError):
    pass


def layout(body, files):
    if files and telegram_html_visible_units(body) <= 1024:
        return files[0]["kind"], body, [], list(enumerate(files))[1:]
    pages = paginate_native_html(body, page_limit=4096)
    return "text", pages[0], pages[1:], list(enumerate(files))


def input_media(item, caption=None):
    path = Path(item.get("path") or "")
    value = (
        item.get("file_id")
        if item.get("native_file_id") or not path.is_file()
        else FSInputFile(path)
    )
    if value is None:
        raise ValueError("Media has neither a local file nor a confirmed Telegram ID")
    if item["kind"] == "video":
        if not item.get("width") or not item.get("height"):
            raise ValueError("Video dimensions must be read before delivery")
        thumbnail = Path(item.get("thumbnail_path") or "")
        return InputMediaVideo(
            media=value,
            caption=caption,
            parse_mode="HTML",
            width=item["width"],
            height=item["height"],
            duration=item["duration"],
            supports_streaming=True,
            thumbnail=FSInputFile(thumbnail)
            if isinstance(value, FSInputFile) and thumbnail.is_file()
            else None,
        )
    return InputMediaPhoto(media=value, caption=caption, parse_mode="HTML")


def refresh_baseline(repository, panel, body):
    if panel["baseline_html"] == body:
        return panel
    from app.telegram_bot.news_panel import get_news_panel

    reader = dict(panel["reader_context"])
    reader.update(
        baseline_html=body,
        baseline_text=telegram_html_visible_text(body),
        baseline_units=utf16_units(telegram_html_visible_text(body)),
    )
    # This repairs a translation/layout of the same saved original post, not its
    # evidence. Preserve the identity of already-paid analyses of that original.
    if not reader.get("context_fingerprint"):
        reader["context_fingerprint"] = hashlib.sha256(
            json.dumps(reader, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    with repository.connection() as db:
        db.execute(
            "UPDATE news_panels SET baseline_html=?,reader_context_json=?,view='base',"
            "previous_view='base',page=0,previous_page=0,displayed_request_id=NULL,"
            "active_request_id=NULL,error_text=NULL,revision=revision+1,updated_at=? "
            "WHERE delivery_id=?",
            (body, json.dumps(reader, ensure_ascii=False), store.stamp(), panel["delivery_id"]),
        )
    return get_news_panel(repository, panel["delivery_id"])


async def once(repository, row, key, send):
    with repository.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        old = db.execute(
            "SELECT * FROM x_repost_outbox WHERE repost_id=? AND part_key=?", (row["id"], key)
        ).fetchone()
        if old:
            if old["status"] != "sent":
                raise UnconfirmedPublication("Previous companion send is unconfirmed")
            return json.loads(old["result_json"])
        db.execute(
            "INSERT INTO x_repost_outbox(repost_id,part_key,status,updated_at) "
            "VALUES(?,?,'preparing',?)",
            (row["id"], key, store.stamp()),
        )
    try:
        result = await send()
    except Exception as exc:
        with repository.connection() as db:
            db.execute(
                "UPDATE x_repost_outbox SET status='uncertain',updated_at=? "
                "WHERE repost_id=? AND part_key=?",
                (store.stamp(), row["id"], key),
            )
        raise UnconfirmedPublication(
            "Companion send is unconfirmed; automatic replay disabled"
        ) from exc
    with repository.connection() as db:
        db.execute(
            "UPDATE x_repost_outbox SET status='sent',telegram_message_id=?,result_json=?,"
            "updated_at=? WHERE repost_id=? AND part_key=?",
            (result["message_id"], json.dumps(result), store.stamp(), row["id"], key),
        )
    return result


async def publish(bot, repository, row, panel, files):
    from app.telegram_bot.news_panel import panel_keyboard

    body = panel["baseline_html"]
    kind, primary, text_parts, extra_media = layout(body, files)
    markup = panel_keyboard(repository, panel)
    try:
        if kind == "text":
            result = await bot.edit_message_text(
                chat_id=row["chat_id"],
                message_id=row["panel_message_id"],
                text=primary,
                parse_mode="HTML",
                reply_markup=markup,
                link_preview_options=LinkPreviewOptions(is_disabled=True),
            )
        else:
            result = await bot.edit_message_media(
                chat_id=row["chat_id"],
                message_id=row["panel_message_id"],
                media=input_media(files[0], primary),
                reply_markup=markup,
                request_timeout=120,
            )
            files[0] = media.remember_file_ids(result, [files[0]])[0]
    except TelegramBadRequest as exc:
        if "message is not modified" not in str(exc).lower():
            raise
    # Persist the actual primary type immediately: reader edits must use captions
    # for native media, and text for a separate full translation.
    with repository.connection() as db:
        db.execute(
            "UPDATE x_reposts SET native_kind=?,media_json=? WHERE id=?",
            (kind, json.dumps(files), row["id"]),
        )
    for index, part in enumerate(text_parts, 1):

        async def send_text(part=part):
            msg = await bot.send_message(
                row["chat_id"],
                part,
                parse_mode="HTML",
                link_preview_options=LinkPreviewOptions(is_disabled=True),
            )
            return {"message_id": msg.message_id}

        await once(repository, row, "text:" + str(index), send_text)
    for index, item in extra_media:

        async def send_media(item=item):
            obj = input_media(item)
            if item["kind"] == "video":
                msg = await bot.send_video(
                    row["chat_id"],
                    video=obj.media,
                    width=obj.width,
                    height=obj.height,
                    duration=obj.duration,
                    supports_streaming=True,
                    thumbnail=obj.thumbnail,
                    request_timeout=120,
                )
            else:
                msg = await bot.send_photo(row["chat_id"], photo=obj.media, request_timeout=120)
            saved = media.remember_file_ids(msg, [item])[0]
            return {"message_id": msg.message_id, "media": saved}

        result = await once(repository, row, "media:" + str(index), send_media)
        files[index] = result["media"]
        with repository.connection() as db:
            db.execute(
                "UPDATE x_reposts SET media_json=? WHERE id=?", (json.dumps(files), row["id"])
            )
    return files
