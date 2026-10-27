"""Explicit, owner-only downloads. No sensor or scheduled feed calls this module."""

from __future__ import annotations

import asyncio
import html
import json
import logging
import time

from aiogram.dispatcher.event.bases import SkipHandler
from aiogram.filters import BaseFilter
from aiogram.filters.callback_data import CallbackData
from aiogram.types import FSInputFile, LinkPreviewOptions, ReplyParameters
from aiogram.utils.keyboard import InlineKeyboardBuilder

from app.media.downloads import (
    Downloader,
    MediaError,
    cache_key,
    enqueue,
    get_request,
    message_links,
    request,
)

logger = logging.getLogger(__name__)
ERRORS = {
    "music_login_required": (
        "Для полного трека нужно один раз подключить аккаунт Яндекс Музыки с Плюсом."
    ),
    "music_full_track_unavailable": "Яндекс не предоставил полный трек для этого аккаунта.",
    "music_preview_only": "Яндекс вернул только фрагмент песни. Короткую версию не отправляю.",
    "music_cover_unavailable": "Не удалось получить обложку. Попробуй позже.",
    "login_required": "Источник требует авторизацию. Файл пока не удалось получить.",
    "rate_limited": "Источник временно ограничил запросы. Автоповтор выключен.",
    "unavailable": "Не удалось получить файл: источник недоступен или ограничил загрузку.",
    "no_video": "В этой публикации нет доступного видео.",
    "invalid_response": "Источник вернул неполные данные. Скачивание остановлено.",
    "too_many_items": "В публикации слишком много медиа. Пришли ссылку на отдельное видео.",
    "too_large": "Этот файл больше доступного лимита. Выбери качество ниже.",
    "disk_full": "Для этого файла сейчас недостаточно свободного места. Загрузка остановлена.",
    "timeout": "Источник слишком долго не отвечал. Загрузка остановлена.",
    "cancelled": "Скачивание отменено.",
    "queue_full": "Очередь загрузок заполнена. Дождись готовности текущих файлов.",
    "invalid_choice": "Этот формат больше не доступен. Обнови список форматов.",
    "delivery_uncertain": "Не удалось подтвердить отправку файла. Проверь чат перед повтором.",
}


class MediaAction(CallbackData, prefix="md"):
    request_id: int
    action: str
    item: int = 0
    choice: str = "-"


def keyboard(req, max_bytes):
    b = InlineKeyboardBuilder()
    if req["platform"] == "yandex_music":
        if req["status"] in {"failed", "cancelled", "ready"}:
            b.button(
                text="Получить MP3",
                callback_data=MediaAction(request_id=req["id"], action="inspect"),
            )
        b.button(text="Отменить", callback_data=MediaAction(request_id=req["id"], action="cancel"))
        return b.as_markup()
    manifest = json.loads(req.get("manifest_json") or "{}")
    items = manifest.get("items", [])
    if items:
        for index, item in enumerate(items):
            for key, choice in item["choices"].items():
                if choice.get("size", 0) > max_bytes:
                    continue
                label = (f"{index + 1} · " if len(items) > 1 else "") + choice["label"]
                if choice.get("size"):
                    label += f" ≈{choice['size'] / 1_000_000:.0f} МБ"
                b.button(
                    text=label,
                    callback_data=MediaAction(
                        request_id=req["id"], action="get", item=index, choice=key
                    ),
                )
            b.adjust(2)
        if len(items) > 1:
            b.row()
            b.button(
                text="Все видео · лучшее доступное",
                callback_data=MediaAction(request_id=req["id"], action="all"),
            )
    else:
        b.button(
            text="Скачать · выбрать качество",
            callback_data=MediaAction(request_id=req["id"], action="inspect"),
        )
    b.row()
    b.button(
        text="Отменить загрузку", callback_data=MediaAction(request_id=req["id"], action="cancel")
    )
    b.button(text="Оригинал", url=req["source_url"])
    return b.as_markup()


async def show(bot, repository, req, settings, text=None):
    if not req.get("panel_message_id") or req["panel_message_id"] <= 0:
        return
    manifest = json.loads(req.get("manifest_json") or "{}")
    title = html.escape(manifest.get("title") or "Скачать медиа")
    body = f"<b>{title}</b>\n\n" + (
        text
        or (
            "Готовлю MP3 с обложкой…"
            if req["platform"] == "yandex_music"
            else "Выбери качество. Готовый файл придёт сюда, в Telegram."
        )
    )
    if req["platform"] != "yandex_music":
        body += f'\n\n<a href="{html.escape(req["source_url"], quote=True)}">Источник</a>'
    try:
        await bot.edit_message_text(
            chat_id=req["chat_id"],
            message_id=req["panel_message_id"],
            text=body,
            reply_markup=keyboard(req, Downloader(settings).max_bytes),
            link_preview_options=LinkPreviewOptions(is_disabled=True),
        )
    except Exception:
        logger.info("Could not update media panel %s", req["id"])


async def ensure_panel(bot, repository, req, settings):
    if req.get("panel_message_id"):
        return req
    with repository.connection() as db:
        changed = db.execute(
            "UPDATE media_requests SET panel_message_id=-1 WHERE id=? AND panel_message_id IS NULL",
            (req["id"],),
        ).rowcount
    if not changed:
        return get_request(repository, req["id"])
    try:
        msg = await bot.send_message(
            req["chat_id"],
            "Готовлю MP3 с обложкой…"
            if req["platform"] == "yandex_music"
            else "<b>Скачать медиа</b>\n\nВыбери действие.",
            reply_markup=keyboard(req, Downloader(settings).max_bytes),
            reply_parameters=ReplyParameters(
                message_id=req["request_message_id"], allow_sending_without_reply=True
            ),
        )
    except Exception:
        logger.warning("Media panel delivery unconfirmed for request %s", req["id"])
        return get_request(repository, req["id"])
    with repository.connection() as db:
        db.execute(
            "UPDATE media_requests SET panel_message_id=? WHERE id=?", (msg.message_id, req["id"])
        )
    return get_request(repository, req["id"])


def add_download_button(repository, panel, builder):
    with repository.connection() as db:
        req = db.execute(
            "SELECT id FROM media_requests WHERE chat_id=? AND video_panel_id=? "
            "ORDER BY id DESC LIMIT 1",
            (panel["chat_id"], panel["id"]),
        ).fetchone()
    if req:
        b = InlineKeyboardBuilder()
        b.button(
            text="Скачать видео / MP3",
            callback_data=MediaAction(request_id=req["id"], action="open"),
        )
        builder.attach(b)


def remember_youtube(repository, message, job_id):
    links = message_links(message)
    if len(links) != 1:
        return
    with repository.connection() as db:
        job = db.execute("SELECT panel_id FROM video_pdf_jobs WHERE id=?", (job_id,)).fetchone()
    if job and job["panel_id"]:
        request(
            repository,
            chat_id=message.chat.id,
            message_id=message.message_id,
            link=links[0],
            video_panel_id=job["panel_id"],
        )


class MediaFilter(BaseFilter):
    async def __call__(self, message):
        text = message.text or message.caption or ""
        command = text.split(maxsplit=1)[0].split("@")[0].lower() if text.strip() else ""
        if command.startswith("/"):
            return command == "/download"
        return any(link[0] in {"x", "yandex_music"} for link in message_links(message))


def register_media_handlers(router, settings, repository):
    from app.telegram_bot.bot import is_allowed

    @router.message(MediaFilter())
    async def incoming(message):
        user = message.from_user.id if message.from_user else None
        if not is_allowed(settings, repository, user) or message.chat.type != "private":
            return
        if not settings.media_downloads_enabled:
            raise SkipHandler()
        explicit = (message.text or "").startswith("/download")
        feedback = repository.feedback_session_for_user(user)
        if not explicit and feedback and feedback["state"] == "awaiting_comment":
            raise SkipHandler()
        links = message_links(message)
        if len(links) != 1:
            await message.answer("Пришли одну ссылку на YouTube, пост X или трек Яндекс Музыки.")
            return
        req = request(
            repository, chat_id=message.chat.id, message_id=message.message_id, link=links[0]
        )
        if req["platform"] == "yandex_music":
            with repository.connection() as db:
                sent = db.execute(
                    "SELECT 1 FROM media_download_jobs WHERE request_id=? AND status='sent'",
                    (req["id"],),
                ).fetchone()
            if sent:
                return  # Telegram redelivery of the same input must not send a second song.
        await ensure_panel(message.bot, repository, req, settings)
        if req["platform"] == "yandex_music":
            if req["status"] == "ready":
                enqueue(repository, req, 0, "mp3", once=True)
            else:
                with repository.connection() as db:
                    db.execute(
                        "UPDATE media_requests SET status='queued',error=NULL WHERE id=? "
                        "AND status IN ('new','failed','cancelled')",
                        (req["id"],),
                    )

    @router.callback_query(MediaAction.filter())
    async def clicked(callback, callback_data):
        user = callback.from_user.id
        req = get_request(repository, callback_data.request_id)
        if (
            not req
            or user != req["chat_id"]
            or not is_allowed(settings, repository, user)
            or callback.message is None
            or callback.message.chat.id != req["chat_id"]
            or callback.message.chat.type != "private"
        ):
            await callback.answer("Эта карточка недоступна", show_alert=True)
            return
        if not settings.media_downloads_enabled:
            await callback.answer("Скачивание выключено", show_alert=True)
            return
        if callback_data.action == "open":
            with repository.connection() as db:
                p = db.execute(
                    "SELECT message_id FROM video_panels WHERE id=?", (req["video_panel_id"],)
                ).fetchone()
            if not p or p[0] != callback.message.message_id:
                await callback.answer("Открой исходную карточку", show_alert=True)
                return
            await callback.answer()
            await ensure_panel(callback.bot, repository, req, settings)
            return
        if callback.message.message_id != req["panel_message_id"]:
            await callback.answer("Открой текущую карточку скачивания", show_alert=True)
            return
        if callback_data.action == "cancel":
            with repository.connection() as db:
                db.execute("UPDATE media_requests SET status='cancelled' WHERE id=?", (req["id"],))
                db.execute(
                    "UPDATE media_download_jobs SET status='cancelled' WHERE request_id=? "
                    "AND status IN ('queued','running')",
                    (req["id"],),
                )
            await callback.answer("Отмена принята")
            await show(callback.bot, repository, req, settings, "Скачивание отменено.")
            return
        if callback_data.action == "inspect":
            with repository.connection() as db:
                if req["platform"] == "yandex_music":
                    uncertain = db.execute(
                        "SELECT 1 FROM media_download_jobs WHERE request_id=? "
                        "AND status='uncertain'",
                        (req["id"],),
                    ).fetchone()
                    if uncertain:
                        await callback.answer(ERRORS["delivery_uncertain"], show_alert=True)
                        return
                    db.execute(
                        "UPDATE media_download_jobs SET status='queued',error=NULL,updated_at=? "
                        "WHERE request_id=? AND status IN ('failed','cancelled')",
                        (time.time(), req["id"]),
                    )
                db.execute(
                    "UPDATE media_requests SET status='queued',error=NULL WHERE id=? "
                    "AND status NOT IN ('queued','preparing')",
                    (req["id"],),
                )
            await callback.answer("Получаю доступные форматы")
            await show(callback.bot, repository, req, settings, "Получаю доступные форматы…")
            return
        try:
            if callback_data.action == "get":
                enqueue(repository, req, callback_data.item, callback_data.choice)
            elif callback_data.action == "all":
                for i, item in enumerate(json.loads(req["manifest_json"])["items"]):
                    choices = [
                        (int(k), k)
                        for k, v in item["choices"].items()
                        if k.isdigit() and v.get("size", 0) <= Downloader(settings).max_bytes
                    ]
                    if choices:
                        enqueue(repository, req, i, max(choices)[1])
            else:
                raise MediaError("invalid_choice")
        except (MediaError, ValueError, KeyError, TypeError) as exc:
            await callback.answer(
                ERRORS.get(getattr(exc, "code", ""), "Обнови список форматов"), show_alert=True
            )
            return
        await callback.answer("Скачивание в очереди")
        await show(callback.bot, repository, req, settings, "Скачивание в очереди…")


class MediaWorker:
    def __init__(self, bot, settings, repository, *, downloader=None):
        self.bot, self.settings, self.repository = bot, settings, repository
        self.downloader = downloader or Downloader(settings)

    def recover(self):
        with self.repository.connection() as db:
            db.execute("UPDATE media_requests SET status='queued' WHERE status='preparing'")
            db.execute("UPDATE media_download_jobs SET status='queued' WHERE status='running'")
            db.execute(
                "UPDATE media_download_jobs SET status='uncertain',error='delivery_uncertain' "
                "WHERE status='sending'"
            )
            music = [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM media_requests WHERE platform='yandex_music' "
                    "AND status='ready' AND manifest_json IS NOT NULL"
                )
            ]
        for req in music:
            try:
                enqueue(self.repository, req, 0, "mp3", once=True)
            except MediaError as exc:
                if exc.code != "queue_full":
                    raise
                break  # The normal loop resumes remaining gaps as space becomes available.

    def cancelled(self, table, ident):
        with self.repository.connection() as db:
            row = db.execute(f"SELECT status FROM {table} WHERE id=?", (ident,)).fetchone()
        return not row or row[0] == "cancelled"

    async def run_once(self):
        with self.repository.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM media_requests WHERE status='queued' ORDER BY id LIMIT 1"
            ).fetchone()
            if row:
                db.execute("UPDATE media_requests SET status='preparing' WHERE id=?", (row["id"],))
            else:
                job = db.execute(
                    "SELECT * FROM media_download_jobs WHERE status='queued' ORDER BY id LIMIT 1"
                ).fetchone()
                if job:
                    db.execute(
                        "UPDATE media_download_jobs SET status='running',updated_at=? WHERE id=?",
                        (time.time(), job["id"]),
                    )
        if row:
            req = dict(row)
            try:
                manifest = await self.downloader.inspect(
                    req, cancelled=lambda: self.cancelled("media_requests", req["id"])
                )
                if self.cancelled("media_requests", req["id"]):
                    raise MediaError("cancelled")
                with self.repository.connection() as db:
                    db.execute(
                        "UPDATE media_requests SET status='ready',manifest_json=?,"
                        "error=NULL,updated_at=? WHERE id=? AND status='preparing'",
                        (json.dumps(manifest, ensure_ascii=False), time.time(), req["id"]),
                    )
                if req["platform"] == "yandex_music":
                    enqueue(
                        self.repository,
                        get_request(self.repository, req["id"]),
                        0,
                        "mp3",
                        once=True,
                    )
                await show(
                    self.bot,
                    self.repository,
                    get_request(self.repository, req["id"]),
                    self.settings,
                )
            except MediaError as exc:
                await self.failure(req, None, exc.code)
            except Exception:
                logger.exception("Media inspection failed for request %s", req["id"])
                await self.failure(req, None, "unavailable")
            return True
        if not job:
            with self.repository.connection() as db:
                pending = db.execute(
                    "SELECT r.* FROM media_requests r WHERE r.platform='yandex_music' "
                    "AND r.status='ready' AND r.manifest_json IS NOT NULL AND NOT EXISTS "
                    "(SELECT 1 FROM media_download_jobs j WHERE j.request_id=r.id) "
                    "ORDER BY r.id LIMIT 1"
                ).fetchone()
            if pending:
                enqueue(self.repository, dict(pending), 0, "mp3", once=True)
                return True
            return False
        job = dict(job)
        req = get_request(self.repository, job["request_id"])
        key = cache_key(req, job["item_index"], job["choice"])
        with self.repository.connection() as db:
            cached = db.execute(
                "SELECT * FROM media_file_cache WHERE cache_key=? AND bot_id=?", (key, self.bot.id)
            ).fetchone()
        try:
            await show(
                self.bot,
                self.repository,
                req,
                self.settings,
                "Отправляю сохранённый файл…" if cached else "Скачиваю и готовлю файл…",
            )
            result = (
                json.loads(cached["metadata_json"])
                if cached
                else await self.downloader.download(
                    req, job, cancelled=lambda: self.cancelled("media_download_jobs", job["id"])
                )
            )
            if self.cancelled("media_download_jobs", job["id"]):
                raise MediaError("cancelled")
            with self.repository.connection() as db:
                db.execute(
                    "UPDATE media_download_jobs SET status='sending',result_json=? WHERE id=?",
                    (json.dumps(result), job["id"]),
                )
            file = cached["file_id"] if cached else FSInputFile(result["path"])
            caption = (
                html.escape(result["title"][:600])
                + '\n\n<a href="'
                + html.escape(req["source_url"], quote=True)
                + '">Источник</a>'
            )
            common = dict(chat_id=req["chat_id"], caption=caption, request_timeout=900)
            music = req["platform"] == "yandex_music"
            if music:
                common.pop("caption")
            try:
                if result["kind"] == "audio":
                    extra = {}
                    if music:
                        extra["performer"] = result["performer"][:300]
                        if not cached:
                            extra["thumbnail"] = FSInputFile(result["thumbnail_path"])
                    msg = await self.bot.send_audio(
                        audio=file,
                        title=result["title"][:300],
                        duration=result["duration"],
                        **extra,
                        **common,
                    )
                    sent_file = msg.audio
                else:
                    msg = await self.bot.send_video(
                        video=file,
                        supports_streaming=True,
                        width=result["width"],
                        height=result["height"],
                        duration=result["duration"],
                        **common,
                    )
                    sent_file = msg.video
            except Exception as exc:
                raise MediaError("delivery_uncertain") from exc
            with self.repository.connection() as db:
                db.execute(
                    "INSERT OR REPLACE INTO media_file_cache VALUES(?,?,?,?,?,?)",
                    (
                        key,
                        self.bot.id,
                        sent_file.file_id,
                        result["kind"],
                        json.dumps(
                            {k: v for k, v in result.items() if k not in {"path", "thumbnail_path"}}
                        ),
                        time.time(),
                    ),
                )
                db.execute(
                    "UPDATE media_download_jobs SET status='sent',sent_message_id=?,"
                    "updated_at=? WHERE id=?",
                    (msg.message_id, time.time(), job["id"]),
                )
            self.downloader.remove_job_files(job["id"])
            if music:
                if req.get("panel_message_id"):
                    try:
                        await self.bot.delete_message(req["chat_id"], req["panel_message_id"])
                        with self.repository.connection() as db:
                            db.execute(
                                "UPDATE media_requests SET panel_message_id=NULL WHERE id=?",
                                (req["id"],),
                            )
                    except Exception:
                        logger.info("Could not remove music progress message %s", req["id"])
                return True
            await show(
                self.bot,
                self.repository,
                req,
                self.settings,
                "Файл отправлен. Его можно сохранить или переслать в канал.",
            )
        except MediaError as exc:
            await self.failure(req, job, exc.code)
        except Exception:
            logger.exception("Media processing failed for job %s", job["id"])
            with self.repository.connection() as db:
                state = db.execute(
                    "SELECT status FROM media_download_jobs WHERE id=?", (job["id"],)
                ).fetchone()[0]
            await self.failure(
                req, job, "delivery_uncertain" if state == "sending" else "unavailable"
            )
        return True

    async def failure(self, req, job, code):
        with self.repository.connection() as db:
            if job:
                state = (
                    "uncertain"
                    if code == "delivery_uncertain"
                    else ("cancelled" if code == "cancelled" else "failed")
                )
                db.execute(
                    "UPDATE media_download_jobs SET status=?,error=?,updated_at=? WHERE id=?",
                    (state, code, time.time(), job["id"]),
                )
            else:
                db.execute(
                    "UPDATE media_requests SET status='failed',error=?,updated_at=? WHERE id=?",
                    (code, time.time(), req["id"]),
                )
        if job and code != "delivery_uncertain":
            self.downloader.remove_job_files(job["id"])
        await show(
            self.bot,
            self.repository,
            req,
            self.settings,
            ERRORS.get(code, "Не удалось получить файл. Автоповтор выключен."),
        )

    async def run_loop(self):
        self.recover()
        last_cleanup = 0.0
        while True:
            try:
                if time.monotonic() - last_cleanup > 3600:
                    with self.repository.connection() as db:
                        active = {
                            r[0]
                            for r in db.execute(
                                "SELECT id FROM media_download_jobs WHERE status IN "
                                "('queued','running','sending')"
                            )
                        }
                    self.downloader.cleanup(active)
                    last_cleanup = time.monotonic()
                if not await self.run_once():
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Media worker failed")
                await asyncio.sleep(3)
