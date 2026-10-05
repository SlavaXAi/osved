"""Owner-only YouTube inbox and one durable text/PDF worker; other handlers stay live."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from contextlib import suppress
from weakref import WeakValueDictionary

from aiogram import F
from aiogram.dispatcher.event.bases import SkipHandler
from aiogram.filters import BaseFilter
from aiogram.types import (
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    LinkPreviewOptions,
    ReplyParameters,
)

from app.config import Settings
from app.db.repository import Repository
from app.telegram_bot.progress import ProgressPulse, progress_text
from app.video.reports import ReportError, VideoReportService
from app.video.transcripts import youtube_video_id

logger = logging.getLogger(__name__)
_progress_locks: WeakValueDictionary[tuple[str, int], asyncio.Lock] = WeakValueDictionary()
ERRORS = {
    "unavailable": "Не удалось получить субтитры этого ролика. Содержание не придумываю.",
    "incomplete_transcript": "Субтитры пришли не целиком. Отчёт по обрезанному видео не создаю.",
    "insufficient_context": "Этот ролик не помещается в лимит одного отчёта. Субтитры не обрезал.",
    "budget_blocked": ("Достигнут лимит расходов OSVED. Следующий платный запрос не запускался."),
    "provider_blocked": "Monid остановил запрос своим лимитом. Проверь ограничения в app.monid.ai.",
    "insufficient_credits": (
        "OpenRouter отклонил запрос из-за нехватки средств или ограничения расходов. "
        "Проверь баланс и лимиты OpenRouter, затем отправь ссылку ещё раз. "
        "Субтитры сохранены; повторно получать их не нужно."
    ),
    "disabled": "Для резюме пока не настроены Monid или OpenRouter.",
    "uncertain": (
        "Результат платного запроса не удалось подтвердить. Автоповтор отключён, "
        "чтобы не списывать деньги повторно. Запрос сохранён для проверки."
    ),
    "invalid_output": (
        "Ответ сохранён, но не прошёл проверку формата. Повторно за генерацию не плачу."
    ),
    "failed": (
        "Резюме не удалось подготовить. Запрос сохранён; "
        "новые платные попытки автоматически не запускаются."
    ),
    "expired": (
        "Субтитры пока не готовы. Запрос провайдера сохранён: "
        "повторная ссылка продолжит проверку без нового извлечения."
    ),
}


def video_links(message) -> list[str]:
    text = message.text or message.caption or ""
    values = re.findall(r"https://[^\s<>]+", text)
    for entity in message.entities or message.caption_entities or []:
        if entity.type == "text_link" and entity.url:
            values.append(entity.url)
    ids = []
    for value in values:
        try:
            video = youtube_video_id(value.rstrip(".,!?)\"'>]"))
        except ValueError:
            continue
        if video not in ids:
            ids.append(video)
    return ids


class VideoReportFilter(BaseFilter):
    async def __call__(self, message):
        text = message.text or message.caption or ""
        command = text.split(maxsplit=1)[0].split("@")[0].lower() if text.strip() else ""
        if command.startswith("/") and command != "/report":
            return False
        return command == "/report" or bool(video_links(message))


def enqueue(
    settings: Settings,
    repository: Repository,
    service: VideoReportService,
    video: str,
    user_id: int,
    chat_id: int,
    message_id: int,
    *,
    output_kind: str = "pdf",
) -> dict:
    if output_kind not in {"pdf", "tldr"}:
        raise ValueError("Unknown video output kind")
    video = youtube_video_id(video)
    now = time.time()
    with repository.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        old = db.execute(
            "SELECT * FROM video_pdf_jobs WHERE chat_id=? AND request_message_id=? "
            "AND output_kind=?",
            (chat_id, message_id, output_kind),
        ).fetchone()
        if old:
            return dict(old, existing=True)
        pending = db.execute(
            "SELECT * FROM video_pdf_jobs WHERE chat_id=? AND video_id=? "
            "AND output_kind=? AND status IN ('queued','running','sending') "
            "ORDER BY id DESC LIMIT 1",
            (chat_id, video, output_kind),
        ).fetchone()
        if pending:
            return dict(pending, existing=True)
        # Owner-requested reports have no queue quota; the worker processes them serially.
        row = db.execute(
            "INSERT INTO video_pdf_jobs"
            "(video_id,user_id,chat_id,request_message_id,output_kind,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (video, user_id, chat_id, message_id, output_kind, now, now),
        )
        return {"id": row.lastrowid, "status": "queued", "existing": False}


async def ensure_video_progress(bot, repository: Repository, job_id: int) -> int | None:
    """One status per job, shared by the button handler and the worker.

    The local lock closes their normal race; the durable -1 claim prevents an
    ambiguous send from being replayed after restart. Cosmetic failure must not
    fail a report or trigger a new paid request.
    """
    key = (str(repository.database_path), job_id)
    lock = _progress_locks.setdefault(key, asyncio.Lock())
    async with lock:
        with repository.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM video_pdf_jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                return None
            job = dict(row)
            known = job["status_message_id"]
            if isinstance(known, int) and known > 0:
                return known
            if known == -1 or job["status"] not in {"queued", "running"}:
                return None
            db.execute("UPDATE video_pdf_jobs SET status_message_id=-1 WHERE id=?", (job_id,))
        try:
            message = await bot.send_message(
                chat_id=job["chat_id"],
                text=progress_text(job["output_kind"], "queued"),
                parse_mode="HTML",
                link_preview_options=LinkPreviewOptions(is_disabled=True),
                reply_parameters=ReplyParameters(
                    message_id=job["request_message_id"], allow_sending_without_reply=True
                )
                if job["request_message_id"] > 0
                else None,
                request_timeout=8,
            )
            if not isinstance(message.message_id, int) or message.message_id <= 0:
                return None
            with repository.connection() as db:
                db.execute(
                    "UPDATE video_pdf_jobs SET status_message_id=? WHERE id=? "
                    "AND status_message_id=-1",
                    (message.message_id, job_id),
                )
            return message.message_id
        except Exception:
            logger.warning("Progress message not confirmed for video job %s", job_id)
            return None


def register_video_report_handlers(router, settings: Settings, repository: Repository) -> None:
    from app.telegram_bot.bot import is_allowed

    @router.callback_query(F.data.startswith("vreport:long:"))
    async def video_long_report_handler(callback):
        if not is_allowed(settings, repository, callback.from_user.id):
            await callback.answer()
            return
        message = callback.message
        job_id = callback.data.removeprefix("vreport:long:")
        with repository.connection() as db:
            row = db.execute(
                "SELECT * FROM video_pdf_jobs WHERE id=? AND user_id=? AND output_kind='tldr'",
                (job_id, callback.from_user.id),
            ).fetchone()
        if (
            row is None
            or message is None
            or message.chat.type != "private"
            or message.chat.id != row["chat_id"]
            or message.message_id != row["text_message_id"]
            or row["status"] != "sent"
        ):
            await callback.answer("Отчёт недоступен")
            return
        if not settings.video_reports_enabled:
            await callback.answer("Отчёты по видео сейчас выключены", show_alert=True)
            return
        result = enqueue(
            settings,
            repository,
            VideoReportService(settings, repository),
            row["video_id"],
            callback.from_user.id,
            row["chat_id"],
            row["request_message_id"],
            output_kind="pdf",
        )
        if result.get("existing"):
            if result["status"] == "sent":
                await callback.answer("Длинный отчёт уже отправлен")
            elif result["status"] in {"failed", "uncertain"}:
                await callback.answer(
                    ERRORS.get(result.get("error"), ERRORS["failed"]), show_alert=True
                )
            else:
                await callback.answer("Длинный отчёт уже в очереди")
        else:
            await callback.answer("Готовлю длинный отчёт · PDF")
            await ensure_video_progress(callback.bot, repository, result["id"])

    @router.message(VideoReportFilter())
    async def video_report_handler(message):
        user_id = message.from_user.id if message.from_user else None
        if not is_allowed(settings, repository, user_id) or message.chat.type != "private":
            return
        text = message.text or message.caption or ""
        explicit = text.split(maxsplit=1)[0].split("@")[0].lower() == "/report"
        session = repository.feedback_session_for_user(user_id)
        if session and session["state"] == "awaiting_comment" and not explicit:
            raise SkipHandler()  # A link can be part of the user's superreaction comment.
        if not settings.video_reports_enabled:
            await message.answer("PDF-отчёты сейчас выключены.")
            return
        links = video_links(message)
        if len(links) != 1:
            await message.answer(
                "Пришли одну ссылку YouTube. Можно так: /report https://youtu.be/…"
            )
            return
        result = enqueue(
            settings,
            repository,
            VideoReportService(settings, repository),
            links[0],
            user_id,
            message.chat.id,
            message.message_id,
            output_kind="pdf" if explicit else "tldr",
        )
        if result.get("existing"):
            return
        await ensure_video_progress(message.bot, repository, result["id"])


class VideoReportWorker:
    def __init__(self, bot, settings: Settings, repository: Repository, *, service=None):
        self.bot, self.settings, self.repository = bot, settings, repository
        self.service = service or VideoReportService(settings, repository)

    def recover(self):
        with self.repository.connection() as db:
            # Generation is protected by durable stage markers. Delivery cannot be replayed
            # after an ambiguous send: Telegram has no client idempotency key for sendDocument.
            db.execute("UPDATE video_pdf_jobs SET status='queued' WHERE status='running'")
            db.execute(
                "UPDATE video_pdf_jobs SET status='uncertain',error='delivery_uncertain' "
                "WHERE status='sending'"
            )

    async def tick(self) -> bool:
        from app.telegram_bot.bot import configured_owner_id

        now = time.time()
        with self.repository.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM video_pdf_jobs WHERE status='queued' AND available_at<=? "
                "ORDER BY id LIMIT 1",
                (now,),
            ).fetchone()
            if not row:
                return False
            job = dict(row)
            db.execute(
                "UPDATE video_pdf_jobs SET status='running',updated_at=? WHERE id=?",
                (now, job["id"]),
            )
        if (
            not self.settings.video_reports_enabled
            or job["user_id"] != configured_owner_id(self.settings, self.repository)
            or job["chat_id"] != job["user_id"]
        ):
            self._finish(job, "failed", "disabled")
            return True
        job["status_message_id"] = await ensure_video_progress(self.bot, self.repository, job["id"])
        try:
            async with ProgressPulse(
                self.bot,
                chat_id=job["chat_id"],
                message_id=lambda: job["status_message_id"],
                kind=job["output_kind"],
                elapsed=time.time() - job["created_at"],
            ) as pulse:
                result = await asyncio.to_thread(
                    self.service.build_tldr if job["output_kind"] == "tldr" else self.service.build,
                    job["video_id"],
                    progress=pulse.set_stage,
                )
        except ReportError as exc:
            if exc.status == "processing" and now - job["created_at"] < 1800:
                with self.repository.connection() as db:
                    db.execute(
                        "UPDATE video_pdf_jobs SET status='queued',available_at=?,updated_at=? "
                        "WHERE id=?",
                        (time.time() + 15, time.time(), job["id"]),
                    )
                await self._notice(
                    job,
                    progress_text(
                        job["output_kind"],
                        "waiting_provider",
                        elapsed=time.time() - job["created_at"],
                    ),
                )
                return True
            status = "expired" if exc.status == "processing" else exc.status
            self._finish(job, "failed", status)
            await self._notice(job, ERRORS.get(status, ERRORS["failed"]))
            return True
        except Exception:
            logger.exception("Video report job %s failed locally", job["id"])
            self._finish(job, "failed", "failed")
            await self._notice(job, ERRORS["failed"])
            return True
        self._finish(job, "sending")
        try:
            reply = (
                ReplyParameters(
                    message_id=job["request_message_id"], allow_sending_without_reply=True
                )
                if job["request_message_id"] > 0
                else None
            )
            if job["output_kind"] == "tldr":
                from app.video.tldr import telegram_tldr

                markup = InlineKeyboardMarkup(
                    inline_keyboard=[
                        [
                            InlineKeyboardButton(
                                text="Длинный отчёт · PDF",
                                callback_data=f"vreport:long:{job['id']}",
                            )
                        ]
                    ]
                )
                if job["status_message_id"]:
                    await self.bot.edit_message_text(
                        chat_id=job["chat_id"],
                        message_id=job["status_message_id"],
                        text=telegram_tldr(result),
                        reply_markup=markup,
                        parse_mode="HTML",
                        link_preview_options=LinkPreviewOptions(is_disabled=True),
                    )
                    delivered_message_id = job["status_message_id"]
                else:
                    response = await self.bot.send_message(
                        chat_id=job["chat_id"],
                        text=telegram_tldr(result),
                        reply_markup=markup,
                        parse_mode="HTML",
                        link_preview_options=LinkPreviewOptions(is_disabled=True),
                        reply_parameters=reply,
                    )
                    delivered_message_id = response.message_id
            else:
                async with ProgressPulse(
                    self.bot,
                    chat_id=job["chat_id"],
                    message_id=lambda: job["status_message_id"],
                    kind="pdf",
                    elapsed=time.time() - job["created_at"],
                ) as pulse:
                    pulse.set_stage("delivery")
                    response = await self.bot.send_document(
                        chat_id=job["chat_id"],
                        document=FSInputFile(result, filename=f"OSVED-{job['video_id']}.pdf"),
                        reply_parameters=reply,
                    )
                delivered_message_id = response.message_id
        except Exception:
            self._finish(job, "uncertain", "delivery_uncertain")
            await self._notice(
                job,
                "Результат готов, но подтверждение доставки не пришло. "
                "Сохранил его; повторную генерацию автоматически не запускаю.",
            )
            return True
        with self.repository.connection() as db:
            field = "text_message_id" if job["output_kind"] == "tldr" else "document_message_id"
            db.execute(
                f"UPDATE video_pdf_jobs SET status='sent',{field}=?,updated_at=? WHERE id=?",
                (delivered_message_id, time.time(), job["id"]),
            )
        # Delivery is already recorded. Cosmetic cleanup must never resend the PDF.
        if (
            job["output_kind"] == "pdf"
            and isinstance(job["status_message_id"], int)
            and job["status_message_id"] > 0
        ):
            try:
                await self.bot.delete_message(
                    chat_id=job["chat_id"], message_id=job["status_message_id"]
                )
            except Exception:
                logger.warning("Progress cleanup failed for video job %s", job["id"])
        # A short request sends only the TL;DR, with no extra 'done' notification.
        return True

    def _finish(self, job, status, error=None):
        with self.repository.connection() as db:
            db.execute(
                "UPDATE video_pdf_jobs SET status=?,error=?,updated_at=? WHERE id=?",
                (status, error, time.time(), job["id"]),
            )

    async def _notice(self, job, text):
        with suppress(Exception):
            if isinstance(job["status_message_id"], int) and job["status_message_id"] > 0:
                await self.bot.edit_message_text(
                    chat_id=job["chat_id"],
                    message_id=job["status_message_id"],
                    text=text,
                    parse_mode="HTML",
                    link_preview_options=LinkPreviewOptions(is_disabled=True),
                )
            else:
                await self.bot.send_message(chat_id=job["chat_id"], text=text)

    async def run_loop(self):
        self.recover()
        while True:
            try:
                await self.tick()
            except Exception:
                logger.exception("Video report worker tick failed")
            await asyncio.sleep(2)
