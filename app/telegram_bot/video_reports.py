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
    LinkPreviewOptions,
    ReplyParameters,
)

from app.config import Settings
from app.db.repository import Repository
from app.telegram_bot.progress import ProgressPulse, progress_text
from app.telegram_bot.video_panel import (
    PanelProgressBot,
    VideoPanelControl,
    VideoResultControl,
    bind_panel_job,
    edit_panel,
    ensure_panel,
    get_panel,
    panel_busy,
    panel_keyboard,
    register_result_message,
    result_keyboard,
    show_cached_summary,
    summary_html,
)
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
    panel_id: int | None = None,
) -> dict:
    if output_kind not in {"pdf", "tldr"}:
        raise ValueError("Unknown video output kind")
    video = youtube_video_id(video)
    now = time.time()
    with repository.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        if panel_id is not None:
            panel = db.execute("SELECT * FROM video_panels WHERE id=?", (panel_id,)).fetchone()
            if (
                not panel
                or panel["video_id"] != video
                or panel["chat_id"] != chat_id
                or user_id != chat_id
            ):
                raise PermissionError("Foreign video panel")
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
            "(video_id,user_id,chat_id,request_message_id,output_kind,created_at,"
            "updated_at,panel_id) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (video, user_id, chat_id, message_id, output_kind, now, now, panel_id),
        )
        return {"id": row.lastrowid, "status": "queued", "existing": False}


async def ensure_video_progress(bot, repository: Repository, job_id: int) -> int | None:
    """Claim an anchor once; bound requests update their original card in place."""
    key = (str(repository.database_path), job_id)
    lock = _progress_locks.setdefault(key, asyncio.Lock())
    async with lock:
        with repository.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM video_pdf_jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                return None
            job = dict(row)
            panel_row = db.execute(
                "SELECT * FROM video_panels WHERE id=?", (job.get("panel_id"),)
            ).fetchone()
            panel = dict(panel_row) if panel_row else None
            if panel:
                if panel["chat_id"] != job["chat_id"] or panel["video_id"] != job["video_id"]:
                    return None
                known = panel["message_id"]
                db.execute(
                    "UPDATE video_pdf_jobs SET status_message_id=? WHERE id=?", (known, job_id)
                )
            else:
                known = job["status_message_id"]
                if isinstance(known, int) and known > 0:
                    return known
                if known == -1 or job["status"] not in {"queued", "running"}:
                    return None
                db.execute("UPDATE video_pdf_jobs SET status_message_id=-1 WHERE id=?", (job_id,))
        if panel:
            try:
                await bot.edit_message_text(
                    chat_id=job["chat_id"],
                    message_id=known,
                    text=progress_text(job["output_kind"], "queued"),
                    parse_mode="HTML",
                    reply_markup=panel_keyboard(repository, panel),
                    link_preview_options=LinkPreviewOptions(is_disabled=True),
                )
            except Exception:
                logger.warning("Panel progress unavailable for job %s", job_id)
            return known
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
            panel = ensure_panel(
                repository,
                chat_id=job["chat_id"],
                message_id=message.message_id,
                video_id=job["video_id"],
                original_html=(
                    '<b>Видео</b>\n\n<a href="https://www.youtube.com/watch?v='
                    + job["video_id"]
                    + '">Смотреть видео</a>'
                ),
            )
            bind_panel_job(repository, panel["id"], job_id)
            with repository.connection() as db:
                db.execute(
                    "UPDATE video_pdf_jobs SET status_message_id=? WHERE id=? "
                    "AND status_message_id=-1",
                    (message.message_id, job_id),
                )
            with suppress(Exception):
                await bot.edit_message_reply_markup(
                    chat_id=job["chat_id"],
                    message_id=message.message_id,
                    reply_markup=panel_keyboard(repository, get_panel(repository, panel["id"])),
                )
            return message.message_id
        except Exception:
            logger.warning("Progress anchor not confirmed for job %s", job_id)
            return None


async def request_panel_output(callback, settings, repository, panel, output_kind):
    if not settings.video_reports_enabled:
        await callback.answer("Отчёты сейчас выключены", show_alert=True)
        return
    if panel_busy(repository, panel):
        await callback.answer("Уже готовлю результат; меню останется здесь")
        return
    service = VideoReportService(settings, repository)
    result = enqueue(
        settings,
        repository,
        service,
        panel["video_id"],
        callback.from_user.id,
        panel["chat_id"],
        panel["message_id"],
        output_kind=output_kind,
        panel_id=panel["id"],
    )
    with repository.connection() as db:
        job = dict(
            db.execute("SELECT * FROM video_pdf_jobs WHERE id=?", (result["id"],)).fetchone()
        )
    if job["status"] == "sent" or (output_kind == "tldr" and job["error"] == "delivery_uncertain"):
        if output_kind == "pdf":
            await callback.answer("PDF уже отправлен в этот чат", show_alert=True)
            return
        await callback.answer("Открываю сохранённую сводку")
        shown = await show_cached_summary(callback.bot, settings, repository, service, job, panel)
        if not shown:
            await edit_panel(
                callback.bot,
                repository,
                panel,
                panel["original_html"]
                + "\n\nСохранённая сводка недоступна. Повторную оплату автоматически не запускаю.",
                view="card",
            )
        return
    if job["status"] in {"failed", "uncertain"}:
        await callback.answer(ERRORS.get(job["error"], ERRORS["failed"]), show_alert=True)
        return
    if not bind_panel_job(repository, panel["id"], job["id"]):
        await callback.answer("Этот выпуск уже готовится; открой сводку чуть позже")
        return
    await callback.answer("Готовлю сводку здесь" if output_kind == "tldr" else "Готовлю PDF")
    await ensure_video_progress(callback.bot, repository, job["id"])


def register_video_report_handlers(router, settings: Settings, repository: Repository) -> None:
    from app.telegram_bot.bot import is_allowed

    @router.callback_query(VideoPanelControl.filter())
    async def video_panel_handler(callback, callback_data: VideoPanelControl):
        panel = get_panel(repository, callback_data.panel_id)
        if (
            not is_allowed(settings, repository, callback.from_user.id)
            or not panel
            or panel["chat_id"] != callback.from_user.id
            or not callback.message
            or callback.message.chat.id != panel["chat_id"]
            or callback.message.message_id != panel["message_id"]
        ):
            await callback.answer("Карточка недоступна", show_alert=True)
            return
        action = callback_data.action
        if panel_busy(repository, panel):
            await callback.answer("Уже готовлю результат; меню останется здесь")
            return
        if action == "card":
            await callback.answer()
            await edit_panel(callback.bot, repository, panel, panel["original_html"], view="card")
            return
        if action == "dismiss" and panel.get("recommendation_id"):
            with repository.connection() as db:
                db.execute(
                    "INSERT INTO video_feedback_events(recommendation_id,action,created_at) "
                    "VALUES(?,'dismiss',datetime('now'))",
                    (panel["recommendation_id"],),
                )
            await callback.answer("Отметка о выборе видео сохранена")
            return
        if action not in ("short", "pdf"):
            await callback.answer()
            return
        await request_panel_output(
            callback, settings, repository, panel, "tldr" if action == "short" else "pdf"
        )

    @router.callback_query(VideoResultControl.filter())
    async def video_result_feedback_handler(callback, callback_data: VideoResultControl):
        from app.video.result_feedback import save_result_feedback

        if not is_allowed(settings, repository, callback.from_user.id) or not callback.message:
            await callback.answer("Недоступно", show_alert=True)
            return
        with repository.connection() as db:
            row = db.execute(
                "SELECT * FROM video_pdf_jobs WHERE id=?", (callback_data.job_id,)
            ).fetchone()
        if row is None or callback.message.chat.id != callback.from_user.id:
            await callback.answer("Результат недоступен")
            return
        job = dict(row)
        with repository.connection() as db:
            origin = db.execute(
                "SELECT * FROM video_panels WHERE chat_id=? AND message_id=?",
                (callback.from_user.id, callback.message.message_id),
            ).fetchone()
        panel = dict(origin) if origin else None
        if (
            job["output_kind"] == "tldr"
            and panel
            and callback.message.message_id == panel["message_id"]
            and (panel["view"] != "summary" or panel["summary_job_id"] != job["id"])
        ):
            await callback.answer("Открой сводку, чтобы оценить её")
            return
        try:
            save_result_feedback(
                repository,
                job_id=job["id"],
                chat_id=callback.message.chat.id,
                user_id=callback.from_user.id,
                reaction=callback_data.reaction,
                message_id=callback.message.message_id,
            )
        except (PermissionError, ValueError):
            await callback.answer("Результат недоступен")
            return
        await callback.answer("Полезность результата сохранена")
        try:
            await callback.message.edit_reply_markup(
                reply_markup=result_keyboard(
                    repository, job, message_id=callback.message.message_id
                )
            )
        except Exception:
            logger.warning("Feedback persisted, keyboard update failed for job %s", job["id"])

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
            alias = db.execute(
                "SELECT 1 FROM video_result_messages WHERE job_id=? AND chat_id=? "
                "AND message_id=? AND output_kind='tldr'",
                (job_id, callback.from_user.id, message.message_id if message else None),
            ).fetchone()
        if (
            row is None
            or message is None
            or message.chat.type != "private"
            or message.chat.id != row["chat_id"]
            or (message.message_id != row["text_message_id"] and not alias)
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
            panel_id=row["panel_id"],
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
            if row["panel_id"]:
                bind_panel_job(repository, row["panel_id"], result["id"])
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
        anchor_id = await ensure_video_progress(self.bot, self.repository, job["id"])
        with self.repository.connection() as db:
            job = dict(
                db.execute("SELECT * FROM video_pdf_jobs WHERE id=?", (job["id"],)).fetchone()
            )
        job["status_message_id"] = anchor_id
        progress_bot = (
            PanelProgressBot(self.bot, self.repository, job) if job.get("panel_id") else self.bot
        )
        try:
            async with ProgressPulse(
                progress_bot,
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
            panel = get_panel(self.repository, job.get("panel_id")) if job.get("panel_id") else None
            reply_id = panel["message_id"] if panel else job["request_message_id"]
            reply = (
                ReplyParameters(message_id=reply_id, allow_sending_without_reply=True)
                if reply_id > 0
                else None
            )
            if job["output_kind"] == "tldr":
                panel = (
                    get_panel(self.repository, job.get("panel_id")) if job.get("panel_id") else None
                )
                rendered = summary_html(self.settings, self.repository, job["video_id"], result)
                if panel:
                    await edit_panel(
                        self.bot,
                        self.repository,
                        panel,
                        rendered,
                        view="summary",
                        summary_job_id=job["id"],
                    )
                    delivered_message_id = panel["message_id"]
                elif job["status_message_id"]:
                    markup = result_keyboard(self.repository, job)
                    await self.bot.edit_message_text(
                        chat_id=job["chat_id"],
                        message_id=job["status_message_id"],
                        text=rendered,
                        reply_markup=markup,
                        parse_mode="HTML",
                        link_preview_options=LinkPreviewOptions(is_disabled=True),
                    )
                    delivered_message_id = job["status_message_id"]
                else:
                    response = await self.bot.send_message(
                        chat_id=job["chat_id"],
                        text=rendered,
                        reply_markup=result_keyboard(self.repository, job),
                        parse_mode="HTML",
                        link_preview_options=LinkPreviewOptions(is_disabled=True),
                        reply_parameters=reply,
                    )
                    delivered_message_id = response.message_id
            else:
                async with ProgressPulse(
                    progress_bot,
                    chat_id=job["chat_id"],
                    message_id=lambda: job["status_message_id"],
                    kind="pdf",
                    elapsed=time.time() - job["created_at"],
                ) as pulse:
                    pulse.set_stage("delivery")
                    response = await self.bot.send_document(
                        chat_id=job["chat_id"],
                        document=FSInputFile(result, filename=f"PARS-{job['video_id']}.pdf"),
                        reply_parameters=reply,
                        reply_markup=result_keyboard(self.repository, job),
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
        register_result_message(
            self.repository, job, delivered_message_id, panel_id=job.get("panel_id")
        )
        if job["output_kind"] == "pdf" and job.get("panel_id"):
            panel = get_panel(self.repository, job["panel_id"])
            with suppress(Exception):
                if panel.get("summary_job_id"):
                    with self.repository.connection() as db:
                        saved = dict(
                            db.execute(
                                "SELECT * FROM video_pdf_jobs WHERE id=?",
                                (panel["summary_job_id"],),
                            ).fetchone()
                        )
                    await show_cached_summary(
                        self.bot, self.settings, self.repository, self.service, saved, panel
                    )
                else:
                    await edit_panel(
                        self.bot, self.repository, panel, panel["original_html"], view="card"
                    )
        # Delivery is already recorded. Cosmetic cleanup must never resend the PDF.
        if (
            job["output_kind"] == "pdf"
            and isinstance(job["status_message_id"], int)
            and job["status_message_id"] > 0
            and not job.get("panel_id")
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
            if job.get("panel_id"):
                panel = get_panel(self.repository, job["panel_id"])
                if panel:
                    await edit_panel(
                        self.bot,
                        self.repository,
                        panel,
                        panel["original_html"] + "\n\n" + text,
                        view="card",
                    )
                    return
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
