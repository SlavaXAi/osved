"""One durable video card, output navigation and result usefulness controls."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime

from aiogram.exceptions import TelegramBadRequest
from aiogram.filters.callback_data import CallbackData
from aiogram.types import LinkPreviewOptions
from aiogram.utils.keyboard import InlineKeyboardBuilder

from app.video.transcripts import youtube_video_id


class VideoPanelControl(CallbackData, prefix="vp"):
    panel_id: int
    action: str


class VideoResultControl(CallbackData, prefix="vr"):
    job_id: int
    reaction: str


def get_panel(repository, panel_id):
    with repository.connection() as db:
        row = db.execute("SELECT * FROM video_panels WHERE id=?", (panel_id,)).fetchone()
    return dict(row) if row else None


def get_panel_for_message(repository, *, chat_id, message_id):
    with repository.connection() as db:
        row = db.execute(
            "SELECT * FROM video_panels WHERE chat_id=? AND message_id=?",
            (chat_id, message_id),
        ).fetchone()
    return dict(row) if row else None


def ensure_panel(
    repository, *, chat_id, message_id, video_id, original_html, recommendation_id=None
):
    video_id = youtube_video_id(video_id)
    now = datetime.now(UTC).isoformat()
    with repository.connection() as db:
        db.execute(
            "INSERT OR IGNORE INTO video_panels(chat_id,message_id,video_id,"
            "recommendation_id,original_html,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
            (chat_id, message_id, video_id, recommendation_id, original_html, now, now),
        )
        row = db.execute(
            "SELECT * FROM video_panels WHERE chat_id=? AND message_id=?", (chat_id, message_id)
        ).fetchone()
        if row["video_id"] != video_id:
            raise ValueError("Panel video identity mismatch")
    return dict(row)


def bind_panel_job(repository, panel_id, job_id, *, loading=True):
    with repository.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        panel = db.execute("SELECT * FROM video_panels WHERE id=?", (panel_id,)).fetchone()
        job = db.execute("SELECT * FROM video_pdf_jobs WHERE id=?", (job_id,)).fetchone()
        if (
            not panel
            or not job
            or panel["chat_id"] != job["chat_id"]
            or panel["video_id"] != job["video_id"]
            or job["user_id"] != panel["chat_id"]
        ):
            raise PermissionError("Foreign panel/job")
        if job["panel_id"] not in (None, panel_id) and job["status"] in (
            "queued",
            "running",
            "sending",
        ):
            return False
        if job["panel_id"] is None or job["status"] not in ("queued", "running", "sending"):
            db.execute("UPDATE video_pdf_jobs SET panel_id=? WHERE id=?", (panel_id, job_id))
        field = "summary_job_id" if job["output_kind"] == "tldr" else "pdf_job_id"
        db.execute(
            f"UPDATE video_panels SET {field}=?,active_job_id=?,view=?,updated_at=? WHERE id=?",
            (
                job_id,
                job_id if loading else None,
                "loading" if loading else panel["view"],
                datetime.now(UTC).isoformat(),
                panel_id,
            ),
        )
    return True


def panel_busy(repository, panel):
    with repository.connection() as db:
        row = db.execute(
            "SELECT status FROM video_pdf_jobs WHERE id=?", (panel.get("active_job_id"),)
        ).fetchone()
    return bool(row and row[0] in ("queued", "running", "sending"))


def source_title(settings, repository, video_id):
    path = settings.data_dir / "video-reports" / video_id / "transcript.json"
    if path.is_file():
        try:
            raw = json.loads(path.read_text())
            if isinstance(raw, dict):
                raw = raw.get("output")
            if (
                isinstance(raw, list)
                and raw
                and isinstance(raw[0], dict)
                and raw[0].get("title")
                and raw[0].get("video_id") in (None, video_id)
            ):
                return str(raw[0]["title"])[:500]
        except (OSError, ValueError):
            pass
    with repository.connection() as db:
        row = db.execute(
            "SELECT original_text FROM source_items WHERE external_id=? ORDER BY id DESC LIMIT 1",
            (video_id,),
        ).fetchone()
    return str(row[0]).split("\n", 1)[0][:500] if row and row[0] else "Краткая сводка видео"


def panel_keyboard(repository, panel):
    from app.video.result_feedback import get_result_feedback

    builder = InlineKeyboardBuilder()
    if panel["view"] == "summary" and panel.get("summary_job_id"):
        current = get_result_feedback(
            repository, job_id=panel["summary_job_id"], chat_id=panel["chat_id"]
        )
        for reaction, label in [("like", "👍 Полезно"), ("dislike", "👎 Не полезно")]:
            builder.button(
                text=("✓ " if current == reaction else "") + label,
                callback_data=VideoResultControl(job_id=panel["summary_job_id"], reaction=reaction),
            )
        builder.adjust(2)
    menu = InlineKeyboardBuilder()
    for action, label in [("short", "Краткая сводка"), ("pdf", "Подробный отчёт · PDF")]:
        menu.button(
            text=label, callback_data=VideoPanelControl(panel_id=panel["id"], action=action)
        )
    menu.adjust(2)
    builder.attach(menu)
    tail = InlineKeyboardBuilder()
    tail.button(text="Открыть видео", url="https://www.youtube.com/watch?v=" + panel["video_id"])
    if panel["view"] == "summary":
        tail.button(
            text="← К описанию",
            callback_data=VideoPanelControl(panel_id=panel["id"], action="card"),
        )
    elif panel["view"] == "card" and panel.get("recommendation_id"):
        tail.button(
            text="Неинтересно",
            callback_data=VideoPanelControl(panel_id=panel["id"], action="dismiss"),
        )
    tail.adjust(1)
    builder.attach(tail)
    return builder.as_markup()


def result_keyboard(repository, job, *, message_id=None):
    from app.video.result_feedback import get_result_feedback

    target = message_id if message_id is not None else job.get("text_message_id")
    if job["output_kind"] == "tldr" and target is not None:
        panel = get_panel_for_message(repository, chat_id=job["chat_id"], message_id=target)
        if panel:
            if (
                panel["video_id"] != job["video_id"]
                or panel["view"] != "summary"
                or panel["summary_job_id"] != job["id"]
            ):
                raise PermissionError("This panel is not displaying the requested result")
            return panel_keyboard(repository, panel)
    current = get_result_feedback(repository, job_id=job["id"], chat_id=job["chat_id"])
    b = InlineKeyboardBuilder()
    for reaction, label in [("like", "👍 Полезно"), ("dislike", "👎 Не полезно")]:
        b.button(
            text=("✓ " if reaction == current else "") + label,
            callback_data=VideoResultControl(job_id=job["id"], reaction=reaction),
        )
    b.adjust(2)
    nav = InlineKeyboardBuilder()
    if job["output_kind"] == "tldr":
        nav.button(text="Подробный отчёт · PDF", callback_data=f"vreport:long:{job['id']}")
    nav.adjust(1)
    b.attach(nav)
    return b.as_markup()


def register_result_message(repository, job, message_id, *, panel_id=None):
    with repository.connection() as db:
        db.execute(
            "INSERT OR IGNORE INTO video_result_messages VALUES(?,?,?,?,?,?)",
            (
                job["id"],
                job["chat_id"],
                message_id,
                job["output_kind"],
                panel_id,
                datetime.now(UTC).isoformat(),
            ),
        )


async def edit_panel(bot, repository, panel, text, *, view, summary_job_id=None):
    updated = {**panel, "view": view}
    if summary_job_id is not None:
        updated["summary_job_id"] = summary_job_id
    try:
        await bot.edit_message_text(
            chat_id=panel["chat_id"],
            message_id=panel["message_id"],
            text=text,
            parse_mode="HTML",
            reply_markup=panel_keyboard(repository, updated),
            link_preview_options=LinkPreviewOptions(is_disabled=True),
        )
    except TelegramBadRequest as exc:
        if "message is not modified" not in str(exc):
            raise
    with repository.connection() as db:
        db.execute(
            "UPDATE video_panels SET view=?,summary_job_id=?,active_job_id=NULL,updated_at=? "
            "WHERE id=?",
            (view, updated["summary_job_id"], datetime.now(UTC).isoformat(), panel["id"]),
        )


class PanelProgressBot:
    """Keep menu and identity while animating a bound panel; other calls pass through."""

    def __init__(self, bot, repository, job):
        self.bot, self.repository, self.job = bot, repository, job

    def __getattr__(self, name):
        return getattr(self.bot, name)

    async def edit_message_text(self, **kwargs):
        panel = get_panel(self.repository, self.job["panel_id"])
        if not panel or panel["active_job_id"] != self.job["id"] or panel["view"] != "loading":
            return None
        kwargs["reply_markup"] = panel_keyboard(self.repository, panel)
        heading = panel["original_html"].split("\n\n", 1)[0]
        if len(heading) < 600:
            kwargs["text"] = heading + "\n\n" + kwargs["text"]
        return await self.bot.edit_message_text(**kwargs)


def summary_html(settings, repository, video_id, text):
    from app.video.tldr import telegram_tldr

    try:
        return telegram_tldr(
            text, title=source_title(settings, repository, video_id), video_id=video_id
        )
    except ValueError:
        return telegram_tldr(text, video_id=video_id)


async def show_cached_summary(bot, settings, repository, service, job, panel):

    text = service.cached_tldr(job["video_id"])
    if text is None:
        return False
    old_message = job.get("text_message_id")
    rendered = summary_html(settings, repository, job["video_id"], text)
    await edit_panel(bot, repository, panel, rendered, view="summary", summary_job_id=job["id"])
    if old_message and old_message > 0:
        register_result_message(repository, job, old_message)
    register_result_message(repository, job, panel["message_id"], panel_id=panel["id"])
    with repository.connection() as db:
        db.execute(
            "UPDATE video_pdf_jobs SET text_message_id=?,panel_id=?,status='sent',error=NULL,"
            "updated_at=? WHERE id=?",
            (panel["message_id"], panel["id"], time.time(), job["id"]),
        )
    return True
