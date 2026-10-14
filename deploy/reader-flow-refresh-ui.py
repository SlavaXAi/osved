"""Refresh owner controls and existing paid summaries in place; never send or regenerate."""

import argparse
import asyncio
import json

from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.types import LinkPreviewOptions

from app.config import Settings
from app.db.repository import Repository
from app.telegram_bot.bot import build_bot, configured_owner_id
from app.telegram_bot.sensory import video_keyboard
from app.telegram_bot.video_panel import (
    get_panel_for_message,
    panel_keyboard,
    register_result_message,
    result_keyboard,
    summary_html,
)
from app.video.reports import VideoReportService

PACE_SECONDS = 1.2
MAX_INLINE_RETRY_SECONDS = 30


class RefreshRateLimited(Exception):
    def __init__(self, retry_after):
        self.retry_after = retry_after
        super().__init__("Telegram rate limit")


async def main(*, results_only=False):
    settings = Settings()
    repo = Repository(settings.database_path)
    owner = configured_owner_id(settings, repo)
    assert owner is not None
    service = VideoReportService(settings, repo)
    with repo.connection() as db:
        recs = [] if results_only else [
            dict(r)
            for r in db.execute(
                "SELECT v.*,i.source_url,(SELECT f.action FROM video_feedback_events f "
                "WHERE f.recommendation_id=v.id AND f.action IN ('open','interest','dismiss') "
                "ORDER BY f.id DESC LIMIT 1) AS last_choice FROM video_recommendations v "
                "JOIN source_items i ON i.id=v.source_item_id "
                "WHERE v.chat_id=? AND v.status='sent' ORDER BY v.id DESC LIMIT 100",
                (owner,),
            )
        ]
        jobs = [
            dict(r)
            for r in db.execute(
                "SELECT * FROM video_pdf_jobs WHERE chat_id=? "
                "AND status='sent' AND created_at>=strftime('%s','now','-7 days') ORDER BY id DESC",
                (owner,),
            )
        ]
    counts = {
        "recommendation_controls": 0,
        "summary_texts": 0,
        "pdf_controls": 0,
        "unchanged": 0,
        "unavailable": 0,
        "missing_cache": 0,
        "completed": True,
        "retry_after": 0,
        "results_only": results_only,
    }
    bot = build_bot(settings)

    async def edit(fn, **kwargs):
        try:
            try:
                await fn(**kwargs)
            except TelegramRetryAfter as exc:
                if exc.retry_after > MAX_INLINE_RETRY_SECONDS:
                    raise RefreshRateLimited(exc.retry_after) from None
                await asyncio.sleep(max(exc.retry_after, PACE_SECONDS))
                await fn(**kwargs)
            return "changed"
        except TelegramRetryAfter as exc:
            # A second refusal stops the batch too; never keep probing a chat
            # whose Telegram cooldown has not cleared.
            raise RefreshRateLimited(exc.retry_after) from None
        except TelegramBadRequest as exc:
            status = "unchanged" if "message is not modified" in str(exc) else "unavailable"
            counts[status] += 1
            return status

    try:
        for row in recs:
            with repo.connection() as db:
                panel = db.execute(
                    "SELECT * FROM video_panels WHERE chat_id=? AND message_id=?",
                    (owner, row["telegram_message_id"]),
                ).fetchone()
            markup = (
                panel_keyboard(repo, dict(panel))
                if panel
                else video_keyboard(
                    row["id"],
                    expanded=row["last_choice"] in ("open", "interest"),
                    url=row["source_url"],
                )
            )
            result = await edit(
                bot.edit_message_reply_markup,
                chat_id=owner,
                message_id=row["telegram_message_id"],
                reply_markup=markup,
            )
            if result == "changed":
                counts["recommendation_controls"] += 1
            await asyncio.sleep(PACE_SECONDS)
        for job in jobs:
            message_id = (
                job["text_message_id"]
                if job["output_kind"] == "tldr"
                else job["document_message_id"]
            )
            if not isinstance(message_id, int) or message_id <= 0:
                continue
            if job["output_kind"] == "tldr":
                panel = get_panel_for_message(repo, chat_id=owner, message_id=message_id)
                if panel and (
                    panel["view"] != "summary"
                    or panel["summary_job_id"] != job["id"]
                    or panel["video_id"] != job["video_id"]
                ):
                    continue
                text = service.cached_tldr(job["video_id"])
                if text is None:
                    counts["missing_cache"] += 1
                    continue
                result = await edit(
                    bot.edit_message_text,
                    chat_id=owner,
                    message_id=message_id,
                    text=summary_html(settings, repo, job["video_id"], text),
                    parse_mode="HTML",
                    reply_markup=result_keyboard(repo, job, message_id=message_id),
                    link_preview_options=LinkPreviewOptions(is_disabled=True),
                )
                if result == "changed":
                    counts["summary_texts"] += 1
            else:
                result = await edit(
                    bot.edit_message_reply_markup,
                    chat_id=owner,
                    message_id=message_id,
                    reply_markup=result_keyboard(repo, job, message_id=message_id),
                )
                if result == "changed":
                    counts["pdf_controls"] += 1
            if result in {"changed", "unchanged"}:
                origin = get_panel_for_message(repo, chat_id=owner, message_id=message_id)
                register_result_message(
                    repo, job, message_id, panel_id=origin["id"] if origin else None
                )
            await asyncio.sleep(PACE_SECONDS)
    except RefreshRateLimited as exc:
        counts["completed"] = False
        counts["retry_after"] = exc.retry_after
    finally:
        await bot.session.close()
    print(json.dumps({**counts, "new_messages": 0, "paid_calls": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-only", action="store_true", help="Skip recommendation controls already refreshed"
    )
    arguments = parser.parse_args()
    asyncio.run(main(results_only=arguments.results_only))
