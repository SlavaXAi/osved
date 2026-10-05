"""Single process supervising Telegram and the bounded scheduled pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime

from app.collectors.ingest import redact_text
from app.config import Settings
from app.db.repository import Repository
from app.ops.feed_health import feed_health
from app.ops.schedule import run_schedule_tick
from app.telegram_bot.bot import run_bot

logger = logging.getLogger(__name__)


async def resume_waiting_reader(settings: Settings, repository: Repository) -> None:
    """A replenished queue resumes after the reader's rating, not hours later."""
    from app.telegram_bot.bot import configured_owner_id
    from app.telegram_bot.news_queue import current_news, queue_enabled, send_next_news

    if (
        not queue_enabled(repository)
        or repository.get_runtime_setting("schedule_enabled") != "true"
        or repository.get_runtime_setting("schedule_dry_run") != "false"
    ):
        return
    owner = configured_owner_id(settings, repository)
    current = current_news(repository, owner) if owner is not None else None
    if current and repository.current_reaction(current["id"]):
        await send_next_news(
            settings,
            repository,
            after_delivery_id=current["id"],
            limit=settings.news_extra_batch_size,
        )


async def schedule_worker(settings: Settings, repository: Repository) -> None:
    while True:
        try:
            await asyncio.to_thread(run_schedule_tick, settings, repository)
            await resume_waiting_reader(settings, repository)
            repository.set_runtime_setting(
                "service_last_tick",
                datetime.now(UTC).isoformat(),
            )
            snapshot = feed_health(repository)
            previous = repository.get_runtime_setting("feed_health_status")
            if previous != snapshot["status"]:
                logger.warning("News feed product health changed: %s", snapshot)
                repository.set_runtime_setting("feed_health_status", snapshot["status"])
            repository.set_runtime_setting("feed_health", json.dumps(snapshot))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Schedule tick failed: %s", redact_text(str(exc))[:500])
        await asyncio.sleep(30)


async def serve(settings: Settings, repository: Repository) -> None:
    # Initial liveness marker: the first bounded radar may take several minutes.
    # Only successful ticks refresh it afterward; a stuck/erroring scheduler
    # therefore becomes unhealthy after the existing fifteen-minute threshold.
    # This marker is not a claim that a scheduled job completed successfully.
    repository.set_runtime_setting("service_last_tick", datetime.now(UTC).isoformat())
    tasks = [
        asyncio.create_task(schedule_worker(settings, repository), name="schedule"),
        asyncio.create_task(run_bot(settings, repository), name="telegram"),
    ]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
