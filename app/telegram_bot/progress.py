"""Quiet, best-effort Telegram progress. No model calls or invented percentages."""

from __future__ import annotations

import asyncio
import html
import time
from collections.abc import Callable
from contextlib import suppress

from aiogram.exceptions import TelegramRetryAfter
from aiogram.types import LinkPreviewOptions

FRAMES = (
    "● ◦ · · · · ·",
    "◦ ● ◦ · · · ·",
    "· ◦ ● ◦ · · ·",
    "· · ◦ ● ◦ · ·",
    "· · · ◦ ● ◦ ·",
    "· · · · ◦ ● ◦",
    "· · · · · ◦ ●",
    "· · · · ◦ ● ◦",
    "· · · ◦ ● ◦ ·",
    "· · ◦ ● ◦ · ·",
    "· ◦ ● ◦ · · ·",
    "◦ ● ◦ · · · ·",
)
TITLES = {"tldr": "Краткая сводка", "pdf": "Подробный PDF", "details": "Подробный разбор"}
STAGES = {
    "queued": "Запрос в очереди",
    "preparing": "Проверяю сохранённые материалы",
    "cache": "Забираю готовый результат из кэша",
    "transcript": "Получаю субтитры",
    "writing": "Собираю связный текст",
    "formatting": "Проверяю структуру текста",
    "pdf": "Верстаю PDF",
    "planning": "Определяю, что нужно уточнить",
    "search": "Ищу дополнительные источники",
    "facts": "Сопоставляю факты и источники",
    "delivery": "Отправляю результат",
    "waiting_provider": "Источник ещё готовит субтитры · проверю снова",
}


def progress_text(kind: str, stage: str, *, frame: int = 0, elapsed: float = 0) -> str:
    seconds = max(0, int(elapsed))
    clock = f"{seconds // 60}:{seconds % 60:02d}"
    title = html.escape(TITLES.get(kind, "Подготовка материала"))
    label = html.escape(STAGES.get(stage, "Обрабатываю запрос"))
    return (
        f"<b>PARS · {title}</b>\n\n"
        f"<code>{FRAMES[frame % len(FRAMES)]}</code>\n{label}\n\n"
        f"<i>Прошло {clock}</i>"
    )


class ProgressPulse:
    """One editable status; stop awaits cancellation before a final response.

    The stage setter is called by a blocking worker thread and does no I/O.
    Telegram failures only slow/disable the animation, never repeat paid work.
    """

    def __init__(
        self,
        bot,
        *,
        chat_id: int,
        message_id: Callable[[], int | None],
        kind: str,
        elapsed: float = 0,
        interval: float = 3.0,
    ):
        self.bot, self.chat_id, self.message_id = bot, chat_id, message_id
        self.kind, self.interval = kind, interval
        self.stage = "preparing"
        self.started = time.monotonic() - max(0, elapsed)
        self.task: asyncio.Task | None = None

    def set_stage(self, stage: str) -> None:
        if stage in STAGES:
            self.stage = stage

    async def __aenter__(self):
        self.task = asyncio.create_task(self._run(), name="telegram-progress-pulse")
        return self

    async def __aexit__(self, *_args):
        if self.task is not None:
            self.task.cancel()
            with suppress(asyncio.CancelledError):
                await self.task
        return False

    async def _run(self):
        frame, failures, delay = 0, 0, self.interval
        while True:
            await asyncio.sleep(delay)
            try:
                message_id = self.message_id()
                if not isinstance(message_id, int) or message_id <= 0:
                    continue
                await self.bot.edit_message_text(
                    chat_id=self.chat_id,
                    message_id=message_id,
                    parse_mode="HTML",
                    text=progress_text(
                        self.kind,
                        self.stage,
                        frame=frame,
                        elapsed=time.monotonic() - self.started,
                    ),
                    link_preview_options=LinkPreviewOptions(is_disabled=True),
                )
                frame += 1
                failures, delay = 0, self.interval
            except TelegramRetryAfter as exc:
                delay = max(self.interval, float(exc.retry_after) + 1)
            except Exception:
                failures += 1
                if failures >= 3:
                    return
                delay = max(self.interval, 15.0)


class DetailProgress:
    """Shares the detail delivery loop, so a pulse can never overwrite its final text."""

    def __init__(self):
        self.frames: dict[int, int] = {}
        self.next_at: dict[int, float] = {}

    async def tick(self, bot, repository) -> None:
        from datetime import UTC, datetime

        with repository.connection() as db:
            rows = db.execute(
                "SELECT id,chat_id,placeholder_message_id,stage,requested_at,depth,delivery_id "
                "FROM detail_requests WHERE status='running' "
                "AND placeholder_message_id>0 ORDER BY id LIMIT 5"
            ).fetchall()
        active = {row["id"] for row in rows}
        self.frames = {k: v for k, v in self.frames.items() if k in active}
        self.next_at = {k: v for k, v in self.next_at.items() if k in active}
        for row in rows:
            key = row["id"]
            now = time.monotonic()
            if self.next_at.get(key, 0) > now:
                continue
            self.next_at[key] = now + 4
            try:
                stamp = datetime.fromisoformat(row["requested_at"].replace("Z", "+00:00"))
                if stamp.tzinfo is None:
                    stamp = stamp.replace(tzinfo=UTC)
                elapsed = (datetime.now(UTC) - stamp).total_seconds()
                frame = self.frames.get(key, 0)
                if row["depth"] != "legacy":
                    from app.telegram_bot.news_panel import update_loading

                    await update_loading(
                        bot,
                        repository,
                        row["delivery_id"],
                        key,
                        progress_text("details", row["stage"], frame=frame, elapsed=elapsed),
                    )
                    self.frames[key] = frame + 1
                    break
                await bot.edit_message_text(
                    chat_id=row["chat_id"],
                    message_id=row["placeholder_message_id"],
                    parse_mode="HTML",
                    text=progress_text(
                        "details",
                        row["stage"],
                        frame=frame,
                        elapsed=elapsed,
                    ),
                    link_preview_options=LinkPreviewOptions(is_disabled=True),
                )
                self.frames[key] = frame + 1
            except TelegramRetryAfter as exc:
                self.next_at[key] = now + max(4, float(exc.retry_after) + 1)
                break
            except Exception:
                self.next_at[key] = now + 30
            # Keep cosmetic I/O bounded even if concurrency is raised later.
            break
