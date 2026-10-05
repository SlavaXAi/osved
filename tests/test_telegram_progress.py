from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.exceptions import TelegramRetryAfter
from aiogram.methods import EditMessageText

from app.telegram_bot.progress import FRAMES, ProgressPulse, progress_text


def test_signal_is_restrained_with_real_stage_not_fake_percentage():
    text = progress_text("tldr", "transcript", elapsed=64)
    assert "Получаю субтитры" in text and "1:04" in text
    assert "<code>" in text and "%" not in text and "⏳" not in text
    assert progress_text("pdf", "pdf", frame=1) != progress_text("pdf", "pdf", frame=2)
    assert len(set(FRAMES)) > 4
    assert "0:00" in progress_text("pdf", "queued", elapsed=-1)


def test_pulse_stops_before_final_result_and_never_edits_parent():
    async def run():
        changed = asyncio.Event()
        edits = []

        async def edit(**kwargs):
            edits.append(kwargs)
            changed.set()

        pulse = ProgressPulse(
            SimpleNamespace(edit_message_text=edit),
            chat_id=7,
            message_id=lambda: 88,
            kind="tldr",
            interval=0.001,
        )
        async with pulse:
            pulse.set_stage("writing")
            await asyncio.wait_for(changed.wait(), timeout=1)
        count = len(edits)
        await asyncio.sleep(0.01)
        assert len(edits) == count
        assert edits[-1]["message_id"] == 88
        assert "Собираю связный текст" in edits[-1]["text"]
        assert all("reply_markup" not in x for x in edits)
        assert pulse.task.done()

    asyncio.run(run())


def test_retry_after_backs_off_without_spinning(monkeypatch):
    delays = []

    async def sleep(delay):
        delays.append(delay)
        if len(delays) == 2:
            raise asyncio.CancelledError

    monkeypatch.setattr("app.telegram_bot.progress.asyncio.sleep", sleep)
    bot = SimpleNamespace(
        edit_message_text=AsyncMock(
            side_effect=TelegramRetryAfter(
                method=EditMessageText(text="x"), message="Too many requests", retry_after=20
            )
        )
    )
    pulse = ProgressPulse(bot, chat_id=7, message_id=lambda: 88, kind="pdf")
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(pulse._run())
    assert delays == [3, 21]
    assert bot.edit_message_text.await_count == 1


def test_detail_pulse_precedes_final_edit_in_the_same_loop(monkeypatch):
    from app.telegram_bot import bot as module

    order = []

    def step(name):
        async def invoke(*_args):
            order.append(name)

        return invoke

    monkeypatch.setattr(module, "DetailProgress", lambda: SimpleNamespace(tick=step("pulse")))
    monkeypatch.setattr(module, "send_detail_placeholders", step("placeholder"))
    monkeypatch.setattr(module, "send_ready_detail_articles", step("final"))
    monkeypatch.setattr(module, "send_terminal_detail_results", step("terminal"))
    monkeypatch.setattr(module, "sync_detail_keyboards", step("buttons"))

    async def stop(_delay):
        raise asyncio.CancelledError

    monkeypatch.setattr(module.asyncio, "sleep", stop)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(module.run_detail_delivery_loop(None, None))
    assert order == ["placeholder", "pulse", "final", "terminal", "buttons"]
