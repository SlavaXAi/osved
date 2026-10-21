from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from app.config import Settings
from app.db.repository import Repository
from app.ops import healthcheck, service


def environment(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, data_dir=tmp_path, telegram_bot_token=None)
    repository = Repository(settings.database_path)
    monkeypatch.setattr(healthcheck, "Settings", lambda: settings)
    return settings, repository


def test_healthcheck_missing_database_is_unhealthy_not_a_new_database(tmp_path, monkeypatch):
    settings, _ = environment(tmp_path, monkeypatch)
    assert healthcheck.check() is False
    assert not settings.database_path.exists()


@pytest.mark.parametrize(
    "timestamp",
    [
        None,
        "not-a-date",
        "2026-09-07T12:00:00",
        lambda: (datetime.now(UTC) - timedelta(minutes=16)).isoformat(),
        lambda: (datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
    ],
)
def test_healthcheck_rejects_absent_stale_invalid_or_future_heartbeat(
    tmp_path, monkeypatch, timestamp
):
    _, repository = environment(tmp_path, monkeypatch)
    repository.migrate()
    # Compute relative timestamps at execution, not collection: a full suite
    # can run for longer than the future offset.
    timestamp = timestamp() if callable(timestamp) else timestamp
    if timestamp is not None:
        repository.set_runtime_setting("service_last_tick", timestamp)
    assert healthcheck.check() is False


def test_healthcheck_accepts_fresh_heartbeat(tmp_path, monkeypatch):
    _, repository = environment(tmp_path, monkeypatch)
    repository.migrate()
    repository.set_runtime_setting("service_last_tick", datetime.now(UTC).isoformat())
    assert healthcheck.check() is True


def test_schedule_worker_records_heartbeat_without_external_calls(tmp_path, monkeypatch):
    settings, repository = environment(tmp_path, monkeypatch)
    repository.migrate()
    calls = []
    monkeypatch.setattr(service, "run_schedule_tick", lambda *args: calls.append(True))

    async def stop_after_one_tick(seconds):
        raise asyncio.CancelledError

    monkeypatch.setattr(service.asyncio, "sleep", stop_after_one_tick)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(service.schedule_worker(settings, repository))
    assert calls == [True]
    assert healthcheck.check() is True


def test_schedule_worker_redacts_error_and_does_not_fake_success(tmp_path, monkeypatch, caplog):
    settings, repository = environment(tmp_path, monkeypatch)
    repository.migrate()
    token = "sk-or-v1-" + "fake" * 12

    def failed_tick(*args):
        raise RuntimeError(f"Provider error {token}")

    async def stop_after_one_tick(seconds):
        raise asyncio.CancelledError

    monkeypatch.setattr(service, "run_schedule_tick", failed_tick)
    monkeypatch.setattr(service.asyncio, "sleep", stop_after_one_tick)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(service.schedule_worker(settings, repository))
    assert token not in caplog.text
    assert "[REDACTED]" in caplog.text
    assert repository.get_runtime_setting("service_last_tick") is None


def test_supervisor_cancels_other_worker_if_bot_fails(tmp_path, monkeypatch):
    settings, repository = environment(tmp_path, monkeypatch)
    repository.migrate()
    cancelled = []

    async def idle_schedule(*args):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    async def failed_bot(*args):
        raise RuntimeError("fixture bot startup failed")

    monkeypatch.setattr(service, "schedule_worker", idle_schedule)
    monkeypatch.setattr(service, "run_bot", failed_bot)
    with pytest.raises(RuntimeError, match="fixture bot startup failed"):
        asyncio.run(service.serve(settings, repository))
    assert cancelled == [True]


def test_supervisor_cancels_bot_when_schedule_stops(tmp_path, monkeypatch):
    settings, repository = environment(tmp_path, monkeypatch)
    repository.migrate()
    cancelled = []

    async def stopped_schedule(*args):
        return None

    async def idle_bot(*args):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    monkeypatch.setattr(service, "schedule_worker", stopped_schedule)
    monkeypatch.setattr(service, "run_bot", idle_bot)
    asyncio.run(service.serve(settings, repository))
    assert cancelled == [True]
    assert healthcheck.check() is True
