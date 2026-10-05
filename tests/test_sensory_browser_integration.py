from __future__ import annotations

from datetime import UTC, datetime

from app.collectors.browser_adapter import BrowserSourceAdapter
from app.collectors.browser_queue import BrowserTaskQueue
from app.collectors.orchestrator import collect_due_sources
from app.config import Settings, SourceConfig, SourceRegistry
from app.db.repository import Repository


def test_browser_snapshot_ingest_survives_restart_limits_and_replay(tmp_path, monkeypatch):
    now = datetime(2026, 9, 7, 12, tzinfo=UTC)
    settings = Settings(app_env="test", data_dir=tmp_path, _env_file=None)
    repository = Repository(settings.database_path)
    repository.migrate()
    source = SourceConfig(
        id="x_for_you",
        name="Visible X",
        kind="x",
        locator="https://x.com/home",
        role="radar",
        region="global",
        language="en",
        adapter_id="browser_x",
        strategy="for_you",
        admission_mode="trigger_only",
        poll_interval_minutes=720,
    )
    registry = SourceRegistry(version=1, sources=[source])
    monkeypatch.setattr("app.collectors.orchestrator.runtime_source_registry", lambda _: registry)
    queue = BrowserTaskQueue(settings.database_path, clock=now.timestamp)
    adapter = BrowserSourceAdapter(queue)

    # A new source must enqueue despite not yet having an authorized browser result.
    first = collect_due_sources(
        settings,
        repository,
        adapters={"browser_x": adapter},
        now=now,
        limit_per_source=3,
    )
    assert first["items_inserted"] == 0
    claim = queue.claim()
    assert claim is not None
    snapshot = {
        "feed_mode": "for_you",
        "posts": [
            {
                "id": f"12345678{index}",
                "author": "example",
                "text": f"Agent release {index}",
                "published_at": "2026-09-07T08:00:00Z",
                "links": [],
            }
            for index in range(3)
        ],
    }
    queue.stage(claim["id"], claim["claim_token"], snapshot)

    # Restart both queue/adapter and lower request limit: the old snapshot is not truncated.
    queue = BrowserTaskQueue(settings.database_path, clock=now.timestamp)
    adapter = BrowserSourceAdapter(queue)
    collected = collect_due_sources(
        settings,
        repository,
        adapters={"browser_x": adapter},
        now=now,
        limit_per_source=1,
        force=True,
    )
    assert collected["items_inserted"] == 3
    assert queue.task_status(claim["id"])["status"] == "completed"
    with repository.connection() as connection:
        rows = connection.execute("SELECT * FROM source_items").fetchall()
    assert len(rows) == 3
    assert all(row["discovery_mode"] == "for_you" for row in rows)
    assert all("visible_posts_only" in row["raw_payload_json"] for row in rows)

    # Re-delivery of the same posts under a later task stays idempotent.
    replay_id = queue.enqueue(source.id, task_key="replay", max_items=3)
    replay = queue.claim()
    assert replay["id"] == replay_id
    queue.stage(replay_id, replay["claim_token"], snapshot)
    collected = collect_due_sources(
        settings,
        repository,
        adapters={"browser_x": adapter},
        now=now,
        force=True,
    )
    assert collected["items_inserted"] == 0
    assert queue.task_status(replay_id)["status"] == "completed"
    with repository.connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM source_items").fetchone()[0] == 3
    # Changing the manual collection limit during the same cadence window must
    # not conflict with that window's immutable task payload.
    already_captured = collect_due_sources(
        settings,
        repository,
        adapters={"browser_x": adapter},
        now=now,
        limit_per_source=10,
        force=True,
    )
    assert already_captured["errors"] == 0
    assert already_captured["items_inserted"] == 0


def test_browser_ingest_error_keeps_staged_result_for_retry(tmp_path, monkeypatch):
    now = datetime(2026, 9, 7, 12, tzinfo=UTC)
    settings = Settings(app_env="test", data_dir=tmp_path, _env_file=None)
    repository = Repository(settings.database_path)
    repository.migrate()
    source = SourceConfig(
        id="x_for_you",
        name="Visible X",
        kind="x",
        locator="https://x.com/home",
        role="radar",
        region="global",
        language="en",
        adapter_id="browser_x",
        strategy="for_you",
        admission_mode="trigger_only",
    )
    monkeypatch.setattr(
        "app.collectors.orchestrator.runtime_source_registry",
        lambda _: SourceRegistry(version=1, sources=[source]),
    )
    queue = BrowserTaskQueue(settings.database_path, clock=now.timestamp)
    task_id = queue.enqueue(source.id, task_key="database-failure")
    task = queue.claim()
    queue.stage(
        task_id,
        task["claim_token"],
        {
            "feed_mode": "for_you",
            "posts": [
                {
                    "id": "1234567890",
                    "author": "example",
                    "text": "AI release",
                    "published_at": "2026-09-07T08:00:00Z",
                    "links": [],
                }
            ],
        },
    )

    def fail_ingest(*args, **kwargs):
        raise RuntimeError("simulated database interruption")

    monkeypatch.setattr("app.collectors.orchestrator.ingest_item", fail_ingest)
    result = collect_due_sources(
        settings,
        repository,
        now=now,
        adapters={
            "browser_x": BrowserSourceAdapter(queue),
        },
    )
    assert result["errors"] == 1
    assert queue.task_status(task_id)["status"] == "staged"
