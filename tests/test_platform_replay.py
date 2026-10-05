import json
from datetime import UTC, datetime, timedelta

from app.config import SourceConfig, SourceRegistry
from app.db.repository import Repository
from app.ops.replay import replay_snapshot

NOW = datetime(2026, 9, 7, 12, tzinfo=UTC)


def setup(tmp_path):
    repository = Repository(tmp_path / "test.sqlite3")
    repository.migrate()
    source = SourceConfig(
        id="reddit_test",
        name="Reddit test",
        kind="reddit",
        locator="LocalLLaMA",
        adapter_id="reddit_monid",
        strategy="new",
        admission_mode="trigger_only",
        role="radar",
        region="global",
        language="en",
        enabled=False,
        options={"per_call_price_cap_usd": 0.0015},
    )
    repository.sync_sources(SourceRegistry(version=1, sources=[source]))
    path = tmp_path / "recorded.json"
    path.write_text(
        json.dumps(
            {
                "posts": [
                    {
                        "id": "abc123",
                        "title": "A new open source framework",
                        "text": "Real saved observation with source text.",
                        "created_at": NOW.isoformat(),
                        "author": "builder",
                    }
                ]
            }
        )
    )
    return repository, source, path


def test_replay_ingests_and_reconciles_once_without_enabling_or_live_success(tmp_path, monkeypatch):
    import app.collectors.platform_transport as transport

    monkeypatch.setattr(
        transport,
        "monid_fetch",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("Replay must never call Monid")),
    )
    repo, source, path = setup(tmp_path)
    first = replay_snapshot(
        repo, source, path, original_run_id="original123", import_charge_usd=0.0015, now=NOW
    )
    again = replay_snapshot(
        repo, source, path, original_run_id="original123", import_charge_usd=0.0015, now=NOW
    )
    assert first["status"] == "replayed"
    assert first["inserted"] == 1
    assert first["accounting_adjustment_usd"] == 0.0015
    assert first["network_calls"] == first["cost_usd"] == 0
    assert again["status"] == "already_replayed"
    assert again["inserted"] == again["accounting_adjustment_usd"] == 0
    with repo.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM source_items").fetchone()[0] == 1
        assert db.execute("SELECT observation_count FROM source_items").fetchone()[0] == 1
        saved = db.execute("SELECT enabled,last_success_at FROM sources").fetchone()
        assert not saved["enabled"]
        assert saved["last_success_at"] is None
        run = db.execute("SELECT status,strategy FROM source_runs").fetchone()
        assert run["status"] == "replayed"
        assert run["strategy"] == "snapshot_replay"
        assert db.execute("SELECT status FROM source_health").fetchone()[0] == "ready_from_replay"
        assert db.execute("SELECT SUM(cost_usd) FROM usage_ledger").fetchone()[0] == 0.0015
        assert (
            db.execute(
                "SELECT request_count FROM usage_ledger WHERE operation='sensor.snapshot_replay'"
            ).fetchone()[0]
            == 0
        )


def test_replay_does_not_duplicate_an_already_accounted_original_charge(tmp_path):
    repo, source, path = setup(tmp_path)
    repo.record_usage(
        "reddit_monid", "sensor.collect", cost_usd=0.0015, metadata={"source_run_id": "original123"}
    )
    result = replay_snapshot(
        repo, source, path, original_run_id="original123", import_charge_usd=0.0015, now=NOW
    )
    assert result["accounting_adjustment_usd"] == 0
    assert repo.current_month_cost() == 0.0015


def test_replay_respects_an_active_live_source_lease(tmp_path):
    repo, source, path = setup(tmp_path)
    with repo.connection() as db:
        db.execute(
            """INSERT INTO source_runs(id,source_id,adapter_id,strategy,status,
                   started_at,lease_until)
                   VALUES('live',?,?,?,'running',?,?)""",
            (
                source.id,
                source.adapter_id,
                source.strategy,
                NOW.isoformat(),
                (NOW + timedelta(minutes=5)).isoformat(),
            ),
        )
    result = replay_snapshot(repo, source, path, original_run_id="original123", now=NOW)
    assert result["status"] == "source_busy"
    assert repo.source_item_count() == 0


def test_invalid_snapshot_is_recorded_as_offline_failure_without_clearing_live_health(tmp_path):
    repo, source, path = setup(tmp_path)
    path.write_text('{"unknown": "schema"}')
    result = replay_snapshot(repo, source, path, original_run_id="original123", now=NOW)
    assert result["status"] == "replay_failed"
    assert result["inserted"] == result["cost_usd"] == 0
    with repo.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM source_health").fetchone()[0] == 0
