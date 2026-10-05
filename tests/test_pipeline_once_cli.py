import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

import app.cli as cli
from app.config import Settings
from app.ops.schedule import ScheduledJobExecutor
from tests.test_radar_feedback import discovery_delivery, repository


def test_explicit_pipeline_uses_shared_steps_and_preserves_limits(
    tmp_path: Path, monkeypatch
) -> None:
    repo = repository(tmp_path)
    settings = Settings(_env_file=None, data_dir=tmp_path, allowed_user_id=123)
    monkeypatch.setattr(cli, "bootstrap", lambda: (settings, repo))
    calls = []

    def radar(self, *, provider, limit_per_source):
        calls.append(("radar", provider, limit_per_source))
        return {"cost_usd": 0.01, "errors": 0}

    def research(self):
        calls.append(("research",))
        return {"events_seen": 1, "cards_created": 1, "cost_usd": 0.02, "errors": 0}

    def delivery(self, *, limit):
        calls.append(("delivery", limit))
        return {"sent": 1, "errors": 0}

    monkeypatch.setattr(ScheduledJobExecutor, "radar", radar)
    monkeypatch.setattr(ScheduledJobExecutor, "research", research)
    monkeypatch.setattr(ScheduledJobExecutor, "delivery", delivery)
    result = CliRunner().invoke(
        cli.app,
        [
            "pipeline-once",
            "--collect",
            "--provider",
            "monid",
            "--collect-limit",
            "7",
            "--generate-limit",
            "2",
            "--deliver-limit",
            "3",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert calls == [("radar", "monid", 7), ("research",), ("research",), ("delivery", 3)]
    assert payload["cards_created"] == 2
    assert payload["cost_usd"] == pytest.approx(0.05)
    assert payload["scheduled"] is False
    assert repo.get_runtime_setting("schedule_enabled") is None


def test_no_deliver_and_unsuccessful_research_stop_repeated_spend(
    tmp_path: Path, monkeypatch
) -> None:
    repo = repository(tmp_path)
    settings = Settings(_env_file=None, data_dir=tmp_path)
    monkeypatch.setattr(cli, "bootstrap", lambda: (settings, repo))
    calls = []

    def research(self):
        calls.append("research")
        return {"events_seen": 1, "cards_created": 0, "errors": 0, "cost_usd": 0.03}

    def forbidden(*args, **kwargs):
        raise AssertionError("Collection or delivery was not requested")

    monkeypatch.setattr(ScheduledJobExecutor, "research", research)
    monkeypatch.setattr(ScheduledJobExecutor, "radar", forbidden)
    monkeypatch.setattr(ScheduledJobExecutor, "delivery", forbidden)
    result = CliRunner().invoke(
        cli.app, ["pipeline-once", "--no-deliver", "--generate-limit", "10"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert calls == ["research"]
    assert payload["radar"]["reason"] == "--no-collect"
    assert payload["delivery"]["reason"] == "--no-deliver"


def test_pipeline_failure_is_accounted_and_delivery_is_skipped(tmp_path: Path, monkeypatch) -> None:
    repo = repository(tmp_path)
    settings = Settings(_env_file=None, data_dir=tmp_path)
    monkeypatch.setattr(cli, "bootstrap", lambda: (settings, repo))

    def research(self):
        repo.record_usage("test", "research", cost_usd=0.04)
        raise RuntimeError("local persistence failed")

    monkeypatch.setattr(ScheduledJobExecutor, "research", research)
    result = CliRunner().invoke(cli.app, ["pipeline-once"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "failed"
    assert payload["delivery"]["reason"] == "upstream_failure"
    assert payload["cost_usd"] == pytest.approx(0.04)
    with repo.connection() as db:
        run = dict(db.execute("SELECT * FROM runs WHERE id=?", (payload["run_id"],)).fetchone())
    assert run["status"] == "failed"
    assert run["finished_at"]


def test_pipeline_real_delivery_excludes_stale_valid_and_fresh_preview(
    tmp_path: Path, monkeypatch
) -> None:
    repo = repository(tmp_path)
    repo.set_runtime_setting("scheduled_video_recommendations_enabled", "false")
    settings = Settings(_env_file=None, data_dir=tmp_path, allowed_user_id=123)
    monkeypatch.setattr(cli, "bootstrap", lambda: (settings, repo))
    discovery_delivery(repo, index=1)
    discovery_delivery(repo, index=2)
    with repo.connection() as db:
        db.execute("DELETE FROM deliveries")
        rows = db.execute("SELECT id FROM events ORDER BY id").fetchall()
        db.execute(
            "UPDATE cards SET validation_status='valid',structured_output_json='{}' "
            "WHERE event_id=?",
            (rows[0][0],),
        )
        db.execute(
            "UPDATE events SET event_at=? WHERE id=?",
            ((datetime.now(UTC) - timedelta(days=8)).isoformat(), rows[0][0]),
        )
        db.execute(
            "UPDATE events SET event_at=? WHERE id=?", (datetime.now(UTC).isoformat(), rows[1][0])
        )
    monkeypatch.setattr(
        ScheduledJobExecutor,
        "research",
        lambda self: {
            "events_seen": 0,
            "cards_created": 0,
            "errors": 0,
            "cost_usd": 0,
        },
    )
    result = CliRunner().invoke(cli.app, ["pipeline-once"])
    assert result.exit_code == 0, result.output
    delivery = json.loads(result.output)["delivery"]
    assert delivery["status"] == "no_fresh_cards"
    assert delivery["sent"] == delivery["news_sent"] == delivery["video_sent"] == 0
    assert delivery["cost_usd"] == delivery["errors"] == 0
