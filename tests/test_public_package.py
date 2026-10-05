"""Exercise the actual public defaults, independently of regression fixtures."""

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from app import cli
from app.config import Settings, runtime_source_registry
from app.ops.schedule import due_slots, run_schedule_tick
from app.radar.telegram_channels import add_channel

ROOT = Path(__file__).parents[1]


def public_settings(tmp_path):
    return Settings(
        _env_file=ROOT / ".env.example",
        data_dir=tmp_path,
        telegram_session_path=tmp_path / "telegram-reader.session",
        sources_file=ROOT / "config/sources.yaml",
        interests_file=ROOT / "config/interests.yaml",
        style_profile_file=ROOT / "config/style-profile.yaml",
        sensory_sources_file=ROOT / "config/sensory-sources.yaml",
        contours_file=ROOT / "config/contours.yaml",
        search_agents_file=ROOT / "config/search-agents.yaml",
        x_authors_file=ROOT / "config/x-authors.yaml",
        weekly_personas_file=ROOT / "config/weekly-personas.yaml",
        frontier_ideas_file=ROOT / "config/frontier-ideas.yaml",
    )


def test_fresh_public_install_has_no_subscriptions_or_paid_background_jobs(tmp_path, monkeypatch):
    for key in ("TELEGRAM_BOT_TOKEN", "OPENROUTER_API_KEY", "MONID_API_KEY", "ALLOWED_USER_ID"):
        monkeypatch.delenv(key, raising=False)
    settings = public_settings(tmp_path)
    monkeypatch.setattr(cli, "Settings", lambda: settings)
    result = CliRunner().invoke(cli.app, ["doctor"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert not any(report["credentials"].values())
    assert report["storage"]["sources"] == 0
    settings, repository = cli.bootstrap()
    assert repository.get_runtime_setting("news_delivery_mode") == "hybrid"
    calls = []
    outcome = run_schedule_tick(settings, repository, executor=lambda job: calls.append(job))
    assert outcome["status"] == "disabled"
    assert calls == []
    assert settings.radar_collect_limit == 100
    assert settings.scheduled_priority_lenses == []


def test_owner_added_channel_survives_neutral_config_restart(tmp_path, monkeypatch):
    settings = public_settings(tmp_path).model_copy(update={"allowed_user_id": 123})
    monkeypatch.setattr(cli, "Settings", lambda: settings)
    _, repository = cli.bootstrap()
    channel = SimpleNamespace(
        id=-1001234567890, username="friends_channel", title="My channel", type="channel"
    )
    added = add_channel(settings, repository, channel, 123)
    _, restarted = cli.bootstrap()
    sources = runtime_source_registry(settings).sources
    assert len(sources) == 1
    assert sources[0].id == added["source_id"]
    assert sources[0].locator == "friends_channel"
    assert restarted.status_counts()["sources"] == 1


def test_owner_schedule_uses_local_date_across_utc_midnight():
    settings = Settings(
        _env_file=None,
        schedule_timezone="America/New_York",
        radar_times=["20:10"],
        research_times=[],
        delivery_times=[],
    )
    slots = due_slots(datetime(2026, 10, 6, 0, 15, tzinfo=UTC), settings=settings)
    assert len(slots) == 1
    assert slots[0].job_type == "radar"
    assert slots[0].slot_at.isoformat() == "2026-10-05T20:10:00-04:00"


@pytest.mark.parametrize(
    "values",
    [
        {"schedule_timezone": "Unknown/Timezone"},
        {"radar_times": ["24:00"]},
        {"delivery_times": ["12:00", "12:00"]},
    ],
)
def test_invalid_owner_schedule_fails_before_startup(values):
    with pytest.raises(ValueError):
        Settings(_env_file=None, **values)
