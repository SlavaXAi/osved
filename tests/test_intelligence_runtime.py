from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from typer.testing import CliRunner

import app.cli as cli_module
from app.config import Settings, load_source_registry
from app.db.repository import Repository
from app.intelligence.service import IntelligenceService

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 8, 25, 12, tzinfo=UTC)


def settings_for(tmp_path: Path) -> Settings:
    return Settings(
        app_env="test",
        data_dir=tmp_path,
        sources_file=PROJECT_ROOT / "tests/fixtures/config/sources.yaml",
        interests_file=PROJECT_ROOT / "tests/fixtures/config/interests.yaml",
        contours_file=PROJECT_ROOT / "tests/fixtures/config/contours.yaml",
        weekly_personas_file=PROJECT_ROOT / "tests/fixtures/config/weekly-personas.yaml",
        x_authors_file=PROJECT_ROOT / "tests/fixtures/config/x-authors.yaml",
        telegram_session_path=tmp_path / "telegram-reader.session",
        _env_file=None,
    )


def repository_with_events(tmp_path: Path) -> tuple[Settings, Repository, list[int]]:
    settings = settings_for(tmp_path)
    repository = Repository(settings.database_path)
    repository.migrate()
    repository.sync_sources(load_source_registry(settings.sources_file))
    event_ids: list[int] = []
    rows = [
        (
            1,
            "YC-стартап запустил AI-агента для строительных смет",
            ["startups", "construction_ai"],
        ),
        (
            3,
            "Стартап привлёк инвестиции для автоматизации строительных закупок",
            ["startups", "construction_ai"],
        ),
        (
            9,
            "Предыдущий seed round construction AI стартапа",
            ["startups", "construction_ai"],
        ),
    ]
    for index, (days_ago, headline, topics) in enumerate(rows, start=1):
        smoke = repository.create_smoke_delivery(
            123,
            f"https://example.com/intelligence-{index}",
        )
        event_ids.append(int(smoke["event_id"]))
        event_at = (NOW - timedelta(days=days_ago)).isoformat()
        with repository.connection() as connection:
            connection.execute(
                """
                UPDATE events
                SET status = 'candidate', headline = ?, event_at = ?, latest_source_at = ?,
                    score = 6.0, topics_json = ?, regions_json = '["global"]'
                WHERE id = ?
                """,
                (
                    headline,
                    event_at,
                    event_at,
                    json.dumps(topics, ensure_ascii=False),
                    smoke["event_id"],
                ),
            )
    return settings, repository, event_ids


def test_contour_backfill_is_deterministic_and_preserves_score_features(tmp_path: Path) -> None:
    settings, repository, event_ids = repository_with_events(tmp_path)
    with repository.connection() as connection:
        connection.execute(
            "UPDATE events SET score_features_json = ? WHERE id = ?",
            (json.dumps({"base_score": 6.0}), event_ids[0]),
        )

    service = IntelligenceService(settings, repository)
    first = service.backfill_contours()
    second = service.backfill_contours()

    assert first.examined == 3
    assert first.updated == 3
    assert first.classified == 3
    assert second.updated == 0
    assert second.unchanged == 3
    with repository.connection() as connection:
        features = json.loads(
            connection.execute(
                "SELECT score_features_json FROM events WHERE id = ?",
                (event_ids[0],),
            ).fetchone()[0]
        )
    assert features["base_score"] == 6.0
    assert features["primary_contour"] == "startups"
    assert features["classifier_version"] == "contours_v1"


def test_build_persists_product_and_reuses_generation_cache(tmp_path: Path) -> None:
    settings, repository, event_ids = repository_with_events(tmp_path)
    service = IntelligenceService(settings, repository)
    service.backfill_contours()

    first = service.build("monthly_theses", now=NOW)
    second = service.build("monthly_theses", now=NOW + timedelta(hours=1))

    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert second["id"] == first["id"]
    assert first["validation_status"] == "valid"
    assert first["source_event_ids"] == event_ids
    assert first["payload"]["metadata"]["generated_without_network"] is True
    assert repository.latest_intelligence_product("monthly_theses")["id"] == first["id"]
    assert repository.intelligence_status()["total"] == 1


def test_empty_product_is_saved_as_insufficient_instead_of_inventing_signal(
    tmp_path: Path,
) -> None:
    settings = settings_for(tmp_path)
    repository = Repository(settings.database_path)
    repository.migrate()
    service = IntelligenceService(settings, repository)

    product = service.build("weekly_map", now=NOW)

    assert product["validation_status"] == "insufficient_data"
    assert product["validation_errors"] == ["no_material_signal_in_period"]
    assert product["source_event_ids"] == []


def test_intelligence_corpus_excludes_unvalidated_candidates(tmp_path: Path) -> None:
    settings, repository, event_ids = repository_with_events(tmp_path)
    unvalidated_event_id = event_ids[-1]
    with repository.connection() as connection:
        connection.execute(
            "UPDATE cards SET validation_status = 'candidate_preview' WHERE event_id = ?",
            (unvalidated_event_id,),
        )

    IntelligenceService(settings, repository).backfill_contours()
    rows = repository.intelligence_events(
        since=(NOW - timedelta(days=35)).isoformat(),
        until=NOW.isoformat(),
    )

    assert {int(row["id"]) for row in rows} == set(event_ids[:-1])


def test_intelligence_cli_build_show_latest_and_status(tmp_path: Path, monkeypatch) -> None:
    settings, repository, _event_ids = repository_with_events(tmp_path)
    monkeypatch.setattr(cli_module, "bootstrap", lambda: (settings, repository))
    runner = CliRunner()

    backfill = runner.invoke(cli_module.app, ["contours-backfill"])
    assert backfill.exit_code == 0, backfill.output
    assert json.loads(backfill.output)["classified"] == 3

    built = runner.invoke(
        cli_module.app,
        ["intelligence-build", "weekly_map", "--now", NOW.isoformat()],
    )
    assert built.exit_code == 0, built.output
    product = json.loads(built.output)
    assert product["kind"] == "weekly_map"
    assert product["validation_status"] == "valid"

    shown = runner.invoke(cli_module.app, ["intelligence-show", str(product["id"])])
    latest = runner.invoke(cli_module.app, ["intelligence-latest", "weekly_map"])
    status = runner.invoke(cli_module.app, ["intelligence-status"])
    assert shown.exit_code == latest.exit_code == status.exit_code == 0
    assert json.loads(shown.output)["id"] == product["id"]
    assert json.loads(latest.output)["id"] == product["id"]
    assert json.loads(status.output)["total"] == 1
    assert json.loads(status.output)["network_requests"] == 0


def test_intelligence_cli_rejects_unknown_kind_without_writing(tmp_path: Path, monkeypatch) -> None:
    settings, repository, _event_ids = repository_with_events(tmp_path)
    monkeypatch.setattr(cli_module, "bootstrap", lambda: (settings, repository))

    result = CliRunner().invoke(cli_module.app, ["intelligence-build", "daily_hype"])

    assert result.exit_code == 2
    assert repository.intelligence_status()["total"] == 0
