from __future__ import annotations

import json
from types import SimpleNamespace

from typer.testing import CliRunner

from app import cli as cli_module
from app.config import Settings
from app.db.repository import Repository
from app.details.worker import DetailWorkerResult


def test_details_status_is_read_only(tmp_path, monkeypatch) -> None:
    settings = Settings(data_dir=tmp_path, _env_file=None)
    repository = Repository(settings.database_path)
    repository.migrate()
    monkeypatch.setattr(cli_module, "bootstrap", lambda: (settings, repository))

    result = CliRunner().invoke(cli_module.app, ["details-status", "--limit", "5"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["recent"] == []
    assert sum(payload["counts"].values()) == 0


def test_details_worker_once_reports_idle_without_external_work(tmp_path, monkeypatch) -> None:
    settings = Settings(data_dir=tmp_path, _env_file=None)
    repository = Repository(settings.database_path)
    repository.migrate()
    monkeypatch.setattr(cli_module, "bootstrap", lambda: (settings, repository))

    class FakeWorker:
        def __init__(self, _settings, _repository) -> None:
            pass

        def run_once(self) -> DetailWorkerResult:
            return DetailWorkerResult(request_id=None, status="idle")

    monkeypatch.setattr(cli_module, "DetailWorker", FakeWorker)
    result = CliRunner().invoke(cli_module.app, ["details-worker-once"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "request_id": None,
        "status": "idle",
        "article_id": None,
        "cost_usd": 0.0,
        "error": None,
        "cache_hit": False,
    }


def test_details_build_is_manual_and_does_not_deliver(tmp_path, monkeypatch) -> None:
    settings = Settings(data_dir=tmp_path, _env_file=None)
    repository = Repository(settings.database_path)
    repository.migrate()
    monkeypatch.setattr(cli_module, "bootstrap", lambda: (settings, repository))
    calls: list[int] = []

    def fake_build(_settings, _repository, event_id: int):
        calls.append(event_id)
        return SimpleNamespace(
            model_dump=lambda **_kwargs: {
                "event_id": event_id,
                "validation_status": "valid",
                "cost_usd": 0.004,
                "rendered_html": "<b>Разбор</b>",
            }
        )

    monkeypatch.setattr(cli_module, "build_detail_article", fake_build)
    result = CliRunner().invoke(cli_module.app, ["details-build", "17"])

    assert result.exit_code == 0, result.output
    assert calls == [17]
    assert json.loads(result.output)["validation_status"] == "valid"


def test_details_retry_surfaces_invalid_state_as_exit_two(tmp_path, monkeypatch) -> None:
    settings = Settings(data_dir=tmp_path, _env_file=None)
    repository = Repository(settings.database_path)
    repository.migrate()
    monkeypatch.setattr(cli_module, "bootstrap", lambda: (settings, repository))

    result = CliRunner().invoke(cli_module.app, ["details-retry", "999"])

    assert result.exit_code == 2
    assert "Unknown detail request: 999" in result.output
