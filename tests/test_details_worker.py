from __future__ import annotations

from pathlib import Path

from app.config import Settings, load_source_registry
from app.db.repository import Repository
from app.details.models import DetailArticleBuildResult
from app.details.research import DetailEvidenceError
from app.details.worker import DetailWorker

PROJECT_ROOT = Path(__file__).parents[1]


def setup_worker(tmp_path: Path) -> tuple[Settings, Repository, dict]:
    settings = Settings(
        app_env="test",
        data_dir=tmp_path,
        sources_file=PROJECT_ROOT / "tests/fixtures/config/sources.yaml",
        interests_file=PROJECT_ROOT / "tests/fixtures/config/interests.yaml",
        contours_file=PROJECT_ROOT / "tests/fixtures/config/contours.yaml",
        telegram_session_path=tmp_path / "telegram-reader.session",
        details_enabled=True,
    )
    repository = Repository(settings.database_path)
    repository.migrate()
    repository.sync_sources(load_source_registry(settings.sources_file))
    delivery = repository.create_smoke_delivery(123, "https://example.com/release")
    repository.mark_delivery_sent(int(delivery["delivery_id"]), 9001)
    repository.enqueue_detail_request(int(delivery["delivery_id"]), 42)
    return settings, repository, delivery


def artifact(event_id: int, *, cost_usd: float = 0.005) -> DetailArticleBuildResult:
    return DetailArticleBuildResult(
        event_id=event_id,
        generation_key=f"detail:{event_id}:1234567890abcdef12345678",
        model="fake/model",
        provider_response_metadata={"provider": "fake"},
        research_plan={"queries": ["example"]},
        fact_snapshot={"facts": [{"text": "Verified"}]},
        document_ids=[1, 2],
        structured_output={"headline": "Detailed"},
        rendered_html="<b>Detailed article</b>",
        source_urls=["https://example.com/a", "https://independent.test/b"],
        validation_status="valid",
        validation_errors=[],
        cost_usd=cost_usd,
        expires_at="2099-09-01T00:00:00+00:00",
    )


def test_worker_builds_one_claimed_request_and_tracks_progress(tmp_path: Path) -> None:
    settings, repository, delivery = setup_worker(tmp_path)
    calls: list[tuple[int, str | None]] = []

    def builder(settings, repository, event_id, *, generation_key=None, progress=None):
        del settings, repository
        calls.append((event_id, generation_key))
        assert progress is not None
        for stage in ("planning", "search", "facts", "writing"):
            progress(stage)
        return artifact(event_id)

    result = DetailWorker(settings, repository, builder=builder).run_once()

    assert result.status == "ready"
    assert result.article_id is not None
    assert result.cost_usd == 0.005
    assert calls == [(int(delivery["event_id"]), None)]
    request = repository.detail_request_for_delivery(int(delivery["delivery_id"]))
    assert request is not None
    assert request["status"] == "ready"
    assert request["stage"] == "delivery"
    assert repository.get_detailed_article(result.article_id)["provider_response_metadata"][
        "research_plan"
    ] == {"queries": ["example"]}
    assert DetailWorker(settings, repository, builder=builder).run_once().status == "idle"


def test_worker_maps_evidence_failure_to_honest_terminal_state(tmp_path: Path) -> None:
    settings, repository, delivery = setup_worker(tmp_path)

    def builder(settings, repository, event_id, *, generation_key=None, progress=None):
        del settings, repository, event_id, generation_key, progress
        raise DetailEvidenceError(["Only one independent full-text domain"], cost_usd=0.001)

    result = DetailWorker(settings, repository, builder=builder).run_once()

    assert result.status == "insufficient_evidence"
    assert result.cost_usd == 0.001
    request = repository.detail_request_for_delivery(int(delivery["delivery_id"]))
    assert request is not None
    assert request["status"] == "insufficient_evidence"
    assert "full-text" in request["last_error"]


def test_worker_enforces_per_request_publish_cap(tmp_path: Path) -> None:
    settings, repository, delivery = setup_worker(tmp_path)

    def builder(settings, repository, event_id, *, generation_key=None, progress=None):
        del settings, repository, generation_key, progress
        return artifact(event_id, cost_usd=0.03)

    result = DetailWorker(settings, repository, builder=builder).run_once()

    assert result.status == "budget_blocked"
    assert result.article_id is None
    request = repository.detail_request_for_delivery(int(delivery["delivery_id"]))
    assert request is not None
    assert request["status"] == "budget_blocked"
    assert repository.status_counts()["valid_detailed_articles"] == 0
