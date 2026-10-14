from __future__ import annotations

import pytest
from test_details_worker import artifact

from app.config import Settings, SourceConfig, SourceRegistry
from app.db.repository import Repository
from app.details.provider_guard import DetailProviderError
from app.details.worker import DetailWorker
from app.research.detail_policy import detail_policy_versions


@pytest.fixture
def reader_queue(tmp_path):
    settings = Settings(_env_file=None, app_env="test", data_dir=tmp_path, allowed_user_id=123)
    repo = Repository(settings.database_path)
    repo.migrate()
    repo.sync_sources(
        SourceRegistry(
            version=1,
            sources=[
                SourceConfig(
                    id="fixture",
                    name="Fixture",
                    kind="rss",
                    locator="https://example.com/rss",
                    role="primary",
                    region="global",
                    language="en",
                )
            ],
        )
    )
    delivery = repo.create_smoke_delivery(123, "https://example.com/release")
    repo.mark_delivery_sent(int(delivery["delivery_id"]), 9001)
    ctx = {
        "baseline_card_id": delivery["card_id"],
        "baseline_text": "Содержательная новость. " * 20,
        "baseline_document_ids": [],
        "baseline_source_urls": ["https://example.com/release"],
        "context_fingerprint": "a" * 64,
    }
    return settings, repo, delivery, ctx


def tier_artifact(event_id, depth, ctx, *, cost=0.03):
    schema, prompt = detail_policy_versions(depth)
    return artifact(event_id, cost_usd=cost).model_copy(
        update={
            "depth": depth,
            "reader_context": ctx,
            "schema_version": schema,
            "prompt_version": prompt,
            "generation_key": f"detail:{event_id}:{depth}:abcdef1234567890",
            "provider_response_metadata": {
                "actual_additional_sources": 3 if depth == "expanded" else 6,
                "target_additional_sources": 3 if depth == "expanded" else 6,
            },
        }
    )


def test_worker_passes_frozen_tier_and_does_not_reuse_expanded_artifact_for_article(reader_queue):
    settings, repo, delivery, ctx = reader_queue
    for depth in ("expanded", "article"):
        repo.enqueue_detail_request(delivery["delivery_id"], 123, depth=depth, reader_context=ctx)
    calls = []

    def builder(settings, repo, event_id, *, generation_key, progress, depth, reader_context):
        assert all(reader_context[key] == value for key, value in ctx.items())
        assert generation_key is None
        if depth == "article":
            assert reader_context.get("article_reference_fingerprint")
            assert "expanded_reference_text" not in reader_context
        calls.append(depth)
        return tier_artifact(event_id, depth, reader_context)

    worker = DetailWorker(settings, repo, builder=builder)
    first, second = worker.run_once(), worker.run_once()
    assert first.status == second.status == "ready"
    assert calls == ["expanded", "article"]
    assert not first.cache_hit and not second.cache_hit
    assert repo.get_detailed_article(first.article_id)["depth"] == "expanded"
    assert repo.get_detailed_article(second.article_id)["depth"] == "article"


def test_worker_maps_unknown_provider_outcome_to_terminal_failure_without_second_dispatch(
    reader_queue,
):
    settings, repo, delivery, ctx = reader_queue
    repo.enqueue_detail_request(delivery["delivery_id"], 123, depth="expanded", reader_context=ctx)
    calls = []

    def builder(*args, **kwargs):
        calls.append(True)
        raise DetailProviderError("uncertain", call_id=7, cost_usd=0.023)

    worker = DetailWorker(settings, repo, builder=builder)
    result = worker.run_once()
    assert result.status == "failed" and result.cost_usd == 0.023
    assert "uncertain" in result.error
    assert worker.run_once().status == "idle" and calls == [True]


def test_worker_rejects_article_for_same_baseline_but_another_reading_reference(reader_queue):
    settings, repo, delivery, ctx = reader_queue
    repo.enqueue_detail_request(delivery["delivery_id"], 123, depth="article", reader_context=ctx)

    def foreign_reference(settings, repo, event_id, *, reader_context, **kwargs):
        foreign = {
            **reader_context,
            "expanded_reference_article_id": 99,
            "expanded_reference_text": "Foreign already read More result.",
            "expanded_reference_units": 39,
        }
        return tier_artifact(event_id, "article", foreign)

    result = DetailWorker(settings, repo, builder=foreign_reference).run_once()
    assert result.status == "failed"
    with repo.connection() as db:
        assert db.execute("SELECT count(*) FROM detailed_articles").fetchone()[0] == 0


def test_worker_rejects_legacy_policy_artifact_for_new_tier(reader_queue):
    settings, repo, delivery, ctx = reader_queue
    repo.enqueue_detail_request(delivery["delivery_id"], 123, depth="expanded", reader_context=ctx)

    def builder(*args, **kwargs):
        return artifact(delivery["event_id"])

    result = DetailWorker(settings, repo, builder=builder).run_once()
    assert result.status == "failed" and result.article_id is None
    assert "outdated evidence-policy" in result.error


@pytest.mark.parametrize("bad", ["foreign_event", "wrong_fingerprint", "missing_context"])
def test_worker_rejects_foreign_or_unscoped_artifact_before_storage(reader_queue, bad):
    settings, repo, delivery, ctx = reader_queue
    repo.enqueue_detail_request(delivery["delivery_id"], 123, depth="expanded", reader_context=ctx)
    result = tier_artifact(delivery["event_id"], "expanded", ctx)
    if bad == "foreign_event":
        result = result.model_copy(update={"event_id": delivery["event_id"] + 1})
    else:
        wrong_context = {} if bad == "missing_context" else {**ctx, "context_fingerprint": "b" * 64}
        result = result.model_copy(update={"reader_context": wrong_context})

    outcome = DetailWorker(settings, repo, builder=lambda *args, **kwargs: result).run_once()
    assert outcome.status == "failed" and outcome.article_id is None
    assert "another event or frozen reader baseline" in outcome.error
    assert outcome.cost_usd == result.cost_usd
    with repo.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM detailed_articles").fetchone()[0] == 0
        assert db.execute("SELECT status FROM detail_requests").fetchone()[0] == "failed"


def test_legacy_worker_also_rejects_foreign_event_artifact(reader_queue):
    settings, repo, delivery, _ = reader_queue
    repo.enqueue_detail_request(delivery["delivery_id"], 123)
    result = artifact(delivery["event_id"] + 1)
    outcome = DetailWorker(settings, repo, builder=lambda *args, **kwargs: result).run_once()
    assert outcome.status == "failed" and outcome.article_id is None
    with repo.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM detailed_articles").fetchone()[0] == 0
