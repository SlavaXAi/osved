import json
from datetime import timedelta
from types import SimpleNamespace

from app.ops.schedule import ScheduledJobExecutor
from app.research.policy import fact_policy_metadata
from tests.test_schedule import NOW, insert_card, insert_event, setup_schedule


def save_sheet(repository, event_id, coverage):
    return repository.create_fact_sheet(
        event_id=event_id,
        schema_version="neutral_fact_sheet_v3",
        facts=[],
        coverage=coverage,
        validation_status="valid",
        expires_at=(NOW + timedelta(days=30)).isoformat(),
    )


def test_old_valid_fact_sheet_does_not_authorize_writing_or_reuse(tmp_path):
    _, repository = setup_schedule(tmp_path)
    event_id = insert_event(repository)
    old = save_sheet(repository, event_id, {"event_type": "other"})
    assert repository.latest_valid_fact_sheet(event_id) is None
    assert repository.fact_sheet_evidence_items(event_id) == []
    assert repository.events_without_cards(event_id=event_id, require_fact_sheet=True) == []
    assert repository.latest_valid_fact_sheet(event_id, require_current_policy=False)["id"] == old
    current = save_sheet(repository, event_id, fact_policy_metadata())
    assert repository.latest_valid_fact_sheet(event_id)["id"] == current
    assert [
        row["id"]
        for row in repository.events_without_cards(event_id=event_id, require_fact_sheet=True)
    ] == [event_id]


def test_wrong_or_malformed_policy_does_not_fall_back_to_legacy_cache(tmp_path):
    _, repository = setup_schedule(tmp_path)
    event_id = insert_event(repository)
    for coverage in (
        {},
        {**fact_policy_metadata(), "validation_policy_version": "older"},
        {**fact_policy_metadata(), "extraction_prompt_version": "older"},
    ):
        save_sheet(repository, event_id, coverage)
        assert repository.latest_valid_fact_sheet(event_id) is None
    with repository.connection() as connection:
        connection.execute("UPDATE event_fact_sheets SET coverage_json='{broken'")
    assert repository.latest_valid_fact_sheet(event_id) is None


def test_scheduler_reextracts_old_facts_from_existing_documents_without_web_search(
    tmp_path, monkeypatch
):
    settings, repository = setup_schedule(tmp_path)
    event_id = insert_event(repository)
    old = save_sheet(repository, event_id, {})
    document_id = repository.upsert_research_document(
        canonical_url="https://example.com/release",
        url="https://example.com/release",
        source_class="primary",
        retrieved_at=NOW.isoformat(),
        expires_at=(NOW + timedelta(days=7)).isoformat(),
        content_text="The company released a concrete technology with useful capabilities. " * 30,
        metadata={"content_kind": "scraped_fulltext"},
    )
    repository.attach_research_document(
        event_id=event_id,
        research_document_id=document_id,
        evidence_role="primary",
    )
    calls = []

    def no_web(*args, **kwargs):
        raise AssertionError("A policy upgrade must reuse full-text evidence, not search again")

    def reextract(*args):
        calls.append("reextract")
        assert repository.latest_valid_fact_sheet(event_id) is None
        save_sheet(repository, event_id, fact_policy_metadata())
        return {"status": "valid", "cost_usd": 0.001}

    def generate(*args, **kwargs):
        calls.append("write")
        assert repository.latest_valid_fact_sheet(event_id)["id"] != old
        insert_card(repository, event_id)
        return SimpleNamespace(cards_created=1, cost_usd=0.001, errors=0)

    monkeypatch.setattr("app.ops.schedule.ResearchEventService", no_web)
    monkeypatch.setattr("app.ops.schedule.build_event_fact_sheet", reextract)
    monkeypatch.setattr("app.ops.schedule.generate_cards", generate)
    result = ScheduledJobExecutor(settings, repository, now=NOW).research()
    assert result["research"]["status"] == "cached_documents_policy_refresh"
    assert calls == ["reextract", "write"]
    with repository.connection() as connection:
        raw = connection.execute("SELECT coverage_json FROM event_fact_sheets WHERE id=?", (old,))
        assert json.loads(raw.fetchone()[0]) == {}  # Never re-stamp a stale fact as validated.
