import json
from datetime import timedelta
from types import SimpleNamespace

import pytest

from app.ops.schedule import RESEARCH_FAILURE_PREFIX, ScheduledJobExecutor, fresh_research_events
from tests.test_schedule import NOW, insert_card, insert_event, setup_schedule


def install_successful_writing(monkeypatch):
    def generate(_settings, repository, *, limit, event_id):
        assert limit == 1
        rows = repository.events_without_cards(event_id=event_id)
        assert [row["id"] for row in rows] == [event_id]
        insert_card(repository, event_id)
        return SimpleNamespace(cards_created=1, cost_usd=0, errors=0)

    monkeypatch.setattr("app.ops.schedule.generate_cards", generate)
    monkeypatch.setattr(
        "app.ops.schedule.build_event_fact_sheet", lambda *a: {"status": "valid", "cost_usd": 0}
    )
    monkeypatch.setattr(
        "app.ops.schedule.rank_event_candidates", lambda rows, **kwargs: rows[: kwargs["limit"]]
    )


def test_network_failure_waits_24h_releases_other_event_then_stops_after_second(
    tmp_path, monkeypatch
):
    settings, repository = setup_schedule(tmp_path)
    settings.scheduled_research_max_attempts = 1  # Isolate the per-event cooldown policy.
    high = insert_event(repository, score=100)
    other = insert_event(repository, score=10)
    calls = []
    install_successful_writing(monkeypatch)

    class Web:
        def research(self, event_id, **kwargs):
            calls.append(event_id)
            if event_id == high:
                repository.record_usage(
                    "fixture", "web.search", cost_usd=0.003, metadata={"event_id": high}
                )
                raise TimeoutError("provider unavailable")
            return SimpleNamespace(cost_usd=0)

    monkeypatch.setattr("app.ops.schedule.ResearchEventService", lambda *a: Web())
    first = ScheduledJobExecutor(settings, repository, now=NOW).research()
    assert first["status"] == "retry_scheduled"
    assert first["cost_usd"] == pytest.approx(0.003)
    assert calls == [high]  # No fallback event or retry inside the same slot.
    assert [row["id"] for row in fresh_research_events(repository, now=NOW, limit=1)] == [other]

    next_slot = ScheduledJobExecutor(settings, repository, now=NOW + timedelta(hours=6)).research()
    assert next_slot["event_id"] == other and next_slot["cards_created"] == 1
    assert calls == [high, other]
    before_boundary = NOW + timedelta(hours=24, seconds=-1)
    assert fresh_research_events(repository, now=before_boundary) == []
    second = ScheduledJobExecutor(settings, repository, now=NOW + timedelta(hours=24)).research()
    assert second["status"] == "research_insufficient"
    assert second["retry"]["attempts"] == 2
    assert second["retry"]["retry_after"] is None
    assert repository.event_context(high)["status"] == "research_insufficient"
    assert "2 failures" in repository.event_context(high)["decision_reason"]
    assert fresh_research_events(repository, now=NOW + timedelta(hours=48)) == []
    assert calls == [high, other, high]


def test_no_usable_documents_gets_one_retry_and_success_clears_it(tmp_path, monkeypatch):
    settings, repository = setup_schedule(tmp_path)
    event_id = insert_event(repository)
    install_successful_writing(monkeypatch)
    attempts = []

    class Web:
        def research(self, event_id, **kwargs):
            attempts.append(event_id)
            return SimpleNamespace(cost_usd=0)

    def facts(*args):
        if len(attempts) == 1:
            raise ValueError("No usable research documents; run research-event first")
        return {"status": "valid", "cost_usd": 0}

    monkeypatch.setattr("app.ops.schedule.ResearchEventService", lambda *a: Web())
    monkeypatch.setattr("app.ops.schedule.build_event_fact_sheet", facts)
    first = ScheduledJobExecutor(settings, repository, now=NOW).research()
    assert first["status"] == "retry_scheduled"
    key = f"{RESEARCH_FAILURE_PREFIX}{event_id}"
    assert json.loads(repository.get_runtime_setting(key))["attempts"] == 1
    second = ScheduledJobExecutor(settings, repository, now=NOW + timedelta(hours=24)).research()
    assert second["cards_created"] == 1
    assert repository.get_runtime_setting(key) is None
    assert fresh_research_events(repository, now=NOW + timedelta(hours=24)) == []


@pytest.mark.parametrize(
    "value",
    [
        "{broken",
        "[]",
        "null",
        '{"attempts":"1","retry_after":"2026-09-06T00:00:00Z"}',
        '{"attempts":1,"retry_after":"2026-09-06T00:00:00"}',
        '{"attempts":1,"retry_after":"not-a-dateZ"}',
        '{"attempts":1,"retry_after":1788710400}',
        '{"attempts":2,"retry_after":null}',
    ],
)
def test_corrupt_or_exhausted_retry_fails_closed_for_one_event_not_whole_queue(tmp_path, value):
    _, repository = setup_schedule(tmp_path)
    blocked = insert_event(repository, score=100)
    other = insert_event(repository, score=10)
    repository.set_runtime_setting(f"{RESEARCH_FAILURE_PREFIX}{blocked}", value)
    assert [row["id"] for row in fresh_research_events(repository, now=NOW, limit=1)] == [other]


def test_retry_compares_absolute_time_across_timezone_offsets(tmp_path):
    _, repository = setup_schedule(tmp_path)
    event_id = insert_event(repository)
    key = f"{RESEARCH_FAILURE_PREFIX}{event_id}"
    repository.set_runtime_setting(
        key,
        json.dumps(
            {
                "attempts": 1,
                "retry_after": "2026-09-07T10:00:00+01:00",
            }
        ),
    )
    assert fresh_research_events(repository, now=NOW - timedelta(seconds=1)) == []
    assert [row["id"] for row in fresh_research_events(repository, now=NOW)] == [event_id]


def test_ordinary_invalid_fact_status_is_not_reclassified_as_network_retry(tmp_path, monkeypatch):
    settings, repository = setup_schedule(tmp_path)
    event_id = insert_event(repository)

    class Web:
        def research(self, *args, **kwargs):
            return SimpleNamespace(cost_usd=0)

    def facts(*args):
        repository.set_event_status(event_id, "research_insufficient", "missing primary evidence")
        return {"status": "invalid", "cost_usd": 0}

    monkeypatch.setattr("app.ops.schedule.ResearchEventService", lambda *a: Web())
    monkeypatch.setattr("app.ops.schedule.build_event_fact_sheet", facts)
    monkeypatch.setattr(
        "app.ops.schedule.generate_cards",
        lambda *a, **k: SimpleNamespace(cards_created=0, cost_usd=0, errors=0),
    )
    result = ScheduledJobExecutor(settings, repository, now=NOW).research()
    assert result["errors"] == 1
    assert repository.get_runtime_setting(f"{RESEARCH_FAILURE_PREFIX}{event_id}") is None
    assert repository.event_context(event_id)["status"] == "research_insufficient"
