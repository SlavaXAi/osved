import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import Settings, load_source_registry
from app.db.repository import Repository
from app.ops.schedule import (
    ScheduledJobExecutor,
    claim_runtime_window,
    configure_schedule,
    due_slots,
    fresh_delivery_card_ids,
    fresh_research_events,
    run_schedule_tick,
)
from app.research.policy import fact_policy_metadata

PROJECT_ROOT = Path(__file__).parents[1]


def setup_schedule(tmp_path: Path) -> tuple[Settings, Repository]:
    settings = Settings(
        app_env="test",
        data_dir=tmp_path,
        sources_file=PROJECT_ROOT / "tests/fixtures/config/sources.yaml",
        interests_file=PROJECT_ROOT / "tests/fixtures/config/interests.yaml",
        telegram_session_path=tmp_path / "telegram-reader.session",
    )
    repository = Repository(settings.database_path)
    repository.migrate()
    repository.sync_sources(load_source_registry(settings.sources_file))
    return settings, repository


def test_schedule_slots_follow_moscow_working_hours() -> None:
    noon = datetime.fromisoformat("2026-08-14T12:05:00+03:00")
    evening = datetime.fromisoformat("2026-08-14T20:05:00+03:00")

    assert [(slot.job_type, slot.slot_at.hour) for slot in due_slots(noon)] == [("radar", 12)]
    assert [(slot.job_type, slot.slot_at.hour) for slot in due_slots(evening)] == [
        ("delivery", 20),
        ("radar", 20),
    ]


def test_runtime_paid_window_has_one_winner_for_parallel_manual_and_schedule(tmp_path):
    _, repository = setup_schedule(tmp_path)
    now = datetime.fromisoformat("2026-09-07T12:00:00+03:00")
    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(
            pool.map(
                lambda _: claim_runtime_window(repository, "fixture-window", now=now), range(8)
            )
        )
    assert sum(claims) == 1


def test_dry_run_is_idempotent_and_performs_no_job(tmp_path: Path) -> None:
    settings, repository = setup_schedule(tmp_path)
    calls: list[str] = []

    first = run_schedule_tick(
        settings,
        repository,
        now=datetime.fromisoformat("2026-08-14T12:05:00+03:00"),
        force_dry_run=True,
        executor=lambda job: calls.append(job) or {},
    )
    second = run_schedule_tick(
        settings,
        repository,
        now=datetime.fromisoformat("2026-08-14T12:06:00+03:00"),
        force_dry_run=True,
        executor=lambda job: calls.append(job) or {},
    )

    assert first["slots"][0]["status"] == "dry_run"
    assert second["slots"][0]["status"] == "already_claimed"
    assert calls == []
    assert repository.recent_schedule_slots()[0]["metrics"]["external_requests"] == 0


def test_live_tick_records_metrics_and_respects_kill_switch(tmp_path: Path) -> None:
    settings, repository = setup_schedule(tmp_path)
    now = datetime.fromisoformat("2026-08-14T15:05:00+03:00")
    assert run_schedule_tick(settings, repository, now=now)["status"] == "disabled"

    configure_schedule(repository, enabled=True, dry_run=False)
    result = run_schedule_tick(
        settings,
        repository,
        now=now,
        executor=lambda job: {
            "job": job,
            "cards_created": 1,
            "errors": 0,
            "cost_usd": 0.01,
        },
    )

    assert result["slots"] == [
        {
            "job_type": "research",
            "slot_at": "2026-08-14T15:00:00+03:00",
            "status": "completed",
        },
    ]
    slot = repository.recent_schedule_slots()[0]
    assert slot["metrics"]["job"] == "research"
    assert repository.status_counts()["runs"] == 1


NOW = datetime.fromisoformat("2026-09-07T12:00:00+03:00")


def insert_event(repository, *, age_days=1, event_at=None, score=10, discovery_mode="fixed"):
    with repository.connection() as connection:
        source_id = connection.execute("SELECT id FROM sources LIMIT 1").fetchone()[0]
        count = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    timestamp = (NOW - timedelta(days=age_days)).isoformat()
    item_id = repository.insert_source_item_with_id(
        source_id=source_id,
        external_id=f"schedule-fixture-{count}",
        source_url=f"https://example.org/news/{count}",
        original_text="A concrete event with supporting evidence and meaningful changes.",
        original_language="en",
        published_at=timestamp,
        outbound_urls=[],
        media_metadata={},
        payload_hash=f"schedule-fixture-hash-{count}",
        raw_payload={},
    )
    with repository.connection() as connection:
        connection.execute(
            "UPDATE source_items SET discovery_mode=? WHERE id=?", (discovery_mode, item_id)
        )
        cursor = connection.execute(
            """INSERT INTO events
            (headline,event_type,event_at,first_seen_at,latest_source_at,primary_source_item_id,score)
            VALUES(?,'release',?,?,?,?,?)""",
            (
                f"Useful recent event {count}",
                event_at,
                NOW.isoformat(),
                NOW.isoformat(),
                item_id,
                score,
            ),
        )
        return int(cursor.lastrowid)


def insert_card(repository, event_id, *, stage="final", validation="valid"):
    return repository.create_card(
        event_id=event_id,
        prompt_version="fixture",
        model="fixture",
        generation_cost_usd=0,
        structured_output={"evaluation_stage": stage},
        rendered_html="<b>Verified news</b>",
        source_urls=["https://example.org"],
        primary_preview_url="https://example.org",
        validation_status=validation,
    )


def test_freshness_uses_event_or_source_date_not_new_card_or_discovery_time(tmp_path):
    _, repository = setup_schedule(tmp_path)
    old = insert_event(repository, age_days=20, score=100)
    repost = insert_event(repository, age_days=1, event_at=(NOW - timedelta(days=20)).isoformat())
    invalid = insert_event(repository, event_at="not-a-date")
    future = insert_event(repository, age_days=-2)
    fresh = insert_event(repository, age_days=2)
    unknown_event_date = insert_event(repository, age_days=3, event_at="")
    events = fresh_research_events(repository, now=NOW, limit=2)
    assert {row["id"] for row in events} == {fresh, unknown_event_date}
    assert {old, repost, invalid, future}.isdisjoint({row["id"] for row in events})


def test_five_day_boundary_and_timezone_are_normalized(tmp_path):
    _, repository = setup_schedule(tmp_path)
    boundary = insert_event(repository, age_days=5)
    insert_event(repository, event_at=(NOW - timedelta(days=5, seconds=1)).isoformat())
    recent = insert_event(repository, event_at="2026-09-03T00:00:00Z")
    assert {row["id"] for row in fresh_research_events(repository, now=NOW)} == {boundary, recent}


@pytest.mark.parametrize(
    "validation,stage",
    [
        ("candidate_preview", "discovery"),
        ("valid", "discovery"),
        ("valid", "candidate_preview"),
        ("invalid", "final"),
    ],
)
def test_search_discovery_previews_are_eligible_for_research_not_delivery(
    tmp_path, validation, stage
):
    _, repository = setup_schedule(tmp_path)
    event_id = insert_event(repository, discovery_mode="web_search")
    with repository.connection() as connection:
        connection.execute(
            """INSERT INTO cards (event_id,card_version,prompt_version,structured_output_json,
            rendered_html,source_urls_json,validation_status)
            VALUES(?,1,'fixture',?,'Preview','[]',?)""",
            (event_id, json.dumps({"evaluation_stage": stage}), validation),
        )
    assert [row["id"] for row in fresh_research_events(repository, now=NOW)] == [event_id]
    assert fresh_delivery_card_ids(repository, owner_id=123, now=NOW) == []
    assert not repository.event_has_valid_card(event_id)
    assert [row["id"] for row in repository.events_without_cards()] == [event_id]

    # A real final card, not a preview, consumes both research and writing queues.
    final_card = insert_card(repository, event_id)
    assert fresh_research_events(repository, now=NOW) == []
    assert repository.events_without_cards() == []
    assert repository.event_has_valid_card(event_id)
    assert fresh_delivery_card_ids(repository, owner_id=123, now=NOW) == [final_card]


@pytest.mark.parametrize(
    "payload,is_finished",
    [
        ("{broken", False),
        ("null", False),
        ("[]", False),
        ('{"headline":"Legacy final without a stage"}', True),
    ],
)
def test_research_and_delivery_share_safe_legacy_card_rule(tmp_path, payload, is_finished):
    _, repository = setup_schedule(tmp_path)
    event_id = insert_event(repository)
    with repository.connection() as connection:
        connection.execute(
            """INSERT INTO cards (event_id,card_version,prompt_version,structured_output_json,
            rendered_html,source_urls_json,validation_status)
            VALUES(?,1,'fixture',?,'Old card','[]','valid')""",
            (event_id, payload),
        )
    assert repository.event_has_valid_card(event_id) is is_finished
    assert bool(fresh_research_events(repository, now=NOW)) is not is_finished
    assert bool(repository.events_without_cards()) is not is_finished
    assert bool(fresh_delivery_card_ids(repository, owner_id=123, now=NOW)) is is_finished


def test_preview_does_not_revive_stale_or_rejected_event(tmp_path):
    _, repository = setup_schedule(tmp_path)
    insert_event(repository, age_days=6)
    for status in ("merged", "duplicate", "rejected", "feedback_rejected", "research_insufficient"):
        event_id = insert_event(repository)
        repository.set_event_status(event_id, status)
    assert fresh_research_events(repository, now=NOW) == []


def test_newer_preview_does_not_hide_an_existing_finished_card(tmp_path):
    _, repository = setup_schedule(tmp_path)
    event_id = insert_event(repository)
    final = insert_card(repository, event_id)
    insert_card(repository, event_id, stage="candidate_preview")
    assert fresh_delivery_card_ids(repository, owner_id=123, now=NOW) == [final]


def test_production_delivery_excludes_stale_and_mislabeled_previews_and_old_versions(tmp_path):
    _, repository = setup_schedule(tmp_path)
    old_card = insert_card(repository, insert_event(repository, age_days=8, score=200))
    preview_card = insert_card(repository, insert_event(repository), stage="discovery")
    valid_event = insert_event(repository)
    old_version = insert_card(repository, valid_event)
    latest = insert_card(repository, valid_event)
    selected = fresh_delivery_card_ids(repository, owner_id=123, now=NOW)
    assert selected == [latest]
    assert {old_card, preview_card, old_version}.isdisjoint(selected)


def test_delivered_event_is_not_resent_after_regeneration_but_failed_card_can_retry(tmp_path):
    _, repository = setup_schedule(tmp_path)
    event_id = insert_event(repository)
    card = insert_card(repository, event_id)
    delivery_id = repository.ensure_delivery(card, 123)
    repository.mark_delivery_sent(delivery_id, 900)
    insert_card(repository, event_id)
    assert fresh_delivery_card_ids(repository, owner_id=123, now=NOW) == []
    fresh = insert_card(repository, insert_event(repository))
    retry = repository.ensure_delivery(fresh, 123)
    repository.mark_delivery_error(retry, "temporary")
    assert fresh_delivery_card_ids(repository, owner_id=123, now=NOW) == [fresh]


def test_scheduled_delivery_uses_explicit_fresh_card_ids(tmp_path, monkeypatch):
    settings, repository = setup_schedule(tmp_path)
    settings.allowed_user_id = 123
    insert_card(repository, insert_event(repository, age_days=9))
    fresh = insert_card(repository, insert_event(repository))
    sent = []

    async def fake_send(settings, repository, card_ids, **kwargs):
        sent.extend(card_ids)
        return {"sent": len(card_ids), "errors": 0}

    monkeypatch.setattr("app.ops.schedule.send_cards_by_ids", fake_send)
    result = ScheduledJobExecutor(settings, repository, now=NOW).delivery()
    assert sent == [fresh]
    assert result["sent"] == 1


def test_generation_retry_reuses_valid_facts_without_paying_for_research(tmp_path, monkeypatch):
    settings, repository = setup_schedule(tmp_path)
    event_id = insert_event(repository)
    repository.create_fact_sheet(
        event_id=event_id,
        schema_version="test",
        facts=[],
        coverage=fact_policy_metadata(),
        validation_status="valid",
        expires_at=(NOW + timedelta(days=2)).isoformat(),
    )
    generated = []

    def no_research(*args, **kwargs):
        raise AssertionError("Valid cached facts must not trigger another paid research pass")

    def fake_generate(settings, repository, *, limit, event_id):
        generated.append(event_id)
        return SimpleNamespace(cards_created=1, cost_usd=0.002, errors=0)

    monkeypatch.setattr("app.ops.schedule.ResearchEventService", no_research)
    monkeypatch.setattr("app.ops.schedule.build_event_fact_sheet", no_research)
    monkeypatch.setattr("app.ops.schedule.generate_cards", fake_generate)
    result = ScheduledJobExecutor(settings, repository, now=NOW).research()
    assert result["fact_sheet"]["cached"] is True
    assert result["research"]["status"] == "cached"
    assert result["cost_usd"] == 0.002
    assert generated == [event_id]


def test_empty_fresh_queue_does_not_open_telegram_session(tmp_path, monkeypatch):
    settings, repository = setup_schedule(tmp_path)
    settings.allowed_user_id = 123
    insert_card(repository, insert_event(repository, age_days=9))

    def forbidden(*args, **kwargs):
        raise AssertionError("Must not open Telegram for a stale queue")

    monkeypatch.setattr("app.ops.schedule.send_cards_by_ids", forbidden)
    assert ScheduledJobExecutor(settings, repository, now=NOW).delivery()["sent"] == 0


def test_video_delivery_has_no_day_quota_and_keeps_toggle(tmp_path, monkeypatch):
    settings, repository = setup_schedule(tmp_path)
    settings.allowed_user_id = 123
    event_id = insert_event(repository)
    with repository.connection() as connection:
        connection.execute(
            """UPDATE sources SET kind='youtube', enabled=1 WHERE id=(SELECT source_id
            FROM source_items JOIN events ON events.primary_source_item_id=source_items.id
            WHERE events.id=?)""",
            (event_id,),
        )
    calls = []
    closed = []

    async def close():
        closed.append(True)

    async def fake_metadata(bot, settings, repository, *, chat_id, limit):
        calls.append((chat_id, limit))
        return {"sent": 1, "errors": 0}

    monkeypatch.setattr(
        "app.ops.schedule.build_bot",
        lambda settings: SimpleNamespace(session=SimpleNamespace(close=close)),
    )
    monkeypatch.setattr("app.ops.schedule.send_video_recommendations", fake_metadata)
    current = NOW + timedelta(minutes=31)
    executor = ScheduledJobExecutor(settings, repository, now=current)
    # Old production claims must not suppress today's newly discovered uploads.
    repository.set_runtime_setting(f"scheduled_video:123:{current.date()}", "completed")
    first = executor.delivery()
    second = executor.delivery()
    assert first["news_sent"] == 0 and first["video_sent"] == 1
    assert second["video_sent"] == 1
    assert calls == [(123, None), (123, None)]
    assert closed == [True, True]
    repository.set_runtime_setting("scheduled_video_recommendations_enabled", "false")
    tomorrow = ScheduledJobExecutor(
        settings, repository, now=current + timedelta(days=1)
    ).delivery()
    assert tomorrow["videos"]["status"] == "disabled"
    assert calls == [(123, None), (123, None)]


def test_scheduled_search_injects_guard_and_accounts_for_unrecorded_failed_response(
    tmp_path, monkeypatch
):
    from pydantic import SecretStr

    from app.ops.budget import SearchBudgetTransport

    settings, repository = setup_schedule(tmp_path)
    settings.openrouter_api_key = SecretStr("not-a-real-key")
    monkeypatch.setattr(
        "app.ops.schedule.collect_due_sources", lambda *a, **k: {"cost_usd": 0, "errors": 0}
    )
    monkeypatch.setattr("app.ops.schedule.prepare_candidates", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr("app.ops.schedule.deduplicate_events", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(
        "app.ops.schedule.load_search_agent_registry",
        lambda *a: SimpleNamespace(enabled_lenses=[SimpleNamespace(id="software")]),
    )

    def fake_radar(*args, transport, **kwargs):
        assert isinstance(transport, SearchBudgetTransport)
        transport.transport = lambda p, k: {"usage": {"cost": 0.001}}
        transport(
            {"model": "fixture", "messages": [{"content": "test"}], "max_tokens": 100}, "unused"
        )
        raise ValueError("Parser failed after provider accepted the request")

    monkeypatch.setattr("app.ops.schedule.run_search_radar", fake_radar)
    result = ScheduledJobExecutor(settings, repository, now=NOW).radar()
    assert result["search"]["status"] == "failed"
    assert result["search"]["budget"]["reported_cost_usd"] == 0.001
    assert repository.current_month_cost() == 0.001
    assert result["cost_usd"] == 0.001
    second = ScheduledJobExecutor(settings, repository, now=NOW).radar()
    assert second["search"]["status"] == "already_attempted"
