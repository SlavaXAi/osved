from datetime import UTC, datetime, timedelta

import pytest

from app.config import load_interest_profile, load_search_agent_registry
from app.ops.budget import SearchBudgetTransport
from app.ops.feed_health import feed_health, feed_health_text
from app.ops.schedule import ScheduledJobExecutor, configure_schedule, fresh_delivery_card_ids
from app.pipeline.selection import PROMOTIONAL_POST
from app.radar.search import OpenRouterSearchAgent
from app.research.web import build_research_query
from tests.test_schedule import NOW, PROJECT_ROOT, insert_card, insert_event, setup_schedule
from tests.test_search_radar import parser_response, search_response


def test_live_request_shape_for_luna_search_and_parser(tmp_path):
    settings, repo = setup_schedule(tmp_path)
    settings.openrouter_search_model = "openai/gpt-5.6-luna"
    from pydantic import SecretStr

    settings.openrouter_api_key = SecretStr("fixture")
    seen = []

    def transport(payload, key):
        assert "temperature" not in payload
        assert payload["reasoning"] == {"effort": "none"}
        assert payload["provider"]["require_parameters"]
        if not payload.get("tools"):
            assert "max_tool_calls" not in payload
        seen.append(payload)
        return search_response() if payload.get("tools") else parser_response()

    now = datetime.now(UTC)
    agent = OpenRouterSearchAgent(
        settings, transport=SearchBudgetTransport(0.10, transport=transport)
    )
    agent.run_lens(
        lens=load_search_agent_registry(
            PROJECT_ROOT / "tests/fixtures/config/search-agents.yaml"
        ).lenses[0],
        profile=load_interest_profile(settings.interests_file),
        window_start=now - timedelta(days=5),
        window_end=now,
        fixed_signals=[],
        seen_events=[],
    )
    assert len(seen) == 2


def test_research_skips_failed_story_and_fills_batch_without_retry_loop(tmp_path, monkeypatch):
    settings, repo = setup_schedule(tmp_path)
    settings.scheduled_news_per_batch = 3
    ids = [insert_event(repo, score=20 - i) for i in range(6)]
    calls = []

    def research(self, event):
        calls.append(event["id"])
        success = event["id"] != ids[0]
        return {
            "event_id": event["id"],
            "cards_created": int(success),
            "errors": int(not success),
            "cost_usd": 0.01,
        }

    monkeypatch.setattr(ScheduledJobExecutor, "research_event", research)
    result = ScheduledJobExecutor(settings, repo, now=NOW).research()
    assert result["cards_created"] == 3
    assert len(calls) == len(set(calls)) == 4
    assert result["errors"] == 1


def test_research_stops_at_attempt_and_cost_limits(tmp_path, monkeypatch):
    settings, repo = setup_schedule(tmp_path)
    for _ in range(10):
        insert_event(repo, score=20)
    settings.scheduled_research_max_attempts = 2
    calls = []

    def fail(self, event):
        calls.append(event["id"])
        return {"cards_created": 0, "errors": 1, "cost_usd": 0.02}

    monkeypatch.setattr(ScheduledJobExecutor, "research_event", fail)
    job = ScheduledJobExecutor(settings, repo, now=NOW)
    assert job.research()["events_seen"] == 2
    settings.scheduled_research_max_cost_usd = 0
    assert job.research()["events_seen"] == 0
    assert len(calls) == 2


def test_fixed_source_scores_cannot_starve_web_discovery(tmp_path, monkeypatch):
    settings, repo = setup_schedule(tmp_path)
    settings.scheduled_research_max_attempts = 2
    for _ in range(5):
        insert_event(repo, score=15)
    web_id = insert_event(repo, score=6, discovery_mode="web_search")
    calls = []

    def research(self, event):
        calls.append(event["id"])
        return {"cards_created": 0, "errors": 1, "cost_usd": 0}

    monkeypatch.setattr(ScheduledJobExecutor, "research_event", research)
    ScheduledJobExecutor(settings, repo, now=NOW).research()
    assert len(calls) == 2 and calls[0] == web_id


def test_legacy_web_items_are_identified_by_real_source_provenance():
    from app.ops.schedule import is_web_discovery

    assert is_web_discovery({"discovery_mode": "feed", "source_id": "web_agent_space"})
    assert is_web_discovery({"source_ids": ["telegram_ai_newz", "web_agent_geopolitics"]})
    assert not is_web_discovery({"discovery_mode": "search", "source_id": "x_search_agents"})


def test_delivery_diversifies_actual_cards_not_only_candidates(tmp_path):
    _, repo = setup_schedule(tmp_path)
    ids = [insert_event(repo, score=s) for s in (12, 11.9, 11.7)]
    cards = [insert_card(repo, eid) for eid in ids]
    with repo.connection() as db:
        for eid, company in zip(ids, ("OpenAI", "OpenAI", "Rocket Lab"), strict=True):
            db.execute(
                "UPDATE events SET entities_json=? WHERE id=?",
                ('{"organizations":["' + company + '"]}', eid),
            )
    assert fresh_delivery_card_ids(repo, owner_id=123, now=NOW, limit=2) == [cards[0], cards[2]]


def test_feed_health_does_not_confuse_live_process_with_delivered_news(tmp_path):
    _, repo = setup_schedule(tmp_path)
    configure_schedule(repo, enabled=True, dry_run=False)
    now = datetime.now(UTC)
    repo.set_runtime_setting("service_last_tick", now.isoformat())
    for job, metrics in (
        ("radar", {"search": {"lenses_succeeded": 0, "errors": ["404"]}}),
        ("research", {"events_seen": 6, "cards_created": 0}),
    ):
        slot = repo.claim_schedule_slot(job, now.isoformat(), "live")
        repo.update_schedule_slot(slot, status="completed", metrics=metrics)
    snapshot = feed_health(repo, now=now)
    assert snapshot["status"] == "degraded"
    assert snapshot["issues"] == ["search_failed", "news_stalled"]
    assert snapshot["news_24h"] == 0
    assert "YouTube и PDF" in feed_health_text(snapshot)
    configure_schedule(repo, enabled=False, dry_run=False)
    assert feed_health(repo, now=now)["status"] == "paused"


def test_query_does_not_follow_incidental_bigtech_mentions():
    query = build_research_query(
        {
            "headline": "Qwen Code adds remote sessions",
            "source_url": "https://github.com/QwenLM/qwen-code",
            "entities_json": '{"organizations":["OpenAI","Anthropic","Qwen"]}',
            "original_text": "OpenAI Anthropic footer related links",
            "event_type": "software_release",
        }
    )
    assert '"Qwen"' in query
    assert "OpenAI" not in query and "Anthropic" not in query


@pytest.mark.parametrize("status", ["planned", "running", "skipped"])
def test_inflight_radar_does_not_erase_last_successful_search(tmp_path, status):
    _, repo = setup_schedule(tmp_path)
    configure_schedule(repo, enabled=True, dry_run=False)
    now = datetime.now(UTC)
    previous = repo.claim_schedule_slot("radar", (now - timedelta(hours=1)).isoformat(), "live")
    repo.update_schedule_slot(
        previous, status="completed", metrics={"search": {"lenses_succeeded": 2}}
    )
    current = repo.claim_schedule_slot("radar", now.isoformat(), "live")
    repo.update_schedule_slot(current, status=status, metrics={})
    snapshot = feed_health(repo, now=now)
    assert snapshot["latest_search"] == "ok"
    assert snapshot["failed_searches_24h"] == 0
    assert "search_failed" not in snapshot["issues"]


def test_promotional_filter_is_not_general_commercial_topic_ban():
    assert PROMOTIONAL_POST.search("Бесплатный буткэмп для будущих основателей стартапов")
    assert PROMOTIONAL_POST.search("Покупайте по промокоду SUPERAI")
    assert not PROMOTIONAL_POST.search(
        "Компания создала модель для борьбы с рекламным мошенничеством"
    )
