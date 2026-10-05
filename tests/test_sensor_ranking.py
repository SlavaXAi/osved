from datetime import UTC, datetime, timedelta

from app.pipeline.ranking import rank_event_candidates


def test_source_platform_and_discovery_fatigue_use_metadata_when_available() -> None:
    history = [{"source_id": "x_author", "source_kind": "x", "discovery_mode": "author"}] * 5
    candidates = [
        {
            "id": 1,
            "score": 7,
            "source_id": "x_author",
            "source_kind": "x",
            "discovery_mode": "author",
        },
        {
            "id": 2,
            "score": 7,
            "source_id": "reddit_topic",
            "source_kind": "reddit",
            "discovery_mode": "search",
        },
    ]
    ranked = rank_event_candidates(candidates, recent_events=history, limit=2)
    assert [row["id"] for row in ranked] == [2, 1]
    assert ranked[1]["saturation_features"]["source_saturation"] < 0
    assert "saturation_features" not in candidates[0]


def test_recent_source_fatigue_decays_and_exceptional_event_survives() -> None:
    now = datetime(2026, 9, 7, tzinfo=UTC)
    event = {"id": 1, "score": 10, "source_id": "repeated"}
    history = [{"source_id": "repeated", "sent_at": (now - timedelta(days=10)).isoformat()}] * 5
    old = rank_event_candidates([event], recent_events=history, now=now)[0]
    fresh = rank_event_candidates([event], recent_events=[{"source_id": "repeated"}] * 5, now=now)[
        0
    ]
    assert old["effective_score"] > fresh["effective_score"]
    ranked = rank_event_candidates([event, {"id": 2, "score": 5}], recent_events=history, limit=1)
    assert ranked[0]["id"] == 1


def test_exploration_is_bounded_and_does_not_rescue_weak_or_invalid_events() -> None:
    candidates = [{"id": index, "score": 7} for index in range(8)]
    candidates += [
        {"id": 20, "score": 6.5, "discovery_mode": "explore"},
        {"id": 21, "score": 2, "discovery_mode": "explore"},
        {"id": 22, "score": float("nan")},
        {"id": 23, "score": "invalid"},
    ]
    ranked = rank_event_candidates(candidates, limit=7)
    assert len(ranked) == 7
    assert 20 in [row["id"] for row in ranked]
    assert not {21, 22, 23} & {row["id"] for row in ranked}
    assert sum(row["saturation_features"].get("exploration_slot", 0) for row in ranked) == 1
    assert rank_event_candidates([{"id": 1, "score": 2}], limit=10) == []
