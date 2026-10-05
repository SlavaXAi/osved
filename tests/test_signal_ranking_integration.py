import json
from datetime import UTC, datetime
from pathlib import Path

from app.config import SourceConfig, SourceRegistry
from app.db.repository import Repository
from app.pipeline.ranking import rank_event_candidates
from app.radar.search import compact_fixed_signals
from tests.test_radar_feedback import discovery_delivery, repository


def test_queries_expose_actual_sensor_metadata_without_conferring_evidence(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    delivery = discovery_delivery(repo, index=1)
    with repo.connection() as db:
        db.execute("UPDATE sources SET role='local_media' WHERE id='web_agent_frontier_global'")
        db.execute(
            "UPDATE source_items SET raw_payload_json=?",
            (
                json.dumps(
                    {
                        "verified": True,
                        "evidence_role": "independent",
                        "original_source_url": "https://origin.example/announcement",
                    }
                ),
            ),
        )
        before = [tuple(row) for row in db.execute("SELECT * FROM events")]
    for query in (repo.events_without_cards, repo.events_needing_research):
        candidate = query(limit=10)[0]
        assert candidate["source_id"] == "web_agent_frontier_global"
        assert candidate["source_kind"] == "web"
        assert candidate["discovery_modes"] == ["search"]
        observation = candidate["signal_observations"][0]
        assert observation["cluster_role"] == "primary"
        assert observation["evidence_role"] == "local_media"
        assert "verified" not in observation
        ranked = rank_event_candidates([candidate])[0]
        quality = ranked["score_features"]["signal_quality"]
        assert quality["evidence_status"] == "unverified"
        assert quality["primary_family_count"] == 0
        assert quality["reference_ids"] == [candidate["primary_source_item_id"]]
        assert ranked["score_features"]["spread_bonus"] == 0
        assert "saturation" in ranked["score_features"]
    with repo.connection() as db:
        assert [tuple(row) for row in db.execute("SELECT * FROM events")] == before
        db.execute("UPDATE cards SET validation_status='valid'")
    repo.mark_delivery_sent(delivery, telegram_message_id=999)
    history = repo.recent_valid_card_events()
    assert len(history) == 1
    assert history[0]["source_kind"] == "web"
    assert history[0]["discovery_modes"] == ["search"]
    assert history[0]["score_features"]["signal_quality"]["evidence_status"] == "unverified"


def test_syndicated_mentions_retain_provenance_but_not_independence(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    discovery_delivery(repo, index=1)
    with repo.connection() as db:
        event = dict(db.execute("SELECT * FROM events").fetchone())
        db.execute("UPDATE sources SET role='radar' WHERE id='web_agent_frontier_global'")
        db.execute(
            "UPDATE source_items SET raw_payload_json=?",
            (json.dumps({"original_source_url": "https://origin.example/item"}),),
        )
    second = repo.insert_source_item_with_id(
        source_id="telegram_ai_newz",
        external_id="other",
        source_url="https://t.me/ai_newz/999",
        original_text="Another publisher repeats the announcement",
        original_language="en",
        published_at=datetime.now(UTC).isoformat(),
        outbound_urls=[],
        media_metadata=None,
        payload_hash="other",
        raw_payload={"syndicated_from": "https://origin.example/item"},
    )
    with repo.connection() as db:
        db.execute(
            "INSERT INTO event_source_items(event_id,source_item_id,evidence_role) VALUES(?,?,?)",
            (event["id"], second, "corroborating"),
        )
    enriched = repo.with_event_signal_metadata([event])[0]
    quality = enriched["signal_quality"]
    assert quality["source_count"] == 2
    assert quality["publisher_family_count"] == 1
    assert quality["evidence_status"] == "unverified"
    assert len(quality["reference_ids"]) == 2
    assert enriched["platforms"] == ["telegram", "web"]


def test_fixed_signal_query_and_prompt_cannot_be_crowded_out_by_x(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "mixed.db")
    repo.migrate()
    kinds = {f"x_author_{index}": "x" for index in range(65)}
    kinds.update({kind: kind for kind in ("reddit", "v2ex", "weibo", "zhihu", "youtube")})
    sources = [
        SourceConfig(
            id=source_id,
            name=source_id,
            kind=kind,
            locator=f"https://example.com/{source_id}",
            role="radar",
            region="global",
            language="en",
            admission_mode="trigger_only",
        )
        for source_id, kind in kinds.items()
    ]
    repo.sync_sources(SourceRegistry(version=1, sources=sources))
    for source_id, kind in kinds.items():
        for index in range(2):
            repo.insert_source_item(
                source_id=source_id,
                external_id=str(index),
                source_url=f"https://example.com/{source_id}/{index}",
                original_text="Company launched a production system with new capabilities.",
                original_language="en",
                published_at=("2026-09-07" if kind == "x" else "2026-09-06") + "T12:00:00+00:00",
                outbound_urls=[],
                media_metadata=None,
                payload_hash=f"{source_id}-{index}",
                raw_payload={},
            )
    rows = repo.recent_fixed_signals("2026-09-05T00:00:00+00:00", limit=120)
    assert len(rows) == 120
    assert {row["source_kind"] for row in rows} == set(kinds.values())
    compact = compact_fixed_signals(rows, limit=40)
    assert {row["kind"] for row in compact} == set(kinds.values())
    assert sum(row["kind"] == "x" for row in compact) <= 16
    assert all(row["admission"] == "trigger_only" for row in compact)
    assert compact_fixed_signals(rows, limit=0) == []
