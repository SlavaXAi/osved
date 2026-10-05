import json
from pathlib import Path

from app.config import load_source_registry
from app.db.repository import Repository
from app.pipeline.dedupe import (
    canonical_evidence_url,
    conflicting_versions,
    deduplicate_events,
    event_match_reason,
    headline_similarity,
    policy_stage,
    safe_shared_evidence,
)

PROJECT_ROOT = Path(__file__).parents[1]


def test_headline_similarity_ignores_common_noise() -> None:
    assert (
        headline_similarity(
            "DeepSeek выпустили собственный харнес и V4 Pro из превью",
            "DeepSeek выкатил V4-Pro и открыл собственный агентский харнес",
        )
        >= 0.45
    )
    assert (
        headline_similarity(
            "DeepSeek выпустили собственный харнес и V4 Pro",
            "Яндекс Роботикс автоматизирует склады",
        )
        == 0
    )


def test_dedupe_merges_event_evidence(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "dedupe.sqlite3")
    repository.migrate()
    registry = load_source_registry(PROJECT_ROOT / "tests/fixtures/config/sources.yaml")
    repository.sync_sources(registry)
    headlines = [
        "DeepSeek выпустили собственный харнес и V4 Pro из превью",
        "DeepSeek выкатил V4-Pro и открыл собственный агентский харнес",
    ]
    event_ids: list[int] = []
    for index, headline in enumerate(headlines):
        repository.insert_source_item(
            source_id=["telegram_ai_newz", "telegram_prompt_design"][index],
            external_id=str(100 + index),
            source_url=f"https://t.me/example/{100 + index}",
            original_text=headline * 4,
            original_language="ru",
            published_at="2026-08-13T12:00:00+00:00",
            outbound_urls=[f"https://example.com/{index}"],
            media_metadata=None,
            payload_hash=f"hash-{index}",
            raw_payload={},
        )
        item_id = repository.unprocessed_source_items()[0]["id"]
        event_id, _ = repository.attach_item_to_event(
            source_item_id=item_id,
            event_type="telegram_signal",
            headline=headline,
            event_at="2026-08-13T12:00:00+00:00",
            primary_url=f"https://example.com/{index}",
            fingerprint=f"fingerprint-{index}",
            score=float(9 - index),
            score_features={},
            decision_reason="test",
            entities={"organizations": ["DeepSeek"]},
            topics=[["ai"], ["agents"]][index],
            regions=[["global"], ["china"]][index],
        )
        event_ids.append(event_id)

    for index, event_id in enumerate(event_ids):
        document_id = repository.upsert_research_document(
            canonical_url=f"https://evidence.example/{index}",
            url=f"https://evidence.example/{index}",
            title=f"Evidence {index}",
            publisher="Evidence",
            source_class="independent",
            retrieved_at="2026-08-13T13:00:00+00:00",
            expires_at="2026-09-13T13:00:00+00:00",
            content_text="Independent evidence for the DeepSeek release.",
        )
        repository.attach_research_document(
            event_id=event_id,
            research_document_id=document_id,
            evidence_role="independent",
            relevance=0.9,
        )

    stats = deduplicate_events(repository)

    assert stats.events_seen == 2
    assert stats.events_merged == 1
    with repository.connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM event_source_items").fetchone()[0] == 2
        row = connection.execute(
            "SELECT entities_json, topics_json, regions_json, score FROM events"
        ).fetchone()
        assert set(json.loads(row["topics_json"])) == {"ai", "agents"}
        assert set(json.loads(row["regions_json"])) == {"global", "china"}
        assert json.loads(row["entities_json"])["organizations"] == ["DeepSeek"]
        assert float(row["score"]) == 9.75
        assert (
            connection.execute("SELECT COUNT(*) FROM event_research_documents").fetchone()[0] == 2
        )


def test_digest_urls_do_not_merge_unrelated_event() -> None:
    digest = {
        "outbound_urls_json": "["
        + ",".join(f'"https://example.com/story-{index}"' for index in range(6))
        + "]"
    }
    single = {"outbound_urls_json": '["https://example.com/story-1"]'}

    assert safe_shared_evidence(digest, single) == set()


def test_canonical_evidence_preserves_article_ids_and_video_ids() -> None:
    assert (
        canonical_evidence_url("https://example.org/?id=1&utm_campaign=test") == "example.org?id=1"
    )
    assert canonical_evidence_url("https://example.org/?id=2") == "example.org?id=2"
    assert (
        canonical_evidence_url("https://youtube.com/watch?v=first&t=5")
        == "youtube.com/watch?v=first"
    )
    assert (
        canonical_evidence_url("https://youtu.be/first?si=tracking") == "youtube.com/watch?v=first"
    )
    assert canonical_evidence_url("https://youtube.com/watch?v=second") != canonical_evidence_url(
        "https://youtu.be/first"
    )
    assert (
        canonical_evidence_url("https://www.example.org/news/Story-A/?utm_source=x")
        == "example.org/news/Story-A"
    )
    assert canonical_evidence_url("https://x.com/example/status/123") == canonical_evidence_url(
        "https://twitter.com/other/status/123"
    )


def test_generic_homepages_profiles_and_latest_release_cannot_merge_events() -> None:
    for url in (
        "https://openai.com/",
        "https://example.org/news",
        "https://x.com/sama",
        "https://github.com/org/repo",
        "https://github.com/org/repo/releases/latest",
        "https://www.youtube.com/@channel",
        "https://t.me/channel/12",
    ):
        assert canonical_evidence_url(url) is None
        assert (
            event_match_reason(
                {"headline": "Новый космический запуск", "primary_url": url},
                {"headline": "Исследование сознания животных", "primary_url": url},
            )
            is None
        )


def test_multilingual_same_article_merges_but_different_model_versions_do_not() -> None:
    assert (
        event_match_reason(
            {"headline": "中国公司开源新模型", "primary_url": "https://company.cn/news/model-a"},
            {
                "headline": "Китайская компания открыла модель",
                "outbound_urls_json": '["https://company.cn/news/model-a?utm_source=tg"]',
            },
        )
        == "shared_specific_source_url"
    )
    assert conflicting_versions("OpenAI GPT-5 launch", "OpenAI GPT-6 launch")
    assert not conflicting_versions("DeepSeek V4-Pro release", "DeepSeek V4 Pro released")
    assert (
        event_match_reason(
            {"headline": "OpenAI GPT-5 launch", "primary_url": "https://example.org/rolling-story"},
            {"headline": "OpenAI GPT-6 launch", "primary_url": "https://example.org/rolling-story"},
        )
        is None
    )


def test_invalid_url_payload_and_generic_headlines_are_not_matches() -> None:
    assert safe_shared_evidence({"outbound_urls_json": "not json"}, {}) == set()
    assert canonical_evidence_url("https://[invalid") is None
    assert (
        event_match_reason(
            {"headline": "Компания представила новую модель искусственного интеллекта"},
            {"headline": "Другая компания представила новую модель искусственного интеллекта"},
        )
        is None
    )


def test_policy_stages_prevent_merging_distinct_developments():
    url = "https://example.com/rolling-sanctions-report"
    for headline, expected in (
        ("Trump signs Russia sanctions bill", "signed"),
        ("Trump plans to sign Russia sanctions bill", "pending_signature"),
        ("Trump will sign Russia sanctions bill", "pending_signature"),
        ("House passes Russia sanctions bill", "house_passed"),
        ("Senate passes Russia sanctions bill", "senate_passed"),
        ("Трамп подпишет закон о санкциях", "pending_signature"),
    ):
        row = {
            "headline": headline,
            "source_url": url,
            "original_text": "Earlier Trump signed another law.",
        }
        assert policy_stage(row) == expected
        if expected != "signed":
            assert (
                event_match_reason(
                    row,
                    {"headline": "Trump signs Russia sanctions bill", "source_url": url},
                )
                is None
            )


def test_unpublished_duplicate_keeps_history_and_redirects_future_evidence(tmp_path):
    repository = Repository(tmp_path / "alias.sqlite3")
    repository.migrate()
    repository.sync_sources(
        load_source_registry(PROJECT_ROOT / "tests/fixtures/config/sources.yaml")
    )
    urls = [
        "https://theguardian.com/us-news/2026/sep/18/trump-signs-russia-sanctions-bill",
        "https://reuters.com/world/us/trump-signs-russia-sanctions-bill-into-law-2026-09-18/",
    ]
    ids = []
    for index, url in enumerate(urls):
        repository.insert_source_item(
            source_id="telegram_ai_newz",
            external_id=f"policy-{index}",
            source_url=url,
            original_text="Trump signs Russia sanctions bill into law.",
            original_language="en",
            published_at="2026-09-18T12:00:00+00:00",
            outbound_urls=[],
            media_metadata=None,
            payload_hash=f"policy-{index}",
            raw_payload={},
        )
        with repository.connection() as db:
            item_id = db.execute("SELECT MAX(id) FROM source_items").fetchone()[0]
        event_id, _ = repository.attach_item_to_event(
            source_item_id=item_id,
            event_type="web_signal",
            headline=(
                "США перевели расширение санкционного давления в законодательный механизм"
                if index == 0
                else "США перевели санкционное давление в законодательную основу"
            ),
            event_at="2026-09-18T12:00:00+00:00",
            primary_url=url,
            fingerprint=f"policy-{index}",
            score=8,
            score_features={},
            decision_reason="test",
        )
        ids.append(event_id)
        repository.create_candidate_preview_card(
            event_id=event_id,
            prompt_version="test",
            model="test",
            generation_cost_usd=0,
            structured_output={},
            rendered_html="Saved preview",
            source_urls=[url],
            primary_preview_url=url,
        )
        repository.create_fact_sheet(
            event_id=event_id,
            schema_version="test",
            facts=[],
            coverage={},
            validation_status="invalid",
            expires_at="2099-01-01T00:00:00+00:00",
        )
        repository.set_event_status(event_id, "research_insufficient")
    stats = deduplicate_events(repository)
    assert stats.events_merged == 1
    assert repository.get_runtime_setting(f"duplicate_event:{ids[1]}") == str(ids[0])
    assert repository.event_context(ids[1])["status"] == "duplicate"
    with repository.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 2
        assert db.execute("SELECT COUNT(*) FROM cards").fetchone()[0] == 2
        assert db.execute("SELECT COUNT(*) FROM event_fact_sheets").fetchone()[0] == 2
        assert (
            db.execute(
                "SELECT COUNT(*) FROM event_source_items WHERE event_id=?", (ids[0],)
            ).fetchone()[0]
            == 2
        )
    # A later collection with the retained duplicate fingerprint stays canonical.
    target_id, created = repository.attach_item_to_event(
        source_item_id=item_id,
        event_type="web_signal",
        headline="Same story again",
        event_at="2026-09-18T13:00:00+00:00",
        primary_url=urls[1],
        fingerprint="policy-1",
        score=8,
        score_features={},
        decision_reason="repeat",
    )
    assert target_id == ids[0] and not created
    assert deduplicate_events(repository).events_merged == 0
    assert not repository.link_duplicate_event(ids[1], ids[0], reason="cannot form cycle")


def test_duplicate_link_never_reassigns_a_delivered_card(tmp_path):
    repository = Repository(tmp_path / "delivered.sqlite3")
    repository.migrate()
    repository.sync_sources(
        load_source_registry(PROJECT_ROOT / "tests/fixtures/config/sources.yaml")
    )
    one = repository.create_smoke_delivery(123, "https://example.org/one")
    two = repository.create_smoke_delivery(123, "https://example.org/two")
    assert not repository.link_duplicate_event(one["event_id"], two["event_id"], reason="test")
    assert repository.get_delivery_context(two["delivery_id"])["event_id"] == two["event_id"]
