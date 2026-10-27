import json
from pathlib import Path

from app.config import load_source_registry
from app.db.repository import Repository
from app.ops.cycle import run_empty_cycle
from app.research.policy import fact_policy_metadata

PROJECT_ROOT = Path(__file__).parents[1]


def test_migrations_and_source_sync_are_idempotent(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "test.sqlite3")
    registry = load_source_registry(PROJECT_ROOT / "tests/fixtures/config/sources.yaml")

    assert repository.migrate() == [
        "0001_initial",
        "0002_feedback_sessions",
        "0003_runtime_settings",
        "0004_event_dedupe",
        "0005_research_memory",
        "0006_event_dossiers",
        "0007_card_media",
        "0008_schedule_slots",
        "0009_author_drafts",
        "0010_detailed_articles",
        "0011_intelligence_products",
        "0012_sensory_sources",
        "0013_browser_tasks",
        "0014_video_recommendations",
        "0015_video_pdf_jobs",
        "0016_youtube_channels",
        "0017_video_output_modes",
        "0018_telegram_channels",
        "0019_biographies",
        "0020_awareness_preferences",
        "0021_video_relevance",
        "0022_video_panel_feedback",
        "0023_news_reader",
        "0025_news_questions",
        "0027_media_downloads",
        "0028_menu_batches",
        "0029_daily_selection",
        "0030_x_profile_subscriptions",
        "0031_x_reposts",
        "0032_native_x_posts",
    ]
    assert repository.migrate() == []
    assert repository.sync_sources(registry) == 26
    assert repository.sync_sources(registry) == 26

    status = repository.status_counts()
    assert status["sources"] == 26
    assert status["enabled_sources"] == 26
    assert status["candidate_sources"] == 25
    assert status["month_cost_usd"] == 0
    assert status["research_documents"] == 0
    assert status["valid_fact_sheets"] == 0
    assert status["ready_event_dossiers"] == 0
    assert status["preview_media_cards"] == 0
    assert status["valid_author_drafts"] == 0
    yield_rows = repository.source_yield_rows()
    assert len(yield_rows) == 26
    assert all(row["items"] == 0 and row["acceptance_rate"] is None for row in yield_rows)


def test_runtime_setting_is_first_writer_wins(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "runtime.sqlite3")
    repository.migrate()

    assert repository.get_runtime_setting("allowed_user_id") is None
    assert repository.claim_runtime_setting("allowed_user_id", "123") == "123"
    assert repository.claim_runtime_setting("allowed_user_id", "456") == "123"
    assert repository.get_runtime_setting("allowed_user_id") == "123"

    repository.set_runtime_setting("delivery_format", "regular")
    assert repository.get_runtime_setting("delivery_format") == "regular"
    repository.set_runtime_setting("delivery_format", "article")
    assert repository.get_runtime_setting("delivery_format") == "article"


def test_empty_cycle_makes_no_usage_entry(tmp_path: Path, monkeypatch) -> None:
    from app.config import Settings

    monkeypatch.chdir(PROJECT_ROOT)
    settings = Settings(
        app_env="test",
        data_dir=tmp_path,
        sources_file=PROJECT_ROOT / "tests/fixtures/config/sources.yaml",
        interests_file=PROJECT_ROOT / "tests/fixtures/config/interests.yaml",
        telegram_session_path=tmp_path / "telegram-reader.session",
    )
    repository = Repository(settings.database_path)
    repository.migrate()

    result = run_empty_cycle(settings, repository)

    assert result.source_count == 26
    assert result.processed_count == 0
    assert repository.current_month_cost() == 0
    assert repository.status_counts()["runs"] == 1


def test_research_cache_is_separate_from_durable_editorial_memory(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "research.sqlite3")
    repository.migrate()
    repository.sync_sources(
        load_source_registry(PROJECT_ROOT / "tests/fixtures/config/sources.yaml")
    )
    smoke = repository.create_smoke_delivery(123, "https://example.com/release")

    repository.record_research_query(
        event_id=smoke["event_id"],
        provider="context.dev",
        query="example release benchmarks",
        freshness="last_week",
        status="completed",
        results=[{"url": "https://example.com/benchmark"}],
        cost_usd=0.0009,
        expires_at="2026-08-01T00:00:00+00:00",
    )
    document_id = repository.upsert_research_document(
        canonical_url="https://example.com/benchmark",
        url="https://example.com/benchmark?utm_source=test",
        title="Independent benchmark",
        publisher="Example Labs",
        source_class="independent",
        retrieved_at="2026-07-01T00:00:00+00:00",
        expires_at="2026-08-01T00:00:00+00:00",
        content_text="The release scored 61 points.",
    )
    repository.attach_research_document(
        event_id=smoke["event_id"],
        research_document_id=document_id,
        evidence_role="independent",
        relevance=0.9,
    )
    repository.create_fact_sheet(
        event_id=smoke["event_id"],
        schema_version="fact_sheet_v1",
        facts=[{"text": "The release scored 61 points.", "document_ids": [document_id]}],
        coverage={"benchmarks": True, "pricing": False, **fact_policy_metadata()},
        validation_status="valid",
        expires_at="2026-08-01T00:00:00+00:00",
        research_cost_usd=0.0009,
    )

    sheet = repository.latest_valid_fact_sheet(smoke["event_id"])
    assert sheet is not None
    assert sheet["coverage"]["benchmarks"] is True
    assert repository.event_research_documents(smoke["event_id"])[0]["publisher"] == (
        "Example Labs"
    )

    replacement_id = repository.create_fact_sheet(
        event_id=smoke["event_id"],
        schema_version="fact_sheet_v1",
        facts=[{"text": "Replacement fact.", "document_ids": [document_id]}],
        coverage={"benchmarks": True, **fact_policy_metadata()},
        validation_status="valid",
        expires_at="2026-08-01T00:00:00+00:00",
    )
    assert repository.latest_valid_fact_sheet(smoke["event_id"])["id"] == replacement_id
    with repository.connection() as connection:
        assert (
            connection.execute(
                """
            SELECT COUNT(*) FROM event_fact_sheets
            WHERE event_id = ? AND validation_status = 'valid'
            """,
                (smoke["event_id"],),
            ).fetchone()[0]
            == 1
        )

    removed = repository.prune_expired_research_cache(now="2026-08-02T00:00:00+00:00")
    assert removed == {"queries": 1, "fact_sheets": 2, "documents": 1}

    retained_document_id = repository.upsert_research_document(
        canonical_url="https://example.com/retained",
        url="https://example.com/retained",
        source_class="primary",
        retrieved_at="2026-07-01T00:00:00+00:00",
        expires_at="2026-08-01T00:00:00+00:00",
        content_text="Official release notes.",
    )
    repository.attach_research_document(
        event_id=smoke["event_id"],
        research_document_id=retained_document_id,
        evidence_role="primary",
    )
    repository.create_fact_sheet(
        event_id=smoke["event_id"],
        schema_version="fact_sheet_v1",
        facts=[{"text": "Official release notes.", "document_ids": [retained_document_id]}],
        coverage={"release": True},
        validation_status="valid",
        expires_at="2026-08-01T00:00:00+00:00",
    )
    repository.mark_delivery_sent(smoke["delivery_id"], 99)

    retained = repository.prune_expired_research_cache(now="2026-08-02T00:00:00+00:00")
    assert retained == {"queries": 0, "fact_sheets": 0, "documents": 0}


def test_search_snippet_cannot_downgrade_cached_fulltext(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "research.sqlite3")
    repository.migrate()
    canonical_url = "https://example.com/research"
    full_text = "Полный проверенный текст. " * 30
    document_id = repository.upsert_research_document(
        canonical_url=canonical_url,
        url=canonical_url,
        source_class="independent",
        title="Full article title",
        publisher="Full Publisher",
        published_at="2026-08-20T09:00:00+00:00",
        retrieved_at="2026-08-20T10:00:00+00:00",
        expires_at="2026-08-27T10:00:00+00:00",
        content_text=full_text,
        content_hash="full-hash",
        metadata={"content_kind": "scraped_fulltext", "canonicalUrl": canonical_url},
    )

    repeated_id = repository.upsert_research_document(
        canonical_url=canonical_url,
        url=canonical_url,
        source_class="independent",
        title="Search title",
        publisher="Search Publisher",
        retrieved_at="2026-08-21T10:00:00+00:00",
        expires_at="2026-08-22T10:00:00+00:00",
        content_text="Короткое описание из поисковой выдачи.",
        content_hash="snippet-hash",
        metadata={"content_kind": "search_snippet", "search_query": "example"},
    )

    assert repeated_id == document_id
    with repository.connection() as connection:
        row = connection.execute(
            "SELECT * FROM research_documents WHERE id = ?",
            (document_id,),
        ).fetchone()
    assert row is not None
    assert row["content_text"] == full_text
    assert row["content_hash"] == "full-hash"
    assert row["title"] == "Full article title"
    assert row["publisher"] == "Full Publisher"
    assert row["published_at"] == "2026-08-20T09:00:00+00:00"
    assert row["retrieved_at"] == "2026-08-20T10:00:00+00:00"
    assert row["expires_at"] == "2026-08-27T10:00:00+00:00"
    assert json.loads(row["metadata_json"])["content_kind"] == "scraped_fulltext"


def test_derived_rebuild_invalidates_intelligence_products(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "rebuild.sqlite3")
    repository.migrate()
    repository.save_intelligence_product(
        kind="weekly_map",
        generation_key="weekly:test",
        schema_version="intelligence_product_v1",
        period_start="2026-08-18T00:00:00+00:00",
        period_end="2026-08-25T00:00:00+00:00",
        title="Weekly map",
        summary="No material signal.",
        payload={"sections": []},
        rendered_html="<b>Weekly map</b>",
        source_event_ids=[],
        validation_status="insufficient_data",
    )

    removed = repository.rebuild_derived_data()

    assert removed["intelligence_products"] == 1
    assert repository.latest_intelligence_product("weekly_map") is None
