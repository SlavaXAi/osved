import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.collectors.x_monid import (
    MONID_CALL_COST_USD,
    QUOTED_POST_LABEL,
    MonidXCollector,
    timeline_to_items,
)
from app.config import Settings, SourceKind, runtime_source_registry
from app.db.repository import Repository

NOW = datetime(2026, 8, 23, 12, 0, tzinfo=UTC)


def make_settings(tmp_path: Path, *, tier: str = "active") -> Settings:
    sources_file = tmp_path / "sources.yaml"
    sources_file.write_text("version: 1\nsources: []\n", encoding="utf-8")
    authors_file = tmp_path / "x-authors.yaml"
    authors_file.write_text(
        f"""version: 1
authors:
  - handle: example_author
    name: Example Author
    topics: [ai, robotics]
    initial_tier: {tier}
    enabled: true
    weekly_persona: false
""",
        encoding="utf-8",
    )
    return Settings(
        app_env="test",
        data_dir=tmp_path,
        sources_file=sources_file,
        x_authors_file=authors_file,
        telegram_session_path=tmp_path / "telegram-reader.session",
    )


def make_repository(settings: Settings) -> Repository:
    repository = Repository(settings.database_path)
    repository.migrate()
    repository.sync_sources(runtime_source_registry(settings))
    return repository


def tweet(
    identifier: str,
    text: str,
    *,
    conversation_id: str | None = None,
    created_at: str = "Sat Aug 22 10:00:00 +0000 2026",
    **extra,
) -> dict:
    return {
        "id_str": identifier,
        "conversation_id_str": conversation_id or identifier,
        "full_text": text,
        "created_at": created_at,
        "user": {"id_str": "42", "screen_name": "example_author"},
        **extra,
    }


def test_cadence_skips_paid_call_unless_forced(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    repository = make_repository(settings)
    source = next(
        source
        for source in runtime_source_registry(settings).sources
        if source.kind == SourceKind.X
    )
    repository.mark_source_success(
        source.id,
        cursor="100",
        success_at="2026-08-23T11:30:00+00:00",
    )
    calls: list[str] = []

    def fake_fetcher(source, output_path):
        calls.append(source.id)
        return {"timeline": [], "user": {}, "next_cursor": None}, MONID_CALL_COST_USD

    collector = MonidXCollector(
        settings,
        repository,
        fetcher=fake_fetcher,
        clock=lambda: NOW,
    )
    skipped = collector.collect()
    forced = collector.collect(force=True)

    assert skipped.sources_attempted == 0
    assert skipped.cost_usd == 0
    assert calls == [source.id]
    assert forced.sources_attempted == forced.sources_succeeded == 1
    assert repository.current_month_cost() == pytest.approx(MONID_CALL_COST_USD)


def test_legacy_author_collector_never_calls_search_or_browser_sources(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    settings.sources_file.write_text(
        """version: 1
sources:
  - id: accidental_query
    name: X topic query using default strategy
    kind: x
    locator: AI agents launched
    adapter_id: x_search_monid
    strategy: feed
    role: radar
    region: global
    language: en
  - id: for_you
    name: X browser
    kind: x
    locator: https://x.com/home
    adapter_id: browser_x
    strategy: for_you
    role: radar
    region: global
    language: en
  - id: invalid_auto_author
    name: Misconfigured default X source
    kind: x
    locator: robot startup launch
    role: radar
    region: global
    language: en
  - id: valid_legacy_author
    name: Existing direct account source
    kind: x
    locator: valid_author
    role: author_radar
    region: global
    language: en
""",
        encoding="utf-8",
    )
    repository = make_repository(settings)
    calls: list[str] = []

    def fetcher(source, output_path):
        calls.append(source.locator)
        return {"timeline": [], "user": {}}, MONID_CALL_COST_USD

    result = MonidXCollector(settings, repository, fetcher=fetcher, clock=lambda: NOW).collect(
        force=True
    )
    assert set(calls) == {"example_author", "valid_author"}
    assert result.sources_attempted == 2
    assert result.cost_usd == pytest.approx(2 * MONID_CALL_COST_USD)
    assert repository.current_month_cost() == pytest.approx(2 * MONID_CALL_COST_USD)


def test_thread_grouping_expands_urls_and_keeps_reply_trace(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    source = next(
        source
        for source in runtime_source_registry(settings).sources
        if source.kind == SourceKind.X
    )
    root = tweet(
        "201",
        "We shipped the robot stack https://t.co/docs",
        conversation_id="201",
        entities={
            "urls": {
                "url": "https://t.co/docs",
                "expanded_url": "https://example.com/robot?utm_source=x",
            },
            "media": [
                {
                    "type": "photo",
                    "media_url_https": "https://cdn.example.com/robot.jpg",
                }
            ],
        },
    )
    reply = tweet(
        "202",
        "It now works in a factory https://t.co/demo",
        conversation_id="201",
        created_at="Sat Aug 22 10:05:00 +0000 2026",
        reply_to="201",
        entities=[
            {
                "url": "https://t.co/demo",
                "expandedUrl": "https://example.com/demo?utm_medium=social",
            }
        ],
    )

    items, observed = timeline_to_items(source, [reply, root], user={}, now=NOW)

    assert observed == ["202", "201"]
    assert len(items) == 1
    item = items[0]
    assert item.external_id == "201"
    assert item.source_url == "https://x.com/example_author/status/201"
    assert "https://example.com/robot" in item.original_text
    assert "https://example.com/demo" in item.original_text
    assert item.outbound_urls == [
        "https://example.com/robot",
        "https://example.com/demo",
    ]
    assert item.media_metadata == {
        "count": 1,
        "items": [{"type": "photo", "url": "https://cdn.example.com/robot.jpg"}],
        "thread": True,
    }
    assert item.raw_payload["tweet_ids"] == ["201", "202"]
    assert item.raw_payload["replies"] == [
        {
            "id": "202",
            "reply_to_tweet_id": "201",
            "reply_to_user_id": None,
            "reply_to_handle": None,
        }
    ]


def test_tikhub_quote_adds_labeled_context_links_and_media_without_duplicates(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    source = next(
        source
        for source in runtime_source_registry(settings).sources
        if source.kind == SourceKind.X
    )
    quote = {
        "tweet_id": "800",
        "conversation_id": "800",
        "text": "This changes inference economics.",
        "created_at": "Sat Aug 22 10:00:00 +0000 2026",
        "entities": {
            "urls": [
                {
                    "expanded_url": "https://example.com/paper?utm_source=x",
                    "url": "https://t.co/shared",
                }
            ]
        },
        "media": [],
        "quoted": {
            "tweet_id": "801",
            "conversation_id": "801",
            "text": "The original benchmark is here https://t.co/shared",
            "created_at": "Sat Aug 22 09:00:00 +0000 2026",
            "entities": {
                "urls": [
                    {
                        "expanded_url": "https://example.com/paper?utm_medium=quote",
                        "url": "https://t.co/shared",
                    }
                ]
            },
            "media": {
                "photo": [
                    {
                        "media_url_https": "https://pbs.twimg.com/media/quoted.jpg",
                        "id": "7001",
                    }
                ]
            },
        },
    }

    items, _ = timeline_to_items(source, [quote], user={}, now=NOW)

    assert len(items) == 1
    item = items[0]
    assert item.original_text.count(QUOTED_POST_LABEL) == 1
    assert "This changes inference economics." in item.original_text
    assert "The original benchmark is here https://example.com/paper" in item.original_text
    assert item.original_text.count("The original benchmark") == 1
    assert item.outbound_urls == [
        "https://example.com/paper",
        "https://x.com/i/web/status/801",
    ]
    assert item.media_metadata == {
        "count": 1,
        "items": [
            {
                "type": "photo",
                "url": "https://pbs.twimg.com/media/quoted.jpg",
                "context": "quoted",
            }
        ],
        "thread": False,
    }


def test_tikhub_photo_and_video_containers_preserve_preview_and_best_mp4(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    source = next(
        source
        for source in runtime_source_registry(settings).sources
        if source.kind == SourceKind.X
    )
    with_media = {
        "tweet_id": "850",
        "conversation_id": "850",
        "text": "Robot demonstration",
        "created_at": "Sat Aug 22 10:00:00 +0000 2026",
        "entities": [],
        "media": {
            "photo": [
                {
                    "media_url_https": "https://pbs.twimg.com/media/photo.jpg",
                    "id": "7101",
                }
            ],
            "video": [
                {
                    "media_url_https": "https://pbs.twimg.com/media/video-preview.jpg",
                    "variants": [
                        {
                            "content_type": "application/x-mpegURL",
                            "url": "https://video.twimg.com/demo/list.m3u8",
                        },
                        {
                            "bitrate": 256000,
                            "content_type": "video/mp4",
                            "url": "https://video.twimg.com/demo/low.mp4?tag=1",
                        },
                        {
                            "bitrate": 2176000,
                            "content_type": "video/mp4",
                            "url": "https://video.twimg.com/demo/high.mp4?tag=1",
                        },
                    ],
                    "id": "7102",
                }
            ],
        },
    }

    items, _ = timeline_to_items(source, [with_media], user={}, now=NOW)

    assert items[0].media_metadata == {
        "count": 2,
        "items": [
            {"type": "photo", "url": "https://pbs.twimg.com/media/photo.jpg"},
            {
                "type": "video",
                "url": "https://pbs.twimg.com/media/video-preview.jpg",
                "content_url": "https://video.twimg.com/demo/high.mp4?tag=1",
            },
        ],
        "thread": False,
    }


def test_pinned_plain_retweet_and_stale_posts_are_skipped_but_quote_is_kept(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    source = next(
        source
        for source in runtime_source_registry(settings).sources
        if source.kind == SourceKind.X
    )
    pinned = tweet("301", "Pinned launch", is_pinned=True)
    retweet = tweet("302", "RT @someone: generic repost")
    stale = tweet(
        "303",
        "Old but ordinary",
        created_at="Sat Aug 01 10:00:00 +0000 2026",
    )
    quote = tweet(
        "304",
        "This changes inference economics https://t.co/quoted",
        is_quote_status=True,
        quoted_tweet={"id_str": "777", "full_text": "Original announcement"},
        entities={
            "urls": [
                {
                    "url": "https://t.co/quoted",
                    "expanded_url": "https://x.com/company/status/777",
                }
            ]
        },
    )

    items, observed = timeline_to_items(
        source,
        [pinned, retweet, stale, quote],
        user={},
        now=NOW,
    )

    assert observed == ["301", "302", "303", "304"]
    assert [item.external_id for item in items] == ["304"]
    assert items[0].outbound_urls == ["https://x.com/company/status/777"]
    assert "changes inference economics" in items[0].original_text


def test_collection_is_idempotent_and_saves_string_max_tweet_cursor(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    repository = make_repository(settings)
    payload = {
        "timeline": [
            tweet("90071992547409931", "First signal"),
            tweet(
                "90071992547409939",
                "Second signal https://t.co/x",
                entities={
                    "urls": [
                        {
                            "url": "https://t.co/x",
                            "expanded_url": "https://example.com/evidence",
                        }
                    ]
                },
            ),
        ],
        "user": {"id_str": "42", "screen_name": "example_author"},
        "next_cursor": "opaque-cursor",
    }
    calls = 0

    def fake_fetcher(source, output_path):
        nonlocal calls
        calls += 1
        return payload, MONID_CALL_COST_USD

    collector = MonidXCollector(
        settings,
        repository,
        fetcher=fake_fetcher,
        clock=lambda: NOW,
    )
    first = collector.collect(force=True)
    second = collector.collect(force=True)
    source_id = "x_example_author"

    assert calls == 2
    assert first.items_seen == 2
    assert first.items_inserted == 2
    assert second.items_seen == 2
    assert second.items_inserted == 0
    assert repository.source_item_count() == 2
    assert repository.get_source_cursor(source_id) == "90071992547409939"
    assert repository.current_month_cost() == pytest.approx(2 * MONID_CALL_COST_USD)

    with repository.connection() as connection:
        rows = connection.execute(
            """
            SELECT external_id, outbound_urls_json, raw_payload_json
            FROM source_items
            ORDER BY external_id
            """
        ).fetchall()
    assert all(isinstance(row["external_id"], str) for row in rows)
    assert json.loads(rows[1]["outbound_urls_json"]) == ["https://example.com/evidence"]
    assert json.loads(rows[1]["raw_payload_json"])["type"] == "x_post"


def test_existing_thread_row_expands_when_same_root_gains_a_reply(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    repository = make_repository(settings)
    root = tweet("901", "Initial thread root", conversation_id="901")
    reply = tweet(
        "902",
        "A material follow-up with evidence",
        conversation_id="901",
        created_at="Sat Aug 22 10:05:00 +0000 2026",
        reply_to="901",
    )
    payloads = [
        {"timeline": [root], "user": {}, "next_cursor": None},
        {"timeline": [reply, root], "user": {}, "next_cursor": None},
    ]

    def fake_fetcher(source, output_path):
        return payloads.pop(0), MONID_CALL_COST_USD

    collector = MonidXCollector(
        settings,
        repository,
        fetcher=fake_fetcher,
        clock=lambda: NOW,
    )
    first = collector.collect(force=True)
    with repository.connection() as connection:
        original = connection.execute(
            "SELECT id, payload_hash FROM source_items WHERE external_id = '901'"
        ).fetchone()
        connection.execute(
            "UPDATE source_items SET processing_status = 'event_attached' WHERE id = ?",
            (original["id"],),
        )

    second = collector.collect(force=True)

    assert first.items_inserted == 1
    assert second.items_inserted == 0
    assert repository.source_item_count() == 1
    with repository.connection() as connection:
        updated = connection.execute(
            """
            SELECT id, original_text, published_at, payload_hash,
                   raw_payload_json, processing_status
            FROM source_items
            WHERE external_id = '901'
            """
        ).fetchone()
        usage = connection.execute(
            "SELECT metadata_json FROM usage_ledger ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert updated["id"] == original["id"]
    assert updated["processing_status"] == "event_attached"
    assert updated["payload_hash"] != original["payload_hash"]
    assert updated["original_text"].count("Initial thread root") == 1
    assert updated["original_text"].count("material follow-up") == 1
    assert updated["published_at"] == "2026-08-22T10:05:00+00:00"
    assert json.loads(updated["raw_payload_json"])["tweet_ids"] == ["901", "902"]
    assert json.loads(usage["metadata_json"])["updated_items"] == 1
