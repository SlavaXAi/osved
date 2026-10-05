from datetime import UTC, datetime
from pathlib import Path

from app.collectors.telegram import (
    canonical_http_url,
    extract_message_urls,
    message_to_item,
    telegram_post_url,
)
from app.config import load_source_registry
from app.db.repository import Repository

PROJECT_ROOT = Path(__file__).parents[1]


class TextUrlEntity:
    def __init__(self, url: str) -> None:
        self.url = url


class UrlEntity:
    pass


class FakeMessage:
    id = 42
    date = datetime(2026, 8, 13, 10, 30, tzinfo=UTC)
    raw_text = "Read https://example.com/news?utm_source=tg and the linked report"
    text = raw_text
    media = None
    photo = None
    video = None
    document = None
    grouped_id = None
    forward = None
    edit_date = None
    web_preview = None

    def get_entities_text(self):
        return [
            (TextUrlEntity("https://primary.example/report#section"), "linked report"),
            (UrlEntity(), "https://example.com/news?utm_source=tg"),
        ]


def test_url_extraction_deduplicates_entities_and_text() -> None:
    assert extract_message_urls(FakeMessage()) == [
        "https://primary.example/report",
        "https://example.com/news",
    ]


def test_canonical_url_rejects_non_http_and_removes_fragment() -> None:
    assert canonical_http_url("tg://resolve?domain=test") is None
    assert canonical_http_url("https://EXAMPLE.com/a#b") == "https://example.com/a"


def test_message_normalization_and_database_idempotency(tmp_path: Path) -> None:
    registry = load_source_registry(PROJECT_ROOT / "tests/fixtures/config/sources.yaml")
    source = next(source for source in registry.sources if source.locator == "ai_newz")
    item = message_to_item(source, FakeMessage())

    assert item.source_url == telegram_post_url("ai_newz", 42)
    assert item.external_id == "42"
    assert item.payload_hash

    repository = Repository(tmp_path / "collector.sqlite3")
    repository.migrate()
    repository.sync_sources(registry)
    kwargs = {
        "source_id": source.id,
        "external_id": item.external_id,
        "source_url": item.source_url,
        "original_text": item.original_text,
        "original_language": item.original_language,
        "published_at": item.published_at,
        "outbound_urls": item.outbound_urls,
        "media_metadata": item.media_metadata,
        "payload_hash": item.payload_hash,
        "raw_payload": item.raw_payload,
    }
    assert repository.insert_source_item(**kwargs)
    assert not repository.insert_source_item(**kwargs)
    assert repository.source_item_count() == 1
