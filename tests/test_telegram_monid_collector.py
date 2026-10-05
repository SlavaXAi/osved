from pathlib import Path

import pytest

from app.collectors.telegram_monid import (
    MONID_CALL_COST_USD,
    MonidTelegramCollector,
    extract_monid_urls,
    monid_message_to_item,
)
from app.config import Settings, SourceKind, load_source_registry
from app.db.repository import Repository

PROJECT_ROOT = Path(__file__).parents[1]


def sample_message(message_id: int = 314) -> dict:
    return {
        "id": message_id,
        "url": f"https://t.me/prompt_design/{message_id}",
        "type": "photo",
        "date": "2026-08-13T14:21:22+00:00",
        "text": "Релиз: https://example.com/release?utm_source=telegram",
        "text_html": (
            'Подробнее в <a href="https://github.com/example/project#readme">репозитории</a>'
        ),
        "media": {
            "photos": ["https://temporary.example/signed.jpg"],
            "videos": [],
            "has_unsupported": False,
        },
    }


def test_monid_message_normalization() -> None:
    source = next(
        source
        for source in load_source_registry(
            PROJECT_ROOT / "tests/fixtures/config/sources.yaml"
        ).sources
        if source.locator == "prompt_design"
    )
    message = sample_message()

    assert extract_monid_urls(message) == [
        "https://example.com/release",
        "https://github.com/example/project",
    ]
    item = monid_message_to_item(source, message)
    assert item.external_id == "314"
    assert item.media_metadata == {
        "kind": "photo",
        "photo_count": 1,
        "video_count": 0,
        "has_unsupported": False,
    }


def test_monid_collector_stores_items_cursors_and_cost(tmp_path: Path) -> None:
    settings = Settings(
        app_env="test",
        data_dir=tmp_path,
        sources_file=PROJECT_ROOT / "tests/fixtures/config/sources.yaml",
        interests_file=PROJECT_ROOT / "tests/fixtures/config/interests.yaml",
        telegram_session_path=tmp_path / "telegram-reader.session",
    )
    repository = Repository(settings.database_path)
    repository.migrate()
    registry = load_source_registry(settings.sources_file)
    repository.sync_sources(registry)
    calls: list[tuple[str, int, int]] = []

    def fake_fetcher(source, limit, after, output_path):
        calls.append((source.id, limit, after))
        return {"messages": [sample_message()]}, MONID_CALL_COST_USD

    stats = MonidTelegramCollector(
        settings,
        repository,
        fetcher=fake_fetcher,
    ).collect(limit_per_source=5)

    assert stats.sources_attempted == 7
    assert stats.sources_succeeded == 7
    assert stats.items_seen == 7
    assert stats.items_inserted == 7
    assert stats.errors == 0
    assert stats.cost_usd == pytest.approx(7 * MONID_CALL_COST_USD)
    assert repository.source_item_count() == 7
    assert repository.current_month_cost() == pytest.approx(7 * MONID_CALL_COST_USD)
    assert all(limit == 5 and after == 0 for _, limit, after in calls)
    assert all(
        repository.get_source_cursor(source.id) == "314"
        for source in registry.sources
        if source.kind == SourceKind.TELEGRAM
    )
