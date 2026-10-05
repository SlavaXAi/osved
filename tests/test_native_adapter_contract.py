from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from app.collectors.base import CollectionRequest
from app.collectors.native_adapters import NativeAdapter
from app.config import Settings, SourceConfig
from tests.test_x_monid_collector import tweet

NOW = datetime(2026, 9, 7, 12, tzinfo=UTC)


def source(kind="x"):
    return SourceConfig(
        id="test_source",
        name="Test source",
        kind=kind,
        locator="example_author",
        role="radar",
        region="global",
        language="en",
        strategy="author_timeline" if kind == "x" else "feed",
    )


@pytest.mark.parametrize("key", ["tweets", "items", "entries", "timeline"])
def test_x_supports_nested_timeline_envelopes(monkeypatch, key):
    row = tweet("12", "A meaningful release with source docs", created_at=NOW.isoformat())
    monkeypatch.setattr(
        "app.collectors.x_monid.fetch_user_timeline",
        lambda *args: ({"timeline": {key: [row]}}, 0.0015),
    )
    result = NativeAdapter(Settings(_env_file=None, app_env="test"), "monid_x_author").collect(
        CollectionRequest(source(), now=NOW)
    )
    assert result.status == "ok"
    assert result.items[0].external_id == "12"
    assert result.cost_usd == 0.0015


def test_x_full_paid_snapshot_is_not_truncated_to_request_limit(monkeypatch):
    rows = [
        tweet(str(index), f"Concrete release {index}", created_at=NOW.isoformat())
        for index in range(1, 21)
    ]
    monkeypatch.setattr(
        "app.collectors.x_monid.fetch_user_timeline", lambda *args: ({"timeline": rows}, 0.0015)
    )
    result = NativeAdapter(Settings(_env_file=None), "monid_x_author").collect(
        CollectionRequest(source(), limit=5, cursor="0", now=NOW)
    )
    assert len(result.items) == 20
    assert result.next_cursor == "20"
    assert result.metadata["complete_snapshot"]
    assert result.metadata["ingest_limit"] == 100


def test_x_one_malformed_row_preserves_good_records_and_cursor(monkeypatch):
    rows = [
        tweet("1", "Good text", created_at=NOW.isoformat()),
        tweet("2", "Bad date", created_at="not a date"),
    ]
    monkeypatch.setattr(
        "app.collectors.x_monid.fetch_user_timeline", lambda *args: ({"timeline": rows}, 0.0015)
    )
    result = NativeAdapter(Settings(_env_file=None), "monid_x_author").collect(
        CollectionRequest(source(), cursor="0", now=NOW)
    )
    assert result.status == "partial"
    assert len(result.items) == 1
    assert result.next_cursor == "0"
    assert result.metadata["malformed"] == 1


@pytest.mark.parametrize(
    "payload,status",
    [
        ({"different_schema": []}, "schema_changed"),
        ({"timeline": [{"unexpected": "shape"}]}, "schema_changed"),
        ({"timeline": []}, "empty"),
    ],
)
def test_x_empty_and_schema_change_are_different(monkeypatch, payload, status):
    monkeypatch.setattr(
        "app.collectors.x_monid.fetch_user_timeline", lambda *args: (payload, 0.0015)
    )
    result = NativeAdapter(Settings(_env_file=None), "monid_x_author").collect(
        CollectionRequest(source(), now=NOW)
    )
    assert result.status == status
    assert result.cost_usd == 0.0015


def test_x_calibration_window_is_respected(monkeypatch):
    row = tweet(
        "1", "Two week old calibration signal", created_at=(NOW - timedelta(days=14)).isoformat()
    )
    monkeypatch.setattr(
        "app.collectors.x_monid.fetch_user_timeline", lambda *args: ({"timeline": [row]}, 0.0015)
    )
    dev = NativeAdapter(Settings(_env_file=None, app_env="test"), "monid_x_author")
    prod = NativeAdapter(Settings(_env_file=None, app_env="production"), "monid_x_author")
    assert len(dev.collect(CollectionRequest(source(), now=NOW)).items) == 1
    assert prod.collect(CollectionRequest(source(), now=NOW)).status == "empty"


@pytest.mark.parametrize(
    "payload,status",
    [
        (b"<html><body>Log in</body></html>", "schema_changed"),
        (b"<rss><channel/></rss>", "empty"),
        (
            b"<rss><channel><item><title>Broken record</title></item></channel></rss>",
            "schema_changed",
        ),
        (b"<!DOCTYPE rss><rss/>", "schema_changed"),
    ],
)
def test_rss_schema_health_is_truthful(monkeypatch, payload, status):
    monkeypatch.setattr("app.collectors.rss.fetch_feed", lambda *args: payload)
    result = NativeAdapter(Settings(_env_file=None), "rss").collect(
        CollectionRequest(source("rss"), now=NOW)
    )
    assert result.status == status
    assert result.cost_usd == 0


def test_github_one_malformed_release_does_not_drop_other_releases(monkeypatch):
    releases = [
        {
            "id": 1,
            "published_at": NOW.isoformat(),
            "html_url": "https://github.com/a/b/releases/tag/1",
            "tag_name": "v1",
        },
        {"id": 2, "published_at": "bad", "html_url": "https://github.com/a/b/releases/tag/2"},
        {"id": 3, "prerelease": True},
    ]
    monkeypatch.setattr("app.collectors.github.fetch_releases", lambda *args: releases)
    result = NativeAdapter(Settings(_env_file=None), "github").collect(
        CollectionRequest(source("github"), now=NOW)
    )
    assert result.status == "partial"
    assert len(result.items) == 1
    assert result.metadata["malformed"] == result.metadata["filtered"] == 1
    assert result.items[0].content_type == "release"


@pytest.mark.parametrize("app_env,after", [("production", 0), ("test", 900)])
def test_telegram_production_reads_latest_window_without_cursor_regression(
    monkeypatch, app_env, after
):
    captured = []

    def fetch(_source, limit, cursor, _path):
        captured.append((limit, cursor))
        return {"messages": [{"id": 100, "date": NOW.isoformat(), "text": "Fresh news"}]}, 0.0015

    monkeypatch.setattr("app.collectors.telegram_monid.fetch_channel_posts", fetch)
    result = NativeAdapter(Settings(_env_file=None, app_env=app_env), "monid_telegram").collect(
        CollectionRequest(source("telegram"), limit=8, cursor="900", now=NOW)
    )
    assert captured == [(8, after)]
    assert result.next_cursor == "900"
    assert result.metadata["coverage"] == (
        "latest_window" if app_env == "production" else "cursor_catchup"
    )
    assert len(result.items) == 1


@pytest.mark.parametrize(
    "rows,status",
    [
        ([{"id": 1, "date": "broken"}], "schema_changed"),
        (
            [{"id": 1, "date": NOW.isoformat(), "text": "ok"}, {"id": 2, "date": "broken"}],
            "partial",
        ),
    ],
)
def test_telegram_malformed_records_do_not_advance_cursor(monkeypatch, rows, status):
    monkeypatch.setattr(
        "app.collectors.telegram_monid.fetch_channel_posts",
        lambda *args: ({"messages": rows}, 0.0015),
    )
    result = NativeAdapter(Settings(_env_file=None), "monid_telegram").collect(
        CollectionRequest(source("telegram"), cursor="0", now=NOW)
    )
    assert result.status == status
    assert result.next_cursor == "0"


@pytest.mark.parametrize(
    "app_env,cursor,expected_min,expected_reverse",
    [
        ("production", "900", 0, False),
        ("test", None, 0, False),
        ("test", "900", 900, True),
    ],
)
def test_mtproto_bootstrap_and_production_read_recent_not_oldest(
    monkeypatch, app_env, cursor, expected_min, expected_reverse
):
    captured = []

    class Client:
        def __init__(self, *args):
            pass

        async def connect(self):
            pass

        async def disconnect(self):
            pass

        async def is_user_authorized(self):
            return True

        async def iter_messages(self, _locator, **kwargs):
            captured.append(kwargs)
            yield SimpleNamespace(id=1000, date=NOW, raw_text="Fresh message", media=None)

    monkeypatch.setattr("telethon.TelegramClient", Client)
    result = NativeAdapter(
        Settings(_env_file=None, app_env=app_env, telegram_api_id=1, telegram_api_hash="test"),
        "telegram_mtproto",
    ).collect(CollectionRequest(source("telegram"), cursor=cursor, now=NOW))
    assert captured[0]["min_id"] == expected_min
    assert captured[0]["reverse"] == expected_reverse
    assert result.status == "ok"
    assert result.next_cursor == "1000"
