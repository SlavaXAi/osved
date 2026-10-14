from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import SecretStr

from app.collectors.base import CollectionRequest
from app.collectors.orchestrator import collect_due_sources
from app.collectors.platform_transport import MonidResponse, PlatformError
from app.collectors.platforms import YouTubeRSSAdapter
from app.collectors.youtube_fallback import (
    CALL_PRICE,
    LIST_ENDPOINT,
    YouTubeFallback,
)
from app.config import Settings, SourceConfig, SourceRegistry
from app.db.repository import Repository
from app.video.channels import ChannelError, ChannelService

CHANNEL = "UC" + "a" * 22
VIDEO = "abcdefghijk"


def listing(channel=CHANNEL, videos=None):
    return {
        "channel": {"id": channel, "name": "Test Channel"},
        "videos": videos
        if videos is not None
        else [{"video_id": VIDEO, "published_time": "1 hour ago", "is_live": False}],
    }


def detail(now, *, video=VIDEO, channel=CHANNEL, published=None):
    return {
        "playerResponse": {
            "videoDetails": {
                "videoId": video,
                "channelId": channel,
                "title": "New release",
                "shortDescription": "New hardware https://example.com/release",
                "author": "Test Channel",
            },
            "microformat": {
                "playerMicroformatRenderer": {
                    "publishDate": published or (now - timedelta(hours=1)).isoformat()
                }
            },
        }
    }


@pytest.fixture
def env(tmp_path):
    settings = Settings(
        _env_file=None,
        app_env="test",
        data_dir=tmp_path,
        monid_api_key=SecretStr("fake"),
        youtube_rss_fallback_enabled=True,
        sensory_sources_file=Path(__file__).parent / "fixtures/config/sensory-sources.yaml",
    )
    repository = Repository(settings.database_path)
    repository.migrate()
    now = datetime.now(UTC)
    source = SourceConfig(
        id="test_youtube",
        name="Test",
        kind="youtube",
        locator=CHANNEL,
        role="radar",
        region="global",
        language="en",
        adapter_id="youtube_rss",
        strategy="channel_uploads",
        options={"not_before": (now - timedelta(hours=2)).isoformat()},
    )
    calls = []

    def paid(endpoint, query):
        calls.append((endpoint, query))
        payload = listing() if endpoint == LIST_ENDPOINT else detail(now, video=query["video_id"])
        return MonidResponse(payload, CALL_PRICE, "test-run")

    fallback = YouTubeFallback(settings, repository, paid=paid)
    return settings, repository, fallback, calls, CollectionRequest(source, limit=100, now=now)


def unavailable(url):
    raise PlatformError("Platform HTTP 404")


def test_fallback_exact_metadata_and_durable_cache(env):
    settings, repo, fallback, calls, request = env
    adapter = YouTubeRSSAdapter(unavailable, fallback=fallback)
    result = adapter.collect(request)
    assert result.status == "ok" and len(result.items) == 1
    item = result.items[0]
    assert item.published_at == (request.now - timedelta(hours=1)).isoformat()
    assert item.outbound_urls == ["https://example.com/release"]
    assert not item.media_metadata["transcript_fetched"]
    assert result.metadata["usage_recorded"] and result.cost_usd == 2 * CALL_PRICE
    assert repo.current_month_cost() == 2 * CALL_PRICE
    reopened = YouTubeFallback(settings, Repository(settings.database_path), paid=fallback.paid)
    repeated = reopened.collect(request, CHANNEL)
    assert repeated.cost_usd == 0 and len(calls) == 2
    assert repo.current_month_cost() == 2 * CALL_PRICE


def test_orchestrator_does_not_double_charge_fallback(env, monkeypatch):
    settings, repo, fallback, calls, request = env
    registry = SourceRegistry(version=1, sources=[request.source])
    monkeypatch.setattr("app.collectors.orchestrator.runtime_source_registry", lambda _: registry)
    result = collect_due_sources(
        settings,
        repo,
        adapters={"youtube_rss": YouTubeRSSAdapter(unavailable, fallback=fallback)},
        now=request.now,
    )
    assert result["items_inserted"] == 1 and result["cost_usd"] == 2 * CALL_PRICE
    assert repo.current_month_cost() == 2 * CALL_PRICE
    with repo.connection() as db:
        assert db.execute("SELECT cost_usd FROM source_runs").fetchone()[0] == 2 * CALL_PRICE


def test_successful_rss_does_not_call_monid(env):
    _, repo, fallback, calls, request = env
    result = YouTubeRSSAdapter(
        lambda _: b'<feed xmlns="http://www.w3.org/2005/Atom"/>', fallback=fallback
    ).collect(request)
    assert result.status == "empty" and result.cost_usd == 0
    assert not calls and repo.current_month_cost() == 0


def test_rss_schema_mismatch_is_not_overridden(env):
    _, repo, fallback, calls, request = env
    result = YouTubeRSSAdapter(lambda _: b"<html/>", fallback=fallback).collect(request)
    assert result.status == "schema_changed" and not calls


def test_opt_out_never_calls_paid_provider(env):
    settings, repo, fallback, calls, request = env
    settings.youtube_rss_fallback_enabled = False
    result = fallback.collect(request, CHANNEL)
    assert result.status == "disabled" and not calls and repo.current_month_cost() == 0


@pytest.mark.parametrize(
    "budget", ["budget_hard_stop_usd", "sensory_daily_cap_usd", "sensory_monthly_cap_usd"]
)
def test_budget_blocks_before_paid_call(env, budget):
    settings, repo, fallback, calls, request = env
    setattr(settings, budget, 0)
    assert fallback.collect(request, CHANNEL).status == "disabled"
    assert not calls and repo.current_month_cost() == 0


def test_budget_exhausted_between_listing_and_details_is_disabled(env):
    settings, repo, fallback, calls, request = env
    settings.sensory_monthly_cap_usd = CALL_PRICE
    result = fallback.collect(request, CHANNEL)
    assert result.status == "disabled" and not result.items
    assert len(calls) == 1
    assert result.cost_usd == repo.current_month_cost() == CALL_PRICE


def test_channel_mismatch_still_accounts_for_billed_request(env):
    _, repo, fallback, calls, request = env

    def wrong(endpoint, query):
        calls.append(endpoint)
        return MonidResponse(listing("UC" + "b" * 22), CALL_PRICE, None)

    fallback.paid = wrong
    assert fallback.collect(request, CHANNEL).status == "schema_changed"
    assert repo.current_month_cost() == CALL_PRICE
    assert fallback.collect(request, CHANNEL).cost_usd == 0
    assert len(calls) == 1


def test_uncertain_request_is_not_replayed_after_restart(env):
    settings, repo, fallback, calls, request = env

    def unfinished(endpoint, query):
        calls.append(endpoint)
        raise PlatformError("Run is still active", status="partial", cost_usd=CALL_PRICE)

    fallback.paid = unfinished
    assert fallback.collect(request, CHANNEL).cost_usd == CALL_PRICE
    reopened = YouTubeFallback(settings, repo, paid=unfinished)
    assert reopened.collect(request, CHANNEL).cost_usd == 0
    assert len(calls) == 1 and repo.current_month_cost() == CALL_PRICE


@pytest.mark.parametrize("published", ["2026-10-05", "2026-10-05T12:00:00"])
def test_date_only_or_unzoned_values_are_not_guessed(env, published):
    _, repo, fallback, calls, request = env

    def dated(endpoint, query):
        calls.append(endpoint)
        return MonidResponse(
            listing() if endpoint == LIST_ENDPOINT else detail(request.now, published=published),
            CALL_PRICE,
            "run",
        )

    fallback.paid = dated
    result = fallback.collect(request, CHANNEL)
    assert result.status == "schema_changed" and not result.items
    assert result.cost_usd == 2 * CALL_PRICE and repo.current_month_cost() == 2 * CALL_PRICE


def test_exact_follow_cutoff_and_old_labels_prevent_archive(env):
    _, _, fallback, calls, request = env

    def dated(endpoint, query):
        calls.append(endpoint)
        return MonidResponse(
            listing(
                videos=[
                    {"video_id": VIDEO, "published_time": "1 hour ago"},
                    {"video_id": "bbbbbbbbbbb", "published_time": "2 days ago"},
                ]
            )
            if endpoint == LIST_ENDPOINT
            else detail(request.now, published=(request.now - timedelta(hours=3)).isoformat()),
            CALL_PRICE,
            "run",
        )

    fallback.paid = dated
    result = fallback.collect(request, CHANNEL)
    assert result.status == "empty" and not result.items and len(calls) == 2


def test_channel_addition_uses_fallback_and_cached_resolution(env, monkeypatch):
    settings, repo, fallback, calls, _ = env
    monkeypatch.setattr("app.collectors.youtube_fallback.YouTubeFallback", lambda *args: fallback)
    with repo.connection() as db:
        db.execute(
            "INSERT INTO youtube_channel_urls(url,channel_id,status,attempted_at) "
            "VALUES(?,?,'ready',?)",
            ("https://www.youtube.com/@testing", CHANNEL, datetime.now(UTC).isoformat()),
        )
    service = ChannelService(settings, repo, fetcher=unavailable)
    result = service.add("https://youtube.com/@Testing", 123)
    assert result["title"] == "Test Channel" and result["channel_id"] == CHANNEL
    assert service.add("https://youtube.com/@Testing", 123)["existing"]
    assert len(calls) == 1 and repo.current_month_cost() == CALL_PRICE


def test_channel_opt_out_reports_real_rss_error(env):
    settings, repo, fallback, calls, _ = env
    settings.youtube_rss_fallback_enabled = False
    with pytest.raises(ChannelError, match="HTTP 404"):
        ChannelService(settings, repo, fetcher=unavailable).add(
            "https://www.youtube.com/channel/" + CHANNEL, 123
        )
    assert not calls
