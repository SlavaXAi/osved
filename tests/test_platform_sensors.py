import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from app.collectors.base import CollectionRequest
from app.collectors.platform_transport import (
    MonidResponse,
    PlatformError,
    fetch_public,
    unwrap_data,
)
from app.collectors.platforms import (
    RedditAdapter,
    V2EXAdapter,
    WeiboSearchAdapter,
    XSearchAdapter,
    YouTubeRSSAdapter,
    ZhihuSearchAdapter,
)
from app.config import SourceConfig

NOW = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)


def request(kind="x", locator="AI agents", strategy="search", options=None, limit=10):
    source = SourceConfig(
        id="fixture",
        name="Fixture",
        kind=kind,
        locator=locator,
        role="radar",
        region="global",
        language="en",
        strategy=strategy,
        options=options or {},
        production_max_age_days=5,
    )
    return CollectionRequest(source=source, limit=limit, now=NOW)


def paid(payload):
    return lambda endpoint, query: MonidResponse(payload, 0.0015)


def tweet(identifier="123", author="person", created="Mon Sep 07 10:00:00 +0000 2026"):
    return {
        "id_str": identifier,
        "full_text": "A useful research release with a link https://example.com/paper",
        "created_at": created,
        "user": {"screen_name": author},
    }


def test_x_search_constrains_recency_and_uses_real_author(tmp_path):
    calls = []

    def fetch(endpoint, query):
        calls.append(query)
        return MonidResponse({"timeline": [tweet(), tweet("124", "second")]}, 0.0015)

    batch = XSearchAdapter(tmp_path, fetch).collect(request())
    assert batch.status == "ok"
    assert calls[0]["keyword"] == "AI agents within_time:5d"
    assert calls[0]["search_type"] == "Top"
    assert {item.author for item in batch.items} == {"person", "second"}
    assert batch.items[0].canonical_url == "https://x.com/person/status/123"
    assert batch.metadata["provider_order"] == "engagement_ranked"
    assert batch.cost_usd == 0.0015


def test_x_search_does_not_require_literal_query_match(tmp_path):
    batch = XSearchAdapter(tmp_path, paid({"timeline": [tweet()]})).collect(
        request(locator="robot factory")
    )
    assert len(batch.items) == 1


def test_x_search_stale_future_and_duplicates_are_removed(tmp_path):
    raw = [
        tweet(),
        tweet(),
        tweet("125", created="Mon Aug 03 10:00:00 +0000 2026"),
        tweet("126", created="Tue Sep 08 10:00:00 +0000 2026"),
    ]
    batch = XSearchAdapter(tmp_path, paid({"timeline": raw})).collect(request())
    assert [item.external_id for item in batch.items] == ["123"]


@pytest.mark.parametrize(
    "payload,status",
    [
        ({"timeline": []}, "empty"),
        ({"unexpected": []}, "schema_changed"),
        ({"timeline": [tweet(created="not-a-date")]}, "schema_changed"),
        ({"code": 429, "data": {}}, "rate_limited"),
        ({"code": 403, "data": {}}, "login_required"),
    ],
)
def test_x_distinguishes_empty_schema_drift_and_access_failures(tmp_path, payload, status):
    result = XSearchAdapter(tmp_path, paid(payload)).collect(request())
    assert result.status == status
    assert result.cost_usd == 0.0015


def test_x_partial_page_keeps_good_data_and_cost(tmp_path):
    result = XSearchAdapter(
        tmp_path, paid({"timeline": [tweet(), tweet("124", created="bad")]})
    ).collect(request())
    assert result.status == "partial"
    assert len(result.items) == 1
    assert result.metadata["items_invalid"] == 1


def test_upstream_failure_does_not_retry_paid_call(tmp_path):
    calls = []

    def fail(endpoint, query):
        calls.append(query)
        raise PlatformError("rate limited", status="rate_limited", cost_usd=0.0015)

    batch = XSearchAdapter(tmp_path, fail).collect(request())
    assert len(calls) == 1
    assert batch.status == "rate_limited"
    assert batch.cost_usd == 0.0015


def reddit_feed():
    return b"""<feed xmlns="http://www.w3.org/2005/Atom"><entry>
      <id>t3_abc123</id><title>New open source tool</title>
      <link href="https://www.reddit.com/r/LocalLLaMA/comments/abc123/new/"/>
      <published>2026-09-07T10:00:00+00:00</published>
      <content type="html">&lt;p&gt;Tool details &lt;a href="https://example.com/tool"&gt;link&lt;/a&gt;&lt;/p&gt;</content>
    </entry></feed>"""


def test_reddit_rss_metadata_keeps_original_link():
    calls = []
    adapter = RedditAdapter(lambda url: calls.append(url) or reddit_feed())
    batch = adapter.collect(request("reddit", "LocalLLaMA", "new"))
    assert batch.status == "ok"
    assert batch.items[0].external_id == "abc123"
    assert batch.items[0].content_type == "forum_post"
    assert "https://example.com/tool" in batch.items[0].outbound_urls
    assert "/r/LocalLLaMA/new.rss" in calls[0]


def test_reddit_search_query_is_encoded():
    calls = []
    adapter = RedditAdapter(lambda url: calls.append(url) or reddit_feed())
    result = adapter.collect(
        request("reddit", "singularity", "search", {"query": "AI welfare & agency"})
    )
    assert result.status == "ok"
    query = parse_qs(urlsplit(calls[0]).query)
    assert query["q"] == ["AI welfare & agency"]
    assert query["restrict_sr"] == ["on"]


def test_reddit_json_filters_adult_ads_and_pins():
    def post(identifier, **extras):
        return {
            "kind": "t3",
            "data": {
                "id": identifier,
                "title": "A real event",
                "permalink": f"/r/test/comments/{identifier}/",
                "created_utc": NOW.timestamp() - 10,
                "author": "reader",
                "score": 100,
                **extras,
            },
        }

    body = {
        "data": {
            "after": "t3_after",
            "children": [
                post("good"),
                post("ad", promoted=True),
                post("nsfw", over_18=True),
                post("pin", stickied=True),
            ],
        }
    }
    batch = RedditAdapter(lambda url: json.dumps(body).encode()).collect(
        request("reddit", "test", "json")
    )
    assert [item.external_id for item in batch.items] == ["good"]
    assert batch.next_cursor == "t3_after"
    assert batch.items[0].engagement_metadata["score"] == 100


def test_reddit_does_not_fetch_arbitrary_url():
    calls = []
    result = RedditAdapter(lambda url: calls.append(url) or b"").collect(
        request("reddit", "https://evil.example/r/test", "new")
    )
    assert result.status == "failed"
    assert not calls


def youtube_feed():
    return b"""<feed xmlns="http://www.w3.org/2005/Atom"
        xmlns:yt="http://www.youtube.com/xml/schemas/2015" xmlns:media="http://search.yahoo.com/mrss/">
      <entry><yt:videoId>abcdefghijk</yt:videoId><title>A conversation about agents</title>
       <published>2026-09-07T10:00:00+00:00</published><author><name>YC</name></author>
       <media:group><media:description>Founders discuss building useful products.
       </media:description></media:group>
      </entry></feed>"""


def test_youtube_only_claims_fetched_metadata():
    batch = YouTubeRSSAdapter(lambda url: youtube_feed()).collect(
        request("youtube", "UCcefcZRL2oaA_uBNeo5UOWg", "channel_uploads")
    )
    assert batch.status == "ok"
    item = batch.items[0]
    assert item.canonical_url == "https://www.youtube.com/watch?v=abcdefghijk"
    assert item.author == "YC"
    assert "Founders discuss" in item.text
    assert item.media_metadata["transcript_fetched"] is False
    assert item.raw_payload["content_basis"] == "metadata_only"
    assert "chapters" not in item.media_metadata
    assert "duration" not in item.media_metadata


def test_youtube_does_not_silently_turn_handles_into_ids():
    calls = []
    batch = YouTubeRSSAdapter(lambda url: calls.append(url) or b"").collect(
        request("youtube", "@ycombinator", "channel_uploads")
    )
    assert batch.status == "failed"
    assert not calls


@pytest.mark.parametrize(
    "content", [b"<html>sign in</html>", b'<!DOCTYPE x [<!ENTITY y "hi">]><feed/>', b"bad xml"]
)
def test_youtube_drift_is_not_empty_success(content):
    batch = YouTubeRSSAdapter(lambda url: content).collect(
        request("youtube", "UCcefcZRL2oaA_uBNeo5UOWg", "channel_uploads")
    )
    assert batch.status == "schema_changed"


def test_v2ex_filters_nodes_and_preserves_original_chinese():
    posts = [
        {
            "id": 10,
            "title": "人工智能新工具",
            "content": "工具已开源",
            "created": NOW.timestamp() - 1,
            "member": {"username": "maker"},
            "node": {"name": "ai"},
            "replies": 5,
        },
        {"id": 11, "title": "offtopic", "created": NOW.timestamp() - 1, "node": {"name": "jobs"}},
    ]
    result = V2EXAdapter(lambda url: json.dumps(posts).encode()).collect(
        request("v2ex", "latest", "latest", {"nodes": ["ai"]})
    )
    assert len(result.items) == 1
    assert result.items[0].text.startswith("人工智能")
    assert result.items[0].canonical_url == "https://www.v2ex.com/t/10"


def test_v2ex_missing_creation_date_is_not_assigned_now():
    result = V2EXAdapter(lambda url: b'[{"id":1,"title":"Missing date"}]').collect(
        request("v2ex", "latest", "latest")
    )
    assert result.status == "schema_changed"
    assert not result.items


def test_weibo_nested_search_cards(tmp_path):
    post = {
        "idstr": "5678",
        "text": "<b>新产品</b>发布",
        "created_at": "Mon Sep 07 10:00:00 +0000 2026",
        "user": {"screen_name": "研究者", "id": 123},
        "reposts_count": 10,
    }
    body = {"code": 200, "data": {"cards": [{"card_group": [{"mblog": post}]}]}}
    result = WeiboSearchAdapter(tmp_path, paid(body)).collect(
        request("weibo", "人工智能", "search")
    )
    assert result.status == "ok"
    assert result.items[0].text.replace(" ", "") == "新产品发布"
    assert result.items[0].author == "研究者"
    assert result.items[0].canonical_url == "https://m.weibo.cn/detail/5678"


def test_weibo_real_app_items_envelope_excludes_ads_and_marks_truncated_posts(tmp_path):
    post = {
        "idstr": "5678",
        "text": "新研究发布，但这里只是开头",
        "created_at": "Mon Sep 07 14:15:34 +0800 2026",
        "user": {"screen_name": "研究者", "id": 123},
        "isLongText": True,
    }
    payload = {
        "items": [
            {"category": "feed", "data": post},
            {"category": "feed", "data": {**post, "idstr": "999", "ad_marked": True}},
            {
                "category": "feed",
                "data": {**post, "idstr": "888", "ad_marked": False, "ad_object": {"ad": True}},
            },
            {"category": "user", "data": {"name": "Profile, not a post"}},
        ]
    }
    result = WeiboSearchAdapter(tmp_path, paid(payload)).collect(request("weibo", "人工智能"))
    assert result.status == "ok"
    assert len(result.items) == 1
    assert result.items[0].raw_payload["content_basis"] == "excerpt"
    assert result.items[0].published_at.startswith("2026-09-07T06:15:34")


def test_weibo_malformed_app_feed_item_is_schema_changed(tmp_path):
    result = WeiboSearchAdapter(tmp_path, paid({"items": [{"category": "feed", "data": None}]}))
    batch = result.collect(request("weibo", "人工智能"))
    assert batch.status == "schema_changed"


def test_zhihu_excerpt_is_not_labelled_full_article(tmp_path):
    post = {
        "object": {
            "id": "123",
            "type": "article",
            "title": "新研究",
            "excerpt": "研究摘要",
            "created_time": NOW.timestamp() - 1,
            "author": {"name": "学者"},
        }
    }
    result = ZhihuSearchAdapter(tmp_path, paid({"data": [post], "paging": {}})).collect(
        request("zhihu", "人工智能", "search")
    )
    assert result.status == "ok"
    assert result.items[0].raw_payload["content_basis"] == "excerpt"
    assert result.items[0].canonical_url == "https://zhuanlan.zhihu.com/p/123"


@pytest.mark.parametrize(
    "adapter,kind", [(WeiboSearchAdapter, "weibo"), (ZhihuSearchAdapter, "zhihu")]
)
def test_chinese_unrecognized_envelope_is_schema_changed(tmp_path, adapter, kind):
    result = adapter(tmp_path, paid({"unexpected": True})).collect(
        request(kind, "人工智能", "search")
    )
    assert result.status == "schema_changed"
    assert result.cost_usd == 0.0015


def test_no_transport_can_fetch_localhost():
    with pytest.raises(PlatformError):
        fetch_public("http://127.0.0.1:8000/admin")


def test_envelope_error_is_not_swallowed():
    with pytest.raises(PlatformError) as caught:
        unwrap_data({"code": 429, "data": {"timeline": []}})
    assert caught.value.status == "rate_limited"


def test_reddit_html_login_page_is_not_empty_success():
    result = RedditAdapter(lambda url: b"<html>Login</html>").collect(
        request("reddit", "LocalLLaMA", "new")
    )
    assert result.status == "schema_changed"


def test_monid_transport_reports_actual_receipt_cost(tmp_path, monkeypatch):
    from subprocess import CompletedProcess

    from app.collectors import platform_transport

    monkeypatch.setattr(platform_transport, "monid_cli_path", lambda: "monid")

    def run(command, **kwargs):
        output = Path(command[command.index("-o") + 1])
        output.write_text('{"timeline":[]}', encoding="utf-8")
        receipt = {"status": "COMPLETED", "id": "run1", "cost": {"value": 0.0009}}
        return CompletedProcess(command, 0, json.dumps(receipt), "")

    monkeypatch.setattr(platform_transport.subprocess, "run", run)
    response = platform_transport.monid_fetch("/test", {}, tmp_path)
    assert response.cost_usd == 0.0009
    assert response.run_id == "run1"
    assert response.payload == {"timeline": []}


@pytest.mark.parametrize(
    "status,expected_cost,expected_status",
    [
        ("BLOCKED", 0, "disabled"),
        ("RUNNING", 0.0015, "partial"),
        ("FAILED", 0.0015, "failed"),
    ],
)
def test_monid_transport_handles_terminal_or_pending_receipt(
    tmp_path, monkeypatch, status, expected_cost, expected_status
):
    from subprocess import CompletedProcess

    from app.collectors import platform_transport

    calls = []
    monkeypatch.setattr(platform_transport, "monid_cli_path", lambda: "monid")

    def run(command, **kwargs):
        calls.append(command)
        return CompletedProcess(command, 0, json.dumps({"status": status, "id": "run1"}), "")

    monkeypatch.setattr(platform_transport.subprocess, "run", run)
    with pytest.raises(PlatformError) as caught:
        platform_transport.monid_fetch("/test", {}, tmp_path)
    assert caught.value.status == expected_status
    assert caught.value.cost_usd == expected_cost
    assert len(calls) == 1


def test_monid_timeout_charges_estimate_without_restarting(tmp_path, monkeypatch):
    from subprocess import TimeoutExpired

    from app.collectors import platform_transport

    monkeypatch.setattr(platform_transport, "monid_cli_path", lambda: "monid")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        raise TimeoutExpired(command, 85)

    monkeypatch.setattr(platform_transport.subprocess, "run", run)
    with pytest.raises(PlatformError) as caught:
        platform_transport.monid_fetch("/test", {}, tmp_path)
    assert caught.value.status == "partial"
    assert caught.value.cost_usd == 0.0015
    assert len(calls) == 1


def reddit_monid_request(*, cap=0.0015, strategy="new", options=None):
    combined = {"cache_ttl_minutes": 60, **(options or {})}
    if cap is not None:
        combined["per_call_price_cap_usd"] = cap
    return request("reddit", "LocalLLaMA", strategy, combined)


def reddit_monid_post(identifier="abc123"):
    return {
        "id": "t3_" + identifier,
        "title": "An open source agent release",
        "text": "Developers release a new agent framework.",
        "created_at": NOW.isoformat(),
        "author": {"name": "builder"},
        "score": 20,
        "comment_count": 5,
    }


@pytest.mark.parametrize("cap", [None, 0.001, True, float("nan")])
def test_paid_reddit_requires_explicit_sufficient_numeric_cap(tmp_path, cap):
    from app.collectors.platform_reddit_monid import RedditMonidAdapter

    calls = []
    adapter = RedditMonidAdapter(tmp_path, fetcher=lambda e, q: calls.append(q))
    result = adapter.collect(reddit_monid_request(cap=cap))
    assert result.status == "disabled"
    assert not calls
    assert adapter.capabilities().estimated_cost_usd == 0.0015


def test_paid_reddit_formatted_feed_normalizes_and_caches(tmp_path):
    from app.collectors.platform_reddit_monid import REDDIT_FEED_ENDPOINT, RedditMonidAdapter

    calls = []

    def fetch(endpoint, query):
        calls.append((endpoint, query))
        return MonidResponse({"posts": [reddit_monid_post()]}, 0.0015)

    adapter = RedditMonidAdapter(tmp_path, fetcher=fetch)
    first = adapter.collect(reddit_monid_request())
    again = RedditMonidAdapter(tmp_path, fetcher=fetch).collect(reddit_monid_request())
    assert first.status == again.status == "ok"
    assert first.items[0].external_id == "abc123"
    assert first.items[0].author == "builder"
    assert first.items[0].discovery_mode == "new"
    assert first.cost_usd == 0.0015
    assert again.cost_usd == 0
    assert again.metadata["cache_hit"] is True
    assert len(calls) == 1
    assert calls[0][0] == REDDIT_FEED_ENDPOINT
    assert calls[0][1] == {
        "subreddit_name": "LocalLLaMA",
        "sort": "NEW",
        "need_format": True,
        "language": "en-US",
    }


def test_paid_reddit_real_graphql_cell_envelope_excludes_ads_and_labels_excerpts(tmp_path):
    from app.collectors.platform_reddit_monid import RedditMonidAdapter

    payload = {
        "subredditfeed": {
            "subredditV3": {
                "elements": {
                    "edges": [
                        {"node": {"groupId": "sortcell", "cells": []}},
                        {"node": {"groupId": "t3_ad123", "adPayload": {}, "cells": []}},
                        {
                            "node": {
                                "groupId": "t3_abc123",
                                "cells": [
                                    {"createdAt": NOW.isoformat(), "authorName": "u/builder"},
                                    {"titleCell": {"title": "A practical open source agent"}},
                                    {"text": "Preview excerpt, not the whole post."},
                                    {"score": 12, "commentCount": 3},
                                ],
                            }
                        },
                        {
                            "node": {
                                "groupId": "t3_def456",
                                "cells": [
                                    {"createdAt": NOW.isoformat(), "authorName": "u/researcher"},
                                    {"title": "How can small local models be evaluated?"},
                                ],
                            }
                        },
                    ]
                }
            }
        }
    }
    result = RedditMonidAdapter(tmp_path, fetcher=paid(payload)).collect(reddit_monid_request())
    assert result.status == "ok"
    assert len(result.items) == 2
    assert result.items[0].external_id == "abc123"
    assert result.items[0].author == "builder"
    assert result.items[0].raw_payload["content_basis"] == "title_and_excerpt"
    assert result.items[1].raw_payload["content_basis"] == "title_only"
    assert result.items[0].canonical_url == "https://www.reddit.com/r/LocalLLaMA/comments/abc123/"


def test_paid_reddit_graphql_known_empty_is_empty_but_malformed_post_is_drift(tmp_path):
    from app.collectors.platform_reddit_monid import RedditMonidAdapter

    def envelope(edges):
        return {"subredditfeed": {"subredditV3": {"elements": {"edges": edges}}}}

    result = RedditMonidAdapter(tmp_path / "empty", fetcher=paid(envelope([]))).collect(
        reddit_monid_request()
    )
    broken = RedditMonidAdapter(
        tmp_path / "broken", fetcher=paid(envelope([{"node": {"groupId": "t3_abc123"}}]))
    ).collect(reddit_monid_request())
    assert result.status == "empty"
    assert broken.status == "schema_changed"


def test_paid_reddit_search_uses_inspected_fields_and_separate_cache(tmp_path):
    from app.collectors.platform_reddit_monid import REDDIT_SEARCH_ENDPOINT, RedditMonidAdapter

    calls = []

    def fetch(endpoint, query):
        calls.append((endpoint, query))
        return MonidResponse({"posts": []}, 0.0015)

    adapter = RedditMonidAdapter(tmp_path, fetcher=fetch)
    first = adapter.collect(reddit_monid_request(strategy="search", options={"query": "agents"}))
    same = adapter.collect(reddit_monid_request(strategy="search", options={"query": "agents"}))
    different = adapter.collect(
        reddit_monid_request(strategy="search", options={"query": "robotics"})
    )
    assert first.status == same.status == different.status == "empty"
    assert same.metadata["cache_hit"] is True
    assert len(calls) == 2
    assert calls[0][0] == REDDIT_SEARCH_ENDPOINT
    assert calls[0][1]["query"] == "subreddit:LocalLLaMA agents"
    assert calls[0][1]["allow_nsfw"] == "0"
    assert calls[0][1]["safe_search"] == "strict"
    assert calls[0][1]["time_range"] == "week"


def test_paid_reddit_cache_expiry_makes_one_new_call(tmp_path):
    from dataclasses import replace
    from datetime import timedelta

    from app.collectors.platform_reddit_monid import RedditMonidAdapter

    calls = []

    def fetch(endpoint, query):
        calls.append(query)
        return MonidResponse({"posts": []}, 0.0015)

    adapter = RedditMonidAdapter(tmp_path, fetcher=fetch)
    original = reddit_monid_request()
    adapter.collect(original)
    result = adapter.collect(replace(original, now=NOW + timedelta(hours=2)))
    assert len(calls) == 2
    assert result.cost_usd == 0.0015


def test_paid_reddit_changed_schema_is_not_empty_success(tmp_path):
    from app.collectors.platform_reddit_monid import RedditMonidAdapter

    result = RedditMonidAdapter(tmp_path, fetcher=paid({"unexpected": []})).collect(
        reddit_monid_request()
    )
    assert result.status == "schema_changed"
    assert result.cost_usd == 0.0015


def test_paid_reddit_invalid_rows_are_not_silently_empty(tmp_path):
    from app.collectors.platform_reddit_monid import RedditMonidAdapter

    result = RedditMonidAdapter(tmp_path, fetcher=paid({"posts": ["bad", {}]})).collect(
        reddit_monid_request()
    )
    assert result.status == "schema_changed"
    assert result.cost_usd == 0.0015


def test_paid_reddit_unexpected_billing_is_accounted_and_stops(tmp_path):
    from app.collectors.platform_reddit_monid import RedditMonidAdapter

    adapter = RedditMonidAdapter(tmp_path, fetcher=lambda e, q: MonidResponse({"posts": []}, 0.01))
    result = adapter.collect(reddit_monid_request())
    assert result.status == "disabled"
    assert result.cost_usd == 0.01
    assert not list(tmp_path.glob("reddit-*.json"))


def test_sensory_configuration_has_valid_contracts():
    from app.collectors.platforms import build_platform_adapters
    from app.config import Settings, load_source_registry

    registry = load_source_registry(Path("tests/fixtures/config/sensory-sources.yaml"))
    adapters = build_platform_adapters(Settings(app_env="test"))
    assert len(registry.sources) >= 10
    for source in registry.sources:
        if source.adapter_id == "browser_x":
            assert not source.enabled
            continue
        assert source.adapter_id in adapters
        assert source.strategy in adapters[source.adapter_id].capabilities().strategies
