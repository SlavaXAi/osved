"""Platform sensors. These return observations; the shared ingest owns events."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlencode, urlsplit
from xml.etree import ElementTree

from app.collectors.base import (
    CollectedItem,
    CollectionBatch,
    CollectionRequest,
    SourceCapabilities,
    SourceHealth,
)
from app.collectors.platform_transport import (
    MonidResponse,
    PlatformError,
    fetch_public,
    monid_fetch,
    unwrap_data,
)
from app.collectors.rss import (
    UNSAFE_XML_RE,
    RSSCollectorError,
    extract_urls,
    normalize_published,
    parse_feed,
    strip_markup,
)
from app.collectors.x_monid import normalize_tweet, parse_twitter_datetime
from app.config import Settings

X_SEARCH_ENDPOINT = "/api/v1/twitter/web/fetch_search_timeline"
WEIBO_SEARCH_ENDPOINT = "/api/v1/weibo/app/fetch_search_all"
ZHIHU_SEARCH_ENDPOINT = "/api/v1/zhihu/web/fetch_article_search_v3"
PublicFetcher = Callable[[str], bytes]
PaidFetcher = Callable[[str, dict[str, Any]], MonidResponse]
MAX_TEXT = 12_000


def _iso(value: Any) -> str:
    if isinstance(value, (int, float)):
        return parse_twitter_datetime(value).isoformat()
    if not value:
        raise ValueError("Missing publication time")
    try:
        return normalize_published(str(value))
    except RSSCollectorError:
        return parse_twitter_datetime(value).isoformat()


def _cutoff(request: CollectionRequest) -> datetime:
    return request.now - timedelta(days=request.source.production_max_age_days)


def _finish(
    request: CollectionRequest,
    items: list[CollectedItem],
    *,
    cost: float = 0,
    seen: int = 0,
    skipped: int = 0,
    cursor: str | None = None,
) -> CollectionBatch:
    fresh = []
    identities: set[str] = set()
    for item in items:
        published = datetime.fromisoformat(item.published_at.replace("Z", "+00:00"))
        if published < _cutoff(request) or published > request.now + timedelta(minutes=10):
            continue
        if item.external_id not in identities:
            identities.add(item.external_id)
            fresh.append(item)
    fresh.sort(key=lambda item: item.published_at, reverse=True)
    status = "partial" if skipped else ("ok" if fresh else "empty")
    if seen and skipped == seen:
        status = "schema_changed"
    return CollectionBatch(
        items=fresh[: request.limit],
        status=status,
        cost_usd=cost,
        next_cursor=cursor,
        error="Some items lacked usable text, identity or dates" if skipped else None,
        metadata={"items_seen": seen, "items_invalid": skipped, "freshness_checked": True},
    )


def _failure(exc: Exception, *, cost: float = 0) -> CollectionBatch:
    if isinstance(exc, PlatformError):
        return CollectionBatch(status=exc.status, error=str(exc), cost_usd=max(cost, exc.cost_usd))
    return CollectionBatch(
        status="schema_changed",
        error=f"Unrecognized platform response: {type(exc).__name__}",
        cost_usd=cost,
    )


class _PublicAdapter:
    strategies: tuple[str, ...] = ("feed",)

    def __init__(self, fetcher: PublicFetcher = fetch_public) -> None:
        self.fetcher = fetcher

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(strategies=self.strategies, estimated_cost_usd=0)

    def healthcheck(self, source: Any) -> SourceHealth:
        return SourceHealth(status="ok")


class RedditAdapter(_PublicAdapter):
    """RSS is the default; JSON is explicit and never a stealth auth workaround."""

    strategies = ("feed", "new", "hot", "search", "json")

    def collect(self, request: CollectionRequest) -> CollectionBatch:
        try:
            source = request.source
            community = source.locator.strip().removeprefix("r/")
            if "://" in community:
                parsed = urlsplit(community)
                if parsed.hostname not in {"reddit.com", "www.reddit.com"}:
                    raise PlatformError("Invalid Reddit source URL")
                match = re.search(r"/r/([A-Za-z0-9_+]+)", parsed.path)
                community = match.group(1) if match else ""
            if not re.fullmatch(r"[A-Za-z0-9_+]{2,120}", community):
                raise PlatformError("Reddit locator must identify a subreddit")
            strategy = source.strategy
            sort = str(source.options.get("sort", "new")) if strategy == "json" else strategy
            if sort not in {"new", "hot", "search"}:
                sort = "new"
            extension = ".json" if strategy == "json" else ".rss"
            query: dict[str, Any] = {"limit": min(request.limit, 100)}
            if strategy == "search":
                query.update(
                    q=str(source.options.get("query", "")), restrict_sr="on", sort="new", t="week"
                )
                if not query["q"]:
                    raise PlatformError("Reddit search requires options.query")
            url = f"https://www.reddit.com/r/{community}/{sort}{extension}?{urlencode(query)}"
            payload = self.fetcher(url)
            if strategy == "json":
                return self._json(request, payload)
            if UNSAFE_XML_RE.search(payload):
                raise PlatformError("Unsafe XML declaration", status="schema_changed")
            root = ElementTree.fromstring(payload)
            if root.tag.rsplit("}", 1)[-1] not in {"feed", "rss"}:
                raise PlatformError("Reddit returned no RSS/Atom feed", status="schema_changed")
            items = parse_feed(payload, limit=min(100, request.limit * 3))
            normalized = [
                CollectedItem(
                    external_id=item.external_id.removeprefix("t3_"),
                    canonical_url=item.source_url,
                    text=item.original_text,
                    published_at=item.published_at,
                    language=source.language,
                    title=item.original_text.split("\n", 1)[0],
                    content_type="forum_post",
                    discovery_mode=strategy,
                    outbound_urls=item.outbound_urls,
                    raw_payload={"subreddit": community, "content_basis": "rss_metadata"},
                )
                for item in items
            ]
            return _finish(request, normalized, seen=len(items))
        except (
            PlatformError,
            ValueError,
            TypeError,
            RSSCollectorError,
            ElementTree.ParseError,
        ) as exc:
            return _failure(exc)

    def _json(self, request: CollectionRequest, payload: bytes) -> CollectionBatch:
        body = json.loads(payload)
        children = body.get("data", {}).get("children") if isinstance(body, dict) else None
        if not isinstance(children, list):
            raise PlatformError("Reddit listing has no children", status="schema_changed")
        items, skipped = [], 0
        for child in children[:100]:
            post = child.get("data", {}) if isinstance(child, dict) else {}
            if post.get("promoted") or post.get("stickied") or post.get("over_18"):
                continue
            try:
                identifier, title = str(post["id"]), str(post["title"]).strip()
                if not identifier or not title or not post.get("permalink"):
                    raise ValueError("Incomplete Reddit post")
                url = "https://www.reddit.com" + str(post["permalink"])
                text = title + "\n\n" + str(post.get("selftext") or "")
                outbound = extract_urls(str(post.get("url") or ""), text, source_url=url)
                items.append(
                    CollectedItem(
                        external_id=identifier,
                        canonical_url=url,
                        title=title,
                        text=text[:MAX_TEXT].strip(),
                        published_at=_iso(post.get("created_utc")),
                        author=str(post.get("author") or ""),
                        language=request.source.language,
                        content_type="forum_post",
                        discovery_mode="json",
                        outbound_urls=outbound,
                        engagement_metadata={
                            "score": post.get("score"),
                            "comments": post.get("num_comments"),
                        },
                        raw_payload={
                            "subreddit": post.get("subreddit"),
                            "crosspost_parent": post.get("crosspost_parent"),
                        },
                    )
                )
            except (ValueError, KeyError, TypeError, RSSCollectorError):
                skipped += 1
        return _finish(
            request, items, seen=len(children), skipped=skipped, cursor=body["data"].get("after")
        )


class YouTubeRSSAdapter(_PublicAdapter):
    strategies = ("feed", "channel_uploads")

    def collect(self, request: CollectionRequest) -> CollectionBatch:
        try:
            locator = request.source.locator
            if "://" in locator:
                parsed = urlsplit(locator)
                if parsed.hostname not in {"www.youtube.com", "youtube.com"}:
                    raise PlatformError("Invalid YouTube source host")
                channel = (parse_qs(parsed.query).get("channel_id") or [""])[0]
                if not channel and parsed.path.startswith("/channel/"):
                    channel = parsed.path.split("/")[2]
            else:
                channel = locator
            if not re.fullmatch(r"UC[A-Za-z0-9_-]{22}", channel):
                raise PlatformError("YouTube source requires a UC channel ID (handles are not IDs)")
            payload = self.fetcher(
                f"https://www.youtube.com/feeds/videos.xml?channel_id={quote(channel)}"
            )
            if UNSAFE_XML_RE.search(payload):
                raise PlatformError("Unsafe XML declaration", status="schema_changed")
            root = ElementTree.fromstring(payload)
            ns = {
                "a": "http://www.w3.org/2005/Atom",
                "yt": "http://www.youtube.com/xml/schemas/2015",
                "m": "http://search.yahoo.com/mrss/",
            }
            if root.tag != "{http://www.w3.org/2005/Atom}feed":
                raise PlatformError("YouTube returned no Atom feed", status="schema_changed")
            entries = root.findall("a:entry", ns)
            items, skipped = [], 0
            for entry in entries[:100]:
                video_id = entry.findtext("yt:videoId", "", ns)
                title = entry.findtext("a:title", "", ns)
                description = entry.findtext("m:group/m:description", "", ns)
                try:
                    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id) or not title:
                        raise ValueError("Incomplete video")
                    url = f"https://www.youtube.com/watch?v={video_id}"
                    items.append(
                        CollectedItem(
                            external_id=video_id,
                            canonical_url=url,
                            title=title,
                            text=(title + "\n\n" + description).strip()[:MAX_TEXT],
                            published_at=_iso(entry.findtext("a:published", "", ns)),
                            author=entry.findtext("a:author/a:name", "", ns),
                            language=request.source.language,
                            content_type="video",
                            discovery_mode="channel_uploads",
                            outbound_urls=extract_urls(description, source_url=url),
                            media_metadata={
                                "video_id": video_id,
                                "channel_id": channel,
                                "content_basis": "title_and_description",
                                "transcript_fetched": False,
                            },
                            raw_payload={
                                "video_id": video_id,
                                "channel_id": channel,
                                "description": description[:MAX_TEXT],
                                "content_basis": "metadata_only",
                            },
                        )
                    )
                except (ValueError, RSSCollectorError):
                    skipped += 1
            return _finish(request, items, seen=len(entries), skipped=skipped)
        except (PlatformError, ValueError, TypeError, ElementTree.ParseError) as exc:
            return _failure(exc)


class V2EXAdapter(_PublicAdapter):
    strategies = ("feed", "latest", "hot")

    def collect(self, request: CollectionRequest) -> CollectionBatch:
        try:
            strategy = "hot" if request.source.strategy == "hot" else "latest"
            payload = json.loads(self.fetcher(f"https://www.v2ex.com/api/topics/{strategy}.json"))
            if not isinstance(payload, list):
                raise PlatformError("V2EX response is not a topics list", status="schema_changed")
            nodes = set(request.source.options.get("nodes", []))
            items, skipped = [], 0
            for post in payload[:100]:
                if not isinstance(post, dict):
                    skipped += 1
                    continue
                node = post.get("node") or {}
                if nodes and node.get("name") not in nodes:
                    continue
                try:
                    identifier, title = str(post["id"]), str(post["title"]).strip()
                    if not identifier.isdigit() or not title:
                        raise ValueError("Invalid V2EX item")
                    url = f"https://www.v2ex.com/t/{identifier}"
                    content = str(
                        post.get("content") or strip_markup(post.get("content_rendered") or "")
                    )
                    items.append(
                        CollectedItem(
                            external_id=identifier,
                            canonical_url=url,
                            title=title,
                            text=(title + "\n\n" + content).strip()[:MAX_TEXT],
                            published_at=_iso(post.get("created")),
                            author=str((post.get("member") or {}).get("username") or ""),
                            language=request.source.language,
                            content_type="forum_post",
                            discovery_mode=strategy,
                            outbound_urls=extract_urls(content, source_url=url),
                            engagement_metadata={"replies": post.get("replies")},
                            raw_payload={
                                "node": node.get("name"),
                                "last_modified": post.get("last_modified"),
                            },
                        )
                    )
                except (ValueError, KeyError, TypeError):
                    skipped += 1
            return _finish(request, items, seen=len(payload), skipped=skipped)
        except (PlatformError, ValueError, TypeError) as exc:
            return _failure(exc)


class _MonidAdapter:
    strategies = ("search",)

    def __init__(self, output_dir: Path, fetcher: PaidFetcher | None = None) -> None:
        self.fetcher = fetcher or (lambda endpoint, query: monid_fetch(endpoint, query, output_dir))

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            strategies=self.strategies, requires_auth=True, estimated_cost_usd=0.0015
        )

    def healthcheck(self, source: Any) -> SourceHealth:
        return SourceHealth(status="ok")


def _timeline(payload: Any) -> list[Any]:
    payload = unwrap_data(payload)
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise PlatformError("Search response has no timeline", status="schema_changed")
    for key in ("timeline", "tweets", "items", "entries"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            return _timeline(value)
    raise PlatformError("Search response has no timeline", status="schema_changed")


class XSearchAdapter(_MonidAdapter):
    def collect(self, request: CollectionRequest) -> CollectionBatch:
        cost = 0.0
        try:
            keyword = request.source.locator.strip()
            if not keyword:
                raise PlatformError("X search query is empty")
            if not re.search(r"(?:within_time:|since:|until:)", keyword):
                keyword += f" within_time:{request.source.production_max_age_days}d"
            response = self.fetcher(X_SEARCH_ENDPOINT, {"keyword": keyword, "search_type": "Top"})
            cost = response.cost_usd
            raw = _timeline(response.payload)
            items, skipped = [], 0
            for value in raw[:200]:
                if not isinstance(value, Mapping):
                    skipped += 1
                    continue
                # Cursor markers are not malformed tweets.
                if "cursor" in str(value.get("entryId", "")).lower():
                    continue
                try:
                    tweet = normalize_tweet(request.source, value, fallback_user={})
                    if tweet is None:
                        continue
                    url = (
                        f"https://x.com/{tweet.author_handle}/status/{tweet.tweet_id}"
                        if tweet.author_handle
                        else f"https://x.com/i/status/{tweet.tweet_id}"
                    )
                    items.append(
                        CollectedItem(
                            external_id=tweet.tweet_id,
                            canonical_url=url,
                            text=tweet.text[:MAX_TEXT],
                            published_at=tweet.published_at.isoformat(),
                            author=tweet.author_handle or "",
                            language=request.source.language,
                            content_type="post",
                            discovery_mode="search",
                            outbound_urls=list(tweet.outbound_urls),
                            media_metadata=tweet.media_metadata or {},
                            raw_payload={
                                "tweet_id": tweet.tweet_id,
                                "conversation_id": tweet.conversation_id,
                                "query": keyword,
                                "is_quote": tweet.is_quote,
                                "provider_order": "engagement_ranked",
                            },
                        )
                    )
                except (ValueError, KeyError, TypeError, OverflowError):
                    skipped += 1
            batch = _finish(request, items, cost=cost, seen=len(raw), skipped=skipped)
            batch.metadata["provider_order"] = "engagement_ranked"
            return batch
        except (PlatformError, ValueError, TypeError) as exc:
            return _failure(exc, cost=cost)


def _weibo_posts(payload: Any) -> list[dict[str, Any]]:
    body = unwrap_data(payload)
    if not isinstance(body, dict):
        raise PlatformError("Weibo returned no search object", status="schema_changed")
    if isinstance(body.get("items"), list):
        # Observed app search schema: post feed records share a list with other
        # search categories. Do not mistake a user card or an ad for a post.
        posts = []
        for item in body["items"][:100]:
            if not isinstance(item, dict) or item.get("category") != "feed":
                continue
            post = item.get("data")
            if not isinstance(post, dict):
                posts.append({})  # Preserve malformed feed records for drift reporting.
            elif not (post.get("ad_marked") or post.get("ad_object") or post.get("ad_state")):
                posts.append(post)
        return posts
    cards = body.get("cards")
    if not isinstance(cards, list):
        raise PlatformError("Weibo search has no cards", status="schema_changed")
    posts: list[dict[str, Any]] = []
    for card in cards[:100]:
        if not isinstance(card, dict):
            continue
        for candidate in [card, *(card.get("card_group") or [])[:100]]:
            if isinstance(candidate, dict) and isinstance(candidate.get("mblog"), dict):
                posts.append(candidate["mblog"])
    return posts


class WeiboSearchAdapter(_MonidAdapter):
    def collect(self, request: CollectionRequest) -> CollectionBatch:
        cost = 0.0
        try:
            response = self.fetcher(
                WEIBO_SEARCH_ENDPOINT,
                {"query": request.source.locator, "page": 1, "search_type": 61},
            )
            cost = response.cost_usd
            raw = _weibo_posts(response.payload)
            items, skipped = [], 0
            for post in raw:
                try:
                    identifier = str(post.get("idstr") or post.get("id") or "")
                    text = strip_markup(str(post.get("text_raw") or post.get("text") or ""))
                    if not identifier or not text:
                        raise ValueError("Incomplete Weibo post")
                    user = post.get("user") or {}
                    url = f"https://m.weibo.cn/detail/{identifier}"
                    items.append(
                        CollectedItem(
                            external_id=identifier,
                            canonical_url=url,
                            text=text[:MAX_TEXT],
                            published_at=_iso(post.get("created_at")),
                            author=str(user.get("screen_name") or ""),
                            language=request.source.language,
                            discovery_mode="search",
                            outbound_urls=extract_urls(str(post.get("text") or ""), source_url=url),
                            engagement_metadata={
                                "reposts": post.get("reposts_count"),
                                "comments": post.get("comments_count"),
                                "likes": post.get("attitudes_count"),
                            },
                            raw_payload={
                                "id": identifier,
                                "user_id": user.get("id"),
                                "query": request.source.locator,
                                "content_basis": "excerpt" if post.get("isLongText") else "post",
                            },
                        )
                    )
                except (ValueError, TypeError, OverflowError):
                    skipped += 1
            return _finish(request, items, cost=cost, seen=len(raw), skipped=skipped)
        except (PlatformError, ValueError, TypeError) as exc:
            return _failure(exc, cost=cost)


class ZhihuSearchAdapter(_MonidAdapter):
    def collect(self, request: CollectionRequest) -> CollectionBatch:
        cost = 0.0
        try:
            response = self.fetcher(
                ZHIHU_SEARCH_ENDPOINT,
                {
                    "keyword": request.source.locator,
                    "offset": "0",
                    "limit": str(min(20, request.limit)),
                    "vertical": "article",
                    "sort": "created_time",
                    "time_interval": "a_week",
                },
            )
            cost = response.cost_usd
            body = unwrap_data(response.payload)
            raw = body.get("data") if isinstance(body, dict) else body
            if not isinstance(raw, list):
                raise PlatformError("Zhihu search has no result list", status="schema_changed")
            items, skipped = [], 0
            for result in raw[:100]:
                try:
                    post = result.get("object", result)
                    kind = post.get("type", "article")
                    if kind not in {"article", "answer"}:
                        continue
                    identifier = str(post["id"])
                    title = strip_markup(
                        post.get("title") or (post.get("question") or {}).get("title") or ""
                    )
                    content = strip_markup(post.get("content") or post.get("excerpt") or "")
                    if not identifier.isdigit() or not (title or content):
                        raise ValueError("Incomplete Zhihu item")
                    url = f"https://zhuanlan.zhihu.com/p/{identifier}"
                    if kind == "answer":
                        question_id = str((post.get("question") or {}).get("id") or "")
                        if not question_id.isdigit():
                            raise ValueError("Answer has no question ID")
                        url = f"https://www.zhihu.com/question/{question_id}/answer/{identifier}"
                    items.append(
                        CollectedItem(
                            external_id=f"{kind}:{identifier}",
                            canonical_url=url,
                            title=title,
                            text=(title + "\n\n" + content).strip()[:MAX_TEXT],
                            published_at=_iso(post.get("created_time") or post.get("created")),
                            author=str((post.get("author") or {}).get("name") or ""),
                            language=request.source.language,
                            content_type="article" if kind == "article" else "forum_post",
                            discovery_mode="search",
                            outbound_urls=extract_urls(post.get("content") or "", source_url=url),
                            engagement_metadata={
                                "votes": post.get("voteup_count"),
                                "comments": post.get("comment_count"),
                            },
                            raw_payload={
                                "query": request.source.locator,
                                "content_basis": "full" if post.get("content") else "excerpt",
                            },
                        )
                    )
                except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
                    skipped += 1
            return _finish(request, items, cost=cost, seen=len(raw), skipped=skipped)
        except (PlatformError, ValueError, TypeError) as exc:
            return _failure(exc, cost=cost)


def build_platform_adapters(settings: Settings, repository: Any = None) -> dict[str, Any]:
    del repository  # State and ingestion belong to the shared orchestrator.
    output_dir = settings.data_dir / "monid-runs"
    from app.collectors.platform_reddit_monid import RedditMonidAdapter

    return {
        "reddit_public": RedditAdapter(),
        "reddit_monid": RedditMonidAdapter(settings.data_dir / "reddit-cache"),
        "youtube_rss": YouTubeRSSAdapter(),
        "v2ex_public": V2EXAdapter(),
        "x_search_monid": XSearchAdapter(output_dir),
        "weibo_search_monid": WeiboSearchAdapter(output_dir),
        "zhihu_search_monid": ZhihuSearchAdapter(output_dir),
    }
