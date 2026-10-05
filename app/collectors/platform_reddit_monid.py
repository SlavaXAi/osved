"""Explicitly selected paid Reddit transport, never an invisible RSS fallback."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from app.collectors.base import CollectionBatch, CollectionRequest, SourceCapabilities, SourceHealth
from app.collectors.platform_transport import MonidResponse, PlatformError, monid_fetch, unwrap_data

REDDIT_FEED_ENDPOINT = "/api/v1/reddit/app/fetch_subreddit_feed"
REDDIT_SEARCH_ENDPOINT = "/api/v1/reddit/app/fetch_dynamic_search"
REDDIT_CALL_COST_USD = 0.0015
PaidFetcher = Callable[[str, dict[str, Any]], MonidResponse]


def _cell_posts(body: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Normalize the observed Reddit app GraphQL feed, excluding ads/layout cells.

    These cells expose a title and sometimes a truncated excerpt, not a complete
    post. The stable t3 group ID is Reddit's real post identity; opaque cell IDs
    are not permalinks. An unknown envelope still fails instead of looking empty.
    """
    feed = body.get("subredditfeed")
    if not isinstance(feed, dict):
        return None
    subreddit = feed.get("subredditV3")
    elements = subreddit.get("elements") if isinstance(subreddit, dict) else None
    edges = elements.get("edges") if isinstance(elements, dict) else None
    if not isinstance(edges, list):
        return None
    posts = []
    for edge in edges[:100]:
        node = edge.get("node") if isinstance(edge, dict) else None
        if not isinstance(node, dict) or node.get("adPayload") is not None:
            continue
        identifier = node.get("groupId")
        if not isinstance(identifier, str) or not re.fullmatch(r"t3_[a-z0-9]+", identifier):
            continue  # Sorting widgets and unrelated app layout are not posts.
        cells = node.get("cells")
        if not isinstance(cells, list):
            posts.append({})
            continue
        post: dict[str, Any] = {"id": identifier, "content_basis": "title_only"}
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            for source, target in (
                ("createdAt", "created_at"),
                ("authorName", "author"),
                ("score", "score"),
                ("commentCount", "num_comments"),
            ):
                if cell.get(source) is not None:
                    post[target] = cell[source]
            title_cell = cell.get("titleCell")
            title = cell.get("title") or (
                title_cell.get("title") if isinstance(title_cell, dict) else None
            )
            if isinstance(title, str):
                post["title"] = title
            if isinstance(cell.get("text"), str) and cell["text"].strip():
                post["text"] = cell["text"]
                post["content_basis"] = "title_and_excerpt"
        if isinstance(post.get("author"), str):
            post["author"] = post["author"].removeprefix("u/")
        posts.append(post)
    return posts


def _subreddit(locator: str) -> str:
    if "://" in locator:
        parsed = urlsplit(locator)
        if parsed.hostname not in {"www.reddit.com", "reddit.com"}:
            raise PlatformError("Reddit locator must identify an actual subreddit")
        match = re.search(r"/r/([A-Za-z0-9_]+)", parsed.path)
        locator = match.group(1) if match else ""
    locator = locator.removeprefix("r/").strip()
    if not re.fullmatch(r"[A-Za-z0-9_]{2,21}", locator):
        raise PlatformError("Reddit Monid feed requires one valid subreddit name")
    return locator


def _extract_posts(payload: Any) -> list[dict[str, Any]]:
    """Accept standard listings and TikHub's formatted list, fail on unknown shapes."""
    body = unwrap_data(payload)
    if isinstance(body, dict) and isinstance(body.get("data"), dict):
        body = body["data"]
    if isinstance(body, dict):
        cell_posts = _cell_posts(body)
        if cell_posts is not None:
            return cell_posts
    if isinstance(body, dict) and isinstance(body.get("children"), list):
        return [
            child["data"] if isinstance(child, dict) and isinstance(child.get("data"), dict) else {}
            for child in body["children"]
        ]
    if isinstance(body, list):
        posts = body
    elif isinstance(body, dict):
        posts = next(
            (body[key] for key in ("posts", "items", "results") if isinstance(body.get(key), list)),
            None,
        )
    else:
        posts = None
    if posts is None:
        raise PlatformError(
            "Reddit Monid returned an unrecognized post list", status="schema_changed"
        )
    normalized = []
    for row in posts[:100]:
        if not isinstance(row, dict):
            normalized.append({})
            continue
        post = row.get("post", row.get("data", row))
        normalized.append(post if isinstance(post, dict) else {})
    return normalized


def _listing(payload: Any, subreddit: str) -> dict[str, Any]:
    posts = _extract_posts(payload)
    children = []
    for post in posts:
        identifier = str(
            post.get("id") or post.get("post_id") or post.get("postId") or ""
        ).removeprefix("t3_")
        title = str(post.get("title") or "").strip()
        # Missing date or identity remains missing: the common parser marks drift.
        created = post.get(
            "created_utc", post.get("created_at", post.get("createdAt", post.get("created")))
        )
        author = post.get("author")
        if isinstance(author, dict):
            author = author.get("name") or author.get("username") or ""
        permalink = post.get("permalink") or post.get("permaLink")
        if isinstance(permalink, str) and permalink.startswith("http"):
            parsed = urlsplit(permalink)
            permalink = parsed.path if parsed.hostname in {"www.reddit.com", "reddit.com"} else ""
        if not permalink and re.fullmatch(r"[a-z0-9]+", identifier):
            permalink = f"/r/{subreddit}/comments/{identifier}/"
        children.append(
            {
                "kind": "t3",
                "data": {
                    "id": identifier,
                    "title": title,
                    "created_utc": created,
                    "author": str(author or ""),
                    "permalink": permalink,
                    "selftext": post.get("selftext") or post.get("text") or post.get("body") or "",
                    "url": post.get("url") or post.get("outbound_url") or "",
                    "score": post.get("score", post.get("upvote_count")),
                    "num_comments": post.get("num_comments", post.get("comment_count")),
                    "subreddit": subreddit,
                    "promoted": bool(post.get("promoted") or post.get("isSponsored")),
                    "stickied": bool(post.get("stickied") or post.get("isStickied")),
                    "over_18": bool(post.get("over_18") or post.get("isNsfw")),
                },
            }
        )
    return {"data": {"children": children, "after": None}}


class RedditMonidAdapter:
    """The shared source claim must reserve capabilities().estimated_cost_usd first.

    Selecting this adapter and explicitly setting per_call_price_cap_usd is
    required. RSS adapters never call it themselves. One page, one paid call,
    bounded durable TTL cache, process lock, and no automatic pagination/retry.
    """

    def __init__(self, cache_dir: Path, *, fetcher: PaidFetcher | None = None) -> None:
        self.cache_dir = Path(cache_dir)
        self.fetcher = fetcher or (
            lambda endpoint, query: monid_fetch(endpoint, query, self.cache_dir / "runs")
        )

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            strategies=("feed", "new", "hot", "search"),
            requires_auth=True,
            estimated_cost_usd=REDDIT_CALL_COST_USD,
        )

    def healthcheck(self, source: Any) -> SourceHealth:
        cap = source.options.get("per_call_price_cap_usd")
        if (
            not isinstance(cap, (int, float))
            or isinstance(cap, bool)
            or not math.isfinite(cap)
            or cap < REDDIT_CALL_COST_USD
        ):
            return SourceHealth(
                status="disabled",
                error="Set per_call_price_cap_usd >= 0.0015 explicitly for paid Reddit",
            )
        return SourceHealth(status="ok")

    def collect(self, request: CollectionRequest) -> CollectionBatch:
        # Also enforce the opt-in when collect() is invoked outside the orchestrator.
        health = self.healthcheck(request.source)
        if health.status != "ok":
            return CollectionBatch(status=health.status, error=health.error)
        cost = 0.0
        try:
            subreddit = _subreddit(request.source.locator)
            query: dict[str, Any] = {"need_format": True, "language": "en-US"}
            endpoint = REDDIT_FEED_ENDPOINT
            if request.source.strategy == "search":
                text = str(request.source.options.get("query") or "").strip()
                if not text or len(text) > 450:
                    raise PlatformError("Reddit search requires a query of 1–450 characters")
                query.update(
                    query=f"subreddit:{subreddit} {text}",
                    search_type="post",
                    sort="NEW",
                    time_range="week",
                    safe_search="strict",
                    allow_nsfw="0",
                )
                endpoint = REDDIT_SEARCH_ENDPOINT
            else:
                query.update(
                    subreddit_name=subreddit,
                    sort="HOT" if request.source.strategy == "hot" else "NEW",
                )
            key = hashlib.sha256(
                json.dumps({"endpoint": endpoint, "query": query}, sort_keys=True).encode()
            ).hexdigest()
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = self.cache_dir / f"reddit-{key}.json"
            ttl = max(5, min(1440, int(request.source.options.get("cache_ttl_minutes", 60)))) * 60
            with (self.cache_dir / f"reddit-{key}.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                payload, cached = None, False
                if cache_path.is_file() and cache_path.stat().st_size <= 4_100_000:
                    try:
                        stored = json.loads(cache_path.read_text(encoding="utf-8"))
                        age = request.now.timestamp() - float(stored["saved_at"])
                        if 0 <= age < ttl:
                            payload, cached = stored["payload"], True
                    except (ValueError, KeyError, TypeError, OSError):
                        pass
                if not cached:
                    response = self.fetcher(endpoint, query)
                    payload, cost = response.payload, response.cost_usd
                    if cost > float(request.source.options["per_call_price_cap_usd"]) + 1e-9:
                        raise PlatformError(
                            "Reddit provider billed above the configured per-call ceiling",
                            status="disabled",
                            cost_usd=cost,
                        )
                    # Verify envelope before caching: stale schema must not look like empty data.
                    _extract_posts(payload)
                    descriptor, temporary = tempfile.mkstemp(prefix=".reddit-", dir=self.cache_dir)
                    try:
                        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                            json.dump(
                                {"saved_at": request.now.timestamp(), "payload": payload},
                                handle,
                                ensure_ascii=False,
                            )
                        os.replace(temporary, cache_path)
                    finally:
                        if os.path.exists(temporary):
                            os.unlink(temporary)
            from app.collectors.platforms import RedditAdapter

            posts = _extract_posts(payload)
            content_bases = {
                str(post.get("id", "")).removeprefix("t3_"): post.get("content_basis")
                for post in posts
            }
            batch = RedditAdapter()._json(
                request, json.dumps(_listing(payload, subreddit)).encode()
            )
            return CollectionBatch(
                items=[
                    replace(
                        item,
                        discovery_mode=request.source.strategy,
                        raw_payload={
                            **item.raw_payload,
                            "transport": "monid/tikhub",
                            "content_basis": content_bases.get(item.external_id.removeprefix("t3_"))
                            or "post_listing",
                        },
                    )
                    for item in batch.items
                ],
                status=batch.status,
                cost_usd=cost,
                error=batch.error,
                metadata={
                    **batch.metadata,
                    "adapter": "reddit_monid",
                    "cache_hit": cached,
                    "provider_cost_ceiling": REDDIT_CALL_COST_USD,
                },
            )
        except (PlatformError, ValueError, TypeError, OSError) as exc:
            return CollectionBatch(
                status=exc.status if isinstance(exc, PlatformError) else "schema_changed",
                error=str(exc)[:500],
                cost_usd=max(cost, getattr(exc, "cost_usd", 0)),
            )
