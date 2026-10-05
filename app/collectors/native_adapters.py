"""Small adapters reuse the existing, tested platform parsers."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Mapping
from xml.etree import ElementTree

from app.collectors.base import (
    CollectedItem,
    CollectionBatch,
    CollectionRequest,
    SourceCapabilities,
    SourceHealth,
)
from app.config import Settings, SourceConfig


def converted(item, source: SourceConfig, *, author: str = "") -> CollectedItem:
    return CollectedItem(
        external_id=item.external_id,
        canonical_url=item.source_url,
        text=item.original_text,
        published_at=item.published_at,
        title=str((item.raw_payload or {}).get("title") or ""),
        author=author,
        language=getattr(item, "original_language", None) or source.language,
        content_type="article" if source.kind.value == "rss" else "post",
        discovery_mode=source.strategy,
        outbound_urls=item.outbound_urls,
        raw_payload=item.raw_payload,
        media_metadata=getattr(item, "media_metadata", None) or {},
    )


class NativeAdapter:
    def __init__(self, settings: Settings, kind: str):
        self.settings, self.kind = settings, kind

    def capabilities(self) -> SourceCapabilities:
        paid = self.kind in {"monid_x_author", "monid_telegram"}
        return SourceCapabilities(
            strategies={
                "rss": ("feed",),
                "github": ("releases", "feed"),
                "monid_x_author": ("author_timeline", "feed"),
                "monid_telegram": ("feed",),
                "telegram_mtproto": ("feed",),
            }.get(self.kind, ()),
            requires_auth=paid or self.kind == "telegram_mtproto",
            estimated_cost_usd=0.0015 if paid else 0,
            incremental=True,
        )

    def healthcheck(self, source: SourceConfig) -> SourceHealth:
        if self.kind.startswith("monid_") and self.settings.monid_api_key is None:
            return SourceHealth("login_required", "Monid credentials missing")
        if (
            self.kind == "telegram_mtproto"
            and not self.settings.credential_status()["telegram_reader"]
        ):
            return SourceHealth("login_required", "Telegram reader credentials missing")
        return SourceHealth()

    def collect(self, request: CollectionRequest) -> CollectionBatch:
        source = request.source
        if self.kind == "rss":
            from app.collectors.rss import (
                UNSAFE_XML_RE,
                RSSCollectorError,
                fetch_feed,
                local_name,
                parse_feed,
            )

            payload = fetch_feed(source)
            try:
                if UNSAFE_XML_RE.search(payload):
                    raise RSSCollectorError("Forbidden XML declaration")
                root = ElementTree.fromstring(payload)
                if local_name(root.tag) not in {"rss", "feed", "RDF"}:
                    raise RSSCollectorError("Expected RSS/Atom, received another document")
                entries = [
                    node for node in root.iter() if local_name(node.tag) in {"entry", "item"}
                ]
                items = [
                    converted(item, source) for item in parse_feed(payload, limit=request.limit)
                ]
            except (ElementTree.ParseError, RSSCollectorError) as exc:
                return CollectionBatch(status="schema_changed", error=str(exc))
            malformed = min(len(entries), request.limit) - len(items)
            return CollectionBatch(
                items,
                "schema_changed"
                if entries and not items
                else "partial"
                if malformed
                else "ok"
                if items
                else "empty",
                max((x.published_at for x in items), default=request.cursor),
                metadata={"items_seen": min(len(entries), request.limit), "malformed": malformed},
            )
        if self.kind == "github":
            from app.collectors.github import fetch_releases, release_text, release_urls
            from app.collectors.rss import RSSCollectorError, normalize_published

            items = []
            malformed = filtered = 0
            releases = fetch_releases(source, request.limit)
            for release in releases:
                if release.get("draft") or release.get("prerelease"):
                    filtered += 1
                    continue
                if not release.get("id"):
                    malformed += 1
                    continue
                url = str(release.get("html_url") or "")
                date = str(release.get("published_at") or release.get("created_at") or "")
                if not url or not date:
                    malformed += 1
                    continue
                try:
                    published = normalize_published(date)
                except RSSCollectorError:
                    malformed += 1
                    continue
                author = release.get("author")
                items.append(
                    CollectedItem(
                        str(release["id"]),
                        url,
                        release_text(source, release),
                        published,
                        author=str(author.get("login") or "") if isinstance(author, dict) else "",
                        content_type="release",
                        discovery_mode="releases",
                        language=source.language,
                        outbound_urls=release_urls(release, url),
                        raw_payload=release,
                    )
                )
            return CollectionBatch(
                items,
                "schema_changed"
                if malformed and not items
                else "partial"
                if malformed
                else "ok"
                if items
                else "empty",
                max((x.published_at for x in items), default=request.cursor),
                metadata={
                    "items_seen": len(releases),
                    "malformed": malformed,
                    "filtered": filtered,
                },
            )
        if self.kind == "telegram_mtproto":
            return asyncio.run(self._telegram(request))
        output = self.settings.data_dir / "monid-runs" / f"{source.id}-{uuid.uuid4().hex}.json"
        if self.kind == "monid_x_author":
            from app.collectors.x_monid import (
                MonidXError,
                _timeline_list,
                _unwrap_tweet,
                fetch_user_timeline,
                normalize_tweet,
                timeline_to_items,
                tweet_id,
            )

            payload, cost = fetch_user_timeline(source, output)
            try:
                timeline = _timeline_list(payload)
            except MonidXError:
                return CollectionBatch(
                    status="schema_changed", cost_usd=cost, error="X response has no timeline array"
                )
            user = payload.get("user")
            user = user if isinstance(user, Mapping) else {}
            usable = []
            malformed = max(0, len(timeline) - 100)
            filtered = 0
            # Validate record-by-record so one broken timestamp does not drop an
            # entire paid page; then reuse grouping for complete author threads.
            for value in timeline[:100]:
                if not isinstance(value, Mapping):
                    malformed += 1
                    continue
                identifier = tweet_id(_unwrap_tweet(value))
                if not identifier:
                    entry_id = str(value.get("entryId") or "")
                    if entry_id.startswith(("cursor-", "promoted-", "who-to-follow")):
                        filtered += 1
                    else:
                        malformed += 1
                    continue
                try:
                    normalize_tweet(source, value, fallback_user=user)
                except (ValueError, KeyError, TypeError, OverflowError):
                    malformed += 1
                    continue
                usable.append(value)
            parsing_source = source
            if self.settings.app_env != "production":
                parsing_source = source.model_copy(
                    update={
                        "production_max_age_days": max(1, source.calibration_lookback_days),
                    }
                )
            items, ids = timeline_to_items(parsing_source, usable, user=user, now=request.now)
            cursor_ids = [str(x) for x in ids if str(x).isdigit()]
            if request.cursor and request.cursor.isdigit():
                cursor_ids.append(request.cursor)
            return CollectionBatch(
                [converted(x, source, author=source.locator) for x in items],
                "schema_changed"
                if malformed and not usable
                else "partial"
                if malformed
                else "ok"
                if items
                else "empty",
                request.cursor
                if malformed
                else max(cursor_ids, key=int)
                if cursor_ids
                else request.cursor,
                cost,
                metadata={
                    "items_seen": len(timeline),
                    "malformed": malformed,
                    "filtered": filtered,
                    "complete_snapshot": True,
                    "ingest_limit": 100,
                    "truncated": len(timeline) > 100,
                },
            )
        if self.kind == "monid_telegram":
            from app.collectors.telegram_monid import fetch_channel_posts, monid_message_to_item

            latest_window = self.settings.app_env == "production"
            payload, cost = fetch_channel_posts(
                source, request.limit, 0 if latest_window else int(request.cursor or 0), output
            )
            messages = payload.get("messages")
            if not isinstance(messages, list):
                return CollectionBatch(
                    status="schema_changed",
                    cost_usd=cost,
                    error="Telegram response has no messages array",
                )
            items = []
            errors = max(0, len(messages) - 100)
            for message in messages[:100]:
                try:
                    items.append(
                        converted(
                            monid_message_to_item(source, message), source, author=source.locator
                        )
                    )
                except (ValueError, KeyError, TypeError):
                    errors += 1
            cursor = str(max([int(request.cursor or 0), *[int(x.external_id) for x in items]]))
            return CollectionBatch(
                items,
                "schema_changed"
                if errors and not items
                else "partial"
                if errors
                else "ok"
                if items
                else "empty",
                request.cursor if errors else cursor,
                cost,
                metadata={
                    "malformed": errors,
                    "items_seen": len(messages),
                    "complete_snapshot": True,
                    "ingest_limit": 100,
                    "coverage": "latest_window" if latest_window else "cursor_catchup",
                },
            )
        raise ValueError(f"Unknown native adapter: {self.kind}")

    async def _telegram(self, request: CollectionRequest) -> CollectionBatch:
        from telethon import TelegramClient

        from app.collectors.telegram import message_to_item

        key = self.settings.telegram_api_hash
        if key is None or self.settings.telegram_api_id is None:
            return CollectionBatch(status="login_required", error="Missing Telegram reader key")
        client = TelegramClient(
            str(self.settings.telegram_session_path),
            self.settings.telegram_api_id,
            key.get_secret_value(),
        )
        await client.connect()
        try:
            if not await client.is_user_authorized():
                return CollectionBatch(
                    status="login_required", error="Telegram session not authorized"
                )
            items = []
            malformed = 0
            cursor = int(request.cursor or 0)
            latest_window = self.settings.app_env == "production"
            async for message in client.iter_messages(
                request.source.locator,
                min_id=0 if latest_window else cursor,
                # Without a cursor, start with current news rather than the
                # channel's oldest post. Later polls consume additions in order.
                reverse=cursor > 0 and not latest_window,
                limit=request.limit,
            ):
                if getattr(message, "date", None) and getattr(message, "id", None):
                    try:
                        items.append(
                            converted(message_to_item(request.source, message), request.source)
                        )
                    except (ValueError, TypeError, KeyError, OverflowError):
                        malformed += 1
            return CollectionBatch(
                items,
                "schema_changed"
                if malformed and not items
                else "partial"
                if malformed
                else "ok"
                if items
                else "empty",
                request.cursor
                if malformed
                else str(max([cursor, *[int(x.external_id) for x in items]])),
                metadata={
                    "malformed": malformed,
                    "coverage": "latest_window" if latest_window else "cursor_catchup",
                },
            )
        finally:
            await client.disconnect()
