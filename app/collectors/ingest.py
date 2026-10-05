"""One bounded, secret-free entry point for source observations."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.collectors.base import CollectedItem
from app.config import SourceConfig
from app.db.repository import Repository

SECRET_KEYS = re.compile(
    r"cookie|authorization|password|secret|(?:^|_)token(?:$|_)|api.?key|ct0|session", re.I
)
SECRET_VALUE = re.compile(
    r"(?:sk-or-v1-[A-Za-z0-9_-]+|monid_live_[A-Za-z0-9_-]+|\d{8,12}:[A-Za-z0-9_-]{30,}|Bearer\s+[A-Za-z0-9._-]+)"
)


def redact_text(value: str) -> str:
    return SECRET_VALUE.sub("[REDACTED]", value)


def sanitize_payload(value: Any, *, depth: int = 0) -> Any:
    if depth > 7:
        return None
    if isinstance(value, dict):
        return {
            str(k)[:100]: sanitize_payload(v, depth=depth + 1)
            for k, v in list(value.items())[:80]
            if not SECRET_KEYS.search(str(k))
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_payload(v, depth=depth + 1) for v in value[:100]]
    if isinstance(value, str):
        return redact_text(value[:16000])
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return redact_text(str(value)[:1000])


def canonical_url(value: str) -> str:
    parts = urlsplit(value.strip())
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username:
        raise ValueError("Source URL must be public HTTP(S) without user credentials")
    host = parts.hostname.casefold()
    if host in {"twitter.com", "www.twitter.com", "www.x.com"}:
        host = "x.com"
    query = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.casefold().startswith("utm_")
        and k.casefold() not in {"fbclid", "gclid", "ref"}
        and not (
            host in {"x.com", "youtube.com", "www.youtube.com", "youtu.be"}
            and k.casefold() in {"s", "t"}
        )
        and not SECRET_KEYS.search(k)
    ]
    # Timestamp links to YouTube remain separate in captions, not event identity.
    return urlunsplit(
        (
            parts.scheme,
            host + (f":{parts.port}" if parts.port else ""),
            parts.path or "/",
            urlencode(query),
            "",
        )
    )


def ingest_item(
    repository: Repository,
    source: SourceConfig,
    item: CollectedItem,
    *,
    now: datetime | None = None,
    max_age_days: int | None = None,
) -> bool:
    current = now or datetime.now(UTC)
    if not item.external_id.strip() or not item.text.strip():
        raise ValueError("Missing stable identity or content")
    published = datetime.fromisoformat(item.published_at.replace("Z", "+00:00"))
    if published.tzinfo is None:
        raise ValueError("Publication time must include timezone")
    published = published.astimezone(UTC)
    if published > current + timedelta(minutes=10):
        raise ValueError("Publication time is in the future")
    if max_age_days is not None and published < current - timedelta(days=max_age_days):
        raise ValueError("Outside source freshness window")
    url = canonical_url(item.canonical_url)
    outbound = []
    for raw_url in item.outbound_urls[:30]:
        try:
            cleaned = canonical_url(raw_url)
        except ValueError:
            continue
        if cleaned != url and cleaned not in outbound:
            outbound.append(cleaned)
    raw = sanitize_payload(item.raw_payload)
    text = redact_text(item.text[:16000])
    # Stable content hash avoids treating engagement changes as new publications.
    payload_hash = hashlib.sha256(
        json.dumps([item.external_id, text, url], ensure_ascii=False).encode()
    ).hexdigest()
    item_id = repository.insert_source_item_with_id(
        source_id=source.id,
        external_id=item.external_id[:512],
        source_url=url,
        original_text=text,
        original_language=item.language or source.language,
        published_at=published.isoformat(),
        outbound_urls=outbound,
        media_metadata=sanitize_payload(item.media_metadata),
        payload_hash=payload_hash,
        raw_payload=raw,
    )
    with repository.connection() as db:
        # Only expand an unconsumed X thread; published evidence snapshots stay immutable.
        if item_id is None and source.kind.value == "x":
            existing = db.execute(
                "SELECT id,raw_payload_json FROM source_items WHERE source_id=? AND external_id=?",
                (source.id, item.external_id[:512]),
            ).fetchone()
            if existing:
                from app.collectors.x_monid import _payload_tweet_ids

                old_ids = _payload_tweet_ids(existing["raw_payload_json"])
                new_ids = _payload_tweet_ids(raw)
                attached = db.execute(
                    "SELECT 1 FROM event_source_items WHERE source_item_id=?", (existing["id"],)
                ).fetchone()
                if old_ids and new_ids > old_ids and not attached:
                    db.execute(
                        "UPDATE source_items SET original_text=?,outbound_urls_json=?,"
                        "raw_payload_json=?,payload_hash=? WHERE id=?",
                        (text, json.dumps(outbound), json.dumps(raw), payload_hash, existing["id"]),
                    )
        db.execute(
            """UPDATE source_items SET canonical_url=?, content_type=?, author_json=?,
            engagement_json=?, discovery_mode=?, last_observed_at=?,
            observation_count=observation_count+?
            WHERE source_id=? AND external_id=?""",
            (
                url,
                item.content_type[:60],
                json.dumps({"name": redact_text(item.author[:300])}),
                json.dumps(sanitize_payload(item.engagement_metadata)),
                item.discovery_mode[:60],
                current.isoformat(),
                int(item_id is None),
                source.id,
                item.external_id[:512],
            ),
        )
    return item_id is not None
