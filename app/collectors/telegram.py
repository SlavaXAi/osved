from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from telethon import TelegramClient

from app.collectors.base import CollectionStats
from app.config import Settings, SourceConfig, SourceKind, load_source_registry
from app.db.repository import Repository

HTTP_URL_RE = re.compile(r"https?://[^\s<>\"]+", re.IGNORECASE)
TRAILING_URL_PUNCTUATION = ".,;:!?)]}»”'"
TRACKING_QUERY_KEYS = {"fbclid", "gclid", "mc_cid", "mc_eid"}


class TelegramReaderNotConfigured(RuntimeError):
    pass


class TelegramSessionNotAuthorized(RuntimeError):
    pass


@dataclass(frozen=True)
class TelegramItem:
    external_id: str
    source_url: str
    original_text: str
    original_language: str
    published_at: str
    outbound_urls: list[str]
    media_metadata: dict[str, Any] | None
    payload_hash: str
    raw_payload: dict[str, Any]


def canonical_http_url(value: str) -> str | None:
    candidate = value.strip().rstrip(TRAILING_URL_PUNCTUATION)
    parts = urlsplit(candidate)
    if parts.scheme.lower() not in {"http", "https"} or not parts.netloc:
        return None
    query = urlencode(
        [
            (key, value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
            if not key.lower().startswith("utm_") and key.lower() not in TRACKING_QUERY_KEYS
        ],
        doseq=True,
    )
    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            parts.path,
            query,
            "",
        )
    )


def extract_message_urls(message: Any) -> list[str]:
    candidates: list[str] = []
    get_entities_text = getattr(message, "get_entities_text", None)
    if callable(get_entities_text):
        for entity, text in get_entities_text():
            candidates.append(getattr(entity, "url", None) or text)

    text = getattr(message, "raw_text", None) or getattr(message, "text", None) or ""
    candidates.extend(HTTP_URL_RE.findall(text))

    webpage = getattr(message, "web_preview", None)
    webpage_url = getattr(webpage, "url", None)
    if webpage_url:
        candidates.append(webpage_url)

    normalized: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        url = canonical_http_url(str(candidate))
        if url and url not in seen:
            seen.add(url)
            normalized.append(url)
    return normalized


def telegram_post_url(locator: str, message_id: int) -> str:
    return f"https://t.me/{locator.lstrip('@')}/{message_id}"


def message_to_item(source: SourceConfig, message: Any) -> TelegramItem:
    message_id = int(message.id)
    source_url = telegram_post_url(source.locator, message_id)
    text = getattr(message, "raw_text", None) or ""
    published_at = message.date
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=UTC)
    published_at_iso = published_at.astimezone(UTC).isoformat()
    outbound_urls = [url for url in extract_message_urls(message) if url != source_url]

    media = getattr(message, "media", None)
    media_metadata = None
    if media is not None:
        media_metadata = {
            "kind": type(media).__name__,
            "has_photo": getattr(message, "photo", None) is not None,
            "has_video": getattr(message, "video", None) is not None,
            "has_document": getattr(message, "document", None) is not None,
        }

    fingerprint_payload = {
        "source_id": source.id,
        "message_id": message_id,
        "published_at": published_at_iso,
        "text": text,
        "outbound_urls": outbound_urls,
    }
    serialized = json.dumps(
        fingerprint_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    payload_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    raw_payload = {
        **fingerprint_payload,
        "grouped_id": getattr(message, "grouped_id", None),
        "forward": bool(getattr(message, "forward", None)),
        "edit_date": str(getattr(message, "edit_date", None) or "") or None,
    }
    return TelegramItem(
        external_id=str(message_id),
        source_url=source_url,
        original_text=text,
        original_language=source.language,
        published_at=published_at_iso,
        outbound_urls=outbound_urls,
        media_metadata=media_metadata,
        payload_hash=payload_hash,
        raw_payload=raw_payload,
    )


class TelegramSourceCollector:
    def __init__(self, settings: Settings, repository: Repository) -> None:
        self.settings = settings
        self.repository = repository

    def _credentials(self) -> tuple[int, str]:
        if self.settings.telegram_api_id is None or self.settings.telegram_api_hash is None:
            raise TelegramReaderNotConfigured("TELEGRAM_API_ID and TELEGRAM_API_HASH are required")
        return (
            self.settings.telegram_api_id,
            self.settings.telegram_api_hash.get_secret_value(),
        )

    async def collect(self, *, limit_per_source: int = 100) -> CollectionStats:
        api_id, api_hash = self._credentials()
        registry = load_source_registry(self.settings.sources_file)
        sources = [
            source for source in registry.enabled_sources if source.kind == SourceKind.TELEGRAM
        ]
        client = TelegramClient(
            str(self.settings.telegram_session_path),
            api_id,
            api_hash,
        )
        await client.connect()
        try:
            if not await client.is_user_authorized():
                raise TelegramSessionNotAuthorized(
                    "Run `newsfeed telegram-session-setup` locally first"
                )

            sources_succeeded = 0
            items_seen = 0
            items_inserted = 0
            errors = 0
            for source in sources:
                try:
                    seen, inserted = await self._collect_source(
                        client,
                        source,
                        limit_per_source=limit_per_source,
                    )
                    sources_succeeded += 1
                    items_seen += seen
                    items_inserted += inserted
                except Exception as exc:
                    errors += 1
                    self.repository.mark_source_error(source.id, str(exc))
            return CollectionStats(
                sources_attempted=len(sources),
                sources_succeeded=sources_succeeded,
                items_seen=items_seen,
                items_inserted=items_inserted,
                errors=errors,
                cost_usd=0,
            )
        finally:
            await client.disconnect()

    async def _collect_source(
        self,
        client: TelegramClient,
        source: SourceConfig,
        *,
        limit_per_source: int,
    ) -> tuple[int, int]:
        cursor = self.repository.get_source_cursor(source.id)
        min_id = int(cursor) if cursor else 0
        messages = []
        async for message in client.iter_messages(
            source.locator,
            min_id=min_id,
            reverse=True,
            limit=limit_per_source,
        ):
            if not getattr(message, "id", None) or not getattr(message, "date", None):
                continue
            messages.append(message)

        inserted = 0
        max_message_id = min_id
        for message in messages:
            item = message_to_item(source, message)
            if self.repository.insert_source_item(
                source_id=source.id,
                external_id=item.external_id,
                source_url=item.source_url,
                original_text=item.original_text,
                original_language=item.original_language,
                published_at=item.published_at,
                outbound_urls=item.outbound_urls,
                media_metadata=item.media_metadata,
                payload_hash=item.payload_hash,
                raw_payload=item.raw_payload,
            ):
                inserted += 1
            max_message_id = max(max_message_id, int(item.external_id))

        self.repository.mark_source_success(
            source.id,
            cursor=str(max_message_id) if max_message_id else cursor,
            success_at=datetime.now(UTC).isoformat(),
        )
        return len(messages), inserted


def setup_telegram_session(settings: Settings) -> dict[str, Any]:
    if settings.telegram_api_id is None or settings.telegram_api_hash is None:
        raise TelegramReaderNotConfigured("TELEGRAM_API_ID and TELEGRAM_API_HASH are required")
    settings.ensure_runtime_directories()
    with TelegramClient(
        str(settings.telegram_session_path),
        settings.telegram_api_id,
        settings.telegram_api_hash.get_secret_value(),
    ) as client:
        me = client.get_me()
        return {
            "authorized": True,
            "user_id": me.id,
            "username": me.username,
            "session_path": str(settings.telegram_session_path),
        }
