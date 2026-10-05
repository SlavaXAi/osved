from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.collectors.base import CollectionStats
from app.collectors.telegram import HTTP_URL_RE, TelegramItem, canonical_http_url
from app.config import Settings, SourceConfig, SourceKind, load_source_registry
from app.db.repository import Repository
from app.ops.budget import evaluate_budget

MONID_PROVIDER = "tikhub"
MONID_ENDPOINT = "/api/v1/telegram/web/fetch_channel_posts"
MONID_CALL_COST_USD = 0.0015
HREF_RE = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)


class MonidTelegramNotConfigured(RuntimeError):
    pass


class MonidTelegramError(RuntimeError):
    pass


FetchResult = tuple[dict[str, Any], float]
ChannelFetcher = Callable[[SourceConfig, int, int, Path], FetchResult]


def monid_cli_path() -> str:
    discovered = shutil.which("monid")
    if discovered:
        return discovered
    user_local = Path.home() / ".local/bin/monid"
    if user_local.is_file():
        return str(user_local)
    raise MonidTelegramNotConfigured("Monid CLI is not installed")


def fetch_channel_posts(
    source: SourceConfig,
    limit: int,
    after: int,
    output_path: Path,
) -> FetchResult:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    query = {
        "channel": source.locator.lstrip("@"),
        "limit": limit,
        "after": after,
    }
    command = [
        monid_cli_path(),
        "run",
        "-p",
        MONID_PROVIDER,
        "-e",
        MONID_ENDPOINT,
        "--query",
        json.dumps(query, separators=(",", ":")),
        "-w",
        "30",
        "-o",
        str(output_path),
        "-j",
    ]
    environment = os.environ.copy()
    environment["NO_COLOR"] = "1"
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=75,
        check=False,
        env=environment,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise MonidTelegramError(detail[-2000:] or "Monid command failed")
    if not output_path.is_file():
        raise MonidTelegramError("Monid completed without writing its output file")
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MonidTelegramError("Monid returned a non-object payload")
    return payload, MONID_CALL_COST_USD


def extract_monid_urls(message: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    text = str(message.get("text") or "")
    candidates.extend(HTTP_URL_RE.findall(text))
    text_html = str(message.get("text_html") or "")
    candidates.extend(html.unescape(value) for value in HREF_RE.findall(text_html))

    preview = message.get("link_preview")
    if isinstance(preview, dict) and preview.get("url"):
        candidates.append(str(preview["url"]))

    normalized: list[str] = []
    seen: set[str] = set()
    source_url = canonical_http_url(str(message.get("url") or ""))
    for candidate in candidates:
        url = canonical_http_url(candidate)
        if url and url != source_url and url not in seen:
            seen.add(url)
            normalized.append(url)
    return normalized


def monid_message_to_item(source: SourceConfig, message: dict[str, Any]) -> TelegramItem:
    message_id = int(message["id"])
    source_url = str(
        message.get("url") or f"https://t.me/{source.locator.lstrip('@')}/{message_id}"
    )
    published = datetime.fromisoformat(str(message["date"]).replace("Z", "+00:00"))
    if published.tzinfo is None:
        published = published.replace(tzinfo=UTC)
    published_at = published.astimezone(UTC).isoformat()
    original_text = str(message.get("text") or "")
    outbound_urls = extract_monid_urls(message)

    media = message.get("media") if isinstance(message.get("media"), dict) else {}
    photos = media.get("photos") if isinstance(media.get("photos"), list) else []
    videos = media.get("videos") if isinstance(media.get("videos"), list) else []
    media_metadata = None
    if photos or videos or message.get("type") not in {None, "text"}:
        media_metadata = {
            "kind": message.get("type"),
            "photo_count": len(photos),
            "video_count": len(videos),
            "has_unsupported": bool(media.get("has_unsupported")),
        }

    fingerprint_payload = {
        "source_id": source.id,
        "message_id": message_id,
        "published_at": published_at,
        "text": original_text,
        "outbound_urls": outbound_urls,
    }
    serialized = json.dumps(
        fingerprint_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return TelegramItem(
        external_id=str(message_id),
        source_url=source_url,
        original_text=original_text,
        original_language=source.language,
        published_at=published_at,
        outbound_urls=outbound_urls,
        media_metadata=media_metadata,
        payload_hash=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        raw_payload=message,
    )


class MonidTelegramCollector:
    def __init__(
        self,
        settings: Settings,
        repository: Repository,
        *,
        fetcher: ChannelFetcher = fetch_channel_posts,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.fetcher = fetcher

    def collect(self, *, limit_per_source: int = 20) -> CollectionStats:
        registry = load_source_registry(self.settings.sources_file)
        sources = [
            source for source in registry.enabled_sources if source.kind == SourceKind.TELEGRAM
        ]
        succeeded = seen = inserted = errors = 0
        total_cost = 0.0

        for source in sources:
            projected = self.repository.current_month_cost() + MONID_CALL_COST_USD
            budget = evaluate_budget(
                projected,
                self.settings.budget_warning_usd,
                self.settings.budget_hard_stop_usd,
            )
            if not budget.paid_operations_allowed:
                self.repository.mark_source_error(source.id, "Monthly hard budget stop")
                errors += 1
                continue

            cursor = int(self.repository.get_source_cursor(source.id) or 0)
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
            output_path = (
                self.settings.data_dir
                / "monid-runs"
                / f"{source.id}-{timestamp}-{uuid.uuid4().hex[:8]}.json"
            )
            try:
                payload, cost = self.fetcher(
                    source,
                    min(limit_per_source, 100),
                    cursor,
                    output_path,
                )
                messages = payload.get("messages")
                if not isinstance(messages, list):
                    raise MonidTelegramError("Response has no messages array")
                source_seen, source_inserted, max_id = self._store_messages(
                    source,
                    messages,
                    cursor,
                )
                self.repository.mark_source_success(
                    source.id,
                    cursor=str(max_id) if max_id else str(cursor) or None,
                    success_at=datetime.now(UTC).isoformat(),
                )
                self.repository.record_usage(
                    "monid/tikhub",
                    "telegram.fetch_channel_posts",
                    cost_usd=cost,
                    metadata={"source_id": source.id, "items": source_seen},
                )
                succeeded += 1
                seen += source_seen
                inserted += source_inserted
                total_cost += cost
            except Exception as exc:
                errors += 1
                self.repository.mark_source_error(source.id, str(exc))

        return CollectionStats(
            sources_attempted=len(sources),
            sources_succeeded=succeeded,
            items_seen=seen,
            items_inserted=inserted,
            errors=errors,
            cost_usd=total_cost,
        )

    def _store_messages(
        self,
        source: SourceConfig,
        messages: list[Any],
        cursor: int,
    ) -> tuple[int, int, int]:
        seen = inserted = 0
        max_id = cursor
        for raw_message in messages:
            if not isinstance(raw_message, dict) or not raw_message.get("id"):
                continue
            item = monid_message_to_item(source, raw_message)
            seen += 1
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
            max_id = max(max_id, int(item.external_id))
        return seen, inserted, max_id
