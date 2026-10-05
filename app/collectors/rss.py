from __future__ import annotations

import hashlib
import html
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from app.collectors.base import CollectionStats
from app.collectors.telegram import HTTP_URL_RE, canonical_http_url
from app.config import Settings, SourceConfig, SourceKind, load_source_registry
from app.db.repository import Repository

MAX_FEED_BYTES = 2_000_000
MAX_ITEM_TEXT = 6_000
UNSAFE_XML_RE = re.compile(rb"<!\s*(?:DOCTYPE|ENTITY)\b", re.IGNORECASE)


class RSSCollectorError(RuntimeError):
    pass


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        value = " ".join(data.split())
        if value:
            self.parts.append(value)


@dataclass(frozen=True)
class FeedItem:
    external_id: str
    source_url: str
    original_text: str
    published_at: str
    outbound_urls: list[str]
    raw_payload: dict[str, Any]


FeedFetcher = Callable[[SourceConfig], bytes]


def strip_markup(value: str) -> str:
    parser = _TextExtractor()
    parser.feed(html.unescape(value))
    parser.close()
    return " ".join(parser.parts)


def fetch_feed(source: SourceConfig) -> bytes:
    request = Request(
        source.locator,
        headers={
            "Accept": "application/atom+xml, application/rss+xml, application/xml, text/xml",
            "User-Agent": "PersonalTechnologyNewsfeed/0.1 (+private single-user reader)",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:  # noqa: S310 - allowlisted URL
            content_length = int(response.headers.get("Content-Length") or 0)
            if content_length > MAX_FEED_BYTES:
                raise RSSCollectorError(f"Feed exceeds {MAX_FEED_BYTES} bytes")
            payload = response.read(MAX_FEED_BYTES + 1)
    except HTTPError as exc:
        raise RSSCollectorError(f"RSS HTTP {exc.code} for {source.id}") from exc
    except URLError as exc:
        raise RSSCollectorError(f"RSS network error for {source.id}: {exc.reason}") from exc
    if len(payload) > MAX_FEED_BYTES:
        raise RSSCollectorError(f"Feed exceeds {MAX_FEED_BYTES} bytes")
    return payload


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def child_text(element: ElementTree.Element, *names: str) -> str:
    wanted = set(names)
    for child in element:
        if local_name(child.tag) in wanted and child.text:
            return child.text.strip()
    return ""


def atom_link(element: ElementTree.Element) -> str:
    fallback = ""
    for child in element:
        if local_name(child.tag) != "link":
            continue
        href = str(child.attrib.get("href") or "").strip()
        if not href:
            continue
        if child.attrib.get("rel", "alternate") == "alternate":
            return href
        fallback = fallback or href
    return fallback


def normalize_published(value: str) -> str:
    if not value:
        raise RSSCollectorError("Feed item has no publication date")
    try:
        published = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        try:
            published = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise RSSCollectorError(f"Unsupported feed date: {value}") from exc
    if published.tzinfo is None:
        published = published.replace(tzinfo=UTC)
    return published.astimezone(UTC).isoformat()


def extract_urls(*values: str, source_url: str) -> list[str]:
    urls: list[str] = []
    for value in values:
        for candidate in HTTP_URL_RE.findall(html.unescape(value)):
            try:
                normalized = canonical_http_url(candidate)
            except ValueError:
                continue
            if normalized and normalized != source_url and normalized not in urls:
                urls.append(normalized)
    return urls[:20]


def parse_feed(payload: bytes, *, limit: int) -> list[FeedItem]:
    if UNSAFE_XML_RE.search(payload):
        raise RSSCollectorError("Feed contains a forbidden DTD/entity declaration")
    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError as exc:
        raise RSSCollectorError(f"Invalid RSS/Atom XML: {exc}") from exc

    root_name = local_name(root.tag)
    if root_name == "feed":
        entries = [child for child in root if local_name(child.tag) == "entry"]
        atom = True
    else:
        channel = next((child for child in root if local_name(child.tag) == "channel"), root)
        entries = [child for child in channel if local_name(child.tag) == "item"]
        atom = False

    parsed: list[FeedItem] = []
    for entry in entries[:limit]:
        title = strip_markup(child_text(entry, "title"))
        try:
            source_url = canonical_http_url(atom_link(entry) if atom else child_text(entry, "link"))
        except ValueError:
            continue
        if not title or not source_url:
            continue
        identifier = child_text(entry, "id", "guid") or source_url
        published_raw = child_text(entry, "published", "updated", "pubDate", "date")
        try:
            published_at = normalize_published(published_raw)
        except RSSCollectorError:
            continue
        summary_html = child_text(entry, "summary", "description")
        content_html = child_text(entry, "content", "encoded")
        summary = strip_markup(summary_html)
        content = strip_markup(content_html)
        if summary and content.startswith(summary):
            content = content[len(summary) :].strip()
        text_parts = [title]
        if summary and summary != title:
            text_parts.append(summary)
        if content and content not in summary and content != title:
            text_parts.append(content)
        original_text = "\n\n".join(text_parts)[:MAX_ITEM_TEXT]
        outbound_urls = extract_urls(
            summary_html,
            content_html,
            source_url=source_url,
        )
        raw_payload = {
            "title": title,
            "id": identifier,
            "link": source_url,
            "published": published_raw,
            "summary": summary,
            "content": content[:MAX_ITEM_TEXT],
        }
        parsed.append(
            FeedItem(
                external_id=identifier,
                source_url=source_url,
                original_text=original_text,
                published_at=published_at,
                outbound_urls=outbound_urls,
                raw_payload=raw_payload,
            )
        )
    return parsed


class RSSCollector:
    def __init__(
        self,
        settings: Settings,
        repository: Repository,
        *,
        fetcher: FeedFetcher = fetch_feed,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.fetcher = fetcher

    def collect(self, *, limit_per_source: int = 20) -> CollectionStats:
        registry = load_source_registry(self.settings.sources_file)
        sources = [source for source in registry.enabled_sources if source.kind == SourceKind.RSS]
        succeeded = seen = inserted = errors = 0
        for source in sources:
            try:
                payload = self.fetcher(source)
                items = parse_feed(payload, limit=min(limit_per_source, 100))
                for item in items:
                    serialized = json.dumps(
                        item.raw_payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    if self.repository.insert_source_item(
                        source_id=source.id,
                        external_id=item.external_id,
                        source_url=item.source_url,
                        original_text=item.original_text,
                        original_language=source.language,
                        published_at=item.published_at,
                        outbound_urls=item.outbound_urls,
                        media_metadata=None,
                        payload_hash=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
                        raw_payload=item.raw_payload,
                    ):
                        inserted += 1
                cursor = max((item.published_at for item in items), default=None)
                self.repository.mark_source_success(
                    source.id,
                    cursor=cursor,
                    success_at=datetime.now(UTC).isoformat(),
                )
                self.repository.record_usage(
                    "direct/rss",
                    "feed.fetch",
                    cost_usd=0,
                    metadata={"source_id": source.id, "items": len(items)},
                )
                seen += len(items)
                succeeded += 1
            except Exception as exc:
                self.repository.mark_source_error(source.id, str(exc))
                errors += 1
        return CollectionStats(
            sources_attempted=len(sources),
            sources_succeeded=succeeded,
            items_seen=seen,
            items_inserted=inserted,
            errors=errors,
            cost_usd=0,
        )
