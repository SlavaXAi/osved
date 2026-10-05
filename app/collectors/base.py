from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from app.config import SourceConfig


@dataclass(frozen=True)
class CollectionStats:
    sources_attempted: int = 0
    sources_succeeded: int = 0
    items_seen: int = 0
    items_inserted: int = 0
    errors: int = 0
    cost_usd: float = 0


@dataclass(frozen=True)
class SourceCapabilities:
    strategies: tuple[str, ...] = ("feed",)
    requires_auth: bool = False
    estimated_cost_usd: float = 0.0
    incremental: bool = False


@dataclass(frozen=True)
class SourceHealth:
    status: str = "ok"
    error: str | None = None


@dataclass(frozen=True)
class CollectionRequest:
    source: SourceConfig
    limit: int = 10
    cursor: str | None = None
    now: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class CollectedItem:
    external_id: str
    canonical_url: str
    text: str
    published_at: str
    title: str = ""
    author: str = ""
    language: str = ""
    content_type: str = "post"
    discovery_mode: str = "feed"
    outbound_urls: list[str] = field(default_factory=list)
    media_metadata: dict[str, Any] = field(default_factory=dict)
    engagement_metadata: dict[str, Any] = field(default_factory=dict)
    raw_payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CollectionBatch:
    items: list[CollectedItem] = field(default_factory=list)
    status: str = "ok"
    next_cursor: str | None = None
    cost_usd: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SourceAdapter(Protocol):
    def capabilities(self) -> SourceCapabilities: ...
    def healthcheck(self, source: SourceConfig) -> SourceHealth: ...
    def collect(self, request: CollectionRequest) -> CollectionBatch: ...
