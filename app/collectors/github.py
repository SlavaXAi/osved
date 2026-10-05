from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.collectors.base import CollectionStats
from app.collectors.telegram import HTTP_URL_RE, canonical_http_url
from app.config import Settings, SourceConfig, SourceKind, load_source_registry
from app.db.repository import Repository

GITHUB_API_VERSION = "2026-03-10"
MAX_GITHUB_BYTES = 2_000_000
MAX_RELEASE_TEXT = 8_000


class GitHubCollectorError(RuntimeError):
    pass


ReleaseFetcher = Callable[[SourceConfig, int], list[dict[str, Any]]]


def fetch_releases(source: SourceConfig, limit: int) -> list[dict[str, Any]]:
    owner_repo = source.locator.strip().strip("/")
    if owner_repo.count("/") != 1:
        raise GitHubCollectorError(f"GitHub locator must be owner/repo: {source.locator}")
    url = f"https://api.github.com/repos/{owner_repo}/releases?per_page={min(limit, 100)}"
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
            "User-Agent": "PersonalTechnologyNewsfeed/0.1",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed GitHub endpoint
            payload = response.read(MAX_GITHUB_BYTES + 1)
    except HTTPError as exc:
        remaining = exc.headers.get("x-ratelimit-remaining")
        detail = f", remaining={remaining}" if remaining is not None else ""
        raise GitHubCollectorError(f"GitHub HTTP {exc.code} for {source.id}{detail}") from exc
    except URLError as exc:
        raise GitHubCollectorError(f"GitHub network error for {source.id}: {exc.reason}") from exc
    if len(payload) > MAX_GITHUB_BYTES:
        raise GitHubCollectorError(f"GitHub response exceeds {MAX_GITHUB_BYTES} bytes")
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GitHubCollectorError("GitHub returned invalid JSON") from exc
    if not isinstance(parsed, list):
        raise GitHubCollectorError("GitHub releases response is not an array")
    return [item for item in parsed if isinstance(item, dict)]


def release_text(source: SourceConfig, release: dict[str, Any]) -> str:
    tag = str(release.get("tag_name") or "untagged")
    name = str(release.get("name") or tag).strip()
    status = "Pre-release" if release.get("prerelease") else "Release"
    body = str(release.get("body") or "").strip()
    heading = f"{source.name}: {name} ({tag}) — {status}"
    return f"{heading}\n\n{body}"[:MAX_RELEASE_TEXT]


def release_urls(release: dict[str, Any], source_url: str) -> list[str]:
    urls: list[str] = []
    for candidate in HTTP_URL_RE.findall(str(release.get("body") or "")):
        try:
            normalized = canonical_http_url(candidate)
        except ValueError:
            continue
        if normalized and normalized != source_url and normalized not in urls:
            urls.append(normalized)
    return urls[:20]


class GitHubReleasesCollector:
    def __init__(
        self,
        settings: Settings,
        repository: Repository,
        *,
        fetcher: ReleaseFetcher = fetch_releases,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.fetcher = fetcher

    def collect(self, *, limit_per_source: int = 10) -> CollectionStats:
        registry = load_source_registry(self.settings.sources_file)
        sources = [
            source for source in registry.enabled_sources if source.kind == SourceKind.GITHUB
        ]
        succeeded = seen = inserted = errors = 0
        for source in sources:
            try:
                releases = self.fetcher(source, min(limit_per_source, 100))
                published_values: list[str] = []
                for release in releases:
                    if release.get("draft") or release.get("prerelease"):
                        continue
                    release_id = release.get("id")
                    source_url = canonical_http_url(str(release.get("html_url") or ""))
                    published_raw = str(
                        release.get("published_at") or release.get("created_at") or ""
                    )
                    if not release_id or not source_url or not published_raw:
                        continue
                    published = datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
                    if published.tzinfo is None:
                        published = published.replace(tzinfo=UTC)
                    published_at = published.astimezone(UTC).isoformat()
                    published_values.append(published_at)
                    seen += 1
                    raw_payload = {
                        "id": release_id,
                        "html_url": source_url,
                        "tag_name": release.get("tag_name"),
                        "name": release.get("name"),
                        "body": release.get("body"),
                        "draft": bool(release.get("draft")),
                        "prerelease": bool(release.get("prerelease")),
                        "published_at": published_raw,
                        "author": (
                            release.get("author", {}).get("login")
                            if isinstance(release.get("author"), dict)
                            else None
                        ),
                    }
                    serialized = json.dumps(
                        raw_payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    if self.repository.insert_source_item(
                        source_id=source.id,
                        external_id=str(release_id),
                        source_url=source_url,
                        original_text=release_text(source, release),
                        original_language=source.language,
                        published_at=published_at,
                        outbound_urls=release_urls(release, source_url),
                        media_metadata=None,
                        payload_hash=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
                        raw_payload=raw_payload,
                    ):
                        inserted += 1
                self.repository.mark_source_success(
                    source.id,
                    cursor=max(published_values, default=None),
                    success_at=datetime.now(UTC).isoformat(),
                )
                self.repository.record_usage(
                    "direct/github",
                    "releases.list",
                    cost_usd=0,
                    metadata={"source_id": source.id, "items": len(published_values)},
                )
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
