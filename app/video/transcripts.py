from __future__ import annotations

import fcntl
import json
import math
import os
import re
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from app.collectors.platform_transport import MonidResponse, PlatformError, monid_fetch, unwrap_data

CAPTIONS_ENDPOINT = "/api/v1/youtube/web_v2/get_video_captions"
CAPTIONS_RESULT_ENDPOINT = "/api/v1/youtube/web_v2/get_video_captions_result"
CAPTION_CALL_COST = 0.012
MAX_TRANSCRIPT_CHARS = 500_000


def _time_value(value: Any, divisor: float = 1) -> float | None:
    """A missing/invalid timestamp is not evidence of 0:00."""
    try:
        parsed = float(value) / divisor
        return parsed if math.isfinite(parsed) and parsed >= 0 else None
    except (ValueError, TypeError, OverflowError):
        return None


def _language_family(code: str) -> str:
    return code.lower().removeprefix("a.").replace("_", "-").split("-", 1)[0]


@dataclass(frozen=True)
class TranscriptSegment:
    start_seconds: float
    text: str
    duration_seconds: float = 0


@dataclass(frozen=True)
class TranscriptResult:
    video_id: str
    status: str
    text: str = ""
    segments: list[TranscriptSegment] = field(default_factory=list)
    language: str = ""
    job_id: str | None = None
    cost_usd: float = 0
    cached: bool = False
    error: str | None = None
    retry_after_seconds: int = 0


def youtube_video_id(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value
    parsed = urlsplit(value)
    identifier = ""
    if parsed.scheme == "https" and parsed.hostname in {
        "www.youtube.com",
        "youtube.com",
        "m.youtube.com",
    }:
        identifier = (parse_qs(parsed.query).get("v") or [""])[0]
        if not identifier and re.match(r"/(?:shorts|embed|live)/", parsed.path):
            identifier = parsed.path.split("/")[2]
    elif parsed.scheme == "https" and parsed.hostname == "youtu.be":
        identifier = parsed.path.lstrip("/").split("/")[0]
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", identifier):
        raise ValueError("A valid YouTube video URL or 11-character ID is required")
    return identifier


def _segments(content: Any) -> tuple[str, list[TranscriptSegment]]:
    if isinstance(content, str):
        if len(content) > MAX_TRANSCRIPT_CHARS:
            raise PlatformError(
                "Transcript exceeds the configured size limit", status="schema_changed"
            )
        if content.lstrip().startswith("{"):
            try:
                return _segments(json.loads(content))
            except ValueError:
                pass
        # Parse SRT when the upstream ignores json3. Never invent timecodes for plain text.
        result: list[TranscriptSegment] = []
        for block in re.split(r"\n\s*\n", content.strip()):
            timing = re.search(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->[^\n]*\n(.+)", block, re.S)
            if timing:
                hour, minute, second, millis = map(int, timing.group(1, 2, 3, 4))
                text = " ".join(timing.group(5).split())
                if text:
                    result.append(
                        TranscriptSegment(hour * 3600 + minute * 60 + second + millis / 1000, text)
                    )
        return (
            ("\n".join(segment.text for segment in result), result)
            if result
            else (content.strip(), [])
        )
    if isinstance(content, dict) and isinstance(content.get("events"), list):
        segments: list[TranscriptSegment] = []
        texts: list[str] = []
        fully_timed = True
        size = 0
        for event in content["events"]:
            if not isinstance(event, dict):
                continue
            text = " ".join(
                "".join(
                    str(seg.get("utf8", ""))
                    for seg in event.get("segs", [])
                    if isinstance(seg, dict)
                ).split()
            )
            if not text:
                continue
            size += len(text)
            if size > MAX_TRANSCRIPT_CHARS:
                raise PlatformError(
                    "Transcript exceeds the configured size limit", status="schema_changed"
                )
            texts.append(text)
            start = _time_value(event.get("tStartMs"), 1000)
            duration = _time_value(event.get("dDurationMs"), 1000) or 0
            if start is None:
                fully_timed = False
            else:
                segments.append(TranscriptSegment(start, text, duration))
        return "\n".join(texts), segments if fully_timed else []
    if isinstance(content, list):
        segments = []
        texts = []
        fully_timed = True
        size = 0
        for item in content:
            if not isinstance(item, dict) or not item.get("text"):
                continue
            text = " ".join(str(item["text"]).split())
            size += len(text)
            if size > MAX_TRANSCRIPT_CHARS:
                raise PlatformError(
                    "Transcript exceeds the configured size limit", status="schema_changed"
                )
            texts.append(text)
            start = _time_value(item.get("start"))
            if start is None:
                fully_timed = False
            else:
                segments.append(
                    TranscriptSegment(start, text, _time_value(item.get("duration")) or 0)
                )
        return "\n".join(texts), segments if fully_timed else []
    raise PlatformError("Unrecognized caption content format", status="schema_changed")


class TranscriptService:
    """Explicit, bounded caption requests with durable positive/negative/job caching.

    A language-list lookup and one content request cost at most $0.024 at the
    inspected tariff. Async-job polling is free and resumed from the cache.
    on_cost is called per billed operation so the caller can persist its ledger.
    """

    def __init__(
        self,
        cache_dir: Path,
        *,
        fetcher: Callable[[str, dict[str, Any]], MonidResponse] | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        on_cost: Callable[[str, float, dict[str, Any]], None] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.fetcher = fetcher or (
            lambda endpoint, query: monid_fetch(
                endpoint,
                query,
                self.cache_dir / "runs",
                estimated_cost=0 if endpoint == CAPTIONS_RESULT_ENDPOINT else CAPTION_CALL_COST,
            )
        )
        self.clock = clock
        self.on_cost = on_cost

    def _store(self, result: TranscriptResult, language_code: str | None = None) -> None:
        payload = {"version": 1, "saved_at": self.clock().timestamp(), "result": asdict(result)}
        descriptor, temporary = tempfile.mkstemp(prefix=".captions-", dir=self.cache_dir)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
            suffix = f".{language_code}" if language_code else ""
            os.replace(temporary, self.cache_dir / f"{result.video_id}{suffix}.json")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _cached(
        self, video_id: str, language_code: str | None = None
    ) -> tuple[TranscriptResult | None, float]:
        suffix = f".{language_code}" if language_code else ""
        path = self.cache_dir / f"{video_id}{suffix}.json"
        if not path.is_file() or path.stat().st_size > 4_000_000:
            return None, 0
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            value = payload["result"]
            value["segments"] = [TranscriptSegment(**item) for item in value.get("segments", [])]
            result = TranscriptResult(**value)
            age = max(0.0, self.clock().timestamp() - float(payload["saved_at"]))
            ttl = {
                "ready": 30 * 86400,
                "unavailable": 86400,
                "processing": 86400,
                "rate_limited": 900,
                "failed": 300,
                "schema_changed": 3600,
                "partial": 3600,
                "disabled": 3600 if result.cost_usd > 0 else 0,
            }.get(result.status, 0)
            if result.video_id == video_id and age < ttl:
                return replace(result, cost_usd=0, cached=True), age
        except (ValueError, KeyError, TypeError, OSError):
            pass
        return None, 0

    def fetch(
        self,
        video: str,
        *,
        preferred_languages: tuple[str, ...] = ("en", "ru", "zh-Hans"),
        cost_limit_usd: float = 0.024,
        language_code: str | None = None,
    ) -> TranscriptResult:
        video_id = youtube_video_id(video)
        if not math.isfinite(cost_limit_usd) or cost_limit_usd < 0:
            return TranscriptResult(video_id, "disabled", error="Caption budget must be finite")
        if language_code is not None and not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,29}", language_code
        ):
            raise ValueError("An explicit caption language code must be a safe language identifier")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with (self.cache_dir / f"{video_id}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return self._fetch_locked(video_id, preferred_languages, cost_limit_usd, language_code)

    def _fetch_locked(
        self,
        video_id: str,
        languages: tuple[str, ...],
        limit: float,
        requested_language: str | None = None,
    ) -> TranscriptResult:
        cached, age = self._cached(video_id, requested_language)
        if requested_language and not cached:
            primary, primary_age = self._cached(video_id)
            if primary and primary.language == requested_language and primary.status == "ready":
                cached, age = primary, primary_age
        if cached and cached.status != "processing":
            return cached
        poll_delay = max(5, cached.retry_after_seconds) if cached else 5
        if cached and age < poll_delay:
            return replace(cached, retry_after_seconds=max(1, int(poll_delay - age)))
        spent = 0.0

        def store(result: TranscriptResult) -> None:
            # An explicit language probe has its own positive/negative/job cache.
            # Failed English probes must not erase already purchased Arabic text.
            self._store(result, requested_language)
            if requested_language and result.status == "ready":
                primary, _ = self._cached(video_id)
                if (
                    primary
                    and primary.status == "ready"
                    and primary.language != result.language
                    and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,29}", primary.language)
                ):
                    self._store(primary, primary.language)
                self._store(result)  # Successful explicit language becomes the preferred cache.

        def run(endpoint: str, query: dict[str, Any]) -> Any:
            nonlocal spent
            expected = 0 if endpoint == CAPTIONS_RESULT_ENDPOINT else CAPTION_CALL_COST
            if spent + expected > limit + 1e-9:
                raise PlatformError(
                    "Caption budget does not cover the next request", status="disabled"
                )
            try:
                response = self.fetcher(endpoint, query)
            except PlatformError as exc:
                spent += exc.cost_usd
                if self.on_cost and exc.cost_usd:
                    self.on_cost(
                        endpoint, exc.cost_usd, {"video_id": video_id, "status": exc.status}
                    )
                raise
            spent += response.cost_usd
            if self.on_cost:
                self.on_cost(
                    endpoint, response.cost_usd, {"video_id": video_id, "status": "completed"}
                )
            if spent > limit + 1e-9:
                raise PlatformError(
                    "Caption billing exceeded the request ceiling", status="disabled"
                )
            return unwrap_data(response.payload)

        language = cached.language if cached else (requested_language or "")
        try:
            if cached and cached.job_id:
                body = run(CAPTIONS_RESULT_ENDPOINT, {"job_id": cached.job_id, "format": "json3"})
            elif requested_language:
                body = run(
                    CAPTIONS_ENDPOINT,
                    {"video_id": video_id, "language_code": requested_language, "format": "json3"},
                )
            else:
                if limit < 2 * CAPTION_CALL_COST - 1e-9:
                    return TranscriptResult(
                        video_id,
                        "disabled",
                        error="Reserve $0.024 for caption discovery and retrieval",
                    )
                body = run(CAPTIONS_ENDPOINT, {"video_id": video_id})
                if not isinstance(body, dict):
                    raise PlatformError(
                        "Caption language response is not an object", status="schema_changed"
                    )
                if body.get("job_id"):
                    result = TranscriptResult(
                        video_id,
                        "processing",
                        job_id=str(body["job_id"]),
                        cost_usd=spent,
                        retry_after_seconds=5,
                    )
                    store(result)
                    return result
                available = body.get("captions")
                if not isinstance(available, list):
                    raise PlatformError(
                        "Caption response has no language list", status="schema_changed"
                    )
                codes = [
                    str(item.get("language_code") or "")
                    for item in available
                    if isinstance(item, dict)
                ]
                codes = [code for code in codes if code]
                if not codes:
                    result = TranscriptResult(
                        video_id,
                        "unavailable",
                        cost_usd=spent,
                        error="The video has no existing captions",
                    )
                    store(result)
                    return result
                language = next(
                    (
                        code
                        for preferred in languages
                        for code in codes
                        if _language_family(code) == _language_family(preferred)
                    ),
                    codes[0],
                )
                body = run(
                    CAPTIONS_ENDPOINT,
                    {"video_id": video_id, "language_code": language, "format": "json3"},
                )
            if isinstance(body, dict) and str(body.get("status", "")).lower() in {
                "processing",
                "queued",
                "active",
            }:
                job_id = str(body.get("job_id") or (cached.job_id if cached else "") or "")
                if not job_id:
                    raise PlatformError(
                        "Async captions response has no job ID", status="schema_changed"
                    )
                result = TranscriptResult(
                    video_id,
                    "processing",
                    language=language,
                    job_id=job_id,
                    cost_usd=spent,
                    retry_after_seconds=5,
                )
            else:
                returned_language = body.get("language_code") if isinstance(body, dict) else None
                if returned_language:
                    if language and _language_family(str(returned_language)) != _language_family(
                        language
                    ):
                        raise PlatformError(
                            "Provider returned captions in a different language than requested",
                            status="unavailable",
                        )
                    language = str(returned_language)
                content = body.get("content", body) if isinstance(body, dict) else body
                text, segments = _segments(content)
                if not text:
                    raise PlatformError("Caption content is empty", status="schema_changed")
                result = TranscriptResult(
                    video_id,
                    "ready",
                    text=text,
                    segments=segments,
                    language=language,
                    cost_usd=spent,
                )
            store(result)
            return result
        except (PlatformError, ValueError, TypeError, KeyError) as exc:
            status = exc.status if isinstance(exc, PlatformError) else "schema_changed"
            if cached and cached.job_id and status in {"rate_limited", "failed", "partial"}:
                # Keep the already purchased asynchronous job through transient poll failures.
                result = TranscriptResult(
                    video_id,
                    "processing",
                    language=cached.language,
                    job_id=cached.job_id,
                    cost_usd=spent,
                    error=str(exc)[:500],
                    retry_after_seconds=30,
                )
            else:
                result = TranscriptResult(video_id, status, cost_usd=spent, error=str(exc)[:500])
            store(result)
            return result
