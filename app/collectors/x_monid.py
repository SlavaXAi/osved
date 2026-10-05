from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import uuid
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from app.collectors.base import CollectionStats
from app.collectors.telegram import HTTP_URL_RE, canonical_http_url
from app.config import Settings, SourceConfig, SourceKind, runtime_source_registry
from app.db.repository import Repository
from app.ops.budget import evaluate_budget

MONID_PROVIDER = "tikhub"
MONID_ENDPOINT = "/api/v1/twitter/web/fetch_user_post_tweet"
MONID_CALL_COST_USD = 0.0015
TWITTER_DATE_FORMAT = "%a %b %d %H:%M:%S %z %Y"
X_STATUS_RE = re.compile(r"https?://(?:www\.)?(?:x|twitter)\.com/[^/]+/status/\d+", re.I)
X_STATUS_ID_RE = re.compile(
    r"https?://(?:www\.)?(?:x|twitter)\.com/.+?/status/(\d+)(?:[/?#]|$)",
    re.I,
)
QUOTED_POST_LABEL = "Quoted post:"


class MonidXNotConfigured(RuntimeError):
    pass


class MonidXError(RuntimeError):
    pass


FetchResult = tuple[dict[str, Any], float]
TimelineFetcher = Callable[[SourceConfig, Path], FetchResult]
SourceStateGetter = Callable[[str], Mapping[str, Any] | None]
Clock = Callable[[], datetime]


@dataclass(frozen=True)
class NormalizedTweet:
    tweet_id: str
    conversation_id: str
    text: str
    published_at: datetime
    outbound_urls: tuple[str, ...]
    media_metadata: dict[str, Any] | None
    author_handle: str | None
    author_id: str | None
    reply_to_tweet_id: str | None
    reply_to_user_id: str | None
    reply_to_handle: str | None
    is_quote: bool
    raw_payload: dict[str, Any]


@dataclass(frozen=True)
class XSourceItem:
    external_id: str
    source_url: str
    original_text: str
    published_at: str
    outbound_urls: list[str]
    media_metadata: dict[str, Any] | None
    payload_hash: str
    raw_payload: dict[str, Any]


def monid_cli_path() -> str:
    discovered = shutil.which("monid")
    if discovered:
        return discovered
    user_local = Path.home() / ".local/bin/monid"
    if user_local.is_file():
        return str(user_local)
    raise MonidXNotConfigured("Monid CLI is not installed")


def fetch_user_timeline(source: SourceConfig, output_path: Path) -> FetchResult:
    """Fetch the newest top page for one X account through TikHub."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    query = {"screen_name": source.locator.lstrip("@")}
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
        raise MonidXError(detail[-2000:] or "Monid command failed")
    if not output_path.is_file():
        raise MonidXError("Monid completed without writing its output file")
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MonidXError("Monid returned a non-object payload")
    return payload, MONID_CALL_COST_USD


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def parse_twitter_datetime(value: Any) -> datetime:
    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 10_000_000_000:
            timestamp /= 1000
        return datetime.fromtimestamp(timestamp, tz=UTC)
    text = str(value or "").strip()
    if not text:
        raise ValueError("Tweet has no creation date")
    try:
        return datetime.strptime(text, TWITTER_DATE_FORMAT).astimezone(UTC)
    except ValueError:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return _as_utc(parsed)


def _legacy(tweet: Mapping[str, Any]) -> Mapping[str, Any]:
    value = tweet.get("legacy")
    return value if isinstance(value, Mapping) else {}


def _field(tweet: Mapping[str, Any], *names: str) -> Any:
    legacy = _legacy(tweet)
    for name in names:
        value = tweet.get(name)
        if value is not None:
            return value
        value = legacy.get(name)
        if value is not None:
            return value
    return None


def _unwrap_tweet(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    """Unwrap common GraphQL/TikHub envelopes without entering quoted tweets."""
    current: Mapping[str, Any] = raw
    for _ in range(7):
        if _field(current, "id_str", "tweet_id", "rest_id", "id") is not None:
            return current
        next_value: Any = None
        for key in ("tweet", "tweet_results", "itemContent", "content", "result"):
            candidate = current.get(key)
            if isinstance(candidate, Mapping):
                next_value = candidate
                break
        if not isinstance(next_value, Mapping):
            break
        current = next_value
    return current


def _string_id(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    return text or None


def tweet_id(tweet: Mapping[str, Any]) -> str | None:
    return _string_id(_field(tweet, "id_str", "tweet_id", "rest_id", "id"))


def _nested_mapping(value: Any, *keys: str) -> Mapping[str, Any] | None:
    current = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current if isinstance(current, Mapping) else None


def _author_details(
    tweet: Mapping[str, Any], fallback_user: Mapping[str, Any]
) -> tuple[str | None, str | None]:
    direct_user = tweet.get("user")
    candidates = [
        direct_user if isinstance(direct_user, Mapping) else None,
        _nested_mapping(tweet, "core", "user_results", "result"),
        fallback_user,
    ]
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        legacy = _legacy(candidate)
        handle = candidate.get("screen_name") or legacy.get("screen_name")
        author_id = (
            candidate.get("id_str")
            or candidate.get("rest_id")
            or legacy.get("id_str")
            or candidate.get("id")
        )
        if handle or author_id:
            return (
                str(handle).lstrip("@") if handle else None,
                _string_id(author_id),
            )
    handle = _field(tweet, "screen_name", "author_screen_name", "user_screen_name")
    author_id = _field(tweet, "user_id_str", "author_id", "user_id")
    return (str(handle).lstrip("@") if handle else None, _string_id(author_id))


def _is_pinned(raw: Mapping[str, Any], tweet: Mapping[str, Any]) -> bool:
    for value in (
        raw.get("is_pinned"),
        raw.get("pinned"),
        tweet.get("is_pinned"),
        tweet.get("pinned"),
        _legacy(tweet).get("is_pinned"),
    ):
        if value is True or (isinstance(value, str) and value.casefold() == "true"):
            return True
    entry_id = str(raw.get("entryId") or raw.get("entry_id") or "").casefold()
    return "pin" in entry_id


def _is_quote(tweet: Mapping[str, Any]) -> bool:
    if bool(_field(tweet, "is_quote_status", "is_quote", "quoted")):
        return True
    return any(
        _field(tweet, key) is not None
        for key in ("quoted_status", "quoted_tweet", "quoted_status_result")
    )


def _quoted_mapping(tweet: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for key in ("quoted", "quoted_status", "quoted_tweet", "quoted_status_result"):
        value = _field(tweet, key)
        if not isinstance(value, Mapping):
            continue
        unwrapped = _unwrap_tweet(value)
        if tweet_id(unwrapped):
            return unwrapped
    return None


def _is_retweet(tweet: Mapping[str, Any]) -> bool:
    if any(
        _field(tweet, key) is not None
        for key in ("retweeted_status", "retweeted_tweet", "retweeted_status_result")
    ):
        return True
    text = str(_field(tweet, "full_text", "text", "tweet_text") or "")
    return text.lstrip().startswith("RT @")


def _url_pairs(value: Any) -> list[tuple[str | None, str]]:
    pairs: list[tuple[str | None, str]] = []
    if isinstance(value, Mapping):
        expanded = (
            value.get("expanded_url")
            or value.get("expandedUrl")
            or value.get("unwound_url")
            or value.get("unwoundUrl")
        )
        short = value.get("url") or value.get("url_str")
        normalized = canonical_http_url(str(expanded or ""))
        if normalized:
            pairs.append((str(short) if short else None, normalized))
        for key, nested in value.items():
            if key.casefold() in {
                "entities",
                "extended_entities",
                "urls",
                "url",
                "media",
                "quoted_status",
                "quoted_tweet",
                "quoted_status_result",
            } and isinstance(nested, (Mapping, list, tuple)):
                pairs.extend(_url_pairs(nested))
    elif isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, str):
                normalized = canonical_http_url(item)
                if normalized:
                    pairs.append((None, normalized))
            else:
                pairs.extend(_url_pairs(item))
    return pairs


def _extract_urls(tweet: Mapping[str, Any], text: str) -> tuple[str, list[str]]:
    pairs: list[tuple[str | None, str]] = []
    for key in (
        "entities",
        "extended_entities",
        "urls",
        "url",
        "media",
        "quoted_status",
        "quoted_tweet",
        "quoted_status_result",
    ):
        value = _field(tweet, key)
        if value is not None:
            pairs.extend(_url_pairs(value))

    expanded_text = text
    for short, expanded in pairs:
        if short and short in expanded_text:
            expanded_text = expanded_text.replace(short, expanded)

    candidates = [expanded for _, expanded in pairs]
    candidates.extend(HTTP_URL_RE.findall(expanded_text))
    urls: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = canonical_http_url(str(candidate))
        if normalized and normalized not in seen:
            seen.add(normalized)
            urls.append(normalized)
    return expanded_text.strip(), urls


def _url_identity(url: str) -> str:
    match = X_STATUS_ID_RE.fullmatch(url)
    return f"x-status:{match.group(1)}" if match else url


def _merge_urls(*groups: list[str] | tuple[str, ...]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for url in group:
            identity = _url_identity(url)
            if identity not in seen:
                seen.add(identity)
                merged.append(url)
    return merged


def _quoted_context(
    tweet: Mapping[str, Any],
) -> tuple[str | None, list[str], dict[str, Any] | None]:
    quoted = _quoted_mapping(tweet)
    if quoted is None:
        return None, [], None
    quoted_text = str(_field(quoted, "full_text", "text", "tweet_text") or "").strip()
    expanded_text, quoted_urls = _extract_urls(quoted, quoted_text)
    quoted_id = tweet_id(quoted)
    if quoted_id:
        quoted_urls = _merge_urls(
            quoted_urls,
            [f"https://x.com/i/web/status/{quoted_id}"],
        )
    media_metadata = _media_metadata(quoted)
    if media_metadata:
        media_metadata = {
            **media_metadata,
            "items": [{**item, "context": "quoted"} for item in media_metadata.get("items", [])],
        }
    return (expanded_text or None, quoted_urls, media_metadata)


def _best_video_url(value: Mapping[str, Any]) -> str | None:
    variants = value.get("variants")
    if not isinstance(variants, list):
        return None
    candidates: list[tuple[int, str]] = []
    for variant in variants:
        if not isinstance(variant, Mapping):
            continue
        content_type = str(variant.get("content_type") or "").casefold()
        normalized = canonical_http_url(str(variant.get("url") or ""))
        if normalized and content_type == "video/mp4":
            bitrate = variant.get("bitrate")
            candidates.append((int(bitrate) if isinstance(bitrate, int) else 0, normalized))
    return max(candidates, default=(0, ""), key=lambda item: item[0])[1] or None


def _media_records(value: Any, *, kind_hint: str | None = None) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    if isinstance(value, Mapping):
        for container_kind in ("photo", "video", "animated_gif", "gif"):
            nested = value.get(container_kind)
            if isinstance(nested, (Mapping, list, tuple)):
                normalized_kind = "video" if container_kind == "animated_gif" else container_kind
                records.extend(_media_records(nested, kind_hint=normalized_kind))

        kind = value.get("type") or value.get("media_type") or kind_hint
        media_url = (
            value.get("media_url_https") or value.get("media_url") or value.get("preview_image_url")
        )
        if media_url:
            record: dict[str, str] = {}
            if kind:
                record["type"] = str(kind)
            normalized = canonical_http_url(str(media_url or ""))
            if normalized:
                record["url"] = normalized
            if str(kind).casefold() in {"video", "gif"}:
                content_url = _best_video_url(value)
                if content_url:
                    record["content_url"] = content_url
            if record:
                records.append(record)
        for key, nested in value.items():
            if key.casefold() in {"media", "extended_entities", "items"}:
                records.extend(_media_records(nested, kind_hint=kind_hint))
    elif isinstance(value, (list, tuple)):
        for item in value:
            records.extend(_media_records(item, kind_hint=kind_hint))
    return records


def _media_metadata(tweet: Mapping[str, Any]) -> dict[str, Any] | None:
    records: list[dict[str, str]] = []
    for key in ("media", "extended_entities", "entities"):
        value = _field(tweet, key)
        if value is not None:
            records.extend(_media_records(value))
    unique: list[dict[str, str]] = []
    seen: set[tuple[str | None, str | None, str | None]] = set()
    for record in records:
        fingerprint = (record.get("type"), record.get("url"), record.get("content_url"))
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(record)
    if not unique:
        return None
    return {"count": len(unique), "items": unique}


def _combine_media_metadata(
    *groups: dict[str, Any] | None,
) -> dict[str, Any] | None:
    items: list[dict[str, str]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for group in groups:
        if not group:
            continue
        for item in group.get("items", []):
            if not isinstance(item, dict):
                continue
            normalized = {str(key): str(value) for key, value in item.items()}
            fingerprint = tuple(sorted(normalized.items()))
            if fingerprint not in seen:
                seen.add(fingerprint)
                items.append(normalized)
    return {"count": len(items), "items": items} if items else None


def _has_quote_comment(text: str) -> bool:
    without_links = HTTP_URL_RE.sub("", text)
    return bool(without_links.strip(" \n\t—:-"))


def _reply_details(
    tweet: Mapping[str, Any],
) -> tuple[str | None, str | None, str | None]:
    reply_value = _field(tweet, "reply_to")
    reply_id: str | None = None
    reply_user_id: str | None = None
    reply_handle: str | None = None
    if isinstance(reply_value, Mapping):
        reply_id = _string_id(
            reply_value.get("tweet_id")
            or reply_value.get("status_id")
            or reply_value.get("id_str")
            or reply_value.get("id")
        )
        reply_user_id = _string_id(
            reply_value.get("user_id_str")
            or reply_value.get("user_id")
            or reply_value.get("author_id")
        )
        handle = reply_value.get("screen_name") or reply_value.get("handle")
        reply_handle = str(handle).lstrip("@") if handle else None
    else:
        reply_id = _string_id(reply_value)

    return (
        _string_id(_field(tweet, "in_reply_to_status_id_str", "in_reply_to_status_id")) or reply_id,
        _string_id(_field(tweet, "in_reply_to_user_id_str", "in_reply_to_user_id"))
        or reply_user_id,
        (str(_field(tweet, "in_reply_to_screen_name") or "").lstrip("@") or None) or reply_handle,
    )


def normalize_tweet(
    source: SourceConfig,
    raw: Mapping[str, Any],
    *,
    fallback_user: Mapping[str, Any],
) -> NormalizedTweet | None:
    tweet = _unwrap_tweet(raw)
    identifier = tweet_id(tweet)
    if not identifier or _is_pinned(raw, tweet):
        return None
    is_quote = _is_quote(tweet)
    if _is_retweet(tweet) and not is_quote:
        return None

    original_text = str(_field(tweet, "full_text", "text", "tweet_text") or "").strip()
    if is_quote and not _has_quote_comment(original_text):
        return None
    expanded_text, outbound_urls = _extract_urls(tweet, original_text)
    if not expanded_text:
        return None
    quoted_text, quoted_urls, quoted_media = _quoted_context(tweet)
    outbound_urls = _merge_urls(outbound_urls, quoted_urls)
    if quoted_text and quoted_text.strip() != expanded_text.strip():
        expanded_text = f"{expanded_text}\n\n{QUOTED_POST_LABEL}\n{quoted_text}"

    created_at = _field(tweet, "created_at", "createdAt", "date", "timestamp")
    published_at = parse_twitter_datetime(created_at)
    conversation_id = (
        _string_id(_field(tweet, "conversation_id_str", "conversation_id", "conversationId"))
        or identifier
    )
    author_handle, author_id = _author_details(tweet, fallback_user)
    reply_to_tweet_id, reply_to_user_id, reply_to_handle = _reply_details(tweet)
    return NormalizedTweet(
        tweet_id=identifier,
        conversation_id=conversation_id,
        text=expanded_text,
        published_at=published_at,
        outbound_urls=tuple(outbound_urls),
        media_metadata=_combine_media_metadata(_media_metadata(tweet), quoted_media),
        author_handle=author_handle,
        author_id=author_id,
        reply_to_tweet_id=reply_to_tweet_id,
        reply_to_user_id=reply_to_user_id,
        reply_to_handle=reply_to_handle,
        is_quote=is_quote,
        raw_payload=dict(raw),
    )


def _is_own_thread(source: SourceConfig, tweets: list[NormalizedTweet]) -> bool:
    if len(tweets) < 2:
        return False
    conversation_id = tweets[0].conversation_id
    if any(tweet.tweet_id == conversation_id for tweet in tweets):
        return True
    tweet_ids = {tweet.tweet_id for tweet in tweets}
    if any(tweet.reply_to_tweet_id in tweet_ids for tweet in tweets):
        return True
    handle = source.locator.lstrip("@").casefold()
    return any(
        tweet.reply_to_handle and tweet.reply_to_handle.casefold() == handle for tweet in tweets
    )


def _merge_media(tweets: list[NormalizedTweet]) -> dict[str, Any] | None:
    items: list[dict[str, str]] = []
    for tweet in tweets:
        if tweet.media_metadata:
            items.extend(tweet.media_metadata.get("items", []))
    if not items:
        return None
    return {"count": len(items), "items": items, "thread": len(tweets) > 1}


def _tweet_group_to_item(source: SourceConfig, tweets: list[NormalizedTweet]) -> XSourceItem:
    ordered = sorted(tweets, key=lambda tweet: (tweet.published_at, int(tweet.tweet_id)))
    root_id = ordered[0].conversation_id if len(ordered) > 1 else ordered[0].tweet_id
    source_url = f"https://x.com/{source.locator.lstrip('@')}/status/{root_id}"
    outbound_urls: list[str] = []
    seen_urls: set[str] = set()
    tweet_ids = {tweet.tweet_id for tweet in ordered}
    for tweet in ordered:
        for url in tweet.outbound_urls:
            if url == source_url or (
                X_STATUS_RE.fullmatch(url) and url.rsplit("/", 1)[-1] in tweet_ids
            ):
                continue
            if url not in seen_urls:
                seen_urls.add(url)
                outbound_urls.append(url)

    raw_payload = {
        "type": "x_thread" if len(ordered) > 1 else "x_post",
        "root_id": root_id,
        "tweet_ids": [tweet.tweet_id for tweet in ordered],
        "links": outbound_urls,
        "replies": [
            {
                "id": tweet.tweet_id,
                "reply_to_tweet_id": tweet.reply_to_tweet_id,
                "reply_to_user_id": tweet.reply_to_user_id,
                "reply_to_handle": tweet.reply_to_handle,
            }
            for tweet in ordered
            if tweet.reply_to_tweet_id
        ],
        "tweets": [tweet.raw_payload for tweet in ordered],
    }
    fingerprint_payload = {
        "source_id": source.id,
        "root_id": root_id,
        "tweets": [
            {
                "id": tweet.tweet_id,
                "text": tweet.text,
                "published_at": tweet.published_at.isoformat(),
                "urls": tweet.outbound_urls,
            }
            for tweet in ordered
        ],
    }
    serialized = json.dumps(
        fingerprint_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return XSourceItem(
        external_id=root_id,
        source_url=source_url,
        original_text="\n\n".join(tweet.text for tweet in ordered),
        published_at=max(tweet.published_at for tweet in ordered).isoformat(),
        outbound_urls=outbound_urls,
        media_metadata=_merge_media(ordered),
        payload_hash=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        raw_payload=raw_payload,
    )


def timeline_to_items(
    source: SourceConfig,
    timeline: list[Any],
    *,
    user: Mapping[str, Any] | None = None,
    now: datetime | None = None,
) -> tuple[list[XSourceItem], list[str]]:
    """Normalize one top page, returning source items and all observed tweet IDs."""
    clock = _as_utc(now or _now_utc())
    cutoff = clock - timedelta(days=source.production_max_age_days)
    fallback_user: Mapping[str, Any] = user or {}
    normalized: list[NormalizedTweet] = []
    observed_ids: list[str] = []
    for value in timeline:
        if not isinstance(value, Mapping):
            continue
        unwrapped = _unwrap_tweet(value)
        identifier = tweet_id(unwrapped)
        if identifier:
            observed_ids.append(identifier)
        tweet = normalize_tweet(source, value, fallback_user=fallback_user)
        if tweet and tweet.published_at >= cutoff:
            normalized.append(tweet)

    by_conversation: dict[str, list[NormalizedTweet]] = defaultdict(list)
    for tweet in normalized:
        by_conversation[tweet.conversation_id].append(tweet)

    groups: list[list[NormalizedTweet]] = []
    for tweets in by_conversation.values():
        if _is_own_thread(source, tweets):
            groups.append(tweets)
        else:
            groups.extend([[tweet] for tweet in tweets])
    groups.sort(key=lambda group: max(tweet.published_at for tweet in group), reverse=True)
    return ([_tweet_group_to_item(source, group) for group in groups], observed_ids)


def _timeline_list(payload: Mapping[str, Any]) -> list[Any]:
    timeline = payload.get("timeline")
    if isinstance(timeline, list):
        return timeline
    if isinstance(timeline, Mapping):
        for key in ("tweets", "items", "entries", "timeline"):
            value = timeline.get(key)
            if isinstance(value, list):
                return value
    raise MonidXError("Response has no timeline array")


def _newest_id(previous: str | None, observed: list[str]) -> str | None:
    values = [value for value in [previous, *observed] if value]
    if not values:
        return None
    numeric = [value for value in values if value.isdigit()]
    if len(numeric) == len(values):
        return max(numeric, key=int)
    return max(values)


def _parse_state_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return parse_twitter_datetime(value)
    except (TypeError, ValueError):
        return None


def source_is_due(
    source: SourceConfig,
    state: Mapping[str, Any] | None,
    *,
    now: datetime,
    force: bool = False,
) -> bool:
    if force:
        return True
    last_success_at = _parse_state_time((state or {}).get("last_success_at"))
    if last_success_at is None:
        return True
    return _as_utc(now) >= last_success_at + timedelta(minutes=source.poll_interval_minutes)


def _payload_tweet_ids(raw_payload: Any) -> set[str]:
    if isinstance(raw_payload, str):
        try:
            raw_payload = json.loads(raw_payload)
        except json.JSONDecodeError:
            return set()
    if not isinstance(raw_payload, Mapping):
        return set()
    tweet_ids = raw_payload.get("tweet_ids")
    if not isinstance(tweet_ids, list):
        return set()
    return {identifier for value in tweet_ids if (identifier := _string_id(value))}


def _store_x_source_item(
    repository: Repository,
    source: SourceConfig,
    item: XSourceItem,
) -> str:
    """Insert an X item, or expand its existing row when a thread gains replies.

    The row ID and processing status deliberately stay unchanged so existing event/evidence
    references remain valid. A partial top page can never shrink a previously stored thread.
    """
    with repository.connection() as connection:
        existing = connection.execute(
            """
            SELECT id, payload_hash, raw_payload_json
            FROM source_items
            WHERE source_id = ? AND external_id = ?
            """,
            (source.id, item.external_id),
        ).fetchone()
        if existing is None:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO source_items(
                    source_id, external_id, source_url, original_text,
                    original_language, published_at, outbound_urls_json,
                    media_metadata_json, payload_hash, raw_payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source.id,
                    item.external_id,
                    item.source_url,
                    item.original_text,
                    source.language,
                    item.published_at,
                    json.dumps(item.outbound_urls, ensure_ascii=False),
                    (
                        json.dumps(item.media_metadata, ensure_ascii=False)
                        if item.media_metadata
                        else None
                    ),
                    item.payload_hash,
                    json.dumps(item.raw_payload, ensure_ascii=False, default=str),
                ),
            )
            return "inserted" if cursor.rowcount == 1 else "unchanged"

        old_tweet_ids = _payload_tweet_ids(existing["raw_payload_json"])
        new_tweet_ids = _payload_tweet_ids(item.raw_payload)
        if not old_tweet_ids or not new_tweet_ids > old_tweet_ids:
            return "unchanged"

        cursor = connection.execute(
            """
            UPDATE source_items
            SET source_url = ?, original_text = ?, original_language = ?,
                published_at = ?, outbound_urls_json = ?, media_metadata_json = ?,
                payload_hash = ?, raw_payload_json = ?
            WHERE id = ?
            """,
            (
                item.source_url,
                item.original_text,
                source.language,
                item.published_at,
                json.dumps(item.outbound_urls, ensure_ascii=False),
                (
                    json.dumps(item.media_metadata, ensure_ascii=False)
                    if item.media_metadata
                    else None
                ),
                item.payload_hash,
                json.dumps(item.raw_payload, ensure_ascii=False, default=str),
                existing["id"],
            ),
        )
        return "updated" if cursor.rowcount == 1 else "unchanged"


class MonidXCollector:
    def __init__(
        self,
        settings: Settings,
        repository: Repository,
        *,
        fetcher: TimelineFetcher = fetch_user_timeline,
        state_getter: SourceStateGetter | None = None,
        clock: Clock = _now_utc,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.fetcher = fetcher
        self.state_getter = state_getter or repository.source_collection_state
        self.clock = clock

    def collect(
        self,
        *,
        force: bool = False,
        limit_per_source: int | None = None,
    ) -> CollectionStats:
        del limit_per_source  # TikHub's endpoint deliberately fetches one latest top page.
        registry = runtime_source_registry(self.settings)
        sources = [
            source
            for source in registry.enabled_sources
            if source.kind == SourceKind.X
            and source.adapter_id in {"auto", "monid_x_author"}
            and source.strategy in {"feed", "author_timeline"}
            and re.fullmatch(r"[A-Za-z0-9_]{1,15}", source.locator.lstrip("@"))
        ]
        attempted = succeeded = seen = inserted = errors = 0
        total_cost = 0.0

        for source in sources:
            now = _as_utc(self.clock())
            state = self.state_getter(source.id)
            if not source_is_due(source, state, now=now, force=force):
                continue

            attempted += 1
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

            timestamp = now.strftime("%Y%m%dT%H%M%S")
            output_path = (
                self.settings.data_dir
                / "monid-runs"
                / f"{source.id}-{timestamp}-{uuid.uuid4().hex[:8]}.json"
            )
            cost: float | None = None
            usage_metadata: dict[str, Any] = {"source_id": source.id, "status": "error"}
            try:
                payload, fetched_cost = self.fetcher(source, output_path)
                cost = max(0.0, float(fetched_cost))
                total_cost += cost
                timeline = _timeline_list(payload)
                user = payload.get("user")
                items, observed_ids = timeline_to_items(
                    source,
                    timeline,
                    user=user if isinstance(user, Mapping) else None,
                    now=now,
                )
                source_inserted = 0
                source_updated = 0
                for item in items:
                    outcome = _store_x_source_item(self.repository, source, item)
                    if outcome == "inserted":
                        source_inserted += 1
                    elif outcome == "updated":
                        source_updated += 1

                previous_cursor = _string_id((state or {}).get("last_cursor"))
                cursor = _newest_id(previous_cursor, observed_ids)
                self.repository.mark_source_success(
                    source.id,
                    cursor=cursor,
                    success_at=now.isoformat(),
                )
                succeeded += 1
                seen += len(items)
                inserted += source_inserted
                usage_metadata = {
                    "source_id": source.id,
                    "status": "success",
                    "timeline_items": len(timeline),
                    "normalized_items": len(items),
                    "inserted_items": source_inserted,
                    "updated_items": source_updated,
                    "next_cursor_present": bool(payload.get("next_cursor")),
                }
            except Exception as exc:
                errors += 1
                usage_metadata["error"] = str(exc)[:500]
                self.repository.mark_source_error(source.id, str(exc))
            finally:
                if cost is not None:
                    self.repository.record_usage(
                        "monid/tikhub",
                        "twitter.fetch_user_post_tweet",
                        cost_usd=cost,
                        metadata=usage_metadata,
                    )

        return CollectionStats(
            sources_attempted=attempted,
            sources_succeeded=succeeded,
            items_seen=seen,
            items_inserted=inserted,
            errors=errors,
            cost_usd=total_cost,
        )


# Keep both word orders available for CLI/import ergonomics.
XMonidCollector = MonidXCollector
