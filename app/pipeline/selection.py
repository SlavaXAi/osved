from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urlsplit

from app.config import Settings, load_interest_profile, runtime_source_registry
from app.db.repository import Repository
from app.pipeline.editorial import extract_editorial_metadata, newsworthiness_features
from app.video.shorts import ShortsClassifier, metadata_type

LOW_VALUE_HOSTS = {
    "t.me",
    "telegram.me",
    "telegram.org",
    "boosty.to",
}

# High precision only: do not reject reporting ABOUT advertising or a startup
# merely because it sells something. Paid placements and direct signup offers
# are not events for this news feed.
PROMOTIONAL_POST = re.compile(
    r"(?:\berid\s*[:=]|#реклама\b|\bна\s+правах\s+рекламы\b|"
    r"\bпо\s+промокоду\b|\bпромокод\s*[:—–-]|"
    r"бесплатн\w*\s+бутк[эе]мп|\bзаписывайтесь\s+на\s+(?:курс|вебинар))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PreparationStats:
    items_seen: int = 0
    events_created: int = 0
    items_attached_to_existing: int = 0
    items_rejected: int = 0


@dataclass(frozen=True)
class ReclassificationStats:
    events_seen: int = 0
    events_updated: int = 0
    events_rejected: int = 0


def preferred_primary_url(
    source_url: str,
    outbound_urls: list[str],
    *,
    source_kind: str = "telegram",
    source_role: str = "radar",
) -> str:
    if source_kind in {"rss", "github", "web"} or source_role == "primary":
        return source_url
    for url in outbound_urls:
        host = (urlsplit(url).hostname or "").lower()
        if host and host not in LOW_VALUE_HOSTS:
            return url
    return outbound_urls[0] if outbound_urls else source_url


def candidate_score(
    item: dict,
    *,
    topics: list[str] | None = None,
    regions: list[str] | None = None,
    interest_topics: set[str] | None = None,
    interest_regions: set[str] | None = None,
) -> tuple[float, dict[str, float | str]]:
    published = datetime.fromisoformat(str(item["published_at"]).replace("Z", "+00:00"))
    if published.tzinfo is None:
        published = published.replace(tzinfo=UTC)
    age_hours = max(0.0, (datetime.now(UTC) - published.astimezone(UTC)).total_seconds() / 3600)
    freshness = max(0.0, 3.0 - age_hours / 48)
    priority = {"p0": 3.0, "p1": 2.0, "p2": 1.0}.get(item["source_priority"], 1.0)
    outbound_urls = json.loads(item["outbound_urls_json"] or "[]")
    external_evidence = 2.0 if item["source_role"] == "primary" else 0.0
    if outbound_urls and external_evidence == 0:
        external_evidence = 1.5
    substance = 1.0 if len(item["original_text"] or "") >= 240 else 0.5
    author_signal = 0.5 if item["source_role"] == "author_radar" else 0.0
    topic_match = bool(set(topics or []) & (interest_topics or set()))
    region_match = bool(set(regions or []) & (interest_regions or set()))
    interest_match = 0.75 if topic_match else 0.0
    if region_match:
        interest_match += 0.25
    features: dict[str, float | str] = {
        "scoring_version": "v7a_1",
        "source_priority": priority,
        "freshness": round(freshness, 3),
        "external_evidence": external_evidence,
        "substance": substance,
        "author_signal": author_signal,
        "interest_match": interest_match,
        **newsworthiness_features(
            str(item.get("original_text") or ""),
            source_role=str(item["source_role"]),
        ),
    }
    score = sum(value for value in features.values() if isinstance(value, float | int))
    return round(float(score), 3), features


def prepare_candidates(
    settings: Settings,
    repository: Repository,
    *,
    limit: int = 200,
) -> PreparationStats:
    rows = repository.unprocessed_source_items(limit)
    registry = runtime_source_registry(settings)
    interests = load_interest_profile(settings.interests_file)
    interest_topics = set(interests.topics)
    interest_regions = set(interests.regions_of_interest)
    source_configs = {source.id: source for source in registry.sources}
    created = attached = rejected = 0
    now = datetime.now(UTC)

    for item in rows:
        item_id = int(item["id"])
        if item["source_kind"] == "youtube":
            metadata = json.loads(item.get("media_metadata_json") or "{}")
            video_id = metadata.get("video_id") or item.get("external_id") or ""
            kind = metadata_type(metadata)
            if kind is None:
                kind = ShortsClassifier(repository).cached(video_id)
            if kind is not False:
                repository.mark_source_item_status(
                    item_id, "filtered_short" if kind else "video_pending_type"
                )
                continue
        if not item["candidate_enabled"] or item["source_role"] == "own_history":
            repository.mark_source_item_status(item_id, "history_only")
            rejected += 1
            continue
        text = str(item["original_text"] or "").strip()
        if len(text) < 80:
            repository.mark_source_item_status(item_id, "rejected", "insufficient_text")
            rejected += 1
            continue
        if PROMOTIONAL_POST.search(text):
            repository.mark_source_item_status(item_id, "rejected", "promotional_placement")
            rejected += 1
            continue
        source_config = source_configs.get(str(item["source_id"]))
        normalized_text = text.casefold()
        if (
            source_config
            and source_config.include_keywords
            and not any(
                keyword.casefold() in normalized_text for keyword in source_config.include_keywords
            )
        ):
            repository.mark_source_item_status(
                item_id,
                "rejected",
                "source_include_keywords_missed",
            )
            rejected += 1
            continue
        if source_config and any(
            keyword.casefold() in normalized_text for keyword in source_config.exclude_keywords
        ):
            repository.mark_source_item_status(
                item_id,
                "rejected",
                "source_exclude_keyword_matched",
            )
            rejected += 1
            continue
        published = datetime.fromisoformat(str(item["published_at"]).replace("Z", "+00:00"))
        if published.tzinfo is None:
            published = published.replace(tzinfo=UTC)
        max_age_days = (
            int(item["calibration_lookback_days"])
            if settings.app_env != "production"
            else int(item["production_max_age_days"])
        )
        if (now - published.astimezone(UTC)).total_seconds() > max_age_days * 86400:
            repository.mark_source_item_status(item_id, "rejected", "outside_age_window")
            rejected += 1
            continue
        if (
            source_config
            and repository.source_event_count_on_date(
                source_config.id,
                published.astimezone(UTC).date().isoformat(),
            )
            >= source_config.daily_candidate_cap
        ):
            repository.mark_source_item_status(item_id, "rejected", "source_daily_cap_reached")
            rejected += 1
            continue

        source_topics = json.loads(item["source_topics_json"] or "[]")
        metadata = extract_editorial_metadata(
            text,
            source_topics=source_topics,
            source_region=str(item.get("source_region") or "global"),
        )
        editorial_features = newsworthiness_features(
            text,
            source_role=str(item["source_role"]),
        )
        if editorial_features["routine_penalty"] < 0 or editorial_features["digest_penalty"] < 0:
            repository.mark_source_item_status(
                item_id,
                "rejected",
                (
                    "low_newsworthiness_digest"
                    if editorial_features["digest_penalty"] < 0
                    else "low_newsworthiness_routine"
                ),
            )
            rejected += 1
            continue

        outbound_urls = json.loads(item["outbound_urls_json"] or "[]")
        primary_url = preferred_primary_url(
            item["source_url"],
            outbound_urls,
            source_kind=item["source_kind"],
            source_role=item["source_role"],
        )
        fingerprint = hashlib.sha256(primary_url.encode("utf-8")).hexdigest()
        score, features = candidate_score(
            item,
            topics=metadata.topics,
            regions=metadata.regions,
            interest_topics=interest_topics,
            interest_regions=interest_regions,
        )
        first_line = next((line.strip() for line in text.splitlines() if line.strip()), text)
        event_type = {
            "rss": "official_feed_entry",
            "github": "github_release",
            "x": "x_signal",
            "web": "web_signal",
        }.get(
            item["source_kind"],
            "telegram_signal" if outbound_urls else "channel_analysis",
        )
        _, was_created = repository.attach_item_to_event(
            source_item_id=item_id,
            event_type=event_type,
            headline=first_line[:180],
            event_at=published.astimezone(UTC).isoformat(),
            primary_url=primary_url,
            fingerprint=fingerprint,
            score=score,
            score_features=features,
            decision_reason=f"accepted:primary={primary_url}",
            entities=metadata.entities,
            topics=metadata.topics,
            regions=metadata.regions,
        )
        if was_created:
            created += 1
        else:
            attached += 1

    return PreparationStats(
        items_seen=len(rows),
        events_created=created,
        items_attached_to_existing=attached,
        items_rejected=rejected,
    )


def reclassify_pending_candidates(
    settings: Settings,
    repository: Repository,
    *,
    limit: int = 500,
) -> ReclassificationStats:
    rows = repository.pending_event_source_items(limit)
    interests = load_interest_profile(settings.interests_file)
    interest_topics = set(interests.topics)
    interest_regions = set(interests.regions_of_interest)
    updated = rejected = 0

    for item in rows:
        text = str(item.get("original_text") or "").strip()
        source_topics = json.loads(item.get("source_topics_json") or "[]")
        metadata = extract_editorial_metadata(
            text,
            source_topics=source_topics,
            source_region=str(item.get("source_region") or "global"),
        )
        score, features = candidate_score(
            item,
            topics=metadata.topics,
            regions=metadata.regions,
            interest_topics=interest_topics,
            interest_regions=interest_regions,
        )
        low_value = features["routine_penalty"] < 0 or features["digest_penalty"] < 0
        repository.update_pending_event_editorial_score(
            int(item["event_id"]),
            base_score=score,
            score_features=features,
            entities=metadata.entities,
            topics=metadata.topics,
            regions=metadata.regions,
            status="rejected_low_value" if low_value else "candidate",
            decision_reason=(
                "v7a_low_newsworthiness_reclassification"
                if low_value
                else "v7a_editorial_reclassification"
            ),
        )
        if low_value:
            rejected += 1
        else:
            updated += 1

    return ReclassificationStats(
        events_seen=len(rows),
        events_updated=updated,
        events_rejected=rejected,
    )
