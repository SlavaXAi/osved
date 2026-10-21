from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from math import isfinite
from typing import Any

from app.pipeline.editorial import event_organizations
from app.pipeline.signal_quality import assess_signal_quality

QUALITY_FLOOR = 3.0
SEMANTIC_EVENT_TYPES = {
    "model_release",
    "software_release",
    "company_event",
    "research_result",
    "public_statement",
    "other",
}


def _list_value(event: dict[str, Any], key: str) -> list[str]:
    value = event.get(key) or []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    return [str(item) for item in value] if isinstance(value, list) else []


def _metadata(event: dict[str, Any]) -> dict[str, Any]:
    value = event.get("score_features") or event.get("score_features_json") or {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = {}
    return {**(value if isinstance(value, dict) else {}), **event}


def _source_keys(event: dict[str, Any], kind: str) -> list[str]:
    metadata = _metadata(event)
    plural, singular = {
        "source": ("source_ids", "source_id"),
        "platform": ("platforms", "source_kind"),
        "discovery": ("discovery_modes", "discovery_mode"),
    }[kind]
    keys = _list_value(metadata, plural)
    value = metadata.get(singular) or (metadata.get("platform") if kind == "platform" else None)
    if value:
        keys.append(str(value))
    return list(dict.fromkeys(keys))


def _recent_weight(event: dict[str, Any], now: datetime) -> float:
    raw = event.get("delivered_at") or event.get("sent_at")
    if not raw:
        return 1.0
    try:
        stamp = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=UTC)
        hours = max(0.0, (now - stamp).total_seconds() / 3600)
        return 2 ** (-hours / 48.0)
    except (TypeError, ValueError):
        return 1.0


def is_exploration_candidate(event: dict[str, Any]) -> bool:
    return bool(
        _metadata(event).get("exploration") is True
        or set(_source_keys(event, "discovery")) & {"explore", "related", "for_you"}
    )


def _seed_counts(
    recent_events: list[dict[str, Any]],
    *,
    now: datetime | None = None,
) -> tuple[Counter[str], Counter[str], Counter[str], Counter[str]]:
    organizations: Counter[str] = Counter()
    topics: Counter[str] = Counter()
    regions: Counter[str] = Counter()
    event_types: Counter[str] = Counter()
    for event in recent_events:
        weight = _recent_weight(event, now) if now else 1
        for name in event_organizations(event):
            organizations[name] += weight
        for topic in _list_value(event, "topics_json"):
            topics[topic] += weight
        for region in _list_value(event, "regions_json"):
            if region != "global":
                regions[region] += weight
        event_type = str(event.get("event_type") or "")
        if event_type in SEMANTIC_EVENT_TYPES:
            event_types[event_type] += weight
    return organizations, topics, regions, event_types


def rank_event_candidates(
    candidates: list[dict[str, Any]],
    *,
    recent_events: list[dict[str, Any]] | None = None,
    limit: int | None = None,
    exploration_fraction: float = 0.15,
    now: datetime | None = None,
    personalization: Any = None,
) -> list[dict[str, Any]]:
    """Greedy soft-diversity reranker; never promotes a candidate below the quality floor."""
    remaining = []
    for candidate in candidates:
        event = dict(candidate)
        try:
            features = event.get("score_features") or event.get("score_features_json") or {}
            features = json.loads(features) if isinstance(features, str) else features
            features = dict(features) if isinstance(features, dict) else {}
        except (ValueError, TypeError):
            features = {}
        observations = event.get("signal_observations")
        if isinstance(observations, list):
            # Observed publisher spread and documentary support are diagnostics,
            # not a popularity bonus or an automatic fact-check approval.
            features["signal_quality"] = assess_signal_quality(
                row for row in observations if isinstance(row, dict)
            )
        if "corroborating_sources" in features:
            features["spread_bonus"] = features["corroborating_sources"]
        event["score_features"] = features
        if personalization is not None:
            features["daily_personalization"] = personalization.adjustment(event)
        remaining.append(event)
    target = len(remaining) if limit is None else max(0, limit)
    now = now or datetime.now(UTC)
    organization_counts, topic_counts, region_counts, type_counts = _seed_counts(
        recent_events or [], now=now if personalization is not None else None
    )
    source_counts: dict[str, Counter[str]] = {
        "source": Counter(),
        "platform": Counter(),
        "discovery": Counter(),
    }
    for event in recent_events or []:
        weight = _recent_weight(event, now)
        for kind, counts in source_counts.items():
            for value in _source_keys(event, kind):
                counts[value] += weight
    # A bounded opportunity for explicitly exploratory candidates; never force a
    # platform, geography or topic quota and never rescue weak-quality events.
    reserve = int(min(0.25, max(0.0, exploration_fraction)) * target) if target >= 4 else 0
    explored = 0
    selected: list[dict[str, Any]] = []

    while remaining and len(selected) < target:
        scored: list[tuple[float, float, int, dict[str, Any], dict[str, float]]] = []
        for index, event in enumerate(remaining):
            raw_score = event.get("score")
            try:
                base_score = QUALITY_FLOOR if raw_score is None else float(raw_score)
            except (TypeError, ValueError):
                continue
            if not isfinite(base_score) or base_score < QUALITY_FLOOR:
                continue
            if personalization is not None:
                preference = event["score_features"]["daily_personalization"]
                # Fixed feeds sum priority, freshness and several source bonuses;
                # search confidence uses a different scale. Calibrate this known
                # heuristic, without replacing the stored evidence/quality score.
                calibrated = (
                    min(8.5, base_score)
                    if event["score_features"].get("scoring_version") == "v7a_1"
                    else base_score
                )
                preference["original_score"] = base_score
                preference["calibrated_score"] = round(calibrated, 3)
                base_score = calibrated + preference["adjustment"]
            organizations = event_organizations(event)
            topics = _list_value(event, "topics_json")
            regions = [
                region for region in _list_value(event, "regions_json") if region != "global"
            ]
            event_type = str(event.get("event_type") or "")
            organization_penalty = min(
                6.0,
                max((organization_counts[name] * 1.5 for name in organizations), default=0.0),
            )
            if personalization is not None:
                pressure = max((organization_counts[n] for n in organizations), default=0.0)
                organization_penalty = min(
                    3.0, min(2.0, pressure) * 0.2 + max(0, pressure - 2) * 0.75
                )
            topic_penalty = min(
                1.0,
                max((topic_counts[topic] * 0.15 for topic in topics), default=0.0),
            )
            if personalization is not None:
                topic_penalty = min(0.5, topic_penalty)
            region_penalty = min(
                0.5,
                max((region_counts[region] * 0.1 for region in regions), default=0.0),
            )
            type_penalty = (
                min(0.75, type_counts[event_type] * 0.15)
                if event_type in SEMANTIC_EVENT_TYPES
                else 0.0
            )
            sensor_penalties = {
                kind: min(
                    cap,
                    max((counts[key] * weight for key in _source_keys(event, kind)), default=0.0),
                )
                for kind, counts, weight, cap in (
                    ("source", source_counts["source"], 0.25, 1.0),
                    ("platform", source_counts["platform"], 0.10, 0.5),
                    ("discovery", source_counts["discovery"], 0.08, 0.4),
                )
            }
            # An exceptional event can survive concentrated source/topic coverage.
            escape = 0.35 if base_score >= 9.0 else 1.0
            total_penalty = escape * (
                organization_penalty
                + topic_penalty
                + region_penalty
                + type_penalty
                + sum(sensor_penalties.values())
            )
            effective_score = round(base_score - total_penalty, 3)
            breakdown = {
                "organization_saturation": -organization_penalty,
                "topic_saturation": -topic_penalty,
                "region_saturation": -region_penalty,
                "event_type_saturation": -type_penalty,
                "source_saturation": -sensor_penalties["source"],
                "platform_saturation": -sensor_penalties["platform"],
                "discovery_saturation": -sensor_penalties["discovery"],
                "exceptional_penalty_factor": escape,
                "effective_score": effective_score,
            }
            scored.append((effective_score, base_score, -index, event, breakdown))

        if not scored:
            break
        _effective, _base, _position, winner, breakdown = max(
            scored,
            key=lambda item: (item[0], item[1], item[2]),
        )
        if reserve > explored and len(selected) >= target - reserve:
            alternatives = [
                item
                for item in scored
                if is_exploration_candidate(item[3])
                and item[1] >= max(QUALITY_FLOOR + 1.0, _base - 2.0)
                and item[0] >= _effective - 1.0
            ]
            if alternatives:
                _effective, _base, _position, winner, breakdown = max(
                    alternatives, key=lambda item: (item[0], item[1], item[2])
                )
                breakdown["exploration_slot"] = 1.0
                explored += 1
        winner["effective_score"] = breakdown["effective_score"]
        winner["saturation_features"] = breakdown
        winner["score_features"] = {**winner["score_features"], "saturation": breakdown}
        selected.append(winner)
        remaining.remove(winner)
        if personalization is not None and winner.get("fingerprint"):
            remaining = [e for e in remaining if e.get("fingerprint") != winner["fingerprint"]]
        organization_counts.update(event_organizations(winner))
        topic_counts.update(_list_value(winner, "topics_json"))
        region_counts.update(
            region for region in _list_value(winner, "regions_json") if region != "global"
        )
        event_type = str(winner.get("event_type") or "")
        if event_type in SEMANTIC_EVENT_TYPES:
            type_counts[event_type] += 1
        for kind, counts in source_counts.items():
            counts.update(_source_keys(winner, kind))

    return selected
