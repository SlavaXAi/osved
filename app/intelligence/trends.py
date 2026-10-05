from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

from app.contours.portfolio import event_primary_contour
from app.intelligence.models import TrendSignal

# Event rows currently contain both event-specific topics and broad defaults
# inherited from their collector/search lens. A trend must therefore be derived
# from concrete event text, not from those untraceable defaults. These patterns
# are a small, closed taxonomy; an unmatched event falls back to its contour.
EVENT_TOPIC_PATTERNS: dict[str, re.Pattern[str]] = {
    "ai_agents": re.compile(r"\b(?:agent\w*|агент\w*)\b", re.I),
    "ai_models": re.compile(r"\b(?:LLM|модел\w*|нейросет\w*|benchmark\w*)\b", re.I),
    "robotics": re.compile(r"\b(?:robot\w*|робот\w*|humanoid\w*)\b", re.I),
    "manufacturing": re.compile(
        r"\b(?:manufactur\w*|factor(?:y|ies)|фабрик\w*|производств\w*|supply chain)\b",
        re.I,
    ),
    "semiconductors": re.compile(
        r"\b(?:chip\w*|semiconductor\w*|GPU\w*|чип\w*|полупроводник\w*)\b",
        re.I,
    ),
    "compute_infrastructure": re.compile(
        r"\b(?:data[ -]?cent(?:er|re)s?|compute cluster\w*|cloud infrastructure|"
        r"дата-центр\w*|вычислительн\w+\s+(?:кластер\w*|инфраструктур\w*))\b",
        re.I,
    ),
    "startup_funding": re.compile(
        r"\b(?:funding|seed round|series [a-z]|венчур\w*|раунд\w*|инвестиц\w*)\b",
        re.I,
    ),
    "startups": re.compile(r"\b(?:startup\w*|стартап\w*|YC\b|a16z)\b", re.I),
    "construction": re.compile(r"\b(?:construction\w*|строител\w*)\b", re.I),
    "space": re.compile(
        r"\b(?:spacecraft\w*|satellite\w*|launch vehicle\w*|косм\w*|спутник\w*)\b",
        re.I,
    ),
    "energy": re.compile(
        r"\b(?:power plant\w*|power grid\w*|energ(?:y|ies)|электроэнерг\w*|энергетик\w*)\b",
        re.I,
    ),
    "logistics": re.compile(r"\b(?:logistic\w*|warehouse\w*|логист\w*|склад\w*)\b", re.I),
    "defense_technology": re.compile(
        r"\b(?:defen[cs]e|military|drone\w*|оборон\w*|военн\w*|дрон\w*)\b",
        re.I,
    ),
    "geopolitics": re.compile(
        r"\b(?:sanction\w*|alliance\w*|геополит\w*|санкц\w*|союз\w*)\b",
        re.I,
    ),
    "macroeconomics": re.compile(
        r"\b(?:inflation|interest rate|tariff\w*|инфляц\w*|ставк\w*|тариф\w*)\b",
        re.I,
    ),
    "education": re.compile(
        r"\b(?:education\w*|universit\w*|literacy|образован\w*|университет\w*)\b",
        re.I,
    ),
    "attention": re.compile(
        r"\b(?:attention economy|digital detox|doomscroll\w*|экономик\w+\s+внимани\w*|"
        r"цифров\w+\s+детокс\w*)\b",
        re.I,
    ),
}


def _event_time(event: dict[str, Any]) -> datetime | None:
    value = (
        event.get("event_at")
        or event.get("latest_source_at")
        or event.get("delivered_at")
        or event.get("created_at")
    )
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _event_topics(event: dict[str, Any]) -> list[str]:
    text = " ".join(
        str(event.get(key) or "")
        for key in ("headline", "original_text", "summary", "material_update")
    )
    return [topic for topic, pattern in EVENT_TOPIC_PATTERNS.items() if pattern.search(text)][:3]


def _event_source_ids(event: dict[str, Any]) -> set[str]:
    values = event.get("source_ids")
    if isinstance(values, str):
        with_json: Any = []
        try:
            with_json = json.loads(values)
        except json.JSONDecodeError:
            with_json = [values]
        values = with_json
    if isinstance(values, list):
        return {str(value) for value in values if str(value)}
    value = event.get("source_id") or event.get("source_name")
    return {str(value)} if value else set()


def _event_quality(event: dict[str, Any]) -> float:
    try:
        return max(0.0, float(event.get("score") or 0))
    except (TypeError, ValueError):
        return 0.0


def detect_trends(
    events: list[dict[str, Any]],
    *,
    now: datetime | None = None,
    window_days: int = 7,
    minimum_current_events: int = 2,
) -> list[TrendSignal]:
    """Compute transparent accumulation signals; no LLM is allowed to invent a trend."""

    end = (now or datetime.now(UTC)).astimezone(UTC)
    current_start = end - timedelta(days=window_days)
    previous_start = current_start - timedelta(days=window_days)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    seen_event_ids: set[str] = set()
    for event in events:
        event_id = event.get("id")
        if event_id is not None:
            event_key = str(event_id)
            if event_key in seen_event_ids:
                continue
            seen_event_ids.add(event_key)
        contour = event_primary_contour(event)
        event_time = _event_time(event)
        if (
            not contour
            or event_time is None
            or _event_quality(event) < 3.0
            or not (previous_start <= event_time <= end)
        ):
            continue
        topics = _event_topics(event) or [contour]
        for topic in topics[:3]:
            enriched = dict(event)
            enriched["_trend_time"] = event_time
            groups[(contour, topic)].append(enriched)

    signals: list[TrendSignal] = []
    for (contour, topic), rows in groups.items():
        current = [row for row in rows if row["_trend_time"] >= current_start]
        previous = [row for row in rows if row["_trend_time"] < current_start]
        # A single excellent event is still an event, not an accumulating trend.
        # Important one-offs stay in the ordinary news flow; this module only
        # promotes a movement after at least two distinct events exist.
        if len(current) < minimum_current_events:
            continue
        source_ids = set().union(*(_event_source_ids(row) for row in current))
        average_quality = (
            sum(_event_quality(row) for row in current) / len(current) if current else 0.0
        )
        acceleration = (len(current) - len(previous)) / max(1, len(previous))
        volume_feature = math.log1p(len(current)) * 1.6
        acceleration_feature = max(-1.0, min(3.0, acceleration))
        diversity_feature = min(2.0, len(source_ids) * 0.4)
        quality_feature = min(2.0, average_quality / 5.0)
        trend_score = round(
            volume_feature + acceleration_feature + diversity_feature + quality_feature,
            3,
        )
        signals.append(
            TrendSignal(
                key=f"{contour}:{topic}",
                label=topic.replace("_", " "),
                primary_contour=contour,
                topic=topic,
                current_event_ids=[int(row["id"]) for row in current if row.get("id")],
                previous_event_ids=[int(row["id"]) for row in previous if row.get("id")],
                current_count=len(current),
                previous_count=len(previous),
                distinct_sources=len(source_ids),
                average_quality=round(average_quality, 3),
                trend_score=trend_score,
                score_features={
                    "volume": round(volume_feature, 3),
                    "acceleration": round(acceleration_feature, 3),
                    "source_diversity": round(diversity_feature, 3),
                    "average_quality": round(quality_feature, 3),
                },
            )
        )

    return sorted(
        signals,
        key=lambda signal: (-signal.trend_score, signal.primary_contour, signal.topic),
    )
