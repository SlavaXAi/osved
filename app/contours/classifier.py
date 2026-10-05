from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from app.contours.models import ContourRegistry

TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё_-]{3,}|[\u3400-\u9fff]+")


@dataclass(frozen=True)
class ContourAssignment:
    primary_contour: str | None
    secondary_contours: tuple[str, ...]
    scores: dict[str, float]
    explanation: str
    classifier_version: str = "contours_v1"
    cross_topic_tags: tuple[str, ...] = ()
    taxonomy_version: int = 1

    def as_metadata(self) -> dict[str, Any]:
        return {
            "classifier_version": self.classifier_version,
            "primary_contour": self.primary_contour,
            "secondary_contours": list(self.secondary_contours),
            "contour_scores": self.scores,
            "contour_reason": self.explanation,
            "cross_topic_tags": list(self.cross_topic_tags),
            "taxonomy_version": self.taxonomy_version,
        }


def _json_list(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    return [str(item).casefold() for item in value if str(item).strip()]


def _entity_values(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, dict):
        return []
    result: list[str] = []
    for values in value.values():
        if isinstance(values, list):
            result.extend(str(item) for item in values)
    return result


def event_classification_text(event: dict[str, Any]) -> str:
    parts = [
        str(event.get("headline") or ""),
        str(event.get("original_text") or event.get("summary") or ""),
        " ".join(_json_list(event.get("topics_json") or event.get("topics"))),
        " ".join(_json_list(event.get("regions_json") or event.get("regions"))),
        " ".join(_entity_values(event.get("entities_json") or event.get("entities"))),
    ]
    return " ".join(" ".join(parts).split()).casefold()


def classify_event(
    event: dict[str, Any],
    registry: ContourRegistry,
    *,
    source_lens_id: str | None = None,
    max_secondary: int = 2,
) -> ContourAssignment:
    """Explainably classify an event without spending tokens.

    A weak single word never forces an event into a contour. Search-lens identity,
    explicit topic metadata and repeated content signals are intentionally stronger.
    Unknown events remain unclassified instead of contaminating a random branch.
    """

    text = event_classification_text(event)
    topics = set(_json_list(event.get("topics_json") or event.get("topics")))
    regions = set(_json_list(event.get("regions_json") or event.get("regions")))
    tokens = set(TOKEN_RE.findall(text))
    cross_tags = set(regions - {"global"})
    cross_tags.update(
        topics
        & {
            "manufacturing",
            "robotics",
            "semiconductors",
            "construction",
            "energy",
            "supply_chains",
            "fintech",
            "compute_infrastructure",
            "cybersecurity",
        }
    )
    if re.search(r"\b(china|chinese|shenzhen)\b|китай|шэньчжэнь|中国|深圳", text):
        cross_tags.add("china")
    scored: list[tuple[float, str, list[str]]] = []

    for contour in registry.enabled_contours:
        score = 0.0
        reasons: list[str] = []
        required_region_hit = regions & {
            region.casefold() for region in contour.required_any_regions
        }
        required_keyword_hit = {
            keyword for keyword in contour.required_any_keywords if keyword.casefold() in text
        }
        if (
            (contour.required_any_regions or contour.required_any_keywords)
            and not required_region_hit
            and not required_keyword_hit
        ):
            scored.append((0.0, contour.id, ["required_context_missing"]))
            continue
        topic_hits = sorted(topics & {topic.casefold() for topic in contour.include_topics})
        if topic_hits:
            score += min(9.0, 3.0 * len(topic_hits))
            reasons.append(f"topics={','.join(topic_hits[:3])}")

        keyword_hits = [
            keyword
            for keyword in contour.include_keywords
            if keyword.casefold() in text
            or (
                " " not in keyword
                and len(keyword) >= 5
                and any(token.startswith(keyword.casefold()) for token in tokens)
            )
        ]
        if keyword_hits:
            score += min(7.5, 1.5 * len(keyword_hits))
            reasons.append(f"keywords={','.join(keyword_hits[:3])}")

        region_hits = sorted(regions & {region.casefold() for region in contour.regions})
        if region_hits and region_hits != ["global"]:
            score += min(1.5, 0.5 * len(region_hits))
            reasons.append(f"regions={','.join(region_hits[:2])}")

        # A search lens is only a weak prior. It must never classify empty or
        # off-topic content by itself.
        if source_lens_id and source_lens_id in contour.search_lens_ids and score > 0:
            score += 1.0
            reasons.append(f"lens={source_lens_id}")

        exclusions = [keyword for keyword in contour.exclude_keywords if keyword.casefold() in text]
        if exclusions:
            score -= min(6.0, 3.0 * len(exclusions))
            reasons.append(f"excluded={','.join(exclusions[:2])}")

        scored.append((round(score, 3), contour.id, reasons))

    scored.sort(key=lambda item: (-item[0], item[1]))
    best_score, best_id, best_reasons = scored[0] if scored else (0.0, "", [])
    if best_score < 2.5:
        return ContourAssignment(
            primary_contour=None,
            secondary_contours=(),
            scores={contour_id: score for score, contour_id, _ in scored if score > 0},
            explanation="insufficient_contour_signal",
            cross_topic_tags=tuple(sorted(cross_tags)),
            taxonomy_version=registry.version,
        )

    secondary_floor = max(2.5, best_score * 0.55)
    secondaries = tuple(
        contour_id for score, contour_id, _reasons in scored[1:] if score >= secondary_floor
    )[:max_secondary]
    explanation = ";".join(best_reasons) or "highest_deterministic_score"
    return ContourAssignment(
        primary_contour=best_id,
        secondary_contours=secondaries,
        scores={contour_id: score for score, contour_id, _ in scored if score > 0},
        explanation=explanation,
        cross_topic_tags=tuple(sorted(cross_tags)),
        taxonomy_version=registry.version,
    )
