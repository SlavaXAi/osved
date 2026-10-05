from __future__ import annotations

import json
from collections import Counter
from typing import Any

from app.contours.models import canonical_contour_id
from app.pipeline.ranking import (
    QUALITY_FLOOR,
    is_exploration_candidate,
    rank_event_candidates,
)


def _score_features(event: dict[str, Any]) -> dict[str, Any]:
    value = event.get("score_features_json") or event.get("score_features") or {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def event_primary_contour(event: dict[str, Any]) -> str | None:
    direct = event.get("primary_contour")
    if direct:
        return canonical_contour_id(str(direct), event)
    value = _score_features(event).get("primary_contour")
    return canonical_contour_id(str(value), event) if value else None


def mix_contour_portfolio(
    candidates: list[dict[str, Any]],
    *,
    recent_events: list[dict[str, Any]] | None = None,
    limit: int | None = None,
    contour_penalty: float = 0.8,
    exceptional_score: float = 9.0,
    exploration_fraction: float = 0.15,
) -> list[dict[str, Any]]:
    """Apply a soft contour saturation penalty after the existing quality reranker.

    There are no quotas: an exceptional event may still win and an empty contour does
    not receive a weak filler. The function copies inputs and keeps the quality floor.
    """

    target = len(candidates) if limit is None else max(0, limit)
    remaining = rank_event_candidates(
        candidates,
        recent_events=recent_events,
        limit=None,
        exploration_fraction=0,
    )
    recent_counts: Counter[str] = Counter(
        contour
        for event in recent_events or []
        if (contour := event_primary_contour(event)) is not None
    )
    batch_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    reserve = int(min(0.25, max(0.0, exploration_fraction)) * target) if target >= 4 else 0
    explored = 0

    while remaining and len(selected) < target:
        choices: list[tuple[float, float, int, dict[str, Any], float]] = []
        for index, candidate in enumerate(remaining):
            raw_score = float(candidate.get("effective_score", candidate.get("score") or 0))
            if raw_score < QUALITY_FLOOR:
                continue
            contour = event_primary_contour(candidate)
            penalty = 0.0
            if contour:
                penalty = min(
                    3.2,
                    recent_counts[contour] * contour_penalty * 0.35
                    + batch_counts[contour] * contour_penalty,
                )
            if raw_score >= exceptional_score:
                penalty *= 0.35
            effective = round(raw_score - penalty, 3)
            choices.append((effective, raw_score, -index, candidate, penalty))

        if not choices:
            break
        effective, _raw, _position, winner, penalty = max(
            choices,
            key=lambda item: (item[0], item[1], item[2]),
        )
        exploration_slot = False
        if reserve > explored and len(selected) >= target - reserve:
            alternatives = [
                item
                for item in choices
                if is_exploration_candidate(item[3])
                and item[1] >= max(QUALITY_FLOOR + 1.0, _raw - 2.0)
                and item[0] >= effective - 1.0
            ]
            if alternatives:
                effective, _raw, _position, winner, penalty = max(
                    alternatives, key=lambda item: (item[0], item[1], item[2])
                )
                exploration_slot = True
                explored += 1
        result = dict(winner)
        result["contour_effective_score"] = effective
        result["contour_saturation_penalty"] = round(-penalty, 3)
        if exploration_slot:
            result["exploration_slot"] = True
        selected.append(result)
        remaining.remove(winner)
        contour = event_primary_contour(winner)
        if contour:
            batch_counts[contour] += 1

    return selected
