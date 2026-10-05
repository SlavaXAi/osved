"""Read-only, evidence-linked proposals for explicit OSVED calibration.

No function here edits config, prompts, reactions or preference weights. Comments
are user-provided evidence, not executable instructions for an automatic agent.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

from app.contours.models import canonical_contour_id
from app.db.repository import Repository
from app.radar.feedback import REACTION_WEIGHTS

REPORT_VERSION = "preference_proposals_v1"
CARD_QUALITY_RE = re.compile(
    r"структур|абзац|заголов|оформл|слишком корот|слишком длин|много цифр|"
    r"больше факт|больше сравн|мало факт|граммат|русск\w* язык|ссылк|"
    r"format|headline|too short|too long|grammar|wording|more context",
    re.I,
)


def _object(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return {}
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return []
    return [str(item) for item in value] if isinstance(value, list) else []


def _patch(
    *,
    scope: str,
    target: str,
    direction: str,
    examples: list[dict[str, Any]],
    rationale: str,
    confidence: float,
) -> dict[str, Any]:
    ids = sorted({int(example["feedback_id"]) for example in examples})
    identity = f"{scope}:{target}:{direction}:{','.join(map(str, ids))}"
    return {
        "id": "pref_" + hashlib.sha256(identity.encode()).hexdigest()[:16],
        "status": "proposed",
        "scope": scope,
        "target": target,
        "direction": direction,
        "confidence": round(confidence, 2),
        "confidence_kind": "heuristic_not_probability",
        "rationale": rationale,
        "feedback_ids": ids,
        "event_ids": sorted({int(example["event_id"]) for example in examples}),
        "source_ids": sorted(
            {example["source_id"] for example in examples if example["source_id"]}
        ),
        "comments": [
            {"feedback_id": example["feedback_id"], "text": example["comment"]}
            for example in examples
            if example["comment"]
        ],
        "requires_review": True,
        "automatic_effect": "none",
    }


def build_preference_report(
    repository: Repository,
    *,
    days: int = 14,
    now: datetime | None = None,
    chat_id: int | None = None,
    limit: int = 200,
    max_patches: int = 20,
) -> dict[str, Any]:
    """Build bounded proposals from the latest reaction per delivery.

    A changed reaction replaces its predecessor for analysis. A newer reaction
    without a comment never inherits an older rationale. Detail reactions are
    analysed separately and never change the original news reaction.
    """
    end = now or datetime.now(UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    start = end - timedelta(days=max(1, min(90, days)))
    with repository.connection() as connection:
        rows = connection.execute(
            """
            WITH latest AS (
                SELECT f.*, d.chat_id,
                    ROW_NUMBER() OVER (PARTITION BY f.delivery_id ORDER BY f.id DESC) AS rn
                FROM feedback_events f JOIN deliveries d ON d.id = f.delivery_id
                WHERE f.reaction IN ('like', 'dislike', 'superlike', 'superdislike')
            )
            SELECT f.id AS feedback_id, f.event_id, f.card_id, f.delivery_id,
                f.reaction, f.comment, f.created_at, e.headline, e.topics_json,
                e.score_features_json, i.source_id, c.structured_output_json
            FROM latest f
            JOIN events e ON e.id = f.event_id
            JOIN cards c ON c.id = f.card_id
            LEFT JOIN source_items i ON i.id = e.primary_source_item_id
            WHERE f.rn = 1 AND datetime(f.created_at) >= datetime(?)
                AND datetime(f.created_at) <= datetime(?)
                AND (? IS NULL OR f.chat_id = ?)
            ORDER BY f.id DESC LIMIT ?
            """,
            (start.isoformat(), end.isoformat(), chat_id, chat_id, max(0, min(limit, 1000))),
        ).fetchall()
        detail_rows = connection.execute(
            """
            WITH latest AS (
                SELECT f.id, f.event_id, f.action, f.metadata_json, d.chat_id,
                    f.created_at,
                    ROW_NUMBER() OVER (
                        PARTITION BY f.delivery_id, json_extract(f.metadata_json, '$.article_id')
                        ORDER BY f.id DESC
                    ) AS rn
                FROM feedback_events f JOIN deliveries d ON d.id = f.delivery_id
                WHERE f.action IN ('detail_useful', 'detail_not_worth_it')
            )
            SELECT id, event_id, action FROM latest
            WHERE rn = 1 AND datetime(created_at) >= datetime(?)
                AND datetime(created_at) <= datetime(?)
                AND (? IS NULL OR chat_id = ?)
            ORDER BY id DESC LIMIT ?
            """,
            (start.isoformat(), end.isoformat(), chat_id, chat_id, max(0, min(limit, 1000))),
        ).fetchall()

    examples: list[dict[str, Any]] = []
    patches: list[dict[str, Any]] = []
    topic_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        event = dict(row)
        features = _object(event.get("score_features_json"))
        payload = _object(event.get("structured_output_json"))
        contour = features.get("primary_contour") or _object(payload.get("contours")).get(
            "primary_contour"
        )
        comment = str(event.get("comment") or "")[:2000]
        scope = "card_quality" if CARD_QUALITY_RE.search(comment) else "event_relevance"
        example = {
            "feedback_id": int(event["feedback_id"]),
            "event_id": int(event["event_id"]),
            "delivery_id": int(event["delivery_id"]),
            "source_id": str(event.get("source_id") or ""),
            "headline": str(event.get("headline") or "")[:240],
            "reaction": event["reaction"],
            "weight": REACTION_WEIGHTS[event["reaction"]],
            "comment": comment,
            "scope": scope,
            "primary_contour": canonical_contour_id(str(contour), event) if contour else None,
            "topics": _list(event.get("topics_json"))[:8],
        }
        examples.append(example)
        if scope == "card_quality":
            patches.append(
                _patch(
                    scope=scope,
                    target=f"card:{event['card_id']}",
                    direction="review_presentation",
                    examples=[example],
                    confidence=0.8,
                    rationale="Комментарий описывает подачу; интерес к теме по нему не снижаем.",
                )
            )
            continue
        if str(example["reaction"]).startswith("super"):
            patches.append(
                _patch(
                    scope="example",
                    target=f"event:{example['event_id']}",
                    direction="positive_example" if example["weight"] > 0 else "negative_example",
                    examples=[example],
                    confidence=0.75 if comment else 0.4,
                    rationale=(
                        "Сильная реакция: разобрать конкретный материал вместе с комментарием."
                        if comment
                        else "Сильная реакция без причины; не выводить запрет всей темы."
                    ),
                )
            )
        for topic in example["topics"]:
            topic_groups[topic].append(example)

    for topic, group in sorted(topic_groups.items()):
        # Repeated cards about one event are a single topical preference example.
        unique = {example["event_id"]: example for example in reversed(group)}
        group = list(unique.values())
        if len(group) < 3:
            continue
        total = sum(example["weight"] for example in group)
        positive = sum(example["weight"] > 0 for example in group)
        consistency = max(positive, len(group) - positive) / len(group)
        if abs(total) < 3 or consistency < 0.75:
            continue
        patches.append(
            _patch(
                scope="topic",
                target=topic,
                direction="increase" if total > 0 else "decrease",
                examples=group,
                confidence=min(0.85, 0.4 + 0.05 * len(group) + 0.15 * consistency),
                rationale=(
                    f"Сходное направление у {len(group)} разных событий; "
                    "предлагается небольшой обратимый сдвиг поискового приоритета."
                ),
            )
        )
    patches.sort(key=lambda patch: (-patch["confidence"], -len(patch["event_ids"]), patch["id"]))
    return {
        "version": REPORT_VERSION,
        "period_start": start.isoformat(),
        "period_end": end.isoformat(),
        "read_only": True,
        "automatic_changes": 0,
        "reaction_counts": dict(Counter(example["reaction"] for example in examples)),
        "examples": examples,
        "details_feedback": [dict(row) for row in detail_rows],
        "proposals": patches[: max(0, min(max_patches, 100))],
        "limits": {"reaction_limit": min(max(0, limit), 1000), "minimum_topic_events": 3},
    }


def render_preference_report(report: dict[str, Any]) -> str:
    """Human-readable Markdown for CLI/export; no generated instructions are applied."""
    lines = [
        "# Предложения по настройке OSVED",
        "",
        f"Период: {report['period_start']} — {report['period_end']}",
        f"Учтено последних реакций: {len(report['examples'])}. Изменения не применены.",
        "",
    ]
    for patch in report["proposals"]:
        refs = ", ".join(f"#{value}" for value in patch["feedback_ids"])
        lines.extend(
            [
                f"- {patch['scope']} / {patch['target']}: {patch['direction']}. "
                f"{patch['rationale']} Реакции: {refs}.",
                "",
            ]
        )
    if not report["proposals"]:
        lines.append("Пока недостаточно согласованного фидбэка для предложений.")
    return "\n".join(lines).rstrip() + "\n"
