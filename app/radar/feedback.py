from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from app.db.repository import Repository

FEEDBACK_EXAMPLES_VERSION = "search_feedback_examples_v1"
MAX_FEEDBACK_EXAMPLES = 12

Reaction = Literal["like", "dislike", "superlike", "superdislike"]

REACTION_WEIGHTS: dict[str, int] = {
    "superdislike": -2,
    "dislike": -1,
    "like": 1,
    "superlike": 2,
}


@dataclass(frozen=True)
class FeedbackExample:
    headline: str
    summary: str
    topics: tuple[str, ...]
    reaction: Reaction
    comment: str
    weight: int

    def as_prompt_dict(self) -> dict[str, Any]:
        return {
            "headline": self.headline,
            "summary": self.summary,
            "topics": list(self.topics),
            "reaction": self.reaction,
            "comment": self.comment,
            "weight": self.weight,
        }


def _compact_text(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def _candidate_payload(structured_output: dict[str, Any]) -> dict[str, Any]:
    candidate = structured_output.get("candidate")
    return candidate if isinstance(candidate, dict) else {}


def load_feedback_examples(
    repository: Repository,
    *,
    limit: int = MAX_FEEDBACK_EXAMPLES,
) -> list[dict[str, Any]]:
    """Return the latest reaction row per discovery delivery as bounded prompt examples.

    The reaction and comment always come from the same feedback_events row. A newer
    reaction with no comment therefore cannot inherit an older comment.
    """
    bounded_limit = max(0, min(int(limit), MAX_FEEDBACK_EXAMPLES))
    if bounded_limit == 0:
        return []

    with repository.connection() as connection:
        rows = connection.execute(
            """
            WITH ranked_feedback AS (
                SELECT
                    feedback_events.id,
                    feedback_events.delivery_id,
                    feedback_events.reaction,
                    feedback_events.comment,
                    cards.structured_output_json,
                    ROW_NUMBER() OVER (
                        PARTITION BY feedback_events.delivery_id
                        ORDER BY feedback_events.id DESC
                    ) AS reaction_rank
                FROM feedback_events
                JOIN deliveries
                  ON deliveries.id = feedback_events.delivery_id
                JOIN cards
                  ON cards.id = deliveries.card_id
                WHERE feedback_events.reaction IN (
                    'like', 'dislike', 'superlike', 'superdislike'
                )
                  AND json_extract(
                      cards.structured_output_json,
                      '$.evaluation_stage'
                  ) = 'discovery'
            )
            SELECT reaction, comment, structured_output_json
            FROM ranked_feedback
            WHERE reaction_rank = 1
            ORDER BY id DESC
            LIMIT ?
            """,
            (bounded_limit,),
        ).fetchall()

    examples: list[dict[str, Any]] = []
    for row in rows:
        reaction = str(row["reaction"])
        weight = REACTION_WEIGHTS.get(reaction)
        if weight is None:
            continue
        try:
            structured_output = json.loads(row["structured_output_json"] or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(structured_output, dict):
            continue
        candidate = _candidate_payload(structured_output)
        headline = _compact_text(candidate.get("headline"), 180)
        summary = _compact_text(candidate.get("summary"), 420)
        if not headline or not summary:
            continue
        raw_topics = candidate.get("topics") or structured_output.get("topics") or []
        topics = tuple(
            dict.fromkeys(
                _compact_text(value, 48) for value in raw_topics if _compact_text(value, 48)
            )
        )[:6]
        example = FeedbackExample(
            headline=headline,
            summary=summary,
            topics=topics,
            reaction=reaction,  # type: ignore[arg-type]
            comment=_compact_text(row["comment"], 360),
            weight=weight,
        )
        examples.append(example.as_prompt_dict())
    return examples
