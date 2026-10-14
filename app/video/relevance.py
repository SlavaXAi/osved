"""Personal metadata relevance. Channel feedback changes selectivity, never a clock."""

from __future__ import annotations

import hashlib
import html
import json
from datetime import UTC, datetime

from app.config import InterestProfile, load_interest_profile
from app.pipeline.editorial import extract_editorial_metadata
from app.video.result_feedback import result_usefulness_examples
from app.video.transcripts import youtube_video_id


def personalization_profile(settings, repository, chat_id):
    profile = (
        load_interest_profile(settings.interests_file)
        if settings.interests_file.exists()
        else InterestProfile(version=1)
    )
    result_examples, rated_videos = result_usefulness_examples(settings, repository, chat_id)
    # Discovery actions are weaker than usefulness after reading a result.
    # Ordinary-news dislikes can concern writing quality and are deliberately
    # not converted into video topic dislikes.
    with repository.connection() as db:
        examples = db.execute(
            """WITH latest AS (
                SELECT f.action,i.original_text,i.source_url,s.name,
                  ROW_NUMBER() OVER (PARTITION BY v.id ORDER BY f.id DESC) AS n,f.id
                FROM video_feedback_events f JOIN video_recommendations v
                  ON v.id=f.recommendation_id JOIN source_items i ON i.id=v.source_item_id
                  JOIN sources s ON s.id=i.source_id
                WHERE v.chat_id=? AND f.action IN ('interest','dismiss')
            ) SELECT action,original_text,source_url,name FROM latest
              WHERE n=1 ORDER BY id DESC LIMIT 100""",
            (chat_id,),
        ).fetchall()
    discovery_examples, seen = [], set()
    for row in examples:
        try:
            video_id = youtube_video_id(str(row["source_url"] or ""))
        except ValueError:
            continue
        if video_id in rated_videos or video_id in seen:
            continue
        seen.add(video_id)
        discovery_examples.append(
            {
                "video_id": video_id,
                "action": row["action"],
                "channel": row["name"],
                "metadata": str(row["original_text"] or "")[:500],
                "signal": "weaker_discovery_choice_not_result_satisfaction",
            }
        )
        if len(discovery_examples) == 8:
            break
    return {
        "interests": profile.model_dump(
            exclude={"production_max_age_days", "calibration_lookback_days"}
        ),
        "video_result_usefulness": result_examples,
        "video_feedback": discovery_examples,
        "video_feedback_policy": {
            "primary_signal": "video_result_usefulness",
            "reaction_meaning": "overall usefulness of the delivered result to the reader",
            "result_format_matters": True,
            "one_video_is_one_example": True,
            "discovery_signal_strength": "weaker",
            "open_or_request_means_satisfaction": False,
            "negative_result_is_topic_or_channel_ban": False,
        },
    }


def local_assessment(item, profile):
    """Conservative free fallback when a model is not configured, not semantic AI."""
    text = str(item.get("original_text") or "")
    detected = set(extract_editorial_metadata(text).topics)
    matched = sorted(detected & set(profile["interests"]["topics"]))
    score = min(0.95, 0.50 + 0.15 * len(matched)) if matched else 0.15
    title = html.escape(text.split("\n", 1)[0][:130] or str(item["source_name"]))
    return {
        "description_html": f"<b>{title}</b>\n\n"
        f"<i>{html.escape(str(item['source_name']))} · по описанию выпуска.</i>",
        "relevance": score,
        "matched_topics": matched,
        "reason": "Совпадение тем по метаданным"
        if matched
        else "Соответствие интересам не установлено",
    }


def validated_assessment(payload, *, item, profile, description):
    value = payload.get("relevance")
    topics = payload.get("matched_topics")
    quotes = payload.get("evidence_quotes")
    reason = payload.get("relevance_reason")
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not 0 <= value <= 1
        or not isinstance(topics, list)
        or not isinstance(quotes, list)
        or not isinstance(reason, str)
        or not 10 <= len(reason) <= 300
    ):
        raise ValueError("Invalid video relevance assessment")
    allowed = set(profile["interests"]["topics"])
    if any(not isinstance(t, str) or t not in allowed for t in topics):
        raise ValueError("Invented profile topic")
    text = str(item.get("original_text") or "")[:6000]
    if any(not isinstance(q, str) or len(q) < 6 or q not in text for q in quotes):
        raise ValueError("Relevance lacks metadata support")
    if value >= 0.45 and (not topics or not quotes):
        raise ValueError("A relevant recommendation needs a topic and an exact metadata quote")
    return {
        "description_html": description,
        "relevance": float(value),
        "matched_topics": topics[:6],
        "reason": reason,
    }


def relevance_threshold(channel_preference):
    return {-1: 0.75, 0: 0.55, 1: 0.45}[channel_preference]


def relevant(assessment, channel_preference):
    return assessment["relevance"] >= relevance_threshold(channel_preference)


class VideoRelevanceService:
    def __init__(self, settings, repository, chat_id, *, assessor=None):
        self.settings, self.repository, self.chat_id = settings, repository, chat_id
        self.profile = personalization_profile(settings, repository, chat_id)
        self.profile_hash = hashlib.sha256(
            json.dumps(self.profile, sort_keys=True).encode()
        ).hexdigest()
        self.remaining_new = 20
        self.assessor = assessor

    def assess(self, item):
        now = datetime.now(UTC).isoformat()
        with self.repository.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute(
                "SELECT * FROM video_relevance_assessments WHERE source_item_id=? "
                "AND chat_id=? AND profile_hash=?",
                (item["id"], self.chat_id, self.profile_hash),
            ).fetchone()
            if old:
                if old["status"] == "ready":
                    return json.loads(old["assessment_json"])
                if old["status"] != "budget_blocked":
                    return None
            if db.execute(
                "SELECT 1 FROM video_relevance_assessments WHERE source_item_id=? AND chat_id=? "
                "AND status IN ('preparing','uncertain') LIMIT 1",
                (item["id"], self.chat_id),
            ).fetchone():
                return None
            if self.settings.openrouter_api_key is not None and self.remaining_new <= 0:
                return None
            if self.settings.openrouter_api_key is not None:
                self.remaining_new -= 1
            if old:
                assessment_id = old["id"]
                db.execute(
                    "UPDATE video_relevance_assessments SET status='preparing',updated_at=? "
                    "WHERE id=?",
                    (now, assessment_id),
                )
            else:
                assessment_id = db.execute(
                    "INSERT INTO video_relevance_assessments(source_item_id,chat_id,profile_hash,"
                    "status,created_at,updated_at) VALUES(?,?,?,'preparing',?,?)",
                    (item["id"], self.chat_id, self.profile_hash, now, now),
                ).lastrowid
        try:
            if self.settings.openrouter_api_key is None:
                assessment = local_assessment(item, self.profile)
            else:
                from app.telegram_bot.sensory import video_description

                assessment = (self.assessor or video_description)(
                    self.settings,
                    self.repository,
                    item,
                    relevance_profile=self.profile,
                    assessment_id=assessment_id,
                )
            status = "ready" if assessment is not None else "budget_blocked"
        except Exception:
            # Existing receipt/error accounting remains in video_description.
            # Even malformed output is retained, and never automatically repaid.
            assessment, status = None, "uncertain"
        with self.repository.connection() as db:
            db.execute(
                "UPDATE video_relevance_assessments SET status=?,assessment_json=?,"
                "updated_at=? WHERE id=?",
                (
                    status,
                    json.dumps(assessment, ensure_ascii=False),
                    datetime.now(UTC).isoformat(),
                    assessment_id,
                ),
            )
        return assessment
