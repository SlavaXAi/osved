from __future__ import annotations

import hashlib
import html
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import cached_property
from math import isfinite

import yaml
from pydantic import BaseModel, Field, field_validator

from app.radar.feedback import REACTION_WEIGHTS

VERSION = "daily_editorial_v1"
QUALITY = re.compile(
    r"структур|абзац|заголов|оформл|формулиров|повтор|однотип|вод[аыу]|поверхност|"
    r"реклам|нечитаб|не\s*ясн|не\s+пон[яи]ма|поинт|споинт|в\s+ч[её]м\s+смысл|"
    r"почему.{0,15}важн|не\s*хват|мало\s+факт|больше\s+факт|преподнес|ссыл[ок]|"
    r"grammar|wording|format|too short|too long|more context",
    re.I,
)
LINK_ONLY = re.compile(r"\s*(?:https?://\S+|@[\w]+)\s*", re.I)


def obj(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return {}
    return value if isinstance(value, dict) else {}


def instant(value):
    stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return stamp.replace(tzinfo=UTC) if stamp.tzinfo is None else stamp.astimezone(UTC)


def event_text(event):
    """Content, not inherited publisher topics, is the preference evidence."""
    # The evaluated card excludes unrelated navigation and related-story links
    # that can appear in a publisher's full page.
    body = event.get("rendered_html") or event.get("summary") or event.get("original_text") or ""
    text = str(event.get("headline") or "") + " " + str(body)
    return html.unescape(re.sub(r"<[^>]*>", " ", text))[:16000]


def traits(event):
    text = event_text(event)
    title = str(event.get("headline") or "")

    def has(pattern):
        return bool(re.search(pattern, text, re.I))

    labels = set()
    ai = has(r"\b(?:AI|ИИ|LLM|Claude|Codex|Cursor|нейросет\w*|агент\w*|agent\w*)\b")
    if (
        has(r"AI[ -]?coding|вайб[ -]?код|кодинг|Claude\s+Code|Codex|Cursor|Qwen\s+Code")
        or ai
        and has(r"coding|codebase|репозитор|написан\w*\s+код|генерац\w*\s+код")
    ):
        labels.add("ai_coding")
    if ai and has(
        r"harness|харнесс|agentic\s+workflow|агентн\w*\s+(?:контур|workflow|процесс)|"
        r"orchestrat|оркестрац|human.in.the.loop|tripartite|context engineering"
    ):
        labels.add("agent_systems")
    if ai and has(
        r"бухгалтер|сверк\w*|закупк\w*|строител|логистик|accounting|procurement|"
        r"bookkeeping|invoice|domain.native|доменн\w*|отраслев\w*"
    ):
        labels.add("domain_workflows")
    if has(r"GitHub|open[ -]source|открыт\w*\s+(?:код|репозитор)|репозиторий"):
        labels.add("open_source")
    if has(r"robotic|робот\w*|physical intelligence|embodied"):
        labels.add("robotics")
    if has(
        r"войн\w*|санкц\w*|таможенн\w*|геополит|торгов\w*\s+(?:поток|баланс|пошлин)|"
        r"interest rates|central bank|ceasefire|ФРС|БРИКС"
    ):
        labels.add("world_affairs")
    if has(r"quantum|квантов|космос|spaceflight|сознани\w*|долголети\w*"):
        labels.add("frontier_science")
    if re.search(
        r"ecosystem|showcase|partners|summit|партн[её]р|экосистем|саммит|"
        r"региональн\w*\s+инновац|regional innovation|live updates|keynote recap",
        title,
        re.I,
    ):
        labels.add("corporate_roundup")
    if has(r"среди примеров|параллельно.{0,50}(?:компани|развива|представ)|ecosystem partners"):
        labels.add("corporate_roundup")
    if has(
        r"пентест|pentest|APT\b|ransomware|malware|CVE-\d|кибербезопас|cybersecurity"
    ) and not has(
        r"government|национальн|государств|критическ\w*\s+инфраструкт|"
        r"CERT|массов\w*\s+атак|миллион\w*\s+(?:устройств|пользовател)"
    ):
        labels.add("niche_security")
    if has(
        r"Ctrl[ -]?Tab|waitlist|списк\w*\s+ожидания|доступ.{0,40}(?:Pro и Max|всем пользовател)|"
        r"UI\s+(?:tweak|fix)|исправлен\w*\s+опечатк"
    ):
        labels.add("access_or_ui")
    if re.search(
        r"funding|valuation|поднял\w*|привл[её]к|инвестиц|оценк\w*\s+компани", title, re.I
    ) and not has(
        r"пользовател|клиент|выручк|customer|revenue|workflow|автоматиз|"
        r"производит|разрабатывает|сервис\w*\s+(?:для|позволя)|product"
    ):
        labels.add("funding_only")
    return labels


GENRES = {"corporate_roundup", "niche_security", "access_or_ui", "funding_only"}


class DailyPolicy(BaseModel):
    version: int = 1
    priorities: dict[str, float] = Field(default_factory=dict)
    genre_penalties: dict[str, float] = Field(default_factory=dict)
    max_adjustment: float = Field(default=2.0, ge=0, le=3)
    minimum_feedback_events: int = Field(default=3, ge=3, le=20)
    feedback_half_life_days: float = Field(default=45, ge=14, le=180)

    @field_validator("priorities", "genre_penalties")
    @classmethod
    def bounded(cls, value):
        if any(not isfinite(v) or not 0 <= v <= 2 for v in value.values()):
            raise ValueError("Preference weights must be finite and between 0 and 2")
        return value


def feedback_scope(comment):
    comment = str(comment or "").strip()
    if LINK_ONLY.fullmatch(comment):
        return "source_request"
    if QUALITY.search(comment):
        return "presentation_or_novelty"
    if re.fullmatch(r"(?:пидары|суки|пиздец)[.!\s]*", comment, re.I):
        return "ambiguous"
    return "result_usefulness"


def feedback_examples(repository, owner, *, now, limit=500):
    """Latest result votes before cutoff; one topical vote per underlying event."""
    with repository.connection() as db:
        rows = db.execute(
            """WITH latest AS (
              SELECT f.*,ROW_NUMBER() OVER(PARTITION BY f.delivery_id ORDER BY f.id DESC) rn
              FROM feedback_events f JOIN deliveries d ON d.id=f.delivery_id
              WHERE d.chat_id=? AND d.delivery_status='sent'
                AND f.reaction IN ('like','superlike','dislike','superdislike')
                AND julianday(f.created_at)<=julianday(?)
            ) SELECT f.*,e.headline,c.rendered_html,i.original_text,e.event_type
              FROM latest f JOIN events e ON e.id=f.event_id JOIN cards c ON c.id=f.card_id
              JOIN source_items i ON i.id=e.primary_source_item_id
              WHERE f.rn=1 ORDER BY f.id DESC LIMIT ?""",
            (owner, now.isoformat(), min(1000, max(0, limit))),
        ).fetchall()
    output = []
    seen_events = set()
    for raw in rows:
        row = dict(raw)
        scope = feedback_scope(row["comment"])
        vote = REACTION_WEIGHTS[row["reaction"]]
        topic_weight = vote
        if row["event_id"] in seen_events:
            # Preserve a different result's presentation lesson, never another
            # independent topical vote for the same underlying event.
            if scope != "presentation_or_novelty":
                continue
            topic_weight = 0
        else:
            seen_events.add(row["event_id"])
        if scope in {"ambiguous", "source_request"} or (vote < 0 and scope != "result_usefulness"):
            topic_weight = 0
        elif scope == "presentation_or_novelty":
            topic_weight *= 0.5
        output.append(
            {
                "feedback_id": row["id"],
                "event_id": row["event_id"],
                "created_at": row["created_at"],
                "headline": row["headline"],
                "summary": re.sub(r"<[^>]*>", " ", row["rendered_html"] or "")[:420],
                "reaction": row["reaction"],
                "comment": str(row["comment"] or "")[:1000],
                "scope": scope,
                "weight": vote,
                "topic_weight": topic_weight,
                "traits": sorted(traits(row)),
            }
        )
    return output


@dataclass
class DailyContext:
    policy: DailyPolicy
    examples: list[dict]
    now: datetime
    engagement: list[dict] = field(default_factory=list)

    @cached_property
    def digest(self):
        payload = [
            VERSION,
            self.policy.model_dump(),
            [(e["feedback_id"], e["topic_weight"]) for e in self.examples],
            self.engagement,
        ]
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]

    @cached_property
    def learned(self):
        groups = defaultdict(list)
        for row in self.examples:
            vote = row["topic_weight"]
            if not vote:
                continue
            age = max(0, (self.now - instant(row["created_at"])).total_seconds() / 86400)
            for label in row["traits"]:
                groups[label].append((vote, 2 ** (-age / self.policy.feedback_half_life_days), row))
        learned = {}
        for label, values in groups.items():
            if len(values) < self.policy.minimum_feedback_events:
                continue
            positives = sum(v > 0 for v, _, _ in values)
            if max(positives, len(values) - positives) / len(values) < 0.75:
                continue
            # Negative usefulness counts against a boost, but is not a topic ban.
            if positives < len(values) / 2 and label not in GENRES:
                continue
            mass = sum(w for _, w, _ in values)
            shift = sum((1 if v > 0 else -1) * w for v, w, _ in values) / (mass + 3)
            learned[label] = {
                "weight": round(0.6 * shift, 3),
                "event_ids": [r["event_id"] for _, _, r in values],
                "feedback_ids": [r["feedback_id"] for _, _, r in values],
            }
        return learned

    def adjustment(self, event):
        labels = traits(event)
        useful = labels - GENRES
        # Cosmetic access changes are not agent engineering advances.
        if "access_or_ui" in labels:
            useful -= {"ai_coding", "agent_systems", "open_source"}
        if "open_source" in useful and not (
            useful & {"ai_coding", "agent_systems", "domain_workflows"}
            or re.search(
                r"[1-9][\d,. ]{2,}\s*(?:stars|зв[её]зд)|тысяч\w*\s+зв[её]зд|"
                r"adoption|пользовател|нов\w*\s+возможност",
                event_text(event),
                re.I,
            )
        ):
            useful.remove("open_source")
        priority = min(
            self.policy.max_adjustment, sum(self.policy.priorities.get(t, 0) for t in useful)
        )
        penalties = min(
            self.policy.max_adjustment, sum(self.policy.genre_penalties.get(t, 0) for t in labels)
        )
        learned = self.learned
        feedback = max(-0.6, min(0.6, sum(learned[t]["weight"] for t in labels if t in learned)))
        usage_ids = {
            r["key"]
            for r in self.engagement
            if set(r["traits"]) & useful
            and any(self.policy.priorities.get(t, 0) > 0 for t in set(r["traits"]) & useful)
        }
        engagement = 0.15 if len(usage_ids) >= 3 else 0.0
        total = max(
            -self.policy.max_adjustment,
            min(self.policy.max_adjustment, priority - penalties + feedback + engagement),
        )
        return {
            "version": VERSION,
            "context": self.digest,
            "traits": sorted(labels),
            "priority": round(priority, 3),
            "genre_penalty": round(-penalties, 3),
            "feedback": round(feedback, 3),
            "engagement": engagement,
            "engagement_ids": sorted(usage_ids)[:20],
            "adjustment": round(total, 3),
            "feedback_ids": sorted(
                {i for t in labels if t in learned for i in learned[t]["feedback_ids"]}
            ),
            "quality_note": "Preference signals do not verify source claims.",
        }


def load_context(settings, repository, *, owner=None, now=None):
    if not settings or not settings.daily_personalization_enabled:
        return None
    owner = owner or settings.allowed_user_id or repository.get_runtime_setting("allowed_user_id")
    if not owner:
        return None
    now = now or datetime.now(UTC)
    path = settings.daily_selection_file
    policy = DailyPolicy.model_validate(yaml.safe_load(path.read_text()) if path.is_file() else {})
    return DailyContext(
        policy,
        feedback_examples(repository, int(owner), now=now),
        now,
        engagement_examples(repository, int(owner), now=now),
    )


def engagement_examples(repository, owner, *, now):
    """Successful reading requests are weak interest; failures and silence are not votes."""
    with repository.connection() as db:
        details = db.execute(
            "SELECT q.event_id,e.headline,i.original_text FROM detail_requests q "
            "JOIN events e ON e.id=q.event_id JOIN source_items i ON i.id=e.primary_source_item_id "
            "WHERE q.chat_id=? AND q.user_id=? AND q.status IN('ready','sent') "
            "AND julianday(COALESCE(q.completed_at,q.updated_at))<=julianday(?) "
            "AND julianday(q.requested_at)>=julianday(?,'-60 days') ORDER BY q.id DESC LIMIT 100",
            (owner, owner, now.isoformat(), now.isoformat()),
        ).fetchall()
        videos = db.execute(
            "SELECT j.video_id,i.original_text FROM video_pdf_jobs j "
            "JOIN source_items i ON i.external_id=j.video_id JOIN sources s ON s.id=i.source_id "
            "WHERE j.chat_id=? AND j.user_id=? AND j.status='sent' AND s.kind='youtube' "
            "AND julianday(j.updated_at)<=julianday(?) "
            "AND julianday(j.created_at)>=julianday(?,'-60 days') "
            "AND NOT EXISTS(SELECT 1 FROM video_result_feedback f JOIN video_pdf_jobs rated "
            "ON rated.id=f.job_id WHERE rated.video_id=j.video_id AND f.chat_id=? "
            "AND f.reaction='dislike' AND julianday(f.updated_at)<=julianday(?)) "
            "ORDER BY j.id DESC LIMIT 150",
            (owner, owner, now.isoformat(), now.isoformat(), owner, now.isoformat()),
        ).fetchall()
    result = {}
    for row in details:
        key = f"event:{row['event_id']}"
        result[key] = {"key": key, "traits": sorted(traits(dict(row)))}
    for row in videos:
        key = "video:" + row["video_id"]
        result[key] = {"key": key, "traits": sorted(traits(dict(row)))}
    return list(result.values())


def prompt_examples(context):
    eligible = [e for e in context.examples if e["scope"] not in {"ambiguous", "source_request"}]
    salient = [e for e in eligible if len(e["comment"]) > 30][:4]
    combined = list({e["feedback_id"]: e for e in [*eligible[:8], *salient, *eligible]}.values())
    return [{**e, "comment": e["comment"][:500]} for e in combined[:12]]


def record_selection(repository, context, *, owner, stage, candidates, selected):
    if context is None or owner is None:
        return
    selected_by_id = {int(r["id"]): r for r in selected}
    rows = []
    for event in candidates:
        chosen = selected_by_id.get(int(event["id"]))
        rows.append(
            {
                "event_id": event["id"],
                "selected": chosen is not None,
                "original_score": (
                    None
                    if isinstance(event.get("score"), float) and not isfinite(event["score"])
                    else event.get("score")
                ),
                "effective_score": chosen.get("effective_score") if chosen else None,
                "reason": (
                    chosen.get("score_features", {}).get("daily_personalization")
                    if chosen
                    else context.adjustment(event)
                ),
                "saturation": chosen.get("saturation_features") if chosen else None,
            }
        )
    payload = json.dumps(
        {"policy": context.digest, "rows": rows},
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    )
    key = hashlib.sha256(f"{owner}:{stage}:{context.now:%Y%m%d%H}:{payload}".encode()).hexdigest()
    with repository.connection() as db:
        db.execute(
            "INSERT OR IGNORE INTO daily_selection_runs(run_key,owner_id,stage,policy_version,"
            "context_hash,payload_json,created_at) VALUES(?,?,?,?,?,?,?)",
            (key, owner, stage, VERSION, context.digest, payload, context.now.isoformat()),
        )
