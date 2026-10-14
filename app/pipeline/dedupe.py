from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING
from urllib.parse import parse_qsl, urlencode, urlsplit

if TYPE_CHECKING:
    from app.db.repository import Repository

TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)
STOPWORDS = {
    "а",
    "без",
    "был",
    "была",
    "было",
    "будет",
    "в",
    "во",
    "вот",
    "для",
    "до",
    "его",
    "и",
    "из",
    "как",
    "на",
    "не",
    "новый",
    "новая",
    "новое",
    "о",
    "он",
    "она",
    "от",
    "по",
    "при",
    "с",
    "со",
    "так",
    "такой",
    "тут",
    "у",
    "что",
    "это",
    "этот",
    "the",
    "a",
    "an",
    "and",
    "for",
    "in",
    "of",
    "to",
    "new",
    "launch",
    "launched",
    "announced",
    "announces",
    "company",
    "companies",
    "компания",
    "компании",
    "анонсировала",
    "представила",
    "представил",
    "выпустила",
    "искусственного",
    "интеллекта",
    "artificial",
    "intelligence",
}

GENERIC_PATHS = {
    "",
    "/",
    "/news",
    "/blog",
    "/updates",
    "/research",
    "/press",
    "/en",
    "/ru",
    "/en/news",
    "/en/blog",
    "/releases",
    "/latest",
    "/explore",
    "/trending",
}
TRACKING_QUERY_KEYS = {"fbclid", "gclid", "ref", "ref_src", "source", "feature", "si"}
VERSION_RE = re.compile(
    r"\b(?P<family>gpt|grok|claude|gemini|deepseek|qwen|llama|v)[ -]?"
    r"(?P<version>\d+(?:[.-]\d+)*)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DedupeStats:
    events_seen: int = 0
    events_merged: int = 0


def headline_tokens(value: str) -> set[str]:
    tokens = {
        token
        for token in TOKEN_RE.findall(value.lower())
        if token not in STOPWORDS and (len(token) >= 3 or any(char.isdigit() for char in token))
    }
    # Chinese has no whitespace word boundary. Long shared spans can be compared
    # conservatively; cross-language joins still require a concrete shared URL.
    for span in re.findall(r"[\u3400-\u9fff]{4,}", value):
        tokens.update(span[index : index + 4] for index in range(len(span) - 3))
    return tokens


def headline_similarity(left: str, right: str) -> float:
    left_tokens = headline_tokens(left)
    right_tokens = headline_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    common = len(left_tokens & right_tokens)
    jaccard = common / len(left_tokens | right_tokens)
    containment = common / min(len(left_tokens), len(right_tokens))
    return max(jaccard, containment)


def canonical_evidence_url(value: str) -> str | None:
    """Identify an actual article/post, preserving content-defining query fields.

    A homepage, profile, latest-release URL or shared video host is not an event.
    We deliberately do not lowercase URL paths: some publishers use case-sensitive IDs.
    """
    try:
        parts = urlsplit(value)
        host = (parts.hostname or "").casefold().removeprefix("www.")
    except (TypeError, ValueError):
        return None
    if parts.scheme not in {"http", "https"} or not host:
        return None
    host = {
        "twitter.com": "x.com",
        "mobile.twitter.com": "x.com",
        "m.youtube.com": "youtube.com",
    }.get(host, host)
    path = parts.path.rstrip("/")
    query = [
        (key, val)
        for key, val in parse_qsl(parts.query, keep_blank_values=True)
        if not key.casefold().startswith("utm_") and key.casefold() not in TRACKING_QUERY_KEYS
    ]
    if host in {"t.me", "telegram.me"}:
        return None
    if host == "youtu.be":
        return f"youtube.com/watch?v={path.lstrip('/')}" if path.count("/") == 1 else None
    if host == "youtube.com":
        video_id = dict(query).get("v")
        if path == "/watch" and video_id:
            return f"youtube.com/watch?v={video_id}"
        if path.startswith(("/shorts/", "/live/")) and len(path.split("/")) == 3:
            return f"youtube.com/watch?v={path.rsplit('/', 1)[-1]}"
        return None
    if host == "x.com":
        match = re.search(r"/status/(\d+)", path)
        return f"x.com/status/{match.group(1)}" if match else None
    if path.casefold() in GENERIC_PATHS and not any(
        key.casefold() in {"id", "p", "article", "story", "post"} for key, _ in query
    ):
        return None
    if host == "github.com" and (
        len(path.strip("/").split("/")) <= 2 or path.endswith("/releases/latest")
    ):
        return None
    suffix = f"?{urlencode(sorted(query))}" if query else ""
    return f"{host}{path}{suffix}"


def _urls(row: dict) -> list[str]:
    raw = row.get("outbound_urls_json") or row.get("outbound_urls") or []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            raw = []
    return [value for value in raw if isinstance(value, str)] if isinstance(raw, list) else []


def evidence_url_identities(row: dict) -> set[str]:
    identities: set[str] = set()
    values = _urls(row)
    values.extend(str(row[key]) for key in ("primary_url", "source_url") if row.get(key))
    for value in values:
        if identity := canonical_evidence_url(value):
            identities.add(identity)
    return identities


def safe_shared_evidence(left: dict, right: dict) -> set[str]:
    left_urls = _urls(left)
    right_urls = _urls(right)
    if len(left_urls) > 4 or len(right_urls) > 4:
        return set()
    return evidence_url_identities(left) & evidence_url_identities(right)


def conflicting_versions(left: str, right: str) -> bool:
    def versions(text: str) -> dict[str, set[str]]:
        result: dict[str, set[str]] = {}
        for match in VERSION_RE.finditer(text):
            result.setdefault(match.group("family").casefold(), set()).add(
                match.group("version").replace("-", ".")
            )
        return result

    first, second = versions(left), versions(right)
    return any(not first[family] & second[family] for family in first.keys() & second.keys())


def policy_stage(row: dict) -> str | None:
    # A proposed/passed/signed law is not the same development. Never erase an
    # update simply because it concerns the same countries or bill. The current
    # headline takes precedence over past developments mentioned in the body.
    for key in ("headline", "source_url", "original_text"):
        text = str(row.get(key) or "")
        for stage, pattern in (
            (
                "pending_signature",
                r"(?:plans?|will|expected).{0,15}sign\b|"
                r"not.signed|не.подписан|планирует подпис|подпишет",
            ),
            ("signed", r"\bsigns\b|\bsigned\b|подписал\w*|подписан\w*"),
            ("house_passed", r"house.passe[sd]|палат\w*.{0,40}(?:одобр|принял|принят)"),
            ("senate_passed", r"senate.passe[sd]|сенат.{0,30}(?:одобр|принял|принят)"),
            ("proposed", r"proposes?|introduces?.bill|предложил\w*|внес[её]н\w*"),
        ):
            if re.search(pattern, text, re.IGNORECASE):
                return stage
    return None


def same_policy_story_slug(left: dict, right: dict) -> bool:
    stage = policy_stage(left)
    if stage is None or stage != policy_stage(right):
        return False

    def tokens(row: dict) -> set[str]:
        path = urlsplit(str(row.get("source_url") or "")).path.rstrip("/").split("/")[-1]
        return (
            set(re.findall(r"[a-z]{3,}", path.lower()))
            - STOPWORDS
            - {
                "html",
                "into",
                "news",
                "world",
                "story",
                "article",
            }
        )

    a, b = tokens(left), tokens(right)
    common = a & b
    return len(common) >= 4 and len(common) / max(1, min(len(a), len(b))) >= 0.85


def event_match_reason(left: dict, right: dict, *, threshold: float = 0.45) -> str | None:
    """Return explainable conservative match evidence, never a claim of truth."""
    left_headline, right_headline = (
        str(left.get("headline") or ""),
        str(right.get("headline") or ""),
    )
    if conflicting_versions(left_headline, right_headline):
        return None
    left_stage, right_stage = policy_stage(left), policy_stage(right)
    if left_stage and right_stage and left_stage != right_stage:
        return None
    if safe_shared_evidence(left, right):
        return "shared_specific_source_url"
    if same_policy_story_slug(left, right):
        return "shared_action_and_story_slug"
    common = headline_tokens(left_headline) & headline_tokens(right_headline)
    if len(common) >= 3 and headline_similarity(left_headline, right_headline) >= threshold:
        return "headline_overlap"
    return None


def deduplicate_events(
    repository: Repository,
    *,
    threshold: float = 0.45,
    window_hours: int = 72,
) -> DedupeStats:
    # A concept explainer and an event report may cite the same post while
    # serving different purposes. Concepts deduplicate by their own identity.
    rows = [row for row in repository.dedupe_event_rows() if row["event_type"] != "concept_brief"]
    active = {int(row["id"]): row for row in rows}
    merged = 0

    for left_index, left in enumerate(rows):
        left_id = int(left["id"])
        if left_id not in active:
            continue
        left_at = datetime.fromisoformat(str(left["event_at"]).replace("Z", "+00:00"))
        for right in rows[left_index + 1 :]:
            right_id = int(right["id"])
            if right_id not in active:
                continue
            right_at = datetime.fromisoformat(str(right["event_at"]).replace("Z", "+00:00"))
            if abs((left_at - right_at).total_seconds()) > window_hours * 3600:
                continue
            reason = event_match_reason(left, right, threshold=threshold)
            if not reason:
                continue

            preserve_history = (
                (left["has_cards"] and right["has_cards"])
                or (left["has_editorial_memory"] and right["has_editorial_memory"])
                or left.get("status") == "research_insufficient"
                or right.get("status") == "research_insufficient"
            )
            if preserve_history:
                strong = reason in {"shared_specific_source_url", "shared_action_and_story_slug"}
                left_headline, right_headline = str(left["headline"]), str(right["headline"])
                strong |= (
                    headline_similarity(left_headline, right_headline) >= 0.85
                    and len(headline_tokens(left_headline) & headline_tokens(right_headline)) >= 4
                )
                if not strong or (left.get("has_delivery") and right.get("has_delivery")):
                    continue
                if right.get("has_delivery") or (
                    right.get("has_valid_card") and not left.get("has_valid_card")
                ):
                    target_id, duplicate_id = right_id, left_id
                elif left.get("has_delivery") or left.get("has_valid_card"):
                    target_id, duplicate_id = left_id, right_id
                else:
                    target_id, duplicate_id = sorted((left_id, right_id))
                if repository.link_duplicate_event(target_id, duplicate_id, reason=reason):
                    active.pop(duplicate_id, None)
                    merged += 1
                    if duplicate_id == left_id:
                        break
                continue
            if right["has_cards"]:
                target_id, duplicate_id = right_id, left_id
            elif left["has_cards"]:
                target_id, duplicate_id = left_id, right_id
            elif right["has_editorial_memory"]:
                target_id, duplicate_id = right_id, left_id
            elif left["has_editorial_memory"]:
                target_id, duplicate_id = left_id, right_id
            elif right["has_research"] and not left["has_research"]:
                target_id, duplicate_id = right_id, left_id
            elif left["has_research"] and not right["has_research"]:
                target_id, duplicate_id = left_id, right_id
            elif float(right["score"] or 0) > float(left["score"] or 0):
                target_id, duplicate_id = right_id, left_id
            else:
                target_id, duplicate_id = left_id, right_id
            repository.merge_events(target_id, duplicate_id)
            active.pop(duplicate_id, None)
            merged += 1
            if duplicate_id == left_id:
                break

    return DedupeStats(events_seen=len(rows), events_merged=merged)
