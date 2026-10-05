from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

ORGANIZATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE))
    for name, pattern in (
        ("OpenAI", r"\bOpenAI\b"),
        ("Google DeepMind", r"\b(?:Google\s+DeepMind|DeepMind)\b"),
        ("Google", r"\bGoogle\b"),
        ("Anthropic", r"\bAnthropic\b"),
        ("NVIDIA", r"\bNVIDIA\b"),
        ("Microsoft", r"\bMicrosoft\b"),
        ("Meta", r"\bMeta\b"),
        ("xAI", r"\b(?:xAI|Grok)\b"),
        ("DeepSeek", r"\bDeepSeek\b"),
        ("Alibaba Qwen", r"\b(?:Alibaba\b|Qwen(?=\b|\d))"),
        ("Mistral", r"\bMistral\b"),
        ("Яндекс", r"\b(?:Яндекс|Yandex)\b"),
        ("Сбер", r"\b(?:Сбер|Sber)\b"),
        ("Kaspersky", r"\b(?:Kaspersky|Касперск\w*)\b"),
        ("Huawei", r"\bHuawei\b"),
        ("Unitree", r"\bUnitree\b"),
        ("DJI", r"\bDJI\b"),
        ("Anduril", r"\bAnduril\b"),
        ("Shield AI", r"\bShield\s+AI\b"),
        ("Helsing", r"\bHelsing\b"),
        ("Palantir", r"\bPalantir\b"),
        ("TSMC", r"\bTSMC\b"),
        ("Samsung", r"\bSamsung\b"),
        ("SK hynix", r"\bSK\s+hynix\b"),
        ("ASML", r"\bASML\b"),
        ("AMD", r"\bAMD\b"),
        ("Intel", r"\bIntel\b"),
        ("Stripe", r"\bStripe\b"),
        ("Ramp", r"\bRamp\b"),
    )
)

TOPIC_PATTERNS: dict[str, re.Pattern[str]] = {
    "ai": re.compile(r"\b(?:AI|ИИ|LLM|модел\w*|нейросет\w*)\b", re.IGNORECASE),
    "agents": re.compile(r"\b(?:agent\w*|агент\w*)\b", re.IGNORECASE),
    "robotics": re.compile(r"\b(?:robot\w*|робот\w*)\b", re.IGNORECASE),
    "semiconductors": re.compile(
        r"\b(?:chip\w*|semiconductor\w*|чип\w*|полупроводник\w*|GPU|NPU)\b",
        re.IGNORECASE,
    ),
    "manufacturing": re.compile(
        r"\b(?:manufactur\w*|factory|factories|производств\w*|фабрик\w*|завод\w*)\b",
        re.IGNORECASE,
    ),
    "cybersecurity": re.compile(
        r"\b(?:cyber\w*|malware|ransomware|уязвимост\w*|кибер\w*|вредонос\w*)\b",
        re.IGNORECASE,
    ),
    "defense_tech": re.compile(
        r"\b(?:defen[cs]e|military|drone\w*|дрон\w*|БПЛА|РЭБ|военн\w*)\b",
        re.IGNORECASE,
    ),
    "technology_economy": re.compile(
        r"\b(?:funding|valuation|revenue|contract|инвестиц\w*|выручк\w*|контракт\w*)\b",
        re.IGNORECASE,
    ),
    "open_source": re.compile(r"\b(?:open[ -]source|открыт\w+\s+код\w*|GitHub)\b", re.IGNORECASE),
}

REGION_PATTERNS: dict[str, re.Pattern[str]] = {
    "china": re.compile(r"\b(?:China|Chinese|Китай\w*|Шэньчжэнь|Пекин)\b", re.IGNORECASE),
    "russia": re.compile(r"\b(?:Russia|Russian|Росси\w*|Москв\w*)\b", re.IGNORECASE),
    "ukraine": re.compile(r"\b(?:Ukraine|Ukrainian|Украин\w*)\b", re.IGNORECASE),
    "united_states": re.compile(r"\b(?:United States|U\.S\.|USA|США)\b", re.IGNORECASE),
    "central_asia": re.compile(
        r"\b(?:Kazakhstan|Armenia|Казахстан\w*|Армени\w*|Узбекистан\w*)\b",
        re.IGNORECASE,
    ),
    "east_asia": re.compile(
        r"\b(?:Taiwan|Japan|Korea|Тайван\w*|Япони\w*|Коре\w*)\b",
        re.IGNORECASE,
    ),
    "europe": re.compile(r"\b(?:Europe|European|Европ\w*|ЕС|EU)\b", re.IGNORECASE),
}

MATERIAL_PATTERNS = re.compile(
    r"\b(?:launch\w*|releas\w*|introduc\w*|deploy\w*|funding|acqui\w*|contract\w*|"
    r"benchmark\w*|record\w*|research\w*|paper\w*|factory|production|manufactur\w*|"
    r"выпуст\w*|представ\w*|запуст\w*|внедр\w*|контракт\w*|инвестиц\w*|"
    r"бенчмарк\w*|рекорд\w*|исследован\w*|производств\w*|фабрик\w*)\b",
    re.IGNORECASE,
)
REAL_WORLD_PATTERNS = re.compile(
    r"\b(?:deploy\w*|production|factory|contract|funding|revenue|users?|customers?|"
    r"внедр\w*|производств\w*|завод\w*|контракт\w*|инвестиц\w*|пользовател\w*)\b",
    re.IGNORECASE,
)
ROUTINE_PATTERNS = re.compile(
    r"\b(?:minor|patch|патч\w*|bug\s*fix\w*|chore|dependency\s+update|typo|documentation\s+update|"
    r"weekly\s+(?:digest|roundup)|UI\s+(?:tweak|fix)|мелк\w+\s+исправлен\w*|"
    r"исправлен\w+\s+опечатк\w*|еженедельн\w+\s+дайджест\w*)\b",
    re.IGNORECASE,
)
DIGEST_PATTERNS = re.compile(
    r"(?:^|\n)\s*(?:📬\s*)?(?:утренн\w+\s+подборк\w*|итоги\s+(?:уходящ\w+\s+)?недел\w*|"
    r"нейродайджест\w*|дайджест\w*|weekly\s+(?:digest|roundup)|"
    r"мероприятия.{0,60}(?:этой|текущей)\s+недел\w*)",
    re.IGNORECASE,
)
SUBSTANTIVE_CHANGE_PATTERNS = re.compile(
    r"\b(?:major|architectur\w*|new\s+(?:model|capabilit\w*|feature\w*)|"
    r"benchmark\w*|record\w*|deploy\w*|funding|acqui\w*|contract\w*|factory|"
    r"production|manufactur\w*|critical|CVE-\d+|zero[ -]day|actively\s+exploited|"
    r"vulnerab\w*|архитектур\w*|нов\w+\s+(?:модел\w*|возможност\w*|"
    r"функц\w*)|бенчмарк\w*|рекорд\w*|внедр\w*|инвестиц\w*|контракт\w*|"
    r"производств\w*|фабрик\w*|завод\w*|критическ\w*|нулев\w+\s+дн\w*|"
    r"эксплуатир\w*|уязвимост\w*)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EditorialMetadata:
    entities: dict[str, list[str]]
    topics: list[str]
    regions: list[str]


def extract_editorial_metadata(
    text: str,
    *,
    source_topics: list[str] | None = None,
    source_region: str | None = None,
) -> EditorialMetadata:
    organizations = [name for name, pattern in ORGANIZATION_PATTERNS if pattern.search(text)]
    topics = list(dict.fromkeys(source_topics or []))
    for topic, pattern in TOPIC_PATTERNS.items():
        if pattern.search(text) and topic not in topics:
            topics.append(topic)
    # A publisher's registration country is not automatically the geography of an event.
    # Keep the argument as an explicit hint for future classifiers, but require textual evidence.
    _ = source_region
    regions: list[str] = []
    for region, pattern in REGION_PATTERNS.items():
        if pattern.search(text) and region not in regions:
            regions.append(region)
    if not regions:
        regions.append("global")
    return EditorialMetadata(
        entities={"organizations": organizations},
        topics=topics,
        regions=regions,
    )


def newsworthiness_features(text: str, *, source_role: str) -> dict[str, float]:
    material = 2.0 if MATERIAL_PATTERNS.search(text) else 0.0
    real_world = 1.0 if REAL_WORLD_PATTERNS.search(text) else 0.0
    primary_evidence = 0.75 if source_role == "primary" else 0.0
    routine = (
        -2.5
        if ROUTINE_PATTERNS.search(text) and not SUBSTANTIVE_CHANGE_PATTERNS.search(text)
        else 0.0
    )
    digest = -4.0 if DIGEST_PATTERNS.search(text[:500]) else 0.0
    return {
        "material_event": material,
        "real_world_impact": real_world,
        "primary_evidence": primary_evidence,
        "routine_penalty": routine,
        "digest_penalty": digest,
    }


def event_organizations(event: dict[str, Any]) -> list[str]:
    raw = event.get("entities_json") or event.get("entities") or {}
    if isinstance(raw, str):
        import json

        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = {}
    if isinstance(raw, dict):
        organizations = raw.get("organizations") or []
        if isinstance(organizations, list) and organizations:
            return [str(value) for value in organizations]
    text = f"{event.get('headline', '')} {event.get('source_name', '')}"
    return [name for name, pattern in ORGANIZATION_PATTERNS if pattern.search(text)]


def saturation_penalty(
    event: dict[str, Any],
    organization_counts: dict[str, int],
) -> tuple[float, dict[str, float]]:
    organizations = event_organizations(event)
    maximum_count = max((organization_counts.get(name, 0) for name in organizations), default=0)
    penalty = min(3.0, maximum_count * 0.75)
    return penalty, {
        "organization_saturation": -penalty,
        "recent_organization_cards": float(maximum_count),
    }
