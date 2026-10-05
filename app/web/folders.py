"""Conservative display-only fallback for historical cards without contour metadata.

These rules organize a reading interface. They never change ranking, research,
personal preferences, historical assignments, or source provenance.
"""

from __future__ import annotations

import re

from app.web.content import plain_text

# Broad domains, not a claim that a company is always relevant to one domain.
# Headline matches have more weight than background mentions in a long article.
PATTERNS = {
    "ai_software": (
        r"\b(?:openai|anthropic|deepmind|chatgpt|claude|gemini|gpt[-‑\s]?\d|llm)\b",
        r"искусственн\w* интеллект|нейросет|\bии\b|\bai\b",
        r"агент\w*|\b(?:agent|inference|benchmark|coding|software)\b",
        r"кибербезопас|фишинг|вредонос|киберинцидент|\b(?:malware|cloudflare|cybersecurity)\b",
        r"программ\w* обеспечен|\b(?:github|api|developer|open.source)\b",
    ),
    "startups": (
        r"стартап|\bstartups?\b|y combinator|\byc\b|\ba16z\b|andreessen horowitz",
        r"венчур|раунд\w* финанс|привл[её]к\w* инвести|"
        r"\b(?:seed round|series [a-d]|venture capital)\b",
        r"бизнес.модел|\b(?:product.market.fit|founders?)\b",
    ),
    "political_economy": (
        r"санкци|пошлин|таможенн|\b(?:sanctions?|tariffs?|customs)\b",
        r"миров\w* экономик|международн\w* торгов|федеральн\w* резерв|\bфрс\b|дедолларизац",
        r"военн\w* конфликт|боев\w* действ|войн|\b(?:war|nato|brics|geopolitics)\b|нато|брикс",
        r"экспортн\w* огранич|импортн\w* огранич|торгов\w* соглашен|export controls|trade deal",
        r"инфляци|центробанк|центральн\w* банк|ставк\w* фрс|federal reserve|interest rates",
    ),
    "space": (
        r"космич|космос|орбит|космонавт|\b(?:spaceflight|spacecraft|orbital|astronaut)\b",
        r"\b(?:nasa|roscosmos|starship|falcon 9|lunar)\b|роскосмос|ракет|спутник",
        r"лунн\w*|посадк\w* на луну|марсиан|\bmars\b",
    ),
    "human_civilization": (
        r"сознани|философ|цивилизац|трансгуман|\b(?:consciousness|philosophy|transhuman)\b",
        r"цифров\w* страдан|бессмерти|эмуляци\w* мозга|ai welfare|brain emulation",
        r"парадокс ферми|\b(?:fermi paradox|acausal|longevity)\b|долголет",
    ),
}
COMPILED = {
    folder: tuple(re.compile(pattern, re.I) for pattern in patterns)
    for folder, patterns in PATTERNS.items()
}


def infer_reading_folder(row: dict) -> str | None:
    title = plain_text(row.get("headline") or row.get("event_headline") or "")
    # Only the publication itself, not a source's long unrelated archive/description.
    body = plain_text(row.get("rendered_html") or row.get("description_html") or row.get("summary"))
    scores = {}
    for folder, patterns in COMPILED.items():
        title_hits = sum(bool(pattern.search(title)) for pattern in patterns)
        body_hits = sum(bool(pattern.search(body)) for pattern in patterns)
        score = 3 * title_hits + body_hits
        if score >= 3:
            scores[folder] = score
    ordered = sorted(scores, key=lambda folder: (-scores[folder], folder))
    if not ordered:
        return None
    if len(ordered) > 1 and scores[ordered[0]] == scores[ordered[1]]:
        return None  # Ambiguous material stays in All/Other, never arbitrarily assigned.
    return ordered[0]
