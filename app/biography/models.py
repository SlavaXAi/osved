from __future__ import annotations

import html
import re
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, model_validator

BODY_MIN_CHARACTERS = 1600
BODY_MAX_CHARACTERS = 2400


class Evidence(BaseModel):
    source_id: int
    fragment_id: int | None = Field(default=None, ge=1)
    quote: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def evidence_reference(self):
        if self.fragment_id is None and len(self.quote) < 20:
            raise ValueError("Evidence needs a source fragment or a verbatim quotation")
        return self


class Fact(BaseModel):
    id: int
    text: str = Field(min_length=15, max_length=600)
    episode_key: str = Field(min_length=3, max_length=100)
    disputed: bool = False
    attribution: str = Field(default="", max_length=120)
    evidence: list[Evidence] = Field(min_length=1, max_length=3)


class Dossier(BaseModel):
    facts: list[Fact] = Field(min_length=6, max_length=16)
    identity_confirmed: bool
    scale_explanation: str = Field(min_length=40)
    subject_deceased_confirmed: bool = False
    death_evidence: list[Evidence] = Field(default_factory=list, max_length=3)


class Image(BaseModel):
    url: str
    page_url: str
    title: str
    description: str
    creator: str
    license: str


class Paragraph(BaseModel):
    text: str = Field(min_length=30)
    fact_ids: list[int] = Field(min_length=1, max_length=8)


class Draft(BaseModel):
    headline: str = Field(min_length=12, max_length=110)
    headline_fact_ids: list[int] = Field(min_length=1, max_length=4)
    paragraphs: list[Paragraph] = Field(min_length=3, max_length=5)
    image_caption: str = Field(min_length=10, max_length=180)
    source_ids: list[int] = Field(min_length=2, max_length=3)

    @property
    def body(self) -> str:
        return "\n\n".join(p.text.strip() for p in self.paragraphs)


class Review(BaseModel):
    supported: bool
    identity_correct: bool
    caption_supported: bool
    no_repeated_episodes: bool
    style_passed: bool
    problems: list[str] = Field(max_length=15)

    @property
    def passed(self) -> bool:
        return (
            all(
                (
                    self.supported,
                    self.identity_correct,
                    self.caption_supported,
                    self.no_repeated_episodes,
                    self.style_passed,
                )
            )
            and not self.problems
        )


def normalized(value: str) -> str:
    value = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", value)
    value = value.replace("**", "").replace("__", "")
    return re.sub(r"\s+", " ", html.unescape(value)).strip().casefold()


def quotation_in_source(quote: str, source: str) -> bool:
    # An exact source slice can start or end inside a Markdown link. Normalizing
    # that slice separately changes its boundary and must not reject verbatim text.
    return len(quote) >= 20 and (quote in source or normalized(quote) in normalized(source))


def dossier_errors(dossier: Dossier, documents: list[dict]) -> list[str]:
    sources = {int(d["id"]): d for d in documents}
    errors = []
    used = set()
    death_publishers = set()
    for evidence in dossier.death_evidence:
        doc = sources.get(evidence.source_id)
        if doc and quotation_in_source(evidence.quote, doc["markdown"]):
            death_publishers.add(doc["publisher"])
        else:
            errors.append("Subject death evidence is not in the source")
    if dossier.subject_deceased_confirmed and len(death_publishers) < 2:
        errors.append("Confirm subject death using two independent publishers")
    ids = [f.id for f in dossier.facts]
    if len(ids) != len(set(ids)) or not dossier.identity_confirmed:
        errors.append("Identity or unique fact IDs are not confirmed")
    for fact in dossier.facts:
        if fact.disputed and not fact.attribution:
            errors.append(f"Disputed fact {fact.id} has no attribution")
        for evidence in fact.evidence:
            doc = sources.get(evidence.source_id)
            if not doc or not quotation_in_source(evidence.quote, doc["markdown"]):
                errors.append(f"Fact {fact.id}: quotation is not in source {evidence.source_id}")
            else:
                used.add(doc["publisher"])
    if len(used) < 2:
        errors.append("Facts need at least two independent publishers")
    return errors


def draft_errors(draft: Draft, dossier: Dossier, documents: list[dict]) -> list[str]:
    errors = []
    if not BODY_MIN_CHARACTERS <= len(draft.body) <= BODY_MAX_CHARACTERS:
        errors.append(
            f"Body has {len(draft.body)} characters; required "
            f"{BODY_MIN_CHARACTERS}–{BODY_MAX_CHARACTERS}"
        )
    facts = {f.id: f for f in dossier.facts}
    selected = {int(d["id"]): d for d in documents if int(d["id"]) in draft.source_ids}
    if (
        len(selected) != len(set(draft.source_ids))
        or len({d["publisher"] for d in selected.values()}) < 2
    ):
        errors.append("Choose 2–3 existing sources from independent publishers")
    cited = set(draft.headline_fact_ids)
    for paragraph in draft.paragraphs:
        cited.update(paragraph.fact_ids)
        # Attribution is checked semantically by the independent reviewer.
        # Several claims may share a natural attribution in one paragraph.
    for fact_id in cited:
        fact = facts.get(fact_id)
        if fact is None or not any(e.source_id in selected for e in fact.evidence):
            errors.append(f"Fact {fact_id} is not covered by the displayed sources")
    if re.search(r"https?://|<[a-zA-Z/]", draft.body + draft.headline):
        errors.append("Prose must be plain text without URLs or markup")
    return errors


def fit_complete_sentences(draft: Draft) -> Draft:
    """Remove whole interior sentences before independent review; never truncate a sentence."""
    value = draft.model_copy(deep=True)
    sentences = [
        re.split(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z«])", paragraph.text.strip())
        for paragraph in value.paragraphs
    ]
    while len(value.body) > BODY_MAX_CHARACTERS:
        candidates = [
            (len(sentence), p, index)
            for p, parts in enumerate(sentences)
            for index, sentence in enumerate(parts)
            if index > 0
            and len(sentence) > 40
            and not (p == len(sentences) - 1 and index == len(parts) - 1)
            and len(value.body) - len(sentence) - 1 >= BODY_MIN_CHARACTERS
        ]
        if not candidates:
            break
        _, p, index = max(candidates)
        sentences[p].pop(index)
        value.paragraphs[p].text = " ".join(sentences[p])
    return value


def render_issue(person_name: str, payload: dict) -> str:
    draft = Draft.model_validate(payload["draft"])
    image = Image.model_validate(payload["image"])
    if urlsplit(image.url).hostname not in {"upload.wikimedia.org", "thumb.wikimedia.org"}:
        raise ValueError("Untrusted illustration URL")
    esc = html.escape
    pieces = [f"<h2>{esc(person_name)} — {esc(draft.headline)}</h2>"]
    pieces.extend(f"<p>{esc(p.text)}</p>" for p in draft.paragraphs)
    pieces.append(f'<img src="{esc(image.url, quote=True)}">')
    pieces.append(
        f'<p>{esc(draft.image_caption)}<br><a href="{esc(image.page_url, quote=True)}">'
        f"{esc(image.creator[:150])} / Wikimedia Commons</a> · {esc(image.license)}</p>"
    )
    links = [
        f'<a href="{esc(s["url"], quote=True)}">{esc(s["title"][:90])}</a>'
        for s in payload["sources"]
    ]
    pieces.append("<p>Источники: " + " · ".join(links) + "</p>")
    return "".join(pieces)
