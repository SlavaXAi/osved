"""Bind model-selected evidence to verbatim, reproducible source fragments."""

from __future__ import annotations

from app.biography.models import Dossier


def source_fragments(document: dict) -> list[dict]:
    text = document["markdown"]
    fragments = []
    start = 0
    while start < len(text):
        end = min(start + 900, len(text))
        if end < len(text):
            boundaries = [
                text.rfind(separator, start + 300, end) + len(separator)
                for separator in ("\n\n", "\n", ". ", " ")
            ]
            end = max([start + 300, *boundaries])
        quote = text[start:end].strip()
        if len(quote) >= 20:
            fragments.append({"id": len(fragments) + 1, "quote": quote, "start": start})
        elif fragments:
            # The final short tail belongs to the preceding source fragment.
            previous = fragments[-1]
            combined = text[previous["start"] : end].strip()
            if len(combined) <= 1000:
                previous["quote"] = combined
        start = end
    return [{"id": f["id"], "quote": f["quote"]} for f in fragments]


def evidence_documents(documents: list[dict]) -> list[dict]:
    return [
        {
            **{key: doc[key] for key in ("id", "url", "publisher", "title")},
            "fragments": source_fragments(doc),
        }
        for doc in documents
    ]


def bind_dossier(dossier: Dossier, documents: list[dict]) -> Dossier:
    result = dossier.model_copy(deep=True)
    fragments = {
        doc["id"]: {f["id"]: f["quote"] for f in source_fragments(doc)} for doc in documents
    }
    evidence = [item for fact in result.facts for item in fact.evidence] + result.death_evidence
    for item in evidence:
        if item.fragment_id is None:
            continue  # Existing dossiers and exact manually cited evidence remain valid.
        quote = fragments.get(item.source_id, {}).get(item.fragment_id)
        if quote is None:
            raise ValueError(f"Unknown source fragment {item.source_id}:{item.fragment_id}")
        item.quote = quote
    return result
