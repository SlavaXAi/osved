from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.db.repository import Repository
from app.research.facts import FactConflict

DOSSIER_SCHEMA_VERSION = "event_dossier_v2"


class TriggerSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_item_id: int
    source_id: str
    source_name: str
    source_kind: str
    source_role: str
    evidence_role: str
    admission_mode: str | None = None
    source_url: str
    published_at: str


class EvidenceDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: int
    source_class: str
    publisher: str | None
    title: str | None
    url: str
    published_at: str | None


class DossierCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    present: list[str]
    required: list[str]
    missing_required: list[str]
    missing_optional: list[str]


class DossierNovelty(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["new_event", "dossier_updated"]
    trigger_count: int = Field(ge=1)
    first_seen_at: str
    latest_source_at: str


class EventDossier(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: int
    canonical_event: str
    event_type: str
    event_at: str | None
    entities: dict[str, Any]
    topics: list[Any]
    regions: list[Any]
    triggers: list[TriggerSource] = Field(min_length=1)
    evidence_documents: list[EvidenceDocument] = Field(min_length=1)
    coverage: DossierCoverage
    conflicts: list[FactConflict]
    novelty: DossierNovelty
    decision: Literal["ready", "needs_research", "blocked_conflict"]
    decision_reason: str


ALL_COVERAGE_FIELDS = (
    "event_identity",
    "capabilities",
    "benchmarks",
    "pricing",
    "availability",
    "limitations",
    "independent_confirmation",
)

REQUIRED_COVERAGE = {
    "model_release": (
        "event_identity",
        "benchmarks",
        "availability",
    ),
    "software_release": ("event_identity", "capabilities"),
    "company_event": ("event_identity",),
    "research_result": ("event_identity",),
    "public_statement": ("event_identity",),
    "other": ("event_identity",),
}


def parse_json_value(value: Any, fallback: Any) -> Any:
    if value is None:
        return fallback
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return fallback
    return value


def canonical_event_text(event: dict[str, Any], facts: list[dict[str, Any]]) -> str:
    release = next(
        (str(fact["text"]) for fact in facts if fact.get("kind") == "release"),
        None,
    )
    return release or str(event.get("headline") or "Технологическое событие")


def build_event_dossier(repository: Repository, event_id: int) -> dict[str, Any]:
    event = repository.event_context(event_id)
    if event is None:
        raise ValueError(f"Unknown event_id: {event_id}")
    fact_sheet = repository.latest_valid_fact_sheet(event_id)
    if fact_sheet is None:
        raise ValueError("No valid fact sheet; build-fact-sheet must succeed first")

    triggers = [
        TriggerSource.model_validate(item) for item in repository.event_trigger_sources(event_id)
    ]
    documents = repository.event_research_documents(event_id)
    evidence_documents = [
        EvidenceDocument(
            document_id=int(document["id"]),
            source_class=str(document["source_class"]),
            publisher=str(document["publisher"]) if document.get("publisher") else None,
            title=str(document["title"]) if document.get("title") else None,
            url=str(document["url"]),
            published_at=(str(document["published_at"]) if document.get("published_at") else None),
        )
        for document in documents
        if document.get("source_class") != "radar_only"
    ]
    if not evidence_documents:
        raise ValueError("No usable evidence documents")

    coverage_payload = dict(fact_sheet["coverage"])
    event_type = str(coverage_payload.get("event_type") or event.get("event_type") or "other")
    if event_type not in REQUIRED_COVERAGE:
        event_type = "other"
    present = [field for field in ALL_COVERAGE_FIELDS if coverage_payload.get(field) is True]
    required = list(REQUIRED_COVERAGE[event_type])
    missing_required = [field for field in required if field not in present]
    missing_optional = [
        field for field in ALL_COVERAGE_FIELDS if field not in required and field not in present
    ]
    conflicts = [
        FactConflict.model_validate(conflict) for conflict in coverage_payload.get("conflicts", [])
    ]
    unresolved_material = [
        conflict
        for conflict in conflicts
        if conflict.severity == "material" and conflict.resolution == "unresolved"
    ]
    has_primary = any(document.source_class == "primary" for document in evidence_documents)
    if not has_primary:
        decision = "needs_research"
        decision_reason = "missing primary evidence document"
    elif missing_required:
        decision = "needs_research"
        decision_reason = "missing required coverage: " + ", ".join(missing_required)
    elif unresolved_material:
        decision = "blocked_conflict"
        decision_reason = "unresolved material conflict: " + ", ".join(
            conflict.subject for conflict in unresolved_material
        )
    else:
        decision = "ready"
        decision_reason = "required evidence coverage is complete"

    previous = repository.latest_event_dossier(event_id)
    dossier = EventDossier(
        event_id=event_id,
        canonical_event=canonical_event_text(event, fact_sheet["facts"]),
        event_type=event_type,
        event_at=str(event["event_at"]) if event.get("event_at") else None,
        entities=parse_json_value(event.get("entities_json"), {}),
        topics=parse_json_value(event.get("topics_json"), []),
        regions=parse_json_value(event.get("regions_json"), []),
        triggers=triggers,
        evidence_documents=evidence_documents,
        coverage=DossierCoverage(
            present=present,
            required=required,
            missing_required=missing_required,
            missing_optional=missing_optional,
        ),
        conflicts=conflicts,
        novelty=DossierNovelty(
            status="dossier_updated" if previous else "new_event",
            trigger_count=len(triggers),
            first_seen_at=str(event["first_seen_at"]),
            latest_source_at=str(event["latest_source_at"]),
        ),
        decision=decision,
        decision_reason=decision_reason,
    )
    dossier_id = repository.create_event_dossier(
        event_id=event_id,
        schema_version=DOSSIER_SCHEMA_VERSION,
        fact_sheet_id=int(fact_sheet["id"]),
        decision=decision,
        decision_reason=decision_reason,
        payload=dossier.model_dump(),
    )
    current_status = str(event["status"])
    if current_status == "candidate" and repository.event_has_valid_card(event_id):
        repository.set_event_status(event_id, "card_ready", decision_reason)
    elif current_status in {"candidate", "research_insufficient", "generation_error"}:
        repository.set_event_status(
            event_id,
            "candidate" if decision == "ready" else "research_insufficient",
            decision_reason,
        )
    return {
        "dossier_id": dossier_id,
        "event_id": event_id,
        "decision": decision,
        "decision_reason": decision_reason,
        "missing_required": missing_required,
        "conflict_count": len(conflicts),
    }
