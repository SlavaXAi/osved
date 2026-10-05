from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, cast

from app.config import (
    Settings,
    load_contour_registry,
    load_interest_profile,
    load_weekly_persona_registry,
)
from app.contours.classifier import classify_event
from app.db.repository import Repository
from app.intelligence.models import IntelligenceProductDraft
from app.intelligence.products import (
    build_monthly_theses,
    build_project_signals,
    build_weekly_map,
    build_weekly_persona_map,
)

IntelligenceKind = Literal[
    "weekly_map",
    "monthly_theses",
    "weekly_personas",
    "project_signals",
]
INTELLIGENCE_KINDS: tuple[IntelligenceKind, ...] = (
    "weekly_map",
    "monthly_theses",
    "weekly_personas",
    "project_signals",
)


def normalize_now(value: datetime | None = None) -> datetime:
    result = value or datetime.now(UTC)
    if result.tzinfo is None:
        result = result.replace(tzinfo=UTC)
    return result.astimezone(UTC)


def parse_intelligence_kind(value: str) -> IntelligenceKind:
    if value not in INTELLIGENCE_KINDS:
        allowed = ", ".join(INTELLIGENCE_KINDS)
        raise ValueError(f"Unsupported intelligence kind {value!r}; choose one of: {allowed}")
    return cast(IntelligenceKind, value)


@dataclass(frozen=True)
class ContourBackfillResult:
    examined: int
    updated: int
    unchanged: int
    classified: int
    unclassified: int
    classifier_version: str = "contours_v1"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class IntelligenceService:
    """Build cached OSVED intelligence products from already stored events only.

    This service deliberately has no collector, browser or LLM dependency. It is a
    deterministic derived view over the factual event corpus and can therefore be
    rebuilt, inspected and tested without spending money.
    """

    def __init__(self, settings: Settings, repository: Repository) -> None:
        self.settings = settings
        self.repository = repository

    def backfill_contours(self, *, limit: int = 10_000) -> ContourBackfillResult:
        registry = load_contour_registry(self.settings.contours_file)
        events = self.repository.events_for_contour_backfill(limit=limit)
        updated = 0
        classified = 0
        for event in events:
            assignment = classify_event(event, registry)
            if assignment.primary_contour is not None:
                classified += 1
            # Persisting the classifier version and an explicit absent contour
            # prevents an ambiguous "not processed" state.
            if self.repository.update_event_contour_assignment(
                int(event["id"]),
                assignment.as_metadata(),
            ):
                updated += 1
        examined = len(events)
        return ContourBackfillResult(
            examined=examined,
            updated=updated,
            unchanged=examined - updated,
            classified=classified,
            unclassified=examined - classified,
        )

    def _events_for_products(self, *, now: datetime) -> list[dict[str, Any]]:
        # The longest product compares two 14-day windows. Seven extra days make
        # timestamp boundaries and late-ingested source material harmless.
        since = now - timedelta(days=35)
        return self.repository.intelligence_events(
            since=since.isoformat(),
            until=now.isoformat(),
        )

    def _build_draft(
        self,
        kind: IntelligenceKind,
        events: list[dict[str, Any]],
        *,
        now: datetime,
    ) -> tuple[IntelligenceProductDraft, dict[str, Any]]:
        if kind == "weekly_map":
            registry = load_contour_registry(self.settings.contours_file)
            return build_weekly_map(events, registry, now=now), {
                "contours": registry.model_dump(mode="json")
            }
        if kind == "monthly_theses":
            registry = load_contour_registry(self.settings.contours_file)
            return build_monthly_theses(events, registry, now=now), {
                "contours": registry.model_dump(mode="json")
            }
        if kind == "weekly_personas":
            personas = load_weekly_persona_registry(self.settings.weekly_personas_file)
            return build_weekly_persona_map(events, personas, now=now), {
                "personas": personas.model_dump(mode="json")
            }
        profile = load_interest_profile(self.settings.interests_file)
        return build_project_signals(events, profile, now=now), {
            "interest_profile": profile.model_dump(mode="json")
        }

    @staticmethod
    def _validate_draft(
        draft: IntelligenceProductDraft,
        *,
        known_event_ids: set[int],
    ) -> tuple[str, list[str]]:
        errors: list[str] = []
        if not draft.rendered_html.strip():
            errors.append("empty_rendered_html")
        unknown_ids = sorted(set(draft.source_event_ids) - known_event_ids)
        if unknown_ids:
            errors.append(f"unknown_source_event_ids:{','.join(map(str, unknown_ids))}")
        if len(draft.source_event_ids) != len(set(draft.source_event_ids)):
            errors.append("duplicate_source_event_ids")
        if errors:
            return "invalid", errors
        if not draft.sections or not draft.source_event_ids:
            return "insufficient_data", ["no_material_signal_in_period"]
        return "valid", []

    @staticmethod
    def _generation_key(
        draft: IntelligenceProductDraft,
        *,
        now: datetime,
        config_snapshot: dict[str, Any],
    ) -> str:
        # Exact clock time is intentionally absent. Re-running the same product on
        # the same UTC day with identical evidence is a $0 cache hit.
        material = {
            "kind": draft.kind,
            "schema_version": draft.schema_version,
            "utc_day": now.date().isoformat(),
            "title": draft.title,
            "summary": draft.summary,
            "sections": [section.model_dump(mode="json") for section in draft.sections],
            "source_event_ids": draft.source_event_ids,
            "config": config_snapshot,
        }
        digest = hashlib.sha256(
            json.dumps(material, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return f"{draft.kind}:{draft.schema_version}:{digest}"

    def build(
        self,
        kind: str,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        parsed_kind = parse_intelligence_kind(kind)
        effective_now = normalize_now(now)
        events = self._events_for_products(now=effective_now)
        draft, config_snapshot = self._build_draft(
            parsed_kind,
            events,
            now=effective_now,
        )
        status, errors = self._validate_draft(
            draft,
            known_event_ids={int(event["id"]) for event in events},
        )
        generation_key = self._generation_key(
            draft,
            now=effective_now,
            config_snapshot=config_snapshot,
        )
        payload = draft.model_dump(mode="json")
        payload["metadata"] = {
            **payload.get("metadata", {}),
            "generated_without_network": True,
            "generated_without_llm": True,
            "corpus_event_count": len(events),
        }
        return self.repository.save_intelligence_product(
            kind=draft.kind,
            generation_key=generation_key,
            schema_version=draft.schema_version,
            period_start=draft.period_start.isoformat(),
            period_end=draft.period_end.isoformat(),
            title=draft.title,
            summary=draft.summary,
            payload=payload,
            rendered_html=draft.rendered_html,
            source_event_ids=draft.source_event_ids,
            validation_status=status,
            validation_errors=errors,
        )

    def show(self, product_id: int) -> dict[str, Any] | None:
        return self.repository.get_intelligence_product(product_id)

    def latest(self, kind: str) -> dict[str, Any] | None:
        return self.repository.latest_intelligence_product(parse_intelligence_kind(kind))

    def status(self) -> dict[str, Any]:
        result = self.repository.intelligence_status()
        result["supported_kinds"] = list(INTELLIGENCE_KINDS)
        result["network_requests"] = 0
        result["llm_requests"] = 0
        return result
