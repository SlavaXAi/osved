from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from app.config import Settings
from app.db.repository import Repository
from app.pipeline.cards import generate_cards
from app.pipeline.ranking import rank_event_candidates
from app.research.facts import build_event_fact_sheet
from app.research.web import ResearchEventService


@dataclass
class CalibrationStats:
    target: int
    events_seen: int = 0
    cards_created: int = 0
    research_errors: int = 0
    fact_sheet_errors: int = 0
    generation_errors: int = 0
    cost_usd: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


ProgressCallback = Callable[[dict[str, Any]], None]


def balanced_research_events(
    repository: Repository,
    *,
    limit: int,
    primary_only: bool = False,
) -> list[dict[str, Any]]:
    candidates = repository.events_needing_research(limit=max(limit * 5, limit))
    if primary_only:
        candidates = [event for event in candidates if event.get("source_role") == "primary"]
    candidates = rank_event_candidates(
        candidates,
        recent_events=repository.recent_valid_card_events(days=7),
        limit=len(candidates),
    )
    buckets: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    source_order: list[str] = []
    for event in candidates:
        trigger_sources = repository.event_trigger_sources(int(event["id"]))
        source_id = str(trigger_sources[0]["source_id"]) if trigger_sources else "unknown"
        if source_id not in buckets:
            source_order.append(source_id)
        buckets[source_id].append(event)

    balanced: list[dict[str, Any]] = []
    while len(balanced) < limit:
        added = False
        for source_id in source_order:
            if not buckets[source_id]:
                continue
            balanced.append(buckets[source_id].popleft())
            added = True
            if len(balanced) >= limit:
                break
        if not added:
            break
    return balanced


def run_calibration_batch(
    settings: Settings,
    repository: Repository,
    *,
    target: int,
    candidate_limit: int,
    max_pages: int = 2,
    primary_only: bool = False,
    progress: ProgressCallback | None = None,
) -> CalibrationStats:
    stats = CalibrationStats(target=target)
    events = balanced_research_events(
        repository,
        limit=candidate_limit,
        primary_only=primary_only,
    )
    service = ResearchEventService(settings, repository)

    for event in events:
        if stats.cards_created >= target:
            break
        event_id = int(event["id"])
        stats.events_seen += 1
        step: dict[str, Any] = {
            "event_id": event_id,
            "headline": str(event.get("headline") or "")[:140],
            "position": stats.events_seen,
            "target": target,
        }
        try:
            cached_documents = repository.event_research_documents(event_id)
            has_primary_content = any(
                document.get("source_class") == "primary"
                and len(str(document.get("content_text") or "")) >= 500
                for document in cached_documents
            )
            if has_primary_content:
                step["research"] = "cached"
            else:
                research = service.research(
                    event_id,
                    query=str(event.get("headline") or ""),
                    primary_url=(
                        str(event["source_url"]) if event.get("source_role") == "primary" else None
                    ),
                    freshness="last_month",
                    max_pages=max_pages,
                )
                stats.cost_usd += research.cost_usd
                step["research"] = "ok"
        except Exception as exc:
            stats.research_errors += 1
            step["research"] = "error"
            step["error"] = str(exc)[:300]
            if progress:
                progress({**step, "totals": stats.as_dict()})
            continue

        try:
            fact_sheet = build_event_fact_sheet(settings, repository, event_id)
            stats.cost_usd += float(fact_sheet.get("cost_usd") or 0)
            step["fact_sheet"] = str(fact_sheet.get("status") or "unknown")
            step["dossier"] = str((fact_sheet.get("dossier") or {}).get("decision") or "none")
            if fact_sheet.get("status") != "valid":
                stats.fact_sheet_errors += 1
                if progress:
                    progress({**step, "totals": stats.as_dict()})
                continue
        except Exception as exc:
            stats.fact_sheet_errors += 1
            step["fact_sheet"] = "error"
            step["error"] = str(exc)[:300]
            if progress:
                progress({**step, "totals": stats.as_dict()})
            continue

        generated = generate_cards(settings, repository, limit=1, event_id=event_id)
        stats.cost_usd += generated.cost_usd
        stats.cards_created += generated.cards_created
        stats.generation_errors += generated.errors
        step["card"] = "created" if generated.cards_created else "rejected"
        if progress:
            progress({**step, "totals": stats.as_dict()})

    return stats
