from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from app.collectors.base import CollectionBatch, CollectionRequest, SourceAdapter
from app.collectors.ingest import ingest_item, redact_text
from app.collectors.registry import adapter_id_for, build_adapter_registry
from app.collectors.state import claim_source, finish_source
from app.config import Settings, runtime_source_registry
from app.db.repository import Repository


def collect_due_sources(
    settings: Settings,
    repository: Repository,
    *,
    source_ids: list[str] | None = None,
    kind: str | None = None,
    provider: str = "auto",
    limit_per_source: int = 10,
    force: bool = False,
    dry_run: bool = False,
    adapters: dict[str, SourceAdapter] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = now or datetime.now(UTC)
    registry = runtime_source_registry(settings)
    repository.sync_sources(registry)
    available = adapters if adapters is not None else build_adapter_registry(settings, repository)
    rows = []
    totals = {
        "sources_attempted": 0,
        "sources_succeeded": 0,
        "items_seen": 0,
        "items_inserted": 0,
        "errors": 0,
        "cost_usd": 0.0,
    }
    for source in registry.sources:
        # A large run may take minutes. Each source has its real poll time;
        # otherwise a legitimately just-published post could look "from the future".
        current = now or datetime.now(UTC)
        override = repository.get_runtime_setting(f"source_enabled:{source.id}")
        enabled = override == "true" if override is not None else source.enabled
        if not enabled or source.admission_mode == "history" or source.options.get("removed"):
            continue
        if source_ids is not None and source.id not in source_ids:
            continue
        if kind not in (None, "all") and source.kind.value != kind:
            continue
        adapter_id = adapter_id_for(source, settings, provider=provider)
        adapter = available.get(adapter_id)
        if adapter is None:
            if source.kind.value == "web":
                continue  # Virtual search lenses are driven by the bounded search runner.
            rows.append({"source_id": source.id, "status": "unsupported", "adapter": adapter_id})
            continue
        if dry_run:
            rows.append(
                {
                    "source_id": source.id,
                    "status": "planned",
                    "adapter": adapter_id,
                    "estimated_cost_usd": adapter.capabilities().estimated_cost_usd,
                }
            )
            continue
        estimated = adapter.capabilities().estimated_cost_usd
        run_id, reason = claim_source(
            repository, settings, source, adapter_id, estimated, now=current, force=force
        )
        if run_id is None:
            rows.append({"source_id": source.id, "status": reason})
            continue
        totals["sources_attempted"] += 1
        started = time.monotonic()
        called = False
        try:
            health = adapter.healthcheck(source)
            if health.status != "ok":
                batch = CollectionBatch(status=health.status, error=health.error)
            else:
                called = True
                batch = adapter.collect(
                    CollectionRequest(
                        source,
                        # Public YouTube RSS is a small upload snapshot. Do not
                        # lose the tail behind the generic per-source sample.
                        limit=100
                        if source.kind.value == "youtube"
                        else max(1, min(limit_per_source, 100)),
                        cursor=repository.get_source_cursor(source.id),
                        now=current,
                    )
                )
            inserted = rejected = ingest_limit = 0
            if batch.status in {"ok", "partial", "empty"}:
                ingest_limit = (
                    100
                    if batch.metadata.get("complete_snapshot") or source.kind.value == "youtube"
                    else 80
                    if batch.metadata.get("ack_required")
                    else max(1, min(limit_per_source, 100))
                )
                for item in batch.items[:ingest_limit]:
                    try:
                        not_before = source.options.get("not_before")
                        if not_before and (
                            not item.published_at
                            or datetime.fromisoformat(item.published_at.replace("Z", "+00:00"))
                            < datetime.fromisoformat(str(not_before))
                        ):
                            continue  # No archive flood when a channel is first followed.
                        inserted += int(
                            ingest_item(
                                repository,
                                source,
                                item,
                                now=current,
                                max_age_days=source.production_max_age_days
                                if settings.app_env == "production"
                                else source.calibration_lookback_days,
                            )
                        )
                    except (ValueError, TypeError, OverflowError):
                        rejected += 1
            status = batch.status
            if rejected:
                status = "partial" if inserted or rejected < len(batch.items) else "filtered"
            acknowledge = getattr(adapter, "acknowledge", None)
            if callable(acknowledge) and batch.metadata.get("ack_required"):
                acknowledge(batch)
            finish_source(
                repository,
                run_id,
                status=status,
                items_seen=len(batch.items),
                items_inserted=inserted,
                items_rejected=rejected,
                cursor=batch.next_cursor if len(batch.items) <= ingest_limit else None,
                cost_usd=batch.cost_usd,
                error=batch.error,
                latency_ms=int((time.monotonic() - started) * 1000),
                now=current,
            )
            if not batch.metadata.get("usage_recorded"):
                repository.record_usage(
                    adapter_id,
                    "sensor.collect",
                    cost_usd=batch.cost_usd,
                    metadata={"source_id": source.id, "source_run_id": run_id, "status": status},
                )
            totals["sources_succeeded"] += int(
                status in {"ok", "empty", "partial", "pending", "filtered"}
            )
            totals["errors"] += int(status not in {"ok", "empty", "partial", "pending", "filtered"})
            totals["items_seen"] += len(batch.items)
            totals["items_inserted"] += inserted
            totals["cost_usd"] += batch.cost_usd
            rows.append(
                {
                    "source_id": source.id,
                    "run_id": run_id,
                    "status": status,
                    "seen": len(batch.items),
                    "inserted": inserted,
                    "rejected": rejected,
                    "cost_usd": batch.cost_usd,
                    "error": redact_text(batch.error or "")[:400],
                }
            )
        except Exception as exc:
            # A timed-out paid request may already have incurred its per-call price.
            cost = estimated if called else 0
            error = redact_text(str(exc))[:1000]
            finish_source(
                repository, run_id, status="failed", error=error, cost_usd=cost, now=current
            )
            if cost:
                repository.record_usage(
                    adapter_id,
                    "sensor.failed",
                    cost_usd=cost,
                    metadata={"source_id": source.id, "source_run_id": run_id, "estimated": True},
                )
            totals["errors"] += 1
            totals["cost_usd"] += cost
            rows.append(
                {"source_id": source.id, "run_id": run_id, "status": "failed", "error": error}
            )
    return {**totals, "sources": rows, "dry_run": dry_run}
