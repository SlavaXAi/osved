"""Explicit offline acceptance of saved Reddit/Weibo responses; never calls a network.

This is an operator tool, not a scheduled collector. Replay runs have a distinct
strategy/status and zero request count. A known historical charge can be imported
once, separately from the replay, without presenting it as a new provider call.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from app.collectors.base import CollectionRequest
from app.collectors.ingest import ingest_item, redact_text
from app.collectors.platform_reddit_monid import RedditMonidAdapter
from app.collectors.platform_transport import MonidResponse
from app.collectors.platforms import WeiboSearchAdapter
from app.config import SourceConfig
from app.db.repository import Repository

REPLAY_VERSION = "2026-09-07-observed-app-layouts-v1"


def replay_snapshot(
    repository: Repository,
    source: SourceConfig,
    snapshot_path: Path,
    *,
    original_run_id: str,
    import_charge_usd: float = 0,
    import_operation: str = "live_smoke.reddit",
    limit: int = 20,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Normalize and ingest one explicitly selected saved response, idempotently.

    Source rows must already exist. No adapter factory, live transport, source
    toggles, news generation, or delivery is invoked. The fake transport returns
    only this snapshot and always reports zero incremental cost.
    """
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        raise ValueError("Replay timestamp must include timezone")
    if source.adapter_id not in {"reddit_monid", "weibo_search_monid"}:
        raise ValueError("Offline replay supports only explicit Reddit/Weibo adapters")
    if not original_run_id or len(original_run_id) > 200:
        raise ValueError("The original run identity is required")
    if not math.isfinite(import_charge_usd) or import_charge_usd < 0:
        raise ValueError("A reconciled historical cost must be finite and nonnegative")
    path = Path(snapshot_path)
    if not path.is_file() or path.stat().st_size > 4_000_000:
        raise ValueError("Snapshot must be an existing JSON file no larger than 4 MB")
    raw = path.read_bytes()
    payload = json.loads(raw)
    fingerprint = hashlib.sha256(raw).hexdigest()
    run_id = (
        "replay-"
        + hashlib.sha256(
            f"{source.id}:{source.adapter_id}:{fingerprint}:{REPLAY_VERSION}".encode()
        ).hexdigest()[:40]
    )
    stamp = current.isoformat()
    accounting = 0.0
    metadata = {
        "source_id": source.id,
        "source_run_id": run_id,
        "original_run_id": original_run_id,
        "snapshot_sha256": fingerprint,
        "snapshot_file": path.name,
        "parser_version": REPLAY_VERSION,
        "network_calls": 0,
        "replayed": True,
    }
    with repository.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        if not db.execute("SELECT 1 FROM sources WHERE id=?", (source.id,)).fetchone():
            raise ValueError("Sync the exact source registry before replay")
        if import_charge_usd:
            existing_charge = db.execute(
                """SELECT 1 FROM usage_ledger WHERE cost_usd > 0 AND (
                json_extract(metadata_json,'$.original_run_id')=? OR
                json_extract(metadata_json,'$.source_run_id')=?) LIMIT 1""",
                (original_run_id, original_run_id),
            ).fetchone()
            if not existing_charge:
                db.execute(
                    """INSERT INTO usage_ledger(provider,operation,request_count,cost_usd,
                    metadata_json) VALUES(?,?,?,?,?)""",
                    (
                        source.adapter_id,
                        import_operation,
                        1,
                        import_charge_usd,
                        json.dumps(
                            {
                                **metadata,
                                "source_run_id": original_run_id,
                                "historical_charge_import": True,
                                "new_request": False,
                            }
                        ),
                    ),
                )
                accounting = import_charge_usd
        existing = db.execute("SELECT * FROM source_runs WHERE id=?", (run_id,)).fetchone()
        if existing and existing["status"] == "replayed":
            return {
                **metadata,
                "status": "already_replayed",
                "seen": existing["items_seen"],
                "inserted": 0,
                "original_inserted": existing["items_inserted"],
                "cost_usd": 0,
                "accounting_adjustment_usd": accounting,
            }
        active = db.execute(
            "SELECT 1 FROM source_runs WHERE source_id=? AND status='running' AND lease_until> ?",
            (source.id, stamp),
        ).fetchone()
        if active:
            return {
                **metadata,
                "status": "source_busy",
                "inserted": 0,
                "cost_usd": 0,
                "accounting_adjustment_usd": accounting,
            }
        db.execute(
            """INSERT INTO source_runs(id,source_id,adapter_id,strategy,status,reserved_usd,
            started_at,lease_until) VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
            status='running',started_at=excluded.started_at,lease_until=excluded.lease_until""",
            (
                run_id,
                source.id,
                "replay:" + source.adapter_id,
                "snapshot_replay",
                "running",
                0,
                stamp,
                (current + timedelta(minutes=5)).isoformat(),
            ),
        )

    started = time.monotonic()
    inserted = rejected = seen = 0
    error = None
    try:
        with tempfile.TemporaryDirectory(prefix="pars-snapshot-replay-") as temporary:

            def saved_response(endpoint: str, query: dict[str, Any]) -> MonidResponse:
                del endpoint, query
                return MonidResponse(payload, cost_usd=0, run_id=original_run_id)

            adapter = (
                RedditMonidAdapter(Path(temporary), fetcher=saved_response)
                if source.adapter_id == "reddit_monid"
                else WeiboSearchAdapter(Path(temporary), saved_response)
            )
            batch = adapter.collect(
                CollectionRequest(source, limit=max(1, min(limit, 100)), now=current)
            )
        seen = len(batch.items)
        if batch.cost_usd != 0:
            raise ValueError("Offline replay cannot report a new provider charge")
        if batch.status not in {"ok", "partial", "empty"}:
            raise ValueError(f"Saved response rejected: {batch.status}: {batch.error or ''}")
        for item in batch.items:
            observed = replace(item, raw_payload={**item.raw_payload, "replay": metadata})
            try:
                inserted += int(
                    ingest_item(
                        repository,
                        source,
                        observed,
                        now=current,
                        max_age_days=source.production_max_age_days,
                    )
                )
            except (ValueError, TypeError, OverflowError):
                rejected += 1
        status = "replayed" if not rejected else "replay_partial"
    except (ValueError, TypeError, KeyError, OSError) as exc:
        status, error = "replay_failed", redact_text(str(exc))[:600]
    with repository.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        db.execute(
            """UPDATE source_runs SET status=?,items_seen=?,items_inserted=?,items_rejected=?,
            cost_usd=0,reserved_usd=0,error=?,latency_ms=?,finished_at=? WHERE id=?""",
            (
                status,
                seen,
                inserted,
                rejected,
                error,
                int((time.monotonic() - started) * 1000),
                stamp,
                run_id,
            ),
        )
        db.execute(
            """INSERT INTO usage_ledger(provider,operation,request_count,cost_usd,metadata_json)
            VALUES(?, 'sensor.snapshot_replay', 0, 0, ?)""",
            (source.adapter_id, json.dumps({**metadata, "status": status})),
        )
        if status == "replayed":
            # This certifies the parser only; do not invent a new network attempt,
            # update live pagination, enable a source, or advance last_success_at.
            db.execute(
                """INSERT INTO source_health(source_id,status,consecutive_errors,last_productive_at)
                VALUES(?, 'ready_from_replay', 0, ?) ON CONFLICT(source_id) DO UPDATE SET
                status='ready_from_replay',consecutive_errors=0,cooldown_until=NULL,
                last_error=NULL,last_productive_at=COALESCE(excluded.last_productive_at,
                source_health.last_productive_at)""",
                (source.id, stamp if inserted else None),
            )
            db.execute("UPDATE sources SET last_error=NULL WHERE id=?", (source.id,))
    return {
        **metadata,
        "status": status,
        "seen": seen,
        "inserted": inserted,
        "rejected": rejected,
        "cost_usd": 0,
        "accounting_adjustment_usd": accounting,
        "error": error,
    }
