"""Durable cadence, reservations and circuit breakers shared by every sensor."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from app.collectors.ingest import redact_text
from app.config import Settings, SourceConfig
from app.db.repository import Repository


def claim_source(
    repository: Repository,
    settings: Settings,
    source: SourceConfig,
    adapter_id: str,
    estimated_cost: float,
    *,
    now: datetime,
    force: bool = False,
) -> tuple[str | None, str]:
    stamp = now.isoformat()
    with repository.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        # Recover the estimated charge into the common ledger exactly once as well.
        db.execute(
            """INSERT INTO usage_ledger(provider,operation,cost_usd,metadata_json)
            SELECT adapter_id,'sensor.abandoned',MAX(cost_usd,reserved_usd),
                   json_object('source_id',source_id,'source_run_id',id,'estimated',1)
            FROM source_runs WHERE status='running' AND lease_until < ?
              AND MAX(cost_usd,reserved_usd)>0""",
            (stamp,),
        )
        # A crashed request may have been charged. Keep its reservation as a conservative cost.
        db.execute(
            """UPDATE source_runs SET status='abandoned', finished_at=?,
            cost_usd=MAX(cost_usd,reserved_usd), reserved_usd=0,
            error='Worker lease expired; cost reserved conservatively'
            WHERE status='running' AND lease_until < ?""",
            (stamp, stamp),
        )
        if db.execute(
            "SELECT 1 FROM source_runs WHERE source_id=? AND status='running'", (source.id,)
        ).fetchone():
            return None, "running"
        toggle = db.execute(
            "SELECT value FROM runtime_settings WHERE key=?", (f"source_enabled:{source.id}",)
        ).fetchone()
        if toggle and str(toggle[0]).lower() == "false":
            return None, "disabled"
        health = db.execute(
            "SELECT * FROM source_health WHERE source_id=?", (source.id,)
        ).fetchone()
        if health and health["cooldown_until"] and health["cooldown_until"] > stamp and not force:
            return None, "cooldown"
        last = db.execute(
            "SELECT started_at FROM source_runs WHERE source_id=? "
            "AND strategy != 'snapshot_replay' ORDER BY started_at DESC LIMIT 1",
            (source.id,),
        ).fetchone()
        if last and not force:
            previous = datetime.fromisoformat(last[0]).astimezone(UTC)
            if now - previous < timedelta(minutes=source.poll_interval_minutes):
                return None, "not_due"
        today, month = stamp[:10], stamp[:7]
        count = db.execute(
            "SELECT COUNT(*) FROM source_runs WHERE source_id=? AND substr(started_at,1,10)=? "
            "AND strategy != 'snapshot_replay'",
            (source.id, today),
        ).fetchone()[0]
        if count >= source.daily_call_cap:
            return None, "daily_call_cap"
        sums = db.execute(
            """SELECT COALESCE(SUM(cost_usd+reserved_usd),0),
            COALESCE(SUM(CASE WHEN substr(started_at,1,10)=?
                THEN cost_usd+reserved_usd ELSE 0 END),0),
            COALESCE(SUM(CASE WHEN source_id=? THEN cost_usd+reserved_usd ELSE 0 END),0)
            FROM source_runs WHERE substr(started_at,1,7)=?""",
            (today, source.id, month),
        ).fetchone()
        total = db.execute(
            "SELECT COALESCE(SUM(cost_usd),0) FROM usage_ledger WHERE substr(created_at,1,7)=?",
            (month,),
        ).fetchone()[0]
        reserved = db.execute(
            "SELECT COALESCE(SUM(reserved_usd),0) FROM source_runs WHERE status='running'"
        ).fetchone()[0]
        if estimated_cost > 0 and (
            sums[0] + estimated_cost > settings.sensory_monthly_cap_usd
            or sums[1] + estimated_cost > settings.sensory_daily_cap_usd
            or sums[2] + estimated_cost > source.monthly_cost_cap_usd
            or total + reserved + estimated_cost >= settings.budget_hard_stop_usd
        ):
            return None, "budget_blocked"
        run_id = uuid.uuid4().hex
        cursor = db.execute("SELECT last_cursor FROM sources WHERE id=?", (source.id,)).fetchone()
        db.execute(
            """INSERT INTO source_runs(id,source_id,adapter_id,strategy,status,
                   cursor_before,reserved_usd,started_at,lease_until) VALUES(?,?,?,?,?,?,?,?,?)""",
            (
                run_id,
                source.id,
                adapter_id,
                source.strategy,
                "running",
                cursor[0] if cursor else None,
                estimated_cost,
                stamp,
                (now + timedelta(minutes=10)).isoformat(),
            ),
        )
        return run_id, "running"


def finish_source(
    repository: Repository,
    run_id: str,
    *,
    status: str,
    items_seen: int = 0,
    items_inserted: int = 0,
    items_rejected: int = 0,
    cursor: str | None = None,
    cost_usd: float = 0,
    error: str | None = None,
    latency_ms: int = 0,
    now: datetime | None = None,
) -> None:
    current = now or datetime.now(UTC)
    stamp = current.isoformat()
    clean_error = redact_text(error or "")[:1000] or None
    good = status in {"ok", "empty", "partial", "pending", "filtered"}
    with repository.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        run = db.execute(
            "SELECT * FROM source_runs WHERE id=? AND status='running'", (run_id,)
        ).fetchone()
        if not run:
            return
        source_id = run["source_id"]
        previous = db.execute(
            "SELECT consecutive_errors FROM source_health WHERE source_id=?", (source_id,)
        ).fetchone()
        errors = 0 if good else min((previous[0] if previous else 0) + 1, 12)
        delay = (
            0
            if good
            else (
                1440
                if status in {"challenge", "login_required"}
                else min(1440, 15 * 2 ** (errors - 1))
            )
        )
        cooldown = (current + timedelta(minutes=delay)).isoformat() if delay else None
        db.execute(
            """UPDATE source_runs SET status=?,cursor_after=?,items_seen=?,items_inserted=?,
            items_rejected=?,cost_usd=?,reserved_usd=0,error=?,latency_ms=?,finished_at=?
            WHERE id=?""",
            (
                status,
                cursor,
                items_seen,
                items_inserted,
                items_rejected,
                cost_usd,
                clean_error,
                latency_ms,
                stamp,
                run_id,
            ),
        )
        db.execute(
            """INSERT INTO source_health(source_id,status,consecutive_errors,cooldown_until,
            last_attempt_at,last_productive_at,last_error) VALUES(?,?,?,?,?,?,?)
            ON CONFLICT(source_id) DO UPDATE SET status=excluded.status,
            consecutive_errors=excluded.consecutive_errors,cooldown_until=excluded.cooldown_until,
            last_attempt_at=excluded.last_attempt_at,
            last_productive_at=COALESCE(excluded.last_productive_at,source_health.last_productive_at),
            last_error=excluded.last_error""",
            (
                source_id,
                status,
                errors,
                cooldown,
                stamp,
                stamp if items_inserted else None,
                clean_error,
            ),
        )
        if good:
            db.execute(
                "UPDATE sources SET last_cursor=COALESCE(?,last_cursor),last_success_at=?,"
                "last_error=NULL WHERE id=?",
                (cursor, stamp, source_id),
            )
        else:
            db.execute("UPDATE sources SET last_error=? WHERE id=?", (clean_error, source_id))


def source_health_report(repository: Repository) -> list[dict[str, Any]]:
    with repository.connection() as db:
        rows = db.execute("""SELECT s.id,s.name,s.kind,s.adapter_id,s.strategy,s.admission_mode,
            s.enabled,h.status,h.cooldown_until,h.last_attempt_at,h.last_productive_at,h.last_error,
            (SELECT COUNT(*) FROM source_items i WHERE i.source_id=s.id) AS items,
            (SELECT COALESCE(SUM(cost_usd),0) FROM source_runs r WHERE r.source_id=s.id) AS cost_usd
            FROM sources s LEFT JOIN source_health h ON h.source_id=s.id
            ORDER BY s.kind,s.id""").fetchall()
    result = [dict(row) for row in rows]
    for row in result:
        override = repository.get_runtime_setting(f"source_enabled:{row['id']}")
        if override is not None:
            row["enabled"] = override == "true"
    return result
