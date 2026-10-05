"""Persistent local paid-test ceiling, independent of the imported server ledger.

Provider ledger entries retain their own actual/estimated reservations. A small
outer hold protects the missing interval when a process dies before a provider
has recorded its receipt. Neither a restart nor a failed request resets spending.
"""

from __future__ import annotations

import math

from app.db.repository import Repository, utc_now


class LocalBudgetBlocked(RuntimeError):
    pass


class LocalBudget:
    def __init__(self, repository: Repository, ceiling: float):
        if (
            not isinstance(ceiling, int | float)
            or not math.isfinite(ceiling)
            or not 0 <= ceiling <= 2
        ):
            raise ValueError("Local paid-test ceiling must be between zero and USD 2")
        self.repository, self.ceiling = repository, float(ceiling)

    def initialize(self, *, enabled: bool) -> None:
        with self.repository.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("""CREATE TABLE IF NOT EXISTS web_budget_meta (
                id INTEGER PRIMARY KEY CHECK(id=1), baseline_ledger_id INTEGER NOT NULL,
                limit_usd REAL NOT NULL, created_at TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS web_paid_operations (
                id TEXT PRIMARY KEY, worst_case_usd REAL NOT NULL,
                starting_ledger_id INTEGER NOT NULL,
                ending_ledger_id INTEGER, status TEXT NOT NULL, created_at TEXT NOT NULL)""")
            if enabled:
                db.execute(
                    "INSERT OR IGNORE INTO web_budget_meta(id,baseline_ledger_id,"
                    "limit_usd,created_at) "
                    "SELECT 1,COALESCE(MAX(id),0),?,? FROM usage_ledger",
                    (self.ceiling, utc_now()),
                )
                # A CLI restart cannot silently reset or expand an existing test budget.
                db.execute(
                    "UPDATE web_budget_meta SET limit_usd=MIN(limit_usd,?) WHERE id=1",
                    (self.ceiling,),
                )
            db.execute(
                "UPDATE web_paid_operations SET status='uncertain',ending_ledger_id="
                "(SELECT COALESCE(MAX(id),0) FROM usage_ledger) WHERE status='active'"
            )

    def _snapshot(self, db) -> dict:
        meta = db.execute("SELECT * FROM web_budget_meta WHERE id=1").fetchone()
        if meta is None:
            return {
                "limit_usd": self.ceiling,
                "spent_usd": 0.0,
                "reserved_usd": 0.0,
                "remaining_usd": self.ceiling,
                "initialized": False,
            }
        spent = float(
            db.execute(
                "SELECT COALESCE(SUM(cost_usd),0) FROM usage_ledger WHERE id>?",
                (meta["baseline_ledger_id"],),
            ).fetchone()[0]
        )
        reserved = 0.0
        for hold in db.execute("SELECT * FROM web_paid_operations WHERE status!='settled'"):
            accounted = float(
                db.execute(
                    "SELECT COALESCE(SUM(cost_usd),0) FROM usage_ledger "
                    "WHERE id>? AND (? IS NULL OR id<=?)",
                    (
                        hold["starting_ledger_id"],
                        hold["ending_ledger_id"],
                        hold["ending_ledger_id"],
                    ),
                ).fetchone()[0]
            )
            reserved += max(0, float(hold["worst_case_usd"]) - accounted)
        return {
            "limit_usd": float(meta["limit_usd"]),
            "spent_usd": round(spent, 8),
            "reserved_usd": round(reserved, 8),
            "remaining_usd": round(max(0, float(meta["limit_usd"]) - spent - reserved), 8),
            "initialized": True,
        }

    def snapshot(self) -> dict:
        with self.repository.connection() as db:
            return self._snapshot(db)

    def reserve(self, operation_id: str, worst_case: float) -> None:
        if not math.isfinite(worst_case) or worst_case < 0:
            raise LocalBudgetBlocked("invalid_reservation")
        with self.repository.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            snapshot = self._snapshot(db)
            if not snapshot["initialized"] or worst_case > snapshot["remaining_usd"] + 1e-9:
                raise LocalBudgetBlocked("local_budget_blocked")
            if db.execute("SELECT 1 FROM web_paid_operations WHERE status='active'").fetchone():
                raise LocalBudgetBlocked("local_paid_request_in_progress")
            if db.execute(
                "SELECT 1 FROM web_paid_operations WHERE id=?", (operation_id,)
            ).fetchone():
                raise LocalBudgetBlocked("operation_already_attempted")
            db.execute(
                "INSERT INTO web_paid_operations(id,worst_case_usd,starting_ledger_id,"
                "status,created_at) "
                "SELECT ?,?,COALESCE(MAX(id),0),'active',? FROM usage_ledger",
                (operation_id, worst_case, utc_now()),
            )

    def finish(self, operation_id: str, *, uncertain: bool = False) -> None:
        with self.repository.connection() as db:
            db.execute(
                "UPDATE web_paid_operations SET status=?,ending_ledger_id="
                "(SELECT COALESCE(MAX(id),0) FROM usage_ledger) WHERE id=? AND status='active'",
                ("uncertain" if uncertain else "settled", operation_id),
            )
