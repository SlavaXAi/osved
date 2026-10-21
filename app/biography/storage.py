from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime, timedelta

from app.biography.catalog import PROFILE, Candidate, identity_key, seed_candidates
from app.db.repository import Repository


class BudgetExceeded(RuntimeError):
    pass


class UncertainOperation(RuntimeError):
    pass


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def unpack(row) -> dict | None:
    if row is None:
        return None
    value = dict(row)
    for key in list(value):
        if key.endswith("_json"):
            value[key[:-5]] = json.loads(value.pop(key))
    return value


class BiographyStore:
    def __init__(self, repository: Repository):
        self.repository = repository

    def initialize(self) -> None:
        if self.repository.get_runtime_setting("biography_profile") is None:
            self.repository.claim_runtime_setting(
                "biography_profile", json.dumps(PROFILE, ensure_ascii=False)
            )
        for index, person in enumerate(seed_candidates()):
            self.add_person(person, approved=True, seed_order=index)

    def add_person(self, person: Candidate, *, approved=False, seed_order=None) -> None:
        with self.repository.connection() as c:
            # Canonical English name plus aliases prevents survey/discovery duplicates.
            existing = c.execute(
                "SELECT name,english_name,aliases_json FROM biography_people"
            ).fetchall()
            keys = {
                identity_key(name)
                for r in existing
                for name in [r["name"], r["english_name"], *json.loads(r["aliases_json"])]
            }
            if any(
                identity_key(name) in keys
                for name in [person.english_name, person.name, *person.aliases]
            ):
                return
            c.execute(
                """INSERT OR IGNORE INTO biography_people
                (identity_key,name,english_name,aliases_json,topic,era,region,life_arc,living,
                 familiar,approved,life_scale,historical_impact,exceptional_quality,rationale,
                 image_query,seed_order,created_at,discovery_urls_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    identity_key(person.english_name),
                    person.name,
                    person.english_name,
                    json.dumps(person.aliases, ensure_ascii=False),
                    person.topic,
                    person.era,
                    person.region,
                    person.life_arc,
                    int(person.living or not approved),
                    int(person.english_name in PROFILE["known_people"]),
                    int(approved),
                    person.life_scale,
                    person.historical_impact,
                    person.exceptional_quality,
                    person.rationale,
                    person.image_query,
                    seed_order,
                    now_iso(),
                    json.dumps(person.discovery_urls),
                ),
            )

    def enabled(self, default=False) -> bool:
        stored = self.repository.get_runtime_setting("biography_enabled")
        return default if stored is None else stored == "true"

    def confirm_deceased(self, person_id: int) -> None:
        with self.repository.connection() as c:
            c.execute("UPDATE biography_people SET living=0 WHERE id=?", (person_id,))

    def set_enabled(self, enabled: bool) -> None:
        self.repository.set_runtime_setting("biography_enabled", str(enabled).lower())

    def recent(self, limit=10) -> list[dict]:
        with self.repository.connection() as c:
            return [
                unpack(r)
                for r in c.execute(
                    """SELECT p.*, i.payload_json,i.dossier_json,
                d.sent_at,d.created_at AS delivery_created_at,f.hero,f.text,f.reaction,f.comment
                FROM biography_deliveries d JOIN biography_issues i ON i.id=d.issue_id
                JOIN biography_people p ON p.id=i.person_id
                LEFT JOIN biography_feedback f ON f.delivery_id=d.id
                WHERE d.status IN ('sent','sending','unknown') ORDER BY d.id DESC LIMIT ?""",
                    (limit,),
                )
            ]

    def pick_candidates(self, *, current: datetime, limit=2, include_failed=False) -> list[dict]:
        recent = self.recent()
        with self.repository.connection() as c:
            people = [unpack(r) for r in c.execute("SELECT * FROM biography_people")]
            unavailable = {
                r[0]
                for r in c.execute(
                    """SELECT i.person_id FROM biography_issues i
                LEFT JOIN biography_deliveries d ON d.issue_id=i.id
                AND d.status IN ('sent','unknown','sending')
                WHERE i.status IN ('ready','reserved','sent','unknown') AND
                (i.status IN ('ready','reserved') OR
                COALESCE(d.sent_at,d.created_at,i.created_at)>?)""",
                    ((current - timedelta(days=120)).isoformat(),),
                )
            }
            failed_attempts = [
                json.loads(r[0]).get("attempted_people", [])
                for r in c.execute(
                    "SELECT artifact_json FROM biography_cycles WHERE created_at>?",
                    ((current - timedelta(days=1)).isoformat(),),
                )
            ]
            if not include_failed:
                unavailable.update(p for attempted in failed_attempts for p in attempted)
            ready_people = [
                unpack(r)
                for r in c.execute("""SELECT p.* FROM biography_issues i
                JOIN biography_people p ON p.id=i.person_id WHERE i.status IN ('ready','reserved')
                ORDER BY i.id DESC""")
            ]
        window = (ready_people + recent)[:9]
        living_allowed = not any(p["living"] for p in window)
        editorial_history = (ready_people + recent)[:10]

        def score(p):
            value = p["life_scale"] * 100 + p["historical_impact"] * 10 + p["exceptional_quality"]
            value += 8 if p["approved"] else 0
            value += 6 if p["topic"] in PROFILE["strong_topics"] else 0
            for past in editorial_history:
                value -= 9 * (past["topic"] == p["topic"])
                value -= 3 * (past["era"] == p["era"])
                value -= 3 * (past["region"] == p["region"])
                value -= 15 * (past["life_arc"] == p["life_arc"])
                # Explicit hero feedback wins; ordinary reactions never double count it.
                signal = past.get("hero")
                if signal is None:
                    signal = {
                        "like": 0.25,
                        "superlike": 0.5,
                        "dislike": -0.25,
                        "superdislike": -0.5,
                    }.get(past.get("reaction"), 0)
                if past["topic"] == p["topic"]:
                    value += signal * 2
            return value, -(p["seed_order"] if p["seed_order"] is not None else 999), -p["id"]

        eligible = [
            p
            for p in people
            if p["id"] not in unavailable
            and (not p["living"] or not p["approved"] or living_allowed)
            and max(p["life_scale"], p["historical_impact"], p["exceptional_quality"]) >= 4
        ]
        # Honour the explicitly agreed three pilots before ranking the ongoing stream.
        eligible.sort(
            key=lambda p: (p["seed_order"] is not None and p["seed_order"] < 3, score(p)),
            reverse=True,
        )
        return eligible[:limit]

    def can_prepare_living(self) -> bool:
        with self.repository.connection() as connection:
            queued = connection.execute(
                "SELECT p.living FROM biography_issues i JOIN biography_people p "
                "ON p.id=i.person_id WHERE i.status IN ('ready','reserved') ORDER BY i.id DESC"
            ).fetchall()
        return not any(row["living"] for row in (list(queued) + self.recent())[:9])

    def person_history(self, person_id: int) -> list[dict]:
        with self.repository.connection() as c:
            return [
                unpack(r)
                for r in c.execute(
                    """SELECT payload_json,dossier_json FROM biography_issues
                WHERE person_id=? AND status IN ('sent','unknown') ORDER BY id DESC""",
                    (person_id,),
                )
            ]

    def start_cycle(
        self, local_date: str, *, max_cost=0.25, daily_limit=3, kind="prepare"
    ) -> int | None:
        with self.repository.connection() as c:
            c.execute("BEGIN IMMEDIATE")
            if c.execute("SELECT 1 FROM biography_cycles WHERE status='running'").fetchone():
                return None
            if c.execute(
                "SELECT 1 FROM biography_usage WHERE status IN ('reserved','unknown')"
            ).fetchone():
                return None
            if (
                c.execute(
                    "SELECT COUNT(*) FROM biography_cycles WHERE local_date=? AND kind=?",
                    (local_date, kind),
                ).fetchone()[0]
                >= daily_limit
            ):
                return None
            cursor = c.execute(
                """INSERT INTO biography_cycles(local_date,kind,max_cost_usd,created_at)
                VALUES (?,?,?,?)""",
                (local_date, kind, max_cost, now_iso()),
            )
            return cursor.lastrowid

    def save_cycle(self, cycle_id: int, artifacts: dict) -> None:
        with self.repository.connection() as c:
            c.execute(
                "UPDATE biography_cycles SET artifact_json=? WHERE id=? AND status='running'",
                (json.dumps(artifacts, ensure_ascii=False), cycle_id),
            )

    def resume_cycle(self, cycle_id: int) -> dict:
        """Operator only: after all receipts are reconciled, keep the original $0.25 cap."""
        with self.repository.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cycle = unpack(
                connection.execute(
                    "SELECT * FROM biography_cycles WHERE id=?", (cycle_id,)
                ).fetchone()
            )
            if not cycle or cycle["status"] not in {"failed", "unknown", "budget_blocked"}:
                raise ValueError("Cycle cannot be resumed")
            if cycle["artifact"].get("manual_resume"):
                raise ValueError("Manual resumption has already been used")
            if (
                connection.execute(
                    "SELECT 1 FROM biography_usage WHERE status IN ('reserved','unknown')"
                ).fetchone()
                or connection.execute(
                    "SELECT 1 FROM biography_cycles WHERE status='running'"
                ).fetchone()
            ):
                raise UncertainOperation("Reconcile every pending receipt before resuming")
            if connection.execute(
                "SELECT 1 FROM biography_issues WHERE cycle_id=?", (cycle_id,)
            ).fetchone():
                raise ValueError("Cycle already has a prepared issue")
            connection.execute(
                "UPDATE biography_cycles SET status='running',finished_at=NULL,artifact_json=? "
                "WHERE id=?",
                (json.dumps({**cycle["artifact"], "manual_resume": now_iso()}), cycle_id),
            )
            return cycle["artifact"]

    def paid_receipt(self, usage_id: int, receipt: dict) -> None:
        with self.repository.connection() as connection:
            connection.execute(
                "UPDATE biography_usage SET receipt_json=? WHERE id=? AND status='reserved'",
                (json.dumps(receipt, ensure_ascii=False), usage_id),
            )

    def finish_cycle(self, cycle_id: int, status: str, error=None) -> None:
        with self.repository.connection() as c:
            c.execute(
                "UPDATE biography_cycles SET status=?,last_error=?,finished_at=? WHERE id=?",
                (status, error, now_iso(), cycle_id),
            )

    def reserve(self, cycle_id: int, key: str, provider: str, amount: float) -> int:
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("Invalid reservation amount")
        with self.repository.connection() as c:
            c.execute("BEGIN IMMEDIATE")
            cycle = c.execute("SELECT * FROM biography_cycles WHERE id=?", (cycle_id,)).fetchone()
            if cycle is None or cycle["status"] != "running":
                raise UncertainOperation("Cycle is not active")
            if c.execute(
                "SELECT 1 FROM biography_usage WHERE cycle_id=? AND status IN "
                "('reserved','unknown')",
                (cycle_id,),
            ).fetchone():
                raise UncertainOperation("Unsettled paid operation; automatic repeat prohibited")
            spent = c.execute(
                "SELECT COALESCE(SUM(cost_usd),0) FROM biography_usage WHERE cycle_id=?",
                (cycle_id,),
            ).fetchone()[0]
            if spent + amount > cycle["max_cost_usd"] + 1e-9:
                raise BudgetExceeded("Biography preparation would exceed $0.25")
            if c.execute(
                "SELECT 1 FROM biography_usage WHERE cycle_id=? AND operation_key=?",
                (cycle_id, key),
            ).fetchone():
                raise UncertainOperation("Paid stage already exists")
            return c.execute(
                """INSERT INTO biography_usage(cycle_id,operation_key,provider,
                reservation_usd,cost_usd,created_at) VALUES (?,?,?,?,?,?)""",
                (cycle_id, key, provider, amount, amount, now_iso()),
            ).lastrowid

    def settle(self, usage_id: int, *, cost: float | None, receipt: dict, response=None) -> None:
        if cost is not None and (not math.isfinite(cost) or cost < 0):
            raise ValueError("Invalid provider cost")
        with self.repository.connection() as c:
            c.execute(
                """UPDATE biography_usage SET status=?,cost_usd=COALESCE(?,reservation_usd),
                receipt_json=?,response_json=? WHERE id=? AND status='reserved'""",
                (
                    "settled" if cost is not None else "unknown",
                    cost,
                    json.dumps(receipt, ensure_ascii=False),
                    json.dumps(response, ensure_ascii=False) if response is not None else None,
                    usage_id,
                ),
            )

    def documents(self, person_id: int) -> list[dict]:
        with self.repository.connection() as c:
            return [
                dict(r)
                for r in c.execute(
                    "SELECT * FROM biography_documents WHERE person_id=? ORDER BY id", (person_id,)
                )
            ]

    def save_document(
        self, person_id: int, url: str, publisher: str, title: str, markdown: str
    ) -> None:
        with self.repository.connection() as c:
            c.execute(
                "INSERT INTO biography_documents(person_id,url,publisher,title,markdown"
                ",content_hash,created_at)\n                VALUES (?,?,?,?,?,?,?) ON "
                "CONFLICT(person_id,url) DO UPDATE SET markdown=excluded.markdown,\n    "
                "            title=excluded.title,content_hash=excluded.content_hash,cr"
                "eated_at=excluded.created_at",
                (
                    person_id,
                    url,
                    publisher,
                    title,
                    markdown,
                    hashlib.sha256(markdown.encode()).hexdigest(),
                    now_iso(),
                ),
            )

    def save_issue(self, person_id: int, cycle_id: int, payload: dict, dossier: dict) -> int:
        with self.repository.connection() as c:
            if c.execute(
                "SELECT 1 FROM biography_usage WHERE cycle_id=? AND status!='settled'", (cycle_id,)
            ).fetchone():
                raise UncertainOperation("A receipt remains unsettled")
            return c.execute(
                "INSERT INTO biography_issues(person_id,cycle_id,payload_json,dossier_j"
                "son,created_at)\n                VALUES (?,?,?,?,?)",
                (
                    person_id,
                    cycle_id,
                    json.dumps(payload, ensure_ascii=False),
                    json.dumps(dossier, ensure_ascii=False),
                    now_iso(),
                ),
            ).lastrowid

    def claim_delivery(
        self,
        owner_id: int,
        slot_key: str,
        *,
        issue_id: int | None = None,
        current: datetime | None = None,
    ) -> dict | None:
        with self.repository.connection() as c:
            c.execute("BEGIN IMMEDIATE")
            existing = c.execute(
                "SELECT * FROM biography_deliveries WHERE owner_id=? AND slot_key=?",
                (owner_id, slot_key),
            ).fetchone()
            if existing:
                return dict(existing) if existing["status"] == "pending" else None
            candidates = c.execute(
                "SELECT i.*,p.living FROM biography_issues i JOIN biography_people p ON "
                "p.id=i.person_id WHERE i.status='ready'"
                + (" AND i.id=?" if issue_id is not None else "")
                + " ORDER BY i.id",
                (issue_id,) if issue_id is not None else (),
            ).fetchall()
            previous = c.execute(
                "SELECT p.living FROM biography_deliveries d JOIN biography_issues i ON "
                "i.id=d.issue_id JOIN biography_people p ON p.id=i.person_id WHERE "
                "d.status IN ('sent','sending','unknown') ORDER BY d.id DESC LIMIT 9"
            ).fetchall()
            living_allowed = not any(r["living"] for r in previous)
            threshold = ((current or datetime.now(UTC)) - timedelta(days=120)).isoformat()
            issue = None
            for candidate in candidates:
                if candidate["living"] and not living_allowed:
                    continue
                repeated = c.execute(
                    "SELECT 1 FROM biography_deliveries d JOIN biography_issues i ON "
                    "i.id=d.issue_id WHERE i.person_id=? AND d.status IN "
                    "('sent','sending','unknown') AND COALESCE(d.sent_at,d.created_at)>?",
                    (candidate["person_id"], threshold),
                ).fetchone()
                if not repeated:
                    issue = candidate
                    break
            if issue is None:
                return None
            cursor = c.execute(
                "INSERT INTO biography_deliveries(issue_id,owner_id,slot_key,created_at"
                ") VALUES (?,?,?,?)",
                (issue["id"], owner_id, slot_key, now_iso()),
            )
            c.execute("UPDATE biography_issues SET status='reserved' WHERE id=?", (issue["id"],))
            return dict(
                c.execute(
                    "SELECT * FROM biography_deliveries WHERE id=?", (cursor.lastrowid,)
                ).fetchone()
            )

    def begin_send(self, delivery_id: int, *, current: datetime) -> bool:
        with self.repository.connection() as c:
            cursor = c.execute(
                """UPDATE biography_deliveries SET status='sending',attempts=attempts+1,
                retry_at=? WHERE id=? AND status='pending' AND attempts<3
                AND (retry_at IS NULL OR retry_at<=?)""",
                (current.isoformat(), delivery_id, current.isoformat()),
            )
            return bool(cursor.rowcount)

    def delivery(self, delivery_id: int) -> dict | None:
        with self.repository.connection() as c:
            return unpack(
                c.execute(
                    "SELECT d.*,p.name,i.payload_json FROM biography_deliveries d\n     "
                    "           JOIN biography_issues i ON i.id=d.issue_id JOIN "
                    "biography_people p ON p.id=i.person_id\n                WHERE "
                    "d.id=?",
                    (delivery_id,),
                ).fetchone()
            )

    def finish_delivery(
        self, delivery_id: int, status: str, *, message_id=None, error=None, retry_at=None
    ) -> None:
        with self.repository.connection() as c:
            c.execute(
                """UPDATE biography_deliveries SET status=?,telegram_message_id=?,last_error=?,
                retry_at=?,sent_at=? WHERE id=? AND status='sending'""",
                (
                    status,
                    message_id,
                    error,
                    retry_at,
                    now_iso() if status == "sent" else None,
                    delivery_id,
                ),
            )
            if status in {"sent", "unknown"}:
                c.execute(
                    "UPDATE biography_issues SET status=? WHERE id=(SELECT issue_id "
                    "FROM biography_deliveries WHERE id=?)",
                    (status, delivery_id),
                )
            elif status == "failed":
                c.execute(
                    "UPDATE biography_issues SET status='ready' WHERE id=(SELECT "
                    "issue_id FROM biography_deliveries WHERE id=?)",
                    (delivery_id,),
                )

    def expire_pending(self, slot_key: str) -> None:
        with self.repository.connection() as c:
            c.execute(
                "UPDATE biography_issues SET status='ready' WHERE id IN\n               "
                " (SELECT issue_id FROM biography_deliveries WHERE status='pending' "
                "AND slot_key=?)",
                (slot_key,),
            )
            # Keep the failed day in the outbox; its issue can be claimed tomorrow.
            c.execute(
                "UPDATE biography_deliveries SET status='failed' WHERE "
                "status='pending' AND slot_key=?",
                (slot_key,),
            )

    def expire_older_pending(self, current_slot: str) -> None:
        with self.repository.connection() as c:
            slots = [
                r[0]
                for r in c.execute(
                    "SELECT slot_key FROM biography_deliveries WHERE status='pending' "
                    "AND slot_key LIKE 'daily:%' AND slot_key<?",
                    (current_slot,),
                )
            ]
        for slot in slots:
            self.expire_pending(slot)

    def recover_stale(self, current: datetime) -> None:
        with self.repository.connection() as c:
            stale = (current - timedelta(minutes=5)).isoformat()
            c.execute(
                "UPDATE biography_issues SET status='unknown' WHERE id IN\n             "
                "   (SELECT issue_id FROM biography_deliveries WHERE status='sending' "
                "AND retry_at<?)",
                (stale,),
            )
            c.execute(
                "UPDATE biography_deliveries SET status='unknown',last_error='Interrupt"
                "ed send; no automatic repeat' WHERE status='sending' AND retry_at<?",
                (stale,),
            )
            c.execute(
                "UPDATE biography_usage SET status='unknown' WHERE status='reserved' "
                "AND cycle_id IN\n                (SELECT id FROM biography_cycles "
                "WHERE status='running' AND "
                "COALESCE(json_extract(artifact_json,'$.manual_resume'),created_at)<?)",
                ((current - timedelta(hours=1)).isoformat(),),
            )
            c.execute(
                "UPDATE biography_cycles SET status='unknown',last_error='Interrupted "
                "preparation; inspect receipts' WHERE status='running' AND "
                "COALESCE(json_extract(artifact_json,'$.manual_resume'),created_at)<?",
                ((current - timedelta(hours=1)).isoformat(),),
            )

    def feedback(self, delivery_id: int, owner_id: int) -> dict:
        with self.repository.connection() as c:
            return dict(
                c.execute(
                    "SELECT * FROM biography_feedback WHERE delivery_id=? AND owner_id=?",
                    (delivery_id, owner_id),
                ).fetchone()
                or {}
            )

    def set_feedback(self, delivery_id: int, owner_id: int, axis: str, value) -> None:
        if axis not in {"hero", "text", "reaction", "comment"}:
            raise ValueError("Unknown feedback axis")
        if axis in {"hero", "text"} and value not in {-1, 1}:
            raise ValueError("Invalid rating")
        if axis == "reaction" and value not in {"like", "dislike", "superlike", "superdislike"}:
            raise ValueError("Invalid reaction")
        with self.repository.connection() as c:
            delivery = c.execute(
                "SELECT owner_id,status FROM biography_deliveries WHERE id=?", (delivery_id,)
            ).fetchone()
            if delivery is None or delivery["owner_id"] != owner_id or delivery["status"] != "sent":
                raise ValueError("Delivery does not belong to this owner")
            c.execute(
                "INSERT OR IGNORE INTO biography_feedback(delivery_id,owner_id,updated_"
                "at) VALUES (?,?,?)",
                (delivery_id, owner_id, now_iso()),
            )
            if axis != "comment":
                c.execute(
                    f"UPDATE biography_feedback SET {axis}=CASE WHEN {axis}=? "
                    "THEN NULL ELSE ? END,updated_at=? WHERE delivery_id=? AND owner_id=?",
                    (value, value, now_iso(), delivery_id, owner_id),
                )
            else:
                c.execute(
                    "UPDATE biography_feedback SET comment=?,updated_at=? WHERE "
                    "delivery_id=? AND owner_id=?",
                    (str(value).strip()[:4000], now_iso(), delivery_id, owner_id),
                )

    def status(self) -> dict:
        with self.repository.connection() as c:
            ready = c.execute(
                "SELECT COUNT(*) FROM biography_issues WHERE status='ready'"
            ).fetchone()[0]
            last = unpack(
                c.execute("SELECT * FROM biography_cycles ORDER BY id DESC LIMIT 1").fetchone()
            )
            cost = c.execute(
                "SELECT COALESCE(SUM(cost_usd),0) FROM biography_usage WHERE "
                "substr(created_at,1,7)=?",
                (datetime.now(UTC).strftime("%Y-%m"),),
            ).fetchone()[0]
            unknown = c.execute(
                "SELECT COUNT(*) FROM biography_usage WHERE status IN ('reserved','unknown')"
            ).fetchone()[0]
        return {
            "ready": ready,
            "last_cycle": last,
            "month_cost_usd": cost,
            "unsettled_receipts": unknown,
        }
