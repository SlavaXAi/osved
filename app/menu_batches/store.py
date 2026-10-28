from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import UTC, datetime, timedelta
from urllib.parse import urlsplit

from app.config import load_weekly_persona_registry
from app.db.card_predicates import finished_card_sql
from app.video.relevance import VideoRelevanceService, relevant

KINDS = {
    "news": ("Новости · сайты и блоги", 1, 7),
    "telegram": ("Telegram-радар", 1, 7),
    "twitter": ("X / Twitter", 1, 7),
    "youtube": ("YouTube", 1, 7),
    "people": ("Люди · действия и высказывания", 7, 3),
    "institutions": ("Институты и организации", 7, 3),
    "bio": ("Биография", 0, 1),
}


def stamp():
    return datetime.now(UTC).isoformat()


def normalized(text):
    return re.sub(r"\W+", " ", str(text).casefold()).strip()


def entity_names(value):
    if isinstance(value, (str, dict)):
        value = [value]
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if isinstance(item, dict):
            item = item.get("name") or item.get("canonical_name")
        if isinstance(item, str) and item.strip():
            result.append(item.strip())
    return result


def x_author(event):
    try:
        url = urlsplit(event.get("source_url") or "")
        if (
            url.scheme == "https"
            and url.hostname
            in {"x.com", "www.x.com", "twitter.com", "www.twitter.com", "mobile.twitter.com"}
            and not url.username
            and not url.password
        ):
            handle = url.path.strip("/").split("/")[0]
            if re.fullmatch(r"[A-Za-z0-9_]{1,15}", handle) and handle not in {
                "i",
                "home",
                "search",
            }:
                return handle.casefold()
    except ValueError:
        pass
    return None


def entity_labels(settings, event, kind):
    entities = event.get("entities_json") or event.get("entities") or {}
    if isinstance(entities, str):
        try:
            entities = json.loads(entities)
        except ValueError:
            entities = {}
    if not isinstance(entities, dict):
        entities = {}
    if kind == "institutions":
        return entity_names(entities.get("organizations"))
    if kind != "people":
        return []
    labels = entity_names(entities.get("persons")) + entity_names(entities.get("people"))
    if settings.weekly_personas_file.is_file():
        corpus = (
            " "
            + normalized(
                " ".join(
                    str(event.get(k) or "") for k in ("headline", "original_text", "rendered_html")
                )
            )
            + " "
        )
        author = x_author(event)
        for person in load_weekly_persona_registry(settings.weekly_personas_file).personas:
            names = [person.name, *person.aliases, *person.source_strategy.x_handles]
            if (
                author and author in {h.casefold() for h in person.source_strategy.x_handles}
            ) or any(normalized(name) and " " + normalized(name) + " " in corpus for name in names):
                labels.append(person.name)
    return list(dict.fromkeys(labels))


class BatchStore:
    def __init__(self, settings, repository):
        self.settings, self.repository = settings, repository

    def get(self, batch_id):
        with self.repository.connection() as db:
            row = db.execute("SELECT * FROM menu_batches WHERE id=?", (batch_id,)).fetchone()
        return dict(row) if row else None

    def items(self, batch_id):
        with self.repository.connection() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM menu_batch_items WHERE batch_id=? ORDER BY id", (batch_id,)
                )
            ]

    def candidates(self, owner, kind, start, end):
        if kind == "bio":
            with self.repository.connection() as db:
                rows = db.execute(
                    "SELECT i.id,p.name FROM biography_issues i JOIN biography_people p "
                    "ON p.id=i.person_id WHERE i.status='ready' ORDER BY i.id"
                ).fetchall()
            return [
                dict(
                    item_kind="bio",
                    entity_id=r["id"],
                    entity_key=f"bio:{r['id']}",
                    payload={"name": r["name"]},
                )
                for r in rows
            ]
        if kind == "youtube":
            profile_hash = VideoRelevanceService(self.settings, self.repository, owner).profile_hash
            with self.repository.connection() as db:
                rows = db.execute(
                    "SELECT i.*,s.name source_name,a.assessment_json,y.channel_id,"
                    "COALESCE(p.preference,0) channel_preference FROM source_items i "
                    "JOIN sources s ON s.id=i.source_id "
                    "JOIN youtube_video_types t ON t.video_id=i.external_id AND t.is_short=0 "
                    "JOIN video_relevance_assessments a ON a.source_item_id=i.id "
                    "AND a.chat_id=? AND a.profile_hash=? AND a.status='ready' "
                    "LEFT JOIN youtube_channels y ON y.source_id=s.id "
                    "LEFT JOIN youtube_channel_preferences p ON p.chat_id=? "
                    "AND p.channel_id=COALESCE(y.channel_id,s.locator) "
                    "WHERE s.kind='youtube' AND s.admission_mode!='history' "
                    "AND COALESCE((SELECT CASE value WHEN 'true' THEN 1 WHEN 'false' THEN 0 END "
                    "FROM runtime_settings WHERE key='source_enabled:'||s.id),s.enabled)=1 "
                    "AND julianday(i.published_at) BETWEEN julianday(?) AND julianday(?) "
                    "AND (y.removed_at IS NULL AND (y.not_before IS NULL "
                    "OR julianday(i.published_at)>=julianday(y.not_before))) "
                    "AND NOT EXISTS(SELECT 1 FROM video_recommendations v "
                    "JOIN source_items seen ON seen.id=v.source_item_id "
                    "WHERE v.chat_id=? AND seen.external_id=i.external_id) "
                    "ORDER BY i.published_at DESC LIMIT 150",
                    (owner, profile_hash, owner, start, end, owner),
                ).fetchall()
            candidates = []
            for row in rows:
                assessment = json.loads(row["assessment_json"])
                if relevant(assessment, row["channel_preference"]):
                    candidates.append(
                        dict(
                            item_kind="video",
                            entity_id=row["id"],
                            entity_key="video:" + row["external_id"],
                            payload={
                                "description_html": assessment["description_html"],
                                "video_id": row["external_id"],
                            },
                            score=assessment["relevance"],
                        )
                    )
            return sorted(candidates, key=lambda r: r["score"], reverse=True)
        with self.repository.connection() as db:
            rows = db.execute(
                f"""SELECT c.*,c.id card_id,e.entities_json,e.headline,e.fingerprint,
                    e.score,e.score_features_json,e.event_type,
                    i.original_text,i.source_url,s.kind source_kind
                FROM cards c JOIN events e ON e.id=c.event_id
                JOIN source_items i ON i.id=e.primary_source_item_id
                JOIN sources s ON s.id=i.source_id
                WHERE {finished_card_sql("c")}
                  AND e.status NOT IN ('merged','rejected','duplicate')
                  AND COALESCE(e.event_type,'')!='system_test'
                  AND s.kind!='youtube' AND s.admission_mode!='history'
                  AND COALESCE((SELECT CASE value WHEN 'true' THEN 1 WHEN 'false' THEN 0 END
                    FROM runtime_settings WHERE key='source_enabled:'||s.id),s.enabled)=1
                  AND julianday(COALESCE(NULLIF(e.event_at,''),i.published_at))
                    BETWEEN julianday(?) AND julianday(?)
                  AND NOT EXISTS(SELECT 1 FROM cards newer WHERE newer.event_id=e.id
                    AND newer.id>c.id AND {finished_card_sql("newer")})
                  AND NOT EXISTS(SELECT 1 FROM deliveries d JOIN cards seen ON seen.id=d.card_id
                    JOIN events se ON se.id=seen.event_id WHERE d.chat_id=?
                    AND (se.id=e.id OR (e.fingerprint IS NOT NULL AND e.fingerprint!=''
                        AND se.fingerprint=e.fingerprint)) AND d.delivery_status!='failed')
                ORDER BY e.score DESC,e.event_at DESC,c.id DESC LIMIT 250""",
                (start, end, owner),
            ).fetchall()
        candidates = []
        external = json.loads(
            self.repository.get_runtime_setting(f"news_external_receipts:{owner}") or "[]"
        )
        seen_urls = {u.rstrip("/") for receipt in external for u in receipt.get("source_urls", [])}
        for raw in rows:
            row = dict(raw)
            allowed = {"news": {"web", "rss"}, "telegram": {"telegram"}, "twitter": {"x"}}.get(kind)
            if allowed and row["source_kind"] not in allowed:
                continue
            if seen_urls.intersection(u.rstrip("/") for u in json.loads(row["source_urls_json"])):
                continue
            labels = entity_labels(self.settings, row, kind)
            if kind in {"people", "institutions"} and not labels:
                continue
            candidates.append(
                dict(
                    item_kind="news",
                    entity_id=row["card_id"],
                    entity_key=f"event:{row['event_id']}",
                    payload=row,
                )
            )
        from app.personalization.daily import load_context, record_selection
        from app.pipeline.ranking import rank_event_candidates

        context = load_context(self.settings, self.repository, owner=owner)
        if context:
            pool = [{**r["payload"], "id": r["payload"]["event_id"]} for r in candidates]
            ranked = rank_event_candidates(pool, limit=len(pool), personalization=context)
            positions = {r["card_id"]: i for i, r in enumerate(ranked)}
            candidates = sorted(
                [r for r in candidates if r["entity_id"] in positions],
                key=lambda r: positions[r["entity_id"]],
            )
            record_selection(
                self.repository,
                context,
                owner=owner,
                stage="menu:" + kind,
                candidates=pool,
                selected=ranked[: KINDS[kind][2]],
            )
        return candidates

    def request(self, owner, kind, request_key, *, now=None, force=False):
        if kind not in KINDS:
            raise ValueError("Unknown edition")
        now = now or datetime.now(UTC)
        start = (now - timedelta(days=KINDS[kind][1])).isoformat()
        end = now.isoformat()
        profile = self.settings.interests_file
        digest = hashlib.sha256(profile.read_bytes() if profile.is_file() else b"").hexdigest()
        candidates = self.candidates(owner, kind, start, end)
        with self.repository.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute(
                "SELECT id FROM menu_batches WHERE request_key=?", (request_key,)
            ).fetchone()
            if not old and not force:
                old = db.execute(
                    "SELECT id FROM menu_batches WHERE owner_id=? AND kind=? AND "
                    "(status IN ('queued','sending') OR julianday(created_at)>=julianday(?)) "
                    "ORDER BY id DESC LIMIT 1",
                    (owner, kind, (now - timedelta(seconds=60)).isoformat()),
                ).fetchone()
            if old:
                return old["id"], False
            batch_id = db.execute(
                "INSERT INTO menu_batches(owner_id,kind,request_key,period_start,period_end,"
                "profile_hash,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
                (owner, kind, request_key, start, end, digest, end, end),
            ).lastrowid
            self._reserve(db, batch_id, owner, candidates, KINDS[kind][2])
            return batch_id, True

    def _reserve(self, db, batch_id, owner, candidates, limit):
        count = 0
        for item in candidates:
            if count >= limit:
                break
            if db.execute(
                "SELECT 1 FROM menu_batch_items WHERE owner_id=? AND entity_key=? "
                "AND status NOT IN('failed','skipped')",
                (owner, item["entity_key"]),
            ).fetchone():
                continue
            delivery_id = None
            if item["item_kind"] == "news":
                event_id = item["payload"]["event_id"]
                if db.execute(
                    "SELECT 1 FROM deliveries d JOIN cards c ON c.id=d.card_id "
                    "JOIN events e ON e.id=c.event_id WHERE d.chat_id=? "
                    "AND (c.event_id=? OR (? IS NOT NULL AND ?!='' AND e.fingerprint=?)) "
                    "AND d.delivery_status!='failed'",
                    (owner, event_id, *([item["payload"].get("fingerprint")] * 3)),
                ).fetchone():
                    continue
                db.execute(
                    "INSERT INTO deliveries(card_id,chat_id,batch_id,delivery_status) "
                    "VALUES(?,?,?,'pending') ON CONFLICT(card_id,chat_id) DO UPDATE SET "
                    "batch_id=excluded.batch_id,delivery_status='pending' "
                    "WHERE deliveries.delivery_status='failed'",
                    (item["entity_id"], owner, f"menu:{batch_id}"),
                )
                delivery_id = db.execute(
                    "SELECT id FROM deliveries WHERE card_id=? AND chat_id=?",
                    (item["entity_id"], owner),
                ).fetchone()[0]
            elif item["item_kind"] == "video":
                inserted = db.execute(
                    "INSERT OR IGNORE INTO video_recommendations(source_item_id,chat_id,"
                    "description_html,status,created_at) VALUES(?,?,?,'preparing',?)",
                    (item["entity_id"], owner, item["payload"]["description_html"], stamp()),
                )
                if not inserted.rowcount:
                    continue
                delivery_id = inserted.lastrowid
            db.execute(
                "INSERT INTO menu_batch_items(batch_id,owner_id,entity_key,item_kind,entity_id,"
                "delivery_id,payload_json) VALUES(?,?,?,?,?,?,?)",
                (
                    batch_id,
                    owner,
                    item["entity_key"],
                    item["item_kind"],
                    item["entity_id"],
                    delivery_id,
                    json.dumps(item["payload"], ensure_ascii=False),
                ),
            )
            count += 1
        if not count:
            db.execute("UPDATE menu_batches SET status='empty' WHERE id=?", (batch_id,))

    def claim_item(self):
        now = time.time()
        with self.repository.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            stale = db.execute(
                "SELECT * FROM menu_batch_items WHERE status='sending' AND lease_until<?", (now,)
            ).fetchall()
            for item in stale:
                self._finish(db, item, "uncertain", error="delivery acknowledgement lost")
            row = db.execute(
                "SELECT i.* FROM menu_batch_items i JOIN menu_batches b ON b.id=i.batch_id "
                "WHERE i.status='queued' AND i.available_at<=? AND b.status_message_id>0 "
                "ORDER BY i.id LIMIT 1",
                (now,),
            ).fetchone()
            if not row:
                return None
            db.execute(
                "UPDATE menu_batch_items SET status='sending',lease_until=? WHERE id=?",
                (now + 180, row["id"]),
            )
            db.execute(
                "UPDATE menu_batches SET status='sending',updated_at=? WHERE id=?",
                (stamp(), row["batch_id"]),
            )
            return dict(row)

    def extend(self, batch_id):
        batch = self.get(batch_id)
        candidates = self.candidates(
            batch["owner_id"], batch["kind"], batch["period_start"], batch["period_end"]
        )
        with self.repository.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute(
                "SELECT count(*) FROM menu_batch_items WHERE batch_id=?", (batch_id,)
            ).fetchone()[0]
            db.execute(
                "UPDATE menu_batches SET status='queued',updated_at=? WHERE id=?",
                (stamp(), batch_id),
            )
            self._reserve(
                db, batch_id, batch["owner_id"], candidates, max(0, KINDS[batch["kind"]][2] - count)
            )
            statuses = [
                r[0]
                for r in db.execute(
                    "SELECT status FROM menu_batch_items WHERE batch_id=?", (batch_id,)
                )
            ]
            status = (
                "queued"
                if any(s in {"queued", "sending"} for s in statuses)
                else "uncertain"
                if "uncertain" in statuses
                else "sent"
                if statuses
                else "empty"
            )
            db.execute("UPDATE menu_batches SET status=? WHERE id=?", (status, batch_id))

    def _finish(self, db, item, status, *, message_id=None, error=None):
        db.execute(
            "UPDATE menu_batch_items SET status=?,telegram_message_id=?,last_error=?,"
            "lease_until=NULL WHERE id=?",
            (status, message_id, error, item["id"]),
        )
        if item["item_kind"] in {"news", "video"}:
            table = "deliveries" if item["item_kind"] == "news" else "video_recommendations"
            column = "delivery_status" if item["item_kind"] == "news" else "status"
            db.execute(
                f"UPDATE {table} SET {column}=?,telegram_message_id=? WHERE id=?",
                ("failed" if status == "skipped" else status, message_id, item["delivery_id"]),
            )
            if item["item_kind"] == "news" and status == "sent":
                db.execute(
                    "UPDATE deliveries SET delivered_at=? WHERE id=?",
                    (stamp(), item["delivery_id"]),
                )
        left = db.execute(
            "SELECT 1 FROM menu_batch_items WHERE batch_id=? AND status IN('queued','sending')",
            (item["batch_id"],),
        ).fetchone()
        if not left:
            uncertain = db.execute(
                "SELECT 1 FROM menu_batch_items WHERE batch_id=? AND status!='sent'",
                (item["batch_id"],),
            ).fetchone()
            db.execute(
                "UPDATE menu_batches SET status=?,updated_at=? WHERE id=?",
                ("uncertain" if uncertain else "sent", stamp(), item["batch_id"]),
            )

    def finish(self, item, status, *, message_id=None, error=None):
        with self.repository.connection() as db:
            self._finish(db, item, status, message_id=message_id, error=error)
