from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from app.config import SourceRegistry
from app.db.card_predicates import finished_card_sql
from app.db.source_availability import available_source_sql
from app.research.detail_policy import (
    DETAIL_PROMPT_VERSION,
    DETAIL_SCHEMA_VERSION,
    current_detail_policy_sql,
)
from app.research.policy import current_fact_policy_sql


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


DETAIL_REQUEST_STATUSES = {
    "queued",
    "running",
    "ready",
    "sent",
    "insufficient_evidence",
    "budget_blocked",
    "failed",
}
DETAIL_REQUEST_STAGES = {
    "queued",
    "planning",
    "search",
    "facts",
    "writing",
    "delivery",
    "complete",
}
DETAIL_TERMINAL_FAILURE_STATUSES = {
    "insufficient_evidence",
    "budget_blocked",
    "failed",
}


def _all_detail_policy_sql(alias="detailed_articles"):
    return (
        "("
        + " OR ".join(
            f"({alias}.depth = '{depth}' AND {current_detail_policy_sql(alias, depth=depth)})"
            for depth in ("legacy", "expanded", "article")
        )
        + ")"
    )


def _as_utc_datetime(value: str | None = None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _detail_request_row(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["request_id"] = int(result["id"])
    result["cache_hit"] = bool(result.get("cache_hit"))
    result["reader_context"] = json.loads(result.get("reader_context_json") or "{}")
    return result


def _detailed_article_row(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    for source, target in (
        ("provider_response_metadata_json", "provider_response_metadata"),
        ("fact_snapshot_json", "fact_snapshot"),
        ("document_ids_json", "document_ids"),
        ("structured_output_json", "structured_output"),
        ("source_urls_json", "source_urls"),
        ("validation_errors_json", "validation_errors"),
        ("reader_context_json", "reader_context"),
    ):
        result[target] = json.loads(result.pop(source))
    return result


def _intelligence_product_row(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    for source, target in (
        ("payload_json", "payload"),
        ("source_event_ids_json", "source_event_ids"),
        ("validation_errors_json", "validation_errors"),
    ):
        result[target] = json.loads(result.pop(source))
    return result


class Repository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.migrations_path = Path(__file__).with_name("migrations")

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 5000")
            # WAL persists in the database. Don't request a mode transition on
            # every read. Concurrent first opens can still race; journal-mode
            # changes may return SQLITE_BUSY without honoring busy_timeout.
            deadline = time.monotonic() + 5
            while True:
                try:
                    mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
                    if mode.lower() != "wal":
                        connection.execute("PRAGMA journal_mode = WAL")
                    break
                except sqlite3.OperationalError as exc:
                    if (
                        getattr(exc, "sqlite_errorcode", None)
                        not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)
                        or time.monotonic() >= deadline
                    ):
                        raise
                    time.sleep(0.025)
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def migrate(self) -> list[str]:
        applied_now: list[str] = []
        with self.connection() as connection:
            # Hold one transaction across version checks and schema changes. executescript
            # would implicitly commit and race another CLI/service starting simultaneously.
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
                """
            )
            applied = {
                row["version"]
                for row in connection.execute("SELECT version FROM schema_migrations")
            }
            for migration_path in sorted(self.migrations_path.glob("*.sql")):
                version = migration_path.stem
                if version in applied:
                    continue
                statement = ""
                for line in migration_path.read_text(encoding="utf-8").splitlines(keepends=True):
                    statement += line
                    if sqlite3.complete_statement(statement):
                        connection.execute(statement)
                        statement = ""
                if statement.strip():
                    connection.execute(statement)
                connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (version, utc_now()),
                )
                applied_now.append(version)
        return applied_now

    def sync_sources(self, registry: SourceRegistry) -> int:
        now = utc_now()
        with self.connection() as connection:
            for source in registry.sources:
                connection.execute(
                    """
                    INSERT INTO sources (
                        id, name, kind, locator, role, enabled, candidate_enabled,
                        priority, region, language, topics_json,
                        poll_interval_minutes, calibration_lookback_days,
                        production_max_age_days, daily_candidate_cap, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        name = excluded.name,
                        kind = excluded.kind,
                        locator = excluded.locator,
                        role = excluded.role,
                        enabled = excluded.enabled,
                        candidate_enabled = excluded.candidate_enabled,
                        priority = excluded.priority,
                        region = excluded.region,
                        language = excluded.language,
                        topics_json = excluded.topics_json,
                        poll_interval_minutes = excluded.poll_interval_minutes,
                        calibration_lookback_days = excluded.calibration_lookback_days,
                        production_max_age_days = excluded.production_max_age_days,
                        daily_candidate_cap = excluded.daily_candidate_cap,
                        updated_at = excluded.updated_at
                    """,
                    (
                        source.id,
                        source.name,
                        source.kind.value,
                        source.locator,
                        source.role.value,
                        int(source.enabled),
                        int(source.candidate_enabled),
                        source.priority,
                        source.region,
                        source.language,
                        json.dumps(source.topics, ensure_ascii=False),
                        source.poll_interval_minutes,
                        source.calibration_lookback_days,
                        source.production_max_age_days,
                        source.daily_candidate_cap,
                        now,
                    ),
                )
                connection.execute(
                    "UPDATE sources SET adapter_id=?, strategy=?, admission_mode=? WHERE id=?",
                    (source.adapter_id, source.strategy, source.admission_mode, source.id),
                )
            # A runtime registry is the source of truth for X handles. Disable
            # stale rows after a rename/removal instead of leaving an old
            # account eligible for collection or radar seeding.
            x_source_ids = [source.id for source in registry.sources if source.kind.value == "x"]
            if x_source_ids:
                placeholders = ",".join("?" for _ in x_source_ids)
                connection.execute(
                    f"""
                    UPDATE sources
                    SET enabled = 0, candidate_enabled = 0, updated_at = ?
                    WHERE kind = 'x' AND id NOT IN ({placeholders})
                    """,
                    (now, *x_source_ids),
                )
        return len(registry.sources)

    def get_source_cursor(self, source_id: str) -> str | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT last_cursor FROM sources WHERE id = ?", (source_id,)
            ).fetchone()
        return row["last_cursor"] if row else None

    def source_collection_state(self, source_id: str) -> dict[str, Any] | None:
        """Return the small state needed to decide whether a paid source is due."""
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT last_cursor, last_success_at, last_error
                FROM sources
                WHERE id = ?
                """,
                (source_id,),
            ).fetchone()
        return dict(row) if row else None

    def mark_source_success(
        self,
        source_id: str,
        *,
        cursor: str | None,
        success_at: str,
    ) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                UPDATE sources
                SET last_cursor = ?, last_success_at = ?, last_error = NULL, updated_at = ?
                WHERE id = ?
                """,
                (cursor, success_at, utc_now(), source_id),
            )

    def mark_source_error(self, source_id: str, error: str) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                UPDATE sources SET last_error = ?, updated_at = ? WHERE id = ?
                """,
                (error[:2000], utc_now(), source_id),
            )

    def insert_source_item(
        self,
        *,
        source_id: str,
        external_id: str,
        source_url: str,
        original_text: str,
        original_language: str,
        published_at: str,
        outbound_urls: list[str],
        media_metadata: dict[str, Any] | None,
        payload_hash: str,
        raw_payload: dict[str, Any],
    ) -> bool:
        return (
            self.insert_source_item_with_id(
                source_id=source_id,
                external_id=external_id,
                source_url=source_url,
                original_text=original_text,
                original_language=original_language,
                published_at=published_at,
                outbound_urls=outbound_urls,
                media_metadata=media_metadata,
                payload_hash=payload_hash,
                raw_payload=raw_payload,
            )
            is not None
        )

    def insert_source_item_with_id(
        self,
        *,
        source_id: str,
        external_id: str,
        source_url: str,
        original_text: str,
        original_language: str,
        published_at: str,
        outbound_urls: list[str],
        media_metadata: dict[str, Any] | None,
        payload_hash: str,
        raw_payload: dict[str, Any],
    ) -> int | None:
        """Insert one raw item and return its ID, or None for an existing duplicate."""
        with self.connection() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO source_items(
                    source_id, external_id, source_url, original_text,
                    original_language, published_at, outbound_urls_json,
                    media_metadata_json, payload_hash, raw_payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_id,
                    external_id,
                    source_url,
                    original_text,
                    original_language,
                    published_at,
                    json.dumps(outbound_urls, ensure_ascii=False),
                    json.dumps(media_metadata, ensure_ascii=False) if media_metadata else None,
                    payload_hash,
                    json.dumps(raw_payload, ensure_ascii=False, default=str),
                ),
            )
            return int(cursor.lastrowid) if cursor.rowcount == 1 else None

    def source_item_count(self) -> int:
        with self.connection() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM source_items").fetchone()[0])

    def source_event_count_on_date(self, source_id: str, event_date: str) -> int:
        with self.connection() as connection:
            return int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM events
                    JOIN source_items
                        ON source_items.id = events.primary_source_item_id
                    WHERE source_items.source_id = ?
                      AND substr(events.event_at, 1, 10) = ?
                    """,
                    (source_id, event_date),
                ).fetchone()[0]
            )

    def source_yield_rows(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    sources.id,
                    sources.name,
                    sources.kind,
                    sources.role,
                    sources.enabled,
                    sources.candidate_enabled,
                    sources.last_success_at,
                    sources.last_error,
                    (SELECT COUNT(*) FROM source_items i
                        WHERE i.source_id = sources.id) AS items,
                    (SELECT COUNT(*) FROM source_items i
                        WHERE i.source_id = sources.id
                          AND i.processing_status = 'event_attached') AS accepted_items,
                    (SELECT COUNT(*) FROM source_items i
                        WHERE i.source_id = sources.id
                          AND i.processing_status IN (
                              'rejected', 'history_only'
                          )) AS rejected_items,
                    (SELECT COUNT(*)
                        FROM events e
                        JOIN source_items i ON i.id = e.primary_source_item_id
                        WHERE i.source_id = sources.id) AS events,
                    (SELECT COUNT(*)
                        FROM cards c
                        JOIN events e ON e.id = c.event_id
                        JOIN source_items i ON i.id = e.primary_source_item_id
                        WHERE i.source_id = sources.id
                          AND c.validation_status = 'valid') AS valid_cards,
                    (SELECT COUNT(*)
                        FROM deliveries d
                        JOIN cards c ON c.id = d.card_id
                        JOIN events e ON e.id = c.event_id
                        JOIN source_items i ON i.id = e.primary_source_item_id
                        WHERE i.source_id = sources.id
                          AND d.delivery_status = 'sent') AS sent,
                    (SELECT COUNT(*)
                        FROM feedback_events f
                        JOIN events e ON e.id = f.event_id
                        JOIN source_items i ON i.id = e.primary_source_item_id
                        WHERE i.source_id = sources.id
                          AND f.reaction IS NOT NULL) AS reactions
                FROM sources
                ORDER BY sources.kind, sources.priority, sources.id
                """
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["acceptance_rate"] = (
                round(item["accepted_items"] / item["items"], 3) if item["items"] else None
            )
            result.append(item)
        return result

    def unprocessed_source_items(self, limit: int = 200) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    source_items.*,
                    sources.name AS source_name,
                    sources.kind AS source_kind,
                    sources.locator AS source_locator,
                    sources.role AS source_role,
                    sources.candidate_enabled,
                    sources.priority AS source_priority,
                    sources.region AS source_region,
                    sources.topics_json AS source_topics_json,
                    sources.production_max_age_days,
                    sources.calibration_lookback_days
                FROM source_items
                JOIN sources ON sources.id = source_items.source_id
                WHERE source_items.processing_status = 'new'
                ORDER BY source_items.published_at DESC, source_items.id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def recent_fixed_signals(self, since: str, limit: int = 40) -> list[dict[str, Any]]:
        """Return compact non-search signals that can seed broad discovery agents."""
        with self.connection() as connection:
            rows = connection.execute(
                """
                WITH ranked_signals AS (
                    SELECT
                        source_items.original_text,
                        source_items.source_url,
                        source_items.outbound_urls_json,
                        source_items.published_at,
                        source_items.id,
                        source_items.discovery_mode,
                        sources.id AS source_id,
                        sources.name AS source_name,
                        sources.kind AS source_kind,
                        sources.role AS source_role,
                        sources.admission_mode,
                        ROW_NUMBER() OVER (
                            PARTITION BY sources.id
                            ORDER BY source_items.published_at DESC, source_items.id DESC
                        ) AS source_rank
                    FROM source_items
                    JOIN sources ON sources.id = source_items.source_id
                    WHERE sources.kind IN ('telegram', 'x', 'rss', 'github', 'reddit',
                                           'youtube', 'v2ex', 'weibo', 'zhihu')
                      AND COALESCE((SELECT CASE WHEN rs.value='true' THEN 1 ELSE 0 END
                          FROM runtime_settings rs WHERE rs.key='source_enabled:'||sources.id),
                          sources.enabled) = 1
                      AND sources.admission_mode != 'history'
                      AND NOT EXISTS(SELECT 1 FROM runtime_settings rs
                          WHERE rs.key='source_enabled:' || sources.id AND rs.value='false')
                      AND source_items.published_at >= ?
                ), mixed_kinds AS (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY source_kind ORDER BY published_at DESC, id DESC
                    ) AS kind_rank
                    FROM ranked_signals WHERE source_rank <= 2
                )
                SELECT
                    id AS source_item_id,
                    source_id,
                    original_text,
                    source_url,
                    outbound_urls_json,
                    published_at,
                    source_name,
                    source_kind,
                    source_role,
                    admission_mode,
                    discovery_mode
                FROM mixed_kinds
                ORDER BY kind_rank, published_at DESC, id DESC
                LIMIT ?
                """,
                (since, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def recent_event_memory(self, since: str, limit: int = 120) -> list[dict[str, Any]]:
        """Return a compact seen-event memory for prompt and exact URL suppression."""
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    events.id AS event_id,
                    events.headline,
                    events.event_at,
                    source_items.source_url,
                    source_items.outbound_urls_json
                FROM events
                JOIN source_items ON source_items.id = events.primary_source_item_id
                JOIN sources ON sources.id = source_items.source_id
                WHERE events.event_at >= ?
                  AND events.status NOT LIKE 'rejected%'
                  AND (
                      sources.kind NOT IN ('telegram', 'x')
                      OR EXISTS (
                          SELECT 1
                          FROM cards
                          WHERE cards.event_id = events.id
                            AND cards.validation_status IN ('valid', 'candidate_preview')
                      )
                  )
                ORDER BY events.event_at DESC, events.id DESC
                LIMIT ?
                """,
                (since, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def pending_event_source_items(self, limit: int = 500) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    events.id AS event_id,
                    events.entities_json,
                    events.topics_json,
                    events.regions_json,
                    source_items.*,
                    sources.name AS source_name,
                    sources.kind AS source_kind,
                    sources.role AS source_role,
                    sources.priority AS source_priority,
                    sources.region AS source_region,
                    sources.topics_json AS source_topics_json
                FROM events
                JOIN source_items ON source_items.id = events.primary_source_item_id
                JOIN sources ON sources.id = source_items.source_id
                WHERE events.status = 'candidate'
                  AND NOT EXISTS (
                      SELECT 1 FROM cards
                      WHERE cards.event_id = events.id
                        AND cards.validation_status = 'valid'
                  )
                ORDER BY events.score DESC, events.latest_source_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def update_pending_event_editorial_score(
        self,
        event_id: int,
        *,
        base_score: float,
        score_features: dict[str, Any],
        entities: dict[str, list[str]],
        topics: list[str],
        regions: list[str],
        status: str = "candidate",
        decision_reason: str = "v7a_editorial_reclassification",
    ) -> None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT entities_json, topics_json, regions_json FROM events WHERE id = ?",
                (event_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"Unknown event_id: {event_id}")
            current_entities = json.loads(row["entities_json"] or "{}")
            merged_entities = {
                key: list(dict.fromkeys([*current_entities.get(key, []), *entities.get(key, [])]))
                for key in set(current_entities) | set(entities)
            }
            merged_topics = list(dict.fromkeys([*json.loads(row["topics_json"] or "[]"), *topics]))
            merged_regions = list(
                dict.fromkeys([*json.loads(row["regions_json"] or "[]"), *regions])
            )
            independent_source_count = int(
                connection.execute(
                    """
                    SELECT COUNT(DISTINCT CASE
                        WHEN sources.kind = 'web' THEN source_items.source_url
                        ELSE source_items.source_id
                    END)
                    FROM event_source_items
                    JOIN source_items
                      ON source_items.id = event_source_items.source_item_id
                    JOIN sources ON sources.id = source_items.source_id
                    WHERE event_source_items.event_id = ?
                    """,
                    (event_id,),
                ).fetchone()[0]
            )
            corroboration_bonus = min(
                1.5,
                max(0, independent_source_count - 1) * 0.75,
            )
            merged_features = {
                **score_features,
                "base_score": base_score,
                "corroborating_sources": corroboration_bonus,
            }
            connection.execute(
                """
                UPDATE events
                SET status = ?, score = ?, score_features_json = ?,
                    entities_json = ?, topics_json = ?, regions_json = ?,
                    decision_reason = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    round(base_score + corroboration_bonus, 3),
                    json.dumps(merged_features, ensure_ascii=False),
                    json.dumps(merged_entities, ensure_ascii=False),
                    json.dumps(merged_topics, ensure_ascii=False),
                    json.dumps(merged_regions, ensure_ascii=False),
                    decision_reason,
                    utc_now(),
                    event_id,
                ),
            )

    def mark_source_item_status(
        self,
        source_item_id: int,
        status: str,
        error: str | None = None,
    ) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                UPDATE source_items
                SET processing_status = ?, processing_error = ?
                WHERE id = ?
                """,
                (status, error[:2000] if error else None, source_item_id),
            )

    def attach_item_to_event(
        self,
        *,
        source_item_id: int,
        event_type: str,
        headline: str,
        event_at: str,
        primary_url: str,
        fingerprint: str,
        score: float,
        score_features: dict[str, Any],
        decision_reason: str,
        entities: dict[str, list[str]] | None = None,
        topics: list[str] | None = None,
        regions: list[str] | None = None,
    ) -> tuple[int, bool]:
        now = utc_now()
        with self.connection() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO events(
                    status, event_type, headline, event_at, first_seen_at,
                    latest_source_at, primary_source_item_id, score,
                    score_features_json, fingerprint, decision_reason
                ) VALUES ('candidate', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_type,
                    headline,
                    event_at,
                    now,
                    event_at,
                    source_item_id,
                    score,
                    json.dumps(score_features, ensure_ascii=False),
                    fingerprint,
                    decision_reason,
                ),
            )
            created = cursor.rowcount == 1
            row = connection.execute(
                """
                SELECT id, entities_json, topics_json, regions_json,
                       score, score_features_json
                FROM events WHERE fingerprint = ?
                """,
                (fingerprint,),
            ).fetchone()
            if row is None:  # pragma: no cover - defensive guard
                raise RuntimeError("Event insert/select failed")
            # An old fingerprint may point to a retained historical duplicate.
            # Route new evidence to its canonical event without deleting history.
            seen_aliases: set[int] = set()
            while int(row["id"]) not in seen_aliases:
                seen_aliases.add(int(row["id"]))
                alias = connection.execute(
                    "SELECT value FROM runtime_settings WHERE key = ?",
                    (f"duplicate_event:{row['id']}",),
                ).fetchone()
                if alias is None or not str(alias["value"]).isdigit():
                    break
                target = connection.execute(
                    "SELECT id, entities_json, topics_json, regions_json, "
                    "score, score_features_json FROM events WHERE id = ?",
                    (int(alias["value"]),),
                ).fetchone()
                if target is None or int(target["id"]) in seen_aliases:
                    break
                row = target
            event_id = int(row["id"])
            current_entities = json.loads(row["entities_json"] or "{}")
            merged_entities: dict[str, list[str]] = {}
            for key in set(current_entities) | set(entities or {}):
                old_values = current_entities.get(key, [])
                new_values = (entities or {}).get(key, [])
                merged_entities[key] = list(dict.fromkeys([*old_values, *new_values]))
            merged_topics = list(
                dict.fromkeys([*json.loads(row["topics_json"] or "[]"), *(topics or [])])
            )
            merged_regions = list(
                dict.fromkeys([*json.loads(row["regions_json"] or "[]"), *(regions or [])])
            )
            existing_features = json.loads(row["score_features_json"] or "{}")
            previous_base_score = float(existing_features.get("base_score", row["score"] or 0))
            base_score = max(previous_base_score, score)
            strongest_features = (
                score_features if score >= previous_base_score else existing_features
            )
            merged_features = {
                **strongest_features,
                "base_score": base_score,
                "corroborating_sources": 0.0,
            }
            evidence_cursor = connection.execute(
                """
                INSERT OR IGNORE INTO event_source_items(
                    event_id, source_item_id, evidence_role
                ) VALUES (?, ?, ?)
                """,
                (event_id, source_item_id, "primary" if created else "corroborating"),
            )
            if evidence_cursor.rowcount == 1:
                connection.execute(
                    """
                    UPDATE events
                    SET status = CASE
                        WHEN status = 'research_insufficient' THEN 'candidate'
                        ELSE status
                    END
                    WHERE id = ?
                    """,
                    (event_id,),
                )
            connection.execute(
                """
                UPDATE events
                SET latest_source_at = MAX(latest_source_at, ?),
                    score = MAX(score, ?),
                    score_features_json = ?,
                    entities_json = ?,
                    topics_json = ?,
                    regions_json = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    event_at,
                    score,
                    json.dumps(merged_features, ensure_ascii=False),
                    json.dumps(merged_entities, ensure_ascii=False),
                    json.dumps(merged_topics, ensure_ascii=False),
                    json.dumps(merged_regions, ensure_ascii=False),
                    now,
                    event_id,
                ),
            )
            independent_source_count = int(
                connection.execute(
                    """
                    SELECT COUNT(DISTINCT CASE
                        WHEN sources.kind = 'web' THEN source_items.source_url
                        ELSE source_items.source_id
                    END)
                    FROM event_source_items
                    JOIN source_items
                      ON source_items.id = event_source_items.source_item_id
                    JOIN sources ON sources.id = source_items.source_id
                    WHERE event_source_items.event_id = ?
                    """,
                    (event_id,),
                ).fetchone()[0]
            )
            corroboration_bonus = min(
                1.5,
                max(0, independent_source_count - 1) * 0.75,
            )
            merged_features["corroborating_sources"] = corroboration_bonus
            connection.execute(
                """
                UPDATE events
                SET score = ?, score_features_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    round(base_score + corroboration_bonus, 3),
                    json.dumps(merged_features, ensure_ascii=False),
                    now,
                    event_id,
                ),
            )
            connection.execute(
                """
                UPDATE source_items
                SET processing_status = 'event_attached', processing_error = NULL
                WHERE id = ?
                """,
                (source_item_id,),
            )
        return event_id, created

    def update_event_classification(
        self,
        event_id: int,
        *,
        event_type: str | None = None,
        entities: dict[str, list[str]] | None = None,
        topics: list[str] | None = None,
        regions: list[str] | None = None,
    ) -> None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT entities_json, topics_json, regions_json FROM events WHERE id = ?",
                (event_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"Unknown event_id: {event_id}")
            current_entities = json.loads(row["entities_json"] or "{}")
            for key, values in (entities or {}).items():
                current_entities[key] = list(
                    dict.fromkeys([*current_entities.get(key, []), *values])
                )
            current_topics = list(
                dict.fromkeys([*json.loads(row["topics_json"] or "[]"), *(topics or [])])
            )
            current_regions = list(
                dict.fromkeys([*json.loads(row["regions_json"] or "[]"), *(regions or [])])
            )
            connection.execute(
                """
                UPDATE events
                SET event_type = COALESCE(?, event_type),
                    entities_json = ?, topics_json = ?, regions_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    event_type,
                    json.dumps(current_entities, ensure_ascii=False),
                    json.dumps(current_topics, ensure_ascii=False),
                    json.dumps(current_regions, ensure_ascii=False),
                    utc_now(),
                    event_id,
                ),
            )

    def events_without_cards(
        self,
        limit: int = 20,
        *,
        event_id: int | None = None,
        require_fact_sheet: bool = False,
        require_ready_dossier: bool = False,
    ) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    events.*,
                    source_items.original_text,
                    source_items.original_language,
                    source_items.source_url,
                    source_items.outbound_urls_json,
                    source_items.published_at,
                    source_items.discovery_mode,
                    sources.id AS source_id,
                    sources.kind AS source_kind,
                    sources.admission_mode,
                    sources.name AS source_name,
                    sources.locator AS source_locator,
                    sources.role AS source_role
                FROM events
                JOIN source_items ON source_items.id = events.primary_source_item_id
                JOIN sources ON sources.id = source_items.source_id
                LEFT JOIN cards
                    ON cards.event_id = events.id AND {finished_card_sql()}
                WHERE events.status = 'candidate'
                  AND cards.id IS NULL
                  AND (? IS NULL OR events.id = ?)
                  AND (
                    ? = 0 OR EXISTS (
                        SELECT 1 FROM event_fact_sheets
                        WHERE event_fact_sheets.event_id = events.id
                      AND event_fact_sheets.validation_status = 'valid'
                      AND {current_fact_policy_sql()}
                    )
                  )
                  AND (
                    ? = 0 OR EXISTS (
                        SELECT 1 FROM event_dossiers
                        WHERE event_dossiers.event_id = events.id
                          AND event_dossiers.validation_status = 'valid'
                          AND event_dossiers.decision = 'ready'
                    )
                  )
                ORDER BY events.score DESC, events.latest_source_at DESC
                LIMIT ?
                """,
                (
                    event_id,
                    event_id,
                    int(require_fact_sheet),
                    int(require_ready_dossier),
                    limit,
                ),
            ).fetchall()
        return self.with_event_signal_metadata([dict(row) for row in rows])

    def events_needing_research(self, limit: int = 1) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    events.*,
                    source_items.source_url,
                    source_items.outbound_urls_json,
                    source_items.discovery_mode,
                    sources.id AS source_id,
                    sources.kind AS source_kind,
                    sources.admission_mode,
                    sources.role AS source_role
                FROM events
                JOIN source_items ON source_items.id = events.primary_source_item_id
                JOIN sources ON sources.id = source_items.source_id
                WHERE events.status = 'candidate'
                  AND NOT EXISTS (
                      SELECT 1 FROM event_fact_sheets
                      WHERE event_fact_sheets.event_id = events.id
                        AND event_fact_sheets.validation_status = 'valid'
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM cards
                      WHERE cards.event_id = events.id
                        AND cards.validation_status = 'valid'
                  )
                ORDER BY events.score DESC, events.latest_source_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return self.with_event_signal_metadata([dict(row) for row in rows])

    def event_evidence_urls(self, event_id: int) -> list[str]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT source_items.source_url, source_items.outbound_urls_json
                FROM event_source_items
                JOIN source_items ON source_items.id = event_source_items.source_item_id
                WHERE event_source_items.event_id = ?
                ORDER BY CASE event_source_items.evidence_role
                    WHEN 'primary' THEN 0 ELSE 1 END,
                    source_items.published_at
                """,
                (event_id,),
            ).fetchall()
        urls: list[str] = []
        for row in rows:
            for url in [*json.loads(row["outbound_urls_json"] or "[]"), row["source_url"]]:
                if url not in urls:
                    urls.append(url)
        return urls

    def event_context(self, event_id: int) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT
                    events.*,
                    source_items.original_text,
                    source_items.source_url,
                    source_items.published_at,
                    sources.name AS source_name,
                    sources.role AS source_role
                FROM events
                JOIN source_items ON source_items.id = events.primary_source_item_id
                JOIN sources ON sources.id = source_items.source_id
                WHERE events.id = ?
                """,
                (event_id,),
            ).fetchone()
        return dict(row) if row else None

    def event_evidence_items(self, event_id: int) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    source_items.id,
                    source_items.original_text,
                    source_items.published_at,
                    source_items.source_url,
                    sources.name AS source_name,
                    sources.role AS source_role
                FROM event_source_items
                JOIN source_items ON source_items.id = event_source_items.source_item_id
                JOIN sources ON sources.id = source_items.source_id
                WHERE event_source_items.event_id = ?
                ORDER BY CASE event_source_items.evidence_role
                    WHEN 'primary' THEN 0 ELSE 1 END,
                    source_items.published_at
                """,
                (event_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def event_trigger_sources(self, event_id: int) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    source_items.id AS source_item_id,
                    source_items.source_url,
                    source_items.published_at,
                    sources.id AS source_id,
                    sources.name AS source_name,
                    sources.kind AS source_kind,
                    sources.role AS source_role,
                    sources.admission_mode,
                    event_source_items.evidence_role
                FROM event_source_items
                JOIN source_items ON source_items.id = event_source_items.source_item_id
                JOIN sources ON sources.id = source_items.source_id
                WHERE event_source_items.event_id = ?
                ORDER BY source_items.published_at, source_items.id
                """,
                (event_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def with_event_signal_metadata(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Batch-load traceable ranking metadata, without certifying source claims.

        event_source_items.primary means the first item in a cluster, not a
        journalistic primary source. Only the configured source role/admission
        is used below; untrusted provider payloads cannot self-verify evidence.
        """
        from app.pipeline.signal_quality import assess_signal_quality

        ids = list(dict.fromkeys(int(event["id"]) for event in events))
        if not ids:
            return []
        observations: dict[int, list[dict[str, Any]]] = {event_id: [] for event_id in ids}
        with self.connection() as connection:
            for start in range(0, len(ids), 500):
                batch = ids[start : start + 500]
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    f"""SELECT esi.event_id, si.id, si.external_id, si.source_url,
                        si.published_at, si.discovery_mode, si.raw_payload_json,
                        s.id AS source_id, s.kind AS source_kind, s.role AS source_role,
                        s.admission_mode, esi.evidence_role AS cluster_role
                    FROM event_source_items esi
                    JOIN source_items si ON si.id = esi.source_item_id
                    JOIN sources s ON s.id = si.source_id
                    WHERE esi.event_id IN ({placeholders}) ORDER BY si.id""",
                    batch,
                ).fetchall()
                for row in rows:
                    item = dict(row)
                    event_id = item.pop("event_id")
                    try:
                        payload = json.loads(item.pop("raw_payload_json") or "{}")
                    except (ValueError, TypeError):
                        payload = {}
                    # Provenance may group syndication, never confer credibility.
                    for key in (
                        "original_source_url",
                        "syndicated_from",
                        "publisher_family",
                        "source_family",
                    ):
                        value = payload.get(key) if isinstance(payload, dict) else None
                        if isinstance(value, str) and 0 < len(value) <= 2048:
                            item[key] = value
                    if (
                        item["source_id"].startswith("web_agent_")
                        and item["discovery_mode"] == "feed"
                    ):
                        item["discovery_mode"] = (
                            "explore"
                            if item["source_id"] == "web_agent_personal_rare_signals"
                            else "search"
                        )
                    item["evidence_role"] = item["source_role"]
                    observations[event_id].append(item)
        enriched = []
        for event in events:
            rows = observations[int(event["id"])]
            try:
                features = event.get("score_features") or event.get("score_features_json") or {}
                if isinstance(features, str):
                    features = json.loads(features)
                features = dict(features) if isinstance(features, dict) else {}
            except (ValueError, TypeError):
                features = {}
            metadata = {
                "source_ids": sorted({row["source_id"] for row in rows}),
                "platforms": sorted({row["source_kind"] for row in rows}),
                "discovery_modes": sorted({row["discovery_mode"] for row in rows}),
                "signal_quality": assess_signal_quality(rows),
            }
            # Historical field remains readable; it is an attention bonus, not verification.
            if "corroborating_sources" in features:
                metadata["spread_bonus"] = features["corroborating_sources"]
            enriched.append(
                {
                    **event,
                    **metadata,
                    "signal_observations": rows,
                    "score_features": {**features, **metadata},
                }
            )
        return enriched

    def event_ids_for_source_items(self, source_item_ids: list[int]) -> dict[int, int]:
        """Map newly inserted source items to their surviving event after dedupe."""
        ordered_ids = list(dict.fromkeys(source_item_ids))
        if not ordered_ids:
            return {}
        placeholders = ", ".join("?" for _ in ordered_ids)
        with self.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT source_item_id, event_id
                FROM event_source_items
                WHERE source_item_id IN ({placeholders})
                """,
                ordered_ids,
            ).fetchall()
        return {int(row["source_item_id"]): int(row["event_id"]) for row in rows}

    def record_research_query(
        self,
        *,
        event_id: int,
        provider: str,
        query: str,
        freshness: str | None,
        status: str,
        results: list[dict[str, Any]],
        cost_usd: float,
        expires_at: str,
    ) -> int:
        with self.connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO research_queries(
                    event_id, provider, query, freshness, status,
                    result_count, results_json, cost_usd, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    provider,
                    query,
                    freshness,
                    status,
                    len(results),
                    json.dumps(results, ensure_ascii=False),
                    cost_usd,
                    expires_at,
                ),
            )
            return int(cursor.lastrowid)

    def upsert_research_document(
        self,
        *,
        canonical_url: str,
        url: str,
        source_class: str,
        retrieved_at: str,
        expires_at: str,
        title: str | None = None,
        publisher: str | None = None,
        published_at: str | None = None,
        content_text: str | None = None,
        content_hash: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        incoming_metadata = metadata or {}
        with self.connection() as connection:
            # A search result may rediscover a canonical URL whose full page was
            # already scraped for another event. Lock and keep the richer artifact:
            # a short snippet must never destroy shared full-text evidence.
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM research_documents WHERE canonical_url = ?",
                (canonical_url,),
            ).fetchone()
            if existing is not None and incoming_metadata.get("content_kind") == ("search_snippet"):
                try:
                    existing_metadata = json.loads(existing["metadata_json"] or "{}")
                except json.JSONDecodeError:
                    existing_metadata = {}
                existing_is_fulltext = existing_metadata.get("content_kind") == (
                    "scraped_fulltext"
                ) or (
                    not existing_metadata.get("content_kind")
                    and len(str(existing["content_text"] or "")) >= 400
                )
                if existing_is_fulltext:
                    url = str(existing["url"])
                    title = existing["title"]
                    publisher = existing["publisher"]
                    source_class = str(existing["source_class"])
                    published_at = existing["published_at"]
                    retrieved_at = str(existing["retrieved_at"])
                    expires_at = str(existing["expires_at"])
                    content_text = existing["content_text"]
                    content_hash = existing["content_hash"]
                    incoming_metadata = existing_metadata
            connection.execute(
                """
                INSERT INTO research_documents(
                    canonical_url, url, title, publisher, source_class,
                    published_at, retrieved_at, expires_at, content_text,
                    content_hash, metadata_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(canonical_url) DO UPDATE SET
                    url = excluded.url,
                    title = excluded.title,
                    publisher = excluded.publisher,
                    source_class = excluded.source_class,
                    published_at = excluded.published_at,
                    retrieved_at = excluded.retrieved_at,
                    expires_at = excluded.expires_at,
                    content_text = excluded.content_text,
                    content_hash = excluded.content_hash,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    canonical_url,
                    url,
                    title,
                    publisher,
                    source_class,
                    published_at,
                    retrieved_at,
                    expires_at,
                    content_text,
                    content_hash,
                    json.dumps(incoming_metadata, ensure_ascii=False),
                    utc_now(),
                ),
            )
            row = connection.execute(
                "SELECT id FROM research_documents WHERE canonical_url = ?",
                (canonical_url,),
            ).fetchone()
        if row is None:  # pragma: no cover - defensive guard
            raise RuntimeError("Research document insert/select failed")
        return int(row["id"])

    def attach_research_document(
        self,
        *,
        event_id: int,
        research_document_id: int,
        evidence_role: str,
        relevance: float | None = None,
    ) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO event_research_documents(
                    event_id, research_document_id, evidence_role, relevance
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(event_id, research_document_id) DO UPDATE SET
                    evidence_role = excluded.evidence_role,
                    relevance = excluded.relevance
                """,
                (event_id, research_document_id, evidence_role, relevance),
            )

    def event_research_documents(self, event_id: int) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT research_documents.*, event_research_documents.evidence_role,
                       event_research_documents.relevance
                FROM event_research_documents
                JOIN research_documents
                  ON research_documents.id = event_research_documents.research_document_id
                WHERE event_research_documents.event_id = ?
                ORDER BY CASE event_research_documents.evidence_role
                    WHEN 'primary' THEN 0
                    WHEN 'independent' THEN 1
                    WHEN 'corroborating' THEN 2
                    ELSE 3 END,
                    event_research_documents.relevance DESC,
                    research_documents.id
                """,
                (event_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def create_fact_sheet(
        self,
        *,
        event_id: int,
        schema_version: str,
        facts: list[dict[str, Any]],
        coverage: dict[str, Any],
        validation_status: str,
        expires_at: str,
        research_cost_usd: float = 0,
        validation_errors: list[str] | None = None,
    ) -> int:
        with self.connection() as connection:
            if validation_status == "valid":
                connection.execute(
                    """
                    UPDATE event_fact_sheets
                    SET validation_status = 'superseded'
                    WHERE event_id = ? AND validation_status = 'valid'
                    """,
                    (event_id,),
                )
            version = int(
                connection.execute(
                    """
                    SELECT COALESCE(MAX(fact_sheet_version), 0) + 1
                    FROM event_fact_sheets WHERE event_id = ?
                    """,
                    (event_id,),
                ).fetchone()[0]
            )
            cursor = connection.execute(
                """
                INSERT INTO event_fact_sheets(
                    event_id, fact_sheet_version, schema_version, facts_json,
                    coverage_json, validation_status, validation_errors_json,
                    research_cost_usd, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    version,
                    schema_version,
                    json.dumps(facts, ensure_ascii=False),
                    json.dumps(coverage, ensure_ascii=False),
                    validation_status,
                    json.dumps(validation_errors or [], ensure_ascii=False),
                    research_cost_usd,
                    expires_at,
                ),
            )
            return int(cursor.lastrowid)

    def latest_valid_fact_sheet(
        self, event_id: int, *, require_current_policy: bool = True
    ) -> dict[str, Any] | None:
        """Only current-policy facts may drive writing; opt-out is for audit/rebuild."""
        with self.connection() as connection:
            row = connection.execute(
                f"""
                SELECT * FROM event_fact_sheets
                WHERE event_id = ? AND validation_status = 'valid'
                  AND (? = 0 OR {current_fact_policy_sql()})
                ORDER BY fact_sheet_version DESC
                LIMIT 1
                """,
                (event_id, int(require_current_policy)),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["facts"] = json.loads(result.pop("facts_json"))
        result["coverage"] = json.loads(result.pop("coverage_json"))
        result["validation_errors"] = json.loads(result.pop("validation_errors_json"))
        return result

    def create_event_dossier(
        self,
        *,
        event_id: int,
        schema_version: str,
        fact_sheet_id: int | None,
        decision: str,
        decision_reason: str,
        payload: dict[str, Any],
        validation_status: str = "valid",
        validation_errors: list[str] | None = None,
    ) -> int:
        with self.connection() as connection:
            version = int(
                connection.execute(
                    """
                    SELECT COALESCE(MAX(dossier_version), 0) + 1
                    FROM event_dossiers WHERE event_id = ?
                    """,
                    (event_id,),
                ).fetchone()[0]
            )
            cursor = connection.execute(
                """
                INSERT INTO event_dossiers(
                    event_id, dossier_version, schema_version, fact_sheet_id,
                    decision, decision_reason, payload_json, validation_status,
                    validation_errors_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    version,
                    schema_version,
                    fact_sheet_id,
                    decision,
                    decision_reason,
                    json.dumps(payload, ensure_ascii=False),
                    validation_status,
                    json.dumps(validation_errors or [], ensure_ascii=False),
                ),
            )
            return int(cursor.lastrowid)

    def latest_event_dossier(self, event_id: int) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM event_dossiers
                WHERE event_id = ? AND validation_status = 'valid'
                ORDER BY dossier_version DESC
                LIMIT 1
                """,
                (event_id,),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["payload"] = json.loads(result.pop("payload_json"))
        result["validation_errors"] = json.loads(result.pop("validation_errors_json"))
        return result

    def event_has_valid_card(self, event_id: int) -> bool:
        with self.connection() as connection:
            row = connection.execute(
                f"""
                SELECT EXISTS(
                    SELECT 1 FROM cards
                    WHERE event_id = ? AND {finished_card_sql()}
                )
                """,
                (event_id,),
            ).fetchone()
        return bool(row[0])

    def create_card_media(
        self,
        *,
        card_id: int,
        policy_version: str,
        mode: str,
        preview_url: str | None,
        image_url: str | None,
        source_document_id: int | None,
        checks: dict[str, Any],
    ) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO card_media(
                    card_id, policy_version, mode, preview_url, image_url,
                    source_document_id, checks_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(card_id) DO UPDATE SET
                    policy_version = excluded.policy_version,
                    mode = excluded.mode,
                    preview_url = excluded.preview_url,
                    image_url = excluded.image_url,
                    source_document_id = excluded.source_document_id,
                    checks_json = excluded.checks_json
                """,
                (
                    card_id,
                    policy_version,
                    mode,
                    preview_url,
                    image_url,
                    source_document_id,
                    json.dumps(checks, ensure_ascii=False),
                ),
            )

    def card_media(self, card_id: int) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM card_media WHERE card_id = ?", (card_id,)
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["checks"] = json.loads(result.pop("checks_json"))
        return result

    def cards_missing_media(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT cards.id AS card_id, cards.event_id, cards.primary_preview_url
                FROM cards
                LEFT JOIN card_media ON card_media.card_id = cards.id
                WHERE cards.validation_status = 'valid' AND card_media.card_id IS NULL
                ORDER BY cards.created_at, cards.id
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def card_author_context(self, card_id: int) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT cards.*, events.event_type, events.headline AS event_headline
                FROM cards
                JOIN events ON events.id = cards.event_id
                WHERE cards.id = ? AND cards.validation_status = 'valid'
                """,
                (card_id,),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["structured_output"] = json.loads(result.pop("structured_output_json"))
        result["source_urls"] = json.loads(result.pop("source_urls_json"))
        result["validation_errors"] = json.loads(result.pop("validation_errors_json"))
        return result

    def create_author_draft(
        self,
        *,
        card_id: int,
        style_profile_version: str,
        prompt_version: str,
        model: str,
        generation_cost_usd: float,
        structured_output: dict[str, Any],
        rendered_html: str,
        validation_status: str = "valid",
        validation_errors: list[str] | None = None,
    ) -> int:
        with self.connection() as connection:
            version = int(
                connection.execute(
                    """
                    SELECT COALESCE(MAX(draft_version), 0) + 1
                    FROM author_drafts WHERE card_id = ?
                    """,
                    (card_id,),
                ).fetchone()[0]
            )
            cursor = connection.execute(
                """
                INSERT INTO author_drafts(
                    card_id, draft_version, style_profile_version, prompt_version,
                    model, generation_cost_usd, structured_output_json, rendered_html,
                    validation_status, validation_errors_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    card_id,
                    version,
                    style_profile_version,
                    prompt_version,
                    model,
                    generation_cost_usd,
                    json.dumps(structured_output, ensure_ascii=False),
                    rendered_html,
                    validation_status,
                    json.dumps(validation_errors or [], ensure_ascii=False),
                ),
            )
        return int(cursor.lastrowid)

    def latest_author_draft(self, card_id: int) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM author_drafts
                WHERE card_id = ? AND validation_status = 'valid'
                ORDER BY draft_version DESC
                LIMIT 1
                """,
                (card_id,),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["structured_output"] = json.loads(result.pop("structured_output_json"))
        result["validation_errors"] = json.loads(result.pop("validation_errors_json"))
        return result

    def fact_sheet_evidence_items(self, event_id: int) -> list[dict[str, Any]]:
        fact_sheet = self.latest_valid_fact_sheet(event_id)
        if fact_sheet is None:
            return []
        documents = {
            int(document["id"]): document for document in self.event_research_documents(event_id)
        }
        evidence: list[dict[str, Any]] = []
        for fact in fact_sheet["facts"]:
            document_ids = [
                int(document_id)
                for document_id in fact.get("document_ids", [])
                if int(document_id) in documents
            ]
            linked_documents = [documents[document_id] for document_id in document_ids]
            publishers = [
                str(document.get("publisher") or document.get("title") or "web research")
                for document in linked_documents
            ]
            evidence.append(
                {
                    "source_name": ", ".join(dict.fromkeys(publishers)) or "fact sheet",
                    "source_role": "research_fact",
                    "published_at": next(
                        (
                            str(document["published_at"])
                            for document in linked_documents
                            if document.get("published_at")
                        ),
                        fact_sheet["created_at"],
                    ),
                    "original_text": str(fact["text"]),
                    "evidence_quotes": fact.get("evidence_quotes", []),
                    # Local validation only; the card prompt omits full source
                    # context to avoid multiplying article tokens per fact.
                    "source_context": "\n\n".join(
                        str(document.get("content_text") or "") for document in linked_documents
                    ),
                    "fact_kind": str(fact.get("kind") or "event"),
                    "document_ids": document_ids,
                    "documents": [
                        {
                            "id": int(document["id"]),
                            "url": str(document["url"]),
                            "source_class": str(document["source_class"]),
                            "publisher": document.get("publisher"),
                            "title": document.get("title"),
                        }
                        for document in linked_documents
                    ],
                    "importance": {
                        "benchmark": 5,
                        "pricing": 5,
                        "limitation": 5,
                        "release": 4,
                        "capability": 3,
                        "availability": 2,
                    }.get(str(fact.get("kind")), 3),
                }
            )
        return evidence

    def prune_expired_research_cache(self, *, now: str | None = None) -> dict[str, int]:
        """Remove expired, unreviewed research while retaining delivered editorial memory."""
        cutoff = now or utc_now()
        with self.connection() as connection:
            queries = connection.execute(
                "DELETE FROM research_queries WHERE expires_at < ?", (cutoff,)
            ).rowcount
            fact_sheets = connection.execute(
                """
                DELETE FROM event_fact_sheets
                WHERE expires_at < ?
                  AND NOT EXISTS (
                    SELECT 1
                    FROM cards
                    LEFT JOIN deliveries ON deliveries.card_id = cards.id
                    LEFT JOIN feedback_events ON feedback_events.card_id = cards.id
                    WHERE cards.event_id = event_fact_sheets.event_id
                      AND (
                        deliveries.delivery_status = 'sent'
                        OR feedback_events.id IS NOT NULL
                      )
                  )
                """,
                (cutoff,),
            ).rowcount
            documents = connection.execute(
                """
                DELETE FROM research_documents
                WHERE expires_at < ?
                  AND NOT EXISTS (
                    SELECT 1
                    FROM event_research_documents
                    JOIN cards
                      ON cards.event_id = event_research_documents.event_id
                    LEFT JOIN deliveries ON deliveries.card_id = cards.id
                    LEFT JOIN feedback_events ON feedback_events.card_id = cards.id
                    WHERE event_research_documents.research_document_id = research_documents.id
                      AND (
                        deliveries.delivery_status = 'sent'
                        OR feedback_events.id IS NOT NULL
                      )
                  )
                """,
                (cutoff,),
            ).rowcount
        return {
            "queries": queries,
            "fact_sheets": fact_sheets,
            "documents": documents,
        }

    def reject_card(self, card_id: int, reason: str) -> None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT event_id FROM cards WHERE id = ?", (card_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"Unknown card_id: {card_id}")
            connection.execute(
                """
                UPDATE cards
                SET validation_status = 'rejected_editorial',
                    validation_errors_json = ?
                WHERE id = ?
                """,
                (json.dumps([reason], ensure_ascii=False), card_id),
            )
            connection.execute(
                "UPDATE events SET status = 'candidate', updated_at = ? WHERE id = ?",
                (utc_now(), row["event_id"]),
            )

    def hold_card_from_feedback(self, card_id: int, reason: str) -> None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT event_id FROM cards WHERE id = ?", (card_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"Unknown card_id: {card_id}")
            connection.execute(
                """
                UPDATE cards
                SET validation_status = 'rejected_user_feedback',
                    validation_errors_json = ?
                WHERE id = ?
                """,
                (json.dumps([reason], ensure_ascii=False), card_id),
            )
            connection.execute(
                """
                UPDATE events
                SET status = 'feedback_rejected', decision_reason = ?, updated_at = ?
                WHERE id = ?
                """,
                (reason, utc_now(), row["event_id"]),
            )

    def set_event_status(self, event_id: int, status: str, reason: str | None = None) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                UPDATE events
                SET status = ?, decision_reason = COALESCE(?, decision_reason), updated_at = ?
                WHERE id = ?
                """,
                (status, reason, utc_now(), event_id),
            )

    def dedupe_event_rows(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    events.id,
                    events.status,
                    events.headline,
                    events.event_type,
                    events.entities_json,
                    events.event_at,
                    events.latest_source_at,
                    events.score,
                    source_items.original_text,
                    source_items.outbound_urls_json,
                    source_items.source_url,
                    source_items.source_id,
                    source_items.discovery_mode,
                    sources.kind AS source_kind,
                    sources.role AS source_role,
                    EXISTS(SELECT 1 FROM cards WHERE cards.event_id = events.id) AS has_cards,
                    EXISTS(SELECT 1 FROM cards WHERE cards.event_id = events.id
                           AND cards.validation_status='valid') AS has_valid_card,
                    EXISTS(SELECT 1 FROM deliveries d JOIN cards c ON d.card_id=c.id
                           WHERE c.event_id=events.id) AS has_delivery,
                    EXISTS(
                        SELECT 1 FROM event_fact_sheets
                        WHERE event_fact_sheets.event_id = events.id
                    ) OR EXISTS(
                        SELECT 1 FROM event_dossiers
                        WHERE event_dossiers.event_id = events.id
                    ) AS has_editorial_memory,
                    EXISTS(
                        SELECT 1 FROM research_queries
                        WHERE research_queries.event_id = events.id
                    ) OR EXISTS(
                        SELECT 1 FROM event_research_documents
                        WHERE event_research_documents.event_id = events.id
                    ) AS has_research
                FROM events
                JOIN source_items ON source_items.id = events.primary_source_item_id
                JOIN sources ON sources.id = source_items.source_id
                WHERE events.status IN ('candidate', 'card_ready', 'research_insufficient')
                ORDER BY events.event_at DESC, events.score DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def link_duplicate_event(
        self, target_event_id: int, duplicate_event_id: int, *, reason: str
    ) -> bool:
        """Alias unpublished repeat discoveries without deleting any historical rows."""
        if target_event_id == duplicate_event_id:
            return False
        with self.connection() as connection:
            delivered = connection.execute(
                "SELECT 1 FROM deliveries d JOIN cards c ON d.card_id=c.id "
                "WHERE c.event_id=? LIMIT 1",
                (duplicate_event_id,),
            ).fetchone()
            if delivered:
                return False
            rows = connection.execute(
                "SELECT id,status FROM events WHERE id IN (?,?)",
                (target_event_id, duplicate_event_id),
            ).fetchall()
            if len(rows) != 2 or any(row["status"] == "duplicate" for row in rows):
                return False
            connection.execute(
                "INSERT OR IGNORE INTO event_source_items(event_id,source_item_id,evidence_role) "
                "SELECT ?,source_item_id,evidence_role FROM event_source_items WHERE event_id=?",
                (target_event_id, duplicate_event_id),
            )
            connection.execute(
                "INSERT OR IGNORE INTO event_research_documents("
                "event_id,research_document_id,evidence_role,relevance,created_at) "
                "SELECT ?,research_document_id,evidence_role,relevance,created_at "
                "FROM event_research_documents WHERE event_id=?",
                (target_event_id, duplicate_event_id),
            )
            connection.execute(
                "INSERT INTO runtime_settings(key,value,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET "
                "value=excluded.value,updated_at=excluded.updated_at",
                (f"duplicate_event:{duplicate_event_id}", str(target_event_id), utc_now()),
            )
            connection.execute(
                "UPDATE events SET status='duplicate',decision_reason=?,updated_at=? WHERE id=?",
                (
                    f"Same event as {target_event_id}: {reason}; history retained",
                    utc_now(),
                    duplicate_event_id,
                ),
            )
        return True

    def merge_events(self, target_event_id: int, duplicate_event_id: int) -> None:
        if target_event_id == duplicate_event_id:
            return
        with self.connection() as connection:
            primary_rows = connection.execute(
                """
                SELECT
                    e.id,
                    e.primary_source_item_id,
                    e.entities_json,
                    e.topics_json,
                    e.regions_json,
                    e.score,
                    e.score_features_json,
                    s.role
                FROM events e
                JOIN source_items i ON i.id = e.primary_source_item_id
                JOIN sources s ON s.id = i.source_id
                WHERE e.id IN (?, ?)
                """,
                (target_event_id, duplicate_event_id),
            ).fetchall()
            primary_by_event = {int(row["id"]): row for row in primary_rows}
            target_primary = primary_by_event.get(target_event_id)
            duplicate_primary = primary_by_event.get(duplicate_event_id)
            target_cards = connection.execute(
                "SELECT COUNT(*) FROM cards WHERE event_id = ?",
                (target_event_id,),
            ).fetchone()[0]
            duplicate_cards = connection.execute(
                "SELECT COUNT(*) FROM cards WHERE event_id = ?",
                (duplicate_event_id,),
            ).fetchone()[0]
            if duplicate_cards:
                raise ValueError("Duplicate event with cards must be selected as merge target")
            duplicate_editorial_memory = connection.execute(
                """
                SELECT
                    EXISTS(SELECT 1 FROM event_fact_sheets WHERE event_id = ?)
                    OR EXISTS(SELECT 1 FROM event_dossiers WHERE event_id = ?)
                """,
                (duplicate_event_id, duplicate_event_id),
            ).fetchone()[0]
            if duplicate_editorial_memory:
                raise ValueError(
                    "Duplicate event with fact-sheet or dossier memory must be the merge target"
                )
            connection.execute(
                """
                INSERT OR IGNORE INTO event_source_items(event_id, source_item_id, evidence_role)
                SELECT ?, source_item_id, evidence_role
                FROM event_source_items
                WHERE event_id = ?
                """,
                (target_event_id, duplicate_event_id),
            )
            connection.execute(
                "DELETE FROM event_source_items WHERE event_id = ?", (duplicate_event_id,)
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO event_research_documents(
                    event_id, research_document_id, evidence_role, relevance, created_at
                )
                SELECT ?, research_document_id, evidence_role, relevance, created_at
                FROM event_research_documents
                WHERE event_id = ?
                """,
                (target_event_id, duplicate_event_id),
            )
            connection.execute(
                "DELETE FROM event_research_documents WHERE event_id = ?",
                (duplicate_event_id,),
            )
            connection.execute(
                "UPDATE research_queries SET event_id = ? WHERE event_id = ?",
                (target_event_id, duplicate_event_id),
            )
            replacement_primary_id = None
            if (
                target_primary is not None
                and duplicate_primary is not None
                and target_primary["role"] == "author_radar"
                and duplicate_primary["role"] != "author_radar"
            ):
                replacement_primary_id = duplicate_primary["primary_source_item_id"]
            target_entities = json.loads(target_primary["entities_json"] or "{}")
            duplicate_entities = json.loads(duplicate_primary["entities_json"] or "{}")
            merged_entities: dict[str, list[str]] = {}
            for key in set(target_entities) | set(duplicate_entities):
                merged_entities[key] = list(
                    dict.fromkeys([*target_entities.get(key, []), *duplicate_entities.get(key, [])])
                )
            merged_topics = list(
                dict.fromkeys(
                    [
                        *json.loads(target_primary["topics_json"] or "[]"),
                        *json.loads(duplicate_primary["topics_json"] or "[]"),
                    ]
                )
            )
            merged_regions = list(
                dict.fromkeys(
                    [
                        *json.loads(target_primary["regions_json"] or "[]"),
                        *json.loads(duplicate_primary["regions_json"] or "[]"),
                    ]
                )
            )
            target_features = json.loads(target_primary["score_features_json"] or "{}")
            duplicate_features = json.loads(duplicate_primary["score_features_json"] or "{}")
            target_base = float(target_features.get("base_score", target_primary["score"] or 0))
            duplicate_base = float(
                duplicate_features.get("base_score", duplicate_primary["score"] or 0)
            )
            base_score = max(target_base, duplicate_base)
            merged_features = dict(
                target_features if target_base >= duplicate_base else duplicate_features
            )
            independent_source_count = int(
                connection.execute(
                    """
                    SELECT COUNT(DISTINCT CASE
                        WHEN sources.kind = 'web' THEN source_items.source_url
                        ELSE source_items.source_id
                    END)
                    FROM event_source_items
                    JOIN source_items
                      ON source_items.id = event_source_items.source_item_id
                    JOIN sources ON sources.id = source_items.source_id
                    WHERE event_source_items.event_id = ?
                    """,
                    (target_event_id,),
                ).fetchone()[0]
            )
            corroboration_bonus = min(
                1.5,
                max(0, independent_source_count - 1) * 0.75,
            )
            merged_features.update(
                {
                    "base_score": base_score,
                    "corroborating_sources": corroboration_bonus,
                }
            )
            connection.execute(
                """
                UPDATE events
                SET latest_source_at = MAX(
                        latest_source_at,
                        (SELECT latest_source_at FROM events WHERE id = ?)
                    ),
                    score = ?,
                    score_features_json = ?,
                    entities_json = ?,
                    topics_json = ?,
                    regions_json = ?,
                    primary_source_item_id = COALESCE(?, primary_source_item_id),
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    duplicate_event_id,
                    round(base_score + corroboration_bonus, 3),
                    json.dumps(merged_features, ensure_ascii=False),
                    json.dumps(merged_entities, ensure_ascii=False),
                    json.dumps(merged_topics, ensure_ascii=False),
                    json.dumps(merged_regions, ensure_ascii=False),
                    replacement_primary_id,
                    utc_now(),
                    target_event_id,
                ),
            )
            connection.execute("DELETE FROM events WHERE id = ?", (duplicate_event_id,))

            if target_cards:  # keep the branch explicit for auditability
                connection.execute(
                    "UPDATE events SET status = 'card_ready' WHERE id = ?", (target_event_id,)
                )

    def rebuild_derived_data(self) -> dict[str, int]:
        """Drop reproducible pipeline outputs while preserving raw inputs and usage."""
        with self.connection() as connection:
            feedback_count = int(
                connection.execute("SELECT COUNT(*) FROM feedback_events").fetchone()[0]
            )
            sent_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM deliveries WHERE delivery_status = 'sent'"
                ).fetchone()[0]
            )
            if feedback_count or sent_count:
                raise RuntimeError(
                    "Refusing derived rebuild after delivery or feedback; preserve user history"
                )
            counts = {
                "deliveries": int(
                    connection.execute("SELECT COUNT(*) FROM deliveries").fetchone()[0]
                ),
                "cards": int(connection.execute("SELECT COUNT(*) FROM cards").fetchone()[0]),
                "events": int(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]),
                "intelligence_products": int(
                    connection.execute("SELECT COUNT(*) FROM intelligence_products").fetchone()[0]
                ),
            }
            connection.execute("DELETE FROM intelligence_products")
            connection.execute("DELETE FROM feedback_sessions")
            connection.execute("DELETE FROM deliveries")
            connection.execute("DELETE FROM cards")
            connection.execute("DELETE FROM event_source_items")
            connection.execute("DELETE FROM events")
            connection.execute(
                """
                UPDATE source_items
                SET processing_status = CASE
                        WHEN source_id IN (
                            SELECT id FROM sources WHERE role = 'own_history'
                        ) THEN 'history_only'
                        ELSE 'new'
                    END,
                    processing_error = NULL
                WHERE processing_status != 'smoke_test'
                """
            )
        return counts

    def create_card(
        self,
        *,
        event_id: int,
        prompt_version: str,
        model: str,
        generation_cost_usd: float,
        structured_output: dict[str, Any],
        rendered_html: str,
        source_urls: list[str],
        primary_preview_url: str,
        validation_status: str = "valid",
        validation_errors: list[str] | None = None,
    ) -> int:
        with self.connection() as connection:
            version = int(
                connection.execute(
                    "SELECT COALESCE(MAX(card_version), 0) + 1 FROM cards WHERE event_id = ?",
                    (event_id,),
                ).fetchone()[0]
            )
            cursor = connection.execute(
                """
                INSERT INTO cards(
                    event_id, card_version, prompt_version, model,
                    generation_cost_usd, structured_output_json, rendered_html,
                    source_urls_json, primary_preview_url, validation_status,
                    validation_errors_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    version,
                    prompt_version,
                    model,
                    generation_cost_usd,
                    json.dumps(structured_output, ensure_ascii=False),
                    rendered_html,
                    json.dumps(source_urls, ensure_ascii=False),
                    primary_preview_url,
                    validation_status,
                    json.dumps(validation_errors or [], ensure_ascii=False),
                ),
            )
            connection.execute(
                "UPDATE events SET status = 'card_ready', updated_at = ? WHERE id = ?",
                (utc_now(), event_id),
            )
            return int(cursor.lastrowid)

    def create_candidate_preview_card(
        self,
        *,
        event_id: int,
        prompt_version: str,
        model: str,
        generation_cost_usd: float,
        structured_output: dict[str, Any],
        rendered_html: str,
        source_urls: list[str],
        primary_preview_url: str,
    ) -> int:
        """Create a feedback-only discovery card without consuming the publication queue."""
        with self.connection() as connection:
            version = int(
                connection.execute(
                    "SELECT COALESCE(MAX(card_version), 0) + 1 FROM cards WHERE event_id = ?",
                    (event_id,),
                ).fetchone()[0]
            )
            cursor = connection.execute(
                """
                INSERT INTO cards(
                    event_id, card_version, prompt_version, model,
                    generation_cost_usd, structured_output_json, rendered_html,
                    source_urls_json, primary_preview_url, validation_status,
                    validation_errors_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'candidate_preview', '[]')
                """,
                (
                    event_id,
                    version,
                    prompt_version,
                    model,
                    generation_cost_usd,
                    json.dumps(structured_output, ensure_ascii=False),
                    rendered_html,
                    json.dumps(source_urls, ensure_ascii=False),
                    primary_preview_url,
                ),
            )
            return int(cursor.lastrowid)

    def create_candidate_batch(
        self,
        *,
        chat_id: int,
        card_ids: list[int],
        batch_id: str,
    ) -> list[int]:
        """Quarantine candidate previews in one explicit delivery batch."""
        ordered_ids = list(dict.fromkeys(card_ids))
        if not ordered_ids:
            return []
        delivery_ids: list[int] = []
        with self.connection() as connection:
            for card_id in ordered_ids:
                card = connection.execute(
                    """
                    SELECT id FROM cards
                    WHERE id = ? AND validation_status = 'candidate_preview'
                    """,
                    (card_id,),
                ).fetchone()
                if card is None:
                    raise ValueError(f"Card {card_id} is not a candidate preview")
                cursor = connection.execute(
                    """
                    INSERT INTO deliveries(card_id, chat_id, batch_id, delivery_status)
                    VALUES (?, ?, ?, 'pending')
                    """,
                    (card_id, chat_id, batch_id),
                )
                delivery_ids.append(int(cursor.lastrowid))
        return delivery_ids

    def candidate_batch_cards(self, chat_id: int, batch_id: str) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    cards.id AS card_id,
                    cards.event_id,
                    cards.structured_output_json,
                    cards.rendered_html,
                    cards.source_urls_json,
                    cards.primary_preview_url,
                    deliveries.id AS delivery_id,
                    deliveries.delivery_status
                FROM deliveries
                JOIN cards ON cards.id = deliveries.card_id
                WHERE deliveries.chat_id = ?
                  AND deliveries.batch_id = ?
                  AND deliveries.delivery_status IN ('pending', 'failed')
                  AND cards.validation_status = 'candidate_preview'
                ORDER BY deliveries.id
                """,
                (chat_id, batch_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def card_source_is_available(self, card_id: int) -> bool:
        with self.connection() as db:
            return (
                db.execute(
                    f"""SELECT 1 FROM cards
                JOIN events ON events.id=cards.event_id
                JOIN source_items ON source_items.id=events.primary_source_item_id
                JOIN sources ON sources.id=source_items.source_id
                WHERE cards.id=? AND cards.validation_status='valid'
                  AND {available_source_sql()}""",
                    (card_id,),
                ).fetchone()
                is not None
            )

    def deliverable_cards(self, chat_id: int, limit: int = 10) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    cards.id AS card_id,
                    cards.event_id,
                    cards.structured_output_json,
                    cards.rendered_html,
                    cards.source_urls_json,
                    cards.primary_preview_url,
                    deliveries.id AS delivery_id,
                    deliveries.delivery_status
                FROM cards
                JOIN events ON events.id=cards.event_id
                JOIN source_items ON source_items.id=events.primary_source_item_id
                JOIN sources ON sources.id=source_items.source_id
                LEFT JOIN deliveries
                    ON deliveries.card_id = cards.id AND deliveries.chat_id = ?
                WHERE cards.validation_status = 'valid'
                  AND {available_source_sql()}
                  AND (deliveries.id IS NULL OR deliveries.delivery_status = 'failed')
                ORDER BY cards.created_at, cards.id
                LIMIT ?
                """,
                (chat_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def deliverable_cards_by_ids(
        self,
        chat_id: int,
        card_ids: list[int],
    ) -> list[dict[str, Any]]:
        """Return only valid, retry-safe cards in the caller's requested order."""
        ordered_ids = list(dict.fromkeys(card_ids))
        if not ordered_ids:
            return []
        placeholders = ", ".join("?" for _ in ordered_ids)
        with self.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    cards.id AS card_id,
                    cards.event_id,
                    cards.structured_output_json,
                    cards.rendered_html,
                    cards.source_urls_json,
                    cards.primary_preview_url,
                    deliveries.id AS delivery_id,
                    deliveries.delivery_status
                FROM cards
                JOIN events ON events.id=cards.event_id
                JOIN source_items ON source_items.id=events.primary_source_item_id
                JOIN sources ON sources.id=source_items.source_id
                LEFT JOIN deliveries
                    ON deliveries.card_id = cards.id AND deliveries.chat_id = ?
                WHERE cards.id IN ({placeholders})
                  AND cards.validation_status = 'valid'
                  AND {available_source_sql()}
                  AND (deliveries.id IS NULL OR deliveries.delivery_status = 'failed')
                """,
                (chat_id, *ordered_ids),
            ).fetchall()
        rows_by_id = {int(row["card_id"]): dict(row) for row in rows}
        return [rows_by_id[card_id] for card_id in ordered_ids if card_id in rows_by_id]

    def ensure_delivery(self, card_id: int, chat_id: int) -> int:
        with self.connection() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO deliveries(card_id, chat_id, delivery_status)
                VALUES (?, ?, 'pending')
                """,
                (card_id, chat_id),
            )
            row = connection.execute(
                "SELECT id FROM deliveries WHERE card_id = ? AND chat_id = ?",
                (card_id, chat_id),
            ).fetchone()
        if row is None:  # pragma: no cover - defensive guard
            raise RuntimeError("Delivery insert/select failed")
        return int(row["id"])

    def get_runtime_setting(self, key: str) -> str | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT value FROM runtime_settings WHERE key = ?", (key,)
            ).fetchone()
        return str(row["value"]) if row else None

    def claim_runtime_setting(self, key: str, value: str) -> str:
        """Set a first-writer-wins value and return the value that won."""
        with self.connection() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO runtime_settings(key, value, updated_at)
                VALUES (?, ?, ?)
                """,
                (key, value, utc_now()),
            )
            row = connection.execute(
                "SELECT value FROM runtime_settings WHERE key = ?", (key,)
            ).fetchone()
        if row is None:  # pragma: no cover - defensive guard
            raise RuntimeError(f"Could not claim runtime setting: {key}")
        return str(row["value"])

    def set_runtime_setting(self, key: str, value: str) -> None:
        """Set a mutable runtime value without changing environment configuration."""
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO runtime_settings(key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (key, value, utc_now()),
            )

    def get_delivery_context(self, delivery_id: int) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT
                    deliveries.id AS delivery_id,
                    deliveries.chat_id,
                    deliveries.telegram_message_id,
                    deliveries.card_id,
                    cards.event_id,
                    cards.rendered_html,
                    cards.structured_output_json,
                    cards.primary_preview_url
                FROM deliveries
                JOIN cards ON cards.id = deliveries.card_id
                WHERE deliveries.id = ?
                """,
                (delivery_id,),
            ).fetchone()
        return dict(row) if row else None

    def current_reaction(self, delivery_id: int) -> str | None:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT reaction
                FROM feedback_events
                WHERE delivery_id = ? AND reaction IS NOT NULL
                ORDER BY id DESC
                LIMIT 1
                """,
                (delivery_id,),
            ).fetchone()
        return row["reaction"] if row else None

    def begin_feedback_session(
        self,
        delivery_id: int,
        user_id: int,
        reaction: str,
        *,
        state: str = "choose_comment",
    ) -> None:
        now = utc_now()
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO feedback_sessions(
                    delivery_id, user_id, reaction, state, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(delivery_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    reaction = excluded.reaction,
                    state = excluded.state,
                    prompt_message_id = NULL,
                    updated_at = excluded.updated_at
                """,
                (delivery_id, user_id, reaction, state, now, now),
            )
            connection.execute(
                "UPDATE deliveries SET keyboard_state = ? WHERE id = ?",
                (state, delivery_id),
            )

    def set_feedback_session_state(
        self,
        delivery_id: int,
        state: str,
        *,
        prompt_message_id: int | None = None,
    ) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                UPDATE feedback_sessions
                SET state = ?, prompt_message_id = ?, updated_at = ?
                WHERE delivery_id = ?
                """,
                (state, prompt_message_id, utc_now(), delivery_id),
            )
            connection.execute(
                "UPDATE deliveries SET keyboard_state = ? WHERE id = ?",
                (state, delivery_id),
            )

    def feedback_session_for_user(self, user_id: int) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM feedback_sessions
                WHERE user_id = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
        return dict(row) if row else None

    def clear_feedback_session(self, delivery_id: int) -> None:
        with self.connection() as connection:
            connection.execute(
                "DELETE FROM feedback_sessions WHERE delivery_id = ?", (delivery_id,)
            )
            connection.execute(
                "UPDATE deliveries SET keyboard_state = 'reactions' WHERE id = ?",
                (delivery_id,),
            )

    def finalize_feedback(
        self,
        delivery_id: int,
        *,
        action: str,
        reaction: str,
        comment: str | None = None,
    ) -> int:
        context = self.get_delivery_context(delivery_id)
        if context is None:
            raise ValueError(f"Unknown delivery_id: {delivery_id}")
        previous_reaction = self.current_reaction(delivery_id)
        with self.connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO feedback_events(
                    delivery_id, event_id, card_id, action,
                    reaction, comment, previous_reaction
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    delivery_id,
                    context["event_id"],
                    context["card_id"],
                    action,
                    reaction,
                    comment,
                    previous_reaction,
                ),
            )
            connection.execute(
                "DELETE FROM feedback_sessions WHERE delivery_id = ?", (delivery_id,)
            )
            connection.execute(
                "UPDATE deliveries SET keyboard_state = 'reactions' WHERE id = ?",
                (delivery_id,),
            )
            return int(cursor.lastrowid)

    def record_feedback_note(
        self,
        delivery_id: int,
        comment: str,
        *,
        reaction: str | None = None,
    ) -> int:
        context = self.get_delivery_context(delivery_id)
        if context is None:
            raise ValueError(f"Unknown delivery_id: {delivery_id}")
        previous_reaction = self.current_reaction(delivery_id) if reaction else None
        with self.connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO feedback_events(
                    delivery_id, event_id, card_id, action,
                    reaction, comment, previous_reaction
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    delivery_id,
                    context["event_id"],
                    context["card_id"],
                    "external_feedback" if reaction else "development_note",
                    reaction,
                    comment,
                    previous_reaction,
                ),
            )
            return int(cursor.lastrowid)

    def create_smoke_delivery(self, chat_id: int, source_url: str) -> dict[str, Any]:
        now = utc_now()
        unique = uuid.uuid4().hex
        rendered_html = (
            "<b>Тестовая карточка персональной ленты</b>\n\n"
            f'Проверяем <a href="{source_url}">контекстную ссылку</a>, '
            "нативное превью Telegram и четыре уровня обратной связи."
        )
        with self.connection() as connection:
            source = connection.execute(
                "SELECT id FROM sources WHERE candidate_enabled = 1 ORDER BY id LIMIT 1"
            ).fetchone()
            if source is None:
                raise RuntimeError("No candidate source is configured")
            item_cursor = connection.execute(
                """
                INSERT INTO source_items(
                    source_id, external_id, source_url, original_text,
                    original_language, published_at, outbound_urls_json,
                    payload_hash, raw_payload_json, processing_status
                ) VALUES (?, ?, ?, ?, 'ru', ?, ?, ?, '{}', 'smoke_test')
                """,
                (
                    source["id"],
                    f"smoke-{unique}",
                    source_url,
                    "Telegram smoke test",
                    now,
                    json.dumps([source_url]),
                    unique,
                ),
            )
            source_item_id = int(item_cursor.lastrowid)
            event_cursor = connection.execute(
                """
                INSERT INTO events(
                    status, event_type, headline, first_seen_at,
                    latest_source_at, primary_source_item_id, decision_reason
                ) VALUES ('test', 'system_test', ?, ?, ?, ?, 'telegram_smoke')
                """,
                ("Тестовая карточка", now, now, source_item_id),
            )
            event_id = int(event_cursor.lastrowid)
            connection.execute(
                """
                INSERT INTO event_source_items(event_id, source_item_id, evidence_role)
                VALUES (?, ?, 'primary')
                """,
                (event_id, source_item_id),
            )
            card_cursor = connection.execute(
                """
                INSERT INTO cards(
                    event_id, card_version, prompt_version, structured_output_json,
                    rendered_html, source_urls_json, primary_preview_url,
                    validation_status
                ) VALUES (?, 1, 'smoke_v1', '{}', ?, ?, ?, 'valid')
                """,
                (event_id, rendered_html, json.dumps([source_url]), source_url),
            )
            card_id = int(card_cursor.lastrowid)
            delivery_cursor = connection.execute(
                """
                INSERT INTO deliveries(card_id, chat_id, delivery_status)
                VALUES (?, ?, 'pending')
                """,
                (card_id, chat_id),
            )
            delivery_id = int(delivery_cursor.lastrowid)
        return {
            "delivery_id": delivery_id,
            "card_id": card_id,
            "event_id": event_id,
            "rendered_html": rendered_html,
            "primary_preview_url": source_url,
        }

    def mark_delivery_sent(self, delivery_id: int, telegram_message_id: int) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                UPDATE deliveries
                SET telegram_message_id = ?, delivery_status = 'sent',
                    delivered_at = ?, attempt_count = attempt_count + 1,
                    last_error = NULL
                WHERE id = ?
                """,
                (telegram_message_id, utc_now(), delivery_id),
            )

    def mark_delivery_error(self, delivery_id: int, error: str) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                UPDATE deliveries
                SET delivery_status = 'failed', attempt_count = attempt_count + 1,
                    last_error = ?
                WHERE id = ?
                """,
                (error[:2000], delivery_id),
            )

    def enqueue_detail_request(
        self,
        delivery_id: int,
        requested_by_user_id: int,
        *,
        generation_key: str | None = None,
        max_daily_new: int = 5,
        now: str | None = None,
        depth: str = "legacy",
        reader_context: dict | None = None,
    ) -> dict[str, Any]:
        """Atomically enqueue one paid on-demand research request per delivery.

        This method performs no network or model calls. Repeated callbacks return
        the existing request. A valid, unexpired article is attached immediately
        without consuming the daily allowance.
        """
        if depth != "legacy":
            from app.db.news_reader import enqueue

            return enqueue(
                self,
                delivery_id,
                requested_by_user_id,
                depth=depth,
                reader_context=reader_context,
                generation_key=generation_key,
                max_daily_new=max_daily_new,
                now=now,
            )
        if max_daily_new < 0:
            raise ValueError("max_daily_new must be non-negative")
        requested_at = _as_utc_datetime(now)
        requested_at_iso = requested_at.isoformat()
        day_start = requested_at.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        day_end = (
            requested_at.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        ).isoformat()

        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            context = connection.execute(
                """
                SELECT
                    deliveries.id AS delivery_id,
                    deliveries.chat_id,
                    deliveries.delivery_status,
                    deliveries.telegram_message_id,
                    deliveries.card_id,
                    cards.event_id
                FROM deliveries
                JOIN cards ON cards.id = deliveries.card_id
                WHERE deliveries.id = ?
                """,
                (delivery_id,),
            ).fetchone()
            if context is None:
                raise ValueError(f"Unknown delivery_id: {delivery_id}")
            if context["delivery_status"] != "sent":
                raise ValueError(f"Delivery {delivery_id} has not been sent")

            resolved_key = generation_key or (
                f"event:{int(context['event_id'])}:{DETAIL_PROMPT_VERSION}"
            )
            existing = connection.execute(
                "SELECT * FROM detail_requests WHERE delivery_id = ? AND depth='legacy'",
                (delivery_id,),
            ).fetchone()
            if existing is not None:
                if existing["status"] in {"ready", "sent"}:
                    reusable = connection.execute(
                        f"""SELECT 1 FROM detailed_articles
                        WHERE id=? AND event_id=? AND validation_status='valid'
                          AND {current_detail_policy_sql()}
                          AND datetime(expires_at)>datetime(?)""",
                        (existing["article_id"], context["event_id"], requested_at_iso),
                    ).fetchone()
                    if reusable is None:
                        # Keep the artifact, feedback and historical message IDs.
                        # This does not enqueue paid work: a separate Retry click
                        # is needed to request a freshly validated article.
                        connection.execute(
                            """UPDATE detail_requests SET status='failed', stage='complete',
                                cache_hit=0,
                                lease_until=NULL, placeholder_lease_until=NULL,
                                last_error='Detail evidence policy or expiry changed; '
                                    || 'explicit retry required',
                                updated_at=? WHERE id=?""",
                            (requested_at_iso, existing["id"]),
                        )
                        existing = connection.execute(
                            "SELECT * FROM detail_requests WHERE id=?", (existing["id"],)
                        ).fetchone()
                result = _detail_request_row(existing)
                result.update({"created": False, "cache_hit": bool(result["cache_hit"])})
                return result

            if generation_key is None:
                cached = connection.execute(
                    f"""
                    SELECT id, generation_key
                    FROM detailed_articles
                    WHERE event_id = ?
                      AND validation_status = 'valid'
                      AND {current_detail_policy_sql()}
                      AND datetime(expires_at) > datetime(?)
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    """,
                    (context["event_id"], requested_at_iso),
                ).fetchone()
            else:
                cached = connection.execute(
                    f"""
                    SELECT id, generation_key
                    FROM detailed_articles
                    WHERE generation_key = ?
                      AND event_id = ?
                      AND validation_status = 'valid'
                      AND {current_detail_policy_sql()}
                      AND datetime(expires_at) > datetime(?)
                    LIMIT 1
                    """,
                    (resolved_key, context["event_id"], requested_at_iso),
                ).fetchone()
            if cached is not None:
                resolved_key = str(cached["generation_key"])
                cursor = connection.execute(
                    """
                    INSERT INTO detail_requests(
                        delivery_id, card_id, event_id, chat_id, user_id,
                        generation_key, status, stage, article_id, cache_hit,
                        requested_at, completed_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'ready', 'delivery', ?, 1, ?, ?, ?)
                    """,
                    (
                        delivery_id,
                        context["card_id"],
                        context["event_id"],
                        context["chat_id"],
                        requested_by_user_id,
                        resolved_key,
                        cached["id"],
                        requested_at_iso,
                        requested_at_iso,
                        requested_at_iso,
                    ),
                )
                row = connection.execute(
                    "SELECT * FROM detail_requests WHERE id = ?", (cursor.lastrowid,)
                ).fetchone()
                if row is None:  # pragma: no cover - defensive guard
                    raise RuntimeError("Detail cache request insert failed")
                result = _detail_request_row(row)
                result.update({"created": True, "cache_hit": True})
                return result

            count = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM detail_requests
                    WHERE user_id = ?
                      AND cache_hit = 0
                      AND requested_at >= ?
                      AND requested_at < ?
                    """,
                    (requested_by_user_id, day_start, day_end),
                ).fetchone()[0]
            )
            if count >= max_daily_new:
                return {
                    "request_id": None,
                    "id": None,
                    "delivery_id": delivery_id,
                    "card_id": int(context["card_id"]),
                    "event_id": int(context["event_id"]),
                    "chat_id": int(context["chat_id"]),
                    "user_id": requested_by_user_id,
                    "generation_key": resolved_key,
                    "status": "daily_limit",
                    "stage": None,
                    "article_id": None,
                    "created": False,
                    "cache_hit": False,
                }

            cursor = connection.execute(
                """
                INSERT INTO detail_requests(
                    delivery_id, card_id, event_id, chat_id, user_id,
                    generation_key, status, stage, requested_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'queued', 'queued', ?, ?)
                """,
                (
                    delivery_id,
                    context["card_id"],
                    context["event_id"],
                    context["chat_id"],
                    requested_by_user_id,
                    resolved_key,
                    requested_at_iso,
                    requested_at_iso,
                ),
            )
            row = connection.execute(
                "SELECT * FROM detail_requests WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
        if row is None:  # pragma: no cover - defensive guard
            raise RuntimeError("Detail request insert failed")
        result = _detail_request_row(row)
        result["request_id"] = int(result["id"])
        result.update({"created": True, "cache_hit": False})
        return result

    def detail_request_for_delivery(
        self, delivery_id: int, *, depth: str = "legacy"
    ) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM detail_requests WHERE delivery_id = ? AND depth = ?",
                (delivery_id, depth),
            ).fetchone()
        if row is None:
            return None
        result = _detail_request_row(row)
        result["request_id"] = int(result["id"])
        return result

    def get_detail_request(self, request_id: int) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM detail_requests WHERE id = ?", (request_id,)
            ).fetchone()
        if row is None:
            return None
        result = _detail_request_row(row)
        result["request_id"] = int(result["id"])
        return result

    def detail_requests_needing_keyboard_sync(
        self,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return cards whose persisted detail label differs from request state."""
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    requests.*,
                    deliveries.telegram_message_id,
                    CASE
                        WHEN requests.status IN ('queued', 'running') THEN 'pending'
                        ELSE requests.status
                    END AS display_status
                FROM detail_requests AS requests
                JOIN deliveries ON deliveries.id = requests.delivery_id
                WHERE COALESCE(requests.keyboard_synced_status, '') != CASE
                    WHEN requests.status IN ('queued', 'running') THEN 'pending'
                    ELSE requests.status
                END
                ORDER BY requests.updated_at, requests.id
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [_detail_request_row(row) for row in rows]

    def mark_detail_keyboard_synced(self, request_id: int, display_status: str) -> bool:
        """Record an edit only if it still represents the request's current state."""
        with self.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE detail_requests
                SET keyboard_synced_status = ?, updated_at = ?
                WHERE id = ?
                  AND ? = CASE
                      WHEN status IN ('queued', 'running') THEN 'pending'
                      ELSE status
                  END
                """,
                (display_status, utc_now(), request_id, display_status),
            )
        return cursor.rowcount == 1

    def claim_next_detail_request(
        self,
        *,
        lease_minutes: int = 10,
        max_concurrency: int = 1,
        now: str | None = None,
    ) -> dict[str, Any] | None:
        """Atomically claim the oldest queued or stale-running request."""
        if lease_minutes < 1:
            raise ValueError("lease_minutes must be positive")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        claimed_at = _as_utc_datetime(now)
        claimed_at_iso = claimed_at.isoformat()
        lease_until = (claimed_at + timedelta(minutes=lease_minutes)).isoformat()
        with self.connection() as connection:
            row = connection.execute(
                """
                UPDATE detail_requests
                SET status = 'running',
                    stage = 'planning',
                    attempt_count = attempt_count + 1,
                    lease_until = ?,
                    started_at = COALESCE(started_at, ?),
                    last_error = NULL,
                    updated_at = ?
                WHERE id = (
                    SELECT candidate.id
                    FROM detail_requests AS candidate
                    WHERE (
                        (
                            candidate.status = 'queued'
                            AND NOT EXISTS (
                                SELECT 1
                                FROM detail_requests AS active
                                WHERE active.generation_key = candidate.generation_key
                                  AND active.id != candidate.id
                                  AND active.status = 'running'
                                  AND (
                                      active.lease_until IS NULL
                                      OR datetime(active.lease_until) > datetime(?)
                                  )
                            )
                        )
                        OR (
                            candidate.status = 'running'
                            AND candidate.lease_until IS NOT NULL
                            AND datetime(candidate.lease_until) <= datetime(?)
                        )
                    )
                      AND (
                          SELECT COUNT(*)
                          FROM detail_requests AS global_active
                          WHERE global_active.status = 'running'
                            AND (
                                global_active.lease_until IS NULL
                                OR datetime(global_active.lease_until) > datetime(?)
                            )
                      ) < ?
                    ORDER BY candidate.requested_at, candidate.id
                    LIMIT 1
                )
                RETURNING *
                """,
                (
                    lease_until,
                    claimed_at_iso,
                    claimed_at_iso,
                    claimed_at_iso,
                    claimed_at_iso,
                    claimed_at_iso,
                    max_concurrency,
                ),
            ).fetchone()
        if row is None:
            return None
        result = _detail_request_row(row)
        result["request_id"] = int(result["id"])
        return result

    def update_detail_request_stage(
        self,
        request_id: int,
        stage: str,
        *,
        lease_minutes: int = 10,
        now: str | None = None,
    ) -> bool:
        if stage not in DETAIL_REQUEST_STAGES - {"queued", "complete"}:
            raise ValueError(f"Invalid running detail stage: {stage}")
        if lease_minutes < 1:
            raise ValueError("lease_minutes must be positive")
        updated_at = _as_utc_datetime(now)
        lease_until = (updated_at + timedelta(minutes=lease_minutes)).isoformat()
        with self.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE detail_requests
                SET stage = ?, lease_until = ?, updated_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (stage, lease_until, updated_at.isoformat(), request_id),
            )
        return cursor.rowcount == 1

    def set_detail_placeholder_message_id(self, request_id: int, message_id: int) -> bool:
        with self.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE detail_requests
                SET placeholder_message_id = ?, placeholder_lease_until = NULL, updated_at = ?
                WHERE id = ?
                """,
                (message_id, utc_now(), request_id),
            )
        return cursor.rowcount == 1

    def claim_detail_placeholder_request(
        self,
        *,
        lease_minutes: int = 2,
        now: str | None = None,
    ) -> dict[str, Any] | None:
        """Claim one placeholder send so two bot processes cannot duplicate it."""
        if lease_minutes < 1:
            raise ValueError("lease_minutes must be positive")
        claimed_at = _as_utc_datetime(now)
        claimed_at_iso = claimed_at.isoformat()
        lease_until = (claimed_at + timedelta(minutes=lease_minutes)).isoformat()
        with self.connection() as connection:
            row = connection.execute(
                """
                UPDATE detail_requests
                SET placeholder_lease_until = ?, updated_at = ?
                WHERE id = (
                    SELECT id
                    FROM detail_requests
                    WHERE status IN ('queued', 'running', 'ready')
                      AND placeholder_message_id IS NULL
                      AND (
                          placeholder_lease_until IS NULL
                          OR datetime(placeholder_lease_until) <= datetime(?)
                      )
                    ORDER BY requested_at, id
                    LIMIT 1
                )
                RETURNING *
                """,
                (lease_until, claimed_at_iso, claimed_at_iso),
            ).fetchone()
            if row is None:
                return None
            parent = connection.execute(
                """
                SELECT telegram_message_id
                FROM deliveries
                WHERE id = ?
                """,
                (row["delivery_id"],),
            ).fetchone()
        result = _detail_request_row(row)
        result["parent_message_id"] = parent["telegram_message_id"] if parent else None
        return result

    def release_detail_placeholder_claim(self, request_id: int, error: str | None = None) -> bool:
        with self.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE detail_requests
                SET placeholder_lease_until = NULL,
                    last_error = COALESCE(?, last_error),
                    updated_at = ?
                WHERE id = ? AND placeholder_message_id IS NULL
                """,
                (error[:2000] if error else None, utc_now(), request_id),
            )
        return cursor.rowcount == 1

    def claim_terminal_detail_placeholder(
        self,
        *,
        lease_minutes: int = 2,
        now: str | None = None,
    ) -> dict[str, Any] | None:
        """Claim one existing progress reply that needs an honest terminal result."""
        if lease_minutes < 1:
            raise ValueError("lease_minutes must be positive")
        claimed_at = _as_utc_datetime(now)
        claimed_at_iso = claimed_at.isoformat()
        lease_until = (claimed_at + timedelta(minutes=lease_minutes)).isoformat()
        with self.connection() as connection:
            row = connection.execute(
                """
                UPDATE detail_requests
                SET placeholder_lease_until = ?, updated_at = ?
                WHERE id = (
                    SELECT id
                    FROM detail_requests
                    WHERE status IN ('insufficient_evidence', 'budget_blocked', 'failed')
                      AND stage = 'delivery'
                      AND placeholder_message_id IS NOT NULL
                      AND (
                          placeholder_lease_until IS NULL
                          OR datetime(placeholder_lease_until) <= datetime(?)
                      )
                    ORDER BY completed_at, id
                    LIMIT 1
                )
                RETURNING *
                """,
                (lease_until, claimed_at_iso, claimed_at_iso),
            ).fetchone()
        return _detail_request_row(row) if row is not None else None

    def mark_terminal_detail_placeholder_sent(
        self,
        request_id: int,
        *,
        status: str,
        claim_lease_until: str,
    ) -> bool:
        if status not in DETAIL_TERMINAL_FAILURE_STATUSES:
            raise ValueError(f"Invalid terminal detail status: {status}")
        with self.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE detail_requests
                SET stage = 'complete', placeholder_lease_until = NULL, updated_at = ?
                WHERE id = ? AND status = ? AND stage = 'delivery'
                  AND placeholder_lease_until = ?
                """,
                (utc_now(), request_id, status, claim_lease_until),
            )
        return cursor.rowcount == 1

    def release_terminal_detail_placeholder_claim(
        self,
        request_id: int,
        *,
        status: str,
        claim_lease_until: str,
    ) -> bool:
        if status not in DETAIL_TERMINAL_FAILURE_STATUSES:
            raise ValueError(f"Invalid terminal detail status: {status}")
        with self.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE detail_requests
                SET placeholder_lease_until = NULL, updated_at = ?
                WHERE id = ? AND status = ? AND stage = 'delivery'
                  AND placeholder_lease_until = ?
                """,
                (utc_now(), request_id, status, claim_lease_until),
            )
        return cursor.rowcount == 1

    def cached_detailed_article(
        self,
        generation_key: str,
        *,
        now: str | None = None,
        depth: str = "legacy",
        context_fingerprint: str | None = None,
        reader_context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if depth != "legacy":
            from app.db.news_reader import cached_article

            return cached_article(
                self,
                depth=depth,
                generation_key=generation_key,
                context_fingerprint=context_fingerprint,
                reader_context=reader_context,
                now=now,
            )
        checked_at = _as_utc_datetime(now).isoformat()
        with self.connection() as connection:
            row = connection.execute(
                f"""
                SELECT *
                FROM detailed_articles
                WHERE generation_key = ?
                  AND validation_status = 'valid' AND depth = 'legacy'
                  AND {current_detail_policy_sql()}
                  AND datetime(expires_at) > datetime(?)
                LIMIT 1
                """,
                (generation_key, checked_at),
            ).fetchone()
        return _detailed_article_row(row) if row else None

    def cached_detailed_article_for_event(
        self,
        event_id: int,
        *,
        now: str | None = None,
        depth: str = "legacy",
        context_fingerprint: str | None = None,
        reader_context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if depth != "legacy":
            from app.db.news_reader import cached_article

            return cached_article(
                self,
                depth=depth,
                event_id=event_id,
                context_fingerprint=context_fingerprint,
                reader_context=reader_context,
                now=now,
            )
        checked_at = _as_utc_datetime(now).isoformat()
        with self.connection() as connection:
            row = connection.execute(
                f"""
                SELECT *
                FROM detailed_articles
                WHERE event_id = ?
                  AND validation_status = 'valid' AND depth = 'legacy'
                  AND {current_detail_policy_sql()}
                  AND datetime(expires_at) > datetime(?)
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (event_id, checked_at),
            ).fetchone()
        return _detailed_article_row(row) if row else None

    def get_detailed_article(self, article_id: int) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM detailed_articles WHERE id = ?", (article_id,)
            ).fetchone()
        return _detailed_article_row(row) if row else None

    def store_detail_article_and_complete(
        self,
        request_id: int,
        *,
        generation_key: str,
        schema_version: str,
        prompt_version: str,
        model: str,
        provider_response_metadata: dict[str, Any],
        fact_snapshot: dict[str, Any],
        document_ids: list[int],
        structured_output: dict[str, Any],
        rendered_html: str,
        source_urls: list[str],
        validation_status: str,
        validation_errors: list[str] | None,
        generation_cost_usd: float,
        expires_at: str,
        now: str | None = None,
        depth: str = "legacy",
        reader_context: dict | None = None,
        context_fingerprint: str | None = None,
    ) -> int:
        """Persist an immutable article and make the claimed request ready atomically."""
        if depth != "legacy":
            from app.db.news_reader import store

            return store(
                self,
                request_id,
                depth=depth,
                reader_context=reader_context,
                context_fingerprint=context_fingerprint,
                generation_key=generation_key,
                schema_version=schema_version,
                prompt_version=prompt_version,
                model=model,
                provider_response_metadata=provider_response_metadata,
                fact_snapshot=fact_snapshot,
                document_ids=document_ids,
                structured_output=structured_output,
                rendered_html=rendered_html,
                source_urls=source_urls,
                validation_status=validation_status,
                validation_errors=validation_errors,
                generation_cost_usd=generation_cost_usd,
                expires_at=expires_at,
                now=now,
            )
        if generation_cost_usd < 0:
            raise ValueError("generation_cost_usd must be non-negative")
        if validation_status != "valid":
            raise ValueError("Only valid detailed articles can complete a request")
        if prompt_version != DETAIL_PROMPT_VERSION or schema_version != DETAIL_SCHEMA_VERSION:
            raise ValueError("Detailed article does not match the current evidence policy")
        completed_at = _as_utc_datetime(now).isoformat()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            request = connection.execute(
                "SELECT event_id, status FROM detail_requests WHERE id = ?", (request_id,)
            ).fetchone()
            if request is None:
                raise ValueError(f"Unknown detail request: {request_id}")
            if request["status"] != "running":
                raise ValueError(f"Detail request {request_id} is not running")
            connection.execute(
                """
                INSERT INTO detailed_articles(
                    event_id, generation_key, schema_version, prompt_version, model,
                    provider_response_metadata_json, fact_snapshot_json, document_ids_json,
                    structured_output_json, rendered_html, source_urls_json,
                    validation_status, validation_errors_json, generation_cost_usd,
                    expires_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(generation_key) DO NOTHING
                """,
                (
                    request["event_id"],
                    generation_key,
                    schema_version,
                    prompt_version,
                    model,
                    json.dumps(provider_response_metadata, ensure_ascii=False),
                    json.dumps(fact_snapshot, ensure_ascii=False),
                    json.dumps(document_ids, ensure_ascii=False),
                    json.dumps(structured_output, ensure_ascii=False),
                    rendered_html,
                    json.dumps(source_urls, ensure_ascii=False),
                    validation_status,
                    json.dumps(validation_errors or [], ensure_ascii=False),
                    generation_cost_usd,
                    expires_at,
                    completed_at,
                ),
            )
            article = connection.execute(
                """
                SELECT id, event_id, validation_status, prompt_version, schema_version
                FROM detailed_articles
                WHERE generation_key = ?
                """,
                (generation_key,),
            ).fetchone()
            if article is None:  # pragma: no cover - defensive guard
                raise RuntimeError("Detail article insert/select failed")
            if int(article["event_id"]) != int(request["event_id"]):
                raise ValueError("generation_key belongs to a different event")
            if article["validation_status"] != "valid":
                raise ValueError("generation_key belongs to an invalid article")
            if (
                article["prompt_version"] != DETAIL_PROMPT_VERSION
                or article["schema_version"] != DETAIL_SCHEMA_VERSION
            ):
                raise ValueError("generation_key belongs to an outdated-policy article")
            cursor = connection.execute(
                """
                UPDATE detail_requests
                SET generation_key = ?, status = 'ready', stage = 'delivery',
                    article_id = ?, lease_until = NULL, last_error = NULL,
                    completed_at = ?, updated_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (
                    generation_key,
                    article["id"],
                    completed_at,
                    completed_at,
                    request_id,
                ),
            )
            if cursor.rowcount != 1:  # pragma: no cover - transaction owns write lock
                raise RuntimeError("Detail request changed while completing")
            return int(article["id"])

    def complete_detail_request_from_cache(
        self,
        request_id: int,
        article_id: int,
        *,
        now: str | None = None,
    ) -> bool:
        """Attach an already-built artifact without re-running paid providers."""
        request = self.get_detail_request(request_id)
        if request and request.get("depth", "legacy") != "legacy":
            from app.db.news_reader import attach_cache

            return attach_cache(self, request_id, article_id, now=now)
        completed_at = _as_utc_datetime(now).isoformat()
        with self.connection() as connection:
            cursor = connection.execute(
                f"""
                UPDATE detail_requests
                SET status = 'ready', stage = 'delivery', article_id = ?,
                    generation_key = (
                        SELECT generation_key FROM detailed_articles WHERE id = ?
                    ),
                    cache_hit = 1, lease_until = NULL, last_error = NULL,
                    completed_at = ?, updated_at = ?
                WHERE id = ?
                  AND status = 'running'
                  AND event_id = (
                      SELECT event_id FROM detailed_articles WHERE id = ?
                  )
                  AND EXISTS (
                      SELECT 1 FROM detailed_articles
                      WHERE id = ?
                        AND validation_status = 'valid'
                        AND {current_detail_policy_sql()}
                        AND datetime(expires_at) > datetime(?)
                  )
                """,
                (
                    article_id,
                    article_id,
                    completed_at,
                    completed_at,
                    request_id,
                    article_id,
                    article_id,
                    completed_at,
                ),
            )
        return cursor.rowcount == 1

    def ready_detail_requests(self, *, limit: int = 10) -> list[dict[str, Any]]:
        """Return delivery-ready artifacts without parsing them in the research worker."""
        with self.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    requests.*,
                    articles.rendered_html,
                    articles.source_urls_json,
                    articles.generation_cost_usd,
                    deliveries.telegram_message_id AS parent_message_id
                FROM detail_requests AS requests
                JOIN detailed_articles AS articles ON articles.id = requests.article_id
                JOIN deliveries ON deliveries.id = requests.delivery_id
                WHERE requests.status = 'ready'
                  AND articles.validation_status = 'valid'
                  AND {_all_detail_policy_sql("articles")}
                  AND datetime(articles.expires_at)>datetime(?)
                ORDER BY requests.completed_at, requests.id
                LIMIT ?
                """,
                (utc_now(), limit),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = _detail_request_row(row)
            item["source_urls"] = json.loads(item.pop("source_urls_json"))
            result.append(item)
        return result

    def claim_ready_detail_request_for_delivery(
        self,
        *,
        lease_minutes: int = 2,
        now: str | None = None,
    ) -> dict[str, Any] | None:
        """Claim one ready artifact for duplicate-safe Telegram edit/send."""
        if lease_minutes < 1:
            raise ValueError("lease_minutes must be positive")
        claimed_at = _as_utc_datetime(now)
        claimed_at_iso = claimed_at.isoformat()
        lease_until = (claimed_at + timedelta(minutes=lease_minutes)).isoformat()
        with self.connection() as connection:
            connection.execute(
                f"""UPDATE detail_requests SET status='failed', stage='delivery', cache_hit=0,
                    lease_until=NULL, placeholder_lease_until=NULL,
                    last_error='Detail evidence policy or expiry changed; explicit retry required',
                    updated_at=?
                WHERE status='ready' AND NOT EXISTS (
                    SELECT 1 FROM detailed_articles WHERE id=detail_requests.article_id
                      AND depth=detail_requests.depth
                      AND (detail_requests.depth='legacy' OR context_fingerprint=
                        json_extract(detail_requests.reader_context_json,'$.context_fingerprint'))
                      AND event_id=detail_requests.event_id AND validation_status='valid'
                      AND {_all_detail_policy_sql()}
                      AND datetime(expires_at)>datetime(?)
                )""",
                (claimed_at_iso, claimed_at_iso),
            )
            row = connection.execute(
                """
                UPDATE detail_requests
                SET lease_until = ?, updated_at = ?
                WHERE id = (
                    SELECT id
                    FROM detail_requests
                    WHERE status = 'ready'
                      AND (depth='legacy' OR EXISTS(SELECT 1 FROM news_panels p
                        WHERE p.delivery_id=detail_requests.delivery_id
                        AND p.active_request_id=detail_requests.id AND p.view='loading'))
                      AND placeholder_message_id IS NOT NULL
                      AND (
                          lease_until IS NULL
                          OR datetime(lease_until) <= datetime(?)
                      )
                    ORDER BY completed_at, id
                    LIMIT 1
                )
                RETURNING *
                """,
                (lease_until, claimed_at_iso, claimed_at_iso),
            ).fetchone()
            if row is None:
                return None
            article = connection.execute(
                """
                SELECT rendered_html, source_urls_json, generation_cost_usd
                FROM detailed_articles
                WHERE id = ?
                """,
                (row["article_id"],),
            ).fetchone()
            parent = connection.execute(
                "SELECT telegram_message_id FROM deliveries WHERE id = ?",
                (row["delivery_id"],),
            ).fetchone()
        if article is None:  # pragma: no cover - foreign key invariant
            raise RuntimeError("Ready detail request has no article")
        result = _detail_request_row(row)
        result["rendered_html"] = article["rendered_html"]
        result["source_urls"] = json.loads(article["source_urls_json"])
        result["generation_cost_usd"] = float(article["generation_cost_usd"])
        result["parent_message_id"] = parent["telegram_message_id"] if parent else None
        return result

    def fail_detail_request(
        self,
        request_id: int,
        *,
        status: str = "failed",
        error: str,
        now: str | None = None,
    ) -> bool:
        if status not in {"failed", "insufficient_evidence", "budget_blocked"}:
            raise ValueError(f"Invalid terminal detail status: {status}")
        finished_at = _as_utc_datetime(now).isoformat()
        with self.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE detail_requests
                SET status = ?, stage = 'delivery', lease_until = NULL,
                    placeholder_lease_until = NULL, last_error = ?,
                    completed_at = ?, updated_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (status, error[:2000], finished_at, finished_at, request_id),
            )
        return cursor.rowcount == 1

    def mark_detail_request_sent(self, request_id: int, *, now: str | None = None) -> bool:
        sent_at = _as_utc_datetime(now).isoformat()
        with self.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE detail_requests
                SET status = 'sent', stage = 'complete', lease_until = NULL, last_error = NULL,
                    completed_at = COALESCE(completed_at, ?), updated_at = ?
                WHERE id = ? AND status IN ('ready', 'sent')
                """,
                (sent_at, sent_at, request_id),
            )
        return cursor.rowcount == 1

    def mark_detail_delivery_error(self, request_id: int, error: str) -> bool:
        """Keep an undelivered artifact ready without reopening a completed send."""
        with self.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE detail_requests
                SET status = 'ready', stage = 'delivery', lease_until = NULL,
                    last_error = ?, updated_at = ?
                WHERE id = ? AND status = 'ready'
                """,
                (error[:2000], utc_now(), request_id),
            )
        return cursor.rowcount == 1

    def retry_detail_request(
        self,
        request_id: int,
        *,
        now: str | None = None,
    ) -> dict[str, Any]:
        retried_at = _as_utc_datetime(now).isoformat()
        with self.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE detail_requests
                SET status = 'queued', stage = 'queued', article_id = NULL,
                    lease_until = NULL, placeholder_lease_until = NULL,
                    last_error = NULL, started_at = NULL,
                    completed_at = NULL, cache_hit = 0,
                    reader_context_json=CASE WHEN depth='legacy' THEN reader_context_json
                        ELSE json_set(reader_context_json,'$.retry_authorization',?,
                            '$.retry_rejected',json('true')) END,
                    updated_at = ?
                WHERE id = ?
                  AND status IN ('failed', 'insufficient_evidence', 'budget_blocked')
                RETURNING *
                """,
                (uuid.uuid4().hex, retried_at, request_id),
            )
            row = cursor.fetchone()
        if row is None:
            current = self.get_detail_request(request_id)
            if current is None:
                raise ValueError(f"Unknown detail request: {request_id}")
            raise ValueError(
                f"Detail request {request_id} cannot be retried from {current['status']}"
            )
        result = _detail_request_row(row)
        result["request_id"] = int(result["id"])
        return result

    def recover_stale_detail_requests(self, *, now: str | None = None) -> int:
        recovered_at = _as_utc_datetime(now).isoformat()
        with self.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE detail_requests
                SET status = 'queued', stage = 'queued', lease_until = NULL,
                    last_error = 'worker lease expired', updated_at = ?
                WHERE status = 'running'
                  AND lease_until IS NOT NULL
                  AND datetime(lease_until) <= datetime(?)
                """,
                (recovered_at, recovered_at),
            )
        return cursor.rowcount

    def recent_detail_requests(self, *, limit: int = 30) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM detail_requests
                ORDER BY requested_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = _detail_request_row(row)
            item["request_id"] = int(item["id"])
            result.append(item)
        return result

    def detail_status_counts(self) -> dict[str, int]:
        counts = {status: 0 for status in sorted(DETAIL_REQUEST_STATUSES)}
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM detail_requests GROUP BY status"
            ).fetchall()
        for row in rows:
            counts[str(row["status"])] = int(row["count"])
        return counts

    def events_for_contour_backfill(self, *, limit: int = 10_000) -> list[dict[str, Any]]:
        """Return durable event text and metadata for the local contour classifier."""
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    events.*,
                    source_items.original_text,
                    source_items.source_url,
                    source_items.published_at,
                    sources.id AS source_id,
                    sources.name AS source_name
                FROM events
                LEFT JOIN source_items ON source_items.id = events.primary_source_item_id
                LEFT JOIN sources ON sources.id = source_items.source_id
                ORDER BY events.id
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def update_event_contour_assignment(
        self,
        event_id: int,
        metadata: dict[str, Any],
    ) -> bool:
        """Merge explainable contour metadata without disturbing ranking features.

        The update is idempotent: an identical deterministic classification does not
        touch ``updated_at`` and therefore cannot invalidate product cache keys.
        """
        with self.connection() as connection:
            row = connection.execute(
                "SELECT score_features_json FROM events WHERE id = ?",
                (event_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"Unknown event_id: {event_id}")
            try:
                current = json.loads(row["score_features_json"] or "{}")
            except json.JSONDecodeError:
                current = {}
            if not isinstance(current, dict):
                current = {}
            merged = {**current, **metadata}
            if merged == current:
                return False
            connection.execute(
                """
                UPDATE events
                SET score_features_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (json.dumps(merged, ensure_ascii=False), utc_now(), event_id),
            )
        return True

    def intelligence_events(
        self,
        *,
        since: str,
        until: str,
        limit: int = 10_000,
    ) -> list[dict[str, Any]]:
        """Return the bounded factual corpus used by derived intelligence products."""
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    events.*,
                    source_items.original_text,
                    source_items.source_url,
                    source_items.published_at,
                    sources.id AS source_id,
                    sources.name AS source_name,
                    GROUP_CONCAT(DISTINCT evidence_sources.id) AS source_ids_csv
                FROM events
                LEFT JOIN source_items ON source_items.id = events.primary_source_item_id
                LEFT JOIN sources ON sources.id = source_items.source_id
                LEFT JOIN event_source_items
                    ON event_source_items.event_id = events.id
                LEFT JOIN source_items AS evidence_items
                    ON evidence_items.id = event_source_items.source_item_id
                LEFT JOIN sources AS evidence_sources
                    ON evidence_sources.id = evidence_items.source_id
                WHERE COALESCE(events.event_at, events.latest_source_at, events.created_at) >= ?
                  AND COALESCE(events.event_at, events.latest_source_at, events.created_at) <= ?
                  AND events.headline IS NOT NULL
                  AND events.status <> 'duplicate'
                  AND events.status <> 'feedback_rejected'
                  AND events.status NOT LIKE 'rejected%'
                  -- Derived intelligence is allowed to summarize only material that
                  -- crossed an existing factual gate. Raw radar candidates, X/TG
                  -- opinions and unverified source items must never become "trends"
                  -- merely because their collector assigned a high score.
                  AND (
                      EXISTS (
                          SELECT 1
                          FROM event_fact_sheets
                          WHERE event_fact_sheets.event_id = events.id
                            AND event_fact_sheets.validation_status = 'valid'
                      )
                      OR EXISTS (
                          SELECT 1
                          FROM cards
                          WHERE cards.event_id = events.id
                            AND cards.validation_status = 'valid'
                      )
                  )
                GROUP BY events.id
                ORDER BY COALESCE(
                    events.event_at,
                    events.latest_source_at,
                    events.created_at
                ) DESC,
                events.id DESC
                LIMIT ?
                """,
                (since, until, limit),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            csv = str(item.pop("source_ids_csv") or "")
            source_ids = [value for value in csv.split(",") if value]
            if item.get("source_id") and item["source_id"] not in source_ids:
                source_ids.append(str(item["source_id"]))
            item["source_ids"] = source_ids
            result.append(item)
        return result

    def save_intelligence_product(
        self,
        *,
        kind: str,
        generation_key: str,
        schema_version: str,
        period_start: str,
        period_end: str,
        title: str,
        summary: str,
        payload: dict[str, Any],
        rendered_html: str,
        source_event_ids: list[int],
        validation_status: str,
        validation_errors: list[str] | None = None,
    ) -> dict[str, Any]:
        if validation_status not in {"valid", "insufficient_data", "invalid"}:
            raise ValueError(f"Unsupported intelligence validation status: {validation_status}")
        now = utc_now()
        with self.connection() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO intelligence_products(
                    kind, generation_key, schema_version, period_start, period_end,
                    title, summary, payload_json, rendered_html,
                    source_event_ids_json, validation_status,
                    validation_errors_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    kind,
                    generation_key,
                    schema_version,
                    period_start,
                    period_end,
                    title,
                    summary,
                    json.dumps(payload, ensure_ascii=False),
                    rendered_html,
                    json.dumps(source_event_ids, ensure_ascii=False),
                    validation_status,
                    json.dumps(validation_errors or [], ensure_ascii=False),
                    now,
                    now,
                ),
            )
            created = cursor.rowcount == 1
            row = connection.execute(
                "SELECT * FROM intelligence_products WHERE generation_key = ?",
                (generation_key,),
            ).fetchone()
        if row is None:  # pragma: no cover - guarded by the insert/select transaction
            raise RuntimeError("Intelligence product insert/select failed")
        result = _intelligence_product_row(row)
        result["cache_hit"] = not created
        return result

    def get_intelligence_product(self, product_id: int) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM intelligence_products WHERE id = ?",
                (product_id,),
            ).fetchone()
        return _intelligence_product_row(row) if row else None

    def latest_intelligence_product(self, kind: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM intelligence_products
                WHERE kind = ?
                ORDER BY period_end DESC, created_at DESC, id DESC
                LIMIT 1
                """,
                (kind,),
            ).fetchone()
        return _intelligence_product_row(row) if row else None

    def intelligence_status(self) -> dict[str, Any]:
        with self.connection() as connection:
            total = int(
                connection.execute("SELECT COUNT(*) FROM intelligence_products").fetchone()[0]
            )
            rows = connection.execute(
                """
                SELECT kind, validation_status, COUNT(*) AS count,
                       MAX(period_end) AS latest_period_end,
                       MAX(created_at) AS latest_created_at
                FROM intelligence_products
                GROUP BY kind, validation_status
                ORDER BY kind, validation_status
                """
            ).fetchall()
        return {"total": total, "products": [dict(row) for row in rows]}

    def record_detail_feedback(
        self,
        delivery_id: int,
        article_id: int,
        *,
        action: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        if action not in {"detail_useful", "detail_not_worth_it"}:
            raise ValueError(f"Unsupported detail feedback action: {action}")
        context = self.get_delivery_context(delivery_id)
        if context is None:
            raise ValueError(f"Unknown delivery_id: {delivery_id}")
        request = self.detail_request_for_delivery(delivery_id)
        if request is None or int(request.get("article_id") or 0) != article_id:
            raise ValueError("Article does not belong to this delivery")
        feedback_metadata = {"article_id": article_id, **(metadata or {})}
        with self.connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO feedback_events(
                    delivery_id, event_id, card_id, action, metadata_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    delivery_id,
                    context["event_id"],
                    context["card_id"],
                    action,
                    json.dumps(feedback_metadata, ensure_ascii=False),
                ),
            )
        return int(cursor.lastrowid)

    def start_run(self, run_type: str, metadata: dict[str, Any] | None = None) -> str:
        run_id = uuid.uuid4().hex
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO runs(id, run_type, status, started_at, metadata_json)
                VALUES (?, ?, 'running', ?, ?)
                """,
                (run_id, run_type, utc_now(), json.dumps(metadata or {}, ensure_ascii=False)),
            )
        return run_id

    def finish_run(
        self,
        run_id: str,
        *,
        status: str = "completed",
        processed_count: int = 0,
        error_count: int = 0,
        cost_usd: float = 0,
        error: str | None = None,
    ) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                UPDATE runs
                SET status = ?, finished_at = ?, processed_count = ?,
                    error_count = ?, cost_usd = ?, error = ?
                WHERE id = ?
                """,
                (
                    status,
                    utc_now(),
                    processed_count,
                    error_count,
                    cost_usd,
                    error,
                    run_id,
                ),
            )

    def claim_schedule_slot(self, job_type: str, slot_at: str, mode: str) -> int | None:
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if (
                mode == "live"
                and connection.execute(
                    "SELECT 1 FROM menu_preparations WHERE status='running' LIMIT 1"
                ).fetchone()
            ):
                # Leave the slot unclaimed: the schedule loop will retry within its grace window.
                return None
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO schedule_slots(job_type, slot_at, mode, status)
                VALUES (?, ?, ?, 'planned')
                """,
                (job_type, slot_at, mode),
            )
            if cursor.rowcount != 1:
                return None
            return int(cursor.lastrowid)

    def update_schedule_slot(
        self,
        slot_id: int,
        *,
        status: str,
        run_id: str | None = None,
        metrics: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                UPDATE schedule_slots
                SET status = ?, run_id = COALESCE(?, run_id), metrics_json = ?,
                    error = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    run_id,
                    json.dumps(metrics or {}, ensure_ascii=False),
                    error,
                    utc_now(),
                    slot_id,
                ),
            )

    def recent_schedule_slots(self, limit: int = 30) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM schedule_slots
                ORDER BY slot_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["metrics"] = json.loads(item.pop("metrics_json"))
            result.append(item)
        return result

    def record_usage(
        self,
        provider: str,
        operation: str,
        *,
        cost_usd: float = 0,
        request_count: int = 1,
        input_units: int | None = None,
        output_units: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO usage_ledger(
                    provider, operation, request_count, input_units,
                    output_units, cost_usd, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    provider,
                    operation,
                    request_count,
                    input_units,
                    output_units,
                    cost_usd,
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )

    def current_month_cost(self) -> float:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT COALESCE(SUM(cost_usd), 0) AS total
                FROM usage_ledger
                WHERE strftime('%Y-%m', created_at) = strftime('%Y-%m', 'now')
                """
            ).fetchone()
        return float(row["total"])

    def status_counts(self) -> dict[str, int | float]:
        with self.connection() as connection:
            source_count = connection.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
            enabled_count = connection.execute(
                "SELECT COUNT(*) FROM sources WHERE enabled = 1"
            ).fetchone()[0]
            candidate_count = connection.execute(
                "SELECT COUNT(*) FROM sources WHERE candidate_enabled = 1"
            ).fetchone()[0]
            run_count = connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            source_item_count = connection.execute("SELECT COUNT(*) FROM source_items").fetchone()[
                0
            ]
            event_count = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            valid_card_count = connection.execute(
                "SELECT COUNT(*) FROM cards WHERE validation_status = 'valid'"
            ).fetchone()[0]
            sent_delivery_count = connection.execute(
                "SELECT COUNT(*) FROM deliveries WHERE delivery_status = 'sent'"
            ).fetchone()[0]
            feedback_count = connection.execute("SELECT COUNT(*) FROM feedback_events").fetchone()[
                0
            ]
            research_document_count = connection.execute(
                "SELECT COUNT(*) FROM research_documents"
            ).fetchone()[0]
            valid_fact_sheet_count = connection.execute(
                """
                SELECT COUNT(*) FROM event_fact_sheets
                WHERE validation_status = 'valid'
                """
            ).fetchone()[0]
            ready_dossier_count = connection.execute(
                """
                SELECT COUNT(*) FROM event_dossiers AS dossier
                WHERE dossier.validation_status = 'valid'
                  AND dossier.decision = 'ready'
                  AND dossier.dossier_version = (
                      SELECT MAX(latest.dossier_version)
                      FROM event_dossiers AS latest
                      WHERE latest.event_id = dossier.event_id
                        AND latest.validation_status = 'valid'
                  )
                """
            ).fetchone()[0]
            preview_media_count = connection.execute(
                "SELECT COUNT(*) FROM card_media WHERE mode = 'telegram_preview'"
            ).fetchone()[0]
            author_draft_count = connection.execute(
                "SELECT COUNT(*) FROM author_drafts WHERE validation_status = 'valid'"
            ).fetchone()[0]
            detail_request_count = connection.execute(
                "SELECT COUNT(*) FROM detail_requests"
            ).fetchone()[0]
            valid_detailed_article_count = connection.execute(
                "SELECT COUNT(*) FROM detailed_articles WHERE validation_status = 'valid'"
            ).fetchone()[0]
        return {
            "sources": source_count,
            "enabled_sources": enabled_count,
            "candidate_sources": candidate_count,
            "source_items": source_item_count,
            "events": event_count,
            "valid_cards": valid_card_count,
            "sent_deliveries": sent_delivery_count,
            "feedback_events": feedback_count,
            "research_documents": research_document_count,
            "valid_fact_sheets": valid_fact_sheet_count,
            "ready_event_dossiers": ready_dossier_count,
            "preview_media_cards": preview_media_count,
            "valid_author_drafts": author_draft_count,
            "detail_requests": detail_request_count,
            "valid_detailed_articles": valid_detailed_article_count,
            "runs": run_count,
            "month_cost_usd": self.current_month_cost(),
        }

    def recent_valid_card_events(self, *, days: int = 7) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    events.id,
                    events.headline,
                    events.event_type,
                    events.entities_json,
                    events.topics_json,
                    events.regions_json,
                    events.score_features_json,
                    source_items.discovery_mode,
                    sources.kind AS source_kind,
                    sources.id AS source_id,
                    sources.name AS source_name,
                    COALESCE(deliveries.delivered_at, deliveries.created_at) AS delivered_at
                FROM cards
                JOIN events ON events.id = cards.event_id
                JOIN deliveries ON deliveries.card_id = cards.id
                JOIN source_items ON source_items.id = events.primary_source_item_id
                JOIN sources ON sources.id = source_items.source_id
                WHERE cards.validation_status = 'valid'
                  AND deliveries.delivery_status = 'sent'
                  AND deliveries.id = (
                      SELECT latest_delivery.id
                      FROM deliveries AS latest_delivery
                      WHERE latest_delivery.card_id = cards.id
                        AND latest_delivery.delivery_status = 'sent'
                      ORDER BY COALESCE(
                          latest_delivery.delivered_at,
                          latest_delivery.created_at
                      ) DESC,
                      latest_delivery.id DESC
                      LIMIT 1
                  )
                  AND COALESCE(deliveries.delivered_at, deliveries.created_at)
                      >= datetime('now', ?)
                ORDER BY COALESCE(deliveries.delivered_at, deliveries.created_at) DESC,
                         cards.id DESC
                """,
                (f"-{max(1, days)} days",),
            ).fetchall()
        return self.with_event_signal_metadata([dict(row) for row in rows])

    def baseline_delivery_rows(self, *, limit: int = 50) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    deliveries.id AS delivery_id,
                    deliveries.telegram_message_id,
                    deliveries.delivered_at,
                    cards.id AS card_id,
                    cards.event_id,
                    cards.card_version,
                    cards.prompt_version,
                    cards.model,
                    cards.generation_cost_usd,
                    cards.rendered_html,
                    cards.source_urls_json,
                    cards.primary_preview_url,
                    cards.created_at AS card_created_at,
                    events.headline AS event_headline,
                    events.event_type,
                    events.entities_json,
                    events.topics_json,
                    events.regions_json,
                    events.score,
                    events.score_features_json,
                    (
                        SELECT COALESCE(SUM(usage_ledger.cost_usd), 0)
                        FROM usage_ledger
                        WHERE CAST(
                            json_extract(usage_ledger.metadata_json, '$.event_id') AS INTEGER
                        ) = events.id
                          AND (
                              usage_ledger.operation LIKE 'web.%'
                              OR usage_ledger.operation LIKE 'research.fact_sheet%'
                          )
                    ) AS research_cost_usd,
                    sources.id AS source_id,
                    sources.name AS source_name,
                    (
                        SELECT feedback_events.reaction
                        FROM feedback_events
                        WHERE feedback_events.delivery_id = deliveries.id
                          AND feedback_events.reaction IS NOT NULL
                        ORDER BY feedback_events.id DESC
                        LIMIT 1
                    ) AS final_reaction,
                    (
                        SELECT feedback_events.comment
                        FROM feedback_events
                        WHERE feedback_events.delivery_id = deliveries.id
                          AND feedback_events.reaction IS NOT NULL
                        ORDER BY feedback_events.id DESC
                        LIMIT 1
                    ) AS final_comment
                FROM deliveries
                JOIN cards ON cards.id = deliveries.card_id
                JOIN events ON events.id = cards.event_id
                JOIN source_items ON source_items.id = events.primary_source_item_id
                JOIN sources ON sources.id = source_items.source_id
                WHERE deliveries.delivery_status = 'sent'
                  AND cards.validation_status = 'valid'
                ORDER BY COALESCE(deliveries.delivered_at, deliveries.created_at) DESC,
                         deliveries.id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def export_feedback_rows(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    feedback_events.id,
                    feedback_events.event_id,
                    feedback_events.card_id,
                    feedback_events.action,
                    feedback_events.reaction,
                    feedback_events.comment,
                    feedback_events.previous_reaction,
                    feedback_events.metadata_json,
                    feedback_events.created_at,
                    cards.rendered_html,
                    cards.source_urls_json
                FROM feedback_events
                JOIN cards ON cards.id = feedback_events.card_id
                ORDER BY feedback_events.created_at
                """
            ).fetchall()
        return [dict(row) for row in rows]
