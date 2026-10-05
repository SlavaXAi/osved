"""Durable queue for a deliberately small, read-only browser sensor.

No browser cookies, executable code, arbitrary target URLs or LLM prompts cross
this boundary. Replaying an accepted result is safe after a server restart.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

MAX_POSTS = 80
MAX_TEXT = 8000
ALLOWED_ERRORS = {
    "no_visible_tab", "wrong_feed", "login_required", "challenge", "empty_feed",
    "page_unavailable", "unsupported_language", "extension_error",
}
_SECRET = re.compile(
    r"(?i)\b(auth_token|ct0|authorization|api[_-]?key)\s*[:=]\s*[^\s,;]+"
)
_SECRET_PARAM = re.compile(r"cookie|token|authorization|password|secret|api.?key|ct0|session", re.I)


class BrowserTaskError(ValueError):
    """Invalid data or a stale/unauthorized task claim."""


def _safe_url(value: Any) -> str | None:
    if not isinstance(value, str) or len(value) > 2048:
        return None
    try:
        parsed = urlsplit(value)
        if parsed.scheme not in {"https", "http"} or not parsed.hostname:
            return None
        if parsed.username or parsed.password:
            return None
        # Keep article identifiers but drop tracking/session parameters. These
        # are evidence links; the bridge itself never fetches them.
        query = [
            (key, value) for key, value in parse_qsl(parsed.query, max_num_fields=50)
            if not _SECRET_PARAM.search(key) and not key.lower().startswith("utm_")
            and key.lower() not in {"fbclid", "gclid", "ref"}
        ]
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query), ""))
    except ValueError:
        return None


def sanitize_snapshot(result: Any, max_items: int = MAX_POSTS) -> dict[str, Any]:
    if not isinstance(result, dict) or set(result) - {"feed_mode", "posts", "observed_at"}:
        raise BrowserTaskError("unexpected_result_fields")
    if result.get("feed_mode") != "for_you":
        raise BrowserTaskError("wrong_feed")
    raw_posts = result.get("posts")
    if not isinstance(raw_posts, list) or len(raw_posts) > MAX_POSTS:
        raise BrowserTaskError("invalid_posts")
    posts, seen = [], set()
    for post in raw_posts:
        if not isinstance(post, dict) or set(post) - {
            "id", "author", "text", "published_at", "url", "links",
        }:
            raise BrowserTaskError("unexpected_post_fields")
        post_id, author = post.get("id"), post.get("author")
        if not isinstance(post_id, str) or not re.fullmatch(r"\d{5,30}", post_id):
            raise BrowserTaskError("invalid_post_id")
        if not isinstance(author, str) or not re.fullmatch(r"[A-Za-z0-9_]{1,15}", author):
            raise BrowserTaskError("invalid_author")
        text = post.get("text", "")
        if not isinstance(text, str) or len(text) > MAX_TEXT:
            raise BrowserTaskError("invalid_post_text")
        published_at = post.get("published_at")
        if not isinstance(published_at, str) or len(published_at) > 50:
            raise BrowserTaskError("invalid_post_date")
        try:
            parsed_date = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            if parsed_date.tzinfo is None:
                raise ValueError("timezone required")
        except ValueError as error:
            raise BrowserTaskError("invalid_post_date") from error
        raw_links = post.get("links", [])
        if not isinstance(raw_links, list) or len(raw_links) > 30:
            raise BrowserTaskError("invalid_links")
        if post_id in seen:
            continue
        seen.add(post_id)
        posts.append({
            "id": post_id,
            "author": author,
            "text": _SECRET.sub(r"\1=[REDACTED]", text),
            "published_at": parsed_date.isoformat(),
            "url": f"https://x.com/{author}/status/{post_id}",
            "links": list(dict.fromkeys(url for link in raw_links if (url := _safe_url(link)))),
        })
        if len(posts) >= max_items:
            break
    return {"feed_mode": "for_you", "posts": posts}


class BrowserTaskQueue:
    def __init__(self, database_path: Path, *, clock: Callable[[], float] = time.time) -> None:
        self.database_path = Path(database_path)
        self.clock = clock

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        migration = Path(__file__).parents[1] / "db/migrations/0013_browser_tasks.sql"
        with self.connection() as connection:
            connection.executescript(migration.read_text(encoding="utf-8"))

    def enqueue(
        self,
        source_id: str,
        *,
        task_key: str,
        max_items: int = 60,
        ttl_seconds: int = 86400,
        max_attempts: int = 3,
    ) -> str:
        if not source_id or len(source_id) > 200 or not task_key or len(task_key) > 240:
            raise BrowserTaskError("invalid_task_identity")
        if not 1 <= max_items <= MAX_POSTS or not 1 <= ttl_seconds <= 86400:
            raise BrowserTaskError("invalid_task_limits")
        if not 1 <= max_attempts <= 5:
            raise BrowserTaskError("invalid_attempt_limit")
        task_id, now = secrets.token_hex(16), self.clock()
        payload = json.dumps({"mode": "for_you", "max_items": max_items})
        with self.connection() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO browser_source_tasks
                (id, task_key, source_id, platform, task_kind, payload_json, status,
                 max_attempts, created_at, expires_at, updated_at)
                VALUES (?, ?, ?, 'x', 'visible_feed_snapshot', ?, 'queued', ?, ?, ?, ?)""",
                (task_id, task_key, source_id, payload, max_attempts, now, now + ttl_seconds, now),
            )
            row = connection.execute(
                "SELECT * FROM browser_source_tasks WHERE task_key = ?", (task_key,),
            ).fetchone()
            assert row is not None
            if row["source_id"] != source_id or row["payload_json"] != payload:
                raise BrowserTaskError("task_key_conflict")
            return str(row["id"])

    def claim(self, *, lease_seconds: int = 90) -> dict[str, Any] | None:
        if not 10 <= lease_seconds <= 300:
            raise BrowserTaskError("invalid_lease")
        now = self.clock()
        with self.connection() as connection:
            tables = {
                row[0] for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name IN ('sources','runtime_settings')"
                )
            }
            if tables == {"sources", "runtime_settings"}:
                # Recheck inside the claim transaction. A collector may have
                # enqueued just after the user's source-off cancellation hook.
                connection.execute(
                    """UPDATE browser_source_tasks SET status='failed',
                    error_code='source_disabled', claim_token=NULL, lease_until=NULL, updated_at=?
                    WHERE status IN ('queued','claimed') AND EXISTS (
                        SELECT 1 FROM sources s WHERE s.id=browser_source_tasks.source_id
                        AND COALESCE((SELECT CASE rs.value
                            WHEN 'true' THEN 1 WHEN 'false' THEN 0 END
                            FROM runtime_settings rs WHERE rs.key='source_enabled:'||s.id),
                            s.enabled)=0)""", (now,),
                )
            connection.execute(
                """UPDATE browser_source_tasks SET status = 'expired', updated_at = ?,
                claim_token = NULL, lease_until = NULL, error_code = 'task_expired'
                WHERE status IN ('queued', 'claimed') AND expires_at <= ?""", (now, now),
            )
            connection.execute(
                """UPDATE browser_source_tasks SET
                status = CASE WHEN attempt_count >= max_attempts THEN 'failed' ELSE 'queued' END,
                error_code = 'lease_expired', claim_token = NULL, lease_until = NULL, updated_at = ?
                WHERE status = 'claimed' AND lease_until <= ?""", (now, now),
            )
            row = connection.execute(
                """SELECT * FROM browser_source_tasks WHERE status = 'queued'
                ORDER BY created_at, id LIMIT 1"""
            ).fetchone()
            if row is None:
                return None
            token = secrets.token_urlsafe(32)
            connection.execute(
                """UPDATE browser_source_tasks SET status = 'claimed', claim_token = ?,
                lease_until = ?, attempt_count = attempt_count + 1, updated_at = ? WHERE id = ?""",
                (token, now + lease_seconds, now, row["id"]),
            )
            return {
                "id": row["id"], "platform": row["platform"], "kind": row["task_kind"],
                "payload": json.loads(row["payload_json"]), "claim_token": token,
                "lease_until": now + lease_seconds,
            }

    def _claim_row(
        self, connection: sqlite3.Connection, task_id: str, claim_token: str,
    ) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM browser_source_tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if row is None or not isinstance(claim_token, str) or not hmac.compare_digest(
            (row["claim_token"] or "").encode(), claim_token.encode(),
        ):
            raise BrowserTaskError("invalid_claim")
        return row

    def stage(self, task_id: str, claim_token: str, result: Any) -> bool:
        """Persist a snapshot. True means first write; False is an identical replay."""
        now = self.clock()
        with self.connection() as connection:
            row = self._claim_row(connection, task_id, claim_token)
            clean = sanitize_snapshot(result, json.loads(row["payload_json"])["max_items"])
            encoded = json.dumps(clean, ensure_ascii=False, sort_keys=True)
            digest = hashlib.sha256(encoded.encode()).hexdigest()
            if row["status"] in {"staged", "committed", "completed"}:
                if row["result_digest"] == digest:
                    return False
                raise BrowserTaskError("result_conflict")
            if row["status"] != "claimed" or row["lease_until"] <= now or row["expires_at"] <= now:
                raise BrowserTaskError("stale_claim")
            connection.execute(
                """UPDATE browser_source_tasks SET status = 'staged', result_json = ?,
                result_digest = ?, updated_at = ? WHERE id = ?""", (encoded, digest, now, task_id),
            )
            return True

    def fail(self, task_id: str, claim_token: str, error_code: str) -> None:
        if error_code not in ALLOWED_ERRORS:
            raise BrowserTaskError("invalid_error_code")
        with self.connection() as connection:
            row = self._claim_row(connection, task_id, claim_token)
            if row["status"] != "claimed" or row["lease_until"] <= self.clock():
                raise BrowserTaskError("stale_claim")
            # A reported error is terminal. Repeated automatic login/challenge
            # attempts or background empty polls would waste the user's attention.
            connection.execute(
                """UPDATE browser_source_tasks SET status = 'failed', error_code = ?,
                claim_token = NULL, lease_until = NULL, updated_at = ? WHERE id = ?""",
                (error_code, self.clock(), task_id),
            )

    def ready_results(self, source_id: str | None = None, *, limit: int = 20) -> list[dict]:
        if not 1 <= limit <= 100:
            raise BrowserTaskError("invalid_result_limit")
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT id, source_id, status, result_json, created_at, updated_at
                FROM browser_source_tasks WHERE status IN ('staged', 'committed')
                AND (? IS NULL OR source_id = ?) ORDER BY created_at LIMIT ?""",
                (source_id, source_id, limit),
            ).fetchall()
            return [{**dict(row), "result": json.loads(row["result_json"])} for row in rows]

    def mark_committed(self, task_id: str) -> None:
        """Call after idempotent source-item inserts have committed."""
        self._advance(task_id, "staged", "committed")

    def complete(self, task_id: str) -> None:
        self._advance(task_id, "committed", "completed")

    def _advance(self, task_id: str, before: str, after: str) -> None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT status FROM browser_source_tasks WHERE id = ?", (task_id,),
            ).fetchone()
            if row is None:
                raise BrowserTaskError("unknown_task")
            if row["status"] in {after, "completed"}:
                return
            if row["status"] != before:
                raise BrowserTaskError("invalid_transition")
            connection.execute(
                "UPDATE browser_source_tasks SET status = ?, updated_at = ? WHERE id = ?",
                (after, self.clock(), task_id),
            )

    def task_status(self, task_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                """SELECT id, source_id, status, attempt_count, max_attempts, error_code,
                created_at, expires_at, updated_at FROM browser_source_tasks WHERE id = ?""",
                (task_id,),
            ).fetchone()
            return dict(row) if row else None

    def latest_source_task(self, source_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                """SELECT id, task_key, status, error_code, created_at, expires_at
                FROM browser_source_tasks
                WHERE source_id = ? ORDER BY created_at DESC, rowid DESC LIMIT 1""", (source_id,),
            ).fetchone()
            return dict(row) if row else None

    def has_open_task(self, source_id: str) -> bool:
        with self.connection() as connection:
            return connection.execute(
                """SELECT 1 FROM browser_source_tasks WHERE source_id = ?
                AND (status IN ('staged', 'committed')
                OR (status IN ('queued', 'claimed') AND expires_at > ?)) LIMIT 1""",
                (source_id, self.clock()),
            ).fetchone() is not None

    def task_exists(self, task_key: str) -> bool:
        with self.connection() as connection:
            return connection.execute(
                "SELECT 1 FROM browser_source_tasks WHERE task_key = ?", (task_key,),
            ).fetchone() is not None

    def cancel_pending(self, source_id: str) -> int:
        """Invalidate unsent claims immediately when the source is disabled."""
        with self.connection() as connection:
            cursor = connection.execute(
                """UPDATE browser_source_tasks
                SET status = 'failed', error_code = 'source_disabled',
                claim_token = NULL, lease_until = NULL, updated_at = ?
                WHERE source_id = ? AND status IN ('queued', 'claimed')""",
                (self.clock(), source_id),
            )
            return cursor.rowcount
