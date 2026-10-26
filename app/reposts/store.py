from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

from app.config import SourceConfig, SourceRegistry


def stamp():
    return datetime.now(UTC).isoformat()


def get(repository, ident):
    with repository.connection() as db:
        row = db.execute("SELECT * FROM x_reposts WHERE id=?", (ident,)).fetchone()
    return dict(row) if row else None


def enqueue(repository, *, owner, message_id, link):
    if link[0] != "x":
        raise ValueError("Only X posts can become reposts")
    repository.sync_sources(
        SourceRegistry(
            version=1,
            sources=[
                SourceConfig(
                    id="owner_x_reposts",
                    name="Owner-selected X posts",
                    kind="x",
                    locator="owner_selected_posts",
                    role="own_history",
                    enabled=False,
                    candidate_enabled=False,
                    region="global",
                    language="und",
                    topics=[],
                )
            ],
        )
    )
    now = stamp()
    with repository.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        old = db.execute(
            "SELECT * FROM x_reposts WHERE chat_id=? AND tweet_id=?", (owner, link[1])
        ).fetchone()
        if old:
            return dict(old), False
        if (
            db.execute(
                "SELECT count(*) FROM x_reposts WHERE status IN ('queued','running')"
            ).fetchone()[0]
            >= 20
        ):
            raise ValueError("В очереди уже 20 публикаций. Дождись готовности текущих.")
        item = db.execute(
            "INSERT INTO source_items(source_id,external_id,source_url,original_text,"
            "original_language,published_at,payload_hash,processing_status) "
            "VALUES('owner_x_reposts',?,?,?,'und',?,?,'history')",
            (
                f"{owner}:{link[1]}",
                link[2],
                "",
                now,
                hashlib.sha256(f"{owner}:{link[1]}".encode()).hexdigest(),
            ),
        ).lastrowid
        event = db.execute(
            "INSERT INTO events(status,event_type,headline,first_seen_at,latest_source_at,"
            "primary_source_item_id,fingerprint) VALUES('user_requested','x_repost',?,?,?,?,?)",
            ("Публикация X", now, now, item, f"x-repost:{owner}:{link[1]}"),
        ).lastrowid
        db.execute(
            "INSERT INTO event_source_items(event_id,source_item_id,evidence_role) "
            "VALUES(?,?,'unverified_post')",
            (event, item),
        )
        ident = db.execute(
            "INSERT INTO x_reposts(chat_id,tweet_id,source_url,request_message_id,event_id,"
            "created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
            (owner, link[1], link[2], message_id, event, now, now),
        ).lastrowid
        db.execute(
            "INSERT INTO x_curation_samples(chat_id,tweet_id,source_url,"
            "provenance,selected_at,imported_at) VALUES(?,?,?,'owner_link',?,?) "
            "ON CONFLICT(chat_id,tweet_id) DO UPDATE SET selected_at=excluded.selected_at,"
            "provenance=excluded.provenance",
            (owner, link[1], link[2], now, now),
        )
    return get(repository, ident), True


def save_post(repository, row, post):
    with repository.connection() as db:
        db.execute(
            "UPDATE x_reposts SET post_json=?,updated_at=? WHERE id=?",
            (json.dumps(post, ensure_ascii=False), stamp(), row["id"]),
        )
        db.execute(
            "UPDATE source_items SET original_text=?,original_language=?,published_at=?,"
            "outbound_urls_json=?,media_metadata_json=? WHERE id=(SELECT "
            "primary_source_item_id FROM events WHERE id=?)",
            (
                post["text"],
                post.get("language", "und"),
                post.get("published_at") or stamp(),
                json.dumps(post.get("urls", [])),
                json.dumps(post.get("media", [])),
                row["event_id"],
            ),
        )
        db.execute(
            "UPDATE x_curation_samples SET author=?,text=? WHERE chat_id=? AND tweet_id=?",
            (post["author"], post["text"], row["chat_id"], row["tweet_id"]),
        )


def vote(repository, row, value):
    if value not in (-1, 1):
        raise ValueError("Unknown vote")
    with repository.connection() as db:
        db.execute(
            "UPDATE x_curation_samples SET vote=?,voted_at=? WHERE chat_id=? AND tweet_id=?",
            (value, stamp(), row["chat_id"], row["tweet_id"]),
        )


def for_delivery(repository, delivery_id):
    with repository.connection() as db:
        row = db.execute("SELECT * FROM x_reposts WHERE delivery_id=?", (delivery_id,)).fetchone()
    return dict(row) if row else None


def retry(repository, row):
    with repository.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        if row.get("delivery_id"):
            panel = db.execute(
                "SELECT view,previous_view FROM news_panels WHERE delivery_id=?",
                (row["delivery_id"],),
            ).fetchone()
            if panel and panel["view"] != "base":
                raise ValueError("Сначала вернись к исходной публикации")
            if db.execute(
                "SELECT 1 FROM detail_requests WHERE delivery_id=? "
                "AND status IN ('queued','running')",
                (row["delivery_id"],),
            ).fetchone():
                raise ValueError("Сначала дождись текущего разбора")
        db.execute(
            "UPDATE x_reposts SET status='queued',updated_at=? "
            "WHERE id=? AND status IN ('failed','ready')",
            (stamp(), row["id"]),
        )


def saved_thesis(repository, panel, input_id):
    """The article request slot is reusable: restore by the saved question fingerprint."""
    with repository.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        previous = db.execute(
            "SELECT * FROM news_question_inputs WHERE chat_id=? AND message_id=?",
            (panel["chat_id"], input_id),
        ).fetchone()
        if not previous or not previous["request_id"]:
            return False, None
        request = db.execute(
            "SELECT * FROM detail_requests WHERE id=? AND delivery_id=?",
            (previous["request_id"], panel["delivery_id"]),
        ).fetchone()
        if not request:
            return True, None
        reader = json.loads(request["reader_context_json"])
        reference = previous["article_reference_fingerprint"]
        if reader.get("article_reference_fingerprint") != reference:
            if request["status"] in {"queued", "running", "ready"}:
                raise ValueError("Сначала открой результат текущего разбора")
            cached = db.execute(
                "SELECT * FROM detailed_articles WHERE event_id=? AND depth='article' "
                "AND context_fingerprint=? AND validation_status='valid' "
                "AND json_extract(reader_context_json,'$.article_reference_fingerprint')=? "
                "ORDER BY id DESC LIMIT 1",
                (request["event_id"], panel["reader_context"]["context_fingerprint"], reference),
            ).fetchone()
            if not cached:
                return True, None
            db.execute(
                "UPDATE detail_requests SET status='ready',stage='delivery',article_id=?,"
                "generation_key=?,reader_context_json=?,cache_hit=1,updated_at=?,"
                "last_error=NULL,keyboard_synced_status=NULL WHERE id=?",
                (
                    cached["id"],
                    cached["generation_key"],
                    cached["reader_context_json"],
                    stamp(),
                    request["id"],
                ),
            )
        request_id = request["id"]
    result = repository.get_detail_request(request_id)
    # Never accidentally show another question's answer from the same mutable slot.
    if result["reader_context"].get("article_reference_fingerprint") != reference:
        return True, None
    return True, result
