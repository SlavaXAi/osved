"""Owner-bound reply sessions and idempotent inputs for question-driven news analysis."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime, timedelta

from app.db.repository import _detail_request_row
from app.research.detail_policy import article_reference_fingerprint, detail_policy_versions


def reply_session(repository, owner, reply_id):
    if not reply_id:
        return None
    with repository.connection() as db:
        row = db.execute(
            "SELECT s.* FROM news_question_sessions s JOIN deliveries d ON d.id=s.delivery_id "
            "WHERE s.chat_id=? AND d.chat_id=? AND d.delivery_status='sent' "
            "AND (d.telegram_message_id=? OR s.prompt_message_id=?) "
            "AND datetime(s.expires_at)>datetime('now')",
            (owner, owner, reply_id, reply_id),
        ).fetchone()
    return dict(row) if row else None


def open_session(repository, panel):
    now = datetime.now(UTC)
    with repository.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        delivery = db.execute(
            "SELECT chat_id,telegram_message_id FROM deliveries WHERE id=? "
            "AND delivery_status='sent'",
            (panel["delivery_id"],),
        ).fetchone()
        if not delivery or (delivery["chat_id"], delivery["telegram_message_id"]) != (
            panel["chat_id"],
            panel["message_id"],
        ):
            raise PermissionError("Foreign news card")
        active = db.execute(
            "SELECT 1 FROM detail_requests WHERE delivery_id=? AND status IN ('queued','running')",
            (panel["delivery_id"],),
        ).fetchone()
        if active:
            raise ValueError("Сначала дождись текущего разбора")
        existing = db.execute(
            "SELECT * FROM news_question_sessions WHERE chat_id=? AND delivery_id=? "
            "AND datetime(expires_at)>datetime(?)",
            (panel["chat_id"], panel["delivery_id"], now.isoformat()),
        ).fetchone()
        if existing:
            return dict(existing)
        db.execute(
            "INSERT INTO news_question_sessions(chat_id,delivery_id,expires_at,created_at) "
            "VALUES(?,?,?,?) ON CONFLICT(chat_id,delivery_id) DO UPDATE SET "
            "prompt_message_id=NULL,expires_at=excluded.expires_at,created_at=excluded.created_at",
            (
                panel["chat_id"],
                panel["delivery_id"],
                (now + timedelta(hours=24)).isoformat(),
                now.isoformat(),
            ),
        )
    return None


def claim_input(repository, session, message_id, kind, *, max_daily):
    now = datetime.now(UTC)
    with repository.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        if not db.execute(
            "SELECT 1 FROM news_question_sessions s JOIN deliveries d ON d.id=s.delivery_id "
            "WHERE s.chat_id=? AND s.delivery_id=? AND d.chat_id=s.chat_id "
            "AND d.delivery_status='sent' AND datetime(s.expires_at)>datetime(?)",
            (session["chat_id"], session["delivery_id"], now.isoformat()),
        ).fetchone():
            raise PermissionError("Вопрос отменён или время ответа истекло")
        if db.execute(
            "SELECT 1 FROM news_question_inputs WHERE chat_id=? AND message_id=?",
            (session["chat_id"], message_id),
        ).fetchone():
            return False
        # Receiving only downloads local input; no paid step has been queued yet.
        db.execute(
            "UPDATE news_question_inputs SET status='failed',error='input lease expired' "
            "WHERE status='receiving' AND datetime(created_at)<datetime(?,'-10 minutes')",
            (now.isoformat(),),
        )
        if (
            db.execute(
                "SELECT 1 FROM news_question_inputs WHERE delivery_id=? AND status='receiving'",
                (session["delivery_id"],),
            ).fetchone()
            or db.execute(
                "SELECT 1 FROM detail_requests WHERE delivery_id=? "
                "AND status IN ('queued','running')",
                (session["delivery_id"],),
            ).fetchone()
        ):
            raise ValueError("Сначала дождись ответа на предыдущий вопрос")
        count = db.execute(
            "SELECT count(*) FROM news_question_inputs WHERE chat_id=? "
            "AND date(created_at)=date(?) AND status IN ('receiving','queued')",
            (session["chat_id"], now.isoformat()),
        ).fetchone()[0]
        if count >= max_daily:
            raise ValueError("Сегодня достигнут лимит новых разборов")
        db.execute(
            "INSERT INTO news_question_inputs(chat_id,message_id,delivery_id,input_kind,status,"
            "created_at) VALUES(?,?,?,?,'receiving',?)",
            (session["chat_id"], message_id, session["delivery_id"], kind, now.isoformat()),
        )
    return True


def queue_question(repository, panel, message_id, question, *, voice=None):
    reader = {
        **panel["reader_context"],
        "reader_mode": "explainer",
        "user_question": question,
        "input_message_id": message_id,
        "baseline_delivery_id": panel["delivery_id"],
        **(voice or {}),
    }
    now = datetime.now(UTC).isoformat()
    with repository.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        incoming = db.execute(
            "SELECT * FROM news_question_inputs WHERE chat_id=? AND message_id=?",
            (panel["chat_id"], message_id),
        ).fetchone()
        if (
            not incoming
            or incoming["status"] != "receiving"
            or incoming["delivery_id"] != panel["delivery_id"]
        ):
            raise ValueError("Этот запрос уже принят")
        if not db.execute(
            "SELECT 1 FROM news_question_sessions WHERE chat_id=? AND delivery_id=? "
            "AND datetime(expires_at)>datetime(?)",
            (panel["chat_id"], panel["delivery_id"], now),
        ).fetchone():
            raise ValueError("Вопрос отменён или время ответа истекло")
        more = db.execute(
            "SELECT a.id,a.rendered_html FROM detail_requests r JOIN detailed_articles a "
            "ON a.id=r.article_id WHERE r.delivery_id=? AND r.depth='expanded' "
            "AND r.status IN ('ready','sent') AND a.validation_status='valid'",
            (panel["delivery_id"],),
        ).fetchone()
        if more:
            from app.details.paging import telegram_html_visible_text

            reader.update(
                expanded_reference_article_id=more["id"],
                expanded_reference_text=telegram_html_visible_text(more["rendered_html"]),
            )
        reference = article_reference_fingerprint(reader)
        reader["article_reference_fingerprint"] = reference
        reader["provider_fingerprint"] = hashlib.sha256(
            (reader["context_fingerprint"] + reference).encode()
        ).hexdigest()
        old = db.execute(
            "SELECT * FROM detail_requests WHERE delivery_id=? AND depth='article'",
            (panel["delivery_id"],),
        ).fetchone()
        if old and old["status"] in {"queued", "running", "ready"}:
            raise ValueError("Сначала открой результат предыдущего разбора")
        schema, prompt = detail_policy_versions("article")
        cached = db.execute(
            "SELECT * FROM detailed_articles WHERE event_id=? AND depth='article' "
            "AND context_fingerprint=? AND schema_version=? AND prompt_version=? "
            "AND json_extract(reader_context_json,'$.article_reference_fingerprint')=? "
            "AND validation_status='valid' AND datetime(expires_at)>datetime(?) "
            "ORDER BY id DESC LIMIT 1",
            (
                reader["baseline_event_id"],
                reader["context_fingerprint"],
                schema,
                prompt,
                reference,
                now,
            ),
        ).fetchone()
        reader.update(retry_authorization=uuid.uuid4().hex, retry_rejected=True)
        key = f"question:{reader['baseline_event_id']}:{reference}"
        values = (
            "ready" if cached else "queued",
            "delivery" if cached else "queued",
            cached["id"] if cached else None,
            cached["generation_key"] if cached else key,
            json.dumps(reader, ensure_ascii=False),
            int(bool(cached)),
            now,
            now,
        )
        if old:
            db.execute(
                "UPDATE detail_requests SET status=?,stage=?,article_id=?,generation_key=?,"
                "reader_context_json=?,cache_hit=?,requested_at=?,updated_at=?,"
                "lease_until=NULL,placeholder_lease_until=NULL,keyboard_synced_status=NULL,"
                "last_error=NULL,started_at=NULL,completed_at=NULL,attempt_count=0 WHERE id=?",
                (*values, old["id"]),
            )
            request_id = old["id"]
        else:
            request_id = db.execute(
                "INSERT INTO detail_requests(status,stage,article_id,generation_key,"
                "reader_context_json,cache_hit,requested_at,updated_at,delivery_id,card_id,"
                "event_id,chat_id,user_id,depth,placeholder_message_id) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,'article',?)",
                (
                    *values,
                    panel["delivery_id"],
                    reader["baseline_card_id"],
                    reader["baseline_event_id"],
                    panel["chat_id"],
                    panel["chat_id"],
                    panel["message_id"],
                ),
            ).lastrowid
        db.execute(
            "UPDATE news_question_inputs SET status='queued',question=?,request_id=?,"
            "article_reference_fingerprint=? WHERE chat_id=? AND message_id=?",
            (question, request_id, reference, panel["chat_id"], message_id),
        )
        db.execute(
            "DELETE FROM news_question_sessions WHERE chat_id=? AND delivery_id=?",
            (panel["chat_id"], panel["delivery_id"]),
        )
        result = _detail_request_row(
            db.execute("SELECT * FROM detail_requests WHERE id=?", (request_id,)).fetchone()
        )
        result["request_id"] = request_id
    return result
