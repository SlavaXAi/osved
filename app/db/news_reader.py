"""Tier-scoped news research persistence; legacy requests stay independent."""

from __future__ import annotations

import json
from datetime import timedelta

from app.db.repository import _as_utc_datetime, _detail_request_row, _detailed_article_row
from app.details.paging import telegram_html_visible_text, telegram_html_visible_units
from app.research.detail_policy import (
    article_reference_fingerprint,
    current_detail_policy_sql,
    detail_policy_versions,
)


def validate_depth(depth):
    if depth not in {"expanded", "article"}:
        raise ValueError("Unknown news reader depth")


def cached_article(
    repository,
    *,
    depth,
    event_id=None,
    generation_key=None,
    context_fingerprint=None,
    reader_context=None,
    now=None,
):
    validate_depth(depth)
    if not context_fingerprint:
        return None
    clauses = [
        "depth=?",
        "validation_status='valid'",
        current_detail_policy_sql(depth=depth),
        "datetime(expires_at)>datetime(?)",
    ]
    args = [depth, _as_utc_datetime(now).isoformat()]
    if depth == "article" and reader_context is not None:
        clauses.append("json_extract(reader_context_json,'$.article_reference_fingerprint')=?")
        args.append(article_reference_fingerprint(reader_context))
    for field, value in [
        ("event_id", event_id),
        ("generation_key", generation_key),
        ("context_fingerprint", context_fingerprint),
    ]:
        if value is not None:
            clauses.append(field + "=?")
            args.append(value)
    with repository.connection() as db:
        row = db.execute(
            "SELECT * FROM detailed_articles WHERE "
            + " AND ".join(clauses)
            + " ORDER BY id DESC LIMIT 1",
            args,
        ).fetchone()
    return _detailed_article_row(row) if row else None


def enqueue(
    repository,
    delivery_id,
    user_id,
    *,
    depth,
    reader_context,
    generation_key=None,
    max_daily_new=5,
    now=None,
):
    validate_depth(depth)
    if max_daily_new < 0:
        raise ValueError("max_daily_new must be non-negative")
    context = dict(reader_context or {})
    fingerprint = str(context.get("context_fingerprint") or "")
    if len(fingerprint) != 64:
        raise ValueError("Reader baseline fingerprint is required")
    stamp = _as_utc_datetime(now)
    start = stamp.replace(hour=0, minute=0, second=0, microsecond=0)
    with repository.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        delivery = db.execute(
            "SELECT d.*,c.event_id FROM deliveries d JOIN cards c ON c.id=d.card_id WHERE d.id=?",
            (delivery_id,),
        ).fetchone()
        if not delivery or delivery["delivery_status"] != "sent":
            raise ValueError("News has not been delivered")
        if delivery["chat_id"] != user_id or context.get("baseline_card_id") != delivery["card_id"]:
            raise PermissionError("Foreign news baseline")
        existing = db.execute(
            "SELECT * FROM detail_requests WHERE delivery_id=? AND depth=?", (delivery_id, depth)
        ).fetchone()
        if existing:
            result = _detail_request_row(existing)
            if result["reader_context"].get("context_fingerprint") != fingerprint:
                raise ValueError("Frozen news baseline changed")
            if result["status"] in {"ready", "sent"}:
                artifact = db.execute(
                    "SELECT * FROM detailed_articles WHERE id=?", (result["article_id"],)
                ).fetchone()
                if (
                    not artifact
                    or artifact["validation_status"] != "valid"
                    or artifact["depth"] != depth
                    or artifact["event_id"] != delivery["event_id"]
                    or artifact["context_fingerprint"] != fingerprint
                    or (
                        depth == "article"
                        and article_reference_fingerprint(
                            json.loads(artifact["reader_context_json"])
                        )
                        != article_reference_fingerprint(result["reader_context"])
                    )
                ):
                    raise ValueError("Reader result belongs to a different baseline")
            return {**result, "created": False}
        if depth == "article":
            # These values come only from this delivery's stored, validated More result.
            # They are frozen once, separately from the original baseline fingerprint.
            for field in (
                "expanded_reference_article_id",
                "expanded_reference_text",
                "expanded_reference_units",
            ):
                context.pop(field, None)
            expanded = db.execute(
                "SELECT a.id,a.rendered_html FROM detail_requests r "
                "JOIN detailed_articles a ON a.id=r.article_id "
                "WHERE r.delivery_id=? AND r.event_id=? AND r.depth='expanded' "
                "AND r.status IN ('ready','sent') AND a.depth='expanded' "
                "AND a.event_id=r.event_id AND a.validation_status='valid' "
                "AND a.context_fingerprint=? "
                "AND json_extract(r.reader_context_json,'$.context_fingerprint')=?",
                (delivery_id, delivery["event_id"], fingerprint, fingerprint),
            ).fetchone()
            if expanded:
                text = telegram_html_visible_text(expanded["rendered_html"])
                units = telegram_html_visible_units(expanded["rendered_html"])
                if text.strip() and units > 0:
                    context.update(
                        expanded_reference_article_id=int(expanded["id"]),
                        expanded_reference_text=text,
                        expanded_reference_units=units,
                    )
            context["article_reference_fingerprint"] = article_reference_fingerprint(context)
        reference_clause = (
            " AND json_extract(reader_context_json,'$.article_reference_fingerprint')=?"
            if depth == "article"
            else ""
        )
        cached = db.execute(
            "SELECT * FROM detailed_articles WHERE event_id=? AND depth=? "
            "AND context_fingerprint=? "
            "AND validation_status='valid' AND "
            + current_detail_policy_sql(depth=depth)
            + reference_clause
            + " AND datetime(expires_at)>datetime(?) ORDER BY id DESC LIMIT 1",
            (
                delivery["event_id"],
                depth,
                fingerprint,
                *([context["article_reference_fingerprint"]] if depth == "article" else []),
                stamp.isoformat(),
            ),
        ).fetchone()
        key = generation_key or (
            f"event:{delivery['event_id']}:{depth}:{fingerprint}:"
            + detail_policy_versions(depth)[1]
            + (":" + context["article_reference_fingerprint"] if depth == "article" else "")
        )
        if not cached:
            count = db.execute(
                "SELECT count(*) FROM detail_requests WHERE user_id=? AND cache_hit=0 "
                "AND datetime(requested_at)>=datetime(?) AND datetime(requested_at)<datetime(?)",
                (user_id, start.isoformat(), (start + timedelta(days=1)).isoformat()),
            ).fetchone()[0]
            if count >= max_daily_new:
                return {
                    "request_id": None,
                    "delivery_id": delivery_id,
                    "status": "daily_limit",
                    "created": False,
                    "cache_hit": False,
                    "depth": depth,
                }
        cursor = db.execute(
            "INSERT INTO detail_requests(delivery_id,card_id,event_id,chat_id,user_id,"
            "generation_key,"
            "depth,reader_context_json,status,stage,article_id,cache_hit,placeholder_message_id,"
            "requested_at,completed_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                delivery_id,
                delivery["card_id"],
                delivery["event_id"],
                delivery["chat_id"],
                user_id,
                cached["generation_key"] if cached else key,
                depth,
                json.dumps(context, ensure_ascii=False),
                "ready" if cached else "queued",
                "delivery" if cached else "queued",
                cached["id"] if cached else None,
                int(bool(cached)),
                delivery["telegram_message_id"],
                stamp.isoformat(),
                stamp.isoformat() if cached else None,
                stamp.isoformat(),
            ),
        )
        row = db.execute("SELECT * FROM detail_requests WHERE id=?", (cursor.lastrowid,)).fetchone()
    return {**_detail_request_row(row), "created": True}


def store(repository, request_id, *, depth, reader_context=None, context_fingerprint=None, **data):
    validate_depth(depth)
    schema, prompt = detail_policy_versions(depth)
    if (data["schema_version"], data["prompt_version"]) != (schema, prompt):
        raise ValueError("Reader artifact has an outdated policy")
    if data["validation_status"] != "valid" or data["generation_cost_usd"] < 0:
        raise ValueError("Reader artifact is not valid")
    context = dict(reader_context or {})
    fingerprint = context_fingerprint or context.get("context_fingerprint")
    if not fingerprint:
        raise ValueError("Reader artifact has no baseline fingerprint")
    stamp = _as_utc_datetime(data.pop("now", None)).isoformat()
    with repository.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        request = db.execute("SELECT * FROM detail_requests WHERE id=?", (request_id,)).fetchone()
        if not request or request["status"] != "running" or request["depth"] != depth:
            raise ValueError("Reader request is not running at this depth")
        original = json.loads(request["reader_context_json"])
        if original.get("context_fingerprint") != fingerprint:
            raise ValueError("Reader artifact has a foreign baseline")
        if depth == "article":
            if article_reference_fingerprint(original) != article_reference_fingerprint(context):
                raise ValueError("Article artifact has a foreign reading reference")
            context["article_reference_fingerprint"] = article_reference_fingerprint(context)
        db.execute(
            "INSERT INTO detailed_articles(event_id,generation_key,schema_version,prompt_version,"
            "model,provider_response_metadata_json,fact_snapshot_json,document_ids_json,"
            "structured_output_json,rendered_html,source_urls_json,validation_status,"
            "validation_errors_json,generation_cost_usd,expires_at,updated_at,depth,"
            "reader_context_json,context_fingerprint) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(generation_key) DO NOTHING",
            (
                request["event_id"],
                data["generation_key"],
                schema,
                prompt,
                data["model"],
                json.dumps(data["provider_response_metadata"], ensure_ascii=False),
                json.dumps(data["fact_snapshot"], ensure_ascii=False),
                json.dumps(data["document_ids"]),
                json.dumps(data["structured_output"], ensure_ascii=False),
                data["rendered_html"],
                json.dumps(data["source_urls"]),
                "valid",
                json.dumps(data["validation_errors"] or []),
                data["generation_cost_usd"],
                data["expires_at"],
                stamp,
                depth,
                json.dumps(context, ensure_ascii=False),
                fingerprint,
            ),
        )
        artifact = db.execute(
            "SELECT * FROM detailed_articles WHERE generation_key=?", (data["generation_key"],)
        ).fetchone()
        if (
            not artifact
            or artifact["event_id"] != request["event_id"]
            or artifact["depth"] != depth
            or artifact["context_fingerprint"] != fingerprint
            or artifact["validation_status"] != "valid"
            or artifact["schema_version"] != schema
            or artifact["prompt_version"] != prompt
            or (
                depth == "article"
                and article_reference_fingerprint(json.loads(artifact["reader_context_json"]))
                != article_reference_fingerprint(context)
            )
            or _as_utc_datetime(artifact["expires_at"]) <= _as_utc_datetime(stamp)
        ):
            raise ValueError("Conflicting reader artifact")
        db.execute(
            "UPDATE detail_requests SET status='ready',stage='delivery',article_id=?,"
            "generation_key=?,lease_until=NULL,last_error=NULL,completed_at=?,updated_at=? "
            ",reader_context_json=? "
            "WHERE id=?",
            (
                artifact["id"],
                data["generation_key"],
                stamp,
                stamp,
                json.dumps(context, ensure_ascii=False),
                request_id,
            ),
        )
    return int(artifact["id"])


def attach_cache(repository, request_id, article_id, now=None):
    stamp = _as_utc_datetime(now).isoformat()
    with repository.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        request = db.execute("SELECT * FROM detail_requests WHERE id=?", (request_id,)).fetchone()
        if not request or request["status"] != "running":
            return False
        depth = request["depth"]
        fingerprint = json.loads(request["reader_context_json"]).get("context_fingerprint")
        article = db.execute(
            "SELECT * FROM detailed_articles WHERE id=? AND event_id=? AND depth=? "
            "AND context_fingerprint=? AND validation_status='valid' AND "
            + current_detail_policy_sql(depth=depth)
            + " AND datetime(expires_at)>datetime(?)",
            (article_id, request["event_id"], depth, fingerprint, stamp),
        ).fetchone()
        if (
            article
            and depth == "article"
            and article_reference_fingerprint(json.loads(article["reader_context_json"]))
            != (article_reference_fingerprint(json.loads(request["reader_context_json"])))
        ):
            return False
        if not article:
            return False
        db.execute(
            "UPDATE detail_requests SET status='ready',stage='delivery',article_id=?,"
            "generation_key=?,cache_hit=1,lease_until=NULL,last_error=NULL,completed_at=?,"
            "updated_at=? WHERE id=?",
            (article_id, article["generation_key"], stamp, stamp, request_id),
        )
    return True
