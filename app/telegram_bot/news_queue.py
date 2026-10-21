"""Durable ordinary-news delivery, with optional rating-gated or hybrid flow."""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from app.collectors.ingest import redact_text
from app.config import DeliveryFormat, Settings
from app.db.repository import Repository

MODE_KEY = "news_delivery_mode"


def queue_enabled(repository: Repository) -> bool:
    return repository.get_runtime_setting(MODE_KEY) in {"on_rating", "hybrid"}


def rating_gate_enabled(repository: Repository) -> bool:
    return repository.get_runtime_setting(MODE_KEY) == "on_rating"


def current_news(repository: Repository, owner_id: int) -> dict[str, Any] | None:
    raw = repository.get_runtime_setting(f"news_queue_current:{owner_id}")
    if not raw or not raw.isdigit():
        return None
    with repository.connection() as db:
        row = db.execute(
            "SELECT * FROM deliveries WHERE id=? AND chat_id=?", (int(raw), owner_id)
        ).fetchone()
    return dict(row) if row else None


async def send_next_news(
    settings: Settings,
    repository: Repository,
    *,
    bot=None,
    after_delivery_id: int | None = None,
    retry: bool = False,
    limit: int = 1,
) -> dict[str, Any]:
    from app.ops.schedule import fresh_delivery_card_ids
    from app.telegram_bot.bot import build_bot, configured_owner_id, send_formatted_card

    result: dict[str, Any] = {"sent": 0, "errors": 0, "cost_usd": 0.0}
    if not queue_enabled(repository):
        return {**result, "status": "queue_disabled"}
    batch_size = 1 if rating_gate_enabled(repository) or retry else max(0, min(limit, 10))
    if not batch_size:
        return {**result, "status": "nothing_requested"}
    owner = configured_owner_id(settings, repository)
    if owner is None:
        return {**result, "status": "owner_not_configured"}
    lock_key = f"news_queue_lock:{owner}"
    token = uuid.uuid4().hex
    with repository.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT value FROM runtime_settings WHERE key=?", (lock_key,)).fetchone()
        try:
            locked_until = float(json.loads(row[0])["until"]) if row else 0
        except (ValueError, TypeError, KeyError):
            return {**result, "status": "queue_lock_invalid", "errors": 1}
        if locked_until > time.time():
            return {**result, "status": "sending"}
        lock_value = json.dumps({"token": token, "until": time.time() + 180})
        db.execute(
            "INSERT INTO runtime_settings(key,value,updated_at) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at",
            (lock_key, lock_value, datetime.now(UTC).isoformat()),
        )
    own_bot = bot is None
    try:
        current = current_news(repository, owner)
        if after_delivery_id is not None and (
            current is None or current["id"] != after_delivery_id
        ):
            return {**result, "status": "already_advanced"}
        if after_delivery_id is not None and not repository.current_reaction(current["id"]):
            return {**result, "status": "waiting_for_rating", "delivery_id": current["id"]}
        retry_card = None
        if current and not repository.current_reaction(current["id"]):
            if current["delivery_status"] == "sent":
                if rating_gate_enabled(repository):
                    return {**result, "status": "waiting_for_rating", "delivery_id": current["id"]}
            elif not retry:
                return {**result, "status": "delivery_confirmation_needed", "errors": 1}
            else:
                with repository.connection() as db:
                    row = db.execute(
                        "SELECT *,id AS card_id FROM cards WHERE id=?", (current["card_id"],)
                    ).fetchone()
                if row is None:
                    return {**result, "status": "card_missing", "errors": 1}
                retry_card = dict(row)
        if retry_card:
            cards = [retry_card]
        else:
            ids = fresh_delivery_card_ids(
                repository, owner_id=owner, now=datetime.now(UTC), limit=batch_size
            )
            if not ids:
                return {**result, "status": "queue_empty"}
            cards = repository.deliverable_cards_by_ids(owner, ids)
            if not cards:
                return {**result, "status": "queue_empty"}
        if bot is None:
            bot = build_bot(settings)
        sent_ids = []
        for card in cards:
            # Keep one cross-process lease for the entire batch; renew before
            # each network request. An old click cannot interleave a new batch.
            renewed = json.dumps({"token": token, "until": time.time() + 180})
            with repository.connection() as db:
                claimed = db.execute(
                    "UPDATE runtime_settings SET value=?,updated_at=? WHERE key=? AND value=?",
                    (renewed, datetime.now(UTC).isoformat(), lock_key, lock_value),
                ).rowcount
            if not claimed:
                return {**result, "status": "queue_lock_lost", "errors": 1, "sent": len(sent_ids)}
            lock_value = renewed
            delivery_id = repository.ensure_delivery(card["card_id"], owner)
            from app.telegram_bot.bot import menu_reserved_delivery

            if menu_reserved_delivery(repository, delivery_id):
                continue
            # The frontier is the last card, not every card in the batch. Only
            # rating this frontier requests more. Old ratings remain feedback.
            repository.set_runtime_setting(f"news_queue_current:{owner}", str(delivery_id))
            try:
                message = await send_formatted_card(
                    bot,
                    chat_id=owner,
                    card=card,
                    delivery_id=delivery_id,
                    delivery_format=DeliveryFormat.REGULAR,
                    include_details=settings.details_enabled,
                )
                repository.mark_delivery_sent(delivery_id, message.message_id)
                sent_ids.append(delivery_id)
            except Exception as exc:
                repository.mark_delivery_error(delivery_id, redact_text(str(exc)))
                return {
                    **result,
                    "status": "delivery_confirmation_needed",
                    "errors": 1,
                    "sent": len(sent_ids),
                    "delivery_ids": sent_ids,
                }
        if not sent_ids:
            return {**result, "status": "queue_empty"}
        return {
            **result,
            "status": "sent",
            "sent": len(sent_ids),
            "delivery_id": sent_ids[-1],
            "delivery_ids": sent_ids,
        }
    finally:
        try:
            if own_bot and bot is not None:
                await bot.session.close()
        finally:
            with repository.connection() as db:
                db.execute(
                    "DELETE FROM runtime_settings WHERE key=? AND value=?", (lock_key, lock_value)
                )
