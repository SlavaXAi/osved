from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta

from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError, TelegramRetryAfter
from aiogram.types import InputRichMessage

from app.biography.engine import MOSCOW, BiographyEngine
from app.biography.models import render_issue
from app.biography.storage import BiographyStore
from app.collectors.ingest import redact_text
from app.config import Settings
from app.db.repository import Repository

logger = logging.getLogger(__name__)


async def send_delivery(store: BiographyStore, bot, delivery: dict, *, current: datetime) -> dict:
    from app.telegram_bot.biography import biography_keyboard

    detail = store.delivery(delivery["id"])
    markup = render_issue(detail["name"], detail["payload"])
    if not store.begin_send(delivery["id"], current=current):
        return {"status": "already_claimed"}
    try:
        message = await bot.send_rich_message(
            chat_id=detail["owner_id"],
            rich_message=InputRichMessage(html=markup),
            reply_markup=biography_keyboard(
                detail["id"], store.feedback(detail["id"], detail["owner_id"])
            ),
        )
    except (TelegramBadRequest, TelegramForbiddenError, TelegramRetryAfter) as exc:
        attempts = store.delivery(detail["id"])["attempts"]
        can_retry = attempts < 3 and not isinstance(exc, TelegramForbiddenError)
        retry_seconds = getattr(exc, "retry_after", 30 if attempts == 1 else 120)
        store.finish_delivery(
            detail["id"],
            "pending" if can_retry else "failed",
            error=redact_text(str(exc))[:500],
            retry_at=(current + timedelta(seconds=retry_seconds)).isoformat()
            if can_retry
            else None,
        )
        return {"status": "retry_wait" if can_retry else "failed"}
    except Exception as exc:
        store.finish_delivery(detail["id"], "unknown", error=redact_text(str(exc))[:500])
        logger.warning("Biography send %s uncertain; automatic repeat stopped", detail["id"])
        return {"status": "unknown"}
    store.finish_delivery(detail["id"], "sent", message_id=message.message_id)
    return {"status": "sent", "delivery_id": detail["id"], "message_id": message.message_id}


async def delivery_tick(settings: Settings, store: BiographyStore, bot, *, current=None) -> dict:
    from app.telegram_bot.bot import configured_owner_id

    current = (current or datetime.now(UTC)).astimezone(UTC)
    store.recover_stale(current)
    if not store.enabled(settings.biography_enabled):
        return {"status": "paused"}
    owner = configured_owner_id(settings, store.repository)
    if owner is None:
        return {"status": "owner_missing"}
    local = current.astimezone(MOSCOW)
    slot = "daily:" + local.date().isoformat()
    store.expire_older_pending(slot)
    if local.hour >= 12:
        store.expire_pending(slot)
        return {"status": "window_closed"}
    if local.hour < 9:
        return {"status": "waiting_09"}
    delivery = store.claim_delivery(owner, slot, current=current)
    if delivery is None:
        return {"status": "already_sent_or_no_ready_issue"}
    return await send_delivery(store, bot, delivery, current=current)


async def preparation_worker(settings: Settings) -> None:
    store = BiographyStore(Repository(settings.database_path))
    store.initialize()
    engine = BiographyEngine(settings, store)
    while True:
        try:
            if store.enabled(settings.biography_enabled):
                store.recover_stale(datetime.now(UTC))
                if store.status()["ready"] < settings.biography_reserve_size:
                    await asyncio.to_thread(engine.prepare_one)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Biography preparation tick: %s", redact_text(str(exc))[:500])
        await asyncio.sleep(60)


async def delivery_worker(settings: Settings) -> None:
    from app.telegram_bot.bot import build_bot

    store = BiographyStore(Repository(settings.database_path))
    bot = None
    try:
        while True:
            try:
                if store.enabled(settings.biography_enabled):
                    if bot is None:
                        bot = build_bot(settings)
                    await delivery_tick(settings, store, bot)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Biography delivery tick: %s", redact_text(str(exc))[:500])
            await asyncio.sleep(30)
    finally:
        if bot is not None:
            await bot.session.close()
