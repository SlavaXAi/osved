from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import UTC, datetime, timedelta

from app.reposts import media, provider, store

logger = logging.getLogger(__name__)


def bind_card(repository, row, post, translation):
    body = media.body_html(post, translation, row["source_url"])
    now = store.stamp()
    with repository.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        db.execute(
            "UPDATE events SET headline=?,updated_at=? WHERE id=?",
            (
                translation.get("thesis") or post["text"][:180] or "Публикация X",
                now,
                row["event_id"],
            ),
        )
        draft = json.dumps({"card_kind": "x_repost", "repost_id": row["id"]})
        db.execute(
            "INSERT OR IGNORE INTO cards(event_id,card_version,prompt_version,"
            "structured_output_json,"
            "rendered_html,source_urls_json,primary_preview_url,validation_status) "
            "VALUES(?,1,'x_repost_v1',?,?,?,?,'adapted')",
            (row["event_id"], draft, body, json.dumps([row["source_url"]]), row["source_url"]),
        )
        card_id = db.execute(
            "SELECT id FROM cards WHERE event_id=? AND card_version=1", (row["event_id"],)
        ).fetchone()[0]
        db.execute(
            "UPDATE cards SET rendered_html=?,validation_status='adapted' WHERE id=?",
            (body, card_id),
        )
        db.execute(
            "INSERT OR IGNORE INTO deliveries(card_id,chat_id,telegram_message_id,"
            "delivery_status,delivered_at) VALUES(?,?,?,'sent',?)",
            (card_id, row["chat_id"], row["panel_message_id"], now),
        )
        delivery_id = db.execute(
            "SELECT id FROM deliveries WHERE card_id=? AND chat_id=?", (card_id, row["chat_id"])
        ).fetchone()[0]
        db.execute("UPDATE x_reposts SET delivery_id=? WHERE id=?", (delivery_id, row["id"]))
    # Preserve the actual post as a read source, explicitly only an author's claim.
    if post["text"]:
        doc_id = repository.upsert_research_document(
            canonical_url=row["source_url"],
            url=row["source_url"],
            source_class="radar_only",
            retrieved_at=now,
            expires_at=(datetime.now(UTC) + timedelta(days=365)).isoformat(),
            title="Пост @" + post["author"],
            publisher="X",
            published_at=post.get("published_at"),
            content_text=post["text"],
            content_hash=hashlib.sha256(post["text"].encode()).hexdigest(),
            metadata={
                "content_kind": "x_post_fulltext",
                "x_repost_id": row["id"],
                "epistemic_role": "author_claim_not_independent_verification",
            },
        )
        repository.attach_research_document(
            event_id=row["event_id"],
            research_document_id=doc_id,
            evidence_role="unverified_post",
            relevance=1,
        )
    return delivery_id


class RepostWorker:
    def __init__(self, bot, settings, repository):
        self.bot, self.settings, self.repository = bot, settings, repository

    async def process(self, row):
        from app.telegram_bot.news_panel import ensure_news_panel

        with self.repository.connection() as db:
            if not db.execute(
                "UPDATE x_reposts SET status='running',error=NULL,updated_at=? "
                "WHERE id=? AND status='queued'",
                (store.stamp(), row["id"]),
            ).rowcount:
                return False
        try:
            post, translation = await asyncio.to_thread(
                provider.prepare, self.settings, self.repository, row
            )
            row = store.get(self.repository, row["id"])
            warning = None
            try:
                files = await asyncio.to_thread(media.acquire, self.settings, row, post)
            except Exception as exc:
                files = []
                warning = (
                    str(exc) if isinstance(exc, ValueError) else "Источник пока не отдал медиафайл"
                )
                logger.warning(
                    "X media unavailable for repost %s: %s", row["id"], type(exc).__name__
                )
            with self.repository.connection() as db:
                db.execute(
                    "UPDATE x_reposts SET media_json=?,error=? WHERE id=?",
                    (json.dumps(files), warning, row["id"]),
                )
            delivery_id = bind_card(self.repository, row, post, translation)
            from app.reposts.native import publish, refresh_baseline

            panel = refresh_baseline(
                self.repository,
                ensure_news_panel(self.repository, delivery_id),
                media.body_html(post, translation, row["source_url"]),
            )
            files = await publish(self.bot, self.repository, row, panel, files)
            with self.repository.connection() as db:
                db.execute(
                    "UPDATE x_reposts SET status='ready',media_json=?,updated_at=? WHERE id=?",
                    (json.dumps(files), store.stamp(), row["id"]),
                )
            return True
        except Exception as exc:
            logger.exception("X repost preparation failed: %s", row["id"])
            with self.repository.connection() as db:
                db.execute(
                    "UPDATE x_reposts SET status='failed',error=?,updated_at=? WHERE id=?",
                    (type(exc).__name__ + ": " + str(exc)[:300], store.stamp(), row["id"]),
                )
            from app.details.provider_guard import DetailProviderError
            from app.telegram_bot.x_reposts import error_keyboard

            error_text = (
                "Достигнут лимит расходов. Пост сохранён; после изменения лимита "
                "обработку можно продолжить этой же кнопкой."
                if isinstance(exc, DetailProviderError) and exc.status == "budget_blocked"
                else "Публикация пока не готова. "
                "Полученные данные и оплаченные ответы сохранены. "
                "Повтор открывает сохранённый результат; неподтверждённые платные запросы "
                "автоматически не повторяются."
            )

            try:
                fresh = store.get(self.repository, row["id"])
                if fresh.get("native_kind"):
                    # A failed companion must not erase the delivered translation.
                    await self.bot.edit_message_reply_markup(
                        chat_id=row["chat_id"],
                        message_id=row["panel_message_id"],
                        reply_markup=error_keyboard(row),
                    )
                else:
                    await self.bot.edit_message_text(
                        chat_id=row["chat_id"],
                        message_id=row["panel_message_id"],
                        text=error_text,
                        reply_markup=error_keyboard(row),
                    )
            except Exception:
                logger.warning("Could not update X repost error panel %s", row["id"])
            return False

    async def run_loop(self):
        with self.repository.connection() as db:
            db.execute("UPDATE x_reposts SET status='queued' WHERE status='running'")
        while True:
            try:
                with self.repository.connection() as db:
                    row = db.execute(
                        "SELECT * FROM x_reposts WHERE status='queued' "
                        "AND panel_message_id IS NOT NULL ORDER BY id LIMIT 1"
                    ).fetchone()
                if row:
                    await self.process(dict(row))
                await asyncio.to_thread(media.cleanup, self.settings, self.repository)
            except Exception:
                logger.exception("X repost worker tick failed")
            await asyncio.sleep(2)
