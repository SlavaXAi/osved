"""Seven explicit editions; each selected item keeps its ordinary reader controls."""

from __future__ import annotations

import asyncio
import html
import json
import logging
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from aiogram.filters import Command
from aiogram.filters.callback_data import CallbackData
from aiogram.types import LinkPreviewOptions
from aiogram.utils.keyboard import InlineKeyboardBuilder

from app.collectors.ingest import redact_text
from app.config import DeliveryFormat
from app.menu_batches.store import KINDS, BatchStore, stamp

logger = logging.getLogger(__name__)
MOSCOW = ZoneInfo("Europe/Moscow")


class BatchAction(CallbackData, prefix="mb"):
    action: str
    value: str


def menu_keyboard():
    b = InlineKeyboardBuilder()
    for kind, (title, _, _) in KINDS.items():
        b.button(text=title, callback_data=BatchAction(action="issue", value=kind))
    b.button(text="Управление", callback_data=BatchAction(action="settings", value="0"))
    b.adjust(1)
    return b.as_markup()


SETTINGS_TEXT = (
    "<b>Управление ОСВЕД</b>\n\n"
    "/sources — источники\n/tgchannels — Telegram-каналы\n/channels — YouTube-каналы\n"
    "/radar — состояние и расходы\n/preferences — интересы и обратная связь\n"
    "/bio status — статус биографий; /bio pause и /bio resume — автоматическая выдача\n"
    "/week — недельная карта\n/people week — прежний недельный обзор людей\n"
    "/download — скачать YouTube или X (до 49 МБ)\n"
    "/help — помощь\n\n/menu — выпуски"
)


def batch_text(store, batch):
    items = store.items(batch["id"])
    sent = sum(i["status"] == "sent" for i in items)
    uncertain = sum(i["status"] == "uncertain" for i in items)
    skipped = sum(i["status"] == "skipped" for i in items)
    period = (
        "Готовая биография из запаса"
        if batch["kind"] == "bio"
        else (
            f"{datetime.fromisoformat(batch['period_start']).astimezone(MOSCOW):%d.%m %H:%M}–"
            f"{datetime.fromisoformat(batch['period_end']).astimezone(MOSCOW):%d.%m %H:%M} МСК"
        )
    )
    text = f"<b>{KINDS[batch['kind']][0]}</b>\n<i>{period}</i>\n\n"
    if not items:
        text += "Новых готовых материалов сейчас нет. Уже присланное повторно не рассылаю."
    elif sent == len(items):
        text += f"Выпуск готов: {sent}. Каждый материал — отдельным сообщением."
    else:
        text += (
            f"Отправлено: {sent} из {len(items)}. "
            "Оценивать каждую карточку для продолжения не нужно."
        )
    if uncertain:
        text += (
            "\nПодтверждение части отправок потеряно. Автоматический повтор остановлен, "
            "чтобы не прислать дубли."
        )
    if skipped:
        text += f"\nПропущено: {skipped}. Источник выключен или карточка больше недоступна."
    with store.repository.connection() as db:
        prep = db.execute(
            "SELECT * FROM menu_preparations WHERE batch_id=?", (batch["id"],)
        ).fetchone()
    if prep:
        if prep["status"] in {"queued", "running"}:
            text += f"\n\nГотовлю новые материалы. Бюджет подготовки: ${prep['limit_usd']:.2f}."
        elif prep["status"] == "offered":
            text += (
                f"\n\nПодготовить новые по свежим сигналам радара? Бюджет: "
                f"${prep['limit_usd']:.2f}. Поиск и генерация могут быть платными. "
                "Сводки и PDF видео заказываются отдельно."
            )
        else:
            label = {
                "complete": "Подготовка завершена",
                "budget_blocked": "Достигнут лимит подготовки",
                "uncertain": "Подготовка прервана; повтор платных шагов остановлен",
                "failed": "Подготовка не завершилась",
            }.get(prep["status"], "Подготовка остановлена")
            text += f"\n\n{label}. Учтено с резервами: ${prep['cost_usd']:.4f}."
            if not items:
                text += " Подходящих готовых материалов пока нет."
    return text


def batch_keyboard(store, batch):
    b = InlineKeyboardBuilder()
    with store.repository.connection() as db:
        prep = db.execute(
            "SELECT * FROM menu_preparations WHERE batch_id=?", (batch["id"],)
        ).fetchone()
    if prep and prep["status"] == "offered":
        b.button(
            text=f"Подготовить · бюджет ${prep['limit_usd']:.2f}",
            callback_data=BatchAction(action="prepare", value=str(batch["id"])),
        )
    elif (
        not prep
        and batch["status"] in {"empty", "sent"}
        and len(store.items(batch["id"])) < KINDS[batch["kind"]][2]
    ):
        b.button(
            text="Подготовить новые",
            callback_data=BatchAction(action="offer", value=str(batch["id"])),
        )
    b.button(
        text="Обновить статус", callback_data=BatchAction(action="status", value=str(batch["id"]))
    )
    b.button(text="Все выпуски", callback_data=BatchAction(action="menu", value="0"))
    b.adjust(1)
    return b.as_markup()


async def refresh_status(bot, store, batch_id):
    batch = store.get(batch_id)
    if not batch or not batch["status_message_id"]:
        return False
    try:
        await bot.edit_message_text(
            chat_id=batch["owner_id"],
            message_id=batch["status_message_id"],
            text=batch_text(store, batch),
            parse_mode="HTML",
            reply_markup=batch_keyboard(store, batch),
            link_preview_options=LinkPreviewOptions(is_disabled=True),
        )
        return True
    except Exception as exc:
        return "message is not modified" in str(exc).lower()


def register_batch_handlers(router, settings, repository):
    if not settings.menu_batches_enabled:
        return
    from app.telegram_bot.bot import is_allowed

    store = BatchStore(settings, repository)

    def private(message, owner):
        return is_allowed(settings, repository, owner) and message.chat.id == owner

    async def issue(message, owner, kind, key):
        batch_id, created = store.request(owner, kind, key)
        batch = store.get(batch_id)
        if (
            not created
            and batch["status_message_id"]
            and await refresh_status(message.bot, store, batch_id)
        ):
            return
        with repository.connection() as db:
            claimed = db.execute(
                "UPDATE menu_batches SET header_status='sending' WHERE id=? "
                "AND header_status!='sending'",
                (batch_id,),
            ).rowcount
        if not claimed:
            return
        try:
            response = await message.answer(
                batch_text(store, batch),
                parse_mode="HTML",
                reply_markup=batch_keyboard(store, batch),
            )
        except Exception:
            with repository.connection() as db:
                db.execute(
                    "UPDATE menu_batches SET header_status='uncertain' WHERE id=?", (batch_id,)
                )
            raise
        with repository.connection() as db:
            db.execute(
                "UPDATE menu_batches SET status_message_id=?,header_status='sent' WHERE id=?",
                (response.message_id, batch_id),
            )

    @router.message(Command("menu", "settings"))
    async def batch_menu_handler(message):
        owner = message.from_user.id if message.from_user else None
        if not private(message, owner):
            return
        if (message.text or "").split()[0].split("@")[0] == "/settings":
            await message.answer(SETTINGS_TEXT)
        else:
            await message.answer(
                "<b>Выпуски ОСВЕД</b>\n\nВыбери раздел. "
                "Готовые новые материалы придут отдельными сообщениями.",
                reply_markup=menu_keyboard(),
            )

    def plain_command(message):
        return len((message.text or "").split()) == 1

    @router.message(Command(*KINDS, "videos"), plain_command)
    async def batch_command_handler(message):
        owner = message.from_user.id if message.from_user else None
        if not private(message, owner):
            return
        kind = (message.text or "").split()[0].split("@")[0].lstrip("/")
        await issue(
            message,
            owner,
            "youtube" if kind == "videos" else kind,
            f"message:{owner}:{message.message_id}",
        )

    @router.callback_query(BatchAction.filter())
    async def batch_action_handler(query, callback_data: BatchAction):
        owner = query.from_user.id
        if not query.message or not private(query.message, owner):
            await query.answer("Недоступно", show_alert=True)
            return
        action, value = callback_data.action, callback_data.value
        if action in {"menu", "settings"}:
            await query.answer()
            # The edition's own status message stays bound to its outbox.
            await query.message.answer(
                SETTINGS_TEXT if action == "settings" else "<b>Выпуски ОСВЕД</b>",
                reply_markup=None if action == "settings" else menu_keyboard(),
            )
            return
        if action == "issue" and value in KINDS:
            await query.answer("Собираю готовые материалы")
            await issue(query.message, owner, value, "callback:" + query.id)
            return
        batch = store.get(int(value)) if value.isdigit() and len(value) < 19 else None
        if (
            not batch
            or batch["owner_id"] != owner
            or batch["status_message_id"] != query.message.message_id
        ):
            await query.answer("Этот выпуск недоступен", show_alert=True)
            return
        if action == "offer":
            ceiling = (
                settings.biography_max_cost_usd
                if batch["kind"] == "bio"
                else settings.menu_preparation_max_cost_usd
            )
            with repository.connection() as db:
                db.execute(
                    "INSERT OR IGNORE INTO menu_preparations(batch_id,owner_id,limit_usd,"
                    "created_at,updated_at) VALUES(?,?,?,?,?)",
                    (batch["id"], owner, ceiling, stamp(), stamp()),
                )
            await query.answer("Перед запуском подтвердите бюджет")
        elif action == "prepare":
            if datetime.fromisoformat(batch["period_end"]) < datetime.now(UTC) - timedelta(
                minutes=30
            ):
                await query.answer(
                    "Выпуск устарел. Открой команду раздела заново.", show_alert=True
                )
                return
            reason = None
            with repository.connection() as db:
                db.execute("BEGIN IMMEDIATE")
                prep = db.execute(
                    "SELECT * FROM menu_preparations WHERE batch_id=?", (batch["id"],)
                ).fetchone()
                busy = db.execute(
                    "SELECT 1 FROM menu_preparations WHERE owner_id=? "
                    "AND status IN('queued','running')",
                    (owner,),
                ).fetchone()
                total = db.execute(
                    "SELECT COALESCE(SUM(cost_usd),0) FROM usage_ledger WHERE "
                    "strftime('%Y-%m',created_at)=strftime('%Y-%m','now')"
                ).fetchone()[0]
                if not prep or prep["status"] != "offered" or busy:
                    reason = "Подготовка уже запущена или завершена"
                elif total + prep["limit_usd"] >= settings.budget_hard_stop_usd:
                    reason = "Подготовка остановлена общим лимитом расходов"
                else:
                    db.execute(
                        "UPDATE menu_preparations SET status='queued',updated_at=? WHERE id=?",
                        (stamp(), prep["id"]),
                    )
            await query.answer(reason or "Подготовка принята", show_alert=bool(reason))
        else:
            await query.answer()
        await refresh_status(query.bot, store, batch["id"])


class BatchWorker:
    def __init__(self, bot, settings, repository):
        self.bot, self.settings, self.repository = bot, settings, repository
        self.store = BatchStore(settings, repository)

    async def send_item(self, item):
        from app.telegram_bot.bot import send_formatted_card
        from app.telegram_bot.sensory import video_keyboard

        payload = json.loads(item["payload_json"])
        if item["item_kind"] in {"news", "video"}:
            with self.repository.connection() as db:
                if item["item_kind"] == "news":
                    source = db.execute(
                        "SELECT s.id,s.enabled,c.validation_status FROM cards c "
                        "JOIN events e ON e.id=c.event_id JOIN source_items i ON "
                        "i.id=e.primary_source_item_id JOIN sources s ON s.id=i.source_id "
                        "WHERE c.id=?",
                        (item["entity_id"],),
                    ).fetchone()
                else:
                    source = db.execute(
                        "SELECT s.id,s.enabled,'valid' validation_status "
                        "FROM source_items i JOIN sources s ON s.id=i.source_id "
                        "WHERE i.id=?",
                        (item["entity_id"],),
                    ).fetchone()
            override = (
                self.repository.get_runtime_setting("source_enabled:" + source["id"])
                if source
                else None
            )
            if (
                not source
                or source["validation_status"] != "valid"
                or override == "false"
                or (override is None and not source["enabled"])
            ):
                raise ValueError("Source disabled or publication withdrawn before delivery")
        if item["item_kind"] == "news":
            return await send_formatted_card(
                self.bot,
                chat_id=item["owner_id"],
                card=payload,
                delivery_id=item["delivery_id"],
                delivery_format=DeliveryFormat.REGULAR,
                include_details=self.settings.details_enabled,
            )
        if item["item_kind"] == "video":
            from app.video.transcripts import youtube_video_id

            video_id = youtube_video_id(payload["video_id"])
            url = "https://www.youtube.com/watch?v=" + video_id
            description = payload["description_html"]
            if url not in description:
                description += (
                    '\n\n<a href="' + html.escape(url, quote=True) + '">Источник · YouTube</a>'
                )
            return await self.bot.send_message(
                chat_id=item["owner_id"],
                text=description,
                parse_mode="HTML",
                reply_markup=video_keyboard(
                    item["delivery_id"],
                    expanded=True,
                    url="https://www.youtube.com/watch?v=" + payload["video_id"],
                ),
                link_preview_options=LinkPreviewOptions(is_disabled=True),
            )
        from app.biography.storage import BiographyStore
        from app.biography.worker import send_delivery

        store = BiographyStore(self.repository)
        delivery = store.claim_delivery(
            item["owner_id"], f"menu:{item['batch_id']}", issue_id=item["entity_id"]
        )
        if not delivery:
            raise ValueError("Biography is already claimed or unavailable")
        result = await send_delivery(store, self.bot, delivery, current=datetime.now(UTC))
        if result["status"] != "sent":
            raise RuntimeError("Biography delivery " + result["status"])
        from types import SimpleNamespace

        return SimpleNamespace(message_id=result["message_id"])

    async def tick(self):
        item = self.store.claim_item()
        if item:
            try:
                response = await self.send_item(item)
                self.store.finish(item, "sent", message_id=response.message_id)
            except ValueError as exc:
                self.store.finish(item, "skipped", error=redact_text(str(exc))[:200])
            except Exception as exc:
                self.store.finish(item, "uncertain", error=redact_text(type(exc).__name__)[:200])
                logger.warning("Edition item %s has no confirmed delivery", item["id"])
            await refresh_status(self.bot, self.store, item["batch_id"])
            return True
        with self.repository.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute(
                "SELECT 1 FROM schedule_slots WHERE mode='live' "
                "AND status IN('planned','running') AND "
                "julianday(created_at)>julianday('now','-15 minutes')"
            ).fetchone():
                return False
            prep = db.execute(
                "SELECT * FROM menu_preparations WHERE status='queued' ORDER BY id LIMIT 1"
            ).fetchone()
            if not prep:
                return False
            db.execute(
                "UPDATE menu_preparations SET status='running',updated_at=? WHERE id=?",
                (stamp(), prep["id"]),
            )
        from app.menu_batches.preparation import prepare

        async def pulse():
            while True:
                await refresh_status(self.bot, self.store, prep["batch_id"])
                await asyncio.sleep(10)

        progress = asyncio.create_task(pulse())
        try:
            await asyncio.to_thread(prepare, self.settings, self.repository, dict(prep))
        finally:
            progress.cancel()
            with suppress(asyncio.CancelledError):
                await progress
        await refresh_status(self.bot, self.store, prep["batch_id"])
        return True

    async def run_loop(self):
        # A restart cannot authorize a second attempt at a paid, possibly completed call.
        from app.menu_batches.preparation import recover_preparations

        recover_preparations(self.repository)
        with self.repository.connection() as db:
            db.execute(
                "UPDATE menu_batches SET header_status='uncertain' WHERE header_status='sending'"
            )
        while True:
            try:
                active = await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Edition worker failed")
                active = False
            await asyncio.sleep(1.1 if active else 2)
