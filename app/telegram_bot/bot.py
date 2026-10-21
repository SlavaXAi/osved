from __future__ import annotations

import asyncio
import html
import logging
from contextlib import suppress
from typing import Any

from aiogram import Bot, Dispatcher, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.types import (
    BotCommand,
    CallbackQuery,
    ForceReply,
    InlineKeyboardMarkup,
    InputRichMessage,
    LinkPreviewOptions,
    Message,
    ReplyParameters,
)

from app.config import DeliveryFormat, Settings
from app.db.repository import Repository
from app.details.worker import DetailWorker
from app.llm.openrouter import CardDraft
from app.pipeline.cards import render_article_rich_html
from app.pipeline.telegram_style import clean_news_html
from app.telegram_bot.feedback import FeedbackAction, Reaction, reaction_from_action
from app.telegram_bot.keyboards import (
    DetailFeedbackAction,
    DetailFeedbackCallback,
    DetailsButtonState,
    DetailsCallback,
    FeedbackCallback,
    NewsDepthCallback,
    detail_feedback_keyboard,
    reaction_keyboard,
    superreaction_keyboard,
)
from app.telegram_bot.news_demo import NewsDemoCallback, handle_news_demo
from app.telegram_bot.news_depth import brief_card_html, depth_key
from app.telegram_bot.news_panel import (
    NewsPanelControl,
    get_news_panel,
    panel_keyboard,
    refresh_markup,
    show_error,
    show_result,
)
from app.telegram_bot.news_reader import (
    handle_news_action,
    record_news_result_feedback,
    terminal_text,
)
from app.telegram_bot.progress import DetailProgress, progress_text

logger = logging.getLogger(__name__)

DETAIL_PLACEHOLDER_HTML = progress_text("details", "queued")
DETAIL_TERMINAL_HTML = {
    "insufficient_evidence": (
        "<b>Подробный разбор не собран</b>\n\n"
        "Не хватило независимых полнотекстовых источников. "
        "Я не буду достраивать факты догадками."
    ),
    "budget_blocked": (
        "<b>Разбор остановлен лимитом расходов</b>\n\n"
        "Система не стала запускать следующий платный шаг сверх установленного лимита."
    ),
    "failed": (
        "<b>Не удалось собрать подробный разбор</b>\n\n"
        "Исходная новость и ваша реакция сохранены. "
        "Разбор можно запустить повторно кнопкой под новостью."
    ),
}


class TelegramBotNotConfigured(RuntimeError):
    pass


class CardsNotDeliverable(RuntimeError):
    def __init__(self, card_ids: list[int]) -> None:
        self.card_ids = card_ids
        super().__init__(
            "Cards are unknown, invalid, already delivered, or currently pending: "
            + ", ".join(str(card_id) for card_id in card_ids)
        )


class CandidateBatchNotDeliverable(RuntimeError):
    pass


OWNER_SETTING = "allowed_user_id"
DELIVERY_FORMAT_SETTING = "delivery_format"

FORMAT_LABELS = {
    DeliveryFormat.REGULAR: "обычные новости",
    DeliveryFormat.ARTICLE: "статьи Telegram",
}


def _details_enabled(settings: Settings) -> bool:
    return bool(getattr(settings, "details_enabled", True))


def _details_daily_limit(settings: Settings) -> int:
    return int(getattr(settings, "details_max_daily_new", 5))


def details_button_state(request: dict[str, Any] | None) -> DetailsButtonState:
    if request is None:
        return DetailsButtonState.REQUEST
    status = str(request.get("status") or "")
    if status in {"queued", "running"}:
        return DetailsButtonState.RUNNING
    if status in {"ready", "sent"}:
        return DetailsButtonState.READY
    if status == "failed":
        return DetailsButtonState.RETRY
    if status == "insufficient_evidence":
        return DetailsButtonState.INSUFFICIENT
    if status in {"budget_blocked", "daily_limit"}:
        return DetailsButtonState.BUDGET_BLOCKED
    return DetailsButtonState.REQUEST


def detail_display_status(request: dict[str, Any]) -> str:
    status = str(request.get("status") or "")
    return "pending" if status in {"queued", "running"} else status


def _delivery_supports_details(repository: Repository, delivery_id: int) -> bool:
    context = repository.get_delivery_context(delivery_id)
    if context is None:
        return False
    event = repository.event_context(int(context["event_id"]))
    return bool(event and event.get("event_type") != "system_test")


def delivery_keyboard(
    repository: Repository,
    delivery_id: int,
    *,
    user_id: int | None = None,
    current: Reaction | None = None,
    details_enabled: bool = True,
) -> InlineKeyboardMarkup:
    """Render reaction and details state together so neither can erase the other."""

    panel = get_news_panel(repository, delivery_id)
    if panel is not None:
        return panel_keyboard(repository, panel)
    include_details = details_enabled and _delivery_supports_details(repository, delivery_id)
    detail_request = (
        repository.detail_request_for_delivery(delivery_id) if include_details else None
    )
    detail_state = details_button_state(detail_request)
    saved_depth = repository.get_runtime_setting(depth_key(delivery_id))
    # Pre-upgrade long cards with an existing article request keep their controls.
    # An explicitly collapsed new card always stays collapsed.
    expanded = saved_depth == "expanded" or (saved_depth is None and detail_request is not None)
    session = repository.feedback_session_for_user(user_id) if user_id is not None else None
    if session is not None and int(session["delivery_id"]) == delivery_id:
        return superreaction_keyboard(
            delivery_id,
            include_details=include_details,
            details_state=detail_state,
            expanded=expanded,
        )
    if current is None:
        current_value = repository.current_reaction(delivery_id)
        current = Reaction(current_value) if current_value else None
    return reaction_keyboard(
        delivery_id,
        current,
        include_details=include_details,
        details_state=detail_state,
        expanded=expanded,
    )


async def _edit_delivery_markup(
    bot, repository, delivery_id, *, user_id=None, current=None, details_enabled=True, message=None
):
    if get_news_panel(repository, delivery_id) is not None:
        return await refresh_markup(bot, repository, delivery_id)
    markup = delivery_keyboard(
        repository, delivery_id, user_id=user_id, current=current, details_enabled=details_enabled
    )
    if message is not None:
        await message.edit_reply_markup(reply_markup=markup)
    else:
        context = repository.get_delivery_context(delivery_id)
        if not context or not context.get("telegram_message_id"):
            return False
        await bot.edit_message_reply_markup(
            chat_id=context["chat_id"],
            message_id=context["telegram_message_id"],
            reply_markup=markup,
        )
    return True


def details_acknowledgement(result: dict[str, Any]) -> tuple[str, bool]:
    """Return callback text and whether Telegram should display it as an alert."""

    status = str(result.get("status") or "")
    if status == "failed" and "explicit retry required" in str(result.get("last_error") or ""):
        return "Проверки обновились. Нажмите повторный разбор для новой проверки", True
    if status == "daily_limit":
        return "Лимит новых подробных разборов на сегодня исчерпан", True
    if status == "budget_blocked":
        return "Запуск остановлен лимитом расходов", True
    if status == "insufficient_evidence":
        return "Для честного подробного разбора пока недостаточно источников", True
    if status in {"ready", "sent"} or result.get("cache_hit"):
        return "Разбор уже готов", False
    if result.get("created"):
        return "Собираю подробный разбор", False
    if status in {"queued", "running"}:
        return "Уже собираю", False
    return "Запрос принят", False


def configured_owner_id(settings: Settings, repository: Repository) -> int | None:
    if settings.allowed_user_id is not None:
        return settings.allowed_user_id
    stored = repository.get_runtime_setting(OWNER_SETTING)
    return int(stored) if stored is not None else None


def is_allowed(settings: Settings, repository: Repository, user_id: int | None) -> bool:
    return user_id is not None and configured_owner_id(settings, repository) == user_id


def configured_delivery_format(settings: Settings, repository: Repository) -> DeliveryFormat:
    stored = repository.get_runtime_setting(DELIVERY_FORMAT_SETTING)
    if stored is None:
        return settings.delivery_format
    try:
        return DeliveryFormat(stored)
    except ValueError:
        return settings.delivery_format


def set_delivery_format(repository: Repository, delivery_format: DeliveryFormat) -> None:
    repository.set_runtime_setting(DELIVERY_FORMAT_SETTING, delivery_format.value)


def build_router(settings: Settings, repository: Repository) -> Router:
    router = Router(name="personal-newsfeed")
    from app.telegram_bot.biography import register_biography_handlers
    from app.telegram_bot.channels import register_channel_handlers
    from app.telegram_bot.intelligence import register_intelligence_handlers
    from app.telegram_bot.media_downloads import register_media_handlers
    from app.telegram_bot.menu_batches import register_batch_handlers
    from app.telegram_bot.sensory import register_sensory_handlers
    from app.telegram_bot.telegram_channels import register_telegram_channel_handlers
    from app.telegram_bot.video_notes import register_video_note_handlers
    from app.telegram_bot.video_reports import register_video_report_handlers

    register_batch_handlers(router, settings, repository)
    register_video_note_handlers(router, settings, repository)
    register_media_handlers(router, settings, repository)
    register_biography_handlers(router, settings, repository)
    register_intelligence_handlers(router, settings, repository)
    register_sensory_handlers(router, settings, repository)
    register_telegram_channel_handlers(router, settings, repository)
    register_channel_handlers(router, settings, repository)
    register_video_report_handlers(router, settings, repository)

    @router.message(Command("start"))
    async def start_handler(message: Message) -> None:
        user_id = message.from_user.id if message.from_user else None
        if user_id is None:
            return
        owner_id = configured_owner_id(settings, repository)
        if owner_id is None:
            owner_id = int(repository.claim_runtime_setting(OWNER_SETTING, str(user_id)))
        if owner_id != user_id:
            return
        if settings.allowed_user_id is None:
            await message.answer(
                "Готово. Вы сохранены как владелец этой персональной ленты.\n"
                f"Telegram user ID: <code>{user_id}</code>"
            )
            return
        if not is_allowed(settings, repository, user_id):
            return
        await message.answer("Персональная технологическая лента готова принимать feedback.")

    @router.message(Command("status"))
    async def status_handler(message: Message) -> None:
        user_id = message.from_user.id if message.from_user else None
        if not is_allowed(settings, repository, user_id):
            return
        status = repository.status_counts()
        delivery_format = configured_delivery_format(settings, repository)
        from app.ops.feed_health import feed_health, feed_health_text

        await message.answer(
            "<b>Статус ленты</b>\n\n" + feed_health_text(feed_health(repository)) + "\n\n"
            f"Источников: {status['enabled_sources']}\n"
            f"Кандидатных источников: {status['candidate_sources']}\n"
            f"Материалов: {repository.source_item_count()}\n"
            f"Формат: {FORMAT_LABELS[delivery_format]}\n"
            f"Учтено за месяц (включая резервы): ${status['month_cost_usd']:.4f}"
        )

    @router.message(Command("format"))
    async def format_handler(message: Message) -> None:
        user_id = message.from_user.id if message.from_user else None
        if not is_allowed(settings, repository, user_id):
            return
        parts = (message.text or "").split(maxsplit=1)
        if len(parts) == 1:
            await message.answer(
                "Сейчас включены <b>обычные новости</b>. "
                "Нативные статьи сохранены как эксперимент, но временно отложены."
            )
            return
        try:
            selected = DeliveryFormat(parts[1].strip().casefold())
        except ValueError:
            await message.answer(
                "Неизвестный формат. Используйте <code>regular</code> или <code>article</code>."
            )
            return
        if selected == DeliveryFormat.ARTICLE:
            await message.answer(
                "Режим статей сейчас отложен. Рабочий формат — <code>regular</code>."
            )
            return
        set_delivery_format(repository, selected)
        await message.answer(
            f"Готово. Следующие публикации придут в формате: <b>{FORMAT_LABELS[selected]}</b>."
        )

    async def advance_news(bot: Bot, delivery_id: int) -> None:
        from app.telegram_bot.news_queue import send_next_news

        try:
            result = await send_next_news(
                settings,
                repository,
                bot=bot,
                after_delivery_id=delivery_id,
                limit=settings.news_extra_batch_size,
            )
            if result.get("errors"):
                logger.warning("News queue needs attention: %s", result.get("status"))
        except Exception:
            # Feedback is already durable. A delivery failure must not undo it.
            logger.exception("Could not advance news queue after feedback")

    @router.message(Command("news"))
    async def news_handler(message: Message) -> None:
        from app.telegram_bot.news_queue import send_next_news

        user_id = message.from_user.id if message.from_user else None
        if not is_allowed(settings, repository, user_id):
            return
        result = await send_next_news(
            settings,
            repository,
            bot=message.bot,
            retry=(message.text or "").split()[1:] == ["retry"],
            limit=settings.news_extra_batch_size,
        )
        explanations = {
            "waiting_for_rating": (
                "Сначала оцени предыдущую новость — затем придёт следующая. "
                "«Подробнее» доступно и после оценки."
            ),
            "queue_empty": (
                "Готовые новости закончились. Радар пополнит очередь при следующем сборе; "
                "повторы присылать не буду."
            ),
            "sending": "Новость уже отправляется.",
            "delivery_confirmation_needed": (
                "Не удалось подтвердить доставку. Если новость не пришла, отправь /news retry. "
                "Если она уже пришла, просто оцени её — повтор может создать дубль."
            ),
            "queue_disabled": "Последовательная лента пока выключена.",
        }
        if result["status"] != "sent":
            await message.answer(
                explanations.get(result["status"], "Очередь требует проверки. Реакции сохранены.")
            )

    @router.message(Command("cancel"))
    async def cancel_handler(message: Message) -> None:
        user_id = message.from_user.id if message.from_user else None
        if not is_allowed(settings, repository, user_id) or user_id is None:
            return
        session = repository.feedback_session_for_user(user_id)
        if session is None:
            await message.answer("Нет ожидающего комментария.")
            return
        repository.clear_feedback_session(session["delivery_id"])
        context = repository.get_delivery_context(session["delivery_id"])
        if context and context["telegram_message_id"]:
            await _edit_delivery_markup(
                message.bot,
                repository,
                session["delivery_id"],
                user_id=user_id,
                details_enabled=_details_enabled(settings),
            )
        await message.answer("Комментарий отменён.")

    @router.callback_query(FeedbackCallback.filter())
    async def feedback_callback_handler(
        query: CallbackQuery,
        callback_data: FeedbackCallback,
    ) -> None:
        user_id = query.from_user.id
        if not is_allowed(settings, repository, user_id):
            await query.answer("Недоступно", show_alert=True)
            return
        delivery_id = callback_data.delivery_id
        context = repository.get_delivery_context(delivery_id)
        if context is None:
            await query.answer("Карточка не найдена", show_alert=True)
            return

        action = callback_data.action
        reaction = reaction_from_action(action)
        if reaction and not reaction.is_super:
            repository.finalize_feedback(
                delivery_id,
                action="reaction",
                reaction=reaction.value,
            )
            await query.answer("Сохранено")
            try:
                if query.message:
                    await _edit_delivery_markup(
                        query.bot,
                        repository,
                        delivery_id,
                        message=query.message,
                        user_id=user_id,
                        current=reaction,
                        details_enabled=_details_enabled(settings),
                    )
            finally:
                await advance_news(query.bot, delivery_id)
            return

        if reaction and reaction.is_super:
            repository.begin_feedback_session(delivery_id, user_id, reaction.value)
            if query.message:
                await _edit_delivery_markup(
                    query.bot,
                    repository,
                    delivery_id,
                    message=query.message,
                    user_id=user_id,
                    details_enabled=_details_enabled(settings),
                )
            await query.answer()
            return

        session = repository.feedback_session_for_user(user_id)
        if session is None or session["delivery_id"] != delivery_id:
            await query.answer("Суперреакция уже не ожидает решения", show_alert=True)
            return

        if action == FeedbackAction.BACK:
            repository.clear_feedback_session(delivery_id)
            if query.message:
                await _edit_delivery_markup(
                    query.bot,
                    repository,
                    delivery_id,
                    message=query.message,
                    user_id=user_id,
                    details_enabled=_details_enabled(settings),
                )
            await query.answer("Отменено")
            return

        if action == FeedbackAction.SILENT:
            reaction = Reaction(session["reaction"])
            repository.finalize_feedback(
                delivery_id,
                action="superreaction_silent",
                reaction=reaction.value,
            )
            await query.answer("Суперреакция сохранена")
            try:
                if query.message:
                    await _edit_delivery_markup(
                        query.bot,
                        repository,
                        delivery_id,
                        message=query.message,
                        user_id=user_id,
                        current=reaction,
                        details_enabled=_details_enabled(settings),
                    )
            finally:
                await advance_news(query.bot, delivery_id)
            return

        if action == FeedbackAction.COMMENT:
            if query.message is None:
                await query.answer("Невозможно запросить комментарий", show_alert=True)
                return
            prompt = await query.message.reply(
                "Коротко объясните, что именно здесь настолько хорошо или плохо. "
                "Можно отменить командой /cancel.",
                reply_markup=ForceReply(selective=True),
            )
            repository.set_feedback_session_state(
                delivery_id,
                "awaiting_comment",
                prompt_message_id=prompt.message_id,
            )
            await query.answer()

    @router.callback_query(NewsDemoCallback.filter())
    async def news_demo_handler(query: CallbackQuery, callback_data: NewsDemoCallback) -> None:
        await handle_news_demo(
            query,
            callback_data,
            repository,
            allowed=is_allowed(settings, repository, query.from_user.id),
        )

    async def dispatch_news_action(query, delivery_id, action, *, page=0, revision=-1):
        if not is_allowed(settings, repository, query.from_user.id):
            await query.answer("Недоступно", show_alert=True)
            return
        try:
            await handle_news_action(
                query, repository, settings, delivery_id, action, page=page, revision=revision
            )
        except (ValueError, PermissionError) as exc:
            await query.answer(str(exc), show_alert=True)
        except TelegramBadRequest as exc:
            if "message is not modified" not in str(exc).lower():
                logger.exception("Could not edit news panel %s", delivery_id)
                await query.answer(
                    "Не удалось обновить карточку. Попробуй открыть её снова.", show_alert=True
                )

    @router.callback_query(NewsPanelControl.filter())
    async def news_panel_handler(query: CallbackQuery, callback_data: NewsPanelControl) -> None:
        await dispatch_news_action(
            query,
            callback_data.delivery_id,
            callback_data.action,
            page=callback_data.page,
            revision=callback_data.revision,
        )

    @router.callback_query(NewsDepthCallback.filter())
    async def news_depth_handler(query: CallbackQuery, callback_data: NewsDepthCallback) -> None:
        await dispatch_news_action(
            query, callback_data.delivery_id, "expanded" if callback_data.expanded else "base"
        )

    @router.callback_query(DetailsCallback.filter())
    async def details_callback_handler(
        query: CallbackQuery, callback_data: DetailsCallback
    ) -> None:
        await dispatch_news_action(query, callback_data.delivery_id, "article")

    @router.callback_query(DetailFeedbackCallback.filter())
    async def detail_feedback_callback_handler(
        query: CallbackQuery,
        callback_data: DetailFeedbackCallback,
    ) -> None:
        user_id = query.from_user.id
        if not is_allowed(settings, repository, user_id):
            await query.answer("Недоступно", show_alert=True)
            return
        panel = get_news_panel(repository, callback_data.delivery_id)
        if panel is not None:
            if (
                not query.message
                or panel["chat_id"] != user_id
                or panel["chat_id"] != query.message.chat.id
                or panel["message_id"] != query.message.message_id
            ):
                await query.answer("Кнопка не относится к этому результату", show_alert=True)
                return
            request = repository.get_detail_request(panel.get("displayed_request_id") or -1)
            if not request or request.get("article_id") != callback_data.article_id:
                await query.answer("Сначала открой этот результат", show_alert=True)
                return
            try:
                record_news_result_feedback(repository, panel, request, callback_data.action.value)
            except PermissionError as exc:
                await query.answer(str(exc), show_alert=True)
                return
            await query.answer("Оценка полезности сохранена")
            await refresh_markup(query.bot, repository, callback_data.delivery_id)
            return
        action = (
            "detail_useful"
            if callback_data.action == DetailFeedbackAction.USEFUL
            else "detail_not_worth_it"
        )
        try:
            repository.record_detail_feedback(
                callback_data.delivery_id,
                callback_data.article_id,
                action=action,
            )
        except ValueError as exc:
            await query.answer(str(exc), show_alert=True)
            return
        await query.answer("Оценка разбора сохранена")
        if query.message:
            await query.message.edit_reply_markup(
                reply_markup=detail_feedback_keyboard(
                    callback_data.delivery_id,
                    callback_data.article_id,
                    current=callback_data.action,
                )
            )

    @router.message()
    async def comment_handler(message: Message) -> None:
        user_id = message.from_user.id if message.from_user else None
        if not is_allowed(settings, repository, user_id) or user_id is None or not message.text:
            return
        session = repository.feedback_session_for_user(user_id)
        if session is None or session["state"] != "awaiting_comment":
            return
        reaction = Reaction(session["reaction"])
        repository.finalize_feedback(
            session["delivery_id"],
            action="superreaction_comment",
            reaction=reaction.value,
            comment=message.text.strip(),
        )
        context = repository.get_delivery_context(session["delivery_id"])
        try:
            if context and context["telegram_message_id"]:
                await _edit_delivery_markup(
                    message.bot,
                    repository,
                    session["delivery_id"],
                    user_id=user_id,
                    current=reaction,
                    details_enabled=_details_enabled(settings),
                )
            await message.answer("Комментарий и суперреакция сохранены.")
        finally:
            await advance_news(message.bot, session["delivery_id"])

    return router


def build_bot(settings: Settings) -> Bot:
    if settings.telegram_bot_token is None:
        raise TelegramBotNotConfigured("TELEGRAM_BOT_TOKEN is required")
    from aiogram.client.session.aiohttp import AiohttpSession
    from aiogram.client.telegram import TelegramAPIServer

    session = None
    if settings.telegram_api_base_url:
        session = AiohttpSession(
            api=TelegramAPIServer.from_base(settings.telegram_api_base_url, is_local=True)
        )
    return Bot(
        session=session,
        token=settings.telegram_bot_token.get_secret_value(),
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )


async def run_bot(settings: Settings, repository: Repository) -> None:
    bot = build_bot(settings)
    try:
        commands = [
            BotCommand(command="help", description="Как пользоваться PARS"),
            BotCommand(command="bio", description="Ежедневные биографии: статус и пауза"),
            BotCommand(command="news", description="Получить ещё свежие новости"),
            BotCommand(command="radar", description="Состояние радара и расходы"),
            BotCommand(command="sources", description="Включить или выключить источники"),
            BotCommand(command="tgchannels", description="Мои Telegram-каналы для радара"),
            BotCommand(command="videos", description="Рекомендации подкастов и видео"),
            BotCommand(command="download", description="Скачать YouTube или X в Telegram"),
            BotCommand(command="circle", description="Превратить видео в кружочек"),
            BotCommand(command="channels", description="Добавить YouTube-каналы и управлять ими"),
            BotCommand(command="report", description="PDF-резюме по ссылке YouTube"),
            BotCommand(command="week", description="Недельная карта проверенных событий"),
            BotCommand(command="people", description="Люди за неделю по сохранённым событиям"),
            BotCommand(command="preferences", description="Предложения по моим реакциям"),
        ]
        if settings.menu_batches_enabled:
            from app.menu_batches.store import KINDS

            commands = [
                BotCommand(command=kind, description=title) for kind, (title, _, _) in KINDS.items()
            ] + [
                BotCommand(command="menu", description="Все выпуски: выбрать кнопкой"),
                BotCommand(command="settings", description="Управление источниками и лентой"),
                *[c for c in commands if c.command not in KINDS],
            ]
        await bot.set_my_commands(commands)
    except Exception:
        logger.warning("Could not update Telegram command menu; polling will still start")
    dispatcher = Dispatcher()
    dispatcher.include_router(build_router(settings, repository))
    details_task: asyncio.Task[None] | None = None
    details_worker_task: asyncio.Task[None] | None = None
    details_worker_stop: asyncio.Event | None = None
    media_task: asyncio.Task[None] | None = None
    batch_task: asyncio.Task[None] | None = None
    if settings.menu_batches_enabled:
        from app.telegram_bot.menu_batches import BatchWorker

        batch_task = asyncio.create_task(
            BatchWorker(bot, settings, repository).run_loop(), name="menu-batch-worker"
        )
    if settings.media_downloads_enabled:
        from app.telegram_bot.media_downloads import MediaWorker

        media_task = asyncio.create_task(
            MediaWorker(bot, settings, repository).run_loop(), name="media-download-worker"
        )
    video_report_task: asyncio.Task[None] | None = None
    if settings.video_reports_enabled:
        from app.telegram_bot.video_reports import VideoReportWorker

        video_report_task = asyncio.create_task(
            VideoReportWorker(bot, settings, repository).run_loop(), name="video-pdf-worker"
        )
    if _details_enabled(settings):
        details_task = asyncio.create_task(
            run_detail_delivery_loop(bot, repository),
            name="details-telegram-delivery",
        )
        details_worker_stop = asyncio.Event()
        details_worker_task = asyncio.create_task(
            DetailWorker(settings, repository).run_loop_async(
                stop_event=details_worker_stop,
            ),
            name="details-research-worker",
        )
    try:
        await dispatcher.start_polling(bot, allowed_updates=dispatcher.resolve_used_update_types())
    finally:
        if batch_task is not None:
            batch_task.cancel()
            with suppress(asyncio.CancelledError):
                await batch_task
        if media_task is not None:
            media_task.cancel()
            with suppress(asyncio.CancelledError):
                await media_task
        if video_report_task is not None:
            video_report_task.cancel()
            with suppress(asyncio.CancelledError):
                await video_report_task
        if details_task is not None:
            details_task.cancel()
            with suppress(asyncio.CancelledError):
                await details_task
        if details_worker_stop is not None:
            details_worker_stop.set()
        if details_worker_task is not None:
            details_worker_task.cancel()
            with suppress(asyncio.CancelledError):
                await details_worker_task
        await bot.session.close()


def _detail_request_id(request: dict[str, Any]) -> int:
    return int(request.get("request_id") or request["id"])


def _detail_link_preview(source_urls: list[str]) -> LinkPreviewOptions | None:
    if not source_urls:
        return None
    return LinkPreviewOptions(
        url=source_urls[0],
        prefer_large_media=True,
        show_above_text=False,
    )


async def send_detail_placeholders(
    bot: Bot,
    repository: Repository,
    *,
    limit: int = 5,
) -> dict[str, int]:
    """Send duplicate-safe progress replies for newly queued detail requests."""

    claimed = sent = errors = 0
    for _ in range(limit):
        request = repository.claim_detail_placeholder_request()
        if request is None:
            break
        claimed += 1
        request_id = _detail_request_id(request)
        parent_message_id = request.get("parent_message_id")
        if parent_message_id is None:
            repository.release_detail_placeholder_claim(
                request_id,
                "parent Telegram message is missing",
            )
            errors += 1
            continue
        try:
            placeholder = await bot.send_message(
                chat_id=int(request["chat_id"]),
                text=DETAIL_PLACEHOLDER_HTML,
                reply_parameters=ReplyParameters(message_id=int(parent_message_id)),
            )
            if not repository.set_detail_placeholder_message_id(
                request_id,
                int(placeholder.message_id),
            ):
                raise RuntimeError(f"Could not persist placeholder for detail {request_id}")
            sent += 1
        except Exception as exc:
            logger.exception("Could not send detail placeholder %s", request_id)
            repository.release_detail_placeholder_claim(request_id, str(exc))
            errors += 1
    return {"claimed": claimed, "sent": sent, "errors": errors}


async def send_ready_detail_articles(
    bot: Bot,
    repository: Repository,
    *,
    limit: int = 5,
) -> dict[str, int]:
    """Edit the progress reply into a final article without rebuilding research."""

    claimed = sent = errors = 0
    for _ in range(limit):
        request = repository.claim_ready_detail_request_for_delivery()
        if request is None:
            break
        claimed += 1
        request_id = _detail_request_id(request)
        if request.get("depth", "legacy") != "legacy":
            panel = get_news_panel(repository, request["delivery_id"])
            artifact = repository.get_detailed_article(request["article_id"])
            try:
                visible = bool(
                    panel
                    and await show_result(
                        bot, repository, panel, request, artifact, from_worker=True
                    )
                )
                if visible:
                    repository.mark_detail_request_sent(request_id)
                    repository.mark_detail_keyboard_synced(request_id, "sent")
                    sent += 1
                else:
                    repository.mark_detail_delivery_error(
                        request_id, "Panel view changed or edit pending"
                    )
            except Exception as exc:
                logger.exception("Could not deliver news-reader result %s", request_id)
                repository.mark_detail_delivery_error(request_id, type(exc).__name__)
                errors += 1
            continue
        chat_id = int(request["chat_id"])
        parent_message_id = request.get("parent_message_id")
        source_urls = list(request.get("source_urls") or [])
        preview = _detail_link_preview(source_urls)
        try:
            placeholder_message_id = request.get("placeholder_message_id")
            if placeholder_message_id is not None:
                await bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=int(placeholder_message_id),
                    text=clean_news_html(str(request["rendered_html"])),
                    parse_mode=ParseMode.HTML,
                    link_preview_options=preview,
                    reply_markup=detail_feedback_keyboard(
                        int(request["delivery_id"]),
                        int(request["article_id"]),
                    ),
                )
            else:
                if parent_message_id is None:
                    raise RuntimeError("Parent Telegram message is missing")
                article_message = await bot.send_message(
                    chat_id=chat_id,
                    text=clean_news_html(str(request["rendered_html"])),
                    parse_mode=ParseMode.HTML,
                    reply_parameters=ReplyParameters(message_id=int(parent_message_id)),
                    link_preview_options=preview,
                    reply_markup=detail_feedback_keyboard(
                        int(request["delivery_id"]),
                        int(request["article_id"]),
                    ),
                )
                if not repository.set_detail_placeholder_message_id(
                    request_id,
                    int(article_message.message_id),
                ):
                    raise RuntimeError(f"Could not persist detail message {request_id}")

            if not repository.mark_detail_request_sent(request_id):
                raise RuntimeError(f"Could not mark detail request {request_id} sent")
            if parent_message_id is not None:
                if not await _edit_delivery_markup(
                    bot,
                    repository,
                    int(request["delivery_id"]),
                    user_id=int(request["user_id"]),
                    details_enabled=True,
                ):
                    raise RuntimeError("Could not refresh the current news menu")
                repository.mark_detail_keyboard_synced(request_id, "sent")
            sent += 1
        except Exception as exc:
            logger.exception("Could not deliver detail article %s", request_id)
            repository.mark_detail_delivery_error(request_id, str(exc))
            errors += 1
    return {"ready": claimed, "sent": sent, "errors": errors}


async def send_terminal_detail_results(
    bot: Bot,
    repository: Repository,
    *,
    limit: int = 5,
) -> dict[str, int]:
    """Replace a progress reply with a durable, non-misleading terminal result."""

    claimed = sent = errors = 0
    for _ in range(limit):
        request = repository.claim_terminal_detail_placeholder()
        if request is None:
            break
        claimed += 1
        request_id = _detail_request_id(request)
        status = str(request["status"])
        claim_lease_until = str(request["placeholder_lease_until"])
        if request.get("depth", "legacy") != "legacy":
            panel = get_news_panel(repository, request["delivery_id"])
            try:
                if panel:
                    await show_error(bot, repository, panel, request, terminal_text(request))
                if repository.mark_terminal_detail_placeholder_sent(
                    request_id, status=status, claim_lease_until=claim_lease_until
                ):
                    sent += 1
            except Exception:
                repository.release_terminal_detail_placeholder_claim(
                    request_id, status=status, claim_lease_until=claim_lease_until
                )
                errors += 1
            continue
        try:
            await bot.edit_message_text(
                chat_id=int(request["chat_id"]),
                message_id=int(request["placeholder_message_id"]),
                text=DETAIL_TERMINAL_HTML[status],
            )
            if not repository.mark_terminal_detail_placeholder_sent(
                request_id,
                status=status,
                claim_lease_until=claim_lease_until,
            ):
                raise RuntimeError(f"Could not finalize terminal detail result {request_id}")
            sent += 1
        except Exception:
            logger.exception("Could not deliver terminal detail result %s", request_id)
            repository.release_terminal_detail_placeholder_claim(
                request_id,
                status=status,
                claim_lease_until=claim_lease_until,
            )
            errors += 1
    return {"claimed": claimed, "sent": sent, "errors": errors}


async def sync_detail_keyboards(
    bot: Bot,
    repository: Repository,
    *,
    limit: int = 20,
) -> dict[str, int]:
    """Reflect worker state without ever replacing the reaction controls."""

    requests = repository.detail_requests_needing_keyboard_sync(limit=limit)
    synced = errors = 0
    for request in requests:
        request_id = _detail_request_id(request)
        telegram_message_id = request.get("telegram_message_id")
        if telegram_message_id is None:
            errors += 1
            continue
        display_status = str(request["display_status"])
        try:
            if not await _edit_delivery_markup(
                bot,
                repository,
                int(request["delivery_id"]),
                user_id=int(request["user_id"]),
                details_enabled=True,
            ):
                errors += 1
                continue
            if repository.mark_detail_keyboard_synced(request_id, display_status):
                synced += 1
        except Exception:
            logger.exception("Could not sync detail keyboard %s", request_id)
            errors += 1
    return {"seen": len(requests), "synced": synced, "errors": errors}


async def run_detail_delivery_loop(
    bot: Bot,
    repository: Repository,
    *,
    poll_interval_seconds: float = 2.0,
) -> None:
    """Keep Telegram I/O separate from the blocking research worker."""

    progress = DetailProgress()
    while True:
        try:
            await send_detail_placeholders(bot, repository)
            await progress.tick(bot, repository)
            await send_ready_detail_articles(bot, repository)
            await send_terminal_detail_results(bot, repository)
            await sync_detail_keyboards(bot, repository)
        except Exception:
            logger.exception("Detail Telegram delivery pump failed")
        await asyncio.sleep(poll_interval_seconds)


async def send_formatted_card(
    bot: Bot,
    *,
    chat_id: int,
    card: dict[str, Any],
    delivery_id: int,
    delivery_format: DeliveryFormat,
    include_details: bool = True,
) -> Message:
    reply_markup = reaction_keyboard(delivery_id, include_details=include_details)
    if delivery_format == DeliveryFormat.ARTICLE:
        draft = CardDraft.model_validate_json(card["structured_output_json"])
        rich_html = render_article_rich_html(draft, card["primary_preview_url"])
        return await bot.send_rich_message(
            chat_id=chat_id,
            rich_message=InputRichMessage(html=rich_html, skip_entity_detection=True),
            reply_markup=reply_markup,
        )
    return await bot.send_message(
        chat_id=chat_id,
        text=brief_card_html(card) if include_details else clean_news_html(card["rendered_html"]),
        parse_mode=ParseMode.HTML,
        reply_markup=reply_markup,
        link_preview_options=LinkPreviewOptions(
            url=card["primary_preview_url"],
            prefer_large_media=True,
            show_above_text=False,
        ),
    )


async def send_smoke_card(
    settings: Settings,
    repository: Repository,
    source_url: str,
    *,
    delivery_format: DeliveryFormat | None = None,
) -> int:
    owner_id = configured_owner_id(settings, repository)
    if owner_id is None:
        raise TelegramBotNotConfigured("Send /start to the bot before the smoke test")
    bot = build_bot(settings)
    delivery = repository.create_smoke_delivery(owner_id, source_url)
    try:
        delivery_format = delivery_format or configured_delivery_format(settings, repository)
        if delivery_format == DeliveryFormat.ARTICLE:
            escaped_url = html.escape(source_url, quote=True)
            message = await bot.send_rich_message(
                chat_id=owner_id,
                rich_message=InputRichMessage(
                    html=(
                        "<h1>Тестовая статья персональной ленты</h1>"
                        f'<p>Проверяем <a href="{escaped_url}">контекстную ссылку</a> и '
                        "нативный Rich Message Telegram.</p>"
                        "<h2>Обратная связь</h2>"
                        "<p>Под статьёй сохраняются четыре уровня редакционной реакции.</p>"
                    ),
                    skip_entity_detection=True,
                ),
                reply_markup=reaction_keyboard(delivery["delivery_id"], include_details=False),
            )
        else:
            message = await bot.send_message(
                chat_id=owner_id,
                text=delivery["rendered_html"],
                reply_markup=reaction_keyboard(delivery["delivery_id"], include_details=False),
                link_preview_options=LinkPreviewOptions(
                    url=delivery["primary_preview_url"],
                    prefer_large_media=True,
                    show_above_text=False,
                ),
            )
        repository.mark_delivery_sent(delivery["delivery_id"], message.message_id)
        return delivery["delivery_id"]
    except Exception as exc:
        repository.mark_delivery_error(delivery["delivery_id"], str(exc))
        raise
    finally:
        await bot.session.close()


def menu_reserved_delivery(repository, delivery_id):
    """An automatic sender must not steal a manually reserved outbox item."""
    with repository.connection() as db:
        row = db.execute("SELECT batch_id FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
        reserved = (
            db.execute(
                "SELECT 1 FROM menu_batch_items WHERE delivery_id=? "
                "AND item_kind='news' AND status NOT IN('skipped','failed')",
                (delivery_id,),
            ).fetchone()
            if row and str(row[0] or "").startswith("menu:")
            else None
        )
    return bool(reserved)


async def send_pending_cards(
    settings: Settings,
    repository: Repository,
    *,
    limit: int = 5,
    delivery_format: DeliveryFormat | None = None,
) -> dict[str, int]:
    owner_id = configured_owner_id(settings, repository)
    if owner_id is None:
        raise TelegramBotNotConfigured("Send /start to the bot before delivery")
    cards = repository.deliverable_cards(owner_id, limit)
    bot = build_bot(settings)
    delivery_format = delivery_format or configured_delivery_format(settings, repository)
    sent = errors = 0
    try:
        for card in cards:
            delivery_id = repository.ensure_delivery(card["card_id"], owner_id)
            if menu_reserved_delivery(repository, delivery_id):
                continue
            try:
                message = await send_formatted_card(
                    bot,
                    chat_id=owner_id,
                    card=card,
                    delivery_id=delivery_id,
                    delivery_format=delivery_format,
                    include_details=_details_enabled(settings),
                )
                repository.mark_delivery_sent(delivery_id, message.message_id)
                sent += 1
            except Exception as exc:
                repository.mark_delivery_error(delivery_id, str(exc))
                errors += 1
    finally:
        await bot.session.close()
    return {"cards_seen": len(cards), "sent": sent, "errors": errors}


async def send_cards_by_ids(
    settings: Settings,
    repository: Repository,
    card_ids: list[int],
    *,
    delivery_format: DeliveryFormat = DeliveryFormat.REGULAR,
) -> dict[str, int]:
    """Deliver an explicit card set without consuming the chronological queue."""
    owner_id = configured_owner_id(settings, repository)
    if owner_id is None:
        raise TelegramBotNotConfigured("Send /start to the bot before delivery")
    ordered_ids = list(dict.fromkeys(card_ids))
    if not ordered_ids:
        raise CardsNotDeliverable([])
    cards = repository.deliverable_cards_by_ids(owner_id, ordered_ids)
    deliverable_ids = {int(card["card_id"]) for card in cards}
    unavailable_ids = [card_id for card_id in ordered_ids if card_id not in deliverable_ids]
    if unavailable_ids:
        # Fail closed before opening a Telegram session or partially sending the requested set.
        raise CardsNotDeliverable(unavailable_ids)

    bot = build_bot(settings)
    sent = errors = 0
    try:
        for card in cards:
            delivery_id = repository.ensure_delivery(card["card_id"], owner_id)
            if menu_reserved_delivery(repository, delivery_id):
                continue
            try:
                message = await send_formatted_card(
                    bot,
                    chat_id=owner_id,
                    card=card,
                    delivery_id=delivery_id,
                    delivery_format=delivery_format,
                    include_details=_details_enabled(settings),
                )
                repository.mark_delivery_sent(delivery_id, message.message_id)
                sent += 1
            except Exception as exc:
                repository.mark_delivery_error(delivery_id, str(exc))
                errors += 1
    finally:
        await bot.session.close()
    return {
        "cards_requested": len(ordered_ids),
        "cards_seen": len(cards),
        "sent": sent,
        "errors": errors,
    }


async def send_candidate_batch(
    settings: Settings,
    repository: Repository,
    *,
    batch_id: str,
    card_ids: list[int] | None = None,
) -> dict[str, int]:
    """Deliver one quarantined discovery batch without exposing it to the normal queue."""
    owner_id = configured_owner_id(settings, repository)
    if owner_id is None:
        raise TelegramBotNotConfigured("Send /start to the bot before delivery")
    if card_ids is not None:
        repository.create_candidate_batch(
            chat_id=owner_id,
            card_ids=card_ids,
            batch_id=batch_id,
        )
    cards = repository.candidate_batch_cards(owner_id, batch_id)
    if not cards:
        raise CandidateBatchNotDeliverable(f"Candidate batch is empty: {batch_id}")

    bot = build_bot(settings)
    sent = errors = 0
    try:
        for card in cards:
            delivery_id = int(card["delivery_id"])
            try:
                message = await send_formatted_card(
                    bot,
                    chat_id=owner_id,
                    card=card,
                    delivery_id=delivery_id,
                    delivery_format=DeliveryFormat.REGULAR,
                    include_details=_details_enabled(settings),
                )
                repository.mark_delivery_sent(delivery_id, message.message_id)
                sent += 1
            except Exception as exc:
                repository.mark_delivery_error(delivery_id, str(exc))
                errors += 1
    finally:
        await bot.session.close()
    return {"cards_seen": len(cards), "sent": sent, "errors": errors}
