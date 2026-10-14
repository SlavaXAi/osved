from __future__ import annotations

from enum import StrEnum

from aiogram.filters.callback_data import CallbackData
from aiogram.types import InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder

from app.telegram_bot.feedback import FeedbackAction, Reaction


class FeedbackCallback(CallbackData, prefix="fb"):
    delivery_id: int
    action: FeedbackAction


class DetailsCallback(CallbackData, prefix="dt"):
    """A separate callback namespace so details never masquerade as feedback."""

    delivery_id: int


class NewsDepthCallback(CallbackData, prefix="nd"):
    delivery_id: int
    expanded: bool


class DetailFeedbackAction(StrEnum):
    USEFUL = "useful"
    NOT_WORTH_IT = "not_worth_it"


class DetailFeedbackCallback(CallbackData, prefix="df"):
    delivery_id: int
    article_id: int
    action: DetailFeedbackAction


class DetailsButtonState(StrEnum):
    REQUEST = "request"
    RUNNING = "running"
    READY = "ready"
    RETRY = "retry"
    INSUFFICIENT = "insufficient"
    BUDGET_BLOCKED = "budget_blocked"


REACTION_LABELS = {
    Reaction.LIKE: "👍",
    Reaction.DISLIKE: "👎",
    Reaction.SUPERLIKE: "🔥",
    Reaction.SUPERDISLIKE: "⛔",
}

REACTION_ORDER = (
    Reaction.SUPERDISLIKE,
    Reaction.DISLIKE,
    Reaction.LIKE,
    Reaction.SUPERLIKE,
)

DETAILS_LABELS = {
    DetailsButtonState.REQUEST: "Статья · глубже",
    DetailsButtonState.RUNNING: "Разбор готовится…",
    DetailsButtonState.READY: "✅ Разбор готов",
    DetailsButtonState.RETRY: "↻ Повторить",
    DetailsButtonState.INSUFFICIENT: "Недостаточно источников",
    DetailsButtonState.BUDGET_BLOCKED: "Лимит расходов",
}


def _add_details_button(
    builder: InlineKeyboardBuilder,
    delivery_id: int,
    state: DetailsButtonState,
    *,
    expanded: bool = False,
) -> None:
    if not expanded:
        builder.button(
            text="Подробнее",
            callback_data=NewsDepthCallback(delivery_id=delivery_id, expanded=True),
        )
        builder.button(
            text="Статья",
            callback_data=DetailsCallback(delivery_id=delivery_id),
        )
        return
    builder.button(
        text=DETAILS_LABELS[state],
        callback_data=DetailsCallback(delivery_id=delivery_id),
    )
    builder.button(
        text="Свернуть",
        callback_data=NewsDepthCallback(delivery_id=delivery_id, expanded=False),
    )


def reaction_keyboard(
    delivery_id: int,
    current: Reaction | None = None,
    *,
    include_details: bool = True,
    details_state: DetailsButtonState = DetailsButtonState.REQUEST,
    expanded: bool = False,
) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for reaction in REACTION_ORDER:
        label = REACTION_LABELS[reaction]
        if current == reaction:
            label = f"✅ {label}"
        builder.button(
            text=label,
            callback_data=FeedbackCallback(
                delivery_id=delivery_id,
                action=FeedbackAction(reaction.value),
            ),
        )
    if include_details:
        _add_details_button(builder, delivery_id, details_state, expanded=expanded)
        builder.adjust(4, 2)
    else:
        builder.adjust(4)
    return builder.as_markup()


def superreaction_keyboard(
    delivery_id: int,
    *,
    include_details: bool = True,
    details_state: DetailsButtonState = DetailsButtonState.REQUEST,
    expanded: bool = False,
) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.button(
        text="💬 Прокомментировать",
        callback_data=FeedbackCallback(
            delivery_id=delivery_id,
            action=FeedbackAction.COMMENT,
        ),
    )
    builder.button(
        text="✅ Отправить молча",
        callback_data=FeedbackCallback(
            delivery_id=delivery_id,
            action=FeedbackAction.SILENT,
        ),
    )
    builder.button(
        text="↩️ Назад",
        callback_data=FeedbackCallback(
            delivery_id=delivery_id,
            action=FeedbackAction.BACK,
        ),
    )
    if include_details:
        _add_details_button(builder, delivery_id, details_state, expanded=expanded)
        builder.adjust(1, 1, 1, 2)
    else:
        builder.adjust(1, 1, 1)
    return builder.as_markup()


def detail_feedback_keyboard(
    delivery_id: int,
    article_id: int,
    *,
    current: DetailFeedbackAction | None = None,
) -> InlineKeyboardMarkup:
    """Evaluate whether the extra research repaid the user's attention and cost."""

    builder = InlineKeyboardBuilder()
    for action, label in (
        (DetailFeedbackAction.USEFUL, "👍 Полезно"),
        (DetailFeedbackAction.NOT_WORTH_IT, "👎 Не полезно"),
    ):
        if current == action:
            label = f"✅ {label}"
        builder.button(
            text=label,
            callback_data=DetailFeedbackCallback(
                delivery_id=delivery_id,
                article_id=article_id,
                action=action,
            ),
        )
    builder.adjust(2)
    return builder.as_markup()
