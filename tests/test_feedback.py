from pathlib import Path

from app.config import load_source_registry
from app.db.repository import Repository
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

PROJECT_ROOT = Path(__file__).parents[1]


def delivery(repository: Repository) -> int:
    repository.migrate()
    repository.sync_sources(
        load_source_registry(PROJECT_ROOT / "tests/fixtures/config/sources.yaml")
    )
    return repository.create_smoke_delivery(123, "https://example.com/news")["delivery_id"]


def test_reaction_types_and_keyboards() -> None:
    assert reaction_from_action(FeedbackAction.LIKE) == Reaction.LIKE
    assert reaction_from_action(FeedbackAction.COMMENT) is None
    assert Reaction.SUPERLIKE.is_super
    assert not Reaction.DISLIKE.is_super
    keyboard = reaction_keyboard(1)
    reaction_buttons = keyboard.inline_keyboard[0]
    assert [button.text for button in reaction_buttons] == ["⛔", "👎", "👍", "🔥"]
    assert [button.text for button in keyboard.inline_keyboard[1]] == ["Подробнее"]
    assert FeedbackCallback.unpack(reaction_buttons[0].callback_data).action == (
        FeedbackAction.SUPERDISLIKE
    )
    assert NewsDepthCallback.unpack(keyboard.inline_keyboard[1][0].callback_data).delivery_id == 1

    super_keyboard = superreaction_keyboard(1)
    assert [row[0].text for row in super_keyboard.inline_keyboard] == [
        "💬 Прокомментировать",
        "✅ Отправить молча",
        "↩️ Назад",
        "Подробнее",
    ]


def test_details_button_state_does_not_replace_reactions_or_superreaction_flow() -> None:
    normal = reaction_keyboard(
        9,
        Reaction.LIKE,
        details_state=DetailsButtonState.RUNNING,
        expanded=True,
    )
    assert [button.text for button in normal.inline_keyboard[0]] == [
        "⛔",
        "👎",
        "✅ 👍",
        "🔥",
    ]
    assert normal.inline_keyboard[1][0].text == "Разбор готовится…"
    assert DetailsCallback.unpack(normal.inline_keyboard[1][0].callback_data).delivery_id == 9
    assert NewsDepthCallback.unpack(normal.inline_keyboard[1][1].callback_data).expanded is False

    superreaction = superreaction_keyboard(
        9,
        details_state=DetailsButtonState.READY,
        expanded=True,
    )
    assert [row[0].text for row in superreaction.inline_keyboard[:3]] == [
        "💬 Прокомментировать",
        "✅ Отправить молча",
        "↩️ Назад",
    ]
    assert superreaction.inline_keyboard[3][0].text == "✅ Разбор готов"


def test_details_can_be_omitted_from_smoke_keyboard() -> None:
    assert len(reaction_keyboard(1, include_details=False).inline_keyboard) == 1
    assert len(superreaction_keyboard(1, include_details=False).inline_keyboard) == 3


def test_detail_article_feedback_keyboard_has_its_own_callback_namespace() -> None:
    keyboard = detail_feedback_keyboard(7, 91)
    buttons = keyboard.inline_keyboard[0]
    assert [button.text for button in buttons] == ["👍 Полезно", "👎 Не окупилось"]
    callback = DetailFeedbackCallback.unpack(buttons[1].callback_data)
    assert callback.delivery_id == 7
    assert callback.article_id == 91
    assert callback.action == DetailFeedbackAction.NOT_WORTH_IT


def test_normal_feedback_is_append_only(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "feedback.sqlite3")
    delivery_id = delivery(repository)

    first_id = repository.finalize_feedback(
        delivery_id,
        action="reaction",
        reaction=Reaction.LIKE.value,
    )
    second_id = repository.finalize_feedback(
        delivery_id,
        action="reaction",
        reaction=Reaction.DISLIKE.value,
    )

    assert second_id > first_id
    assert repository.current_reaction(delivery_id) == Reaction.DISLIKE.value
    rows = repository.export_feedback_rows()
    assert len(rows) == 2
    assert rows[1]["previous_reaction"] == Reaction.LIKE.value


def test_external_feedback_can_reject_and_hold_a_delivered_card(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "feedback.sqlite3")
    delivery_id = delivery(repository)
    context = repository.get_delivery_context(delivery_id)
    assert context is not None

    feedback_id = repository.record_feedback_note(
        delivery_id,
        "The card mixes an author's advice into a neutral event summary.",
        reaction=Reaction.DISLIKE.value,
    )
    repository.hold_card_from_feedback(
        context["card_id"],
        "user rejected subjective spillover",
    )

    assert feedback_id > 0
    assert repository.current_reaction(delivery_id) == Reaction.DISLIKE.value
    rows = repository.export_feedback_rows()
    assert rows[-1]["action"] == "external_feedback"
    with repository.connection() as connection:
        card = connection.execute(
            "SELECT validation_status FROM cards WHERE id = ?",
            (context["card_id"],),
        ).fetchone()
        event = connection.execute(
            "SELECT status FROM events WHERE id = ?",
            (context["event_id"],),
        ).fetchone()
    assert card["validation_status"] == "rejected_user_feedback"
    assert event["status"] == "feedback_rejected"


def test_superreaction_comment_session_survives_repository_reopen(tmp_path: Path) -> None:
    database_path = tmp_path / "feedback.sqlite3"
    repository = Repository(database_path)
    delivery_id = delivery(repository)
    repository.begin_feedback_session(delivery_id, 123, Reaction.SUPERDISLIKE.value)
    repository.set_feedback_session_state(delivery_id, "awaiting_comment", prompt_message_id=99)

    reopened = Repository(database_path)
    session = reopened.feedback_session_for_user(123)
    assert session is not None
    assert session["delivery_id"] == delivery_id
    assert session["reaction"] == Reaction.SUPERDISLIKE.value
    assert session["state"] == "awaiting_comment"

    reopened.finalize_feedback(
        delivery_id,
        action="superreaction_comment",
        reaction=Reaction.SUPERDISLIKE.value,
        comment="Never send unverified tool roundups.",
    )
    assert reopened.feedback_session_for_user(123) is None
    rows = reopened.export_feedback_rows()
    assert rows[-1]["comment"] == "Never send unverified tool roundups."
