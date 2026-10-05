import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.telegram_bot.news_demo import (
    NewsDemoCallback,
    demo_key,
    demo_keyboard,
    handle_news_demo,
    publish_news_demos,
)


class Repository:
    def __init__(self):
        self.runtime = {}

    def get_runtime_setting(self, key):
        return self.runtime.get(key)

    def set_runtime_setting(self, key, value):
        self.runtime[key] = value


def demo():
    return {
        "id": "sep20A",
        "variant": "A",
        "group": "ab",
        "navigation": "steps",
        "labels": {
            "context": "Подробнее",
            "article": "Статья",
            "back": "Назад",
            "collapse": "Свернуть",
            "short": "Событие",
        },
        "levels": {"short": "Кратко", "context": "Средний текст", "article": "Статья"},
    }


def test_demo_publishes_once_and_callbacks_do_not_use_news_feedback_or_research():
    repo = Repository()
    bot = SimpleNamespace(send_message=AsyncMock(return_value=SimpleNamespace(message_id=700)))
    message = SimpleNamespace(chat=SimpleNamespace(id=123), message_id=700, edit_text=AsyncMock())
    query = SimpleNamespace(from_user=SimpleNamespace(id=123), message=message, answer=AsyncMock())

    async def exercise():
        first = await publish_news_demos(bot, repo, chat_id=123, demos=[demo()])
        again = await publish_news_demos(bot, repo, chat_id=123, demos=[demo()])
        assert first == again == [700]
        await handle_news_demo(
            query, NewsDemoCallback(demo_id="sep20A", action="context"), repo, allowed=True
        )
        assert message.edit_text.call_args.args[0] == "Средний текст"
        await handle_news_demo(
            query, NewsDemoCallback(demo_id="sep20A", action="article"), repo, allowed=True
        )
        assert message.edit_text.call_args.args[0] == "Статья"
        await handle_news_demo(
            query, NewsDemoCallback(demo_id="sep20A", action="choose"), repo, allowed=True
        )

    asyncio.run(exercise())
    assert bot.send_message.await_count == 1
    assert json.loads(repo.runtime["news_ux_vote:ab:123"])["variant"] == "A"
    assert set(repo.runtime) == {demo_key("sep20A"), "news_ux_vote:ab:123"}


def test_demo_rejects_foreign_message_and_owner():
    repo = Repository()
    repo.set_runtime_setting(
        demo_key("sep20A"), json.dumps({**demo(), "chat_id": 123, "message_id": 700})
    )
    query = SimpleNamespace(
        from_user=SimpleNamespace(id=999),
        answer=AsyncMock(),
        message=SimpleNamespace(
            chat=SimpleNamespace(id=999), message_id=700, edit_text=AsyncMock()
        ),
    )
    for allowed in (False, True):
        asyncio.run(
            handle_news_demo(
                query, NewsDemoCallback(demo_id="sep20A", action="context"), repo, allowed=allowed
            )
        )
    query.message.edit_text.assert_not_called()


def test_tab_navigation_keeps_three_levels_and_choice_separate():
    keyboard = demo_keyboard({**demo(), "navigation": "tabs"}, "context").inline_keyboard
    assert [len(row) for row in keyboard] == [3, 1]
    assert keyboard[0][1].text == "· Подробнее"
    assert all(len(button.callback_data.encode()) <= 64 for row in keyboard for button in row)
