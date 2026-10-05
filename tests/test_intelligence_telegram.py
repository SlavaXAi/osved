from __future__ import annotations

import asyncio
import html
import re
from types import SimpleNamespace

import pytest
from test_sensory_telegram import FakeMessage, database_snapshot, handler

from app.config import Settings
from app.db.repository import Repository
from app.telegram_bot.bot import build_router
from app.telegram_bot.intelligence import MAX_VISIBLE_UNITS, intelligence_message


@pytest.fixture
def environment(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path, allowed_user_id=123)
    repository = Repository(settings.database_path)
    repository.migrate()
    return settings, repository, build_router(settings, repository)


@pytest.mark.parametrize("name", ["week_handler", "people_handler"])
@pytest.mark.parametrize("user_id,chat_id", [(999, 999), (123, -100111)])
def test_non_owner_or_group_cannot_read_or_build_private_views(environment, name, user_id, chat_id):
    _, repository, router = environment
    before = database_snapshot(repository)
    message = FakeMessage(user_id, chat_id=chat_id)
    asyncio.run(handler(router, name, message=True)(message))
    assert not message.answers
    assert database_snapshot(repository) == before


@pytest.mark.parametrize("name", ["week_handler", "people_handler"])
def test_empty_stored_corpus_gives_honest_insufficient_data_without_spending(environment, name):
    _, repository, router = environment
    message = FakeMessage(123)
    asyncio.run(handler(router, name, message=True)(message))
    text, options = message.answers[0]
    assert "недостаточно данных" in text
    assert "не означает" in text
    assert "не полный обзор" in text
    assert "Новый поиск не запускался" in text
    assert options["link_preview_options"].is_disabled
    assert repository.current_month_cost() == 0
    with repository.connection() as db:
        product = db.execute("SELECT * FROM intelligence_products").fetchone()
        assert product["validation_status"] == "insufficient_data"
        assert db.execute("SELECT count(*) FROM runs").fetchone()[0] == 0


@pytest.mark.parametrize(
    "name,kind",
    [
        ("week_handler", "weekly_map"),
        ("people_handler", "weekly_personas"),
    ],
)
def test_owner_gets_existing_rendered_product_with_scope_notice(
    environment, monkeypatch, name, kind
):
    _, _, router = environment
    calls = []

    def build(selected_kind):
        calls.append(selected_kind)
        return {
            "validation_status": "valid",
            "rendered_html": (
                "<b>Недельная карта</b>\n"
                '<a href="https://example.org/report">Проверенное событие</a>'
            ),
        }

    monkeypatch.setattr(
        "app.telegram_bot.intelligence.IntelligenceService", lambda *a: SimpleNamespace(build=build)
    )
    message = FakeMessage(123)
    asyncio.run(handler(router, name, message=True)(message))
    assert calls == [kind]
    text, options = message.answers[0]
    assert '<a href="https://example.org/report">Проверенное событие</a>' in text
    assert "Только сохранённые проверенные события" in text
    assert "не полная история действий людей" in text
    assert options["parse_mode"] == "HTML"


def test_long_unicode_html_is_bounded_balanced_and_annotated():
    rendered = '<b>Карта</b><blockquote><a href="https://example.org">' + "🤖 &amp; " * 2000
    rendered += "</a></blockquote><b>Это не должно попасть в обрезанный текст</b>"
    result = intelligence_message(
        {"validation_status": "valid", "rendered_html": rendered}, "weekly_map"
    )
    visible = html.unescape(re.sub(r"<[^>]+>", "", result))
    assert len(visible.encode("utf-16-le")) // 2 <= MAX_VISIBLE_UNITS
    assert result.count("<a ") == result.count("</a>")
    assert result.count("<blockquote>") == result.count("</blockquote>")
    assert "сокращённая карта" in result
    assert "Это не должно" not in result
    assert "&amp;" in result


def test_invalid_product_never_shows_unverified_render():
    result = intelligence_message(
        {
            "validation_status": "invalid",
            "rendered_html": "Необоснованная сенсация",
        },
        "weekly_map",
    )
    assert "Необоснованная сенсация" not in result
    assert "не прошёл проверку" in result


def test_week_people_commands_are_registered_before_feedback_catchall(environment):
    _, _, router = environment
    names = [item.callback.__name__ for item in router.message.handlers]
    assert names.index("week_handler") < len(names) - 1
    assert names.index("people_handler") < len(names) - 1
    assert names[-1] not in {"week_handler", "people_handler"}
