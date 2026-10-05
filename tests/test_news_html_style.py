import asyncio
from html.parser import HTMLParser
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.config import DeliveryFormat
from app.llm.openrouter import CardSection, build_request
from app.pipeline.cards import ResolvedAnchor, render_card_html
from app.pipeline.telegram_style import NEWS_STYLE_INSTRUCTION, clean_news_html, clean_news_text
from app.telegram_bot.bot import send_formatted_card
from tests.test_card_pipeline import valid_draft


def test_actual_writer_prompt_contains_the_style_contract():
    request = build_request(
        model="test/model",
        source_name="Example",
        source_role="primary",
        published_at="2026-09-19T10:00:00Z",
        original_text="Компания выпустила продукт.",
        known_urls=["https://example.com/release"],
    )
    assert NEWS_STYLE_INSTRUCTION in request["messages"][0]["content"]
    assert "parse_mode=HTML" in NEWS_STYLE_INSTRUCTION
    assert "полные пути" in NEWS_STYLE_INSTRUCTION
    assert "не модель" in NEWS_STYLE_INSTRUCTION


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "## Заголовок\n**Важная мысль** и *короткое пояснение*.",
            "Заголовок\nВажная мысль и короткое пояснение.",
        ),
        (
            "Репозиторий `itorrents[.]org` и файл `msedge.exe`.",
            "Репозиторий itorrents[.]org и файл msedge.exe.",
        ),
        (
            "snake_case, GPT_6, C++, C#, 2 * 3, 60%, 800 VDC, CVE-2020-0688",
            "snake_case, GPT_6, C++, C#, 2 * 3, 60%, 800 VDC, CVE-2020-0688",
        ),
        (
            r"Файл `C:\ProgramData\Microsoft\Windows\Telemetry\msedge.exe`.",
            r"Файл C:\ProgramData\Microsoft\Windows\Telemetry\msedge.exe.",
        ),
        ("#ИИ — рубрика, а не заголовок Markdown.", "#ИИ — рубрика, а не заголовок Markdown."),
        ("Метод `__init__` и идентификатор __name__.", "Метод __init__ и идентификатор __name__."),
    ],
)
def test_cosmetic_cleanup_never_rewrites_facts(raw, expected):
    assert clean_news_text(raw) == expected


def test_screenshot_regression_cached_footer_and_source_link():
    source = (
        '<b><a href="https://example.com/report?a=1&amp;b=2">MovieReaper</a></b>\n\n'
        'Публичный репозиторий <a href="https://example.com/source">`itorrents[.]org`</a>.'
        "\n\n<i>Что важно учитывать: Программа маскировалась под "
        "`C:\\ProgramData\\Microsoft\\Windows\\Telemetry\\msedge.exe`. "
        "У оператора было 21 команда.</i>"
    )
    rendered = clean_news_html(source)
    assert "`" not in rendered and "<i>" not in rendered
    assert "Что важно учитывать" not in rendered
    assert '<a href="https://example.com/source">itorrents[.]org</a>' in rendered
    assert "?a=1&amp;b=2" in rendered and "&amp;amp;" not in rendered
    assert "21 команда" in rendered
    assert r"C:\ProgramData\Microsoft\Windows\Telemetry\msedge.exe" in rendered
    assert clean_news_html(rendered) == rendered


def test_rendered_news_has_bold_title_paragraphs_and_no_italic_dump():
    draft = valid_draft().model_copy(
        update={
            "sections": [
                CardSection(
                    kind="limitations",
                    text=(
                        "Инструмент пока работает только с текстом; обработка изображений "
                        "и звука в текущем релизе не поддерживается."
                    ),
                    evidence_ids=[0],
                )
            ]
        }
    )
    rendered = render_card_html(draft, "https://example.com/release")
    assert rendered.startswith("<b>") and "\n\n" in rendered
    assert "<i>" not in rendered and "Что важно учитывать" not in rendered
    assert "обработка изображений" in rendered


def test_linked_phrase_keeps_its_evidence_url_after_backticks_are_removed():
    draft = valid_draft().model_copy(
        update={
            "sections": [
                CardSection(
                    kind="details",
                    text=(
                        "Новый параметр `parallel_tasks` позволяет управлять "
                        "несколькими задачами в одной рабочей сессии."
                    ),
                    evidence_ids=[0],
                )
            ]
        }
    )
    rendered = render_card_html(
        draft,
        "https://example.com/release",
        [ResolvedAnchor("`parallel_tasks`", "https://example.com/docs")],
    )
    assert '<a href="https://example.com/docs">parallel_tasks</a>' in rendered
    assert "`" not in rendered


def test_cleaner_only_emits_supported_tags_and_does_not_add_unsafe_links():
    class Tags(HTMLParser):
        def __init__(self):
            super().__init__()
            self.tags = []

        def handle_starttag(self, tag, attrs):
            self.tags.append(tag)

    rendered = clean_news_html(
        "<p><strong>Заголовок</strong></p>\n\n"
        '<a href="javascript:alert(1)">Слова</a> &lt; 20 &amp; 30.'
        "<blockquote>Цитата <b>с выделением</b></blockquote>"
    )
    parser = Tags()
    parser.feed(rendered)
    assert set(parser.tags) <= {"b", "a", "code", "pre", "blockquote"}
    assert "javascript:" not in rendered
    assert "&lt; 20 &amp; 30" in rendered


def test_send_boundary_cleans_queued_html_without_paid_regeneration_or_db_rewrite():
    card = {
        "rendered_html": "<b>Новость</b>\n\n<i>Что важно учитывать: "
        "Файл `msedge.exe` остался в системе.</i>",
        "primary_preview_url": "https://example.com/report",
    }
    old = dict(card)
    bot = SimpleNamespace(send_message=AsyncMock(return_value=SimpleNamespace(message_id=1)))
    asyncio.run(
        send_formatted_card(
            bot,
            chat_id=123,
            card=card,
            delivery_id=42,
            delivery_format=DeliveryFormat.REGULAR,
        )
    )
    sent = bot.send_message.call_args.kwargs
    assert sent["parse_mode"] == "HTML"
    assert "`" not in sent["text"] and "<i>" not in sent["text"]
    assert "msedge.exe" in sent["text"]
    assert len(sent["reply_markup"].inline_keyboard[0]) == 4
    assert sent["reply_markup"].inline_keyboard[-1][0].text.endswith("Подробнее")
    assert card == old
