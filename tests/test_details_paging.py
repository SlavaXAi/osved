from __future__ import annotations

import pytest

from app.details.paging import (
    TelegramPagingError,
    paginate_telegram_html,
    telegram_html_visible_text,
    telegram_html_visible_units,
)
from app.details.renderer import (
    DetailRenderError,
    detail_article_visible_text,
    render_detail_article_html,
    render_detail_article_pages,
)
from tests.test_details_research import fact_documents, valid_article


def test_pages_repeat_linked_header_and_keep_citations_whole():
    header = '<b><a href="https://source.example/a?x=1&amp;y=2">Событие &amp; смысл</a></b>'
    paragraphs = [
        f'Абзац {index}: <a href="https://source.example/{index}">проверяемая причина</a> '
        + "содержательная деталь " * 4
        + "."
        for index in range(4)
    ]
    pages = paginate_telegram_html(header + "\n\n" + "\n\n".join(paragraphs), page_limit=180)
    assert len(pages) == 4
    assert all(page.startswith(header + "\n\n") for page in pages)
    assert all(telegram_html_visible_units(page) <= 180 for page in pages)
    for index, paragraph in enumerate(paragraphs):
        assert pages[index].endswith(paragraph)
        assert sum(paragraph in page for page in pages) == 1


def test_limit_counts_decoded_entities_and_utf16_including_header_and_separator():
    header = "<b>Заголовок</b>"
    body = "😀" * 30 + " &amp; &lt; &gt; &quot;"
    full = header + "\n\n" + body
    units = telegram_html_visible_units(full)
    assert units == len("Заголовок\n\n".encode("utf-16-le")) // 2 + 60 + 8
    assert paginate_telegram_html(full, page_limit=units) == [full]
    with pytest.raises(TelegramPagingError, match="complete Telegram paragraph"):
        paginate_telegram_html(full, page_limit=units - 1)


def test_hidden_url_length_is_not_counted_and_numeric_entities_preserve_text():
    value = '<a href="https://source.example/' + "x" * 5000 + '">&#128512; &#x27;</a>'
    pages = paginate_telegram_html(value, page_limit=4)
    assert telegram_html_visible_text(pages[0]) == "😀 '"
    assert telegram_html_visible_units(pages[0]) == 4


def test_repeat_header_false_keeps_every_paragraph_once():
    full = "<b>Заголовок</b>\n\n" + "а" * 100 + "\n\n" + "б" * 100
    pages = paginate_telegram_html(full, repeat_header=False, page_limit=115)
    assert len(pages) == 2
    assert pages[0].startswith("<b>Заголовок</b>")
    assert "Заголовок" not in pages[1]
    assert "\n\n".join(telegram_html_visible_text(page) for page in pages) == (
        telegram_html_visible_text(full)
    )


def test_plain_first_paragraph_is_not_treated_as_repeatable_headline():
    paragraphs = ["а" * 90, "б" * 90]
    assert paginate_telegram_html("\n\n".join(paragraphs), page_limit=100) == paragraphs


def test_headline_only_stays_a_single_page():
    assert paginate_telegram_html("<b>Заголовок</b>") == ["<b>Заголовок</b>"]


def test_indivisible_formatted_paragraph_is_never_split_at_internal_blank_lines():
    full = '<b>Заголовок</b>\n\n<a href="https://source.example">' + "а" * 60
    full += "\n\n" + "б" * 60 + "</a>"
    with pytest.raises(TelegramPagingError, match="complete Telegram paragraph"):
        paginate_telegram_html(full, page_limit=100)


def test_visible_spaces_inside_formatting_are_counted():
    assert telegram_html_visible_units("<b>  Текст  </b>") == 9
    with pytest.raises(TelegramPagingError):
        paginate_telegram_html("<b>  Текст  </b>", page_limit=8)


@pytest.mark.parametrize(
    "value",
    [
        "<b>Незакрытый текст",
        '<b><a href="https://source.example">Текст</b></a>',
        '<a href="javascript:alert(1)">Текст</a>',
        '<a href="https://source.example" onclick="alert(1)">Текст</a>',
        "<script>Текст</script>",
        "Текст &unknown;",
        "Текст &#xD800;",
        "Текст &#0;",
        "Текст & данные",
        "Текст < данные",
        "<!-- comment -->Текст",
    ],
)
def test_unsafe_or_unbalanced_html_is_rejected(value):
    with pytest.raises(TelegramPagingError):
        paginate_telegram_html(value)


@pytest.mark.parametrize("limit", [0, 4097, True, 3.5])
def test_page_limit_cannot_exceed_telegram_or_be_noninteger(limit):
    with pytest.raises(TelegramPagingError):
        paginate_telegram_html("Текст", page_limit=limit)


def test_prose_render_omits_labels_but_preserves_content_and_links():
    draft = valid_article()
    prose = render_detail_article_html(draft, fact_documents(), prose=True)
    assert "Что именно выпустили" not in prose
    assert "Что пока неизвестно" not in prose
    assert '<a href="https://wire.example/atlas">Независимый тест</a>' in prose
    assert draft.lead in telegram_html_visible_text(prose)
    assert draft.limitations.text in telegram_html_visible_text(prose)
    assert telegram_html_visible_text(prose) == detail_article_visible_text(draft, prose=True)
    legacy = render_detail_article_html(draft, fact_documents())
    assert "<b>Что именно выпустили</b>" in legacy


def test_long_full_artifact_can_be_saved_and_delivered_as_complete_pages():
    draft = valid_article()
    paragraphs = [paragraph for section in draft.sections for paragraph in section.paragraphs]
    sections = [
        draft.sections[0].model_copy(update={"paragraphs": paragraphs[:2]}),
        draft.sections[1].model_copy(update={"paragraphs": paragraphs[2:]}),
        draft.sections[0].model_copy(update={"paragraphs": paragraphs[:2]}),
        draft.sections[1].model_copy(update={"paragraphs": paragraphs[2:]}),
        draft.sections[0].model_copy(update={"paragraphs": paragraphs[:2]}),
    ]
    draft = draft.model_copy(update={"sections": sections})
    full = render_detail_article_html(draft, fact_documents(), prose=True, visible_limit=None)
    pages = render_detail_article_pages(draft, fact_documents(), page_limit=1000)
    assert len(pages) > 1 and all(telegram_html_visible_units(page) <= 1000 for page in pages)
    headline = detail_article_visible_text(draft, prose=True).split("\n\n", 1)[0]
    bodies = [telegram_html_visible_text(page).removeprefix(headline + "\n\n") for page in pages]
    assert "\n\n".join(bodies) == telegram_html_visible_text(full).removeprefix(headline + "\n\n")


def test_single_message_default_counts_astral_characters_in_utf16():
    draft = valid_article()
    draft = draft.model_copy(update={"lead": "😀" * 2050 + "."})
    with pytest.raises(DetailRenderError, match="UTF-16"):
        render_detail_article_html(draft, fact_documents())


def test_page_renderer_propagates_indivisible_paragraph_error_without_cutting():
    with pytest.raises(DetailRenderError, match="complete Telegram paragraph"):
        render_detail_article_pages(valid_article(), fact_documents(), page_limit=150)
