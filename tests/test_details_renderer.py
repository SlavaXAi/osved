import pytest

from app.details.renderer import (
    DetailRenderError,
    detail_article_visible_text,
    render_detail_article_html,
)
from tests.test_details_research import fact_documents, valid_article


def test_detail_renderer_embeds_sources_and_stays_within_telegram_limit() -> None:
    draft = valid_article()

    rendered = render_detail_article_html(draft, fact_documents())

    assert '<a href="https://openai.com/atlas">' in rendered
    assert '<a href="https://wire.example/atlas">Независимый тест</a>' in rendered
    assert "<b>Что именно выпустили</b>" in rendered
    assert len(detail_article_visible_text(draft)) <= 4096


def test_renderer_reports_overlap_instead_of_silently_losing_citation() -> None:
    draft = valid_article()
    overlap = draft.source_anchors[1].model_copy(
        update={"phrase": "обрабатывает документы и вызывает инструменты"}
    )
    draft = draft.model_copy(update={"source_anchors": [draft.source_anchors[0], overlap]})

    with pytest.raises(DetailRenderError, match="Source anchor phrases overlap"):
        render_detail_article_html(draft, fact_documents())


def test_renderer_does_not_invent_anchor_when_words_are_absent() -> None:
    draft = valid_article()
    wrong = draft.source_anchors[1].model_copy(update={"phrase": "Вымышленная проверка"})
    draft = draft.model_copy(update={"source_anchors": [draft.source_anchors[0], wrong]})

    with pytest.raises(DetailRenderError, match="Source anchor phrase is absent"):
        render_detail_article_html(draft, fact_documents())
