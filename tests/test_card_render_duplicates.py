import pytest

from app.llm.openrouter import CardDraft, CardSection, GroundedFact
from app.pipeline.cards import ResolvedAnchor, is_redundant_text, render_card_html

# Exact duplicated price wording from the locally inspected card 189.
HIGHLIGHT = "Тариф Catch составляет 99 долларов в месяц, а пробный период — 10 дней."
ECONOMICS = (
    "Тариф стоит 99 долларов в месяц при помесячной оплате; отменить его можно в любое время. "
    "На сайте также указан 10-дневный бесплатный пробный период."
)


def price_draft() -> CardDraft:
    return CardDraft(
        headline="Catch помогает делегировать административные задачи",
        headline_evidence_ids=[0],
        lead=(
            "Сервис помогает руководителю работать с электронной почтой, организовывать встречи "
            "и поездки, подключаясь к существующим рабочим аккаунтам."
        ),
        lead_evidence_ids=[0],
        sections=[CardSection(kind="economics", text=ECONOMICS, evidence_ids=[0])],
        highlight=GroundedFact(text=HIGHLIGHT, evidence_ids=[0], kind="metric"),
        source_mode="reported_event",
    )


@pytest.mark.parametrize(
    "anchor_phrase", [None, "10-дневный бесплатный пробный период", "99 долларов в месяц"]
)
def test_duplicate_price_highlight_drops_but_full_section_and_its_anchor_remain(anchor_phrase):
    draft = price_draft()
    assert not is_redundant_text(HIGHLIGHT, [draft.lead])
    assert is_redundant_text(HIGHLIGHT, [ECONOMICS])
    anchors = (
        [ResolvedAnchor(anchor_phrase, "https://example.com/pricing")] if anchor_phrase else []
    )
    rendered = render_card_html(draft, "https://example.com/catch", anchors)
    assert "<blockquote>" not in rendered
    assert rendered.count("99 долларов в месяц") == 1
    assert "<b>Цена.</b>" in rendered
    assert "отменить его можно в любое время" in rendered
    if anchor_phrase:
        assert rendered.count('href="https://example.com/pricing"') == 1
        assert f">{anchor_phrase}</a>" in rendered


def test_redundant_highlight_is_kept_if_it_alone_contains_a_contextual_anchor():
    rendered = render_card_html(
        price_draft(),
        "https://example.com/catch",
        [ResolvedAnchor("Тариф Catch", "https://example.com/pricing")],
    )
    assert '<blockquote><a href="https://example.com/pricing">Тариф Catch</a>' in rendered
    assert rendered.count('href="https://example.com/pricing"') == 1


def test_distinct_useful_highlight_keeps_blockquote_and_link():
    quote = "В сложных случаях Catch запрашивает решение человека, а не действует без согласования."
    draft = price_draft().model_copy(
        update={
            "highlight": GroundedFact(text=quote, evidence_ids=[0], kind="attributed_statement")
        }
    )
    assert not is_redundant_text(quote, [draft.lead, ECONOMICS])
    rendered = render_card_html(
        draft,
        "https://example.com/catch",
        [ResolvedAnchor("решение человека", "https://example.com/control")],
    )
    assert "<blockquote>" in rendered
    assert '<a href="https://example.com/control">решение человека</a>' in rendered
    assert "</blockquote>" in rendered
    assert "<b>Цена.</b>" in rendered
