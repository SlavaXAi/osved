import json

import pytest

from app.pipeline.numbers import extract_numeric_values
from app.research.facts import (
    ResearchFact,
    build_fact_sheet_request,
    canonical_hook_claim_strength,
    derive_editorial_hook,
    evidence_quote_bank,
    fact_safety_errors,
    visible_evidence_text,
)


@pytest.mark.parametrize(
    "text",
    [
        "WeatherNext 3 демонстрирует улучшение точности прогнозирования осадков до 60%.",
        "WeatherNext 3 улучшает прогноз погоды на 60%.",
        "WeatherNext 3 improves weather forecasts by 60%.",
        "WeatherNext 3 improves forecasting accuracy by 60%.",
        "WeatherNext 3 improves forecasts of precipitation by 60%.",
        "WeatherNext 3 демонстрирует улучшение точности прогнозирования осадков до 60% "
        "по сравнению с IMERG в среднесрочных глобальных прогнозах.",
        "WeatherNext 3 improves accuracy by 60% in medium-range global forecasts.",
    ],
)
def test_forecasting_capability_is_not_a_future_business_promise(text: str) -> None:
    fact = ResearchFact(text=text, document_ids=[1], evidence_quotes=[], kind="benchmark")
    assert (
        canonical_hook_claim_strength(
            fact, {1: {"source_class": "primary"}}, event_type="model_release"
        )
        == "organization_claim"
    )


@pytest.mark.parametrize(
    "text",
    [
        "Google планирует улучшить прогноз погоды на 60%.",
        "Google plans to improve weather forecasts by 60%.",
        "Google forecasts revenue growth of 60% next year.",
        "По прогнозу компании выручка вырастет на 60%.",
        "Компания прогнозирует рост выручки на 60%.",
        "Компания повысила прогноз выручки на 60%.",
        "Google increased its revenue forecast by 60%.",
    ],
)
def test_actual_future_targets_remain_attributed_forecasts(text: str) -> None:
    fact = ResearchFact(text=text, document_ids=[1], evidence_quotes=[], kind="capability")
    assert (
        canonical_hook_claim_strength(fact, {1: {"source_class": "primary"}}, event_type="other")
        == "forecast_or_target"
    )


def test_comma_thousands_are_not_decimals_but_fractional_commas_stay_valid() -> None:
    assert extract_numeric_values("12,000 meetings; 20,000 tasks; 200,000 emails") == {
        "12000",
        "20000",
        "200000",
    }
    assert extract_numeric_values("$1,234.56 and 5,000+") == {"1234.56", "5000"}
    assert extract_numeric_values("0,125; 1,5; 1,234%; 1,234 million") == {
        "0.125",
        "1.5",
        "1.234%",
        "1234000",
    }


def test_visible_literal_quote_supports_link_and_bold_without_fabrication() -> None:
    raw = "[Catch](https://catch.example/) processed **200,000 emails** for its users."
    quote = "Catch processed 200,000 emails for its users."
    docs = {1: {"source_class": "primary", "title": "Catch", "content_text": raw}}
    fact = ResearchFact(
        text="Catch обработал 200 тысяч писем.",
        document_ids=[1],
        evidence_quotes=[{"document_id": 1, "quote": quote}],
        kind="capability",
    )
    assert fact_safety_errors(fact, docs) == []
    bad = fact.model_copy(update={"text": "Catch обработал 250 тысяч писем."})
    assert any(
        "unsupported numeric value 250000" in error for error in fact_safety_errors(bad, docs)
    )
    wrong_quote = ResearchFact(
        text="Catch обработал 200 тысяч писем.",
        document_ids=[1],
        evidence_quotes=[{"document_id": 1, "quote": quote.replace("processed", "deleted")}],
        kind="capability",
    )
    assert any("literal excerpt" in error for error in fact_safety_errors(wrong_quote, docs))


def test_ai_hyphen_russian_suffix_is_not_an_invented_product_variant() -> None:
    quote = "Your AI assistant for email, scheduling, travel, and calls."
    docs = {1: {"source_class": "primary", "title": "Catch AI Assistant", "content_text": quote}}
    fact = ResearchFact(
        text="AI-ассистент Catch работает с электронной почтой и поездками.",
        document_ids=[1],
        evidence_quotes=[{"document_id": 1, "quote": quote}],
        kind="capability",
    )
    assert not any("Latin product" in error for error in fact_safety_errors(fact, docs))
    bad = fact.model_copy(update={"text": "CatchPro-7 работает с электронной почтой."})
    assert any("Latin product" in error for error in fact_safety_errors(bad, docs))


def test_bank_quotes_are_literal_and_prioritize_late_benchmarks() -> None:
    paragraphs = [
        f"This ordinary paragraph number {index} gives background on the company."
        for index in range(50)
    ]
    important = (
        "The evaluation reports a CRPS improvement of 60% compared with the previous baseline."
    )
    content = "\n\n".join([*paragraphs, important])
    bank = evidence_quote_bank(content)
    assert important in bank
    visible = " ".join(visible_evidence_text(content).split())
    assert all(" ".join(quote.split()) in visible for quote in bank)
    assert all(len(quote) <= 240 for quote in bank)


def test_prompt_protects_technical_meaning_and_selects_event_change() -> None:
    request = build_fact_sheet_request(
        model="test",
        event={"headline": "Test"},
        documents=[
            {
                "id": 1,
                "source_class": "primary",
                "content_text": "[Catch](https://example.com) runs tasks.",
            }
        ],
    )
    prompt = request["messages"][0]["content"]
    assert "Cybersecurity capability" in prompt
    assert "возможности в кибероперациях" in prompt
    assert "а не уровень защищённости" in prompt
    assert "НОВОЕ ИЗМЕНЕНИЕ" in prompt
    assert "не самое большое число" in prompt
    evidence = json.loads(request["messages"][1]["content"].split("\n", 1)[1])
    assert evidence["documents"][0]["content"] == "Catch runs tasks."


def test_alliance_formation_outranks_background_trial_count() -> None:
    facts = [
        ResearchFact(
            text="Brown University проводит 500 клинических испытаний старения.",
            document_ids=[1],
            kind="capability",
        ),
        ResearchFact(
            text="Brown University создаёт альянс для исследований старения "
            "и клинических испытаний.",
            document_ids=[1],
            kind="release",
        ),
    ]
    hook = derive_editorial_hook(
        facts,
        {1: {"source_class": "primary"}},
        event_type="company_event",
        event_headline="Brown University создаёт альянс для клинических испытаний старения",
    )
    assert hook.hook_fact == facts[1].text
