import copy
import json
from pathlib import Path

import pytest

from app.details.models import DeepArticleDraft, DetailFactSnapshot, DetailProviderCall
from app.details.research import (
    DetailBudgetError,
    DetailGenerationError,
    _unique_link_phrase,
    build_deep_article_request,
    compact_generated_deep_article_payload,
    deep_article_prose_errors,
    generate_deep_article,
    validate_deep_article,
)
from tests.test_details_research import deep_generator_context as deep_generator_context
from tests.test_details_research import valid_article


def test_actual_luna_snapshot_rejects_clipping_and_bracket_markers_without_erasing_them():
    fixture = Path(__file__).parent / "fixtures/details_precise_audit_v7.json"
    data = json.loads(fixture.read_text())
    draft = DeepArticleDraft.model_validate(data["draft"])
    snapshot = DetailFactSnapshot.model_validate(data["snapshot"])
    assert snapshot.event_id == 491 and len(snapshot.facts) == 6
    errors = deep_article_prose_errors(draft)
    assert any("headline must be rewritten" in error for error in errors)
    assert any("highlight ends with an incomplete sentence" in error for error in errors)
    assert any("bracket citation" in error for error in errors)
    assert any("независимим" in error for error in errors)
    compacted = compact_generated_deep_article_payload(copy.deepcopy(data["draft"]))
    assert compacted["headline"] == data["draft"]["headline"]
    assert compacted["highlight"] == data["draft"]["highlight"]
    assert compacted["sections"] == data["draft"]["sections"]


def test_details_provider_does_not_force_character_clipping(deep_generator_context):
    _, repo, snapshot, docs = deep_generator_context
    payload = build_deep_article_request(
        model="openai/gpt-5.6-luna", event=repo.event, snapshot=snapshot, documents=docs,
    )
    schema = json.dumps(payload["response_format"]["json_schema"]["schema"])
    assert "maxLength" not in schema and "minLength" not in schema
    assert "maxItems" in schema  # Keep structural bounds and local Pydantic validation.
    prompt = payload["messages"][0]["content"]
    assert "не обрывай слово" in prompt
    assert "квадратных скобках" in prompt
    assert "для создания" in prompt


class ArticleSequence:
    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        index = len(self.calls) - 1
        assert index < len(self.outcomes), "Unexpected extra paid attempt"
        draft = valid_article()
        if self.outcomes[index] == "bad":
            draft = draft.model_copy(update={"lead": draft.lead + " [source marker]"})
        return draft, DetailProviderCall(
            model=kwargs["model"], response_id=f"fixture-{index}", prompt_tokens=100,
            completion_tokens=50, reported_cost_usd=0.002, accounted_cost_usd=0.002,
        )


@pytest.mark.parametrize("outcomes", [["good"], ["bad", "good"], ["bad", "bad"]])
def test_details_repair_is_bounded_with_unchanged_facts_and_full_cost(
    deep_generator_context, outcomes,
):
    settings, repo, snapshot, docs = deep_generator_context
    settings = settings.model_copy(update={
        "openrouter_card_model": "fixture/model", "openrouter_fallback_model": "fixture/model",
        "openrouter_model": "fixture/model",
    })
    generator = ArticleSequence(outcomes)
    before = snapshot.model_dump()
    if outcomes[-1] == "bad":
        with pytest.raises(DetailGenerationError) as raised:
            generate_deep_article(
                settings, repo, 7, snapshot, docs, generator=generator, max_cost_usd=0.02,
            )
        cost = raised.value.cost_usd
    else:
        result = generate_deep_article(
            settings, repo, 7, snapshot, docs, generator=generator, max_cost_usd=0.02,
        )
        cost = result.cost_usd
        assert validate_deep_article(result.draft, snapshot=snapshot, documents=docs) == []
    assert len(generator.calls) == len(outcomes)
    assert cost == pytest.approx(len(outcomes) * 0.002)
    assert snapshot.model_dump() == before
    if len(outcomes) == 2:
        assert generator.calls[0]["snapshot"] == generator.calls[1]["snapshot"]
        assert generator.calls[0]["documents"] == generator.calls[1]["documents"]
        assert "repair_context" not in generator.calls[0]
        assert "bracket citation" in generator.calls[1]["repair_context"]["validation_error"]
        assert repo.usage[-1][1]["metadata"]["is_repair"] is True
        assert repo.usage[-1][1]["metadata"]["repair_error_refs"][0]["response_id"] == "fixture-0"


def test_details_repair_obeys_remaining_request_budget(deep_generator_context):
    settings, repo, snapshot, docs = deep_generator_context
    generator = ArticleSequence(["bad"])
    with pytest.raises(DetailBudgetError) as raised:
        generate_deep_article(
            settings, repo, 7, snapshot, docs, generator=generator, max_cost_usd=0.003,
        )
    assert raised.value.cost_usd == 0.002
    assert len(generator.calls) == 1


@pytest.mark.parametrize("internal", [
    "Доступные facts позволяют описать заявленные меры.",
    "Во facts содержится описание продукта.",
    "В предоставленном evidence нет сведений об аудите.",
])
def test_actual_internal_packet_jargon_is_not_published(internal):
    draft = valid_article()
    draft = draft.model_copy(update={"lead": draft.lead + " " + internal})
    assert any("internal evidence-packet" in error for error in deep_article_prose_errors(draft))


def test_english_brand_is_not_misread_as_internal_packet_jargon():
    draft = valid_article()
    draft = draft.model_copy(update={"lead": draft.lead + " Компания Facts представила продукт."})
    assert not any(
        "internal evidence-packet" in error for error in deep_article_prose_errors(draft)
    )


def test_automatic_anchor_fallback_keeps_two_to_six_words_without_dangling_preposition():
    text = "Catch работает через почту, Slack и телефон после подключения существующих аккаунтов."
    phrase = _unique_link_phrase(text, text, set())
    assert phrase and 2 <= len(phrase.split()) <= 6
    assert phrase.split()[-1] not in {"и", "с", "для", "по"}
    assert phrase in text
