from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.config import load_source_registry
from app.db.repository import Repository
from app.llm.openrouter import (
    CardDraft,
    CardSection,
    CardValidationError,
    GeneratedCardDraft,
    SourceAnchor,
    compact_generated_card_payload,
    headline_strength_errors,
    resolve_verified_headline,
    semantic_hook_errors,
    validate_grounding,
)
from app.ops.baseline import export_baseline
from app.pipeline.cards import (
    ResolvedAnchor,
    render_card_html,
    resolve_contextual_anchors,
)
from app.pipeline.editorial import (
    event_organizations,
    extract_editorial_metadata,
    newsworthiness_features,
)
from app.pipeline.ranking import rank_event_candidates
from app.pipeline.selection import candidate_score
from app.research.facts import (
    EditorialHook,
    EvidenceQuote,
    FactCoverage,
    FactSheetDraft,
    ResearchFact,
    fact_safety_errors,
    sanitize_fact_sheet,
    validate_fact_sheet,
)

PROJECT_ROOT = Path(__file__).parents[1]


def valid_draft() -> CardDraft:
    return CardDraft(
        headline="DeepSeek представила новую агентскую систему",
        headline_evidence_ids=[0],
        lead=(
            "Компания выпустила агентскую систему для работы с программным кодом и "
            "опубликовала описание её основных возможностей для разработчиков."
        ),
        lead_evidence_ids=[0],
        sections=[
            CardSection(
                kind="details",
                text=(
                    "Систему можно развернуть самостоятельно и встроить в существующий "
                    "процесс работы с репозиториями."
                ),
                evidence_ids=[0],
            )
        ],
        highlight=None,
        source_mode="reported_event",
    )


def evidence_item(
    text: str,
    *,
    document_id: int,
    url: str,
    source_class: str,
) -> dict:
    return {
        "source_name": f"Document {document_id}",
        "source_role": "research_fact",
        "published_at": "2026-08-18T00:00:00+00:00",
        "original_text": text,
        "fact_kind": "release",
        "importance": 5,
        "document_ids": [document_id],
        "documents": [
            {
                "id": document_id,
                "url": url,
                "source_class": source_class,
            }
        ],
    }


def test_fact_sheet_derives_a_verbatim_verified_hook() -> None:
    documents = [
        {"id": 1, "source_class": "primary", "content_text": "Official release."},
        {"id": 2, "source_class": "independent", "content_text": "Score: 71 points."},
    ]
    draft = FactSheetDraft(
        event_type="model_release",
        facts=[
            ResearchFact(
                text="Компания выпустила новую модель.",
                document_ids=[1],
                kind="release",
            ),
            ResearchFact(
                text="Модель набрала 71 балл в независимом тесте.",
                document_ids=[2],
                kind="benchmark",
            ),
            ResearchFact(
                text="Модель доступна через API.",
                document_ids=[1],
                kind="availability",
            ),
        ],
        coverage=FactCoverage(
            event_identity=True,
            capabilities=False,
            benchmarks=True,
            pricing=False,
            availability=True,
            limitations=False,
            independent_confirmation=True,
        ),
        conflicts=[],
    )

    sanitized, warnings = sanitize_fact_sheet(draft, documents)

    assert sanitized.editorial_hook is not None
    assert sanitized.editorial_hook.hook_fact == draft.facts[1].text
    assert sanitized.editorial_hook.document_ids == [2]
    assert sanitized.editorial_hook.hook_type == "scale"
    assert sanitized.facts[1].kind == "benchmark"
    assert sanitized.coverage.benchmarks is True
    assert sanitized.coverage.event_identity is True
    assert any("editorial hook is missing" in warning for warning in warnings)
    assert validate_fact_sheet(sanitized, documents) == []


def test_verified_hook_prefers_facts_relevant_to_the_trigger_event() -> None:
    documents = [{"id": 1, "source_class": "primary", "content_text": "Official page."}]
    relevant = ResearchFact(
        text="ChatGPT for Teens включает дополнительные средства защиты для подростков.",
        document_ids=[1],
        kind="capability",
    )
    unrelated_release = ResearchFact(
        text="OpenAI и CodeAI объявили отдельное партнерство для преподавателей.",
        document_ids=[1],
        kind="release",
    )
    draft = FactSheetDraft(
        event_type="software_release",
        facts=[
            relevant,
            unrelated_release,
            ResearchFact(
                text="Режим доступен пользователям в возрасте от 13 до 17 лет.",
                document_ids=[1],
                kind="availability",
            ),
        ],
        coverage=FactCoverage(
            event_identity=True,
            capabilities=True,
            benchmarks=False,
            pricing=False,
            availability=True,
            limitations=False,
            independent_confirmation=False,
        ),
        conflicts=[],
    )

    sanitized, _warnings = sanitize_fact_sheet(
        draft,
        documents,
        event_headline="Introducing ChatGPT for Teens with stronger protections",
    )

    assert sanitized.editorial_hook is not None
    assert sanitized.editorial_hook.hook_fact == relevant.text


def test_verified_hook_uses_event_concept_before_a_generic_numeric_fact() -> None:
    documents = [
        {
            "id": 1,
            "source_class": "primary",
            "content_text": (
                "Новая таксономия включает два уровня и 30 тем. "
                "Количество категорий с детализацией по тематикам видео выросло в 10 раз. "
                "Нововведения уже работают в рекламных алгоритмах RUTUBE."
            ),
        }
    ]
    video_fact = ResearchFact(
        text="Количество категорий с детализацией по тематикам видео выросло в 10 раз.",
        document_ids=[1],
        kind="release",
    )
    draft = FactSheetDraft(
        event_type="other",
        facts=[
            ResearchFact(
                text="Новая таксономия включает два уровня и 30 тем.",
                document_ids=[1],
                kind="release",
            ),
            video_fact,
            ResearchFact(
                text="Нововведения уже работают в рекламных алгоритмах RUTUBE.",
                document_ids=[1],
                kind="release",
            ),
        ],
        coverage=FactCoverage(
            event_identity=True,
            capabilities=False,
            benchmarks=False,
            pricing=False,
            availability=False,
            limitations=False,
            independent_confirmation=False,
        ),
        conflicts=[],
    )

    sanitized, _warnings = sanitize_fact_sheet(
        draft,
        documents,
        event_headline="RUTUBE внедрил ML-модель для категоризации видео с точностью 93%",
    )

    assert sanitized.editorial_hook is not None
    assert sanitized.editorial_hook.hook_fact == video_fact.text


def test_verified_hook_prefers_an_exact_event_metric_over_weak_stem_overlap() -> None:
    documents = [
        {
            "id": 1,
            "source_class": "primary",
            "content_text": (
                "RUTUBE запустил модель для категоризации видео с точностью более 93%. "
                "Новая таксономия содержит два уровня и включает 30 тем."
            ),
        }
    ]
    exact_metric = ResearchFact(
        text="RUTUBE определяет тематику видео с точностью более 93%.",
        document_ids=[1],
        kind="capability",
    )
    draft = FactSheetDraft(
        event_type="other",
        facts=[
            ResearchFact(
                text="Новая таксономия содержит два уровня и включает 30 тем.",
                document_ids=[1],
                kind="release",
            ),
            exact_metric,
            ResearchFact(
                text="Нововведения уже работают в рекламных алгоритмах RUTUBE.",
                document_ids=[1],
                kind="release",
            ),
        ],
        coverage=FactCoverage(
            event_identity=True,
            capabilities=True,
            benchmarks=False,
            pricing=False,
            availability=False,
            limitations=False,
            independent_confirmation=False,
        ),
        conflicts=[],
    )

    sanitized, _warnings = sanitize_fact_sheet(
        draft,
        documents,
        event_headline="RUTUBE внедрил ML-модель для категоризации видео с точностью более 93%",
    )

    assert sanitized.editorial_hook is not None
    assert sanitized.editorial_hook.hook_fact == exact_metric.text
    assert sanitized.coverage.event_identity is True


def test_fact_sheet_fails_closed_without_an_event_relevant_safe_hook() -> None:
    documents = [
        {
            "id": 1,
            "source_class": "primary",
            "content_text": "Official page containing several unrelated announcements.",
        }
    ]
    draft = FactSheetDraft(
        event_type="software_release",
        facts=[
            ResearchFact(
                text="OpenAI и CodeAI объявили отдельное партнерство для преподавателей.",
                document_ids=[1],
                kind="release",
            ),
            ResearchFact(
                text="Компания открыла учебные материалы для преподавателей.",
                document_ids=[1],
                kind="capability",
            ),
            ResearchFact(
                text="Материалы доступны зарегистрированным пользователям.",
                document_ids=[1],
                kind="availability",
            ),
        ],
        coverage=FactCoverage(
            event_identity=True,
            capabilities=True,
            benchmarks=False,
            pricing=False,
            availability=True,
            limitations=False,
            independent_confirmation=False,
        ),
        conflicts=[],
    )

    sanitized, warnings = sanitize_fact_sheet(
        draft,
        documents,
        event_headline="RUTUBE внедрил ML-модель для категоризации видео с точностью 93%",
    )
    errors = validate_fact_sheet(sanitized, documents)

    assert sanitized.editorial_hook is None
    assert sanitized.coverage.event_identity is False
    assert any("no safe fact is relevant" in warning for warning in warnings)
    assert "editorial hook is missing" in errors


def test_model_configuration_steps_are_not_counted_as_a_benchmark() -> None:
    documents = [
        {
            "id": 1,
            "source_class": "primary",
            "content_text": (
                "Sand.ai выпустила MAGI-2 Preview. Модель генерирует видео и звук. "
                "MAGI-2 Preview, base | 100 preview + 5 refiner."
            ),
        }
    ]
    configuration = ResearchFact(
        text=(
            "Базовая модель MAGI-2 Preview использует 100 шагов предварительной обработки "
            "и 5 шагов постобработки."
        ),
        document_ids=[1],
        kind="benchmark",
    )
    draft = FactSheetDraft(
        event_type="other",
        facts=[
            ResearchFact(
                text="Sand.ai выпустила открытую модель MAGI-2 Preview.",
                document_ids=[1],
                kind="release",
            ),
            ResearchFact(
                text="MAGI-2 Preview генерирует видео и звук.",
                document_ids=[1],
                kind="capability",
            ),
            configuration,
        ],
        coverage=FactCoverage(
            event_identity=True,
            capabilities=True,
            benchmarks=True,
            pricing=False,
            availability=False,
            limitations=False,
            independent_confirmation=False,
        ),
        conflicts=[],
    )

    sanitized, warnings = sanitize_fact_sheet(
        draft,
        documents,
        event_headline="Открытая MoE-видеомодель MAGI-2 Preview от Sand.ai",
    )

    normalized = next(fact for fact in sanitized.facts if fact.text == configuration.text)
    assert normalized.kind == "capability"
    assert sanitized.coverage.benchmarks is False
    assert sanitized.coverage.capabilities is True
    assert any("benchmark reclassified as capability" in warning for warning in warnings)


def test_hook_headline_may_add_context_that_its_cited_fact_supports() -> None:
    errors = semantic_hook_errors(
        "У RUTUBE новая таксономия включает 30 тем",
        "Новая таксономия включает 30 тем.",
        supporting_text="Нововведения уже работают в рекламных алгоритмах RUTUBE.",
    )

    assert errors == []

    substitution = semantic_hook_errors(
        "Сбер опубликовал решения десяти математических задач",
        "OpenAI опубликовала решения десяти математических задач.",
        supporting_text="Сбер открыл новый офис.",
    )

    assert any("organization" in error for error in substitution)


def test_benchmark_hook_can_cite_its_release_fact_but_not_uncited_dossier_context() -> None:
    benchmark = "WeatherNext 3 улучшила точность прогнозирования осадков на 60%."
    release = "Google выпустила WeatherNext 3 для прогнозирования погоды."
    snippets = [
        {"id": 0, "text": benchmark, "document_ids": [1]},
        {"id": 1, "text": release, "document_ids": [1]},
    ]
    hook = {
        "hook_fact": benchmark,
        "document_ids": [1],
        "claim_strength": "organization_claim",
        "forbidden_strengthenings": [],
    }
    headline = (
        "По данным Google, компания выпустила WeatherNext 3 "
        "с улучшенной точностью прогнозирования на 60%"
    )

    assert (
        headline_strength_errors(headline, [0, 1], evidence_snippets=snippets, editorial_hook=hook)
        == []
    )
    uncited = headline_strength_errors(
        headline, [0], evidence_snippets=snippets, editorial_hook=hook
    )
    assert any("organization" in error for error in uncited)
    assert any("action" in error for error in uncited)
    no_hook = headline_strength_errors(
        headline, [1], evidence_snippets=snippets, editorial_hook=hook
    )
    assert any("does not cite the selected hook" in error for error in no_hook)


def test_supporting_context_does_not_replace_hook_action() -> None:
    errors = semantic_hook_errors(
        "OpenAI закрыла лабораторию математических задач",
        "OpenAI опубликовала решения математических задач.",
        supporting_text="OpenAI закрыла старую лабораторию.",
    )
    assert any("action" in error for error in errors)


def test_generated_card_normalizes_local_mode_and_overlong_sections() -> None:
    payload = valid_draft().model_dump()
    payload.update(
        {
            "safe_headline": valid_draft().headline,
            "safe_headline_evidence_ids": [0],
            "headline_mode": "default",
        }
    )
    payload["sections"][0]["text"] = "Проверяемая техническая подробность. " * 30
    long_anchor = "Проверяемая техническая подробность " * 5
    payload["source_anchors"] = [
        {
            "phrase": long_anchor.strip(),
            "evidence_id": 0,
            "document_id": 1,
            "role": "technical",
        }
    ]

    compacted = compact_generated_card_payload(payload)
    parsed = GeneratedCardDraft.model_validate(compacted)

    assert parsed.headline_mode == "verified_hook"
    assert len(parsed.sections[0].text) <= 650
    assert parsed.sections[0].text.endswith(".")
    assert parsed.source_anchors == []


def test_unsupported_hook_headline_uses_same_call_safe_fallback() -> None:
    text = "OpenAI опубликовала решения десяти математических задач."
    evidence = [
        evidence_item(
            text,
            document_id=1,
            url="https://openai.com/research/ten-problems",
            source_class="primary",
        )
    ]
    hook = EditorialHook(
        hook_fact=text,
        hook_type="scale",
        document_ids=[1],
        claim_strength="confirmed_fact",
        forbidden_strengthenings=["научное открытие", "прорыв", "впервые"],
    ).model_dump()
    draft = valid_draft().model_copy(
        update={
            "headline": "OpenAI совершила научное открытие в математике",
            "headline_evidence_ids": [0],
            "safe_headline": "OpenAI опубликовала решения десяти математических задач",
            "safe_headline_evidence_ids": [0],
        }
    )

    resolved, reasons = resolve_verified_headline(
        draft,
        source_name="Fact sheet",
        source_role="research_fact",
        published_at="2026-08-18T00:00:00+00:00",
        original_text=text,
        evidence_items=evidence,
        editorial_hook=hook,
    )

    assert resolved.headline == draft.safe_headline
    assert resolved.headline_mode == "safe_fallback"
    assert any("научное открытие" in reason for reason in reasons)


def test_primary_only_benchmark_headline_gets_deterministic_attribution() -> None:
    text = "Qwen3.8-Max набрала 93.0 балла в PaperBench."
    evidence = [
        evidence_item(
            text,
            document_id=1,
            url="https://qwen.ai/research/model",
            source_class="primary",
        )
    ]
    hook = EditorialHook(
        hook_fact=text,
        hook_type="scale",
        document_ids=[1],
        claim_strength="organization_claim",
        forbidden_strengthenings=[],
    ).model_dump()
    draft = valid_draft().model_copy(
        update={
            "headline": "Qwen3.8-Max набрала 93.0 балла в PaperBench",
            "headline_evidence_ids": [0],
            "safe_headline": "Qwen3.8-Max набрала 93.0 балла в PaperBench",
            "safe_headline_evidence_ids": [0],
        }
    )

    resolved, reasons = resolve_verified_headline(
        draft,
        source_name="Fact sheet",
        source_role="research_fact",
        published_at="2026-08-18T00:00:00+00:00",
        original_text=text,
        evidence_items=evidence,
        editorial_hook=hook,
    )

    assert resolved.headline.startswith("По данным ")
    assert "Qwen" in resolved.headline
    assert reasons == []


def test_research_claim_gets_natural_lowercase_attribution() -> None:
    text = "Разработана атака Download More RAM, обходящая защиту Windows."
    evidence = [
        evidence_item(
            text,
            document_id=1,
            url="https://example.com/security-paper",
            source_class="primary",
        )
    ]
    hook = EditorialHook(
        hook_fact=text,
        hook_type="risk",
        document_ids=[1],
        claim_strength="organization_claim",
        forbidden_strengthenings=[],
    ).model_dump()
    draft = valid_draft().model_copy(
        update={
            "headline": "Атака Download More RAM обходит защиту Windows",
            "headline_evidence_ids": [0],
            "safe_headline": "Разработана атака Download More RAM, обходящая защиту Windows",
            "safe_headline_evidence_ids": [0],
        }
    )

    resolved, reasons = resolve_verified_headline(
        draft,
        source_name="Security paper",
        source_role="research_fact",
        published_at="2026-08-18T00:00:00+00:00",
        original_text=text,
        evidence_items=evidence,
        editorial_hook=hook,
    )

    assert resolved.headline.startswith("По данным исследователей, атака")
    assert not resolved.headline.endswith(" на")
    assert reasons == []


def test_semantically_unrelated_headline_falls_back_and_bad_fallback_fails() -> None:
    text = "OpenAI опубликовала решения десяти математических задач."
    evidence = [
        evidence_item(
            text,
            document_id=1,
            url="https://openai.com/research/ten-problems",
            source_class="primary",
        )
    ]
    hook = EditorialHook(
        hook_fact=text,
        hook_type="scale",
        document_ids=[1],
        claim_strength="confirmed_fact",
        forbidden_strengthenings=[],
    ).model_dump()
    draft = valid_draft().model_copy(
        update={
            "headline": "OpenAI полностью закрыла лабораторию климатических исследований",
            "headline_evidence_ids": [0],
            "safe_headline": "OpenAI опубликовала решения десяти математических задач",
            "safe_headline_evidence_ids": [0],
        }
    )

    resolved, reasons = resolve_verified_headline(
        draft,
        source_name="Fact sheet",
        source_role="research_fact",
        published_at="2026-08-18T00:00:00+00:00",
        original_text=text,
        evidence_items=evidence,
        editorial_hook=hook,
    )

    assert resolved.headline_mode == "safe_fallback"
    assert any("conservative paraphrase" in reason for reason in reasons)

    with pytest.raises(CardValidationError):
        resolve_verified_headline(
            draft.model_copy(update={"safe_headline": "OpenAI купила новую лабораторию"}),
            source_name="Fact sheet",
            source_role="research_fact",
            published_at="2026-08-18T00:00:00+00:00",
            original_text=text,
            evidence_items=evidence,
            editorial_hook=hook,
        )


@pytest.mark.parametrize(
    "unsupported_headline",
    [
        "OpenAI опубликовала решения десяти климатических задач",
        "Сбер опубликовал решения десяти математических задач",
    ],
)
def test_hook_guard_rejects_subject_or_organization_substitution(
    unsupported_headline: str,
) -> None:
    text = "OpenAI опубликовала решения десяти математических задач."
    evidence = [
        evidence_item(
            text,
            document_id=1,
            url="https://openai.com/research/ten-problems",
            source_class="primary",
        )
    ]
    hook = EditorialHook(
        hook_fact=text,
        hook_type="scale",
        document_ids=[1],
        claim_strength="confirmed_fact",
        forbidden_strengthenings=[],
    ).model_dump()
    draft = valid_draft().model_copy(
        update={
            "headline": unsupported_headline,
            "headline_evidence_ids": [0],
            "safe_headline": "OpenAI опубликовала решения десяти математических задач",
            "safe_headline_evidence_ids": [0],
        }
    )

    resolved, reasons = resolve_verified_headline(
        draft,
        source_name="Fact sheet",
        source_role="research_fact",
        published_at="2026-08-18T00:00:00+00:00",
        original_text=text,
        evidence_items=evidence,
        editorial_hook=hook,
    )

    assert resolved.headline_mode == "safe_fallback"
    assert reasons


def test_literal_quote_must_support_fact_action_and_every_cited_document() -> None:
    documents = {
        1: {
            "id": 1,
            "source_class": "primary",
            "content_text": "Компания выпустила проект и наняла команду.",
        },
        2: {
            "id": 2,
            "source_class": "independent",
            "content_text": "Independent report confirms the launch.",
        },
    }
    false_fact = ResearchFact(
        text="Компания закрыла проект и уволила команду.",
        document_ids=[1],
        evidence_quotes=[
            EvidenceQuote(
                document_id=1,
                quote="Компания выпустила проект и наняла команду.",
            )
        ],
        kind="release",
    )
    uncovered_independent = ResearchFact(
        text="Компания выпустила проект и наняла команду.",
        document_ids=[1, 2],
        evidence_quotes=[
            EvidenceQuote(
                document_id=1,
                quote="Компания выпустила проект и наняла команду.",
            )
        ],
        kind="release",
    )

    false_errors = fact_safety_errors(false_fact, documents)
    coverage_errors = fact_safety_errors(uncovered_independent, documents)

    assert any("action absent" in error for error in false_errors)
    assert any("cover every document" in error for error in coverage_errors)

    product_swap = ResearchFact(
        text="Компания выпустила модель Qwen3.8-Max.",
        document_ids=[1],
        evidence_quotes=[
            EvidenceQuote(
                document_id=1,
                quote="Компания выпустила модель Qwen3.8.",
            )
        ],
        kind="release",
    )
    documents[1]["content_text"] += " Компания выпустила модель Qwen3.8."

    assert any("Latin product" in error for error in fact_safety_errors(product_swap, documents))

    documents[1].update(
        {
            "publisher": "Qwen",
            "title": "Qwen3.8 model card",
            "url": "https://example.com/Qwen3.8",
        }
    )
    supported_product = product_swap.model_copy(
        update={"text": "Компания выпустила модель Qwen3.8."}
    )
    assert not any(
        "Latin product" in error for error in fact_safety_errors(supported_product, documents)
    )
    assert any("Latin product" in error for error in fact_safety_errors(product_swap, documents))

    documents[1]["title"] = "Qwen3.8-2.4T-A95B model card"
    assert not any(
        "Latin product" in error for error in fact_safety_errors(supported_product, documents)
    )
    assert any("Latin product" in error for error in fact_safety_errors(product_swap, documents))


def test_primary_only_capability_hook_remains_an_attributed_company_claim() -> None:
    documents = [
        {
            "id": 1,
            "source_class": "primary",
            "content_text": "Company documentation.",
        }
    ]
    capability = ResearchFact(
        text="Компания заявляет, что модель умеет автономно писать программный код.",
        document_ids=[1],
        kind="capability",
    )
    draft = FactSheetDraft(
        event_type="other",
        facts=[
            capability,
            ResearchFact(
                text="Система поддерживает самостоятельное развёртывание.",
                document_ids=[1],
                kind="availability",
            ),
            ResearchFact(
                text="Документация описывает ограничения текущей версии.",
                document_ids=[1],
                kind="limitation",
            ),
        ],
        coverage=FactCoverage(
            event_identity=False,
            capabilities=True,
            benchmarks=False,
            pricing=False,
            availability=True,
            limitations=True,
            independent_confirmation=False,
        ),
        conflicts=[],
        editorial_hook=EditorialHook(
            hook_fact=capability.text,
            hook_type="capability",
            document_ids=[1],
            claim_strength="confirmed_fact",
            forbidden_strengthenings=[],
        ),
    )

    sanitized, _warnings = sanitize_fact_sheet(draft, documents)

    assert sanitized.editorial_hook is not None
    assert sanitized.editorial_hook.claim_strength == "organization_claim"


def test_two_body_anchors_are_grounded_and_do_not_change_preview() -> None:
    items = [
        evidence_item(
            "DeepSeek выпустила новую агентскую систему.",
            document_id=1,
            url="https://deepseek.com/release",
            source_class="primary",
        ),
        evidence_item(
            "Компания опубликовала технический отчёт об архитектуре системы.",
            document_id=2,
            url="https://deepseek.com/technical-report",
            source_class="primary",
        ),
        evidence_item(
            "Независимый тест подтвердил работу системы на реальных задачах.",
            document_id=3,
            url="https://benchmark.example/deepseek",
            source_class="independent",
        ),
    ]
    draft = valid_draft().model_copy(
        update={
            "headline_evidence_ids": [0],
            "lead": (
                "Компания выпустила агентскую систему и отдельный технический отчёт, "
                "который описывает её архитектуру и порядок самостоятельного запуска."
            ),
            "lead_evidence_ids": [0, 1],
            "sections": [
                CardSection(
                    kind="benchmarks",
                    text=(
                        "Независимый тест подтвердил работу системы на реальных задачах "
                        "без дополнительных заявлений разработчика."
                    ),
                    evidence_ids=[2],
                )
            ],
            "source_anchors": [
                SourceAnchor(
                    phrase="технический отчёт",
                    evidence_id=1,
                    document_id=2,
                    role="technical",
                ),
                SourceAnchor(
                    phrase="Независимый тест",
                    evidence_id=2,
                    document_id=3,
                    role="independent",
                ),
            ],
        }
    )
    hook = {
        "hook_fact": items[0]["original_text"],
        "hook_type": "none",
        "document_ids": [1],
        "claim_strength": "confirmed_fact",
        "forbidden_strengthenings": [],
    }

    validate_grounding(
        draft,
        original_text=items[0]["original_text"],
        known_urls=[item["documents"][0]["url"] for item in items],
        source_name="Fact sheet",
        source_role="research_fact",
        published_at="2026-08-18T00:00:00+00:00",
        evidence_items=items,
        editorial_hook=hook,
    )
    documents = [item["documents"][0] for item in items]
    anchors = resolve_contextual_anchors(
        draft,
        evidence_items=items,
        research_documents=documents,
        primary_url="https://deepseek.com/release",
    )
    rendered = render_card_html(draft, "https://deepseek.com/release", anchors)

    assert rendered.count("<a href=") == 3
    assert 'href="https://deepseek.com/release"' in rendered
    assert 'href="https://deepseek.com/technical-report"' in rendered
    assert 'href="https://benchmark.example/deepseek"' in rendered


def test_metadata_does_not_confuse_source_country_with_event_geography() -> None:
    metadata = extract_editorial_metadata(
        "OpenAI запустила новую модель для разработчиков.",
        source_topics=["ai"],
        source_region="united_states",
    )

    assert metadata.entities["organizations"] == ["OpenAI"]
    assert metadata.topics == ["ai"]
    assert metadata.regions == ["global"]


def test_legacy_empty_metadata_still_recovers_subject_organization() -> None:
    event = {
        "headline": "OpenAI выпустила новую модель",
        "source_name": "OpenAI News",
        "entities_json": "{}",
    }

    assert event_organizations(event) == ["OpenAI"]
    assert event_organizations({**event, "entities_json": '{"organizations": []}'}) == ["OpenAI"]


def test_routine_patch_is_not_promoted_by_the_word_release() -> None:
    routine = newsworthiness_features(
        "Released a minor patch with bug fixes and a dependency update.",
        source_role="primary",
    )
    substantive = newsworthiness_features(
        "Released a major new model architecture with independent benchmark results.",
        source_role="primary",
    )
    critical_security_patch = newsworthiness_features(
        "Released a critical patch for an actively exploited zero-day vulnerability.",
        source_role="primary",
    )

    assert routine["routine_penalty"] < 0
    assert substantive["routine_penalty"] == 0
    assert critical_security_patch["routine_penalty"] == 0


def test_digest_is_rejected_even_when_its_body_contains_material_keywords() -> None:
    features = newsworthiness_features(
        "📬Утренняя подборка\nOpenAI выпустила модель. NVIDIA построила фабрику.",
        source_role="radar",
    )

    assert features["material_event"] > 0
    assert features["digest_penalty"] < 0


def test_primary_evidence_outranks_a_radar_link_with_other_factors_equal() -> None:
    base = {
        "published_at": datetime.now(UTC).isoformat(),
        "source_priority": "p0",
        "original_text": "Компания представила новую модель для программных агентов. " * 5,
    }
    primary_score, _ = candidate_score(
        {
            **base,
            "source_role": "primary",
            "outbound_urls_json": "[]",
        }
    )
    radar_score, _ = candidate_score(
        {
            **base,
            "source_role": "radar",
            "outbound_urls_json": '["https://company.example/release"]',
        }
    )

    assert primary_score > radar_score


def test_soft_saturation_prefers_alternative_but_keeps_quality_escape_clause() -> None:
    recent = [
        {
            "headline": "OpenAI release",
            "entities_json": json.dumps({"organizations": ["OpenAI"]}),
        }
        for _ in range(4)
    ]
    equal_candidates = [
        {
            "id": 1,
            "score": 8.0,
            "headline": "OpenAI update",
            "entities_json": json.dumps({"organizations": ["OpenAI"]}),
        },
        {
            "id": 2,
            "score": 8.0,
            "headline": "DeepSeek update",
            "entities_json": json.dumps({"organizations": ["DeepSeek"]}),
        },
    ]
    assert rank_event_candidates(equal_candidates, recent_events=recent, limit=1)[0]["id"] == 2

    exceptional = [
        {**equal_candidates[0], "score": 12.0},
        {**equal_candidates[1], "score": 3.1},
    ]
    assert rank_event_candidates(exceptional, recent_events=recent, limit=1)[0]["id"] == 1


def test_corroboration_bonus_counts_distinct_sources_and_is_idempotent(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "corroboration.sqlite3")
    repository.migrate()
    repository.sync_sources(
        load_source_registry(PROJECT_ROOT / "tests/fixtures/config/sources.yaml")
    )

    def insert_item(source_id: str, external_id: str) -> int:
        repository.insert_source_item(
            source_id=source_id,
            external_id=external_id,
            source_url=f"https://example.com/{external_id}",
            original_text="DeepSeek представила новую агентскую систему для программного кода.",
            original_language="ru",
            published_at="2026-08-18T00:00:00+00:00",
            outbound_urls=["https://deepseek.com/release"],
            media_metadata=None,
            payload_hash=f"hash-{external_id}",
            raw_payload={},
        )
        with repository.connection() as connection:
            return int(
                connection.execute(
                    "SELECT id FROM source_items WHERE source_id = ? AND external_id = ?",
                    (source_id, external_id),
                ).fetchone()[0]
            )

    first_id = insert_item("telegram_ai_newz", "first")
    same_source_id = insert_item("telegram_ai_newz", "same-source")
    independent_source_id = insert_item("telegram_prompt_design", "independent-source")

    event_id, _ = repository.attach_item_to_event(
        source_item_id=first_id,
        event_type="telegram_signal",
        headline="DeepSeek представила новую агентскую систему",
        event_at="2026-08-18T00:00:00+00:00",
        primary_url="https://deepseek.com/release",
        fingerprint="same-event",
        score=8.0,
        score_features={"material_event": 2.0},
        decision_reason="test",
    )
    repository.attach_item_to_event(
        source_item_id=same_source_id,
        event_type="telegram_signal",
        headline="DeepSeek представила новую агентскую систему",
        event_at="2026-08-18T00:01:00+00:00",
        primary_url="https://deepseek.com/release",
        fingerprint="same-event",
        score=7.0,
        score_features={"material_event": 2.0},
        decision_reason="test",
    )
    with repository.connection() as connection:
        same_source_score = float(
            connection.execute("SELECT score FROM events WHERE id = ?", (event_id,)).fetchone()[0]
        )
    repository.attach_item_to_event(
        source_item_id=independent_source_id,
        event_type="telegram_signal",
        headline="DeepSeek представила новую агентскую систему",
        event_at="2026-08-18T00:02:00+00:00",
        primary_url="https://deepseek.com/release",
        fingerprint="same-event",
        score=7.0,
        score_features={"material_event": 2.0},
        decision_reason="test",
    )
    repository.attach_item_to_event(
        source_item_id=independent_source_id,
        event_type="telegram_signal",
        headline="DeepSeek представила новую агентскую систему",
        event_at="2026-08-18T00:02:00+00:00",
        primary_url="https://deepseek.com/release",
        fingerprint="same-event",
        score=7.0,
        score_features={"material_event": 2.0},
        decision_reason="test",
    )
    with repository.connection() as connection:
        row = connection.execute(
            "SELECT score, score_features_json FROM events WHERE id = ?",
            (event_id,),
        ).fetchone()

    assert same_source_score == 8.0
    assert float(row["score"]) == 8.75
    assert json.loads(row["score_features_json"])["corroborating_sources"] == 0.75


def test_renderer_keeps_a_context_link_when_its_section_is_textually_redundant() -> None:
    draft = valid_draft().model_copy(
        update={
            "lead": (
                "Компания выпустила агентскую систему для программного кода, описала "
                "архитектуру, самостоятельный запуск и интеграцию с репозиториями."
            ),
            "sections": [
                CardSection(
                    kind="details",
                    text=(
                        "Компания выпустила агентскую систему для программного кода, описала "
                        "архитектуру, самостоятельный запуск и интеграцию с репозиториями "
                        "в техническом отчёте."
                    ),
                    evidence_ids=[0],
                )
            ],
        }
    )

    rendered = render_card_html(
        draft,
        "https://deepseek.com/release",
        [ResolvedAnchor("техническом отчёте", "https://deepseek.com/report")],
    )

    assert 'href="https://deepseek.com/report"' in rendered


def test_private_baseline_export_scrubs_runtime_settings(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "live.sqlite3")
    repository.migrate()
    repository.sync_sources(
        load_source_registry(PROJECT_ROOT / "tests/fixtures/config/sources.yaml")
    )
    smoke = repository.create_smoke_delivery(123, "https://example.com/release")
    repository.mark_delivery_sent(smoke["delivery_id"], 99)
    repository.set_runtime_setting("allowed_user_id", "123")
    output_dir = tmp_path / "baseline"

    result = export_baseline(repository, output_dir=output_dir, limit=50)

    assert result.integrity_check == "ok"
    assert result.foreign_key_violations == 0
    assert result.delivered_cards == 1
    assert (output_dir / "checksums.sha256").exists()
    with sqlite3.connect(output_dir / "snapshot.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM runtime_settings").fetchone()[0] == 0
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["private_artifact"] is True
    assert manifest["contains_runtime_settings"] is False

    with pytest.raises(FileExistsError):
        export_baseline(repository, output_dir=output_dir, limit=50)
