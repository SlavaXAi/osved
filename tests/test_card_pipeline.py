import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from app.config import DeliveryFormat, Settings, load_source_registry
from app.db.repository import Repository
from app.llm.openrouter import (
    CardDraft,
    CardSection,
    CardValidationError,
    GroundedFact,
    OpenRouterCardGenerator,
    OpenRouterResult,
    build_request,
    normalize_audit_kinds,
    normalize_russian_text,
    russian_quality_errors,
    split_evidence_text,
    validate_grounding,
)
from app.pipeline.cards import (
    choose_link_phrase,
    generate_cards,
    generation_models,
    is_redundant_text,
    render_article_rich_html,
    render_card_html,
    unique_known_urls,
)
from app.pipeline.selection import preferred_primary_url, prepare_candidates
from app.research.dossier import build_event_dossier
from app.research.policy import fact_policy_metadata
from app.telegram_bot.bot import send_formatted_card

PROJECT_ROOT = Path(__file__).parents[1]


def valid_draft() -> CardDraft:
    return CardDraft(
        headline="DeepSeek представил новый агентский инструмент",
        headline_evidence_ids=[0],
        lead=(
            "Компания открыла агентский инструмент для работы с программным кодом, "
            "а веса модели опубликовала в открытом доступе. Инструмент разбивает задачу "
            "на отдельные шаги и позволяет проверять промежуточный результат."
        ),
        lead_evidence_ids=[0],
        sections=[
            CardSection(
                kind="details",
                text=(
                    "Релиз можно развернуть самостоятельно и встроить в существующий процесс "
                    "работы с репозиториями без передачи кода внешнему сервису."
                ),
                evidence_ids=[0],
            )
        ],
        highlight=None,
        source_mode="reported_event",
    )


def test_openrouter_request_requires_structured_outputs() -> None:
    payload = build_request(
        model="google/gemini-2.5-flash-lite",
        source_name="Test",
        source_role="radar",
        published_at="2026-08-13T10:00:00+00:00",
        original_text="Компания DeepSeek сообщила о релизе, веса открыты.",
        known_urls=["https://example.com/release"],
    )

    assert payload["provider"] == {"require_parameters": True}
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["strict"] is True
    import json

    schema_text = json.dumps(payload["response_format"]["json_schema"]["schema"])
    assert "maxLength" not in schema_text
    assert "minLength" not in schema_text
    assert "maxItems" in schema_text
    assert "самая большая цифра" in payload["messages"][0]["content"]


def test_card_quality_rejects_cut_sentence_without_rewriting_facts() -> None:
    draft = valid_draft().model_copy(update={"lead": valid_draft().lead + " В числе изменений,"})
    assert any("incomplete sentence" in error for error in russian_quality_errors(draft))
    broken = valid_draft().model_copy(update={"lead": valid_draft().lead + " Проекты,,"})
    assert any("broken punctuation" in error for error in russian_quality_errors(broken))


def test_reasoning_editor_omits_unsupported_temperature_and_caps_price() -> None:
    payload = build_request(
        model="openai/gpt-5.6-luna",
        source_name="Test",
        source_role="primary",
        published_at="2026-09-07T00:00:00+00:00",
        original_text="Компания выпустила новый продукт.",
        known_urls=["https://example.com/release"],
    )
    assert "temperature" not in payload
    assert payload["max_tokens"] == 3200
    assert payload["reasoning"]["effort"] == "low"
    assert payload["provider"]["max_price"] == {"prompt": 0.5, "completion": 2.0}


def test_measured_improvement_is_not_rejected_as_vague_adjective() -> None:
    draft = valid_draft().model_copy(
        update={"headline": "DeepSeek сообщил об улучшении точности на 60%"}
    )
    validate_grounding(
        draft,
        original_text="DeepSeek сообщил об улучшении точности на 60%.",
        known_urls=["https://example.com/benchmark"],
    )
    with pytest.raises(CardValidationError, match="evaluative"):
        validate_grounding(
            draft.model_copy(update={"headline": "DeepSeek представил улучшенную модель"}),
            original_text="DeepSeek представил улучшенную модель.",
            known_urls=["https://example.com/benchmark"],
        )


def test_generator_parses_usage_and_grounding_is_separate() -> None:
    draft = valid_draft()

    def transport(payload, api_key):
        assert api_key == "test-key"
        return {
            "id": "generation-1",
            "model": payload["model"],
            "choices": [{"message": {"content": draft.model_dump_json()}}],
            "usage": {
                "cost": 0.00012,
                "prompt_tokens": 100,
                "completion_tokens": 50,
            },
        }

    settings = Settings(openrouter_api_key=SecretStr("test-key"))
    result = OpenRouterCardGenerator(settings, transport=transport).generate(
        source_name="Test",
        source_role="radar",
        published_at="2026-08-13T10:00:00+00:00",
        original_text="Компания DeepSeek сообщила о релизе, веса открыты.",
        known_urls=["https://example.com/release"],
    )

    assert result.cost_usd == pytest.approx(0.00012)
    assert result.prompt_tokens == 100
    assert result.draft == draft
    validate_grounding(
        result.draft,
        original_text="Компания DeepSeek сообщила о релизе, веса открыты.",
        known_urls=["https://example.com/release"],
    )

    broken = draft.model_copy(
        update={
            "sections": [
                CardSection(
                    kind="details",
                    text=(
                        "Веса модели опубликованы в открытом доступе и доступны для "
                        "самостоятельного развертывания."
                    ),
                    evidence_ids=[99],
                )
            ]
        }
    )
    with pytest.raises(CardValidationError, match="unknown evidence ID"):
        validate_grounding(
            broken,
            original_text="Компания DeepSeek сообщила о релизе, веса открыты.",
            known_urls=["https://example.com/release"],
        )


def test_generation_uses_one_fallback_after_validation_failure(tmp_path: Path) -> None:
    settings = Settings(
        app_env="test",
        data_dir=tmp_path,
        sources_file=PROJECT_ROOT / "tests/fixtures/config/sources.yaml",
        interests_file=PROJECT_ROOT / "tests/fixtures/config/interests.yaml",
        telegram_session_path=tmp_path / "telegram-reader.session",
        openrouter_card_model="cheap/model",
        openrouter_model="strong/model",
        openrouter_fallback_model="strong/model",
    )
    repository = Repository(settings.database_path)
    repository.migrate()
    repository.sync_sources(load_source_registry(settings.sources_file))
    repository.insert_source_item(
        source_id="telegram_prompt_design",
        external_id="fallback-test",
        source_url="https://example.com/fallback-test",
        original_text=(
            "Компания DeepSeek выпустила агентский инструмент с открытыми весами и новым "
            "режимом длительной работы над задачами."
        ),
        original_language="ru",
        published_at=datetime.now(UTC).isoformat(),
        outbound_urls=["https://example.com/fallback-test"],
        media_metadata=None,
        payload_hash="fallback-test-hash",
        raw_payload={},
    )
    prepare_candidates(settings, repository)
    event_id = repository.events_without_cards()[0]["id"]
    document_id = repository.upsert_research_document(
        canonical_url="https://example.com/researched-fallback-test",
        url="https://example.com/researched-fallback-test",
        title="Release evidence",
        publisher="Example",
        source_class="primary",
        retrieved_at=datetime.now(UTC).isoformat(),
        expires_at="2099-01-01T00:00:00+00:00",
        content_text=(
            "Компания DeepSeek выпустила агентский инструмент с открытыми весами и новым "
            "режимом длительной работы над задачами."
        ),
    )
    repository.attach_research_document(
        event_id=event_id,
        research_document_id=document_id,
        evidence_role="primary",
    )
    repository.create_fact_sheet(
        event_id=event_id,
        schema_version="fact_sheet_v1",
        facts=[
            {
                "text": (
                    "Компания DeepSeek выпустила агентский инструмент с открытыми весами и новым "
                    "режимом длительной работы над задачами."
                ),
                "document_ids": [document_id],
            }
        ],
        coverage={
            **fact_policy_metadata(),
            "event_type": "other",
            "event_identity": True,
            "conflicts": [],
        },
        validation_status="valid",
        expires_at="2099-01-01T00:00:00+00:00",
    )
    build_event_dossier(repository, event_id)
    calls: list[str] = []

    class FakeGenerator:
        def generate(self, **kwargs):
            model = kwargs["model"]
            calls.append(model)
            draft = valid_draft()
            if model == "cheap/model":
                draft = draft.model_copy(update={"headline_evidence_ids": [99]})
            return OpenRouterResult(
                draft=draft,
                model=model,
                cost_usd=0.001,
                prompt_tokens=100,
                completion_tokens=50,
                response_id=f"response-{model}",
            )

    stats = generate_cards(settings, repository, limit=1, generator=FakeGenerator())

    assert generation_models(settings) == ["cheap/model", "strong/model"]
    assert calls == ["cheap/model", "strong/model"]
    assert stats.cards_created == 1
    assert stats.errors == 0
    assert stats.cost_usd == pytest.approx(0.002)
    assert repository.status_counts()["valid_cards"] == 1
    with repository.connection() as connection:
        card = connection.execute(
            "SELECT id, primary_preview_url FROM cards WHERE event_id = ?", (event_id,)
        ).fetchone()
    assert card["primary_preview_url"] == "https://example.com/researched-fallback-test"
    media = repository.card_media(card["id"])
    assert media is not None
    assert media["mode"] == "telegram_preview"


def test_evidence_is_programmatically_numbered() -> None:
    assert split_evidence_text(
        "Первый факт. Второй факт!\n\nТретий блок https://example.com/x"
    ) == [
        "Первый факт.",
        "Второй факт!",
        "Третий блок [ссылка]",
    ]


def test_request_combines_evidence_and_preserves_source_roles() -> None:
    payload = build_request(
        model="google/gemini-2.5-flash-lite",
        source_name="Primary",
        source_role="radar",
        published_at="2026-08-13T10:00:00+00:00",
        original_text="Основной факт.",
        known_urls=["https://example.com/release"],
        evidence_items=[
            {
                "source_name": "Prompt Design",
                "source_role": "radar",
                "published_at": "2026-08-13T10:00:00+00:00",
                "original_text": "Компания выпустила модель. https://example.com/release",
            },
            {
                "source_name": "Crypto Essay",
                "source_role": "author_radar",
                "published_at": "2026-08-13T11:00:00+00:00",
                "original_text": "По оценке автора, модель почти на 90% дешевле конкурента.",
            },
        ],
    )

    evidence = payload["messages"][1]["content"]
    assert '"source_name":"Prompt Design"' in evidence
    assert '"source_role":"author_radar"' in evidence
    assert "https://example.com/release" not in evidence
    assert "[ссылка]" in evidence


def test_author_evidence_deterministically_sets_audit_kind() -> None:
    draft = valid_draft().model_copy(
        update={
            "sections": [
                CardSection(
                    text="Crypto Essay оценивает инструмент как более дешёвый.",
                    evidence_ids=[1],
                    kind="details",
                )
            ],
        }
    )
    normalized = normalize_audit_kinds(
        draft,
        [
            {"id": 0, "source_name": "Prompt Design", "source_role": "radar"},
            {"id": 1, "source_name": "Crypto Essay", "source_role": "author_radar"},
        ],
        event_type="public_statement",
    )

    assert normalized.sections[0].kind == "statement"
    assert normalized.headline == draft.headline


def test_routine_press_release_statement_is_dropped() -> None:
    draft = valid_draft().model_copy(
        update={
            "sections": [
                *valid_draft().sections,
                CardSection(
                    kind="statement",
                    text=(
                        "Руководитель компании назвал партнерство важным шагом для дальнейшего "
                        "развития совместных решений."
                    ),
                    evidence_ids=[0],
                ),
            ]
        }
    )

    normalized = normalize_audit_kinds(
        draft,
        [{"id": 0, "source_name": "Press release", "source_role": "research_fact"}],
    )

    assert [section.kind for section in normalized.sections] == ["details"]


def test_section_label_requires_matching_fact_kind() -> None:
    draft = valid_draft().model_copy(
        update={
            "sections": [
                CardSection(
                    kind="benchmarks",
                    text=(
                        "Модель использует сто шагов предварительной обработки и пять шагов "
                        "постобработки при генерации результата."
                    ),
                    evidence_ids=[0],
                ),
                CardSection(
                    kind="benchmarks",
                    text=(
                        "Модель набрала восемьдесят пять процентов в проверяемом тесте и "
                        "превысила средний человеческий результат."
                    ),
                    evidence_ids=[1],
                ),
            ]
        }
    )

    normalized = normalize_audit_kinds(
        draft,
        [
            {
                "id": 0,
                "source_name": "Primary",
                "source_role": "research_fact",
                "fact_kind": "capability",
            },
            {
                "id": 1,
                "source_name": "Primary",
                "source_role": "research_fact",
                "fact_kind": "benchmark",
            },
        ],
    )

    assert [section.kind for section in normalized.sections] == ["details", "benchmarks"]


def test_grounding_rejects_raw_urls_in_generated_text() -> None:
    draft = valid_draft().model_copy(
        update={
            "lead": (
                "Подробности релиза и технические характеристики опубликованы по адресу "
                "https://example.com/release в официальной документации компании."
            )
        }
    )
    with pytest.raises(CardValidationError, match="raw URL"):
        validate_grounding(
            draft,
            original_text="Компания сообщила о релизе, веса открыты.",
            known_urls=["https://example.com/release"],
        )


def test_grounding_rejects_numbers_missing_from_cited_facts() -> None:
    draft = valid_draft().model_copy(
        update={
            "lead": valid_draft().lead + " Система выполняет задачу в среднем за 53 шага.",
        }
    )
    with pytest.raises(CardValidationError, match="unsupported numeric value 53"):
        validate_grounding(
            draft,
            original_text="Компания сообщила о релизе, веса открыты.",
            known_urls=["https://example.com/release"],
        )


def test_grounding_keeps_author_radar_out_of_headline_and_lead() -> None:
    evidence_items = [
        {
            "source_name": "Prompt Design",
            "source_role": "radar",
            "published_at": "2026-08-13T10:00:00+00:00",
            "original_text": "Компания выпустила новый инструмент.",
        },
        {
            "source_name": "Crypto Essay",
            "source_role": "author_radar",
            "published_at": "2026-08-13T11:00:00+00:00",
            "original_text": "По оценке автора, инструмент почти на 90% дешевле конкурента.",
        },
    ]
    unattributed = valid_draft().model_copy(update={"lead_evidence_ids": [0, 1]})
    with pytest.raises(CardValidationError, match="cannot be grounded in an author-radar"):
        validate_grounding(
            unattributed,
            original_text=evidence_items[0]["original_text"],
            known_urls=["https://example.com/release"],
            evidence_items=evidence_items,
        )

    unattributed_section = valid_draft().model_copy(
        update={
            "sections": [
                CardSection(
                    kind="statement",
                    text=(
                        "По опубликованной оценке этот инструмент почти на 90% дешевле "
                        "сопоставимого решения конкурента."
                    ),
                    evidence_ids=[1],
                )
            ]
        }
    )
    with pytest.raises(CardValidationError, match="must attribute author evidence"):
        validate_grounding(
            unattributed_section,
            original_text=evidence_items[0]["original_text"],
            known_urls=["https://example.com/release"],
            evidence_items=evidence_items,
            event_type="public_statement",
        )


def test_renderer_has_one_contextual_link_and_escapes_html() -> None:
    draft = valid_draft().model_copy(
        update={
            "lead": (
                "Компания <не копирует HTML> из источника и публикует техническое описание "
                "релиза отдельным документом для разработчиков."
            ),
            "sections": [
                CardSection(
                    kind="benchmarks",
                    text=(
                        "В проверяемом наборе задач инструмент выполнил заявленные сценарии "
                        "без передачи исходного кода внешнему сервису."
                    ),
                    evidence_ids=[0],
                ),
                CardSection(
                    kind="limitations",
                    text=(
                        "Пока опубликовано только описание разработчика без отдельной "
                        "независимой проверки результата."
                    ),
                    evidence_ids=[0],
                ),
            ],
            "highlight": GroundedFact(
                text="Веса модели опубликованы в открытом доступе.",
                evidence_ids=[0],
                kind="metric",
            ),
        }
    )
    rendered = render_card_html(
        draft,
        "https://example.com/release?a=1&b=2",
    )

    assert rendered.count("<a href=") == 1
    assert "DeepSeek</a>" in rendered
    assert "&lt;не копирует HTML&gt;" in rendered
    assert "a=1&amp;b=2" in rendered
    assert "<b>Бенчмарки.</b>" in rendered
    assert "<blockquote>" in rendered
    assert "<i>" not in rendered and "Что важно учитывать:" not in rendered
    assert "без отдельной независимой проверки результата" in rendered
    assert "•" not in rendered
    assert "Авторская аналитика" not in rendered


def test_russian_quality_rejects_known_case_error_and_calque() -> None:
    broken = valid_draft().model_copy(
        update={
            "lead": (
                "Релиз создан с акцентом на долгосрочные агенты и достигает 61 балла "
                "в независимом наборе задач для программных систем."
            )
        }
    )

    errors = russian_quality_errors(broken)

    assert len(errors) == 2
    assert any("incorrect case" in error for error in errors)
    assert any("benchmark calque" in error for error in errors)


def test_russian_quality_rejects_mixed_script_product_name() -> None:
    broken = valid_draft().model_copy(
        update={
            "headline": "Kaspersky сообщила об эксплуатации уязвимости BlueHаm в атаках",
        }
    )

    errors = russian_quality_errors(broken)

    assert any("mixed-script token" in error for error in errors)


def test_russian_quality_rejects_english_quarter_inside_russian_case() -> None:
    broken = valid_draft().model_copy(
        update={
            "headline": "В Q2 2026 года Kaspersky зафиксировала рост числа новых угроз",
        }
    )

    errors = russian_quality_errors(broken)

    assert any("quarter construction" in error for error in errors)


def test_computing_capacity_is_not_mistaken_for_powerful_adjective() -> None:
    draft = valid_draft().model_copy(
        update={"headline": "NVIDIA создаёт платформу финансирования вычислительных мощностей"}
    )

    validate_grounding(
        draft,
        original_text=(
            "NVIDIA создаёт платформу финансирования вычислительных мощностей. "
            "Новая структура объединяет нескольких финансовых партнёров."
        ),
        known_urls=["https://example.com/release"],
    )


def test_russian_normalizer_fixes_quarter_construction() -> None:
    assert normalize_russian_text("В Q2 2026 года модель вышла") == (
        "Во втором квартале 2026 года модель вышла"
    )


def test_russian_normalizer_removes_accidental_repeated_word() -> None:
    assert normalize_russian_text("игры игры получились нестабильными") == (
        "игры получились нестабильными"
    )


def test_russian_normalizer_rewrites_benchmark_calque() -> None:
    assert normalize_russian_text("Модель достигла 28,3 балла") == "Модель набрала 28,3 балла"


def test_grounding_rejects_truncated_headline() -> None:
    draft = valid_draft().model_copy(
        update={"headline": "Anthropic исследовала координацию агентов Sonnet и М"}
    )

    with pytest.raises(CardValidationError, match="appears truncated"):
        validate_grounding(
            draft,
            original_text=(
                "Anthropic исследовала координацию агентов Sonnet и Mythos. "
                "DeepSeek представил новый агентский инструмент с контекстом 500000 токенов. "
                "Система распределяет задачи между несколькими агентами и сохраняет историю."
            ),
            known_urls=["https://example.com/release"],
        )


def test_grounding_rejects_headline_ending_with_preposition() -> None:
    draft = valid_draft().model_copy(
        update={"headline": "DeepSeek представила агентскую систему для работы на"}
    )
    with pytest.raises(CardValidationError, match="dangling function word"):
        validate_grounding(
            draft,
            original_text="DeepSeek представила агентскую систему для работы с кодом.",
            known_urls=["https://example.com/release"],
        )


def test_grounding_rejects_truncated_latin_entity() -> None:
    draft = valid_draft().model_copy(
        update={
            "headline": "Kaspersky зафиксировала эксплуатацию уязвимости BlueHam в атаках",
            "lead": (
                "Исследователи описали активную эксплуатацию уязвимости в операционной системе "
                "и связали её с несколькими зафиксированными атаками."
            ),
            "sections": [
                CardSection(
                    kind="details",
                    text=(
                        "В опубликованном отчёте перечислены наблюдаемые способы эксплуатации "
                        "и рекомендации по установке исправлений."
                    ),
                    evidence_ids=[0],
                )
            ],
        }
    )

    with pytest.raises(CardValidationError, match="unsupported Latin entity BlueHam"):
        validate_grounding(
            draft,
            original_text=(
                "Kaspersky зафиксировала эксплуатацию уязвимости BlueHammer в атаках и "
                "опубликовала рекомендации по установке исправлений."
            ),
            known_urls=["https://example.com/report"],
        )


def test_renderer_does_not_repeat_same_section_label() -> None:
    section_one = CardSection(
        kind="context",
        text="Первый контекстный абзац объясняет происхождение и текущее состояние события.",
        evidence_ids=[0],
    )
    section_two = CardSection(
        kind="context",
        text="Второй абзац добавляет отдельную деталь без повторения визуальной рубрики.",
        evidence_ids=[0],
    )
    draft = valid_draft().model_copy(update={"sections": [section_one, section_two]})

    rendered = render_card_html(draft, "https://example.com/release")

    assert rendered.count("<b>Контекст.</b>") == 1


def test_redundant_text_detects_repeated_fact_but_keeps_added_detail() -> None:
    lead = (
        "Дорожная карта Ethereum предусматривает семь форков к 2029 году, примерно по одному "
        "форку каждые шесть месяцев, и будет обновляться не реже раза в квартал."
    )
    repeated = (
        "Документ намечает семь форков к 2029 году с частотой один форк каждые шесть месяцев "
        "и ежеквартальными обновлениями."
    )
    expanded = (
        "Квантовая устойчивость и конфиденциальность подняты в списке приоритетов, а часть "
        "прежних направлений развития EVM получила меньший приоритет."
    )

    assert is_redundant_text(repeated, [lead])
    assert not is_redundant_text(expanded, [lead])


def test_primary_urls_prefer_official_domains() -> None:
    event = {
        "outbound_urls_json": (
            '["https://dslab.tech/ai/models/grok","https://x.ai/news/grok-4-6"]'
        ),
        "source_url": "https://t.me/channel/1",
    }

    assert unique_known_urls(event) == [
        "https://x.ai/news/grok-4-6",
        "https://dslab.tech/ai/models/grok",
        "https://t.me/channel/1",
    ]


def test_unrelated_github_repo_is_not_treated_as_primary() -> None:
    event = {
        "headline": "DeepSeek выпустили Harness и V4 Pro",
        "original_text": "DeepSeek Harness стал модульным",
        "outbound_urls_json": (
            '["https://github.com/cordiverse/paper",'
            '"https://github.com/deepseek-ai/deepseek-harness"]'
        ),
        "source_url": "https://t.me/channel/1",
    }

    assert unique_known_urls(event)[0] == "https://github.com/deepseek-ai/deepseek-harness"


def test_official_feed_entry_remains_primary_over_links_inside_article() -> None:
    assert (
        preferred_primary_url(
            "https://openai.com/index/release",
            ["https://github.com/example/supporting-code"],
            source_kind="rss",
            source_role="primary",
        )
        == "https://openai.com/index/release"
    )

    event = {
        "source_role": "primary",
        "headline": "Qwen Code release",
        "original_text": "Agent plugins",
        "outbound_urls_json": '["https://github.com/QwenLM/qwen-code/pull/8834"]',
        "source_url": "https://github.com/QwenLM/qwen-code/releases/tag/v0.21.11",
    }
    assert unique_known_urls(event)[0] == event["source_url"]


def test_link_phrase_is_programmatic() -> None:
    assert choose_link_phrase("Вышел Grok 4.6 с новыми агентами") == "Grok 4.6"
    assert choose_link_phrase("Группа Head Mare провела новую атаку") == "Head Mare"
    assert choose_link_phrase("APT-группа Head Mare провела новую атаку") == "Head Mare"
    assert choose_link_phrase("«Сбер» предложил создать совет") == "Сбер"
    assert (
        choose_link_phrase(
            "Во втором квартале 2026 года обнаружено 2538 новых вариантов программ-вымогателей"
        )
        == "2538 новых вариантов программ-вымогателей"
    )


def test_article_renderer_uses_native_rich_blocks_and_semantic_link() -> None:
    draft = valid_draft().model_copy(
        update={
            "highlight": GroundedFact(
                text="Ключевое сравнение подтверждено отдельным измерением.",
                evidence_ids=[0],
                kind="event",
            ),
            "sections": [
                *valid_draft().sections,
                CardSection(
                    kind="limitations",
                    text=(
                        "Для самостоятельного запуска требуется отдельная инфраструктура, "
                        "а управляемый облачный вариант пока не заявлен."
                    ),
                    evidence_ids=[0],
                ),
            ],
        }
    )

    rendered = render_article_rich_html(draft, "https://deepseek.com/release")

    assert rendered.startswith("<h1>")
    assert '<a href="https://deepseek.com/release">DeepSeek</a>' in rendered
    assert "<aside>Ключевое сравнение подтверждено" in rendered
    assert "<h2>Что изменилось</h2>" in rendered
    assert "<h2>Ограничения</h2>" in rendered
    assert "Источник" not in rendered


def test_delivery_adapter_keeps_regular_and_article_transports_separate() -> None:
    class FakeBot:
        def __init__(self) -> None:
            self.regular: dict | None = None
            self.article: dict | None = None

        async def send_message(self, **kwargs):
            self.regular = kwargs
            return SimpleNamespace(message_id=1)

        async def send_rich_message(self, **kwargs):
            self.article = kwargs
            return SimpleNamespace(message_id=2)

    draft = valid_draft()
    card = {
        "structured_output_json": draft.model_dump_json(),
        "rendered_html": render_card_html(draft, "https://deepseek.com/release"),
        "primary_preview_url": "https://deepseek.com/release",
    }
    bot = FakeBot()

    asyncio.run(
        send_formatted_card(
            bot,
            chat_id=123,
            card=card,
            delivery_id=7,
            delivery_format=DeliveryFormat.REGULAR,
        )
    )
    assert bot.regular is not None
    assert bot.article is None
    from app.telegram_bot.news_depth import brief_card_html

    assert bot.regular["text"] == brief_card_html(card)
    assert len(bot.regular["text"]) < len(card["rendered_html"])
    assert bot.regular["link_preview_options"].url == card["primary_preview_url"]

    asyncio.run(
        send_formatted_card(
            bot,
            chat_id=123,
            card=card,
            delivery_id=8,
            delivery_format=DeliveryFormat.ARTICLE,
        )
    )
    assert bot.article is not None
    assert bot.article["rich_message"].html.startswith("<h1>")
    assert bot.article["rich_message"].skip_entity_detection is True


def test_candidate_preparation_rejects_history_and_deduplicates_urls(tmp_path: Path) -> None:
    settings = Settings(
        app_env="test",
        data_dir=tmp_path,
        sources_file=PROJECT_ROOT / "tests/fixtures/config/sources.yaml",
        interests_file=PROJECT_ROOT / "tests/fixtures/config/interests.yaml",
        telegram_session_path=tmp_path / "telegram-reader.session",
    )
    repository = Repository(settings.database_path)
    repository.migrate()
    registry = load_source_registry(settings.sources_file)
    repository.sync_sources(registry)
    published_at = datetime.now(UTC).isoformat()
    source_ids = [
        "telegram_prompt_design",
        "telegram_ai_newz",
        "telegram_test_history",
    ]
    for index, source_id in enumerate(source_ids):
        repository.insert_source_item(
            source_id=source_id,
            external_id=str(100 + index),
            source_url=f"https://t.me/example/{100 + index}",
            original_text="Содержательный технологический текст " * 8,
            original_language="ru",
            published_at=published_at,
            outbound_urls=["https://example.com/same-release"],
            media_metadata=None,
            payload_hash=f"hash-{index}",
            raw_payload={},
        )

    stats = prepare_candidates(settings, repository)

    assert stats.items_seen == 3
    assert stats.events_created == 1
    assert stats.items_attached_to_existing == 1
    assert stats.items_rejected == 1
    events = repository.events_without_cards()
    assert len(events) == 1
    assert events[0]["score"] > 0


def test_source_keyword_filters_keep_signal_and_reject_feed_noise(tmp_path: Path) -> None:
    settings = Settings(
        app_env="test",
        data_dir=tmp_path,
        sources_file=PROJECT_ROOT / "tests/fixtures/config/sources.yaml",
        interests_file=PROJECT_ROOT / "tests/fixtures/config/interests.yaml",
        telegram_session_path=tmp_path / "telegram-reader.session",
    )
    repository = Repository(settings.database_path)
    repository.migrate()
    repository.sync_sources(load_source_registry(settings.sources_file))
    published_at = datetime.now(UTC).isoformat()
    repository.insert_source_item(
        source_id="rss_nvidia_blog",
        external_id="gaming",
        source_url="https://blogs.nvidia.com/blog/geforce-now-game-release/",
        original_text=(
            "GeForce NOW game release expands the cloud gaming catalog for players. " * 3
        ),
        original_language="en",
        published_at=published_at,
        outbound_urls=[],
        media_metadata=None,
        payload_hash="gaming-hash",
        raw_payload={},
    )
    repository.insert_source_item(
        source_id="rss_nvidia_blog",
        external_id="inference",
        source_url="https://blogs.nvidia.com/blog/new-inference-chip/",
        original_text=(
            "NVIDIA introduced a new AI inference chip for data center workloads and agents. " * 3
        ),
        original_language="en",
        published_at=published_at,
        outbound_urls=[],
        media_metadata=None,
        payload_hash="inference-hash",
        raw_payload={},
    )

    stats = prepare_candidates(settings, repository)

    assert stats.items_seen == 2
    assert stats.events_created == 1
    assert stats.items_rejected == 1


def test_daily_candidate_cap_prevents_one_source_from_flooding_queue(tmp_path: Path) -> None:
    settings = Settings(
        app_env="test",
        data_dir=tmp_path,
        sources_file=PROJECT_ROOT / "tests/fixtures/config/sources.yaml",
        interests_file=PROJECT_ROOT / "tests/fixtures/config/interests.yaml",
        telegram_session_path=tmp_path / "telegram-reader.session",
    )
    repository = Repository(settings.database_path)
    repository.migrate()
    repository.sync_sources(load_source_registry(settings.sources_file))
    published_at = datetime.now(UTC).isoformat()
    for index in range(7):
        repository.insert_source_item(
            source_id="rss_securelist",
            external_id=f"report-{index}",
            source_url=f"https://securelist.com/report/{index}/",
            original_text=(
                f"Kaspersky reports a distinct cybersecurity campaign number {index}. " * 3
            ),
            original_language="en",
            published_at=published_at,
            outbound_urls=[],
            media_metadata=None,
            payload_hash=f"report-hash-{index}",
            raw_payload={},
        )

    stats = prepare_candidates(settings, repository)

    assert stats.events_created == 2
    assert stats.items_rejected == 5
