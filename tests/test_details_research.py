from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from app.config import Settings
from app.details.models import (
    DeepArticleDraft,
    DeepArticleParagraph,
    DeepArticleSection,
    DeepSourceAnchor,
    GeneratedDeepArticleDraft,
)
from app.details.renderer import render_detail_article_html
from app.details.research import (
    BoundedDetailResearchService,
    DetailBudgetError,
    DetailEvidenceError,
    DetailResearchPlanner,
    OpenRouterDeepArticleGenerator,
    _unsupported_latin_entities,
    build_deep_article_request,
    build_detail_fact_snapshot,
    compact_generated_deep_article_payload,
    document_content_kind,
    evaluate_detail_source_gate,
    generate_deep_article,
    repair_deep_article_metadata,
    select_detail_documents,
    validate_deep_article,
)
from app.llm.openrouter import OpenRouterResponseError
from app.research.facts import (
    EvidenceQuote,
    FactCoverage,
    FactSheetDraft,
    FactSheetGeneration,
    ResearchFact,
    select_research_documents,
)
from app.research.web import ScrapeResponse, SearchResponse, WebResearchError


def document(
    document_id: int,
    url: str,
    source_class: str,
    content: str,
    *,
    kind: str = "scraped_fulltext",
) -> dict:
    return {
        "id": document_id,
        "canonical_url": url,
        "url": url,
        "title": f"Document {document_id}",
        "publisher": url.split("/")[2],
        "source_class": source_class,
        "content_text": content,
        "metadata_json": f'{{"content_kind":"{kind}"}}',
        "relevance": 1.0,
    }


def test_search_snippet_never_passes_full_text_source_gate() -> None:
    snippet = document(
        1,
        "https://official.example/release",
        "primary",
        "A very long search description. " * 30,
        kind="search_snippet",
    )
    independent = document(
        2,
        "https://news.example/report",
        "independent",
        "An independently scraped report. " * 30,
    )

    gate = evaluate_detail_source_gate([snippet, independent])

    assert document_content_kind(snippet) == "search_snippet"
    assert gate.valid is False
    assert gate.eligible_document_ids == [2]


def test_regular_fact_selection_also_excludes_explicit_search_snippets() -> None:
    snippet = document(
        1,
        "https://official.example/release",
        "primary",
        "A long search description which must not become evidence. " * 30,
        kind="search_snippet",
    )
    full_text = document(
        2,
        "https://news.example/report",
        "independent",
        "A fully scraped independent report with evidence. " * 30,
    )

    selected = select_research_documents([snippet, full_text], event_headline="Example report")

    assert [item["id"] for item in selected] == [2]


def test_source_gate_rejects_syndicated_copy_on_a_second_domain() -> None:
    same_body = "The exact same syndicated body with facts and figures. " * 30
    documents = [
        document(1, "https://official.example/release", "primary", same_body),
        document(2, "https://copy.example/story", "independent", same_body),
    ]

    gate = evaluate_detail_source_gate(documents)

    assert gate.valid is False
    assert gate.eligible_document_ids == [1]


def test_source_gate_counts_organization_domains_not_subdomains() -> None:
    documents = [
        document(1, "https://news.example.com/release", "primary", "Official facts. " * 40),
        document(
            2,
            "https://blog.example.com/analysis",
            "independent",
            "Different analysis from the same organization. " * 40,
        ),
    ]

    gate = evaluate_detail_source_gate(documents)

    assert gate.valid is False
    assert gate.distinct_domains == ["example.com"]


def test_detail_selection_reserves_independent_source_across_primary_subdomains() -> None:
    documents = [
        document(
            index,
            f"https://section-{index}.vendor.example/release",
            "primary",
            f"Distinct official page number {index}. " * 40,
        )
        for index in range(1, 5)
    ]
    documents.append(
        document(
            5,
            "https://reuters.com/technology/report",
            "independent",
            "Independent reporting with a distinct account. " * 40,
        )
    )

    selected = select_detail_documents(
        documents,
        event={"headline": "Vendor released a new product"},
        limit=4,
    )

    assert 5 in {int(item["id"]) for item in selected}
    assert evaluate_detail_source_gate(selected).valid is True


def test_event_evidence_role_overrides_global_document_class() -> None:
    primary = document(
        1,
        "https://unknown-vendor.example/release",
        "corroborating",
        "Official product release. " * 40,
    )
    primary["evidence_role"] = "primary"
    independent = document(
        2,
        "https://news.example/report",
        "independent",
        "Independent reporting with different facts. " * 40,
    )

    gate = evaluate_detail_source_gate([primary, independent])

    assert gate.valid is True
    assert gate.primary_document_ids == [1]


def test_cached_unrelated_unknown_pages_do_not_fill_the_article_window():
    documents = [
        document(1, "https://magentic.com/launch", "primary", "Magentic launches. " * 40),
        document(
            2,
            "https://thenextweb.com/news/magentic",
            "corroborating",
            "Magentic independent report. " * 40,
        ),
        document(
            3,
            "https://clinic.example/diabetes",
            "corroborating",
            "Diabetes treatment unrelated to the company. " * 40,
        ),
    ]
    selected = select_detail_documents(documents, event={"headline": "Magentic launches"})
    assert {item["id"] for item in selected} == {1, 2}
    assert evaluate_detail_source_gate(selected).valid


def test_high_risk_event_requires_two_independent_domains() -> None:
    documents = [
        document(1, "https://government.example/order", "primary", "Official order. " * 40),
        document(2, "https://wire.example/story", "independent", "Independent report. " * 40),
    ]

    gate = evaluate_detail_source_gate(
        documents,
        event={"headline": "Правительство ввело новые санкции"},
    )

    assert gate.valid is False
    assert "high-risk event requires two independent full-text domains" in gate.errors


def test_detail_planner_is_bounded_to_two_queries() -> None:
    plan = DetailResearchPlanner().plan(
        {"headline": "YC профинансировал новую платформу для строительных стартапов"},
        [],
    )

    assert plan.contour == "Стартапы"
    assert 1 <= len(plan.queries) <= 2
    assert "первичный или официальный источник" in plan.gaps


def test_provider_article_schema_requires_nullable_optional_blocks() -> None:
    schema = GeneratedDeepArticleDraft.model_json_schema()

    assert "highlight" in schema["required"]
    assert "limitations" in schema["required"]


def test_generated_article_compacts_unique_overlong_anchor_without_changing_source() -> None:
    payload = valid_article().model_dump()
    payload["source_anchors"][0]["phrase"] = payload["lead"]

    compacted = compact_generated_deep_article_payload(payload)
    typed = GeneratedDeepArticleDraft.model_validate(compacted)

    assert len(typed.source_anchors[0].phrase) <= 120
    assert payload["lead"].count(typed.source_anchors[0].phrase) == 1
    assert typed.source_anchors[0].fact_id == 2
    assert typed.source_anchors[0].document_id == 1


def test_versioned_latin_entity_supports_only_its_unversioned_family() -> None:
    cited = "Z.ai выпустила GLM-5.3 для программирования."

    assert _unsupported_latin_entities("GLM стала доступна через API", cited) == set()
    assert _unsupported_latin_entities("GLM-6 стала доступна через API", cited) == {"glm-6"}


class MemoryRepository:
    def __init__(self, documents: list[dict] | None = None) -> None:
        self.event = {
            "id": 7,
            "headline": "OpenAI выпустила Atlas 2 для корпоративной автоматизации",
            "event_type": "software_release",
            "source_role": "radar",
            "source_url": "https://t.me/radar/7",
            "topics_json": "[]",
            "regions_json": "[]",
        }
        self.documents = deepcopy(documents or [])
        self.usage: list[tuple] = []
        self.queries: list[dict] = []
        self._next_id = max([int(item["id"]) for item in self.documents] or [0]) + 1

    def event_context(self, event_id):
        return deepcopy(self.event) if event_id == 7 else None

    def event_research_documents(self, event_id):
        return deepcopy(self.documents)

    def event_evidence_urls(self, event_id):
        return []

    def latest_valid_fact_sheet(self, event_id):
        return None

    def current_month_cost(self):
        return 0.0

    def record_research_query(self, **kwargs):
        self.queries.append(kwargs)

    def record_usage(self, *args, **kwargs):
        self.usage.append((args, kwargs))

    def upsert_research_document(self, **kwargs):
        existing = next(
            (item for item in self.documents if item["canonical_url"] == kwargs["canonical_url"]),
            None,
        )
        payload = {
            **kwargs,
            "id": int(existing["id"]) if existing else self._next_id,
            "metadata_json": __import__("json").dumps(kwargs.get("metadata") or {}),
            "relevance": 1.0,
        }
        payload.pop("metadata", None)
        if existing:
            self.documents[self.documents.index(existing)] = payload
        else:
            self.documents.append(payload)
            self._next_id += 1
        return int(payload["id"])

    def attach_research_document(self, **kwargs):
        return None


class FirstSearchSucceedsClient:
    def __init__(self) -> None:
        self.search_calls = 0
        self.scrape_calls = 0

    def search(self, query, *, freshness, num_results):
        self.search_calls += 1
        return SearchResponse(
            query=query,
            results=[
                {
                    "url": "https://openai.com/release",
                    "title": "OpenAI Atlas 2 official release",
                    "description": "Official snippet",
                    "relevance": "high",
                },
                {
                    "url": "https://reuters.com/report",
                    "title": "OpenAI Atlas 2 independent report",
                    "description": "Independent snippet",
                    "relevance": "high",
                },
            ],
            cost_usd=0.00018,
        )

    def scrape(self, url):
        self.scrape_calls += 1
        return ScrapeResponse(
            url=url,
            markdown=("Full official page. " if "openai.com" in url else "Full report. ") * 40,
            metadata={"canonicalUrl": url, "title": "Page"},
            cost_usd=0.0009,
        )


def test_bounded_research_does_not_run_second_search_after_gate_passes(tmp_path: Path) -> None:
    settings = Settings(app_env="test", data_dir=tmp_path)
    repository = MemoryRepository()
    client = FirstSearchSucceedsClient()

    result = BoundedDetailResearchService(
        settings,
        repository,  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
    ).collect(7)

    assert result.gate.valid is True
    assert result.search_count == 1
    assert client.search_calls == 1
    assert client.scrape_calls == 2


def test_bounded_research_prefers_new_organization_domain_over_primary_subdomain(
    tmp_path: Path,
) -> None:
    settings = Settings(app_env="test", data_dir=tmp_path, details_max_scrapes=1)
    repository = MemoryRepository(
        [
            document(
                1,
                "https://news.openai.com/release",
                "primary",
                "Existing official full-text release. " * 40,
            )
        ]
    )

    class DomainAwareClient:
        def __init__(self) -> None:
            self.scrape_calls: list[str] = []

        def search(self, query, *, freshness, num_results):
            return SearchResponse(
                query=query,
                results=[
                    {
                        "url": "https://blog.openai.com/more",
                        "title": "OpenAI Atlas 2: more official context",
                        "description": "Official snippet",
                        "relevance": "high",
                    },
                    {
                        "url": "https://reuters.com/technology/report",
                        "title": "OpenAI Atlas 2 independent report",
                        "description": "Independent snippet",
                        "relevance": "high",
                    },
                ],
                cost_usd=0.00018,
            )

        def scrape(self, url):
            self.scrape_calls.append(url)
            return ScrapeResponse(
                url=url,
                markdown="Independent full-text report with distinct evidence. " * 40,
                metadata={"canonicalUrl": url, "title": "Report"},
                cost_usd=0.0009,
            )

    client = DomainAwareClient()
    result = BoundedDetailResearchService(
        settings,
        repository,  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
    ).collect(7)

    assert result.gate.valid is True
    assert client.scrape_calls == ["https://reuters.com/technology/report"]


def test_failed_scrapes_still_consume_the_hard_attempt_limit(tmp_path: Path) -> None:
    settings = Settings(app_env="test", data_dir=tmp_path, details_max_scrapes=2)
    repository = MemoryRepository()

    class AlwaysFailingClient:
        def __init__(self) -> None:
            self.scrape_calls = 0

        def search(self, query, *, freshness, num_results):
            return SearchResponse(
                query=query,
                results=[
                    {
                        "url": f"https://source-{index}.example/report",
                        "title": f"OpenAI Atlas 2 source {index}",
                        "description": "Search description",
                        "relevance": "high",
                    }
                    for index in range(6)
                ],
                cost_usd=0.00054,
            )

        def scrape(self, url):
            self.scrape_calls += 1
            raise WebResearchError(f"could not scrape {url}")

    client = AlwaysFailingClient()
    with pytest.raises(DetailEvidenceError):
        BoundedDetailResearchService(
            settings,
            repository,  # type: ignore[arg-type]
            client=client,  # type: ignore[arg-type]
        ).collect(7)

    assert client.scrape_calls == 2


def fact_documents() -> list[dict]:
    primary_facts = [
        "OpenAI выпустила Atlas 2 для корпоративной автоматизации.",
        "Atlas 2 поддерживает обработку документов и вызов инструментов.",
    ]
    independent_facts = [
        "Независимый тест измерил точность Atlas 2 на уровне 72 процентов.",
        "В независимом тесте Atlas 2 допускала ошибки на длинных документах.",
    ]
    return [
        document(1, "https://openai.com/atlas", "primary", "\n".join(primary_facts) * 20),
        document(
            2,
            "https://wire.example/atlas",
            "independent",
            "\n".join(independent_facts) * 20,
        ),
    ]


def fact_sheet() -> FactSheetDraft:
    rows = [
        (1, "OpenAI выпустила Atlas 2 для корпоративной автоматизации.", "release"),
        (1, "Atlas 2 поддерживает обработку документов и вызов инструментов.", "capability"),
        (2, "Независимый тест измерил точность Atlas 2 на уровне 72 процентов.", "benchmark"),
        (2, "В независимом тесте Atlas 2 допускала ошибки на длинных документах.", "limitation"),
    ]
    return FactSheetDraft(
        event_type="software_release",
        facts=[
            ResearchFact(
                text=text,
                document_ids=[document_id],
                evidence_quotes=[EvidenceQuote(document_id=document_id, quote=text)],
                kind=kind,
            )
            for document_id, text, kind in rows
        ],
        coverage=FactCoverage(
            event_identity=True,
            capabilities=True,
            benchmarks=True,
            pricing=False,
            availability=False,
            limitations=True,
            independent_confirmation=True,
        ),
        conflicts=[],
    )


class FactGenerator:
    def generate(self, *, event, documents, model):
        return FactSheetGeneration(
            draft=fact_sheet(),
            model=model,
            cost_usd=0.001,
            prompt_tokens=100,
            completion_tokens=100,
            response_id="fact-response",
        )


def test_fact_fallback_is_not_started_without_reserved_request_budget(tmp_path: Path) -> None:
    settings = Settings(
        app_env="test",
        data_dir=tmp_path,
        openrouter_model="cheap/primary",
        openrouter_fallback_model="cheap/fallback",
    )
    repository = MemoryRepository(fact_documents())

    class CostlyInvalidGenerator:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, *, event, documents, model):
            self.calls += 1
            raise OpenRouterResponseError(
                "invalid JSON",
                model=model,
                cost_usd=0.006,
                prompt_tokens=1000,
                completion_tokens=500,
                response_id="invalid-fact",
            )

    generator = CostlyInvalidGenerator()
    with pytest.raises(DetailBudgetError) as raised:
        build_detail_fact_snapshot(
            settings,
            repository,  # type: ignore[arg-type]
            7,
            fact_documents(),
            generator=generator,  # type: ignore[arg-type]
            max_cost_usd=0.01,
        )

    assert generator.calls == 1
    assert raised.value.cost_usd == 0.006


def test_detail_fact_snapshot_does_not_mutate_event_status(tmp_path: Path) -> None:
    settings = Settings(app_env="test", data_dir=tmp_path)
    repository = MemoryRepository(fact_documents())
    before = deepcopy(repository.event)

    result = build_detail_fact_snapshot(
        settings,
        repository,  # type: ignore[arg-type]
        7,
        fact_documents(),
        generator=FactGenerator(),  # type: ignore[arg-type]
    )

    assert len(result.snapshot.facts) == 4
    assert result.snapshot.document_ids == [1, 2]
    assert repository.event == before


def valid_article() -> DeepArticleDraft:
    lead = (
        "OpenAI представила Atlas 2 для корпоративной автоматизации: модель обрабатывает "
        "документы и вызывает инструменты, а независимая проверка добавляет измеримый ориентир."
    )
    sections = [
        DeepArticleSection(
            heading="Что именно выпустили",
            paragraphs=[
                DeepArticleParagraph(
                    text=(
                        "Платформа рассчитана на работу с корпоративными документами и может "
                        "вызывать внешние инструменты внутри одного рабочего процесса. "
                        "Это две прямо заявленные возможности Atlas 2 без дополнительных выводов."
                    ),
                    fact_ids=[1, 2],
                ),
                DeepArticleParagraph(
                    text=(
                        "В официальном сообщении продукт описан как инструмент корпоративной "
                        "автоматизации; других сценариев подтверждённая факт-карта не добавляет."
                    ),
                    fact_ids=[1],
                ),
            ],
        ),
        DeepArticleSection(
            heading="Что показала внешняя проверка",
            paragraphs=[
                DeepArticleParagraph(
                    text=(
                        "Независимый тест оценил точность Atlas 2 в 72 процента. Результат даёт "
                        "конкретную метрику для сравнения, но сам по себе не описывает все режимы "
                        "работы продукта и не отменяет наблюдавшиеся ограничения."
                    ),
                    fact_ids=[3, 4],
                ),
                DeepArticleParagraph(
                    text=(
                        "Проверяющие также зафиксировали ошибки на длинных документах. Поэтому "
                        "результат испытания нельзя переносить на любой корпоративный процесс "
                        "без дополнительной проверки на собственных данных."
                    ),
                    fact_ids=[4],
                ),
            ],
        ),
    ]
    return DeepArticleDraft(
        headline="OpenAI выпустила Atlas 2 для корпоративной автоматизации",
        headline_fact_ids=[1],
        lead=lead,
        lead_fact_ids=[1, 2, 3],
        sections=sections,
        highlight=None,
        limitations=DeepArticleParagraph(
            text=(
                "В независимом тесте модель допускала ошибки на длинных документах; другие "
                "ограничения доступная факт-карта не подтверждает."
            ),
            fact_ids=[4],
        ),
        source_anchors=[
            DeepSourceAnchor(
                phrase="модель обрабатывает",
                fact_id=2,
                document_id=1,
                role="primary",
            ),
            DeepSourceAnchor(
                phrase="Независимый тест",
                fact_id=3,
                document_id=2,
                role="independent",
            ),
        ],
        used_document_ids=[1, 2],
    )


@pytest.fixture
def deep_generator_context(tmp_path):
    settings = Settings(
        _env_file=None, app_env="test", data_dir=tmp_path, openrouter_api_key="fake-key"
    )
    docs = fact_documents()
    repository = MemoryRepository(docs)
    facts = build_detail_fact_snapshot(settings, repository, 7, docs, generator=FactGenerator())
    return settings, repository, facts.snapshot, docs


@pytest.mark.parametrize("model", ["openai/gpt-5.6-luna", "openai/gpt-4.1-mini"])
def test_deep_article_payload_preserves_model_parameter_compatibility(
    deep_generator_context, model
):
    from app.pipeline.telegram_style import NEWS_STYLE_INSTRUCTION

    _, repository, snapshot, docs = deep_generator_context
    payload = build_deep_article_request(
        model=model, event=repository.event, snapshot=snapshot, documents=docs
    )
    assert payload["max_tokens"] == 3200
    assert payload["provider"]["require_parameters"] is True
    assert NEWS_STYLE_INSTRUCTION in payload["messages"][0]["content"]
    if "gpt-5" in model:
        assert "temperature" not in payload
        assert payload["reasoning"] == {"effort": "low", "exclude": True}
        assert payload["provider"]["max_price"] == {"prompt": 0.5, "completion": 2.0}
    else:
        assert payload["temperature"] == 0
        assert "reasoning" not in payload
        assert payload["provider"] == {"require_parameters": True}


@pytest.mark.parametrize("invalid_json", [False, True])
@pytest.mark.parametrize(
    "router_cost,is_byok,upstream_cost,expected",
    [(0, True, 0.004, 0.004), (0.001, True, 0.004, 0.005), (0.002, False, 0.004, 0.002)],
)
def test_deep_article_actual_generator_accounts_byok_and_invalid_responses(
    deep_generator_context, invalid_json, router_cost, is_byok, upstream_cost, expected
):
    settings, repository, snapshot, docs = deep_generator_context
    calls = []

    def transport(payload, key):
        calls.append(payload)
        return {
            "id": "deep-receipt",
            "model": payload["model"],
            "usage": {
                "cost": router_cost,
                "is_byok": is_byok,
                "cost_details": {"upstream_inference_cost": upstream_cost},
                "prompt_tokens": 1000,
                "completion_tokens": 400,
            },
            "choices": [
                {
                    "message": {
                        "content": "broken" if invalid_json else valid_article().model_dump_json()
                    }
                }
            ],
        }

    generator = OpenRouterDeepArticleGenerator(settings, transport=transport)
    kwargs = dict(
        event=repository.event, snapshot=snapshot, documents=docs, model="openai/gpt-5.6-luna"
    )
    if invalid_json:
        with pytest.raises(OpenRouterResponseError) as raised:
            generator.generate(**kwargs)
        assert raised.value.cost_usd == expected
        metadata = raised.value.billing_metadata
    else:
        _, provider = generator.generate(**kwargs)
        assert provider.reported_cost_usd == router_cost
        assert provider.accounted_cost_usd == expected
        metadata = provider.billing_metadata
    assert len(calls) == 1
    assert "temperature" not in calls[0]
    assert metadata["is_byok"] is is_byok
    assert metadata["cost_estimated"] is False


def test_deep_article_byok_accounting_reaches_request_budget_and_ledger(deep_generator_context):
    settings, repository, snapshot, docs = deep_generator_context

    def transport(payload, key):
        return {
            "model": payload["model"],
            "usage": {
                "cost": 0,
                "is_byok": True,
                "cost_details": {"upstream_inference_cost": 0.004},
                "prompt_tokens": 1000,
                "completion_tokens": 400,
            },
            "choices": [{"message": {"content": json.dumps(valid_article().model_dump())}}],
        }

    result = generate_deep_article(
        settings,
        repository,
        7,
        snapshot,
        docs,
        generator=OpenRouterDeepArticleGenerator(settings, transport=transport),
        max_cost_usd=0.02,
    )
    assert result.cost_usd == 0.004
    operation, receipt = repository.usage[-1]
    assert operation == ("openrouter", "details.article")
    assert receipt["cost_usd"] == 0.004
    assert receipt["metadata"]["billing_basis"] == "openrouter_plus_byok_upstream"
    assert receipt["metadata"]["event_id"] == 7


def test_details_full_paragraph_context_is_not_misread_as_release(deep_generator_context):
    _, repository, snapshot, docs = deep_generator_context
    draft = valid_article()
    section = draft.sections[1]
    paragraph = section.paragraphs[0].model_copy(
        update={
            "text": "В представленных материалах приведены результаты проверки. "
            + section.paragraphs[0].text
        }
    )
    draft = draft.model_copy(
        update={
            "sections": [
                draft.sections[0],
                section.model_copy(update={"paragraphs": [paragraph, section.paragraphs[1]]}),
            ]
        }
    )
    assert (
        validate_deep_article(draft, snapshot=snapshot, documents=docs, event=repository.event)
        == []
    )


def test_deep_article_validator_catches_unsupported_number(tmp_path: Path) -> None:
    settings = Settings(app_env="test", data_dir=tmp_path)
    repository = MemoryRepository(fact_documents())
    snapshot = build_detail_fact_snapshot(
        settings,
        repository,  # type: ignore[arg-type]
        7,
        fact_documents(),
        generator=FactGenerator(),  # type: ignore[arg-type]
    ).snapshot
    draft = valid_article()
    corrupted = draft.model_copy(
        update={"headline": "OpenAI выпустила Atlas 2 с точностью 99 процентов"}
    )

    errors = validate_deep_article(corrupted, snapshot=snapshot, documents=fact_documents())

    assert any("unsupported numbers" in error for error in errors)


def test_deep_article_validator_accepts_grounded_multisource_article(tmp_path: Path) -> None:
    settings = Settings(app_env="test", data_dir=tmp_path)
    repository = MemoryRepository(fact_documents())
    snapshot = build_detail_fact_snapshot(
        settings,
        repository,  # type: ignore[arg-type]
        7,
        fact_documents(),
        generator=FactGenerator(),  # type: ignore[arg-type]
    ).snapshot

    errors = validate_deep_article(
        valid_article(),
        snapshot=snapshot,
        documents=fact_documents(),
    )

    assert errors == []


def test_article_metadata_repair_adds_missing_citations_and_derives_anchors(
    tmp_path: Path,
) -> None:
    settings = Settings(app_env="test", data_dir=tmp_path)
    repository = MemoryRepository(fact_documents())
    snapshot = build_detail_fact_snapshot(
        settings,
        repository,  # type: ignore[arg-type]
        7,
        fact_documents(),
        generator=FactGenerator(),  # type: ignore[arg-type]
    ).snapshot
    draft = valid_article()
    broken = draft.model_copy(
        update={
            "lead_fact_ids": [2, 3],
            "source_anchors": [
                anchor.model_copy(update={"phrase": "результат", "role": "context"})
                for anchor in draft.source_anchors
            ],
            "used_document_ids": [1],
        }
    )

    repaired = repair_deep_article_metadata(
        broken,
        snapshot=snapshot,
        documents=fact_documents(),
    )

    assert repaired.lead == broken.lead
    assert 1 in repaired.lead_fact_ids
    assert {anchor.role for anchor in repaired.source_anchors} == {"primary", "independent"}
    assert repaired.used_document_ids == [1, 2]
    # Two documents cited by the lead used to produce shifted, overlapping
    # ten-word phrases: validation passed but rendering dropped one source.
    rendered = render_detail_article_html(repaired, fact_documents())
    assert all(anchor.phrase in rendered for anchor in repaired.source_anchors)
    assert all(f'href="{document["url"]}"' in rendered for document in fact_documents())
    assert (
        validate_deep_article(
            repaired,
            snapshot=snapshot,
            documents=fact_documents(),
        )
        == []
    )


def test_article_validator_rejects_overlapping_source_anchors(tmp_path: Path) -> None:
    settings = Settings(app_env="test", data_dir=tmp_path)
    snapshot = build_detail_fact_snapshot(
        settings,
        MemoryRepository(fact_documents()),  # type: ignore[arg-type]
        7,
        fact_documents(),
        generator=FactGenerator(),  # type: ignore[arg-type]
    ).snapshot
    draft = valid_article()
    overlap = draft.source_anchors[1].model_copy(
        update={"phrase": "обрабатывает документы и вызывает инструменты"}
    )
    draft = draft.model_copy(update={"source_anchors": [draft.source_anchors[0], overlap]})

    errors = validate_deep_article(draft, snapshot=snapshot, documents=fact_documents())

    assert any("overlaps source_anchors[0]" in error for error in errors)


def test_article_validator_preserves_high_risk_source_policy(tmp_path: Path) -> None:
    settings = Settings(app_env="test", data_dir=tmp_path)
    repository = MemoryRepository(fact_documents())
    snapshot = build_detail_fact_snapshot(
        settings,
        repository,  # type: ignore[arg-type]
        7,
        fact_documents(),
        generator=FactGenerator(),  # type: ignore[arg-type]
    ).snapshot

    errors = validate_deep_article(
        valid_article(),
        snapshot=snapshot,
        documents=fact_documents(),
        event={"headline": "Правительство ввело новые санкции"},
    )

    assert "high-risk event requires two independent full-text domains" in errors


def test_article_validator_rejects_new_headline_action_and_wrong_anchor_role(
    tmp_path: Path,
) -> None:
    settings = Settings(app_env="test", data_dir=tmp_path)
    repository = MemoryRepository(fact_documents())
    snapshot = build_detail_fact_snapshot(
        settings,
        repository,  # type: ignore[arg-type]
        7,
        fact_documents(),
        generator=FactGenerator(),  # type: ignore[arg-type]
    ).snapshot
    draft = valid_article()
    anchors = list(draft.source_anchors)
    anchors[0] = anchors[0].model_copy(update={"role": "independent"})
    corrupted = draft.model_copy(
        update={
            "headline": draft.headline.replace("выпустила", "приобрела"),
            "source_anchors": anchors,
        }
    )

    errors = validate_deep_article(
        corrupted,
        snapshot=snapshot,
        documents=fact_documents(),
    )

    assert any("introduces an action" in error for error in errors)
    assert any("role independent does not match primary" in error for error in errors)
