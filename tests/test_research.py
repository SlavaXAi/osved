from pathlib import Path

from app.config import Settings, load_source_registry
from app.db.repository import Repository
from app.research.dossier import build_event_dossier
from app.research.facts import (
    ConflictClaim,
    FactConflict,
    FactCoverage,
    FactSheetDraft,
    FactSheetGeneration,
    GeneratedFactSheetDraft,
    ResearchFact,
    build_event_fact_sheet,
    build_fact_sheet_request,
    compact_generated_evidence_quotes,
    deduplicate_fact_sheet,
    normalize_event_type,
    sanitize_fact_sheet,
    select_research_documents,
    validate_fact_sheet,
)
from app.research.policy import fact_policy_metadata
from app.research.web import (
    ResearchEventService,
    ScrapeResponse,
    SearchResponse,
    classify_source,
    infer_event_primary_url,
)

PROJECT_ROOT = Path(__file__).parents[1]


def test_fact_sheet_schema_keeps_shape_without_provider_state_explosion() -> None:
    payload = build_fact_sheet_request(
        model="google/gemini-2.5-flash-lite",
        event={"headline": "Компания представила ИИ-сервис", "event_at": None},
        documents=[
            {
                "id": 1,
                "source_class": "primary",
                "publisher": "Example",
                "title": "Release",
                "published_at": None,
                "content_text": "Компания представила ИИ-сервис для бизнеса.",
            }
        ],
    )

    schema = payload["response_format"]["json_schema"]["schema"]
    encoded = str(schema)

    assert schema["required"] == [
        "event_type",
        "facts",
        "coverage",
        "conflicts",
        "editorial_hook",
    ]
    assert schema["additionalProperties"] is False
    assert "maxItems" not in encoded
    assert "minItems" not in encoded
    assert "maxLength" not in encoded
    assert "minLength" not in encoded


def test_generated_literal_quotes_are_bounded_locally_after_decoding() -> None:
    long_quote = "Компания представила корпоративную ИИ-платформу. " * 12
    payload = {
        "facts": [
            {
                "text": "Компания представила корпоративную ИИ-платформу.",
                "document_ids": [1],
                "evidence_quotes": [{"document_id": 1, "quote": long_quote}],
                "kind": "release",
            }
        ]
    }

    compacted = compact_generated_evidence_quotes(payload)
    quotes = [item["quote"] for item in compacted["facts"][0]["evidence_quotes"]]

    assert all(len(quote) <= 240 and quote in long_quote for quote in quotes)
    assert "".join(quotes) == long_quote


def test_generated_fact_sheet_drops_structurally_invalid_optional_rows() -> None:
    facts = [
        {
            "text": f"Компания подтвердила проверяемый факт номер {index}.",
            "document_ids": [1],
            "evidence_quotes": [
                {"document_id": 1, "quote": "Компания подтвердила проверяемый факт."}
            ],
            "kind": "release",
        }
        for index in range(3)
    ]
    payload = {
        "event_type": "other",
        "facts": [
            *facts,
            {
                "text": "Строка без какого-либо источника должна быть удалена.",
                "document_ids": [],
                "evidence_quotes": [],
                "kind": "release",
            },
        ],
        "coverage": {
            "event_identity": True,
            "capabilities": False,
            "benchmarks": False,
            "pricing": False,
            "availability": False,
            "limitations": False,
            "independent_confirmation": False,
        },
        "conflicts": [
            {
                "subject": "invalid conflict",
                "severity": "material",
                "claims": [
                    {"text": "Одна сторона заявила одно.", "document_id": 0},
                    {"text": "Другая сторона заявила другое.", "document_id": 1},
                ],
                "resolution": "unresolved",
                "resolved_document_id": None,
            }
        ],
        "editorial_hook": {
            "hook_fact": "",
            "hook_type": "none",
            "document_ids": [],
            "claim_strength": "confirmed_fact",
            "forbidden_strengthenings": [],
        },
    }
    payload["facts"][0]["evidence_quotes"].append(
        {"document_id": 0, "quote": "Невалидная цитата не должна отравить весь ответ."}
    )

    compacted = compact_generated_evidence_quotes(payload)
    parsed = GeneratedFactSheetDraft.model_validate(compacted)

    assert len(parsed.facts) == 3
    assert parsed.conflicts == []
    assert parsed.editorial_hook.hook_fact == facts[0]["text"]


def setup_event(tmp_path: Path) -> tuple[Settings, Repository, dict]:
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
    smoke = repository.create_smoke_delivery(123, "https://example.com/release")
    with repository.connection() as connection:
        connection.execute(
            "UPDATE events SET headline = ? WHERE id = ?",
            (
                "Компания выпустила модель с результатом 70 баллов и ценой 2 доллара",
                smoke["event_id"],
            ),
        )
    return settings, repository, smoke


def test_source_classification_separates_evidence_from_radars() -> None:
    assert classify_source("https://x.ai/news/model") == "primary"
    assert classify_source("https://artificialanalysis.ai/articles/model") == "independent"
    assert classify_source("https://x.com/someone/status/1") == "radar_only"
    assert (
        classify_source(
            "https://example.com/release",
            known_primary_urls={"https://example.com/release"},
        )
        == "primary"
    )


def test_radar_post_can_identify_an_exact_event_brand_domain() -> None:
    event = {
        "headline": "ITGlobal представила корпоративную ИИ-инфраструктуру",
        "source_role": "radar",
        "source_url": "https://t.me/ict_moscow_ai/5047",
    }
    official = "https://itglobal.com/ru-ru/company/news/ai-release"

    assert infer_event_primary_url(event, [official, event["source_url"]]) == official


def test_radar_post_does_not_guess_between_unrelated_external_pages() -> None:
    event = {
        "headline": "Новый корпоративный ИИ-сервис",
        "source_role": "radar",
        "source_url": "https://t.me/example/1",
    }

    assert (
        infer_event_primary_url(
            event,
            ["https://example.org/report", "https://another.example/story"],
        )
        is None
    )


def test_guide_is_not_misclassified_as_model_release() -> None:
    draft = FactSheetDraft(
        event_type="model_release",
        facts=[
            ResearchFact(text="Компания выпустила модель.", document_ids=[1], kind="release"),
            ResearchFact(
                text="Модель поддерживает вызов инструментов.",
                document_ids=[1],
                kind="capability",
            ),
            ResearchFact(text="Модель набрала 70 баллов.", document_ids=[1], kind="benchmark"),
        ],
        coverage=FactCoverage(
            event_identity=False,
            capabilities=False,
            benchmarks=False,
            pricing=False,
            availability=False,
            limitations=False,
            independent_confirmation=False,
        ),
        conflicts=[],
    )

    normalized = normalize_event_type(draft, "The builder's guide to GPT-5.6")

    assert normalized.event_type == "other"


def test_research_document_selection_prefers_event_specific_independent_source() -> None:
    documents = [
        {
            "id": 1,
            "source_class": "primary",
            "title": "GLM-5.3 release",
            "content_text": "GLM-5.3 details " * 20,
            "relevance": 1.0,
        },
        {
            "id": 2,
            "source_class": "independent",
            "title": "Large generic model leaderboard",
            "content_text": "Many unrelated models " * 1000,
            "relevance": 1.0,
        },
        {
            "id": 3,
            "source_class": "independent",
            "title": "Independent GLM-5.3 review",
            "content_text": "GLM-5.3 benchmark confirmation " * 20,
            "relevance": 0.8,
        },
    ]

    selected = select_research_documents(documents, event_headline="Встречайте — GLM 5.3")

    assert [document["id"] for document in selected[:2]] == [1, 3]


def test_research_event_is_bounded_and_persists_documents(tmp_path: Path) -> None:
    settings, repository, smoke = setup_event(tmp_path)

    class FakeClient:
        def search(self, query, *, freshness, num_results):
            assert freshness == "last_week"
            assert num_results == 10
            return SearchResponse(
                query=query,
                results=[
                    {
                        "url": "https://example.com/release",
                        "title": "Official release",
                        "description": "Official model release.",
                        "relevance": "high",
                    },
                    {
                        "url": "https://artificialanalysis.ai/articles/release",
                        "title": "Independent benchmarks",
                        "description": "Independent score and price comparison.",
                        "relevance": "high",
                    },
                ],
                cost_usd=0.00018,
            )

        def scrape(self, url):
            is_primary = url == "https://example.com/release"
            return ScrapeResponse(
                url=url,
                markdown=(
                    "Official release with price and benchmark table."
                    if is_primary
                    else "Independent benchmark confirms the score and comparison."
                ),
                metadata={
                    "canonicalUrl": url,
                    "title": "Official" if is_primary else "Independent",
                    "publishedTime": "2026-08-12",
                },
                cost_usd=0.0009,
            )

    stats = ResearchEventService(
        settings,
        repository,
        client=FakeClient(),
    ).research(
        smoke["event_id"],
        query="model release benchmarks",
        primary_url="https://example.com/release",
        freshness="last_week",
        max_pages=2,
    )

    assert stats.pages_scraped == 2
    assert stats.search_results == 2
    documents = repository.event_research_documents(smoke["event_id"])
    assert {document["source_class"] for document in documents} == {
        "primary",
        "independent",
    }
    assert {document["evidence_role"] for document in documents} == {
        "primary",
        "independent",
    }
    assert repository.current_month_cost() == stats.cost_usd


def valid_fact_sheet(primary_id: int, independent_id: int) -> FactSheetDraft:
    return FactSheetDraft(
        event_type="model_release",
        facts=[
            ResearchFact(
                text="Компания выпустила новую версию модели.",
                document_ids=[primary_id],
                kind="release",
            ),
            ResearchFact(
                text="Цена составляет 2 доллара за миллион входных токенов.",
                document_ids=[primary_id, independent_id],
                kind="pricing",
            ),
            ResearchFact(
                text="Независимый индекс оценил модель в 61 балл.",
                document_ids=[independent_id],
                kind="benchmark",
            ),
            ResearchFact(
                text="Модель доступна через программный интерфейс.",
                document_ids=[primary_id],
                kind="availability",
            ),
        ],
        coverage=FactCoverage(
            event_identity=True,
            capabilities=False,
            benchmarks=True,
            pricing=True,
            availability=True,
            limitations=False,
            independent_confirmation=True,
        ),
        conflicts=[],
    )


def test_fact_sheet_validation_and_persistence(tmp_path: Path) -> None:
    settings, repository, smoke = setup_event(tmp_path)
    primary_id = repository.upsert_research_document(
        canonical_url="https://example.com/release",
        url="https://example.com/release",
        source_class="primary",
        retrieved_at="2026-08-13T00:00:00+00:00",
        expires_at="2099-01-01T00:00:00+00:00",
        content_text=(
            "Official model release with a price of 2 dollars for input and 6 dollars for "
            "output plus API availability. "
        )
        * 2,
    )
    independent_id = repository.upsert_research_document(
        canonical_url="https://artificialanalysis.ai/articles/release",
        url="https://artificialanalysis.ai/articles/release",
        source_class="independent",
        retrieved_at="2026-08-13T00:00:00+00:00",
        expires_at="2099-01-01T00:00:00+00:00",
        content_text="Independent benchmark scored the model at 61 points." * 2,
    )
    for document_id, role in [(primary_id, "primary"), (independent_id, "independent")]:
        repository.attach_research_document(
            event_id=smoke["event_id"],
            research_document_id=document_id,
            evidence_role=role,
        )
    draft, _warnings = sanitize_fact_sheet(
        valid_fact_sheet(primary_id, independent_id),
        repository.event_research_documents(smoke["event_id"]),
    )

    duplicated = draft.model_copy(
        update={
            "facts": [
                *draft.facts,
                ResearchFact(
                    text=(
                        "Цена за миллион входных токенов составляет 2 доллара, а за миллион "
                        "выходных токенов — 6 долларов."
                    ),
                    document_ids=[primary_id],
                    kind="pricing",
                ),
            ]
        }
    )
    assert len(deduplicate_fact_sheet(duplicated).facts) == len(draft.facts)

    documents = repository.event_research_documents(smoke["event_id"])
    assert validate_fact_sheet(draft, documents) == []
    bad = draft.model_copy(
        update={
            "facts": [
                *draft.facts,
                ResearchFact(
                    text="Автор советует планировать карьеру с учётом удешевления ИИ.",
                    document_ids=[independent_id],
                    kind="capability",
                ),
            ]
        }
    )
    assert any("editorial" in error for error in validate_fact_sheet(bad, documents))

    hallucinated = draft.model_copy(
        update={
            "facts": [
                *draft.facts,
                ResearchFact(
                    text="Модель стоит на 90% дешевле конкурента.",
                    document_ids=[independent_id],
                    kind="pricing",
                ),
            ]
        }
    )
    sanitized, warnings = sanitize_fact_sheet(hallucinated, documents)
    assert len(sanitized.facts) == len(draft.facts)
    assert any("unsupported numeric value 90%" in warning for warning in warnings)

    class FakeGenerator:
        def generate(self, *, event, documents, model):
            return FactSheetGeneration(
                draft=draft,
                model=model,
                cost_usd=0.001,
                prompt_tokens=100,
                completion_tokens=50,
                response_id="fact-sheet-1",
            )

    result = build_event_fact_sheet(
        settings,
        repository,
        smoke["event_id"],
        generator=FakeGenerator(),
    )

    assert result["status"] == "valid"
    assert result["fact_count"] == 4
    assert result["dossier"]["decision"] == "ready"
    assert repository.latest_valid_fact_sheet(smoke["event_id"]) is not None
    dossier = repository.latest_event_dossier(smoke["event_id"])
    assert dossier is not None
    assert dossier["payload"]["coverage"]["missing_required"] == []
    assert repository.event_context(smoke["event_id"])["status"] == "test"


def test_model_release_without_announced_price_can_still_be_ready(tmp_path: Path) -> None:
    _, repository, smoke = setup_event(tmp_path)
    primary_id = repository.upsert_research_document(
        canonical_url="https://example.com/model",
        url="https://example.com/model",
        source_class="primary",
        retrieved_at="2026-08-14T00:00:00+00:00",
        expires_at="2099-01-01T00:00:00+00:00",
        content_text="The model scored 70 points and weights will be released in two weeks." * 3,
    )
    independent_id = repository.upsert_research_document(
        canonical_url="https://unite.ai/model-review",
        url="https://unite.ai/model-review",
        source_class="independent",
        retrieved_at="2026-08-14T00:00:00+00:00",
        expires_at="2099-01-01T00:00:00+00:00",
        content_text="Independent reporting confirms the 70-point benchmark result." * 3,
    )
    for document_id, role in [(primary_id, "primary"), (independent_id, "independent")]:
        repository.attach_research_document(
            event_id=smoke["event_id"],
            research_document_id=document_id,
            evidence_role=role,
        )
    draft = FactSheetDraft(
        event_type="model_release",
        facts=[
            ResearchFact(
                text="Компания выпустила новую версию модели.",
                document_ids=[primary_id],
                kind="release",
            ),
            ResearchFact(
                text="Модель набрала 70 баллов в заявленном тесте.",
                document_ids=[primary_id, independent_id],
                kind="benchmark",
            ),
            ResearchFact(
                text="Веса модели обещают опубликовать через две недели.",
                document_ids=[primary_id],
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

    documents = repository.event_research_documents(smoke["event_id"])
    draft, _warnings = sanitize_fact_sheet(draft, documents)
    assert validate_fact_sheet(draft, documents) == []


def test_primary_only_model_release_can_enter_calibration(tmp_path: Path) -> None:
    _, repository, smoke = setup_event(tmp_path)
    primary_id = repository.upsert_research_document(
        canonical_url="https://example.com/model-primary",
        url="https://example.com/model-primary",
        source_class="primary",
        retrieved_at="2026-08-15T00:00:00+00:00",
        expires_at="2099-01-01T00:00:00+00:00",
        content_text="The model scored 70 points and is available through an API." * 3,
    )
    repository.attach_research_document(
        event_id=smoke["event_id"],
        research_document_id=primary_id,
        evidence_role="primary",
    )
    draft = FactSheetDraft(
        event_type="model_release",
        facts=[
            ResearchFact(
                text="Компания выпустила новую модель.", document_ids=[primary_id], kind="release"
            ),
            ResearchFact(
                text="Модель набрала 70 баллов.", document_ids=[primary_id], kind="benchmark"
            ),
            ResearchFact(
                text="Модель доступна через API.", document_ids=[primary_id], kind="availability"
            ),
        ],
        coverage=FactCoverage(
            event_identity=True,
            capabilities=False,
            benchmarks=True,
            pricing=False,
            availability=True,
            limitations=False,
            independent_confirmation=False,
        ),
        conflicts=[],
    )

    documents = repository.event_research_documents(smoke["event_id"])
    draft, _warnings = sanitize_fact_sheet(draft, documents)
    assert validate_fact_sheet(draft, documents) == []


def test_event_dossier_blocks_unresolved_material_conflict(tmp_path: Path) -> None:
    _, repository, smoke = setup_event(tmp_path)
    primary_id = repository.upsert_research_document(
        canonical_url="https://example.com/release",
        url="https://example.com/release",
        source_class="primary",
        retrieved_at="2026-08-13T00:00:00+00:00",
        expires_at="2099-01-01T00:00:00+00:00",
        content_text="Official price is 2 dollars per million tokens." * 4,
    )
    independent_id = repository.upsert_research_document(
        canonical_url="https://artificialanalysis.ai/release",
        url="https://artificialanalysis.ai/release",
        source_class="independent",
        retrieved_at="2026-08-13T00:00:00+00:00",
        expires_at="2099-01-01T00:00:00+00:00",
        content_text="Measured price is 3 dollars per million tokens." * 4,
    )
    for document_id, role in [(primary_id, "primary"), (independent_id, "independent")]:
        repository.attach_research_document(
            event_id=smoke["event_id"],
            research_document_id=document_id,
            evidence_role=role,
        )
    conflict = FactConflict(
        subject="input token price",
        severity="material",
        claims=[
            ConflictClaim(text="Official price is 2 dollars.", document_id=primary_id),
            ConflictClaim(text="Measured price is 3 dollars.", document_id=independent_id),
        ],
        resolution="unresolved",
        resolved_document_id=None,
    )
    repository.create_fact_sheet(
        event_id=smoke["event_id"],
        schema_version="neutral_fact_sheet_v2",
        facts=[
            {
                "text": "The company released a model.",
                "kind": "release",
                "document_ids": [primary_id],
            },
            {
                "text": "The model is available through an API.",
                "kind": "availability",
                "document_ids": [primary_id],
            },
            {
                "text": "The benchmark score is 61 points.",
                "kind": "benchmark",
                "document_ids": [independent_id],
            },
        ],
        coverage={
            **fact_policy_metadata(),
            "event_type": "other",
            "event_identity": True,
            "capabilities": False,
            "benchmarks": True,
            "pricing": True,
            "availability": True,
            "limitations": False,
            "independent_confirmation": True,
            "conflicts": [conflict.model_dump()],
        },
        validation_status="valid",
        expires_at="2099-01-01T00:00:00+00:00",
    )

    result = build_event_dossier(repository, smoke["event_id"])

    assert result["decision"] == "blocked_conflict"
    assert "input token price" in result["decision_reason"]


def test_absence_based_conflict_is_removed_during_sanitization() -> None:
    documents = [
        {"id": 1, "source_class": "primary", "content_text": "Model scored 62.1%."},
        {"id": 2, "source_class": "independent", "content_text": "Public leaderboard."},
    ]
    draft = FactSheetDraft(
        event_type="other",
        facts=[
            ResearchFact(text="Компания выпустила модель.", document_ids=[1], kind="release"),
            ResearchFact(text="Модель набрала 62.1%.", document_ids=[1], kind="benchmark"),
            ResearchFact(text="Модель доступна через API.", document_ids=[1], kind="availability"),
        ],
        coverage=FactCoverage(
            event_identity=True,
            capabilities=False,
            benchmarks=True,
            pricing=False,
            availability=True,
            limitations=False,
            independent_confirmation=False,
        ),
        conflicts=[
            FactConflict(
                subject="leaderboard",
                severity="material",
                claims=[
                    ConflictClaim(text="Модель набрала 62.1%.", document_id=1),
                    ConflictClaim(text="Модель отсутствует в топ-10.", document_id=2),
                ],
                resolution="unresolved",
                resolved_document_id=None,
            )
        ],
    )

    sanitized, warnings = sanitize_fact_sheet(draft, documents)

    assert sanitized.conflicts == []
    assert any("absence or non-mention" in warning for warning in warnings)


def test_complementary_benchmark_claims_are_not_treated_as_conflict() -> None:
    documents = [
        {"id": 1, "source_class": "primary", "content_text": "Terminal score: 28.3."},
        {
            "id": 2,
            "source_class": "independent",
            "content_text": "The model trails two competitors on Terminal Bench 3.0.",
        },
    ]
    draft = FactSheetDraft(
        event_type="model_release",
        facts=[
            ResearchFact(text="Компания выпустила модель.", document_ids=[1], kind="release"),
            ResearchFact(text="Модель набрала 28.3.", document_ids=[1], kind="benchmark"),
            ResearchFact(text="Модель доступна через API.", document_ids=[1], kind="availability"),
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
        conflicts=[
            FactConflict(
                subject="Terminal Bench 3.0",
                severity="material",
                claims=[
                    ConflictClaim(text="GLM-5.3 набрала 28.3.", document_id=1),
                    ConflictClaim(
                        text="GLM-5.3 уступает двум конкурентам на Terminal Bench 3.0.",
                        document_id=2,
                    ),
                ],
                resolution="unresolved",
                resolved_document_id=None,
            )
        ],
    )

    sanitized, warnings = sanitize_fact_sheet(draft, documents)

    assert sanitized.conflicts == []
    assert any("not materially contradictory" in warning for warning in warnings)


def test_agent_failure_is_reclassified_as_limitation() -> None:
    documents = [
        {
            "id": 1,
            "source_class": "primary",
            "content_text": "Агенты отключали друг друга и запускали вредоносные скрипты.",
        }
    ]
    draft = FactSheetDraft(
        event_type="research_result",
        facts=[
            ResearchFact(
                text="Компания опубликовала исследование.", document_ids=[1], kind="release"
            ),
            ResearchFact(
                text="Агенты отключали друг друга и запускали вредоносные скрипты.",
                document_ids=[1],
                kind="capability",
            ),
            ResearchFact(
                text="Исследование доступно публично.", document_ids=[1], kind="availability"
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

    sanitized, _warnings = sanitize_fact_sheet(draft, documents)

    assert sanitized.facts[1].kind == "limitation"
    assert sanitized.coverage.limitations is True
