from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.config import (
    Settings,
    load_interest_profile,
    load_search_agent_registry,
    load_source_registry,
    runtime_source_registry,
)
from app.db.repository import Repository
from app.llm.openrouter import OpenRouterRequestError
from app.radar.feedback import FEEDBACK_EXAMPLES_VERSION
from app.radar.search import (
    OpenRouterSearchAgent,
    SearchRadarError,
    actionable_x_trigger,
    common_search_prompt,
    compact_fixed_signals,
    parse_json_content,
    parser_system_prompt,
    run_search_radar,
)

PROJECT_ROOT = Path(__file__).parents[1]
PRIMARY_URL = "https://example.com/robot-factory"
SUPPORTING_URL = "https://example.org/robot-factory-analysis"
CONTEXT_URL = "https://example.net/robot-factory-comparison"


def test_search_prompts_do_not_turn_the_feed_into_a_financial_ticker() -> None:
    discovery = common_search_prompt(4)
    parser = parser_system_prompt()

    assert "максимум один заголовок" in discovery
    assert "Не превращай выдачу в финансовый журнал" in discovery
    assert "Числа по умолчанию переноси в summary" in parser
    assert "а не просто сколько денег" in parser


def test_malformed_parser_json_is_reported_as_search_error() -> None:
    with pytest.raises(SearchRadarError, match="malformed JSON"):
        parse_json_content('{"events": [{"headline": "обрезано}]')


def search_response() -> dict:
    return {
        "model": "openai/gpt-5.6-luna",
        "choices": [
            {
                "message": {
                    "content": (
                        "A robotics company opened a mass-production factory. "
                        f"[Primary]({PRIMARY_URL}) [Analysis]({SUPPORTING_URL})"
                    ),
                    "annotations": [
                        {
                            "type": "url_citation",
                            "url_citation": {
                                "url": PRIMARY_URL,
                                "title": "Robot factory launch",
                                "content": "The company opened a factory for mass production.",
                            },
                        },
                        {
                            "type": "url_citation",
                            "url_citation": {
                                "url": SUPPORTING_URL,
                                "title": "Independent analysis",
                                "content": "The facility targets 10,000 robots per year.",
                            },
                        },
                    ],
                }
            }
        ],
        "usage": {
            "cost": 0.007,
            "prompt_tokens": 1000,
            "completion_tokens": 300,
            "server_tool_use_details": {"web_search_requests": 1},
        },
    }


def parser_response(*, primary_url: str = PRIMARY_URL) -> dict:
    event_at = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    return {
        "model": "openai/gpt-5.6-luna",
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "events": [
                                {
                                    "headline": "В Азии запустили массовое производство роботов",
                                    "summary": (
                                        "Компания открыла завод серийного производства роботов "
                                        "с заявленной мощностью до 10 тысяч единиц в год."
                                    ),
                                    "why_relevant": (
                                        "Это измеримый переход embodied AI от прототипов "
                                        "к промышленному масштабу."
                                    ),
                                    "material_update": (
                                        "Предприятие начало работу и раскрыло годовую мощность."
                                    ),
                                    "event_at": event_at,
                                    "event_type": "manufacturing_launch",
                                    "primary_url": primary_url,
                                    "supporting_urls": [SUPPORTING_URL],
                                    "topics": ["robotics", "manufacturing"],
                                    "regions": ["east_asia"],
                                    "organizations": ["Example Robotics"],
                                    "confidence": 0.92,
                                }
                            ]
                        },
                        ensure_ascii=False,
                    )
                }
            }
        ],
        "usage": {"cost": 0.001, "prompt_tokens": 800, "completion_tokens": 250},
    }


def structured_response(content: dict, *, cost: float = 0.001) -> dict:
    return {
        "model": "openai/gpt-5.6-luna",
        "choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}}],
        "usage": {"cost": cost, "prompt_tokens": 100, "completion_tokens": 40},
    }


def context_accept_response(payload: dict) -> dict:
    request = json.loads(payload["messages"][1]["content"])
    return structured_response(
        {
            "decisions": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "decision": "accept",
                    "research_question": "",
                }
                for candidate in request["candidates"]
            ]
        }
    )


def fake_transport(payload: dict, _api_key: str) -> dict:
    if payload.get("tools"):
        assert payload["tools"][0]["type"] == "openrouter:web_search"
        assert payload["tools"][0]["parameters"]["max_uses"] == 2
        assert payload["max_tool_calls"] == 2
        return search_response()
    assert payload["response_format"]["type"] == "json_schema"
    schema_name = payload["response_format"]["json_schema"]["name"]
    if schema_name == "context_decision_batch":
        return context_accept_response(payload)
    schema = payload["response_format"]["json_schema"]["schema"]
    event_schema = schema["$defs"]["SearchEventCandidate"]
    assert set(event_schema["required"]) == set(event_schema["properties"])
    return parser_response()


def test_search_agent_uses_citations_then_strict_parser(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        app_env="test",
        data_dir=tmp_path,
        openrouter_api_key="test-key",
        openrouter_search_model="openai/gpt-5.6-luna",
    )
    lens = load_search_agent_registry(
        PROJECT_ROOT / "tests/fixtures/config/search-agents.yaml"
    ).lenses[0]
    profile = load_interest_profile(PROJECT_ROOT / "tests/fixtures/config/interests.yaml")
    now = datetime.now(UTC)

    result = OpenRouterSearchAgent(settings, transport=fake_transport).run_lens(
        lens=lens,
        profile=profile,
        window_start=now - timedelta(hours=120),
        window_end=now,
        fixed_signals=[],
        seen_events=[],
    )

    assert [event.primary_url for event in result.events] == [PRIMARY_URL]
    assert result.events[0].supporting_urls == [SUPPORTING_URL]
    assert result.search_requests == 1
    assert result.cost_usd == 0.008


def test_search_lens_receives_weighted_feedback_examples(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        app_env="test",
        data_dir=tmp_path,
        openrouter_api_key="test-key",
        openrouter_search_model="openai/gpt-5.6-luna",
    )
    lens = load_search_agent_registry(
        PROJECT_ROOT / "tests/fixtures/config/search-agents.yaml"
    ).lenses[0]
    profile = load_interest_profile(PROJECT_ROOT / "tests/fixtures/config/interests.yaml")
    now = datetime.now(UTC)
    examples = [
        {
            "headline": "ИИ-чип вышел в серийное производство",
            "summary": "Компания начала поставки и раскрыла измеримую производительность.",
            "topics": ["semiconductors"],
            "reaction": "superlike",
            "comment": "Нужна более понятная глубина изложения.",
            "weight": 2,
        }
    ]

    def transport(payload: dict, _api_key: str) -> dict:
        if payload.get("tools"):
            request = json.loads(payload["messages"][1]["content"])
            assert request["FEEDBACK_EXAMPLES_VERSION"] == FEEDBACK_EXAMPLES_VERSION
            assert request["FEEDBACK_EXAMPLES"] == examples
            return search_response()
        return parser_response()

    OpenRouterSearchAgent(settings, transport=transport).run_lens(
        lens=lens,
        profile=profile,
        window_start=now - timedelta(hours=120),
        window_end=now,
        fixed_signals=[],
        seen_events=[],
        feedback_examples=examples,
    )

    prompt = common_search_prompt(4)
    assert "не означает неприязнь к теме" in prompt
    assert "не жёсткие правила" in prompt


def test_search_agent_rejects_url_missing_from_citations(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        app_env="test",
        data_dir=tmp_path,
        openrouter_api_key="test-key",
    )
    lens = load_search_agent_registry(
        PROJECT_ROOT / "tests/fixtures/config/search-agents.yaml"
    ).lenses[0]
    profile = load_interest_profile(PROJECT_ROOT / "tests/fixtures/config/interests.yaml")
    now = datetime.now(UTC)

    def ungrounded_transport(payload: dict, _api_key: str) -> dict:
        return (
            search_response()
            if payload.get("tools")
            else parser_response(primary_url="https://hallucinated.example/news")
        )

    result = OpenRouterSearchAgent(settings, transport=ungrounded_transport).run_lens(
        lens=lens,
        profile=profile,
        window_start=now - timedelta(hours=120),
        window_end=now,
        fixed_signals=[],
        seen_events=[],
    )

    assert result.events == []


def test_new_space_lens_source_is_registered_before_paid_call(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        app_env="test",
        data_dir=tmp_path,
        sources_file=PROJECT_ROOT / "tests/fixtures/config/sources.yaml",
        interests_file=PROJECT_ROOT / "tests/fixtures/config/interests.yaml",
        search_agents_file=PROJECT_ROOT / "tests/fixtures/config/search-agents.yaml",
        contours_file=PROJECT_ROOT / "tests/fixtures/config/contours.yaml",
        openrouter_api_key="test-key",
    )
    repo = Repository(settings.database_path)
    repo.migrate()

    def check_preflight(payload: dict, api_key: str) -> dict:
        with repo.connection() as db:
            source = db.execute(
                "SELECT role, admission_mode FROM sources WHERE id='web_agent_space'"
            ).fetchone()
        assert dict(source) == {"role": "radar", "admission_mode": "event_candidate"}
        return fake_transport(payload, api_key)

    result = run_search_radar(settings, repo, lens_ids=["space"], transport=check_preflight)
    assert result.events_inserted == 1
    assert result.preview_cards == 1


@pytest.mark.parametrize(
    "crash_stage,expected_calls,expected_cost",
    [
        ("source_persistence", 2, 0.008),
        ("context_ledger", 3, 0.009),
    ],
)
def test_search_crash_finalizes_run_and_recovers_known_cost_without_retry(
    tmp_path: Path,
    monkeypatch,
    crash_stage: str,
    expected_calls: int,
    expected_cost: float,
) -> None:
    settings = Settings(
        _env_file=None,
        app_env="test",
        data_dir=tmp_path,
        sources_file=PROJECT_ROOT / "tests/fixtures/config/sources.yaml",
        interests_file=PROJECT_ROOT / "tests/fixtures/config/interests.yaml",
        search_agents_file=PROJECT_ROOT / "tests/fixtures/config/search-agents.yaml",
        contours_file=PROJECT_ROOT / "tests/fixtures/config/contours.yaml",
        openrouter_api_key="test-key",
    )
    repo = Repository(settings.database_path)
    repo.migrate()
    calls = []

    def observed(payload, api_key):
        calls.append(payload)
        return fake_transport(payload, api_key)

    def crash(*args, **kwargs):
        raise RuntimeError("local persistence crash")

    if crash_stage == "source_persistence":
        monkeypatch.setattr(repo, "insert_source_item_with_id", crash)
    else:
        monkeypatch.setattr("app.radar.search.record_context_usage", crash)
    with pytest.raises(RuntimeError, match="local persistence crash"):
        run_search_radar(settings, repo, lens_ids=["space"], transport=observed)
    assert len(calls) == expected_calls
    assert repo.current_month_cost() == pytest.approx(expected_cost)
    with repo.connection() as db:
        run = dict(db.execute("SELECT * FROM runs WHERE run_type='search_radar'").fetchone())
        recovery_rows = db.execute(
            "SELECT COUNT(*) FROM usage_ledger WHERE operation='search_radar_accounting_recovery'"
        ).fetchone()[0]
    assert run["status"] == "failed"
    assert run["finished_at"]
    assert run["cost_usd"] == pytest.approx(expected_cost)
    assert recovery_rows == int(crash_stage == "context_ledger")


def test_radar_preview_is_quarantined_from_publication_queue(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        app_env="test",
        data_dir=tmp_path,
        sources_file=PROJECT_ROOT / "tests/fixtures/config/sources.yaml",
        interests_file=PROJECT_ROOT / "tests/fixtures/config/interests.yaml",
        search_agents_file=PROJECT_ROOT / "tests/fixtures/config/search-agents.yaml",
        telegram_session_path=tmp_path / "reader.session",
        openrouter_api_key="test-key",
        allowed_user_id=123,
    )
    repository = Repository(settings.database_path)
    repository.migrate()
    repository.sync_sources(load_source_registry(settings.sources_file))

    stats = run_search_radar(
        settings,
        repository,
        lookback_hours=120,
        preview_limit=4,
        lens_ids=["frontier_global"],
        transport=fake_transport,
    )

    assert stats.preview_cards == 1
    assert stats.cost_usd == pytest.approx(0.009)
    assert stats.context_candidates == 1
    assert stats.context_research_requested == 0
    assert stats.context_enriched == 0
    assert stats.context_cost_usd == pytest.approx(0.001)
    assert repository.deliverable_cards(123, 10) == []
    delivery_ids = repository.create_candidate_batch(
        chat_id=123,
        card_ids=stats.card_ids,
        batch_id=stats.batch_id or "",
    )
    assert len(delivery_ids) == 1
    batch = repository.candidate_batch_cards(123, stats.batch_id or "")
    assert [row["card_id"] for row in batch] == stats.card_ids
    with repository.connection() as connection:
        card = connection.execute(
            "SELECT validation_status, structured_output_json, rendered_html "
            "FROM cards WHERE id = ?",
            (stats.card_ids[0],),
        ).fetchone()
        event = connection.execute(
            "SELECT status, score_features_json FROM events WHERE id = ?",
            (batch[0]["event_id"],),
        ).fetchone()
    assert card["validation_status"] == "candidate_preview"
    assert event["status"] == "candidate"
    card_payload = json.loads(card["structured_output_json"])
    event_features = json.loads(event["score_features_json"])
    assert card_payload["contours"]["primary_contour"] == "startups"
    assert event_features["primary_contour"] == "startups"
    assert event_features["signal_quality"]["evidence_status"] == "unverified"
    assert event_features["signal_quality"]["trigger_count"] == 1
    assert event_features["source_kind"] == "web"
    assert event_features["discovery_mode"] == "search"
    assert (
        event_features["exploration_fraction"]
        == load_interest_profile(settings.interests_file).exploration_fraction
    )
    assert "saturation" in event_features
    assert "Контур: Стартапы и технологические компании" in card["rendered_html"]
    assert "<b>Блок: Стартапы и технологические компании</b>" in card["rendered_html"]


def test_context_accept_adds_no_second_web_search(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        app_env="test",
        data_dir=tmp_path,
        sources_file=PROJECT_ROOT / "tests/fixtures/config/sources.yaml",
        interests_file=PROJECT_ROOT / "tests/fixtures/config/interests.yaml",
        search_agents_file=PROJECT_ROOT / "tests/fixtures/config/search-agents.yaml",
        telegram_session_path=tmp_path / "reader.session",
        openrouter_api_key="test-key",
        allowed_user_id=123,
    )
    repository = Repository(settings.database_path)
    repository.migrate()
    repository.sync_sources(load_source_registry(settings.sources_file))
    calls: list[dict] = []

    def transport(payload: dict, api_key: str) -> dict:
        calls.append(payload)
        return fake_transport(payload, api_key)

    stats = run_search_radar(
        settings,
        repository,
        lookback_hours=120,
        preview_limit=4,
        lens_ids=["frontier_global"],
        transport=transport,
    )

    tool_calls = [payload for payload in calls if payload.get("tools")]
    assert len(tool_calls) == 1
    assert tool_calls[0]["tools"][0]["parameters"]["max_uses"] == 2
    assert stats.context_research_attempted == 0
    assert stats.context_cost_usd == pytest.approx(0.001)
    with repository.connection() as connection:
        usage = connection.execute(
            "SELECT cost_usd FROM usage_ledger WHERE operation = 'context_validator'"
        ).fetchone()
        source_item = connection.execute(
            "SELECT raw_payload_json FROM source_items WHERE source_url = ?", (PRIMARY_URL,)
        ).fetchone()
    assert float(usage["cost_usd"]) == pytest.approx(0.001)
    raw_payload = json.loads(source_item["raw_payload_json"])
    assert raw_payload["context_validation"]["outcome"] == "accepted"


def test_context_drop_is_saved_as_trace_but_not_made_deliverable(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        app_env="test",
        data_dir=tmp_path,
        sources_file=PROJECT_ROOT / "tests/fixtures/config/sources.yaml",
        interests_file=PROJECT_ROOT / "tests/fixtures/config/interests.yaml",
        search_agents_file=PROJECT_ROOT / "tests/fixtures/config/search-agents.yaml",
        telegram_session_path=tmp_path / "reader.session",
        openrouter_api_key="test-key",
        allowed_user_id=123,
    )
    repository = Repository(settings.database_path)
    repository.migrate()
    repository.sync_sources(load_source_registry(settings.sources_file))

    def transport(payload: dict, api_key: str) -> dict:
        if payload.get("tools"):
            return search_response()
        schema_name = payload["response_format"]["json_schema"]["name"]
        if schema_name == "search_event_batch":
            return parser_response()
        if schema_name == "context_decision_batch":
            return structured_response(
                {
                    "decisions": [
                        {
                            "candidate_id": "event-1",
                            "decision": "drop",
                            "research_question": "",
                        }
                    ]
                }
            )
        return fake_transport(payload, api_key)

    stats = run_search_radar(
        settings,
        repository,
        lookback_hours=120,
        preview_limit=4,
        lens_ids=["frontier_global"],
        transport=transport,
    )

    assert stats.preview_cards == 0
    assert stats.context_rejected == 1
    with repository.connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM cards").fetchone()[0] == 0
        source_item = connection.execute(
            "SELECT raw_payload_json FROM source_items WHERE source_url = ?", (PRIMARY_URL,)
        ).fetchone()
    metadata = json.loads(source_item["raw_payload_json"])["context_validation"]
    assert metadata["outcome"] == "rejected_too_thin"


def test_context_research_enriches_card_metadata_and_cost(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        app_env="test",
        data_dir=tmp_path,
        sources_file=PROJECT_ROOT / "tests/fixtures/config/sources.yaml",
        interests_file=PROJECT_ROOT / "tests/fixtures/config/interests.yaml",
        search_agents_file=PROJECT_ROOT / "tests/fixtures/config/search-agents.yaml",
        telegram_session_path=tmp_path / "reader.session",
        openrouter_api_key="test-key",
        allowed_user_id=123,
    )
    repository = Repository(settings.database_path)
    repository.migrate()
    repository.sync_sources(load_source_registry(settings.sources_file))
    calls: list[dict] = []

    def transport(payload: dict, _api_key: str) -> dict:
        calls.append(payload)
        if payload.get("tools"):
            max_uses = payload["tools"][0]["parameters"]["max_uses"]
            if max_uses == 2:
                return search_response()
            assert max_uses == 1
            return {
                "model": "openai/gpt-5.6-luna",
                "choices": [
                    {
                        "message": {
                            "content": "Источник сравнивает мощность с пилотной линией.",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "url_citation": {
                                        "url": CONTEXT_URL,
                                        "title": "Factory comparison",
                                        "content": (
                                            "The factory has five times the pilot capacity."
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {
                    "cost": 0.007,
                    "prompt_tokens": 200,
                    "completion_tokens": 80,
                    "server_tool_use_details": {"web_search_requests": 1},
                },
            }

        schema_name = payload["response_format"]["json_schema"]["name"]
        if schema_name == "search_event_batch":
            return parser_response()
        request = json.loads(payload["messages"][1]["content"])
        if schema_name == "context_decision_batch":
            candidate_id = request["candidates"][0]["candidate_id"]
            return structured_response(
                {
                    "decisions": [
                        {
                            "candidate_id": candidate_id,
                            "decision": "research",
                            "research_question": (
                                "Насколько мощность выше пилотного производства?"
                            ),
                        }
                    ]
                }
            )
        assert schema_name == "context_merge"
        candidate_id = request["candidate"]["candidate_id"]
        return structured_response(
            {
                "candidate_id": candidate_id,
                "resolved": True,
                "summary": (
                    "Компания открыла завод серийного производства роботов мощностью "
                    "до 10 тысяч единиц в год — это в пять раз больше пилотной линии."
                ),
                "why_relevant": (
                    "Сопоставимый рост мощности показывает измеримый переход embodied AI "
                    "от пилота к промышленному масштабу."
                ),
                "supporting_urls": [CONTEXT_URL],
            }
        )

    stats = run_search_radar(
        settings,
        repository,
        lookback_hours=120,
        preview_limit=4,
        lens_ids=["frontier_global"],
        transport=transport,
    )

    assert stats.preview_cards == 1
    assert stats.context_research_requested == 1
    assert stats.context_research_attempted == 1
    assert stats.context_enriched == 1
    assert stats.context_fail_open == 0
    assert stats.context_cost_usd == pytest.approx(0.009)
    assert stats.cost_usd == pytest.approx(0.017)
    assert [
        payload["tools"][0]["parameters"]["max_uses"] for payload in calls if payload.get("tools")
    ] == [2, 1]

    with repository.connection() as connection:
        card = connection.execute(
            """
            SELECT structured_output_json, rendered_html, source_urls_json
            FROM cards WHERE id = ?
            """,
            (stats.card_ids[0],),
        ).fetchone()
        source_item = connection.execute(
            """
            SELECT outbound_urls_json, raw_payload_json
            FROM source_items WHERE source_url = ?
            """,
            (PRIMARY_URL,),
        ).fetchone()
        usage_rows = connection.execute(
            """
            SELECT operation, cost_usd, metadata_json
            FROM usage_ledger
            WHERE operation IN ('context_validator', 'context_research_merge')
            ORDER BY operation
            """
        ).fetchall()

    structured = json.loads(card["structured_output_json"])
    assert "в пять раз" in structured["candidate"]["summary"]
    assert structured["context_validation"]["outcome"] == "enriched"
    assert structured["context_validation"]["citations"][0]["url"] == CONTEXT_URL
    assert CONTEXT_URL in json.loads(card["source_urls_json"])
    assert "в пять раз" in card["rendered_html"]

    raw_payload = json.loads(source_item["raw_payload_json"])
    assert raw_payload["context_validation"]["enriched"] is True
    assert raw_payload["candidate"]["supporting_urls"][0] == CONTEXT_URL
    assert CONTEXT_URL in json.loads(source_item["outbound_urls_json"])
    usage = {row["operation"]: row for row in usage_rows}
    assert float(usage["context_validator"]["cost_usd"]) == pytest.approx(0.001)
    assert float(usage["context_research_merge"]["cost_usd"]) == pytest.approx(0.008)
    research_metadata = json.loads(usage["context_research_merge"]["metadata_json"])
    assert research_metadata["enrichments_succeeded"] == 1


def test_context_validator_failure_holds_preliminary_candidate_from_delivery(
    tmp_path: Path,
) -> None:
    settings = Settings(
        _env_file=None,
        app_env="test",
        data_dir=tmp_path,
        sources_file=PROJECT_ROOT / "tests/fixtures/config/sources.yaml",
        interests_file=PROJECT_ROOT / "tests/fixtures/config/interests.yaml",
        search_agents_file=PROJECT_ROOT / "tests/fixtures/config/search-agents.yaml",
        telegram_session_path=tmp_path / "reader.session",
        openrouter_api_key="test-key",
        allowed_user_id=123,
    )
    repository = Repository(settings.database_path)
    repository.migrate()
    repository.sync_sources(load_source_registry(settings.sources_file))

    def transport(payload: dict, _api_key: str) -> dict:
        if payload.get("tools"):
            return search_response()
        schema_name = payload["response_format"]["json_schema"]["name"]
        if schema_name == "search_event_batch":
            return parser_response()
        raise OpenRouterRequestError("validator unavailable")

    stats = run_search_radar(
        settings,
        repository,
        lookback_hours=120,
        preview_limit=4,
        lens_ids=["frontier_global"],
        transport=transport,
    )

    assert stats.preview_cards == 0
    assert stats.context_fail_open == 1
    assert stats.context_cost_usd == 0
    assert stats.cost_usd == pytest.approx(0.008)
    with repository.connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM cards").fetchone()[0] == 0
        source_item = connection.execute(
            "SELECT raw_payload_json FROM source_items WHERE source_url = ?", (PRIMARY_URL,)
        ).fetchone()
    raw_payload = json.loads(source_item["raw_payload_json"])
    assert "10 тысяч" in raw_payload["candidate"]["summary"]
    assert raw_payload["context_validation"]["outcome"] == "validator_fail_open"


def test_multiple_search_lenses_do_not_fake_independent_corroboration(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "corroboration.sqlite3")
    repository.migrate()
    repository.sync_sources(
        load_source_registry(PROJECT_ROOT / "tests/fixtures/config/sources.yaml")
    )
    item_ids: list[int] = []
    for index, source_id in enumerate(
        ["web_agent_frontier_global", "web_agent_personal_rare_signals"]
    ):
        item_id = repository.insert_source_item_with_id(
            source_id=source_id,
            external_id=f"same-{index}",
            source_url=PRIMARY_URL,
            original_text=("Mass production robot factory launched. " * 4),
            original_language="en",
            published_at="2026-08-22T12:00:00+00:00",
            outbound_urls=[],
            media_metadata=None,
            payload_hash=f"same-hash-{index}",
            raw_payload={},
        )
        assert item_id is not None
        item_ids.append(item_id)
        repository.attach_item_to_event(
            source_item_id=item_id,
            event_type="web_signal",
            headline="Robot factory starts mass production",
            event_at="2026-08-22T12:00:00+00:00",
            primary_url=PRIMARY_URL,
            fingerprint="same-fingerprint",
            score=5,
            score_features={"base_score": 5},
            decision_reason="test",
        )

    with repository.connection() as connection:
        row = connection.execute("SELECT score, score_features_json FROM events").fetchone()
    assert float(row["score"]) == 5
    assert json.loads(row["score_features_json"])["corroborating_sources"] == 0


def test_failed_normalization_still_records_paid_search_usage(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        app_env="test",
        data_dir=tmp_path,
        sources_file=PROJECT_ROOT / "tests/fixtures/config/sources.yaml",
        interests_file=PROJECT_ROOT / "tests/fixtures/config/interests.yaml",
        search_agents_file=PROJECT_ROOT / "tests/fixtures/config/search-agents.yaml",
        telegram_session_path=tmp_path / "reader.session",
        openrouter_api_key="test-key",
        allowed_user_id=123,
    )
    repository = Repository(settings.database_path)
    repository.migrate()
    repository.sync_sources(load_source_registry(settings.sources_file))

    def failing_parser_transport(payload: dict, _api_key: str) -> dict:
        if payload.get("tools"):
            return search_response()
        raise OpenRouterRequestError("invalid structured-output schema")

    stats = run_search_radar(
        settings,
        repository,
        lookback_hours=120,
        preview_limit=4,
        lens_ids=["frontier_global"],
        transport=failing_parser_transport,
    )

    assert stats.lenses_succeeded == 0
    assert stats.cost_usd == 0.007
    assert repository.current_month_cost() == 0.007
    assert "normalization failed" in stats.errors[0]


def test_recent_fixed_signals_caps_each_source_to_preserve_radar_breadth(
    tmp_path: Path,
) -> None:
    repository = Repository(tmp_path / "signals.sqlite3")
    repository.migrate()
    repository.sync_sources(
        load_source_registry(PROJECT_ROOT / "tests/fixtures/config/sources.yaml")
    )
    published = "2026-08-23T10:00:00+00:00"
    for index in range(5):
        repository.insert_source_item(
            source_id="telegram_ai_newz",
            external_id=f"busy-{index}",
            source_url=f"https://t.me/ai_newz/{index + 1}",
            original_text=f"Busy source signal {index}",
            original_language="ru",
            published_at=published,
            outbound_urls=[],
            media_metadata=None,
            payload_hash=f"busy-hash-{index}",
            raw_payload={},
        )
    repository.insert_source_item(
        source_id="telegram_blockchain_rf",
        external_id="other-1",
        source_url="https://t.me/blockchainRF/1",
        original_text="Independent source signal",
        original_language="ru",
        published_at=published,
        outbound_urls=[],
        media_metadata=None,
        payload_hash="other-hash-1",
        raw_payload={},
    )

    rows = repository.recent_fixed_signals("2026-08-23T00:00:00+00:00", limit=40)

    assert sum(row["source_name"] == "AI Newz" for row in rows) == 2
    assert sum(row["source_name"] == "BlockchainRF" for row in rows) == 1


def test_recent_fixed_signals_uses_enabled_x_as_trigger_only(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        sources_file=PROJECT_ROOT / "tests/fixtures/config/sources.yaml",
        x_authors_file=PROJECT_ROOT / "tests/fixtures/config/x-authors.yaml",
    )
    repository = Repository(tmp_path / "x-signals.sqlite3")
    repository.migrate()
    runtime = runtime_source_registry(settings)
    corrected_tunguz = next(source for source in runtime.sources if source.id == "x_ttunguz")
    stale_tunguz = corrected_tunguz.model_copy(
        update={"id": "x_tunguz", "locator": "tunguz", "enabled": True}
    )
    repository.sync_sources(
        runtime.model_copy(update={"sources": [*runtime.sources, stale_tunguz]})
    )
    repository.sync_sources(runtime)

    with repository.connection() as connection:
        stale_row = connection.execute(
            "SELECT enabled, candidate_enabled FROM sources WHERE id = 'x_tunguz'"
        ).fetchone()
    assert stale_row is not None
    assert stale_row["enabled"] == 0
    assert stale_row["candidate_enabled"] == 0

    x_sources = [source for source in runtime.sources if source.kind.value == "x"]
    assert x_sources
    assert all(not source.candidate_enabled for source in x_sources)

    repository.insert_source_item(
        source_id="x_rowancheung",
        external_id="rowan-trigger",
        source_url="https://x.com/rowancheung/status/1",
        original_text="A fresh X signal that should seed the search radar.",
        original_language="en",
        published_at="2026-08-23T10:00:00+00:00",
        outbound_urls=[],
        media_metadata=None,
        payload_hash="rowan-trigger-hash",
        raw_payload={},
    )
    repository.insert_source_item(
        source_id="x_kirkdborne",
        external_id="disabled-trigger",
        source_url="https://x.com/KirkDBorne/status/2",
        original_text="A disabled X source must not seed the search radar.",
        original_language="en",
        published_at="2026-08-23T10:00:00+00:00",
        outbound_urls=[],
        media_metadata=None,
        payload_hash="disabled-trigger-hash",
        raw_payload={},
    )

    rows = repository.recent_fixed_signals("2026-08-23T00:00:00+00:00", limit=40)

    assert [row["source_name"] for row in rows] == ["X — Rowan Cheung"]


def test_raw_x_trigger_does_not_pollute_seen_event_memory(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        sources_file=PROJECT_ROOT / "tests/fixtures/config/sources.yaml",
        x_authors_file=PROJECT_ROOT / "tests/fixtures/config/x-authors.yaml",
    )
    repository = Repository(tmp_path / "memory.sqlite3")
    repository.migrate()
    repository.sync_sources(runtime_source_registry(settings))
    published = "2026-08-23T10:00:00+00:00"

    for source_id, external_id, url in [
        ("x_rowancheung", "x-raw", "https://x.com/rowancheung/status/10"),
        ("web_agent_frontier_global", "web-confirmed", PRIMARY_URL),
    ]:
        item_id = repository.insert_source_item_with_id(
            source_id=source_id,
            external_id=external_id,
            source_url=url,
            original_text="A concrete technology event with enough detail for event memory.",
            original_language="en",
            published_at=published,
            outbound_urls=[],
            media_metadata=None,
            payload_hash=f"hash-{external_id}",
            raw_payload={},
        )
        assert item_id is not None
        repository.attach_item_to_event(
            source_item_id=item_id,
            event_type="test_signal",
            headline=f"Event from {source_id}",
            event_at=published,
            primary_url=url,
            fingerprint=f"fingerprint-{external_id}",
            score=5,
            score_features={"base_score": 5},
            decision_reason="test",
        )

    rows = repository.recent_event_memory("2026-08-23T00:00:00+00:00", limit=20)

    assert [row["source_url"] for row in rows] == [PRIMARY_URL]


def test_x_trigger_filter_rejects_opinion_but_keeps_concrete_event_clues() -> None:
    base = {
        "source_kind": "x",
        "source_name": "X author",
        "published_at": "2026-08-23T10:00:00+00:00",
        "source_url": "https://x.com/author/status/1",
        "outbound_urls_json": "[]",
    }

    assert not actionable_x_trigger(
        {**base, "original_text": "Your standards should be just short of impossibly high."}
    )
    assert actionable_x_trigger(
        {**base, "original_text": "The company launched a new inference chip for factories."}
    )
    assert actionable_x_trigger(
        {**base, "original_text": "Agents now use 5x more tokens than human users."}
    )
    assert actionable_x_trigger(
        {
            **base,
            "original_text": "A detailed technical analysis is available here.",
            "outbound_urls_json": '["https://example.com/research"]',
        }
    )


def test_fixed_signal_compactor_caps_x_before_llm_prompt() -> None:
    rows: list[dict] = []
    for kind, count in [("x", 30), ("telegram", 20), ("rss", 10), ("github", 10)]:
        for index in range(count):
            rows.append(
                {
                    "source_kind": kind,
                    "source_name": f"{kind}-{index}",
                    "published_at": "2026-08-23T10:00:00+00:00",
                    "source_url": f"https://example.com/{kind}/{index}",
                    "outbound_urls_json": "[]",
                    "original_text": (
                        f"Company launched production system number {index}."
                        if kind == "x"
                        else f"Concrete {kind} signal number {index}."
                    ),
                }
            )

    compacted = compact_fixed_signals(rows, limit=40)
    counts = {
        kind: sum(item["kind"] == kind for item in compacted)
        for kind in {item["kind"] for item in compacted}
    }

    assert counts == {"x": 16, "telegram": 12, "rss": 6, "github": 6}
    assert len(compacted) == 40
