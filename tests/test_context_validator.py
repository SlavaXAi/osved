from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.radar.context import (
    ContextCandidate,
    OpenRouterContextValidator,
    _validator_system_prompt,
)

SOURCE_URL = "https://example.com/chip-benchmark"


def test_validator_prompt_requires_product_context_for_funding_and_point_for_benchmark() -> None:
    prompt = _validator_system_prompt()

    assert "сумма денег сама по себе недостаточна" in prompt
    assert "что именно компания продаёт или строит" in prompt
    assert "Benchmark-only" in prompt
    assert "какое решение для продукта" in prompt


def candidate(candidate_id: str = "event-1") -> ContextCandidate:
    return ContextCandidate(
        candidate_id=candidate_id,
        headline="Китайская компания представила новый ИИ-чип",
        summary=(
            "Производитель выпустил ускоритель для локального инференса с заявленной "
            "производительностью 64 TOPS."
        ),
        why_relevant=(
            "Чип может расширить локальную аппаратную базу для роботов и автономных систем."
        ),
    )


def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        app_env="test",
        data_dir=tmp_path,
        openrouter_api_key="test-key",
        openrouter_search_model="openai/gpt-5.6-luna",
    )


def response(content: dict[str, Any], *, cost: float = 0.001) -> dict[str, Any]:
    return {
        "model": "openai/gpt-5.6-luna",
        "choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}}],
        "usage": {
            "cost": cost,
            "prompt_tokens": 100,
            "completion_tokens": 40,
        },
    }


def search_response() -> dict[str, Any]:
    return {
        "model": "openai/gpt-5.6-luna",
        "choices": [
            {
                "message": {
                    "content": "Источник сравнивает чип с предыдущим поколением.",
                    "annotations": [
                        {
                            "type": "url_citation",
                            "url_citation": {
                                "url": SOURCE_URL,
                                "title": "Chip benchmark",
                                "content": "The new chip is twice as fast as its predecessor.",
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


def assert_all_properties_required(schema: dict[str, Any]) -> None:
    def visit(node: Any) -> None:
        if isinstance(node, dict):
            properties = node.get("properties")
            if isinstance(properties, dict):
                assert set(node.get("required") or []) == set(properties)
                assert node.get("additionalProperties") is False
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(schema)


def test_accept_never_opens_web_search(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def transport(payload: dict[str, Any], _key: str) -> dict[str, Any]:
        calls.append(payload)
        assert "tools" not in payload
        assert_all_properties_required(payload["response_format"]["json_schema"]["schema"])
        return response(
            {
                "decisions": [
                    {
                        "candidate_id": "event-1",
                        "decision": "accept",
                        "research_question": "",
                    }
                ]
            }
        )

    result = OpenRouterContextValidator(settings(tmp_path), transport=transport).run([candidate()])

    assert len(calls) == 1
    assert result.enrichments_requested == 0
    assert result.enrichments_attempted == 0
    assert result.candidates[0].outcome == "accepted"
    assert result.candidates[0].enriched is False
    assert result.cost_usd == 0.001


def test_thin_trend_is_rejected_without_web_search(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def transport(payload: dict[str, Any], _key: str) -> dict[str, Any]:
        calls.append(payload)
        return response(
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

    result = OpenRouterContextValidator(settings(tmp_path), transport=transport).run([candidate()])

    assert len(calls) == 1
    assert result.candidates[0].decision == "drop"
    assert result.candidates[0].outcome == "rejected_too_thin"
    assert result.enrichments_attempted == 0


def test_missing_candidate_decision_is_not_silently_accepted(tmp_path: Path) -> None:
    def transport(_payload: dict[str, Any], _key: str) -> dict[str, Any]:
        return response({"decisions": []})

    result = OpenRouterContextValidator(settings(tmp_path), transport=transport).run([candidate()])

    assert result.candidates[0].decision == "drop"
    assert result.candidates[0].outcome == "rejected_too_thin"


def test_research_gets_exactly_one_search_then_strict_merge(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def transport(payload: dict[str, Any], _key: str) -> dict[str, Any]:
        calls.append(payload)
        if len(calls) == 1:
            return response(
                {
                    "decisions": [
                        {
                            "candidate_id": "event-1",
                            "decision": "research",
                            "research_question": ("С чем сравнима производительность 64 TOPS?"),
                        }
                    ]
                }
            )
        if payload.get("tools"):
            tool = payload["tools"][0]
            assert tool["type"] == "openrouter:web_search"
            assert tool["parameters"]["max_uses"] == 1
            assert payload["max_tool_calls"] == 1
            return search_response()
        assert_all_properties_required(payload["response_format"]["json_schema"]["schema"])
        return response(
            {
                "candidate_id": "event-1",
                "resolved": True,
                "headline": "Компания уточнила возможности ускорителя для локального инференса",
                "summary": (
                    "Производитель выпустил ускоритель для локального инференса на "
                    "64 TOPS — вдвое быстрее предыдущего поколения."
                ),
                "why_relevant": (
                    "Рост производительности расширяет локальную аппаратную базу для "
                    "роботов и автономных систем."
                ),
                "supporting_urls": [SOURCE_URL],
            }
        )

    result = OpenRouterContextValidator(settings(tmp_path), transport=transport).run([candidate()])

    assert len(calls) == 3
    assert sum(bool(payload.get("tools")) for payload in calls) == 1
    assert result.enrichments_attempted == 1
    assert result.enrichments_succeeded == 1
    assert result.candidates[0].outcome == "enriched"
    assert result.candidates[0].enriched is True
    assert result.candidates[0].headline.startswith("Компания уточнила")
    assert "вдвое" in result.candidates[0].summary
    assert [item.url for item in result.candidates[0].citations] == [SOURCE_URL]
    assert result.usage.search_requests == 1
    assert result.cost_usd == pytest.approx(0.009)


def test_unresolved_research_fails_open_with_original_text(tmp_path: Path) -> None:
    original = candidate()
    calls = 0

    def transport(payload: dict[str, Any], _key: str) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return response(
                {
                    "decisions": [
                        {
                            "candidate_id": "event-1",
                            "decision": "research",
                            "research_question": "Каков независимый benchmark чипа?",
                        }
                    ]
                }
            )
        if payload.get("tools"):
            return search_response()
        return response(
            {
                "candidate_id": "event-1",
                "resolved": False,
                "summary": original.summary,
                "why_relevant": original.why_relevant,
                "supporting_urls": [],
            }
        )

    result = OpenRouterContextValidator(settings(tmp_path), transport=transport).run([original])

    item = result.candidates[0]
    assert calls == 3
    assert item.outcome == "unresolved_fail_open"
    assert item.enriched is False
    assert item.summary == original.summary
    assert item.why_relevant == original.why_relevant
    assert item.error is not None
    assert result.enrichments_attempted == 1
    assert result.enrichments_succeeded == 0
    assert result.usage.search_requests == 1


def test_enrichment_is_hard_capped_at_three(tmp_path: Path) -> None:
    items = [candidate(f"event-{index}") for index in range(4)]
    search_calls = 0

    def transport(payload: dict[str, Any], _key: str) -> dict[str, Any]:
        nonlocal search_calls
        if len(payload["messages"]) and "candidates" in str(payload["messages"][1]):
            return response(
                {
                    "decisions": [
                        {
                            "candidate_id": item.candidate_id,
                            "decision": "research",
                            "research_question": f"Каков benchmark для {item.candidate_id}?",
                        }
                        for item in items
                    ]
                }
            )
        if payload.get("tools"):
            search_calls += 1
            return search_response()
        user_payload = json.loads(payload["messages"][1]["content"])
        source = user_payload["candidate"]
        return response(
            {
                "candidate_id": source["candidate_id"],
                "resolved": True,
                "summary": source["summary"],
                "why_relevant": source["why_relevant"],
                "supporting_urls": [SOURCE_URL],
            }
        )

    result = OpenRouterContextValidator(settings(tmp_path), transport=transport).run(
        items, max_enrichments=99
    )

    assert result.enrichments_requested == 4
    assert result.enrichments_attempted == 3
    assert result.enrichments_succeeded == 3
    assert search_calls == 3
    assert result.candidates[3].outcome == "enrichment_cap_fail_open"
    assert result.candidates[3].enriched is False
