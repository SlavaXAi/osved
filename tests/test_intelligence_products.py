from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.config import (
    load_contour_registry,
    load_interest_profile,
    load_weekly_persona_registry,
)
from app.intelligence.models import TrendSignal
from app.intelligence.products import (
    build_monthly_theses,
    build_project_signals,
    build_weekly_map,
    build_weekly_persona_map,
)
from app.intelligence.trends import detect_trends

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 8, 25, 12, tzinfo=UTC)


def event(
    event_id: int,
    *,
    days_ago: int,
    contour: str,
    topic: str,
    headline: str,
    source_id: str,
    score: float = 6.0,
) -> dict:
    return {
        "id": event_id,
        "event_at": (NOW - timedelta(days=days_ago)).isoformat(),
        "headline": headline,
        "topics_json": f'["{topic}"]',
        "score_features_json": f'{{"primary_contour":"{contour}"}}',
        "source_id": source_id,
        "source_url": f"https://example.com/{event_id}",
        "score": score,
    }


def corpus() -> list[dict]:
    return [
        event(
            1,
            days_ago=1,
            contour="startups",
            topic="construction_ai",
            headline="YC-стартап запустил AI-агента для строительных смет",
            source_id="yc",
        ),
        event(
            2,
            days_ago=3,
            contour="startups",
            topic="construction_ai",
            headline="Фонд инвестировал в автоматизацию строительных закупок",
            source_id="fund",
        ),
        event(
            3,
            days_ago=9,
            contour="startups",
            topic="construction_ai",
            headline="Предыдущий раунд construction AI",
            source_id="archive",
        ),
        event(
            4,
            days_ago=2,
            contour="people_institutions",
            topic="frontier_ai",
            headline="Sam Altman представил новую стратегию OpenAI",
            source_id="openai",
        ),
    ]


def test_trend_engine_exposes_feature_breakdown() -> None:
    trends = detect_trends(corpus(), now=NOW)
    construction = next(item for item in trends if item.topic == "construction")
    assert construction.current_count == 2
    assert construction.previous_count == 1
    assert construction.distinct_sources == 2
    assert construction.score_features.keys() == {
        "volume",
        "acceleration",
        "source_diversity",
        "average_quality",
    }


def test_trends_normalize_topic_aliases_and_ignore_low_quality_noise() -> None:
    rows = [
        event(
            10,
            days_ago=1,
            contour="ai_software",
            topic="agents",
            headline="Новый агент планирует рабочие процессы",
            source_id="one",
        ),
        event(
            11,
            days_ago=2,
            contour="ai_software",
            topic="agentic_ai",
            headline="AI agents автоматизируют поддержку",
            source_id="two",
        ),
        event(
            12,
            days_ago=2,
            contour="ai_software",
            topic="agents",
            headline="Шумный агент без достаточного качества",
            source_id="noise",
            score=2.0,
        ),
    ]
    trends = detect_trends(rows, now=NOW)
    agents = next(item for item in trends if item.topic == "ai_agents")
    assert agents.current_count == 2
    assert agents.current_event_ids == [10, 11]


def test_trends_do_not_treat_search_lens_defaults_as_event_topics() -> None:
    rows = [
        event(
            event_id,
            days_ago=event_id - 20,
            contour="ai_software",
            topic="technology_news",
            headline="Компания уточнила дорожную карту продукта",
            source_id="web_agent_frontier_global",
        )
        for event_id in (21, 22)
    ]
    for row in rows:
        row["topics_json"] = (
            '["technology_news", "ai", "agents", "semiconductors", "compute_infrastructure"]'
        )

    trends = detect_trends(rows, now=NOW)

    assert [trend.topic for trend in trends] == ["ai_software"]


def test_trends_count_a_duplicate_event_id_only_once() -> None:
    row = event(
        30,
        days_ago=1,
        contour="startups",
        topic="startups",
        headline="Стартап запустил новый продукт",
        source_id="yc",
    )

    assert detect_trends([row, dict(row)], now=NOW) == []


def test_exceptional_single_event_remains_an_event_not_a_trend() -> None:
    row = event(
        31,
        days_ago=1,
        contour="physical_economy",
        topic="robotics",
        headline="Робот установил новый мировой рекорд",
        source_id="primary",
        score=10.0,
    )

    assert detect_trends([row], now=NOW) == []


def test_weekly_map_is_grounded_in_source_events_and_links() -> None:
    product = build_weekly_map(
        corpus(),
        load_contour_registry(PROJECT_ROOT / "tests/fixtures/config/contours.yaml"),
        now=NOW,
    )
    assert product.kind == "weekly_map"
    # The rendered comparison says "against one event previously", so that
    # previous event belongs to provenance as well.
    assert product.source_event_ids == [1, 2, 3]
    assert "https://example.com/1" in product.rendered_html
    assert "2 события за неделю против 1" in product.rendered_html


def test_monthly_theses_are_explicitly_candidates_for_review() -> None:
    product = build_monthly_theses(
        corpus(),
        load_contour_registry(PROJECT_ROOT / "tests/fixtures/config/contours.yaml"),
        now=NOW,
    )
    assert product.metadata["requires_user_review"] is True
    assert "не автоматически принятые убеждения" in product.summary
    assert product.source_event_ids == [1, 2, 3]


def test_persona_map_skips_people_without_material_updates() -> None:
    product = build_weekly_persona_map(
        corpus(),
        load_weekly_persona_registry(PROJECT_ROOT / "tests/fixtures/config/weekly-personas.yaml"),
        now=NOW,
    )
    assert [section.id for section in product.sections] == ["sam_altman"]
    assert product.source_event_ids == [4]


def test_empty_persona_map_is_explicitly_a_shared_corpus_scaffold() -> None:
    product = build_weekly_persona_map(
        corpus()[:3],
        load_weekly_persona_registry(PROJECT_ROOT / "tests/fixtures/config/weekly-personas.yaml"),
        now=NOW,
    )

    assert product.sections == []
    assert product.metadata["scaffold_only"] is True
    assert product.metadata["dedicated_persona_collection_enabled"] is False
    assert "не означает, что человек ничего не делал" in product.summary


def test_product_provenance_excludes_trends_omitted_by_render_cap(monkeypatch) -> None:
    rows = [
        event(
            event_id,
            days_ago=1,
            contour="startups",
            topic="startups",
            headline=f"Стартап-событие {event_id}",
            source_id=f"source-{event_id}",
        )
        for event_id in [*range(1, 11), *range(101, 111)]
    ]
    trends = [
        TrendSignal(
            key=f"startups:topic_{index}",
            label=f"topic {index}",
            primary_contour="startups",
            topic=f"topic_{index}",
            current_event_ids=[index],
            previous_event_ids=[100 + index],
            current_count=1,
            previous_count=2 if index == 1 else 1,
            distinct_sources=1,
            average_quality=6.0,
            trend_score=float(20 - index),
            score_features={},
        )
        for index in range(1, 11)
    ]
    monkeypatch.setattr("app.intelligence.products.detect_trends", lambda *_args, **_kwargs: trends)

    weekly = build_weekly_map(
        rows,
        load_contour_registry(PROJECT_ROOT / "tests/fixtures/config/contours.yaml"),
        now=NOW,
    )
    monthly = build_monthly_theses(
        rows,
        load_contour_registry(PROJECT_ROOT / "tests/fixtures/config/contours.yaml"),
        now=NOW,
    )

    assert 10 not in weekly.source_event_ids
    assert 110 not in weekly.source_event_ids
    assert 9 not in monthly.source_event_ids
    assert 109 not in monthly.source_event_ids
    assert "ослабевает" in monthly.sections[0].summary


def test_project_signals_use_current_projects_without_claiming_relevance_as_fact() -> None:
    profile = load_interest_profile(PROJECT_ROOT / "tests/fixtures/config/interests.yaml")
    profile = profile.model_copy(update={"current_projects": ["Пример проекта в строительстве"]})
    product = build_project_signals(
        corpus(),
        profile,
        now=NOW,
    )
    assert any("строитель" in section.title.casefold() for section in product.sections)
    assert "человеческой проверки" in product.rendered_html
