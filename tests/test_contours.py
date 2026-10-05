from __future__ import annotations

from pathlib import Path

import yaml

from app.contours.classifier import classify_event
from app.contours.models import ContourRegistry, canonical_contour_id
from app.contours.portfolio import event_primary_contour, mix_contour_portfolio

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def registry() -> ContourRegistry:
    payload = yaml.safe_load((PROJECT_ROOT / "tests/fixtures/config/contours.yaml").read_text())
    return ContourRegistry.model_validate(payload)


def test_registry_has_the_seven_agreed_contours() -> None:
    contours = registry()
    assert len(contours.contours) == 7
    assert {contour.id for contour in contours.contours} == {
        "startups",
        "ai_software",
        "space",
        "world_economy",
        "geopolitics",
        "people_institutions",
        "human_civilization",
    }


def test_classifier_uses_content_not_only_the_search_lens() -> None:
    assignment = classify_event(
        {
            "headline": "Китайская фабрика запустила линию промышленных роботов",
            "topics": ["manufacturing", "robotics"],
            "regions": ["china"],
        },
        registry(),
        source_lens_id="frontier_global",
    )
    assert assignment.primary_contour == "startups"
    assert "china" in assignment.cross_topic_tags
    assert "manufacturing" in assignment.cross_topic_tags
    assert assignment.as_metadata()["taxonomy_version"] == 2


def test_classifier_keeps_weak_unknown_event_unclassified() -> None:
    assignment = classify_event(
        {"headline": "Компания сообщила о небольшом внутреннем изменении"},
        registry(),
    )
    assert assignment.primary_contour is None
    assert assignment.explanation == "insufficient_contour_signal"

    lens_only = classify_event(
        {"headline": "Компания сообщила о небольшом внутреннем изменении"},
        registry(),
        source_lens_id="frontier_global",
    )
    assert lens_only.primary_contour is None


def test_geography_is_cross_topic_not_an_automatic_primary_contour() -> None:
    texas = classify_event(
        {
            "headline": "В Техасе открыли фабрику полупроводников",
            "topics": ["manufacturing", "semiconductors"],
            "regions": ["united_states"],
        },
        registry(),
        source_lens_id="frontier_global",
    )
    assert texas.primary_contour != "china_manufacturing"

    china = classify_event(
        {
            "headline": "В Китае открыли фабрику полупроводников",
            "topics": ["manufacturing", "semiconductors"],
            "regions": ["china"],
        },
        registry(),
    )
    assert china.primary_contour == texas.primary_contour == "startups"
    assert "china" in china.cross_topic_tags
    assert "china" not in texas.cross_topic_tags


def test_space_and_philosophy_have_distinct_current_homes() -> None:
    space = classify_event(
        {"headline": "SpaceX запустила спутник на орбиту", "topics": ["space"]}, registry()
    )
    assert space.primary_contour == "space"
    philosophy = classify_event(
        {"headline": "Новый спор о сознании и AI welfare", "topics": ["consciousness"]}, registry()
    )
    assert philosophy.primary_contour == "human_civilization"
    assert registry().by_id()["human_civilization"].title == "Философия, человек и цивилизация"


def test_historical_contour_ids_remain_resolvable_without_rewriting_history() -> None:
    assert registry().by_id()["company_capital"].id == "startups"
    assert canonical_contour_id("physical_economy", {"headline": "SpaceX launch"}) == "space"
    historical = {"primary_contour": "physical_economy", "headline": "Завод промышленных роботов"}
    assert event_primary_contour(historical) == "startups"
    assert historical["primary_contour"] == "physical_economy"


def test_chinese_content_classification_and_disabled_registry_are_safe() -> None:
    assignment = classify_event(
        {"headline": "中国火箭将卫星送入轨道", "topics": ["space"]}, registry()
    )
    assert assignment.primary_contour == "space"
    assert "china" in assignment.cross_topic_tags
    disabled = registry()
    for contour in disabled.contours:
        contour.enabled = False
    assert (
        classify_event({"headline": "SpaceX launched a satellite"}, disabled).primary_contour
        is None
    )


def test_portfolio_mixer_softly_breaks_a_single_contour_monopoly() -> None:
    candidates = [
        {"id": 1, "score": 7.0, "primary_contour": "ai_software"},
        {"id": 2, "score": 6.8, "primary_contour": "ai_software"},
        {"id": 3, "score": 6.5, "primary_contour": "startups"},
    ]
    mixed = mix_contour_portfolio(candidates, limit=3)
    assert [row["id"] for row in mixed] == [1, 3, 2]


def test_portfolio_mixer_does_not_drop_exceptional_same_contour_event() -> None:
    candidates = [
        {"id": 1, "score": 10.0, "primary_contour": "ai_software"},
        {"id": 2, "score": 9.5, "primary_contour": "ai_software"},
        {"id": 3, "score": 5.0, "primary_contour": "startups"},
    ]
    mixed = mix_contour_portfolio(candidates, limit=2)
    assert [row["id"] for row in mixed] == [1, 2]


def test_portfolio_preserves_bounded_exploration_after_contour_reranking() -> None:
    candidates = [{"id": index, "score": 7} for index in range(8)]
    candidates.append({"id": 99, "score": 6.5, "discovery_mode": "explore"})
    mixed = mix_contour_portfolio(candidates, limit=7)
    assert len(mixed) == 7
    assert mixed[-1]["id"] == 99
    assert mixed[-1]["exploration_slot"]
    no_exploration = mix_contour_portfolio(candidates, limit=7, exploration_fraction=0)
    assert 99 not in [event["id"] for event in no_exploration]
