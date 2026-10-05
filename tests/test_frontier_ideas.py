from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import load_contour_registry, load_frontier_ideas_registry
from app.frontier_ideas.config import FrontierEvidenceRole
from app.frontier_ideas.models import (
    EpistemicEvidence,
    EpistemicStatus,
    FrontierIdeaCandidate,
    FrontierItemKind,
)
from app.frontier_ideas.scoring import (
    FrontierFeedbackSignal,
    FrontierReaction,
    score_frontier_idea,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 8, 25, 12, tzinfo=UTC)


def candidate(
    *,
    kind: FrontierItemKind = FrontierItemKind.EVERGREEN_IDEA,
    status: EpistemicStatus = EpistemicStatus.TESTABLE_HYPOTHESIS,
    event_at: datetime | None = None,
    origin_year: int | None = 2008,
    material_update: str | None = None,
    liveness: float = 0.7,
) -> FrontierIdeaCandidate:
    return FrontierIdeaCandidate(
        kind=kind,
        title="Полная эмуляция мозга как исследовательская программа",
        thesis=(
            "Достаточно точная цифровая модель структуры и динамики мозга может "
            "воспроизвести его когнитивные функции, но масштаб и требования неизвестны."
        ),
        epistemic_status=status,
        evidence=EpistemicEvidence(
            source_count=4,
            independent_source_count=2,
            primary_source_present=True,
            empirical_support=0.4,
            formal_rigor=0.7,
            falsifiability=0.65,
            objections_addressed=0.6,
        ),
        source_ids=["arxiv", "openalex"],
        source_urls=["https://arxiv.org/abs/1234.5678", "https://example.org/critique"],
        topics=["whole_brain_emulation", "consciousness"],
        observed_at=NOW,
        event_at=event_at,
        material_update=material_update,
        origin_year=origin_year,
        program_liveness=liveness,
        intellectual_surprise=0.9,
        consequence_magnitude=0.95,
        objection_quality=0.75,
        personal_relevance=0.9,
    )


def test_frontier_ideas_is_a_lens_within_the_seven_contour_map() -> None:
    frontier = load_frontier_ideas_registry(
        PROJECT_ROOT / "tests/fixtures/config/frontier-ideas.yaml"
    )
    contours = load_contour_registry(PROJECT_ROOT / "tests/fixtures/config/contours.yaml")

    assert frontier.lens.id == "frontier_ideas"
    assert set(frontier.lens.candidate_kinds) == {
        FrontierItemKind.EVENT,
        FrontierItemKind.EVERGREEN_IDEA,
    }
    assert len(contours.contours) == 7
    assert "frontier_ideas" not in {contour.id for contour in contours.contours}


def test_frontier_source_registry_is_free_and_separates_triggers_from_evidence() -> None:
    registry = load_frontier_ideas_registry(
        PROJECT_ROOT / "tests/fixtures/config/frontier-ideas.yaml"
    )
    sources = {source.id: source for source in registry.enabled_sources}

    assert sources.keys() >= {
        "openalex",
        "semantic_scholar",
        "crossref",
        "arxiv",
        "pubmed",
        "pmc",
        "lesswrong_curated",
        "alignment_forum",
        "astral_codex_ten",
        "manifold",
    }
    assert all(source.cost_tier == "free" for source in sources.values())
    assert {source.id for source in registry.evidence_sources} == {"arxiv", "pmc"}
    assert not sources["lesswrong_curated"].can_support_claim
    assert not sources["manifold"].can_support_claim
    assert sources["lesswrong_curated"].evidence_role == FrontierEvidenceRole.TRIGGER_ONLY


def test_candidate_model_keeps_events_and_evergreen_ideas_distinct() -> None:
    event = candidate(
        kind=FrontierItemKind.EVENT,
        event_at=NOW - timedelta(days=1),
        origin_year=None,
        material_update="Лаборатория опубликовала новый измеримый результат программы.",
    )
    evergreen = candidate()

    assert event.event_at is not None
    assert evergreen.origin_year == 2008
    assert evergreen.event_at is None

    with pytest.raises(ValidationError, match="event candidates require"):
        candidate(
            kind=FrontierItemKind.EVENT,
            event_at=None,
            origin_year=None,
            material_update=None,
        )

    with pytest.raises(ValidationError, match="cannot masquerade"):
        candidate(event_at=NOW - timedelta(days=1))


def test_feedback_changes_personal_ranking_but_not_epistemic_score() -> None:
    idea = candidate()
    superlike = score_frontier_idea(
        idea,
        feedback=FrontierFeedbackSignal(reaction=FrontierReaction.SUPERLIKE),
        now=NOW,
    )
    superdislike = score_frontier_idea(
        idea,
        feedback=FrontierFeedbackSignal(reaction=FrontierReaction.SUPERDISLIKE),
        now=NOW,
    )

    assert superlike.epistemic_score == superdislike.epistemic_score
    assert superlike.timeliness_score == superdislike.timeliness_score
    assert superlike.intrinsic_interest_score == superdislike.intrinsic_interest_score
    assert superlike.total_score - superdislike.total_score == pytest.approx(20)


def test_superlike_cannot_promote_fiction_to_empirical_standing() -> None:
    fiction = candidate(status=EpistemicStatus.FICTION)
    score = score_frontier_idea(
        fiction,
        feedback=FrontierFeedbackSignal(reaction=FrontierReaction.SUPERLIKE),
        now=NOW,
    )

    assert score.epistemic_score <= 25
    assert score.feedback_adjustment == 10


def test_fresh_event_decays_but_evergreen_idea_uses_program_liveness() -> None:
    fresh_event = candidate(
        kind=FrontierItemKind.EVENT,
        event_at=NOW - timedelta(days=1),
        origin_year=None,
        material_update="Исследователи начали новую проверяемую экспериментальную программу.",
    )
    old_event = fresh_event.model_copy(update={"event_at": NOW - timedelta(days=30)})
    evergreen = candidate(liveness=0.8)

    fresh_score = score_frontier_idea(fresh_event, now=NOW)
    old_score = score_frontier_idea(old_event, now=NOW)
    evergreen_score = score_frontier_idea(evergreen, now=NOW)

    assert fresh_score.timeliness_score > old_score.timeliness_score
    assert old_score.timeliness_score == 0
    assert evergreen_score.timeliness_score == pytest.approx(87)
