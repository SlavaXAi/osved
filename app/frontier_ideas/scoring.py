from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from app.frontier_ideas.models import (
    EpistemicStatus,
    FrontierIdeaCandidate,
    FrontierItemKind,
)

SCORING_VERSION = "frontier_ideas_v1"

# This is a grounding prior, not a truth probability. The ceiling prevents a
# brilliantly presented fictional or futurist idea from acquiring the epistemic
# standing of an experiment simply through high user interest.
STATUS_GROUNDING_PRIOR: dict[EpistemicStatus, float] = {
    EpistemicStatus.EMPIRICAL: 0.90,
    EpistemicStatus.FORMAL_MODEL: 0.78,
    EpistemicStatus.TESTABLE_HYPOTHESIS: 0.62,
    EpistemicStatus.PHILOSOPHY: 0.42,
    EpistemicStatus.FUTURISM: 0.28,
    EpistemicStatus.FICTION: 0.10,
}
STATUS_GROUNDING_CEILING: dict[EpistemicStatus, float] = {
    EpistemicStatus.EMPIRICAL: 100,
    EpistemicStatus.FORMAL_MODEL: 92,
    EpistemicStatus.TESTABLE_HYPOTHESIS: 80,
    EpistemicStatus.PHILOSOPHY: 60,
    EpistemicStatus.FUTURISM: 45,
    EpistemicStatus.FICTION: 25,
}


class FrontierReaction(StrEnum):
    SUPERDISLIKE = "superdislike"
    DISLIKE = "dislike"
    LIKE = "like"
    SUPERLIKE = "superlike"


FEEDBACK_WEIGHTS: dict[FrontierReaction, int] = {
    FrontierReaction.SUPERDISLIKE: -2,
    FrontierReaction.DISLIKE: -1,
    FrontierReaction.LIKE: 1,
    FrontierReaction.SUPERLIKE: 2,
}


class FrontierFeedbackSignal(BaseModel):
    """A preference signal; it may adjust ranking but never evidence."""

    reaction: FrontierReaction
    comment: str = Field(default="", max_length=600)

    @property
    def weight(self) -> int:
        return FEEDBACK_WEIGHTS[self.reaction]


class FrontierIdeaScore(BaseModel):
    epistemic_score: float = Field(ge=0, le=100)
    timeliness_score: float = Field(ge=0, le=100)
    intrinsic_interest_score: float = Field(ge=0, le=100)
    feedback_adjustment: float = Field(ge=-10, le=10)
    total_score: float = Field(ge=0, le=100)
    scoring_version: str = SCORING_VERSION


def _epistemic_score(candidate: FrontierIdeaCandidate) -> float:
    evidence = candidate.evidence
    independent_ratio = min(1.0, evidence.independent_source_count / 3)
    evidence_signal = (
        0.30 * evidence.empirical_support
        + 0.22 * evidence.formal_rigor
        + 0.18 * evidence.falsifiability
        + 0.15 * evidence.objections_addressed
        + 0.10 * float(evidence.primary_source_present)
        + 0.05 * independent_ratio
    )
    raw = 100 * (0.45 * STATUS_GROUNDING_PRIOR[candidate.epistemic_status] + 0.55 * evidence_signal)
    return round(min(raw, STATUS_GROUNDING_CEILING[candidate.epistemic_status]), 3)


def _timeliness_score(candidate: FrontierIdeaCandidate, *, now: datetime) -> float:
    if candidate.kind == FrontierItemKind.EVERGREEN_IDEA:
        # Evergreen material does not pretend to be fresh. It stays discoverable
        # when an actual research programme around it remains alive.
        return round(100 * (0.35 + 0.65 * candidate.program_liveness), 3)

    assert candidate.event_at is not None
    age_days = max(0.0, (now - candidate.event_at.astimezone(UTC)).total_seconds() / 86400)
    return round(100 * max(0.0, 1 - age_days / 21), 3)


def _intrinsic_interest_score(candidate: FrontierIdeaCandidate) -> float:
    return round(
        100
        * (
            0.25 * candidate.program_liveness
            + 0.25 * candidate.intellectual_surprise
            + 0.25 * candidate.consequence_magnitude
            + 0.15 * candidate.objection_quality
            + 0.10 * candidate.personal_relevance
        ),
        3,
    )


def score_frontier_idea(
    candidate: FrontierIdeaCandidate,
    *,
    feedback: FrontierFeedbackSignal | None = None,
    now: datetime | None = None,
) -> FrontierIdeaScore:
    """Score an idea while preserving the preference/evidence boundary.

    Feedback is deliberately added only after the epistemic and timeliness scores
    have been finalized. A superlike can surface a speculative idea; it cannot make
    that idea more empirical, rigorous, falsifiable or independently supported.
    """

    current_time = now or datetime.now(UTC)
    if current_time.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    epistemic = _epistemic_score(candidate)
    timeliness = _timeliness_score(candidate, now=current_time.astimezone(UTC))
    intrinsic_interest = _intrinsic_interest_score(candidate)
    feedback_adjustment = float((feedback.weight if feedback else 0) * 5)
    total = 0.45 * epistemic + 0.15 * timeliness + 0.40 * intrinsic_interest + feedback_adjustment
    return FrontierIdeaScore(
        epistemic_score=epistemic,
        timeliness_score=timeliness,
        intrinsic_interest_score=intrinsic_interest,
        feedback_adjustment=feedback_adjustment,
        total_score=round(min(100.0, max(0.0, total)), 3),
    )
