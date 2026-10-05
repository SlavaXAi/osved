"""The "Frontier ideas" discovery lens.

This package intentionally lives outside :mod:`app.contours`: it discovers
epistemically unusual ideas across several existing contours instead of adding a
tenth topic bucket.
"""

from app.frontier_ideas.config import FrontierIdeasRegistry
from app.frontier_ideas.models import (
    EpistemicEvidence,
    EpistemicStatus,
    FrontierIdeaCandidate,
    FrontierItemKind,
)
from app.frontier_ideas.scoring import (
    FrontierFeedbackSignal,
    FrontierIdeaScore,
    FrontierReaction,
    score_frontier_idea,
)

__all__ = [
    "EpistemicEvidence",
    "EpistemicStatus",
    "FrontierFeedbackSignal",
    "FrontierIdeaCandidate",
    "FrontierIdeaScore",
    "FrontierIdeasRegistry",
    "FrontierItemKind",
    "FrontierReaction",
    "score_frontier_idea",
]
