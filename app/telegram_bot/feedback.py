from __future__ import annotations

from enum import StrEnum


class Reaction(StrEnum):
    LIKE = "like"
    DISLIKE = "dislike"
    SUPERLIKE = "superlike"
    SUPERDISLIKE = "superdislike"

    @property
    def is_super(self) -> bool:
        return self in {Reaction.SUPERLIKE, Reaction.SUPERDISLIKE}


class FeedbackAction(StrEnum):
    LIKE = "like"
    DISLIKE = "dislike"
    SUPERLIKE = "superlike"
    SUPERDISLIKE = "superdislike"
    COMMENT = "comment"
    SILENT = "silent"
    BACK = "back"


def reaction_from_action(action: FeedbackAction) -> Reaction | None:
    if action in {
        FeedbackAction.LIKE,
        FeedbackAction.DISLIKE,
        FeedbackAction.SUPERLIKE,
        FeedbackAction.SUPERDISLIKE,
    }:
        return Reaction(action.value)
    return None
