from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Self
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator, model_validator


class FrontierItemKind(StrEnum):
    """Why an item entered the lens.

    An event needs a dated material change. An evergreen idea is allowed to be old;
    it earns attention through liveness and consequences rather than fake freshness.
    """

    EVENT = "event"
    EVERGREEN_IDEA = "evergreen_idea"


class EpistemicStatus(StrEnum):
    """Type of claim, not a probability that the claim is true."""

    EMPIRICAL = "empirical"
    FORMAL_MODEL = "formal_model"
    TESTABLE_HYPOTHESIS = "testable_hypothesis"
    PHILOSOPHY = "philosophy"
    FUTURISM = "futurism"
    FICTION = "fiction"


class EpistemicEvidence(BaseModel):
    """Evidence features controlled by validation, never by user preference."""

    source_count: int = Field(ge=1, le=50)
    independent_source_count: int = Field(default=0, ge=0, le=50)
    primary_source_present: bool = False
    empirical_support: float = Field(default=0, ge=0, le=1)
    formal_rigor: float = Field(default=0, ge=0, le=1)
    falsifiability: float = Field(default=0, ge=0, le=1)
    objections_addressed: float = Field(default=0, ge=0, le=1)

    @model_validator(mode="after")
    def independent_sources_cannot_exceed_total(self) -> Self:
        if self.independent_source_count > self.source_count:
            raise ValueError("independent_source_count cannot exceed source_count")
        return self


class FrontierIdeaCandidate(BaseModel):
    """A normalized candidate emitted by the Frontier Ideas discovery lens."""

    kind: FrontierItemKind
    title: str = Field(min_length=12, max_length=180)
    thesis: str = Field(min_length=40, max_length=1200)
    epistemic_status: EpistemicStatus
    evidence: EpistemicEvidence
    source_ids: list[str] = Field(min_length=1, max_length=12)
    source_urls: list[str] = Field(min_length=1, max_length=12)
    topics: list[str] = Field(default_factory=list, max_length=12)

    observed_at: datetime
    event_at: datetime | None = None
    material_update: str | None = Field(default=None, max_length=600)
    origin_year: int | None = Field(default=None, ge=1000, le=3000)

    program_liveness: float = Field(ge=0, le=1)
    intellectual_surprise: float = Field(ge=0, le=1)
    consequence_magnitude: float = Field(ge=0, le=1)
    objection_quality: float = Field(ge=0, le=1)
    personal_relevance: float = Field(ge=0, le=1)

    @field_validator("source_ids")
    @classmethod
    def source_ids_are_unique(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values if value.strip()]
        if not normalized:
            raise ValueError("at least one source id is required")
        if len(normalized) != len(set(normalized)):
            raise ValueError("source ids must be unique")
        return normalized

    @field_validator("source_urls")
    @classmethod
    def source_urls_are_http_and_unique(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            url = value.strip()
            parsed = urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError("source URLs must be absolute HTTP(S) URLs")
            if url not in normalized:
                normalized.append(url)
        return normalized

    @model_validator(mode="after")
    def event_and_evergreen_fields_do_not_blur(self) -> Self:
        if self.observed_at.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")
        if self.kind == FrontierItemKind.EVENT:
            if self.event_at is None or self.event_at.tzinfo is None:
                raise ValueError("event candidates require a timezone-aware event_at")
            if not self.material_update or len(self.material_update.strip()) < 20:
                raise ValueError("event candidates require a concrete material_update")
            if self.origin_year is not None:
                raise ValueError("event candidates must not use origin_year")
            return self

        if self.event_at is not None or self.material_update is not None:
            raise ValueError("evergreen ideas cannot masquerade as dated events")
        if self.origin_year is None:
            raise ValueError("evergreen ideas require origin_year")
        return self
