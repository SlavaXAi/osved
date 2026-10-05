from __future__ import annotations

from enum import StrEnum
from typing import Self
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator, model_validator

from app.frontier_ideas.models import EpistemicStatus, FrontierItemKind


class FrontierSourceKind(StrEnum):
    API = "api"
    RSS = "rss"


class FrontierSourceRole(StrEnum):
    PRIMARY_REPOSITORY = "primary_repository"
    RESEARCH_INDEX = "research_index"
    IDEA_TRIGGER = "idea_trigger"
    FORECAST_TRIGGER = "forecast_trigger"


class FrontierEvidenceRole(StrEnum):
    PRIMARY = "primary"
    DISCOVERY_ONLY = "discovery_only"
    TRIGGER_ONLY = "trigger_only"


class FrontierAuthMode(StrEnum):
    NONE = "none"
    OPTIONAL_FREE_KEY = "optional_free_key"
    FREE_KEY = "free_key"


class FrontierSourceConfig(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    title: str = Field(min_length=3)
    kind: FrontierSourceKind
    role: FrontierSourceRole
    evidence_role: FrontierEvidenceRole
    endpoint: str
    enabled: bool = True
    auth: FrontierAuthMode = FrontierAuthMode.NONE
    cost_tier: str = Field(default="free", pattern=r"^free$")
    request_cap_per_run: int = Field(default=1, ge=1, le=5)
    topics: list[str] = Field(default_factory=list)
    notes: str = Field(min_length=20)

    @field_validator("endpoint")
    @classmethod
    def endpoint_is_http(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("frontier source endpoint must be absolute HTTPS")
        return value

    @property
    def can_support_claim(self) -> bool:
        return self.evidence_role == FrontierEvidenceRole.PRIMARY


class FrontierDiscoveryLens(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    title: str = Field(min_length=3)
    enabled: bool = True
    prompt_version: str
    candidate_kinds: list[FrontierItemKind] = Field(min_length=1)
    epistemic_statuses: list[EpistemicStatus] = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1)
    max_candidates: int = Field(default=5, ge=1, le=12)
    event_lookback_days: int = Field(default=14, ge=1, le=60)
    min_epistemic_score: float = Field(default=20, ge=0, le=100)
    instructions: str = Field(min_length=100)


class FrontierIdeasRegistry(BaseModel):
    version: int = Field(ge=1)
    lens: FrontierDiscoveryLens
    sources: list[FrontierSourceConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def source_ids_are_unique_and_referenced(self) -> Self:
        source_ids = [source.id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("frontier source ids must be unique")
        unknown = set(self.lens.source_ids) - set(source_ids)
        if unknown:
            raise ValueError(f"frontier lens references unknown sources: {sorted(unknown)}")
        if self.lens.id != "frontier_ideas":
            raise ValueError("frontier ideas registry must use the frontier_ideas lens id")
        return self

    @property
    def enabled_sources(self) -> list[FrontierSourceConfig]:
        selected = set(self.lens.source_ids)
        return [source for source in self.sources if source.enabled and source.id in selected]

    @property
    def evidence_sources(self) -> list[FrontierSourceConfig]:
        return [source for source in self.enabled_sources if source.can_support_claim]
