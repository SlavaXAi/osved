from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.research.detail_policy import DETAIL_PROMPT_VERSION, DETAIL_SCHEMA_VERSION
from app.research.facts import EditorialHook, EvidenceQuote, FactConflict, FactCoverage

DETAIL_FACT_SNAPSHOT_VERSION = "detail_fact_snapshot_v1"
NewsReaderDepth = Literal["legacy", "expanded", "article"]


class DetailResearchPlan(BaseModel):
    """A deliberately finite research plan; it is not an autonomous agent loop."""

    model_config = ConfigDict(extra="forbid")

    contour: str = Field(min_length=2, max_length=80)
    known_fact_count: int = Field(ge=0, le=100)
    gaps: list[str] = Field(default_factory=list, max_length=6)
    queries: list[str] = Field(min_length=1, max_length=2)
    desired_source_roles: list[Literal["primary", "independent", "context", "official_data"]] = (
        Field(min_length=1, max_length=4)
    )
    stop_condition: str = Field(min_length=20, max_length=400)


class DetailSourceGate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid: bool
    eligible_document_ids: list[int] = Field(default_factory=list)
    distinct_domains: list[str] = Field(default_factory=list)
    primary_document_ids: list[int] = Field(default_factory=list)
    independent_document_ids: list[int] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class DetailFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fact_id: int = Field(ge=1)
    text: str = Field(min_length=10, max_length=600)
    document_ids: list[int] = Field(min_length=1, max_length=3)
    evidence_quotes: list[EvidenceQuote] = Field(min_length=1, max_length=6)
    kind: Literal[
        "release",
        "capability",
        "benchmark",
        "pricing",
        "availability",
        "limitation",
    ]


class DetailFactSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = DETAIL_FACT_SNAPSHOT_VERSION
    event_id: int = Field(ge=1)
    event_type: str
    facts: list[DetailFact] = Field(min_length=3, max_length=14)
    coverage: FactCoverage
    conflicts: list[FactConflict] = Field(default_factory=list, max_length=4)
    editorial_hook: EditorialHook | None = None
    document_ids: list[int] = Field(min_length=1, max_length=16)
    source_domains: list[str] = Field(min_length=1, max_length=16)
    warnings: list[str] = Field(default_factory=list)


class DeepArticleParagraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=60, max_length=900)
    fact_ids: list[int] = Field(min_length=1, max_length=6)


class DeepArticleSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    heading: str = Field(min_length=3, max_length=80)
    paragraphs: list[DeepArticleParagraph] = Field(min_length=1, max_length=3)


class DeepArticleHighlight(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=20, max_length=320)
    fact_ids: list[int] = Field(min_length=1, max_length=3)


class DeepSourceAnchor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phrase: str = Field(min_length=3, max_length=120)
    fact_id: int = Field(ge=1)
    document_id: int = Field(ge=1)
    role: Literal["primary", "independent", "context"]


class DeepArticleDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    headline: str = Field(min_length=10, max_length=140)
    headline_fact_ids: list[int] = Field(min_length=1, max_length=3)
    lead: str = Field(min_length=80, max_length=600)
    lead_fact_ids: list[int] = Field(min_length=1, max_length=6)
    sections: list[DeepArticleSection] = Field(min_length=2, max_length=5)
    highlight: DeepArticleHighlight | None = None
    limitations: DeepArticleParagraph | None = None
    source_anchors: list[DeepSourceAnchor] = Field(min_length=1, max_length=12)
    used_document_ids: list[int] = Field(min_length=1, max_length=16)


class GeneratedDeepArticleDraft(DeepArticleDraft):
    """Provider schema keeps optional blocks required-but-nullable for strict JSON."""

    highlight: DeepArticleHighlight | None
    limitations: DeepArticleParagraph | None


class DetailProviderCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    response_id: str | None = None
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    reported_cost_usd: float = Field(ge=0)
    accounted_cost_usd: float = Field(ge=0)
    billing_metadata: dict[str, Any] = Field(default_factory=dict)


class DetailArticleBuildResult(BaseModel):
    """Transport-independent artifact consumed by the queue worker."""

    model_config = ConfigDict(extra="forbid")

    event_id: int = Field(ge=1)
    generation_key: str = Field(min_length=16, max_length=128)
    schema_version: str = DETAIL_SCHEMA_VERSION
    prompt_version: str = DETAIL_PROMPT_VERSION
    model: str
    depth: NewsReaderDepth = "legacy"
    reader_context: dict[str, Any] = Field(default_factory=dict)
    provider_response_metadata: dict[str, Any] = Field(default_factory=dict)
    research_plan: dict[str, Any]
    fact_snapshot: dict[str, Any]
    document_ids: list[int] = Field(min_length=1, max_length=16)
    structured_output: dict[str, Any]
    rendered_html: str = Field(min_length=1, max_length=40000)
    source_urls: list[str] = Field(min_length=1, max_length=16)
    validation_status: Literal["valid", "invalid", "insufficient_evidence"]
    validation_errors: list[str] = Field(default_factory=list)
    cost_usd: float = Field(ge=0)
    expires_at: str
