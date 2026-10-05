from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class TrendSignal(BaseModel):
    key: str
    label: str
    primary_contour: str
    topic: str
    current_event_ids: list[int]
    previous_event_ids: list[int]
    current_count: int = Field(ge=0)
    previous_count: int = Field(ge=0)
    distinct_sources: int = Field(ge=0)
    average_quality: float = Field(ge=0)
    trend_score: float
    score_features: dict[str, float]


class ProductItem(BaseModel):
    headline: str
    url: str | None = None
    event_id: int | None = None
    note: str | None = None


class ProductSection(BaseModel):
    id: str
    title: str
    summary: str
    items: list[ProductItem] = Field(default_factory=list)


class IntelligenceProductDraft(BaseModel):
    kind: Literal["weekly_map", "monthly_theses", "weekly_personas", "project_signals"]
    schema_version: str = "intelligence_product_v1"
    period_start: datetime
    period_end: datetime
    title: str
    summary: str
    sections: list[ProductSection]
    source_event_ids: list[int]
    rendered_html: str
    metadata: dict[str, Any] = Field(default_factory=dict)
