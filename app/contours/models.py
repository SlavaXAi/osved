from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

LEGACY_CONTOUR_ALIASES = {
    "company_capital": "startups",
    "china_manufacturing": "world_economy",
    "physical_economy": "startups",
    "philosophy_civilization": "human_civilization",
}


def canonical_contour_id(contour_id: str, event: dict[str, Any] | None = None) -> str:
    """Resolve historical labels for display/ranking without rewriting stored history.

    The former physical branch covered much more than space, so use explicit event
    context when available and otherwise preserve the broad technology-company home.
    """
    if contour_id == "physical_economy" and event:
        text = " ".join(
            str(event.get(key) or "") for key in ("headline", "topics_json", "topics", "summary")
        ).casefold()
        if any(
            marker in text
            for marker in (
                "spacex",
                "spaceflight",
                "satellite",
                "spacecraft",
                '"space"',
                "космос",
                "космичес",
                "спутник",
                "орбит",
                "starship",
                "航天",
                "火箭",
            )
        ):
            return "space"
        if any(marker in text for marker in ("ai agent", "software", "автоматиза", "агент")):
            return "ai_software"
    return LEGACY_CONTOUR_ALIASES.get(contour_id, contour_id)


class ContourConfig(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    title: str = Field(min_length=3)
    mission: str = Field(min_length=30)
    enabled: bool = True
    include_topics: list[str] = Field(default_factory=list)
    include_keywords: list[str] = Field(default_factory=list)
    exclude_keywords: list[str] = Field(default_factory=list)
    required_any_regions: list[str] = Field(default_factory=list)
    required_any_keywords: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)
    key_questions: list[str] = Field(default_factory=list, min_length=1)
    max_age_days: int = Field(default=7, ge=1, le=31)
    search_lens_ids: list[str] = Field(default_factory=list)
    source_hints: list[str] = Field(default_factory=list)


class ContourRegistry(BaseModel):
    version: int = Field(ge=1)
    contours: list[ContourConfig] = Field(min_length=1)
    aliases: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def contour_ids_are_unique(self) -> ContourRegistry:
        ids = [contour.id for contour in self.contours]
        if len(ids) != len(set(ids)):
            raise ValueError("contour ids must be unique")
        if set(self.aliases) & set(ids):
            raise ValueError("contour alias cannot shadow a current id")
        if any(target not in ids for target in self.aliases.values()):
            raise ValueError("contour aliases must point to a current id")
        return self

    @property
    def enabled_contours(self) -> list[ContourConfig]:
        return [contour for contour in self.contours if contour.enabled]

    def by_id(self, *, include_aliases: bool = True) -> dict[str, ContourConfig]:
        result = {contour.id: contour for contour in self.contours}
        if include_aliases:
            result.update({alias: result[target] for alias, target in self.aliases.items()})
        return result
