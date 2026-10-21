"""Neutral example catalogue; replace it with the owner-approved interests."""

from __future__ import annotations

import re
import unicodedata

from pydantic import BaseModel, Field

PROFILE = {
    "version": "2026-10-neutral",
    "lens": "Learning, practical work, decisions and their consequences",
    "priorities": ["documented learning", "practical applications", "historical impact"],
    "strong_topics": ["science", "technology", "thought"],
    "avoid": ["unsupported motivational stories"],
    "style": "Concrete, calm narrative supported by sources.",
    "known_people": [],
    "excluded_survey_signals": [],
}


class Candidate(BaseModel):
    name: str = Field(min_length=3, max_length=90)
    english_name: str = Field(min_length=3, max_length=90)
    aliases: list[str] = Field(default_factory=list, max_length=6)
    topic: str = Field(min_length=3, max_length=30)
    era: str = Field(min_length=3, max_length=30)
    region: str = Field(min_length=3, max_length=50)
    life_arc: str = Field(min_length=3, max_length=60)
    living: bool
    life_scale: int = Field(ge=1, le=5)
    historical_impact: int = Field(ge=1, le=5)
    exceptional_quality: int = Field(ge=1, le=5)
    rationale: str = Field(min_length=30, max_length=500)
    image_query: str = Field(min_length=3, max_length=120)
    discovery_urls: list[str] = Field(default_factory=list, max_length=3)
    discovery_quote: str = ""


class Discovery(BaseModel):
    candidates: list[Candidate] = Field(min_length=1, max_length=8)


def identity_key(name: str) -> str:
    value = unicodedata.normalize("NFKD", name).casefold()
    value = "".join(c for c in value if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9а-яё]+", "-", value).strip("-")


# name, canonical English name, topic, era, region, life pattern, living, image search
SEEDS = [
    (
        "Ада Лавлейс",
        "Ada Lovelace",
        "science",
        "industrial",
        "Britain",
        "mathematics_computing",
        False,
        "Ada Lovelace",
    ),
    (
        "Майкл Фарадей",
        "Michael Faraday",
        "science",
        "industrial",
        "Britain",
        "experiments_discovery",
        False,
        "Michael Faraday",
    ),
    (
        "Сергей Королёв",
        "Sergei Korolev",
        "space",
        "modern",
        "Russia",
        "engineering_space",
        False,
        "Sergei Korolev",
    ),
]


def seed_candidates() -> list[Candidate]:
    return [
        Candidate(
            name=r[0],
            english_name=r[1],
            topic=r[2],
            era=r[3],
            region=r[4],
            life_arc=r[5],
            living=r[6],
            image_query=r[7],
            life_scale=5,
            historical_impact=4,
            exceptional_quality=4,
            rationale="Стартовый пример для исследования по источникам; изучить масштаб "
            "жизни, решения и влияние по источникам.",
        )
        for r in SEEDS
    ]
