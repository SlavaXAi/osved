"""Example profiles for the inherited regression suite; never used by the service."""

from pathlib import Path

import pytest

CONFIG = Path(__file__).parent / "fixtures" / "config"
PROFILE_ENV = {
    "SOURCES_FILE": "sources.yaml",
    "INTERESTS_FILE": "interests.yaml",
    "STYLE_PROFILE_FILE": "style-profile.yaml",
    "CONTOURS_FILE": "contours.yaml",
    "SEARCH_AGENTS_FILE": "search-agents.yaml",
    "X_AUTHORS_FILE": "x-authors.yaml",
    "WEEKLY_PERSONAS_FILE": "weekly-personas.yaml",
    "FRONTIER_IDEAS_FILE": "frontier-ideas.yaml",
}


@pytest.fixture(autouse=True)
def example_profiles(monkeypatch):
    for key, filename in PROFILE_ENV.items():
        monkeypatch.setenv(key, str(CONFIG / filename))
