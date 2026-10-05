"""Derived OSVED products: trends, weekly maps, theses and persona dossiers."""

from app.intelligence.products import (
    build_monthly_theses,
    build_project_signals,
    build_weekly_map,
    build_weekly_persona_map,
)
from app.intelligence.service import INTELLIGENCE_KINDS, IntelligenceService
from app.intelligence.trends import detect_trends

__all__ = [
    "build_monthly_theses",
    "build_project_signals",
    "build_weekly_map",
    "build_weekly_persona_map",
    "detect_trends",
    "INTELLIGENCE_KINDS",
    "IntelligenceService",
]
