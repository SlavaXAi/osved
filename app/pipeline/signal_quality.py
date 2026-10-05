"""Cheap, auditable event spread and evidence accounting.

Source popularity is an observation, not proof. This module never upgrades a
trigger to independent evidence merely because several accounts repeated it.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from urllib.parse import urlsplit

from app.pipeline.dedupe import canonical_evidence_url

TRIGGER_ROLES = {"radar", "author_radar", "trigger_only", "discovery", "history", "own_history"}
SOCIAL_HOSTS = {"t.me", "telegram.me", "x.com", "twitter.com", "reddit.com", "youtube.com",
                "youtu.be", "weibo.com", "zhihu.com", "v2ex.com"}


def _host(url: Any) -> str:
    try:
        return (urlsplit(str(url or "")).hostname or "").casefold().removeprefix("www.")
    except ValueError:
        return ""


def assess_signal_quality(observations: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Separate repetition, independent publisher families and verified evidence.

    Expected optional fields: id, source_id, source_url, source_kind/platform,
    source_family/publisher_family, original_source_url/syndicated_from,
    source_role/evidence_role, admission_mode, verified, published_at. The result
    is descriptive; it does not change a card's factual validation status.
    """
    rows = list(observations)
    seen: set[str] = set()
    sources: set[str] = set()
    platforms: set[str] = set()
    families: set[str] = set()
    primary: set[str] = set()
    independent: set[str] = set()
    trigger_count = 0
    references: list[int | str] = []
    for row in rows:
        url = str(row.get("source_url") or row.get("url") or "")
        source = str(row.get("source_id") or _host(url) or "unknown")
        identity = str(row.get("id") or "") or f"{source}:{row.get('external_id') or url}"
        if identity in seen:
            continue
        seen.add(identity)
        sources.add(source)
        if platform := str(row.get("platform") or row.get("source_kind") or ""):
            platforms.add(platform)
        original = row.get("original_source_url") or row.get("syndicated_from")
        explicit_family = row.get("source_family") or row.get("publisher_family")
        host = _host(url)
        # Telegram/Reddit/X are delivery platforms, not one editorial publisher.
        publisher = source if host in SOCIAL_HOSTS else host
        family = str(explicit_family or _host(original) or publisher or source)
        families.add(family)
        role = str(row.get("evidence_role") or row.get("source_role") or "")
        admission = str(row.get("admission_mode") or "")
        is_trigger = (
            role in TRIGGER_ROLES
            or str(row.get("source_role") or "") in TRIGGER_ROLES
            or admission in {"trigger_only", "history"}
        )
        trigger_count += int(is_trigger)
        if row.get("id") is not None:
            references.append(row["id"])
        # A publisher's own assertion is primary evidence of the assertion. The
        # fact-checking validator still decides which factual claims it supports.
        source_role = str(row.get("source_role") or "")
        if not is_trigger and (
            source_role == "primary"
            or admission == "primary_evidence"
            or (not source_role and role == "primary")
        ):
            primary.add(family)
        elif not is_trigger and role == "independent" and row.get("verified") is True:
            independent.add(family)
    independent -= primary
    spread_score = round(min(100.0, len(sources) * 10 + max(0, len(platforms) - 1) * 10), 1)
    evidence_status = (
        "primary_and_independent"
        if primary and independent
        else "primary_only"
        if primary
        else "independent_only"
        if independent
        else "unverified"
    )
    return {
        "version": "signal_quality_v1",
        "mention_count": len(seen),
        "source_count": len(sources),
        "platform_count": len(platforms),
        "publisher_family_count": len(families),
        "primary_family_count": len(primary),
        "verified_independent_family_count": len(independent),
        "trigger_count": trigger_count,
        "spread_score": spread_score,
        "evidence_status": evidence_status,
        "requires_verification": not (primary and independent),
        "reference_ids": references,
        "note": "Spread measures attention; evidence status does not certify every claim.",
    }


def syndication_identity(observation: dict[str, Any]) -> str | None:
    """Known shared original used to collapse syndications without deleting mentions."""
    original = observation.get("original_source_url") or observation.get("syndicated_from")
    return canonical_evidence_url(str(original)) if original else None
