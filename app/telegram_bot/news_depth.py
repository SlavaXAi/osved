"""Progressive disclosure of a validated card; opening context costs no model call."""

from __future__ import annotations

import html
import json
from typing import Any

from app.pipeline.cards import linked_headline_html
from app.pipeline.telegram_style import clean_news_html, clean_news_text


def depth_key(delivery_id: int) -> str:
    return f"news_depth:{delivery_id}"


def brief_card_html(card: dict[str, Any]) -> str:
    """Use the complete grounded lead, never a cut sentence or a new AI summary.

    Legacy cards without structured content retain their original rendering.
    The full validated card stays in storage for the free second level.
    """
    try:
        draft = json.loads(card.get("structured_output_json") or "{}")
        if isinstance(draft, dict) and draft.get("card_kind") in {"concept_brief", "people_brief"}:
            return clean_news_html(str(card["rendered_html"]))
        headline = clean_news_text(str(draft["headline"]))
        lead = clean_news_text(str(draft["lead"]))
        if not headline.strip() or not lead.strip():
            raise ValueError("empty card")
        linked = linked_headline_html(headline, str(card["primary_preview_url"]))
        return clean_news_html(f"<b>{linked}</b>\n\n{html.escape(lead)}")
    except (ValueError, TypeError, KeyError):
        return clean_news_html(str(card["rendered_html"]))
