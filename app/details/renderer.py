from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from app.details.models import DeepArticleDraft
from app.pipeline.cards import choose_link_phrase, linkify
from app.pipeline.telegram_style import clean_news_html

TELEGRAM_MESSAGE_LIMIT = 4096


class DetailRenderError(ValueError):
    pass


@dataclass(frozen=True)
class ResolvedDetailAnchor:
    phrase: str
    url: str


def _absolute_http_url(value: str) -> bool:
    parts = urlsplit(value)
    return parts.scheme in {"http", "https"} and bool(parts.netloc)


def document_map(documents: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(document["id"]): document for document in documents}


def resolve_detail_anchors(
    draft: DeepArticleDraft,
    documents: list[dict[str, Any]],
) -> list[ResolvedDetailAnchor]:
    by_id = document_map(documents)
    resolved: list[ResolvedDetailAnchor] = []
    seen_phrases: set[str] = set()
    for anchor in draft.source_anchors:
        document = by_id.get(anchor.document_id)
        url = str((document or {}).get("url") or "")
        if not document or not _absolute_http_url(url) or anchor.phrase in seen_phrases:
            continue
        resolved.append(ResolvedDetailAnchor(phrase=anchor.phrase, url=url))
        seen_phrases.add(anchor.phrase)
    return resolved


def _render_anchored_text(
    value: str,
    anchors: list[ResolvedDetailAnchor],
    used_phrases: set[str],
) -> str:
    matches: list[tuple[int, int, ResolvedDetailAnchor]] = []
    for anchor in anchors:
        if anchor.phrase in used_phrases:
            continue
        position = value.find(anchor.phrase)
        if position >= 0:
            matches.append((position, position + len(anchor.phrase), anchor))
    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    selected: list[tuple[int, int, ResolvedDetailAnchor]] = []
    for match in matches:
        if selected and match[0] < selected[-1][1]:
            raise DetailRenderError(
                "Source anchor phrases overlap: "
                + selected[-1][2].phrase + " / " + match[2].phrase
            )
        selected.append(match)
    if not selected:
        return html.escape(value)

    blocks: list[str] = []
    cursor = 0
    for start, end, anchor in selected:
        blocks.append(html.escape(value[cursor:start]))
        blocks.append(
            f'<a href="{html.escape(anchor.url, quote=True)}">{html.escape(value[start:end])}</a>'
        )
        used_phrases.add(anchor.phrase)
        cursor = end
    blocks.append(html.escape(value[cursor:]))
    return "".join(blocks)


def detail_article_visible_text(draft: DeepArticleDraft) -> str:
    blocks = [draft.headline, draft.lead]
    if draft.highlight is not None:
        blocks.append(draft.highlight.text)
    for section in draft.sections:
        blocks.append(section.heading)
        blocks.extend(paragraph.text for paragraph in section.paragraphs)
    if draft.limitations is not None:
        blocks.extend(["Что пока неизвестно", draft.limitations.text])
    return "\n\n".join(blocks)


def preferred_detail_source_url(
    draft: DeepArticleDraft,
    documents: list[dict[str, Any]],
) -> str:
    by_id = document_map(documents)
    for source_class in ("primary", "independent", "corroborating"):
        for document_id in draft.used_document_ids:
            document = by_id.get(document_id)
            url = str((document or {}).get("url") or "")
            if (
                document
                and document.get("source_class") == source_class
                and _absolute_http_url(url)
            ):
                return url
    for document_id in draft.used_document_ids:
        url = str((by_id.get(document_id) or {}).get("url") or "")
        if _absolute_http_url(url):
            return url
    raise DetailRenderError("Detailed article has no absolute source URL")


def render_detail_article_html(
    draft: DeepArticleDraft,
    documents: list[dict[str, Any]],
    *,
    headline_url: str | None = None,
    visible_limit: int = TELEGRAM_MESSAGE_LIMIT,
) -> str:
    """Render a grounded draft into one ordinary Telegram HTML message.

    Telegram applies its 4096-character limit to visible text after entity parsing, so
    long href values are intentionally excluded from the length calculation.
    """

    visible = detail_article_visible_text(draft)
    if len(visible) > visible_limit:
        raise DetailRenderError(
            f"Detailed article has {len(visible)} visible characters; limit is {visible_limit}"
        )

    target_url = headline_url or preferred_detail_source_url(draft, documents)
    if not _absolute_http_url(target_url):
        raise DetailRenderError("Headline URL must be absolute HTTP(S)")
    headline_phrase = choose_link_phrase(draft.headline)
    linked_headline, linked = linkify(draft.headline, headline_phrase, target_url)
    if not linked:
        linked_headline = (
            f'<a href="{html.escape(target_url, quote=True)}">{html.escape(draft.headline)}</a>'
        )

    anchors = resolve_detail_anchors(draft, documents)
    used_phrases: set[str] = set()
    blocks = [f"<b>{linked_headline}</b>", _render_anchored_text(draft.lead, anchors, used_phrases)]

    if draft.highlight is not None:
        blocks.append(
            "<blockquote>"
            + _render_anchored_text(draft.highlight.text, anchors, used_phrases)
            + "</blockquote>"
        )

    for section in draft.sections:
        blocks.append(f"<b>{html.escape(section.heading)}</b>")
        blocks.extend(
            _render_anchored_text(paragraph.text, anchors, used_phrases)
            for paragraph in section.paragraphs
        )

    if draft.limitations is not None:
        blocks.append("<b>Что пока неизвестно</b>")
        blocks.append(_render_anchored_text(draft.limitations.text, anchors, used_phrases))

    missing = [anchor.phrase for anchor in anchors if anchor.phrase not in used_phrases]
    if missing:
        raise DetailRenderError(
            "Source anchor phrase is absent from article: " + ", ".join(missing)
        )
    return clean_news_html("\n\n".join(blocks))


def source_urls_for_draft(
    draft: DeepArticleDraft,
    documents: list[dict[str, Any]],
) -> list[str]:
    by_id = document_map(documents)
    urls: list[str] = []
    for document_id in draft.used_document_ids:
        url = str((by_id.get(document_id) or {}).get("url") or "")
        if _absolute_http_url(url) and url not in urls:
            urls.append(url)
    return urls
