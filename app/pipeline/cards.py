from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from app.config import Settings
from app.db.repository import Repository
from app.llm.openrouter import (
    PROMPT_VERSION,
    CardDraft,
    CardValidationError,
    OpenRouterCardGenerator,
    OpenRouterResponseError,
    resolve_verified_headline,
    validate_grounding,
)
from app.ops.budget import evaluate_budget
from app.pipeline.media import MEDIA_POLICY_VERSION, select_preview_first_media
from app.pipeline.ranking import rank_event_candidates
from app.pipeline.telegram_style import clean_news_html


@dataclass(frozen=True)
class GenerationStats:
    events_seen: int = 0
    cards_created: int = 0
    errors: int = 0
    cost_usd: float = 0


@dataclass(frozen=True)
class ResolvedAnchor:
    phrase: str
    url: str


def generation_models(settings: Settings) -> list[str]:
    candidates = [
        settings.openrouter_card_model or settings.openrouter_fallback_model,
        settings.openrouter_model or "google/gemini-2.5-flash-lite",
    ]
    models: list[str] = []
    for model in candidates:
        if model and model not in models:
            models.append(model)
    return models


PRIMARY_DOMAIN_SUFFIXES = {
    "anthropic.com",
    "deepseek.com",
    "google.com",
    "microsoft.com",
    "nvidia.com",
    "openai.com",
    "qwen.ai",
    "x.ai",
}
REPOSITORY_HOSTS = {"github.com", "huggingface.co"}
LOW_PRIORITY_HOSTS = {
    "dslab.tech",
    "t.me",
    "telegram.me",
    "techcrunch.com",
}


def url_priority(url: str, context: str = "") -> tuple[int, int]:
    parts = urlsplit(url)
    host = (parts.hostname or "").lower().removeprefix("www.")
    if any(host == suffix or host.endswith(f".{suffix}") for suffix in PRIMARY_DOMAIN_SUFFIXES):
        return (0, 0)
    if host in REPOSITORY_HOSTS:
        context_tokens = {
            token for token in re.findall(r"[a-z0-9]+", context.lower()) if len(token) >= 3
        }
        path_tokens = set(re.findall(r"[a-z0-9]+", parts.path.lower()))
        overlap = len(context_tokens & path_tokens)
        return (1 if overlap else 2, -overlap)
    if host in LOW_PRIORITY_HOSTS:
        return (3, 0)
    return (2, 0)


def unique_known_urls(event: dict, extra_urls: list[str] | None = None) -> list[str]:
    urls = json.loads(event["outbound_urls_json"] or "[]")
    urls.append(event["source_url"])
    urls.extend(extra_urls or [])
    unique: list[str] = []
    for url in urls:
        if url not in unique:
            unique.append(url)
    original_order = {url: index for index, url in enumerate(unique)}
    context = f"{event.get('headline', '')} {event.get('original_text', '')[:1000]}"
    ranked = sorted(
        unique,
        key=lambda url: (*url_priority(url, context), original_order[url]),
    )
    if event.get("source_role") == "primary" and event["source_url"] in ranked:
        ranked.remove(event["source_url"])
        ranked.insert(0, event["source_url"])
    return ranked[:12]


def linkify(value: str, phrase: str, url: str) -> tuple[str, bool]:
    position = value.find(phrase)
    if position < 0:
        return html.escape(value), False
    before = html.escape(value[:position])
    linked = f'<a href="{html.escape(url, quote=True)}">{html.escape(phrase)}</a>'
    after = html.escape(value[position + len(phrase) :])
    return before + linked + after, True


def resolve_contextual_anchors(
    draft: CardDraft,
    *,
    evidence_items: list[dict],
    research_documents: list[dict],
    primary_url: str,
) -> list[ResolvedAnchor]:
    documents_by_id = {
        int(document["id"]): document
        for document in research_documents
        if document.get("source_class") != "radar_only"
    }
    resolved: list[ResolvedAnchor] = []
    seen_urls = {primary_url}
    for anchor in draft.source_anchors:
        document = documents_by_id.get(anchor.document_id)
        if document is None:
            continue
        url = str(document.get("url") or "")
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.netloc or url in seen_urls:
            continue
        # Provenance was already validated against evidence IDs. Keep URLs renderer-only.
        if not any(
            anchor.document_id in {int(value) for value in item.get("document_ids", [])}
            for item in evidence_items
        ):
            continue
        resolved.append(ResolvedAnchor(phrase=anchor.phrase, url=url))
        seen_urls.add(url)
        if len(resolved) == 2:
            break
    return resolved


def render_anchored_text(
    value: str,
    anchors: list[ResolvedAnchor],
    used_phrases: set[str],
) -> str:
    matches: list[tuple[int, int, ResolvedAnchor]] = []
    for anchor in anchors:
        if anchor.phrase in used_phrases:
            continue
        position = value.find(anchor.phrase)
        if position >= 0:
            matches.append((position, position + len(anchor.phrase), anchor))
    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    selected: list[tuple[int, int, ResolvedAnchor]] = []
    for match in matches:
        if selected and match[0] < selected[-1][1]:
            continue
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


def has_pending_anchor(
    value: str,
    anchors: list[ResolvedAnchor],
    used_phrases: set[str],
) -> bool:
    return any(anchor.phrase not in used_phrases and anchor.phrase in value for anchor in anchors)


def choose_link_phrase(headline: str) -> str:
    latin_entities = re.finditer(
        r"\b[A-Z][A-Za-z0-9._-]*(?:\s+(?:[A-Z][A-Za-z0-9._-]*|[A-Za-z]*\d[A-Za-z0-9._-]*)){0,3}",
        headline,
    )
    for latin_entity in latin_entities:
        phrase = latin_entity.group(0).strip("-")
        if phrase.upper() not in {"AI", "API", "APT", "CPU", "GPU", "LLM"}:
            return phrase
    quoted = re.search(r"«([^»]{3,80})»", headline)
    if quoted:
        return quoted.group(1)
    metric_phrases = list(
        re.finditer(
            r"\b\d[\d\s.,]*\s+(?:нов\w+\s+)?[А-Яа-яЁё-]+(?:\s+[А-Яа-яЁё-]+){0,2}",
            headline,
        )
    )
    if metric_phrases:
        metric_phrase = max(
            metric_phrases,
            key=lambda match: (
                bool(re.search(r"\bнов\w*\b", match.group(0), re.IGNORECASE)),
                len(match.group(0).split()),
                -match.start(),
            ),
        )
        return metric_phrase.group(0)
    words = re.findall(r"[А-Яа-яЁё0-9-]+", headline)
    return " ".join(words[: min(3, len(words))]) or headline


SECTION_LABELS = {
    "benchmarks": "Бенчмарки.",
    "economics": "Цена.",
    "availability": "Статус.",
    "context": "Контекст.",
    "statement": "Заявление.",
}
ARTICLE_SECTION_HEADINGS = {
    "details": "Что изменилось",
    "benchmarks": "Бенчмарки",
    "economics": "Цена",
    "availability": "Статус",
    "limitations": "Ограничения",
    "context": "Контекст",
    "statement": "Заявление",
}
CONTENT_STOPWORDS = {
    "более",
    "будет",
    "были",
    "версии",
    "доступен",
    "доступна",
    "который",
    "которая",
    "модель",
    "можно",
    "новый",
    "новая",
    "релиз",
    "системы",
    "также",
    "через",
}


def comparable_tokens(text: str) -> set[str]:
    tokens = set()
    for token in re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", text.casefold()):
        if token in CONTENT_STOPWORDS:
            continue
        if token.isdigit() or len(token) >= 4:
            tokens.add(token[:5])
    return tokens


def is_redundant_text(text: str, previous_texts: list[str]) -> bool:
    tokens = comparable_tokens(text)
    if len(tokens) < 5:
        return False
    return any(
        len(tokens & comparable_tokens(previous)) / len(tokens) >= 0.55
        for previous in previous_texts
    )


def render_section_html(
    kind: str,
    text: str,
    *,
    anchors: list[ResolvedAnchor] | None = None,
    used_phrases: set[str] | None = None,
) -> str:
    escaped = render_anchored_text(
        text,
        anchors or [],
        used_phrases if used_phrases is not None else set(),
    )
    if kind == "details":
        return escaped
    if kind == "limitations":
        return escaped
    label = SECTION_LABELS.get(kind)
    if label is None:
        raise CardValidationError(f"Unknown card section kind: {kind}")
    return f"<b>{label}</b> {escaped}"


def linked_headline_html(headline: str, primary_url: str) -> str:
    parts = urlsplit(primary_url)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise CardValidationError("Primary card URL must be an absolute HTTP(S) URL")
    link_phrase = choose_link_phrase(headline)
    linked_headline, linked = linkify(headline, link_phrase, primary_url)
    if not linked:
        linked_headline = f'<a href="{html.escape(primary_url, quote=True)}">{linked_headline}</a>'
    return linked_headline


def render_card_html(
    draft: CardDraft,
    primary_url: str,
    contextual_anchors: list[ResolvedAnchor] | None = None,
) -> str:
    headline = linked_headline_html(draft.headline, primary_url)
    anchors = contextual_anchors or []
    used_phrases: set[str] = set()
    blocks = [
        f"<b>{headline}</b>",
        render_anchored_text(draft.lead, anchors, used_phrases),
    ]
    visible_texts = [draft.lead]
    section_texts = [section.text for section in draft.sections]
    # Prefer the complete section over a preview of the same fact. A pending
    # link keeps the highlight only when no later section can carry that link;
    # sections with pending anchors are already protected from dedup below.
    highlight_has_unique_anchor = draft.highlight is not None and any(
        anchor.phrase not in used_phrases
        and anchor.phrase in draft.highlight.text
        and not any(anchor.phrase in text for text in section_texts)
        for anchor in anchors
    )
    if draft.highlight is not None and (
        not is_redundant_text(draft.highlight.text, visible_texts + section_texts)
        or highlight_has_unique_anchor
    ):
        blocks.append(
            "<blockquote>"
            + render_anchored_text(draft.highlight.text, anchors, used_phrases)
            + "</blockquote>"
        )
        visible_texts.append(draft.highlight.text)
    labelled_kinds: set[str] = set()
    for section in draft.sections:
        if is_redundant_text(section.text, visible_texts) and not has_pending_anchor(
            section.text,
            anchors,
            used_phrases,
        ):
            continue
        if section.kind in labelled_kinds and section.kind not in {"details", "limitations"}:
            blocks.append(render_anchored_text(section.text, anchors, used_phrases))
        else:
            blocks.append(
                render_section_html(
                    section.kind,
                    section.text,
                    anchors=anchors,
                    used_phrases=used_phrases,
                )
            )
            labelled_kinds.add(section.kind)
        visible_texts.append(section.text)
    rendered = clean_news_html("\n\n".join(blocks))
    if len(rendered) > 4096:
        raise CardValidationError("Rendered Telegram card exceeds 4096 characters")
    return rendered


def render_article_rich_html(draft: CardDraft, primary_url: str) -> str:
    """Render the same grounded draft as a native Telegram Rich Message article."""
    headline = linked_headline_html(draft.headline, primary_url)
    blocks = [f"<h1>{headline}</h1>", f"<p>{html.escape(draft.lead)}</p>"]
    visible_texts = [draft.lead]
    if draft.highlight is not None and not is_redundant_text(draft.highlight.text, visible_texts):
        blocks.append(f"<aside>{html.escape(draft.highlight.text)}</aside>")
        visible_texts.append(draft.highlight.text)

    headed_kinds: set[str] = set()
    for section in draft.sections:
        if is_redundant_text(section.text, visible_texts):
            continue
        heading = ARTICLE_SECTION_HEADINGS.get(section.kind)
        if heading is None:
            raise CardValidationError(f"Unknown card section kind: {section.kind}")
        if section.kind not in headed_kinds:
            blocks.append(f"<h2>{heading}</h2>")
            headed_kinds.add(section.kind)
        blocks.append(f"<p>{html.escape(section.text)}</p>")
        visible_texts.append(section.text)

    rendered = "\n".join(blocks)
    if len(rendered) > 32768:
        raise CardValidationError("Rendered Telegram rich article exceeds 32768 characters")
    return rendered


def generate_cards(
    settings: Settings,
    repository: Repository,
    *,
    limit: int = 3,
    event_id: int | None = None,
    generator: OpenRouterCardGenerator | None = None,
) -> GenerationStats:
    from app.personalization.daily import load_context

    context = load_context(settings, repository)
    event_pool = repository.events_without_cards(
        limit if event_id is not None else max(limit * 5, 50),
        event_id=event_id,
        require_fact_sheet=True,
        require_ready_dossier=True,
    )
    events = (
        event_pool
        if event_id is not None
        else rank_event_candidates(
            event_pool,
            recent_events=repository.recent_valid_card_events(days=7),
            limit=limit,
            personalization=context,
        )
    )
    generator = generator or OpenRouterCardGenerator(settings)
    created = errors = 0
    total_cost = 0.0

    for event in events:
        budget = evaluate_budget(
            repository.current_month_cost(),
            settings.budget_warning_usd,
            settings.budget_hard_stop_usd,
        )
        if not budget.paid_operations_allowed:
            break
        research_documents = repository.event_research_documents(event["id"])
        primary_document_urls = [
            document["url"]
            for document in research_documents
            if document.get("source_class") == "primary"
        ]
        known_urls = unique_known_urls(
            event,
            [
                *repository.event_evidence_urls(event["id"]),
                *(document["url"] for document in research_documents),
            ],
        )
        evidence_items = repository.fact_sheet_evidence_items(event["id"])
        fact_sheet = repository.latest_valid_fact_sheet(event["id"])
        editorial_hook = (
            dict(fact_sheet["coverage"].get("editorial_hook") or {}) if fact_sheet else {}
        )
        attempt_errors: list[str] = []
        event_cost = 0.0
        card_created = False
        models = generation_models(settings)[:2]
        single_model = len(models) == 1
        if single_model:
            models.append(models[0])
        repair_context = None
        repair_error_refs: list[dict] = []
        for attempt, model in enumerate(models, start=1):
            # With one configured model, retry only a parsed draft rejected by
            # the factual validator. Never replay a network/unknown failure.
            if attempt == 2 and single_model and repair_context is None:
                break
            budget = evaluate_budget(
                repository.current_month_cost(),
                settings.budget_warning_usd,
                settings.budget_hard_stop_usd,
            )
            if not budget.paid_operations_allowed:
                attempt_errors.append("monthly hard budget stop before fallback")
                break
            try:
                result = generator.generate(
                    source_name=str(event.get("source_name") or "validated fact sheet"),
                    source_role="research_fact",
                    published_at=event["published_at"],
                    original_text=event["original_text"],
                    known_urls=known_urls,
                    evidence_items=evidence_items,
                    event_type=event["event_type"],
                    editorial_hook=editorial_hook,
                    model=model,
                    **({"repair_context": repair_context} if repair_context else {}),
                )
                event_cost += result.cost_usd
                total_cost += result.cost_usd
                try:
                    resolved_draft, fallback_reasons = resolve_verified_headline(
                        result.draft,
                        source_name=str(event.get("source_name") or "validated fact sheet"),
                        source_role="research_fact",
                        published_at=event["published_at"],
                        original_text=event["original_text"],
                        evidence_items=evidence_items,
                        editorial_hook=editorial_hook,
                    )
                    validate_grounding(
                        resolved_draft,
                        original_text=event["original_text"],
                        known_urls=known_urls,
                        source_name=str(event.get("source_name") or "validated fact sheet"),
                        source_role="research_fact",
                        published_at=event["published_at"],
                        evidence_items=evidence_items,
                        event_type=event["event_type"],
                        editorial_hook=editorial_hook,
                    )
                except CardValidationError as exc:
                    repository.record_usage(
                        "openrouter",
                        "card.generate.validation_failed",
                        cost_usd=result.cost_usd,
                        input_units=result.prompt_tokens,
                        output_units=result.completion_tokens,
                        metadata={
                            "event_id": event["id"],
                            "attempt": attempt,
                            "model": result.model,
                            "response_id": result.response_id,
                            "is_repair": repair_context is not None,
                            "repair_error_refs": repair_error_refs,
                            "error": str(exc),
                            "rejected_draft": result.draft.model_dump(),
                            **getattr(result, "billing_metadata", {}),
                        },
                    )
                    attempt_errors.append(f"{model}: {exc}")
                    repair_context = {
                        "validation_error": str(exc),
                        "rejected_draft": result.draft.model_dump(),
                    }
                    repair_error_refs = [
                        {
                            "event_id": event["id"],
                            "attempt": attempt,
                            "response_id": result.response_id,
                        }
                    ]
                    continue

                if fallback_reasons:
                    repository.record_usage(
                        "local",
                        "card.headline_fallback_used",
                        metadata={
                            "event_id": event["id"],
                            "attempt": attempt,
                            "rejected_headline": result.draft.headline,
                            "accepted_headline": resolved_draft.headline,
                            "reasons": fallback_reasons,
                        },
                    )

                repository.record_usage(
                    "openrouter",
                    "card.generate",
                    cost_usd=result.cost_usd,
                    input_units=result.prompt_tokens,
                    output_units=result.completion_tokens,
                    metadata={
                        "event_id": event["id"],
                        "attempt": attempt,
                        "model": result.model,
                        "response_id": result.response_id,
                        "is_repair": repair_context is not None,
                        "repair_error_refs": repair_error_refs,
                        **getattr(result, "billing_metadata", {}),
                    },
                )
                media = select_preview_first_media(
                    repository.event_research_documents(event["id"]),
                    fallback_urls=primary_document_urls or known_urls,
                    preferred_document_ids=[
                        int(value) for value in editorial_hook.get("document_ids", [])
                    ],
                )
                primary_url = media.preview_url or (
                    primary_document_urls[0] if primary_document_urls else known_urls[0]
                )
                contextual_anchors = resolve_contextual_anchors(
                    resolved_draft,
                    evidence_items=evidence_items,
                    research_documents=research_documents,
                    primary_url=primary_url,
                )
                rendered_html = render_card_html(
                    resolved_draft,
                    primary_url,
                    contextual_anchors,
                )
                card_id = repository.create_card(
                    event_id=event["id"],
                    prompt_version=PROMPT_VERSION,
                    model=result.model,
                    generation_cost_usd=event_cost,
                    structured_output=resolved_draft.model_dump(),
                    rendered_html=rendered_html,
                    source_urls=known_urls,
                    primary_preview_url=primary_url,
                )
                repository.create_card_media(
                    card_id=card_id,
                    policy_version=MEDIA_POLICY_VERSION,
                    mode=media.mode,
                    preview_url=media.preview_url,
                    image_url=media.image_url,
                    source_document_id=media.source_document_id,
                    checks=media.checks,
                )
                created += 1
                card_created = True
                break
            except OpenRouterResponseError as exc:
                repository.record_usage(
                    "openrouter",
                    "card.generate.invalid_response",
                    cost_usd=exc.cost_usd,
                    input_units=exc.prompt_tokens,
                    output_units=exc.completion_tokens,
                    metadata={
                        "event_id": event["id"],
                        "attempt": attempt,
                        "model": exc.model,
                        "response_id": exc.response_id,
                        "is_repair": repair_context is not None,
                        "repair_error_refs": repair_error_refs,
                        **getattr(exc, "billing_metadata", {}),
                    },
                )
                event_cost += exc.cost_usd
                total_cost += exc.cost_usd
                attempt_errors.append(f"{model}: {exc}")
            except Exception as exc:
                attempt_errors.append(f"{model}: {exc}")
                break
        if not card_created:
            reason = " | ".join(attempt_errors) or "No generation model available"
            repository.set_event_status(event["id"], "generation_error", reason)
            errors += 1

    return GenerationStats(
        events_seen=len(events),
        cards_created=created,
        errors=errors,
        cost_usd=total_cost,
    )
