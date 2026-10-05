from __future__ import annotations

import html
import json
import re
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import Any

from app.config import InterestProfile, WeeklyPersonaRegistry
from app.contours.models import ContourRegistry
from app.intelligence.models import IntelligenceProductDraft, ProductItem, ProductSection
from app.intelligence.trends import detect_trends

TAG_RE = re.compile(r"<[^>]+>")
PROJECT_STOPWORDS = {
    "для",
    "the",
    "and",
    "или",
    "как",
    "это",
    "система",
    "применение",
    "персональная",
    "сложных",
    "продукты",
}


def _event_time(event: dict[str, Any]) -> datetime | None:
    value = event.get("event_at") or event.get("latest_source_at") or event.get("created_at")
    if not value:
        return None
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if result.tzinfo is None:
        result = result.replace(tzinfo=UTC)
    return result.astimezone(UTC)


def _event_url(event: dict[str, Any]) -> str | None:
    for key in ("primary_url", "source_url"):
        value = str(event.get(key) or "").strip()
        if value.startswith(("http://", "https://")):
            return value
    return None


def _event_item(event: dict[str, Any], *, note: str | None = None) -> ProductItem:
    return ProductItem(
        headline=str(event.get("headline") or "Событие без заголовка")[:240],
        url=_event_url(event),
        event_id=int(event["id"]) if event.get("id") else None,
        note=note,
    )


def _render_item(item: ProductItem) -> str:
    headline = html.escape(item.headline)
    if item.url:
        headline = f'<a href="{html.escape(item.url, quote=True)}">{headline}</a>'
    note = f" — {html.escape(item.note)}" if item.note else ""
    return f"• {headline}{note}"


def _render_product(
    *,
    title: str,
    summary: str,
    sections: list[ProductSection],
    max_visible_chars: int = 3900,
) -> str:
    chunks = [f"<b>{html.escape(title)}</b>", "", html.escape(summary)]
    for section in sections:
        candidate = ["", f"<b>{html.escape(section.title)}</b>", html.escape(section.summary)]
        candidate.extend(_render_item(item) for item in section.items)
        visible = len(TAG_RE.sub("", "\n".join([*chunks, *candidate])))
        if visible > max_visible_chars:
            break
        chunks.extend(candidate)
    return "\n".join(chunks)


def _period_events(
    events: list[dict[str, Any]],
    *,
    start: datetime,
    end: datetime,
) -> list[dict[str, Any]]:
    return [
        event
        for event in events
        if (event_time := _event_time(event)) is not None and start <= event_time <= end
    ]


def build_weekly_map(
    events: list[dict[str, Any]],
    registry: ContourRegistry,
    *,
    now: datetime | None = None,
) -> IntelligenceProductDraft:
    end = (now or datetime.now(UTC)).astimezone(UTC)
    start = end - timedelta(days=7)
    current_events = _period_events(events, start=start, end=end)
    events_by_id = {int(event["id"]): event for event in events if event.get("id")}
    contour_titles = {contour.id: contour.title for contour in registry.contours}
    trends = detect_trends(events, now=end)
    sections: list[ProductSection] = []

    rendered_trends = trends[:9]
    for trend in rendered_trends:
        items = [
            _event_item(events_by_id[event_id])
            for event_id in trend.current_event_ids[:3]
            if event_id in events_by_id
        ]
        sections.append(
            ProductSection(
                id=trend.key,
                title=f"{contour_titles.get(trend.primary_contour, trend.primary_contour)}: "
                f"{trend.label}",
                summary=(
                    f"{trend.current_count} события за неделю против "
                    f"{trend.previous_count} неделей ранее; "
                    f"различных источников в корпусе: {trend.distinct_sources}."
                ),
                items=items,
            )
        )

    summary = (
        f"За семь дней система проверила {len(current_events)} событий и выделила "
        f"{len(trends)} накапливающихся информационных движения."
    )
    title = f"Недельная карта OSVED · {start:%d.%m}–{end:%d.%m.%Y}"
    rendered = _render_product(title=title, summary=summary, sections=sections)
    return IntelligenceProductDraft(
        kind="weekly_map",
        period_start=start,
        period_end=end,
        title=title,
        summary=summary,
        sections=sections,
        # Provenance must describe only claims that made it into the rendered
        # product. The comparison mentions both windows, so both sides belong
        # to the audit trail; trends omitted by the nine-section Telegram cap do not.
        source_event_ids=sorted(
            {
                event_id
                for trend in rendered_trends
                for event_id in [*trend.current_event_ids, *trend.previous_event_ids]
            }
        ),
        rendered_html=rendered,
        metadata={"trend_count": len(trends), "event_count": len(current_events)},
    )


def build_monthly_theses(
    events: list[dict[str, Any]],
    registry: ContourRegistry,
    *,
    now: datetime | None = None,
) -> IntelligenceProductDraft:
    end = (now or datetime.now(UTC)).astimezone(UTC)
    start = end - timedelta(days=28)
    contour_titles = {contour.id: contour.title for contour in registry.contours}
    trends = detect_trends(events, now=end, window_days=14)
    sections: list[ProductSection] = []
    rendered_trends = trends[:8]
    for trend in rendered_trends:
        if trend.current_count > trend.previous_count:
            direction = "усиливается"
        elif trend.current_count < trend.previous_count:
            direction = "ослабевает"
        else:
            direction = "сохраняется"
        sections.append(
            ProductSection(
                id=trend.key,
                title=contour_titles.get(trend.primary_contour, trend.primary_contour),
                summary=(
                    f"Рабочий тезис: движение «{trend.label}» {direction}. "
                    f"Последние 14 дней: {trend.current_count} события; "
                    f"предыдущие 14 дней: {trend.previous_count}."
                ),
            )
        )
    summary = (
        "Это не предсказания и не автоматически принятые убеждения. Это проверяемые "
        "кандидаты на изменение картины мира, выведенные из накопленных событий."
    )
    title = f"Рабочие тезисы OSVED · {start:%d.%m}–{end:%d.%m.%Y}"
    return IntelligenceProductDraft(
        kind="monthly_theses",
        period_start=start,
        period_end=end,
        title=title,
        summary=summary,
        sections=sections,
        source_event_ids=sorted(
            {
                event_id
                for trend in rendered_trends
                for event_id in [*trend.current_event_ids, *trend.previous_event_ids]
            }
        ),
        rendered_html=_render_product(title=title, summary=summary, sections=sections),
        metadata={"thesis_candidates": len(sections), "requires_user_review": True},
    )


def _event_searchable_text(event: dict[str, Any]) -> str:
    parts = [str(event.get("headline") or ""), str(event.get("original_text") or "")]
    for key in ("entities_json", "topics_json"):
        value = event.get(key)
        if isinstance(value, str):
            with_value: Any = value
            with suppress(json.JSONDecodeError):
                with_value = json.loads(value)
            value = with_value
        parts.append(json.dumps(value, ensure_ascii=False) if value else "")
    return " ".join(parts).casefold()


def build_weekly_persona_map(
    events: list[dict[str, Any]],
    personas: WeeklyPersonaRegistry,
    *,
    now: datetime | None = None,
) -> IntelligenceProductDraft:
    end = (now or datetime.now(UTC)).astimezone(UTC)
    start = end - timedelta(days=7)
    recent = _period_events(events, start=start, end=end)
    sections: list[ProductSection] = []
    used_event_ids: set[int] = set()
    for persona in personas.personas:
        needles = {
            persona.name.casefold(),
            *(handle.casefold() for handle in persona.source_strategy.x_handles),
        }
        matching = [
            event
            for event in recent
            if any(needle and needle in _event_searchable_text(event) for needle in needles)
            and float(event.get("score") or 0) >= 3.0
        ]
        if not matching and persona.skip_if_no_material_update:
            continue
        matching.sort(key=lambda event: float(event.get("score") or 0), reverse=True)
        items = [_event_item(event) for event in matching[:4]]
        used_event_ids.update(item.event_id for item in items if item.event_id)
        sections.append(
            ProductSection(
                id=persona.id,
                title=persona.name,
                summary=f"{persona.role}. Существенных обновлений: {len(matching)}.",
                items=items,
            )
        )
    title = f"Люди и институты · {start:%d.%m}–{end:%d.%m.%Y}"
    summary = (
        "Пилотный слой проверяет только уже подтверждённые события из общего корпуса OSVED. "
        "Отдельный поиск по интервью, выступлениям и аккаунтам персон пока не запускается; "
        "поэтому пустая сводка не означает, что человек ничего не делал."
    )
    return IntelligenceProductDraft(
        kind="weekly_personas",
        period_start=start,
        period_end=end,
        title=title,
        summary=summary,
        sections=sections,
        source_event_ids=sorted(used_event_ids),
        rendered_html=_render_product(title=title, summary=summary, sections=sections),
        metadata={
            "personas_with_material_updates": len(sections),
            "coverage_mode": "shared_verified_corpus_only",
            "dedicated_persona_collection_enabled": False,
            "scaffold_only": True,
        },
    )


def _project_terms(project: str) -> set[str]:
    return {
        term.casefold()
        for term in re.findall(r"[A-Za-zА-Яа-яЁё-]{3,}", project)
        if term.casefold() not in PROJECT_STOPWORDS
    }


def build_project_signals(
    events: list[dict[str, Any]],
    profile: InterestProfile,
    *,
    now: datetime | None = None,
) -> IntelligenceProductDraft:
    end = (now or datetime.now(UTC)).astimezone(UTC)
    start = end - timedelta(days=7)
    recent = _period_events(events, start=start, end=end)
    sections: list[ProductSection] = []
    used_event_ids: set[int] = set()
    for index, project in enumerate(profile.current_projects):
        terms = _project_terms(project)
        matches: list[tuple[int, dict[str, Any]]] = []
        for event in recent:
            text = _event_searchable_text(event)
            score = sum(term in text or (len(term) >= 7 and term[:6] in text) for term in terms)
            if score:
                matches.append((score, event))
        matches.sort(
            key=lambda pair: (pair[0], float(pair[1].get("score") or 0)),
            reverse=True,
        )
        items = [
            _event_item(event, note=f"совпавших проектных признаков: {score}")
            for score, event in matches[:4]
        ]
        used_event_ids.update(item.event_id for item in items if item.event_id)
        if items:
            sections.append(
                ProductSection(
                    id=f"project_{index + 1}",
                    title=project,
                    summary="Сигналы требуют человеческой проверки применимости к проекту.",
                    items=items,
                )
            )
    title = f"Проектные сигналы OSVED · {start:%d.%m}–{end:%d.%m.%Y}"
    summary = "События, которые пересекаются с активными проектами пользователя."
    return IntelligenceProductDraft(
        kind="project_signals",
        period_start=start,
        period_end=end,
        title=title,
        summary=summary,
        sections=sections,
        source_event_ids=sorted(used_event_ids),
        rendered_html=_render_product(title=title, summary=summary, sections=sections),
        metadata={"projects_with_signals": len(sections)},
    )
