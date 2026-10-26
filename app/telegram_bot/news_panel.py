"""Durable news depth navigation inside the original delivered message."""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
from datetime import UTC, datetime
from html.parser import HTMLParser
from urllib.parse import urlsplit
from weakref import WeakValueDictionary
from zoneinfo import ZoneInfo

from aiogram.exceptions import TelegramAPIError, TelegramBadRequest
from aiogram.filters.callback_data import CallbackData
from aiogram.types import LinkPreviewOptions
from aiogram.utils.keyboard import InlineKeyboardBuilder

from app.details.paging import (
    paginate_telegram_html,
    telegram_html_visible_text,
    telegram_html_visible_units,
    utf16_units,
)
from app.llm.openrouter import build_evidence_snippets
from app.research.detail_policy import article_reference_fingerprint, detail_policy_versions
from app.telegram_bot.feedback import FeedbackAction
from app.telegram_bot.keyboards import (
    REACTION_LABELS,
    REACTION_ORDER,
    DetailFeedbackAction,
    DetailFeedbackCallback,
    FeedbackCallback,
)
from app.telegram_bot.news_depth import brief_card_html
from app.telegram_bot.progress import progress_text

_LOCKS: WeakValueDictionary[tuple[str, int], asyncio.Lock] = WeakValueDictionary()
_FORMAT_VERSION = "news_panel_v1"
_STATUS_RESERVE = 420


class NewsPanelControl(CallbackData, prefix="np"):
    delivery_id: int
    action: str
    page: int = 0
    revision: int = -1


def _lock(repository, delivery_id):
    key = (str(repository.database_path), int(delivery_id))
    return _LOCKS.setdefault(key, asyncio.Lock())


def _stamp():
    return datetime.now(UTC).isoformat()


def get_news_panel(repository, delivery_id) -> dict | None:
    with repository.connection() as db:
        row = db.execute("SELECT * FROM news_panels WHERE delivery_id=?", (delivery_id,)).fetchone()
    if not row:
        return None
    panel = dict(row)
    panel["reader_context"] = json.loads(panel["reader_context_json"])
    return panel


class _Links(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.urls = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            url = dict(attrs).get("href")
            if url:
                self.urls.append(url)


def _http_url(url):
    try:
        parts = urlsplit(url)
        return (
            parts.scheme in {"http", "https"}
            and bool(parts.netloc)
            and not any(ord(character) < 32 for character in url)
        )
    except (ValueError, TypeError):
        return False


def _baseline_sources(repository, card, draft, baseline_html):
    parser = _Links()
    parser.feed(baseline_html)
    urls = {url for url in parser.urls if _http_url(url)}
    ids = set()
    brief = bool(draft.get("headline") and draft.get("lead"))
    evidence_ids = set(draft.get("headline_evidence_ids", []) + draft.get("lead_evidence_ids", []))
    if not brief:
        for section in draft.get("sections", []):
            evidence_ids.update(section.get("evidence_ids", []))
        evidence_ids.update((draft.get("highlight") or {}).get("evidence_ids", []))
    # Select the snapshot that existed when this delivered card was written,
    # not today's enlarged fact sheet or every document attached to the event.
    with repository.connection() as db:
        fact_sheet = db.execute(
            "SELECT facts_json FROM event_fact_sheets WHERE event_id=? "
            "AND validation_status='valid' AND julianday(created_at)<=julianday(?) "
            "ORDER BY fact_sheet_version DESC LIMIT 1",
            (card["event_id"], card["created_at"]),
        ).fetchone()
        if fact_sheet and evidence_ids:
            facts = json.loads(fact_sheet[0])
            snippets = build_evidence_snippets(
                source_name="Frozen card facts",
                source_role="research_fact",
                published_at=card["created_at"],
                original_text="",
                evidence_items=[
                    {
                        "source_name": "Frozen card facts",
                        "source_role": "research_fact",
                        "original_text": fact["text"],
                        "document_ids": fact.get("document_ids", []),
                    }
                    for fact in facts
                ],
            )
            for snippet in snippets:
                if snippet["id"] in evidence_ids:
                    ids.update(snippet.get("document_ids", []))
        documents = db.execute(
            "SELECT id,url,canonical_url FROM research_documents WHERE id IN "
            "(SELECT research_document_id FROM event_research_documents WHERE event_id=?)",
            (card["event_id"],),
        ).fetchall()
    for document in documents:
        if document["id"] in ids:
            urls.update(
                url for url in (document["url"], document["canonical_url"]) if _http_url(url)
            )
        elif document["url"] in urls or document["canonical_url"] in urls:
            ids.add(document["id"])
    visible_urls = list(dict.fromkeys(url for url in parser.urls if _http_url(url)))
    return visible_urls + sorted(urls - set(visible_urls)), sorted(ids)


def ensure_news_panel(repository, delivery_id) -> dict:
    existing = get_news_panel(repository, delivery_id)
    if existing:
        return existing
    context = repository.get_delivery_context(delivery_id)
    if (
        not context
        or not isinstance(context.get("telegram_message_id"), int)
        or context["telegram_message_id"] <= 0
    ):
        raise ValueError("News has no confirmed original message")
    with repository.connection() as db:
        delivery = db.execute(
            "SELECT delivery_status FROM deliveries WHERE id=?", (delivery_id,)
        ).fetchone()
        card = dict(db.execute("SELECT * FROM cards WHERE id=?", (context["card_id"],)).fetchone())
    if delivery[0] != "sent":
        raise ValueError("Only delivered news can open a reader panel")
    baseline_html = brief_card_html(context)
    baseline_text = telegram_html_visible_text(baseline_html)
    try:
        draft = json.loads(card["structured_output_json"] or "{}")
    except (TypeError, ValueError):
        draft = {}
    if not isinstance(draft, dict):
        draft = {}
    if draft.get("card_kind") == "x_repost":
        # A repost is an adaptation of an author, never a validated news lead.
        baseline_html = card["rendered_html"]
        baseline_text = telegram_html_visible_text(baseline_html)
    urls, ids = _baseline_sources(repository, card, draft, baseline_html)
    reader = {
        "baseline_card_id": context["card_id"],
        "baseline_event_id": context["event_id"],
        "baseline_html": baseline_html,
        "baseline_text": baseline_text,
        "baseline_units": utf16_units(baseline_text),
        "baseline_source_urls": urls,
        "baseline_document_ids": ids,
    }
    if draft.get("card_kind") == "x_repost":
        reader["x_repost_id"] = draft["repost_id"]
    reader["context_fingerprint"] = hashlib.sha256(
        json.dumps(reader, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with repository.connection() as db:
        db.execute(
            "INSERT OR IGNORE INTO news_panels(delivery_id,chat_id,message_id,"
            "baseline_html,reader_context_json) "
            "VALUES(?,?,?,?,?)",
            (
                delivery_id,
                context["chat_id"],
                context["telegram_message_id"],
                baseline_html,
                json.dumps(reader, ensure_ascii=False),
            ),
        )
    return get_news_panel(repository, delivery_id)


def _request_matches(panel, request):
    reader = panel["reader_context"]
    return bool(
        request
        and request.get("depth") in {"expanded", "article"}
        and request.get("delivery_id") == panel["delivery_id"]
        and request.get("chat_id") == request.get("user_id") == panel["chat_id"]
        and request.get("card_id") == reader["baseline_card_id"]
        and request.get("event_id") == reader["baseline_event_id"]
        and request.get("reader_context", {}).get("context_fingerprint")
        == reader["context_fingerprint"]
    )


def _artifact_matches(panel, request, artifact):
    if not _request_matches(panel, request) or not artifact:
        return False
    return bool(
        request.get("status") in {"ready", "sent"}
        and request.get("article_id") == artifact.get("id")
        and artifact.get("event_id") == reader_event(panel)
        and artifact.get("depth") == request["depth"]
        and artifact.get("context_fingerprint") == panel["reader_context"]["context_fingerprint"]
        and artifact.get("reader_context", {}).get("context_fingerprint")
        == panel["reader_context"]["context_fingerprint"]
        and bool(artifact.get("schema_version"))
        and bool(artifact.get("prompt_version"))
        and artifact.get("validation_status") == "valid"
        and (
            request["depth"] != "article"
            or article_reference_fingerprint(request["reader_context"])
            == article_reference_fingerprint(artifact.get("reader_context") or {})
        )
    )


def reader_event(panel):
    return panel["reader_context"]["baseline_event_id"]


def _source_footer(artifact):
    urls = list(dict.fromkeys(url for url in artifact.get("source_urls", []) if _http_url(url)))
    if not urls:
        raise ValueError("Reader result has no valid source URLs")
    links = []
    for index, url in enumerate(urls, 1):
        label = (urlsplit(url).hostname or "").removeprefix("www.")
        if utf16_units(label) > 80:
            label = f"Источник {index}"
        links.append(f'<a href="{html.escape(url, quote=True)}">{html.escape(label)}</a>')
    notes = []
    schema, prompt = detail_policy_versions(artifact["depth"])
    try:
        expires = datetime.fromisoformat(artifact["expires_at"].replace("Z", "+00:00"))
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=UTC)
        expired = expires <= datetime.now(UTC)
    except (KeyError, ValueError, TypeError):
        expired = True
    if expired or (artifact["schema_version"], artifact["prompt_version"]) != (schema, prompt):
        try:
            created = datetime.fromisoformat(artifact["created_at"].replace("Z", "+00:00"))
            if created.tzinfo is None:
                created = created.replace(tzinfo=UTC)
            stamp = created.astimezone(ZoneInfo("Europe/Moscow")).strftime("%d.%m.%Y")
            notes.append("Сохранённый разбор от " + stamp)
        except (KeyError, ValueError, TypeError):
            notes.append("Сохранённый разбор")
    metadata = artifact.get("provider_response_metadata") or {}
    actual = metadata.get("actual_additional_sources")
    target = metadata.get("target_additional_sources")
    if type(actual) is int and type(target) is int and 0 <= actual < target and target in {3, 6}:
        notes.append(f"Дополнительных источников: {actual} из {target}")
    notes.append("Источники: " + " · ".join(links))
    return "\n".join(notes)


def artifact_pages(repository, artifact):
    footer = _source_footer(artifact)
    material = [artifact["rendered_html"], footer, _FORMAT_VERSION]
    digest = hashlib.sha256(json.dumps(material, ensure_ascii=False).encode()).hexdigest()
    key = f"news_reader_pages:{artifact['id']}:{digest}"
    cached = repository.get_runtime_setting(key)
    if cached:
        try:
            pages = json.loads(cached)
            if pages and all(
                isinstance(page, str) and telegram_html_visible_units(page) <= 4096
                for page in pages
            ):
                return pages
        except (ValueError, TypeError):
            pass
    limit = (
        4096
        - telegram_html_visible_units(footer)
        - utf16_units("Страница 999/999")
        - 4
        - _STATUS_RESERVE
    )
    pages = paginate_telegram_html(artifact["rendered_html"], page_limit=limit)
    result = [
        page + "\n\n" + footer + (f"\n\nСтраница {index}/{len(pages)}" if len(pages) > 1 else "")
        for index, page in enumerate(pages, 1)
    ]
    if any(telegram_html_visible_units(page) > 4096 - _STATUS_RESERVE for page in result):
        raise ValueError("Reader page leaves no room for inline progress")
    repository.set_runtime_setting(key, json.dumps(result, ensure_ascii=False))
    return result


def cached_back_target(repository, panel):
    depth = (
        panel["previous_view"]
        if panel["view"] in {"loading", "error"}
        else ("expanded" if panel["view"] == "article" else "base")
    )
    if depth in {"expanded", "article"}:
        request = repository.detail_request_for_delivery(panel["delivery_id"], depth=depth)
        artifact = (
            repository.get_detailed_article(request["article_id"])
            if request and request.get("article_id")
            else None
        )
        if _artifact_matches(panel, request, artifact):
            return depth, request, artifact
    return "base", None, None


def panel_keyboard(repository, panel, *, request=None, artifact=None, page_count=None):
    b = InlineKeyboardBuilder()

    def callback(action, page=0):
        return NewsPanelControl(
            delivery_id=panel["delivery_id"], action=action, page=page, revision=panel["revision"]
        )

    view = panel["view"]
    if view in {"expanded", "article"} and artifact is None and panel.get("displayed_request_id"):
        request = repository.get_detail_request(panel["displayed_request_id"])
        artifact = (
            repository.get_detailed_article(request["article_id"])
            if request and request.get("article_id")
            else None
        )
        if not _artifact_matches(panel, request, artifact):
            artifact = None
    if view in {"loading", "error"} and request is None and panel.get("active_request_id"):
        request = repository.get_detail_request(panel["active_request_id"])
    if page_count is None:
        page_count = len(artifact_pages(repository, artifact)) if artifact else 1
    if view == "base" and panel["reader_context"].get("x_repost_id"):
        from app.telegram_bot.x_reposts import feedback_buttons

        b.attach(feedback_buttons(repository, panel))
    elif view == "base":
        session = repository.feedback_session_for_user(panel["chat_id"])
        if session and session["delivery_id"] == panel["delivery_id"]:
            for action, label in [
                (FeedbackAction.COMMENT, "💬 Прокомментировать"),
                (FeedbackAction.SILENT, "✅ Отправить молча"),
                (FeedbackAction.BACK, "↩️ Назад"),
            ]:
                b.button(
                    text=label,
                    callback_data=FeedbackCallback(delivery_id=panel["delivery_id"], action=action),
                )
            b.adjust(1)
        else:
            current = repository.current_reaction(panel["delivery_id"])
            for reaction in REACTION_ORDER:
                label = REACTION_LABELS[reaction]
                b.button(
                    text=("✅ " if current == reaction.value else "") + label,
                    callback_data=FeedbackCallback(
                        delivery_id=panel["delivery_id"], action=FeedbackAction(reaction.value)
                    ),
                )
            b.adjust(4)
    elif view in {"expanded", "article"} and artifact:
        with repository.connection() as db:
            row = db.execute(
                "SELECT action FROM news_result_feedback WHERE article_id=? AND chat_id=?",
                (artifact["id"], panel["chat_id"]),
            ).fetchone()
        current = row[0] if row else None
        thumbs = InlineKeyboardBuilder()
        for action, label in [
            (DetailFeedbackAction.USEFUL, "👍 Полезно"),
            (DetailFeedbackAction.NOT_WORTH_IT, "👎 Не полезно"),
        ]:
            thumbs.button(
                text=("✅ " if current == action.value else "") + label,
                callback_data=DetailFeedbackCallback(
                    delivery_id=panel["delivery_id"], article_id=artifact["id"], action=action
                ),
            )
        thumbs.adjust(2)
        b.attach(thumbs)
    menu = InlineKeyboardBuilder()
    question_label = (
        "Задать вопрос" if panel["reader_context"].get("x_repost_id") else "Запросить разбор"
    )
    for depth, label in [("expanded", "Подробнее"), ("ask", question_label)]:
        if (
            view == "loading"
            and request
            and request["depth"] == ("article" if depth == "ask" else depth)
        ):
            label = "Готовлю разбор…" if depth == "ask" else "Готовлю подробности…"
        elif view == depth:
            label = "✓ " + label
        menu.button(text=label, callback_data=callback(depth))
    menu.adjust(2)
    b.attach(menu)
    saved = repository.detail_request_for_delivery(panel["delivery_id"], depth="article")
    if saved and saved["status"] in {"ready", "sent"} and view != "article":
        b.button(text="Открыть готовый разбор", callback_data=callback("article"))
    with repository.connection() as db:
        awaiting = db.execute(
            "SELECT 1 FROM news_question_sessions WHERE chat_id=? AND delivery_id=? "
            "AND datetime(expires_at)>datetime('now')",
            (panel["chat_id"], panel["delivery_id"]),
        ).fetchone()
    if awaiting:
        b.button(text="Отменить вопрос", callback_data=callback("cancel_question"))
    if view in {"expanded", "article"} and page_count > 1:
        paging = InlineKeyboardBuilder()
        if panel["page"] > 0:
            paging.button(text="← Предыдущая", callback_data=callback("prev", panel["page"] - 1))
        if panel["page"] + 1 < page_count:
            paging.button(text="Следующая →", callback_data=callback("next", panel["page"] + 1))
        paging.adjust(2)
        b.attach(paging)
    tail = InlineKeyboardBuilder()
    if view == "error" and request:
        tail.button(text="↻ Повторить", callback_data=callback("retry"))
    if view != "base":
        target, _, _ = cached_back_target(repository, panel)
        tail.button(
            text="← К подробностям" if target == "expanded" else "← Назад",
            callback_data=callback("back"),
        )
        if target != "base":
            tail.button(text="К новости", callback_data=callback("base"))
    urls = panel["reader_context"]["baseline_source_urls"]
    if urls:
        tail.button(text="Открыть первоисточник", url=urls[0])
    tail.adjust(1)
    b.attach(tail)
    return b.as_markup()


def _revision_matches(panel, expected_revision):
    return expected_revision is None or panel["revision"] == expected_revision


def _current_content(repository, panel):
    view = panel["previous_view"] if panel["view"] in {"loading", "error"} else panel["view"]
    page = panel["previous_page"] if panel["view"] in {"loading", "error"} else panel["page"]
    if view in {"expanded", "article"} and panel.get("displayed_request_id"):
        request = repository.get_detail_request(panel["displayed_request_id"])
        artifact = (
            repository.get_detailed_article(request["article_id"])
            if request and request.get("article_id")
            else None
        )
        if _artifact_matches(panel, request, artifact):
            pages = artifact_pages(repository, artifact)
            return pages[min(page, len(pages) - 1)]
    return panel["baseline_html"]


async def _edit_and_commit(
    bot, repository, current, updated, text, *, request=None, artifact=None, page_count=1
):
    repost_id = current["reader_context"].get("x_repost_id")
    if not repost_id and telegram_html_visible_units(text) > 4096:
        raise ValueError("News panel text exceeds Telegram")
    content = {
        "text": text,
        "parse_mode": "HTML",
        "link_preview_options": LinkPreviewOptions(is_disabled=True),
    }
    if repost_id:
        from app.reposts.media import rich_content
        from app.reposts.store import get

        repost = get(repository, repost_id)
        files = json.loads(repost["media_json"] or "[]") if repost else []
        visible_view = (
            updated["previous_view"] if updated["view"] in {"loading", "error"} else updated["view"]
        )
        content = {"rich_message": rich_content(text, files if visible_view == "base" else [])}
    try:
        await bot.edit_message_text(
            chat_id=current["chat_id"],
            message_id=current["message_id"],
            **content,
            reply_markup=panel_keyboard(
                repository, updated, request=request, artifact=artifact, page_count=page_count
            ),
            request_timeout=8,
        )
    except TelegramBadRequest as exc:
        if (
            "message is not modified" not in str(exc)
            and "message is not modified" not in exc.message.lower()
        ):
            return False
    except (TelegramAPIError, TimeoutError, ConnectionError):
        return False
    with repository.connection() as db:
        result = db.execute(
            "UPDATE news_panels SET view=?,displayed_request_id=?,active_request_id=?,"
            "previous_view=?,page=?,previous_page=?,revision=?,error_text=?,updated_at=? "
            "WHERE delivery_id=? AND revision=? AND chat_id=? AND message_id=?",
            (
                updated["view"],
                updated.get("displayed_request_id"),
                updated.get("active_request_id"),
                updated["previous_view"],
                updated["page"],
                updated["previous_page"],
                updated["revision"],
                updated.get("error_text"),
                _stamp(),
                current["delivery_id"],
                current["revision"],
                current["chat_id"],
                current["message_id"],
            ),
        )
        if result.rowcount == 1 and updated["view"] in {"expanded", "article"}:
            db.execute(
                "INSERT INTO runtime_settings(key,value,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value,"
                "updated_at=excluded.updated_at",
                (
                    f"news_reader_position:{current['delivery_id']}:{updated['view']}",
                    str(updated["page"]),
                    _stamp(),
                ),
            )
        return result.rowcount == 1


async def show_base(bot, repository, panel, *, expected_revision=None):
    async with _lock(repository, panel["delivery_id"]):
        current = get_news_panel(repository, panel["delivery_id"])
        if not current or not _revision_matches(current, expected_revision):
            return False
        updated = {
            **current,
            "view": "base",
            "page": 0,
            "previous_view": "base",
            "previous_page": 0,
            "active_request_id": None,
            "displayed_request_id": None,
            "revision": current["revision"] + 1,
            "error_text": None,
        }
        return await _edit_and_commit(bot, repository, current, updated, current["baseline_html"])


async def refresh_markup(bot, repository, delivery_id):
    """Refresh the current menu under the same lock as text and progress edits."""
    async with _lock(repository, delivery_id):
        current = get_news_panel(repository, delivery_id)
        if not current:
            return False
        try:
            await bot.edit_message_reply_markup(
                chat_id=current["chat_id"],
                message_id=current["message_id"],
                reply_markup=panel_keyboard(repository, current),
                request_timeout=8,
            )
        except TelegramBadRequest as exc:
            return "message is not modified" in str(exc).lower()
        except (TelegramAPIError, TimeoutError, ConnectionError):
            return False
        return True


async def show_result(
    bot, repository, panel, request, artifact, *, page=0, expected_revision=None, from_worker=False
):
    async with _lock(repository, panel["delivery_id"]):
        current = get_news_panel(repository, panel["delivery_id"])
        request = repository.get_detail_request(request.get("request_id") or request["id"])
        artifact = repository.get_detailed_article(artifact["id"])
        if (
            not current
            or not _revision_matches(current, expected_revision)
            or not _artifact_matches(current, request, artifact)
        ):
            return False
        if from_worker and (
            current["view"] != "loading" or current["active_request_id"] != request["request_id"]
        ):
            return False
        pages = artifact_pages(repository, artifact)
        if isinstance(page, bool) or not isinstance(page, int) or not 0 <= page < len(pages):
            return False
        updated = {
            **current,
            "view": request["depth"],
            "page": page,
            "previous_view": request["depth"],
            "previous_page": page,
            "active_request_id": None,
            "displayed_request_id": request["request_id"],
            "revision": current["revision"] + 1,
            "error_text": None,
        }
        return await _edit_and_commit(
            bot,
            repository,
            current,
            updated,
            pages[page],
            request=request,
            artifact=artifact,
            page_count=len(pages),
        )


async def set_loading(
    bot, repository, panel, request, *, progress_html=None, expected_revision=None
):
    async with _lock(repository, panel["delivery_id"]):
        current = get_news_panel(repository, panel["delivery_id"])
        request = repository.get_detail_request(request.get("request_id") or request["id"])
        if (
            not current
            or not _revision_matches(current, expected_revision)
            or not _request_matches(current, request)
        ):
            return False
        if request["status"] not in {"queued", "running", "ready"}:
            return False
        if current["view"] == "loading" and current["active_request_id"] != request["request_id"]:
            return False
        previous_view = (
            current["previous_view"] if current["view"] in {"loading", "error"} else current["view"]
        )
        previous_page = (
            current["previous_page"] if current["view"] in {"loading", "error"} else current["page"]
        )
        updated = {
            **current,
            "view": "loading",
            "previous_view": previous_view,
            "previous_page": previous_page,
            "active_request_id": request["request_id"],
            "revision": current["revision"] + int(current["view"] != "loading"),
            "error_text": None,
        }
        status = progress_html or progress_text("details", request.get("stage") or "queued")
        text = _current_content(repository, current) + "\n\n" + status
        return await _edit_and_commit(bot, repository, current, updated, text, request=request)


async def update_loading(bot, repository, delivery_id, request_id, progress_html):
    async with _lock(repository, delivery_id):
        current = get_news_panel(repository, delivery_id)
        if (
            not current
            or current["view"] != "loading"
            or current["active_request_id"] != request_id
        ):
            return False
        request = repository.get_detail_request(request_id)
        if not _request_matches(current, request) or request["status"] not in {"queued", "running"}:
            return False
        text = _current_content(repository, current) + "\n\n" + progress_html
        return await _edit_and_commit(bot, repository, current, current, text, request=request)


async def show_error(bot, repository, panel, request, error_text, *, from_worker=True):
    async with _lock(repository, panel["delivery_id"]):
        current = get_news_panel(repository, panel["delivery_id"])
        request = repository.get_detail_request(request.get("request_id") or request["id"])
        if not current or not _request_matches(current, request):
            return False
        if from_worker and (
            current["view"] != "loading" or current["active_request_id"] != request["request_id"]
        ):
            return False
        updated = {
            **current,
            "view": "error",
            "active_request_id": request["request_id"],
            "revision": current["revision"] + 1,
            "error_text": error_text,
        }
        text = _current_content(repository, current) + "\n\n" + html.escape(error_text)
        return await _edit_and_commit(bot, repository, current, updated, text, request=request)


class NewsPanelProgressBot:
    def __init__(self, bot, repository, delivery_id, request_id):
        self.bot, self.repository = bot, repository
        self.delivery_id, self.request_id = delivery_id, request_id

    def __getattr__(self, name):
        return getattr(self.bot, name)

    async def edit_message_text(self, **kwargs):
        panel = get_news_panel(self.repository, self.delivery_id)
        if (
            not panel
            or kwargs.get("chat_id") != panel["chat_id"]
            or kwargs.get("message_id") != panel["message_id"]
        ):
            return False
        return await update_loading(
            self.bot, self.repository, self.delivery_id, self.request_id, kwargs["text"]
        )
