"""Loopback-only private web adapter for a LOCAL SNAPSHOT of OSVED.

This is not a replacement Telegram process. Explicit on-demand web jobs use their
own queue; existing Telegram jobs are never claimed or delivered by this module.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hmac
import ipaddress
import json
import secrets
import time
from pathlib import Path
from urllib.parse import urlsplit

from aiohttp import web

from app.config import Settings
from app.db.repository import Repository, utc_now
from app.details.research import (
    DetailBudgetError,
    DetailEvidenceError,
    DetailGenerationError,
    build_detail_article,
)
from app.video.channels import PRICE as CHANNEL_RESOLUTION_PRICE
from app.video.channels import ChannelError, ChannelService, channel_url
from app.video.reports import ReportError
from app.video.tldr import telegram_tldr
from app.web.budget import LocalBudget, LocalBudgetBlocked
from app.web.content import safe_html
from app.web.store import FOLDERS, REACTIONS, DashboardStore

STATE = web.AppKey("dashboard_state", object)
COOKIE = "pars_dashboard_session"
CONTENT_SECURITY_POLICY = "; ".join(
    (
        "default-src 'self'",
        "script-src 'self' 'wasm-unsafe-eval'",
        "style-src 'self' 'unsafe-inline'",
        "img-src 'self' data: blob:",
        "font-src 'self' data:",
        "worker-src 'self' blob:",
        "connect-src 'self' blob:",
        "object-src 'none'",
        "base-uri 'none'",
        "form-action 'self'",
        "frame-ancestors 'self'",
    )
)
REGIONS = {
    "moscow": ("москв", "moscow"),
    "beijing": ("пекин", "beijing"),
    "gba": (
        "greater bay area",
        "гуандун",
        "guangdong",
        "hong kong",
        "гонконг",
        "шэньчж",
        "shenzhen",
        "гуанчж",
        "guangzhou",
        "макао",
        "macau",
    ),
    "rostov": ("ростов", "rostov"),
    "tehran": ("тегеран", "tehran"),
}


def failure(code: str, message: str, status: int = 400) -> web.Response:
    return web.json_response({"error": {"code": code, "message": message}}, status=status)


def local_host(host: str) -> bool:
    try:
        name = urlsplit("http://" + host).hostname
        return name == "localhost" or ipaddress.ip_address(name or "").is_loopback
    except ValueError:
        return False


class DashboardState:
    def __init__(
        self,
        settings,
        repository,
        access_token,
        allow_paid,
        static_dir,
        reports=None,
        article_builder=build_detail_article,
        channel_service=None,
        paid_budget_usd=2.0,
    ):
        if not isinstance(access_token, str) or len(access_token) < 24:
            raise ValueError("Dashboard access token must contain at least 24 characters")
        self.settings, self.repository = settings, repository
        self.store = DashboardStore(settings, repository, reports=reports)
        self.access_token = access_token
        self.allow_paid = allow_paid
        self.budget = LocalBudget(repository, paid_budget_usd)
        self.static_dir = Path(static_dir).resolve() if static_dir else None
        self.sessions: dict[str, tuple[str, float]] = {}
        self.login_attempts: list[float] = []
        self.tasks: set[asyncio.Task] = set()
        self.semaphore = asyncio.Semaphore(1)
        self.article_builder = article_builder
        # Keys removed when paid access is disabled, including @handle resolution.
        channel_settings = (
            settings if allow_paid else settings.model_copy(update={"monid_api_key": None})
        )
        self.channel_service = channel_service or ChannelService(channel_settings, repository)

    async def call(self, method, *args, **kwargs):
        # Each Repository operation opens its own SQLite connection in this thread.
        return await asyncio.to_thread(method, *args, **kwargs)

    def session(self, request):
        session = self.sessions.get(request.cookies.get(COOKIE, ""))
        return session if session and session[1] > time.monotonic() else None


@web.middleware
async def security_headers(request, handler):
    # Outermost middleware covers denied login, Host, and CSRF requests too.
    response = await handler(request)
    response.headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    return response


@web.middleware
async def protection(request, handler):
    state = request.app[STATE]
    if not local_host(request.host):
        return failure("invalid_host", "Доступ разрешён только через localhost.", 403)
    if request.remote:
        try:
            if not ipaddress.ip_address(request.remote).is_loopback:
                return failure("not_local", "Этот интерфейс доступен только локально.", 403)
        except ValueError:
            return failure("not_local", "Локальный адрес не подтверждён.", 403)
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        origin = request.headers.get("Origin")
        expected = f"{request.scheme}://{request.host}"
        if origin is not None and origin != expected:
            return failure("invalid_origin", "Запрос пришёл с другой страницы.", 403)
        if request.headers.get("Sec-Fetch-Site") == "cross-site":
            return failure("invalid_origin", "Межсайтовые запросы запрещены.", 403)
        if request.content_type != "application/json":
            return failure("invalid_content_type", "Ожидается JSON.", 415)
    if request.path.startswith("/api/") and request.path != "/api/session":
        session = state.session(request)
        if session is None:
            return failure("unauthorized", "Открой приватную ссылку для входа.", 401)
        if request.method not in {"GET", "HEAD"} and not hmac.compare_digest(
            request.headers.get("X-CSRF-Token", ""), session[0]
        ):
            return failure("csrf", "Обнови страницу и повтори действие.", 403)
    try:
        response = await handler(request)
    except web.HTTPException as exc:
        response = failure("http_error", "Запрос недоступен.", exc.status)
    except (ValueError, TypeError, KeyError, binascii.Error):
        response = failure("invalid_request", "Проверь параметры запроса.")
    except Exception:
        # Never leak provider errors, paths, credentials, or source raw payloads.
        response = failure("internal_error", "Не удалось выполнить запрос. Попробуй позже.", 500)
    return response


async def payload(request) -> dict:
    result = await request.json()
    if not isinstance(result, dict):
        raise ValueError("Expected object")
    return result


def item_key(request) -> tuple[str, int]:
    kind, item_id = request.match_info["kind"], int(request.match_info["id"])
    if kind not in {"news", "video"} or item_id < 1:
        raise ValueError("Invalid item")
    return kind, item_id


async def session_create(request):
    state = request.app[STATE]
    now = time.monotonic()
    state.login_attempts = [stamp for stamp in state.login_attempts if now - stamp < 60]
    if len(state.login_attempts) >= 10:
        return failure("rate_limited", "Слишком много попыток. Подожди минуту.", 429)
    value = (await payload(request)).get("token")
    if not isinstance(value, str) or not hmac.compare_digest(value, state.access_token):
        state.login_attempts.append(now)
        return failure("unauthorized", "Неверный ключ приватной ссылки.", 401)
    state.sessions = {key: session for key, session in state.sessions.items() if session[1] > now}
    if len(state.sessions) >= 20:
        state.sessions.pop(next(iter(state.sessions)))
    session_id, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(24)
    state.sessions[session_id] = (csrf, now + 12 * 3600)
    response = web.json_response({"ok": True, "csrf": csrf})
    response.set_cookie(
        COOKIE,
        session_id,
        httponly=True,
        samesite="Strict",
        max_age=43200,
        secure=request.secure,
        path="/",
    )
    return response


async def session_delete(request):
    state = request.app[STATE]
    state.sessions.pop(request.cookies.get(COOKIE, ""), None)
    response = web.json_response({"ok": True})
    response.del_cookie(COOKIE, path="/")
    return response


async def bootstrap(request):
    state = request.app[STATE]
    data = await state.call(state.store.overview)
    data["status"]["budget_scope"] = "local_snapshot_not_live_account"
    data["paid_budget"] = {
        **await state.call(state.budget.snapshot),
        "enabled": state.allow_paid,
        "scope": "local_dashboard_since_baseline",
    }
    data.update(
        {
            "csrf": state.session(request)[0],
            "profile": {"name": "OSVED"},
            "features": {
                "reactions": True,
                "saved": True,
                "news_context": True,
                "news_article": True,
                "video_short": True,
                "video_long": True,
                "channels_add": True,
                "paid_generation": state.allow_paid,
                "region_filter": "text_match",
                "telegram_delivery": False,
                "automatic_collection": False,
            },
        }
    )
    try:
        metadata = json.loads((state.settings.data_dir / "snapshot.json").read_text())
        data["status"]["snapshot_at"] = metadata.get("snapshot_at") or metadata.get("created_at")
    except (OSError, ValueError):
        data["status"]["snapshot_at"] = None
    return web.json_response(data)


async def feed(request):
    state = request.app[STATE]
    folder = request.query.get("folder", "all")
    if folder not in FOLDERS:
        raise ValueError("Unknown folder")
    limit = max(1, min(50, int(request.query.get("limit", 20))))
    saved = request.query.get("saved", "0") in {"true", "1"}
    query = request.query.get("q", "").strip().casefold()[:200]
    region = request.query.get("region", "")
    if region and region not in REGIONS:
        raise ValueError("Unknown region")
    items = await state.call(state.store.all_items)
    items = [
        item
        for item in items
        if (folder == "all" or item["folder"] == folder) and (not saved or item["saved"])
    ]
    if query or region:

        def matches(item):
            text = " ".join((item["title"], item["summary"], " ".join(item["regions"]))).casefold()
            return (not query or query in text) and (
                not region or any(word in text for word in REGIONS[region])
            )

        items = [item for item in items if matches(item)]
    total = len(items)
    cursor = request.query.get("cursor")
    if cursor:
        if len(cursor) > 512:
            raise ValueError("Invalid cursor")
        boundary = json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())
        if (
            not isinstance(boundary, list)
            or len(boundary) != 2
            or not all(isinstance(part, str) for part in boundary)
        ):
            raise ValueError("Invalid cursor")
        items = [item for item in items if [item["published_at"], item["key"]] < boundary]
    page = items[:limit]
    has_more = len(items) > limit
    next_cursor = (
        base64.urlsafe_b64encode(
            json.dumps(
                [
                    page[-1]["published_at"],
                    page[-1]["key"],
                ]
            ).encode()
        ).decode()
        if page and has_more
        else None
    )
    return web.json_response(
        {
            "items": page,
            "cursor": next_cursor,
            "has_more": has_more,
            "total": total,
            "region_filter": "text_match" if region else None,
        }
    )


async def item_get(request):
    state = request.app[STATE]
    item = await state.call(state.store.item, *item_key(request))
    return web.json_response(item) if item else failure("not_found", "Публикация не найдена.", 404)


async def item_write(request):
    state = request.app[STATE]
    kind, item_id = item_key(request)
    row = await state.call(state.store.raw_item, kind, item_id)
    if row is None:
        return failure("not_found", "Публикация не найдена.", 404)
    data = await payload(request)
    if request.match_info["action"] == "reaction":
        reaction = data.get("reaction")
        if "reaction" not in data or reaction not in REACTIONS:
            raise ValueError("Unknown reaction")
        await state.call(state.store.react, kind, row, reaction)
    else:
        if type(data.get("saved")) is not bool:
            raise ValueError("Expected boolean")
        await state.call(state.store.set_saved, kind, item_id, data["saved"])
    return web.json_response({"item": await state.call(state.store.item, kind, item_id)})


def cached_expansion(item: dict, level: str) -> dict | None:
    content = item["content"].get(level)
    if content or (level == "long" and item["pdf_url"]):
        return {
            "status": "ready",
            "content_html": content,
            "pdf_url": item["pdf_url"] if level == "long" else None,
            "cached": True,
        }
    return None


def job_update(state, job_id, *, status=None, stage=None, result=None, error=None):
    with state.repository.connection() as db:
        db.execute(
            "UPDATE web_jobs SET status=COALESCE(?,status),stage=COALESCE(?,stage),"
            "result_json=COALESCE(?,result_json),error=?,updated_at=? WHERE id=?",
            (status, stage, json.dumps(result) if result else None, error, utc_now(), job_id),
        )


def execute_job(state, job_id, kind, row, level):
    job_update(state, job_id, status="running", stage="preparing")

    def progress(stage):
        job_update(state, job_id, stage=stage)

    try:
        if state.repository.current_month_cost() >= state.settings.budget_hard_stop_usd:
            raise ReportError("budget_blocked")
        maximum = (
            state.settings.details_max_cost_usd
            if kind == "news"
            else state.settings.video_report_max_cost_usd
        )
        state.budget.reserve(job_id, maximum)
        if kind == "news":
            artifact = state.article_builder(
                state.settings, state.repository, row["event_id"], progress=progress
            )
            if artifact.validation_status != "valid":
                raise ReportError("insufficient_evidence")
            state.store.store_article(artifact)
            result = {"content_html": safe_html(artifact.rendered_html), "pdf_url": None}
        elif level == "short":
            text = state.store.reports.build_tldr(row["source_url"], progress=progress)
            result = {"content_html": safe_html(telegram_tldr(text)), "pdf_url": None}
        else:
            state.store.reports.build(row["source_url"], progress=progress)
            result = {"content_html": None, "pdf_url": f"/api/items/video/{row['id']}/pdf"}
        job_update(state, job_id, status="ready", stage="complete", result=result)
        state.budget.finish(job_id)
    except LocalBudgetBlocked:
        job_update(state, job_id, status="failed", stage="complete", error="local_budget_blocked")
    except (ReportError, DetailBudgetError, DetailEvidenceError, DetailGenerationError) as exc:
        code = getattr(exc, "status", None) or {
            DetailBudgetError: "budget_blocked",
            DetailEvidenceError: "insufficient_evidence",
        }.get(type(exc), "generation_failed")
        job_update(state, job_id, status="failed", stage="complete", error=code)
        state.budget.finish(job_id, uncertain=code in {"uncertain", "processing"})
    except Exception:
        # Uncertain paid requests never autoretry, even after process restart.
        job_update(state, job_id, status="failed", stage="complete", error="generation_failed")
        state.budget.finish(job_id, uncertain=True)


async def run_job(state, job_id, kind, row, level):
    async with state.semaphore:
        await state.call(execute_job, state, job_id, kind, row, level)


async def expand(request):
    state = request.app[STATE]
    kind, item_id = item_key(request)
    level = (await payload(request)).get("level")
    if level not in ({"context", "article"} if kind == "news" else {"short", "long"}):
        raise ValueError("Unknown expansion level")
    item = await state.call(state.store.item, kind, item_id)
    if item is None:
        return failure("not_found", "Публикация не найдена.", 404)
    cached = cached_expansion(item, level)
    if cached:
        return web.json_response(cached)
    if not state.allow_paid:
        return failure(
            "generation_disabled",
            "В локальной версии платные запросы выключены. Доступны уже сохранённые тексты и PDF.",
            409,
        )
    if (kind == "news" and not state.settings.details_enabled) or (
        kind == "video" and not state.settings.video_reports_enabled
    ):
        return failure("generation_disabled", "Этот вид генерации выключен в настройках.", 409)
    budget = await state.call(state.budget.snapshot)
    maximum = (
        state.settings.details_max_cost_usd
        if kind == "news"
        else state.settings.video_report_max_cost_usd
    )
    if maximum > budget["remaining_usd"] + 1e-9:
        return failure(
            "local_budget_blocked",
            "Локальный бюджет теста исчерпан: "
            "не хватает резерва на запрос. Сохранённые материалы доступны.",
            409,
        )
    row = await state.call(state.store.raw_item, kind, item_id)
    # Single DB uniqueness key prevents parallel clicks and duplicate provider calls.
    key = f"{kind}:{row.get('event_id', item_id)}:{level}"

    def reserve():
        with state.repository.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT id,status FROM web_jobs WHERE job_key=?", (key,)).fetchone()
            if old:
                return old["id"], old["status"], False
            if (
                db.execute(
                    "SELECT COUNT(*) FROM web_jobs WHERE status IN ('queued','running')"
                ).fetchone()[0]
                >= 3
            ):
                return None, "queue_full", False
            job_id = secrets.token_urlsafe(18)
            db.execute(
                "INSERT INTO web_jobs(id,job_key,kind,item_id,level,status,stage,"
                "created_at,updated_at) "
                "VALUES(?,?,?,?,?,'queued','queued',?,?)",
                (job_id, key, kind, item_id, level, utc_now(), utc_now()),
            )
            return job_id, "queued", True

    job_id, status, created = await state.call(reserve)
    if not job_id:
        return failure("queue_full", "Уже готовятся три запроса. Подожди завершения.", 429)
    if created:
        task = asyncio.create_task(run_job(state, job_id, kind, row, level))
        state.tasks.add(task)
        task.add_done_callback(state.tasks.discard)
    return web.json_response({"status": status, "job_id": job_id}, status=202)


async def job_get(request):
    state = request.app[STATE]

    def read():
        with state.repository.connection() as db:
            row = db.execute(
                "SELECT * FROM web_jobs WHERE id=?", (request.match_info["id"],)
            ).fetchone()
        if not row:
            return None
        result = json.loads(row["result_json"] or "{}")
        return {
            "status": row["status"],
            "stage": row["stage"],
            "error": row["error"],
            "content_html": result.get("content_html"),
            "pdf_url": result.get("pdf_url"),
        }

    result = await state.call(read)
    return web.json_response(result) if result else failure("not_found", "Задача не найдена.", 404)


async def sources_get(request):
    state = request.app[STATE]
    folder = request.query.get("folder", "all")
    if folder not in FOLDERS:
        raise ValueError("Invalid folder")
    return web.json_response({"sources": await state.call(state.store.sources, folder)})


async def channels_get(request):
    state = request.app[STATE]
    return web.json_response({"channels": await state.call(state.store.channels)})


async def channels_add(request):
    state = request.app[STATE]
    data = await payload(request)
    try:
        url = channel_url(str(data.get("url", "")))
        # Resolution can spend money, never implicit in dev mode.
        if not state.allow_paid and "/channel/UC" not in url:
            return failure(
                "resolution_disabled",
                "Для локального теста используй ссылку "
                "youtube.com/channel/UC…: распознавание @имени платное и выключено.",
                409,
            )
        operation_id = "channel:" + secrets.token_urlsafe(18)
        async with state.semaphore:
            paid_resolution = "/channel/UC" not in url
            if paid_resolution:
                await state.call(state.budget.reserve, operation_id, CHANNEL_RESOLUTION_PRICE)
            try:
                row = await state.call(
                    state.channel_service.add, url, state.settings.allowed_user_id or 0
                )
            except Exception:
                if paid_resolution:
                    await state.call(state.budget.finish, operation_id, uncertain=True)
                raise
            else:
                if paid_resolution:
                    await state.call(state.budget.finish, operation_id)
    except LocalBudgetBlocked:
        return failure(
            "local_budget_blocked", "Недостаточно локального бюджета на распознавание канала.", 409
        )
    except ChannelError as exc:
        return failure("channel_error", str(exc), 400)
    return web.json_response(
        {
            "channel": {
                "id": row["source_id"],
                "name": row.get("title", "YouTube"),
                "url": url,
                "enabled": True,
            },
            "local_only": True,
        },
        status=201,
    )


async def pdf_get(request):
    state = request.app[STATE]
    kind, item_id = item_key(request)
    if kind != "video":
        return failure("not_found", "PDF не найден.", 404)
    row = await state.call(state.store.raw_item, kind, item_id)
    path = await state.call(state.store.reports.cached_pdf, row["source_url"]) if row else None
    if path is None:
        return failure("not_found", "PDF ещё не готов.", 404)
    # Only a service-verified report, never a URL-provided file path or arbitrary data file.
    path = Path(path).resolve()
    root = (state.settings.data_dir / "video-reports").resolve()
    if not path.is_relative_to(root) or path.suffix != ".pdf":
        return failure("not_found", "PDF недоступен.", 404)
    return web.FileResponse(
        path,
        headers={
            "Content-Type": "application/pdf",
            "Content-Disposition": 'inline; filename="OSVED-report.pdf"',
        },
    )


async def static_get(request):
    state = request.app[STATE]
    if state.static_dir is None:
        raise web.HTTPNotFound()
    name = request.match_info.get("path") or "index.html"
    path = (state.static_dir / name).resolve()
    if (
        not path.is_relative_to(state.static_dir)
        or not path.is_file()
        or any(part.startswith(".") for part in Path(name).parts)
    ):
        raise web.HTTPNotFound()
    return web.FileResponse(path)


async def startup(app):
    state = app[STATE]
    await state.call(state.store.initialize)
    await state.call(state.budget.initialize, enabled=state.allow_paid)

    def recover():
        with state.repository.connection() as db:
            db.execute(
                "UPDATE web_jobs SET status='failed',stage='complete',"
                "error='interrupted_no_retry',updated_at=? WHERE status IN ('queued','running')",
                (utc_now(),),
            )

    await state.call(recover)


async def cleanup(app):
    # Do not cancel an in-flight paid request (its underlying thread cannot be cancelled).
    state = app[STATE]
    if state.tasks:
        await asyncio.gather(*state.tasks, return_exceptions=True)


def create_app(
    settings: Settings,
    repository: Repository | None = None,
    *,
    access_token: str,
    static_dir: Path | None = None,
    allow_paid: bool = False,
    paid_budget_usd: float = 2.0,
    reports=None,
    article_builder=build_detail_article,
    channel_service=None,
) -> web.Application:
    """Build the app over an already-migrated isolated snapshot; no bot start/migration.

    Bind only 127.0.0.1. The Host/peer checks also reject accidental public binding.
    The caller is responsible for copying and migrating a nonproduction database.
    """
    app = web.Application(middlewares=[security_headers, protection], client_max_size=16_384)
    app[STATE] = DashboardState(
        settings,
        repository or Repository(settings.database_path),
        access_token,
        allow_paid,
        static_dir,
        reports,
        article_builder,
        channel_service,
        paid_budget_usd,
    )
    app.router.add_post("/api/session", session_create)
    app.router.add_post("/api/logout", session_delete)
    app.router.add_get("/api/bootstrap", bootstrap)
    app.router.add_get("/api/feed", feed)
    app.router.add_get("/api/items/{kind}/{id}", item_get)
    app.router.add_post("/api/items/{kind}/{id}/{action:reaction|saved}", item_write)
    app.router.add_post("/api/items/{kind}/{id}/expand", expand)
    app.router.add_get("/api/items/{kind}/{id}/pdf", pdf_get)
    app.router.add_get("/api/jobs/{id}", job_get)
    app.router.add_get("/api/sources", sources_get)
    app.router.add_get("/api/channels", channels_get)
    app.router.add_post("/api/channels", channels_add)
    app.router.add_get("/static/{path:.*}", static_get)
    app.router.add_get("/{path:.*}", static_get)
    app.on_startup.append(startup)
    app.on_cleanup.append(cleanup)
    return app
