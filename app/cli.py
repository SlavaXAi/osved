from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime
from getpass import getpass
from pathlib import Path
from typing import Annotated

import typer

from app import __version__
from app.collectors.orchestrator import collect_due_sources
from app.collectors.state import source_health_report
from app.collectors.telegram import (
    TelegramReaderNotConfigured,
    TelegramSessionNotAuthorized,
    setup_telegram_session,
)
from app.config import DeliveryFormat, Settings, runtime_source_registry
from app.db.repository import Repository
from app.details.research import DetailResearchError, build_detail_article
from app.details.worker import DetailWorker
from app.intelligence.service import INTELLIGENCE_KINDS, IntelligenceService, normalize_now
from app.ops.baseline import export_baseline
from app.ops.budget import evaluate_budget
from app.ops.calibration import run_calibration_batch
from app.ops.cycle import run_empty_cycle
from app.ops.schedule import (
    SCHEDULE_DRY_RUN_SETTING,
    SCHEDULE_ENABLED_SETTING,
    configure_schedule,
    run_schedule_loop,
    run_schedule_tick,
    setting_bool,
)
from app.pipeline.cards import generate_cards
from app.pipeline.dedupe import deduplicate_events
from app.pipeline.media import MEDIA_POLICY_VERSION, select_preview_first_media
from app.pipeline.selection import prepare_candidates, reclassify_pending_candidates
from app.pipeline.voice import generate_author_draft
from app.radar.search import SearchRadarError, run_search_radar
from app.research.dossier import build_event_dossier
from app.research.facts import build_event_fact_sheet
from app.research.web import ResearchEventService
from app.telegram_bot.bot import (
    CandidateBatchNotDeliverable,
    CardsNotDeliverable,
    TelegramBotNotConfigured,
    run_bot,
    send_candidate_batch,
    send_cards_by_ids,
    send_pending_cards,
    send_smoke_card,
)

app = typer.Typer(
    name="newsfeed",
    help="OSVED: personal Telegram news radar.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


def runtime() -> tuple[Settings, Repository]:
    settings = Settings()
    settings.ensure_runtime_directories()
    repository = Repository(settings.database_path)
    return settings, repository


def bootstrap() -> tuple[Settings, Repository]:
    settings, repository = runtime()
    repository.migrate()
    repository.sync_sources(runtime_source_registry(settings))
    repository.claim_runtime_setting("news_delivery_mode", settings.news_delivery_mode)
    return settings, repository


def prompt_secret(label: str, *, required: bool) -> str:
    while True:
        value = getpass(f"{label}: ").strip()
        if value or not required:
            return value
        typer.echo(f"{label} is required.", err=True)


def write_private_env(path: Path, values: dict[str, str]) -> None:
    lines = [
        "# Generated locally by newsfeed configure-secrets. Never commit this file.",
        "APP_ENV=development",
        "DATA_DIR=data",
        "SOURCES_FILE=config/sources.yaml",
        "INTERESTS_FILE=config/interests.yaml",
        "STYLE_PROFILE_FILE=config/style-profile.yaml",
        "LOG_LEVEL=INFO",
        "TELEGRAM_SESSION_PATH=data/telegram-reader.session",
        "BUDGET_TARGET_USD=8",
        "BUDGET_WARNING_USD=9",
        "BUDGET_HARD_STOP_USD=10",
        "VIDEO_REPORT_BUDGET_ENABLED=true",
        "RADAR_COLLECT_LIMIT=100",
    ]
    lines.extend(f"{key}={value}" for key, value in values.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


def update_env_value(path: Path, key: str, value: str) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    prefix = f"{key}="
    updated = False
    result: list[str] = []
    for line in lines:
        if line.startswith(prefix):
            result.append(f"{prefix}{value}")
            updated = True
        else:
            result.append(line)
    if not updated:
        result.append(f"{prefix}{value}")
    path.write_text("\n".join(result) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


@app.command("version")
def version_command() -> None:
    typer.echo(__version__)


@app.command("configure-secrets")
def configure_secrets(
    env_path: Annotated[Path, typer.Option("--env-path")] = Path(".env"),
) -> None:
    """Prompt locally for credentials and write a private mode-0600 .env file."""
    if env_path.exists() and not typer.confirm(f"Replace existing {env_path}?"):
        raise typer.Abort()

    typer.echo("Values are hidden while typing and are never printed back.")
    values = {
        "TELEGRAM_BOT_TOKEN": prompt_secret("Telegram bot token", required=True),
        "TELEGRAM_API_ID": prompt_secret("Telegram API ID (optional MTProto)", required=False),
        "TELEGRAM_API_HASH": prompt_secret("Telegram API hash (optional MTProto)", required=False),
        "ALLOWED_USER_ID": prompt_secret("Allowed Telegram user ID", required=True),
        "OPENROUTER_API_KEY": prompt_secret(
            "OpenRouter API key (may be left blank)", required=False
        ),
        "OPENROUTER_MODEL": "",
        "OPENROUTER_CARD_MODEL": "openai/gpt-5.6-luna",
        "OPENROUTER_FALLBACK_MODEL": "openai/gpt-4.1-mini",
        "MONID_API_KEY": prompt_secret("Monid API key (may be left blank)", required=False),
    }
    write_private_env(env_path, values)
    typer.echo(f"Saved {env_path} with mode 0600. Secret values were not displayed.")


@app.command("set-owner")
def set_owner(
    user_id: int,
    env_path: Annotated[Path, typer.Option("--env-path")] = Path(".env"),
) -> None:
    """Set ALLOWED_USER_ID without reading or printing any other secret."""
    try:
        update_env_value(env_path, "ALLOWED_USER_ID", str(user_id))
    except FileNotFoundError:
        typer.echo(f"{env_path} does not exist; run configure-secrets first", err=True)
        raise typer.Exit(code=2) from None
    typer.echo(f"Owner user ID saved in {env_path}.")


@app.command("init-db")
def init_db() -> None:
    """Create or upgrade the local SQLite database."""
    settings, repository = runtime()
    applied = repository.migrate()
    typer.echo(
        json.dumps(
            {
                "database": str(settings.database_path),
                "applied_migrations": applied,
            },
            ensure_ascii=False,
        )
    )


@app.command("sync-sources")
def sync_sources() -> None:
    """Validate sources.yaml and copy it into SQLite."""
    settings, repository = runtime()
    repository.migrate()
    registry = runtime_source_registry(settings)
    count = repository.sync_sources(registry)
    typer.echo(json.dumps({"synced_sources": count}, ensure_ascii=False))


@app.command("doctor")
def doctor() -> None:
    """Show safe configuration status without printing secret values."""
    settings, repository = bootstrap()
    credentials = settings.credential_status()
    credentials["allowed_user"] = bool(
        settings.allowed_user_id or repository.get_runtime_setting("allowed_user_id")
    )
    month_cost = repository.current_month_cost()
    budget = evaluate_budget(
        month_cost,
        settings.budget_warning_usd,
        settings.budget_hard_stop_usd,
    )
    typer.echo(
        json.dumps(
            {
                "app_env": settings.app_env,
                "database": str(settings.database_path),
                "credentials": credentials,
                "storage": repository.status_counts(),
                "budget_state": budget.state.value,
                "paid_operations_allowed": budget.paid_operations_allowed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("source-report")
def source_report() -> None:
    """Show source health and measured yield without performing external requests."""
    _, repository = bootstrap()
    typer.echo(json.dumps(repository.source_yield_rows(), ensure_ascii=False, indent=2))


@app.command("baseline-export")
def baseline_export_command(
    limit: Annotated[int, typer.Option("--limit", min=1, max=500)] = 50,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
) -> None:
    """Create a private, reproducible Phase-0 snapshot without external requests."""
    settings, repository = runtime()
    if not repository.database_path.exists():
        raise typer.BadParameter("Database does not exist; run init-db first")
    target = output_dir or (
        settings.data_dir
        / "baselines"
        / f"v7-phase0-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S')}"
    )
    try:
        result = export_baseline(repository, output_dir=target, limit=limit)
    except FileExistsError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(result.__dict__, ensure_ascii=False, indent=2))


@app.command("run")
def run(once: bool = typer.Option(False, "--once", help="Run one safe cycle and exit.")) -> None:
    """Run the worker. Phase 0 intentionally supports only a single empty cycle."""
    if not once:
        raise typer.BadParameter("Phase 0 requires --once; scheduling starts in phase 1.")
    settings, repository = bootstrap()
    result = run_empty_cycle(settings, repository)
    typer.echo(
        json.dumps(
            {
                "run_id": result.run_id,
                "phase": 0,
                "sources": result.source_count,
                "processed": result.processed_count,
                "external_requests": 0,
                "cost_usd": 0,
                "budget_state": result.budget_state.value,
            },
            ensure_ascii=False,
        )
    )


def phase_zero_noop(run_type: str) -> None:
    settings, repository = bootstrap()
    run_id = repository.start_run(run_type, {"phase": 0, "implemented": False})
    repository.finish_run(run_id, status="skipped")
    typer.echo(
        json.dumps(
            {
                "run_id": run_id,
                "status": "skipped",
                "reason": f"{run_type} starts in a later phase",
                "external_requests": 0,
                "cost_usd": 0,
                "app_env": settings.app_env,
            },
            ensure_ascii=False,
        )
    )


@app.command("collect-once")
def collect_once(
    kind: Annotated[str, typer.Option("--kind")] = "none",
    provider: Annotated[str, typer.Option("--provider")] = "auto",
    limit_per_source: Annotated[int, typer.Option("--limit-per-source", min=1, max=500)] = 100,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Ignore the per-source cadence for this run (mainly useful for X diagnostics).",
        ),
    ] = False,
) -> None:
    """Collect one source class; defaults to a safe no-op."""
    if kind == "none":
        phase_zero_noop("collect_once")
        return
    if kind not in {
        "telegram",
        "rss",
        "github",
        "x",
        "reddit",
        "youtube",
        "v2ex",
        "weibo",
        "zhihu",
        "all",
    }:
        raise typer.BadParameter("Unknown source kind; use all for all due sources")
    if provider not in {"auto", "mtproto", "monid"}:
        raise typer.BadParameter("--provider must be auto, mtproto, or monid")

    settings, repository = bootstrap()
    selected_provider = provider if kind == "telegram" else ("monid" if kind == "x" else "direct")
    if kind == "x" and provider not in {"auto", "monid"}:
        raise typer.BadParameter("X collection currently supports --provider auto or monid")
    if kind == "telegram" and selected_provider == "auto":
        selected_provider = (
            "mtproto"
            if settings.telegram_api_id is not None and settings.telegram_api_hash is not None
            else "monid"
        )
    run_id = repository.start_run(
        f"collect_{kind}",
        {
            "limit_per_source": limit_per_source,
            "provider": selected_provider,
            "force": force,
        },
    )
    try:
        stats = collect_due_sources(
            settings,
            repository,
            kind=kind,
            provider=provider,
            force=force,
            limit_per_source=limit_per_source,
        )
        repository.finish_run(
            run_id,
            processed_count=stats["items_inserted"],
            error_count=stats["errors"],
            cost_usd=stats["cost_usd"],
        )
    except (TelegramReaderNotConfigured, TelegramSessionNotAuthorized) as exc:
        repository.finish_run(run_id, status="failed", error_count=1, error=str(exc))
        typer.echo(f"Telegram reader is not ready: {exc}", err=True)
        raise typer.Exit(code=2) from None
    except Exception as exc:
        repository.finish_run(run_id, status="failed", error_count=1, error=str(exc))
        raise
    typer.echo(
        json.dumps(
            {"run_id": run_id, "provider": selected_provider, **stats},
            ensure_ascii=False,
        )
    )


@app.command("telegram-session-setup")
def telegram_session_setup() -> None:
    """Authorize the local Telegram user session interactively."""
    settings, _ = bootstrap()
    result = setup_telegram_session(settings)
    typer.echo(json.dumps(result, ensure_ascii=False))


@app.command("bot")
def bot_command() -> None:
    """Run the private delivery bot and feedback handlers."""
    settings, repository = bootstrap()
    try:
        asyncio.run(run_bot(settings, repository))
    except TelegramBotNotConfigured as exc:
        typer.echo(f"Telegram bot is not ready: {exc}", err=True)
        raise typer.Exit(code=2) from None


@app.command("telegram-smoke")
def telegram_smoke(
    source_url: Annotated[str, typer.Option("--source-url")] = (
        "https://core.telegram.org/bots/api"
    ),
    publication_format: Annotated[
        DeliveryFormat | None,
        typer.Option("--format", help="Override regular/article for this smoke message."),
    ] = None,
) -> None:
    """Send one real preview-and-feedback card to the configured owner."""
    settings, repository = bootstrap()
    try:
        delivery_id = asyncio.run(
            send_smoke_card(
                settings,
                repository,
                source_url,
                delivery_format=publication_format,
            )
        )
    except TelegramBotNotConfigured as exc:
        typer.echo(f"Telegram bot is not ready: {exc}", err=True)
        raise typer.Exit(code=2) from None
    typer.echo(json.dumps({"delivery_id": delivery_id, "status": "sent"}))


@app.command("prepare-candidates")
def prepare_candidates_command(
    limit: Annotated[int, typer.Option("--limit", min=1, max=1000)] = 200,
) -> None:
    """Apply cheap gates, URL dedupe, and transparent scoring."""
    settings, repository = bootstrap()
    stats = prepare_candidates(settings, repository, limit=limit)
    typer.echo(json.dumps(stats.__dict__, ensure_ascii=False))


@app.command("search-radar")
def search_radar_command(
    lookback_hours: Annotated[int, typer.Option("--lookback-hours", min=24, max=504)] = 120,
    preview_limit: Annotated[int, typer.Option("--preview-limit", min=1, max=32)] = 16,
    lens_ids: Annotated[list[str] | None, typer.Option("--lens")] = None,
    deliver: Annotated[bool, typer.Option("--deliver/--no-deliver")] = False,
) -> None:
    """Run bounded search agents and optionally deliver an isolated feedback batch."""
    settings, repository = bootstrap()
    try:
        stats = run_search_radar(
            settings,
            repository,
            lookback_hours=lookback_hours,
            preview_limit=preview_limit,
            lens_ids=lens_ids,
        )
        result: dict[str, object] = stats.as_dict()
        if deliver and stats.batch_id and stats.card_ids:
            result["delivery"] = asyncio.run(
                send_candidate_batch(
                    settings,
                    repository,
                    batch_id=stats.batch_id,
                    card_ids=stats.card_ids,
                )
            )
    except (SearchRadarError, CandidateBatchNotDeliverable, TelegramBotNotConfigured) as exc:
        typer.echo(f"Search radar failed: {exc}", err=True)
        raise typer.Exit(code=2) from None
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("deliver-candidate-batch")
def deliver_candidate_batch_command(batch_id: str) -> None:
    """Retry one explicit candidate batch; never consumes normal publication cards."""
    settings, repository = bootstrap()
    try:
        result = asyncio.run(send_candidate_batch(settings, repository, batch_id=batch_id))
    except (CandidateBatchNotDeliverable, TelegramBotNotConfigured) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from None
    typer.echo(json.dumps(result, ensure_ascii=False))


@app.command("deliver-candidate-cards")
def deliver_candidate_cards_command(card_ids: list[int]) -> None:
    """Create and send a quarantined batch from explicitly reviewed preview cards."""
    settings, repository = bootstrap()
    batch_id = f"candidate:{uuid.uuid4().hex[:12]}"
    try:
        result = asyncio.run(
            send_candidate_batch(
                settings,
                repository,
                batch_id=batch_id,
                card_ids=card_ids,
            )
        )
    except (CandidateBatchNotDeliverable, TelegramBotNotConfigured) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from None
    typer.echo(json.dumps({"batch_id": batch_id, **result}, ensure_ascii=False))


@app.command("reclassify-pending")
def reclassify_pending_command(
    limit: Annotated[int, typer.Option("--limit", min=1, max=5000)] = 500,
) -> None:
    """Backfill V7-A metadata/scoring and reject obvious routine pending events."""
    settings, repository = bootstrap()
    stats = reclassify_pending_candidates(settings, repository, limit=limit)
    typer.echo(json.dumps(stats.__dict__, ensure_ascii=False))


@app.command("generate-cards")
def generate_cards_command(
    limit: Annotated[int, typer.Option("--limit", min=1, max=50)] = 1,
    event_id: Annotated[int | None, typer.Option("--event-id")] = None,
) -> None:
    """Generate grounded Russian cards for the highest-scoring candidates."""
    settings, repository = bootstrap()
    stats = generate_cards(settings, repository, limit=limit, event_id=event_id)
    typer.echo(json.dumps(stats.__dict__, ensure_ascii=False))


@app.command("research-event")
def research_event_command(
    event_id: int,
    query: Annotated[str | None, typer.Option("--query")] = None,
    primary_url: Annotated[str | None, typer.Option("--primary-url")] = None,
    freshness: Annotated[str, typer.Option("--freshness")] = "last_month",
    max_pages: Annotated[int, typer.Option("--max-pages", min=1, max=2)] = 2,
) -> None:
    """Search and read a bounded evidence bundle for one selected event."""
    if freshness not in {"last_24_hours", "last_week", "last_month", "last_year"}:
        raise typer.BadParameter("Unsupported freshness window")
    settings, repository = bootstrap()
    stats = ResearchEventService(settings, repository).research(
        event_id,
        query=query,
        primary_url=primary_url,
        freshness=freshness,
        max_pages=max_pages,
    )
    typer.echo(json.dumps(stats.__dict__, ensure_ascii=False))


@app.command("build-fact-sheet")
def build_fact_sheet_command(event_id: int) -> None:
    """Extract and validate neutral sourced facts for one researched event."""
    settings, repository = bootstrap()
    result = build_event_fact_sheet(settings, repository, event_id)
    typer.echo(json.dumps(result, ensure_ascii=False))


@app.command("build-dossier")
def build_dossier_command(event_id: int) -> None:
    """Build a versioned decision snapshot from stored facts and evidence."""
    _, repository = bootstrap()
    result = build_event_dossier(repository, event_id)
    typer.echo(json.dumps(result, ensure_ascii=False))


@app.command("show-dossier")
def show_dossier_command(event_id: int) -> None:
    """Show the latest valid event dossier without new external requests."""
    _, repository = bootstrap()
    dossier = repository.latest_event_dossier(event_id)
    if dossier is None:
        typer.echo(f"No dossier for event_id={event_id}", err=True)
        raise typer.Exit(code=2)
    typer.echo(json.dumps(dossier, ensure_ascii=False, indent=2))


@app.command("backfill-media")
def backfill_media_command(
    limit: Annotated[int, typer.Option("--limit", min=1, max=1000)] = 100,
) -> None:
    """Apply the cheap preview-first media policy to existing valid cards."""
    _, repository = bootstrap()
    rows = repository.cards_missing_media(limit)
    previews = text_only = 0
    for row in rows:
        fallback_urls = [str(row["primary_preview_url"])] if row["primary_preview_url"] else []
        media = select_preview_first_media(
            repository.event_research_documents(int(row["event_id"])),
            fallback_urls=fallback_urls,
        )
        repository.create_card_media(
            card_id=int(row["card_id"]),
            policy_version=MEDIA_POLICY_VERSION,
            mode=media.mode,
            preview_url=media.preview_url,
            image_url=media.image_url,
            source_document_id=media.source_document_id,
            checks=media.checks,
        )
        if media.mode == "telegram_preview":
            previews += 1
        else:
            text_only += 1
    typer.echo(
        json.dumps(
            {"cards_seen": len(rows), "telegram_preview": previews, "text_only": text_only},
            ensure_ascii=False,
        )
    )


@app.command("generate-author-draft")
def generate_author_draft_command(card_id: int) -> None:
    """Create a separate grounded voice-layer draft; never deliver it automatically."""
    settings, repository = bootstrap()
    result = generate_author_draft(settings, repository, card_id)
    typer.echo(json.dumps(result, ensure_ascii=False))


@app.command("show-author-draft")
def show_author_draft_command(card_id: int) -> None:
    """Show the latest valid author draft without an external request."""
    _, repository = bootstrap()
    draft = repository.latest_author_draft(card_id)
    if draft is None:
        typer.echo(f"No valid author draft for card_id={card_id}", err=True)
        raise typer.Exit(code=2)
    typer.echo(json.dumps(draft, ensure_ascii=False, indent=2))


@app.command("dedupe-events")
def dedupe_events_command() -> None:
    """Merge near-identical event headlines inside a short time window."""
    _, repository = bootstrap()
    stats = deduplicate_events(repository)
    typer.echo(json.dumps(stats.__dict__, ensure_ascii=False))


@app.command("rebuild-derived")
def rebuild_derived_command() -> None:
    """Rebuild events/cards from raw inputs; refuses after any delivery or feedback."""
    _, repository = bootstrap()
    try:
        removed = repository.rebuild_derived_data()
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from None
    typer.echo(json.dumps({"removed": removed, "raw_items_preserved": True}))


@app.command("reject-card")
def reject_card_command(
    card_id: int,
    reason: Annotated[str, typer.Option("--reason")] = "manual_editorial_rejection",
) -> None:
    """Reject a generated card and return its event to the generation queue."""
    _, repository = bootstrap()
    try:
        repository.reject_card(card_id, reason)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from None
    typer.echo(json.dumps({"card_id": card_id, "status": "rejected"}))


@app.command("hold-card")
def hold_card_command(
    card_id: int,
    reason: Annotated[str, typer.Option("--reason")],
) -> None:
    """Reject a delivered card from feedback without requeueing the old pipeline."""
    _, repository = bootstrap()
    try:
        repository.hold_card_from_feedback(card_id, reason)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from None
    typer.echo(json.dumps({"card_id": card_id, "status": "feedback_rejected"}))


@app.command("retry-event")
def retry_event_command(event_id: int) -> None:
    """Return one failed event to the explicit generation queue."""
    _, repository = bootstrap()
    repository.set_event_status(event_id, "candidate", "manual_generation_retry")
    typer.echo(json.dumps({"event_id": event_id, "status": "candidate"}))


@app.command("deliver-once")
def deliver_once(
    limit: Annotated[int, typer.Option("--limit", min=1, max=50)] = 5,
    publication_format: Annotated[
        DeliveryFormat | None,
        typer.Option("--format", help="Override regular/article for this delivery."),
    ] = None,
) -> None:
    """Deliver valid undelivered cards to the private owner chat."""
    settings, repository = bootstrap()
    try:
        result = asyncio.run(
            send_pending_cards(
                settings,
                repository,
                limit=limit,
                delivery_format=publication_format,
            )
        )
    except TelegramBotNotConfigured as exc:
        typer.echo(f"Telegram bot is not ready: {exc}", err=True)
        raise typer.Exit(code=2) from None
    typer.echo(json.dumps(result, ensure_ascii=False))


@app.command("deliver-cards")
def deliver_cards_command(
    card_ids: Annotated[
        list[int],
        typer.Argument(help="Explicit card IDs to deliver, in delivery order."),
    ],
    publication_format: Annotated[
        DeliveryFormat,
        typer.Option("--format", help="Delivery format for this explicit batch."),
    ] = DeliveryFormat.REGULAR,
) -> None:
    """Deliver only the selected valid, undelivered cards to the private owner chat."""
    settings, repository = bootstrap()
    if any(card_id < 1 for card_id in card_ids):
        raise typer.BadParameter("CARD_ID values must be positive integers")
    try:
        result = asyncio.run(
            send_cards_by_ids(
                settings,
                repository,
                card_ids,
                delivery_format=publication_format,
            )
        )
    except CardsNotDeliverable as exc:
        unavailable = ", ".join(str(card_id) for card_id in exc.card_ids) or "none supplied"
        typer.echo(f"Cards are not deliverable: {unavailable}", err=True)
        raise typer.Exit(code=2) from None
    except TelegramBotNotConfigured as exc:
        typer.echo(f"Telegram bot is not ready: {exc}", err=True)
        raise typer.Exit(code=2) from None
    typer.echo(json.dumps(result, ensure_ascii=False))


@app.command("pipeline-once")
def pipeline_once(
    collect: Annotated[bool, typer.Option("--collect/--no-collect")] = False,
    provider: Annotated[str, typer.Option("--provider")] = "auto",
    collect_limit: Annotated[int, typer.Option("--collect-limit", min=1, max=100)] = 20,
    generate_limit: Annotated[int, typer.Option("--generate-limit", min=1, max=10)] = 1,
    deliver_limit: Annotated[int, typer.Option("--deliver-limit", min=1, max=10)] = 1,
    deliver: Annotated[bool, typer.Option("--deliver/--no-deliver")] = True,
) -> None:
    """Run one fresh radar→research→verified-message cycle; never schedules itself."""
    from app.collectors.ingest import redact_text
    from app.ops.schedule import ScheduledJobExecutor

    if provider not in {"auto", "mtproto", "monid"}:
        raise typer.BadParameter("--provider must be auto, mtproto, or monid")
    settings, repository = bootstrap()
    run_id = repository.start_run(
        "pipeline_once",
        {
            "collect": collect,
            "provider": provider,
            "collect_limit": collect_limit,
            "generate_limit": generate_limit,
            "deliver_limit": deliver_limit,
            "deliver": deliver,
        },
    )
    result: dict[str, object] = {"run_id": run_id, "status": "running", "scheduled": False}
    executor = ScheduledJobExecutor(settings, repository)
    ledger_before = repository.current_month_cost()
    cost = 0.0
    errors = 0
    cards_created = 0
    research_steps: list[dict] = []
    try:
        if collect:
            radar = executor.radar(provider=provider, limit_per_source=collect_limit)
            result["radar"] = radar
            cost += float(radar.get("cost_usd") or 0)
            errors += int(radar.get("errors") or 0)
        else:
            result["radar"] = {"status": "skipped", "reason": "--no-collect"}

        for _ in range(generate_limit):
            # The shared executor researches primary sources and creates a
            # verified card for one fresh event, never a raw trigger/preview.
            step = executor.research()
            research_steps.append(step)
            cost += float(step.get("cost_usd") or 0)
            errors += int(step.get("errors") or 0)
            cards_created += int(step.get("cards_created") or 0)
            if step.get("errors") or not step.get("cards_created"):
                # Do not repeatedly pay for the same unsuccessful candidate.
                break
        result["research"] = research_steps
        if not deliver or errors:
            result["delivery"] = {
                "status": "skipped",
                "sent": 0,
                "reason": "--no-deliver" if not deliver else "upstream_errors",
            }
        else:
            try:
                delivery = executor.delivery(limit=deliver_limit)
                errors += int(delivery.get("errors") or 0)
                result["delivery"] = delivery
            except TelegramBotNotConfigured as exc:
                result["delivery"] = {
                    "status": "skipped",
                    "sent": 0,
                    "reason": "telegram_not_configured",
                    "detail": redact_text(str(exc)),
                }
        result["status"] = "failed" if errors else "completed"
    except Exception as exc:
        errors += 1
        cost = max(cost, repository.current_month_cost() - ledger_before)
        result.update({"status": "failed", "error": redact_text(str(exc))[:1000]})
        result["research"] = research_steps
        result.setdefault(
            "delivery", {"status": "skipped", "reason": "upstream_failure", "sent": 0}
        )
    result.update({"cards_created": cards_created, "errors": errors, "cost_usd": cost})
    repository.finish_run(
        run_id,
        status=str(result["status"]),
        processed_count=cards_created,
        error_count=errors,
        cost_usd=cost,
        error=str(result.get("error") or "") or None,
    )
    typer.echo(json.dumps(result, ensure_ascii=False))
    if errors:
        raise typer.Exit(code=1)


@app.command("calibration-batch")
def calibration_batch(
    target: Annotated[int, typer.Option("--target", min=1, max=100)] = 50,
    candidate_limit: Annotated[int, typer.Option("--candidate-limit", min=1, max=300)] = 100,
    max_pages: Annotated[int, typer.Option("--max-pages", min=1, max=2)] = 2,
    primary_only: Annotated[bool, typer.Option("--primary-only/--all-sources")] = False,
    deliver: Annotated[bool, typer.Option("--deliver/--no-deliver")] = False,
) -> None:
    """Build a source-balanced feedback batch and optionally deliver it."""
    settings, repository = bootstrap()

    def show_progress(payload: dict[str, object]) -> None:
        typer.echo(json.dumps(payload, ensure_ascii=False))

    stats = run_calibration_batch(
        settings,
        repository,
        target=target,
        candidate_limit=candidate_limit,
        max_pages=max_pages,
        primary_only=primary_only,
        progress=show_progress,
    )
    result: dict[str, object] = {"generation": stats.as_dict()}
    if deliver and stats.cards_created:
        result["delivery"] = asyncio.run(
            send_pending_cards(
                settings,
                repository,
                limit=stats.cards_created,
                delivery_format=DeliveryFormat.REGULAR,
            )
        )
    typer.echo(json.dumps(result, ensure_ascii=False))


@app.command("schedule-enable")
def schedule_enable(
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run/--live", help="Keep external actions disabled or run live jobs."),
    ] = True,
) -> None:
    """Enable the scheduler; dry-run is the safe default."""
    _, repository = bootstrap()
    configure_schedule(repository, enabled=True, dry_run=dry_run)
    typer.echo(json.dumps({"enabled": True, "dry_run": dry_run}))


@app.command("schedule-disable")
def schedule_disable() -> None:
    """Activate the scheduler kill switch without deleting history."""
    _, repository = bootstrap()
    configure_schedule(repository, enabled=False, dry_run=True)
    typer.echo(json.dumps({"enabled": False, "dry_run": True}))


@app.command("schedule-status")
def schedule_status(
    limit: Annotated[int, typer.Option("--limit", min=1, max=100)] = 20,
) -> None:
    """Show scheduler mode and recent idempotent slots."""
    _, repository = bootstrap()
    typer.echo(
        json.dumps(
            {
                "enabled": setting_bool(repository, SCHEDULE_ENABLED_SETTING, False),
                "dry_run": setting_bool(repository, SCHEDULE_DRY_RUN_SETTING, True),
                "recent_slots": repository.recent_schedule_slots(limit),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("schedule-tick")
def schedule_tick(
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Plan due jobs even if the scheduler kill switch is on."),
    ] = False,
    at: Annotated[
        str | None, typer.Option("--at", help="ISO timestamp for a dry-run audit.")
    ] = None,
) -> None:
    """Run one idempotent scheduler tick and exit."""
    settings, repository = bootstrap()
    if at and not dry_run:
        raise typer.BadParameter("--at is only allowed together with --dry-run")
    try:
        now = datetime.fromisoformat(at) if at else None
    except ValueError as exc:
        raise typer.BadParameter("--at must be a valid ISO timestamp") from exc
    result = run_schedule_tick(
        settings,
        repository,
        now=now,
        force_dry_run=dry_run,
    )
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("schedule-loop")
def schedule_loop(
    interval_seconds: Annotated[int, typer.Option("--interval-seconds", min=30, max=300)] = 60,
) -> None:
    """Run the local scheduler loop; the database kill switch still applies."""
    settings, repository = bootstrap()
    run_schedule_loop(settings, repository, interval_seconds=interval_seconds)


def _parse_intelligence_now(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        return normalize_now(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError as exc:
        raise typer.BadParameter("--now must be a valid ISO timestamp") from exc


@app.command("contours-backfill")
def contours_backfill_command(
    limit: Annotated[int, typer.Option("--limit", min=1, max=100_000)] = 10_000,
) -> None:
    """Deterministically tag existing events with the seven active OSVED contours."""
    settings, repository = bootstrap()
    result = IntelligenceService(settings, repository).backfill_contours(limit=limit)
    typer.echo(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))


@app.command("intelligence-build")
def intelligence_build_command(
    kind: str,
    now: Annotated[
        str | None,
        typer.Option("--now", help="Optional ISO timestamp for a reproducible build."),
    ] = None,
) -> None:
    """Build and cache one derived product without network or LLM calls."""
    if kind not in INTELLIGENCE_KINDS:
        raise typer.BadParameter(f"KIND must be one of: {', '.join(INTELLIGENCE_KINDS)}")
    settings, repository = bootstrap()
    product = IntelligenceService(settings, repository).build(
        kind,
        now=_parse_intelligence_now(now),
    )
    typer.echo(json.dumps(product, ensure_ascii=False, indent=2))


@app.command("intelligence-show")
def intelligence_show_command(product_id: int) -> None:
    """Show one persisted intelligence product by ID."""
    settings, repository = bootstrap()
    product = IntelligenceService(settings, repository).show(product_id)
    if product is None:
        typer.echo(f"Unknown intelligence product: {product_id}", err=True)
        raise typer.Exit(code=2)
    typer.echo(json.dumps(product, ensure_ascii=False, indent=2))


@app.command("intelligence-latest")
def intelligence_latest_command(kind: str) -> None:
    """Show the latest persisted product of one kind."""
    if kind not in INTELLIGENCE_KINDS:
        raise typer.BadParameter(f"KIND must be one of: {', '.join(INTELLIGENCE_KINDS)}")
    settings, repository = bootstrap()
    product = IntelligenceService(settings, repository).latest(kind)
    if product is None:
        typer.echo(f"No intelligence product for kind: {kind}", err=True)
        raise typer.Exit(code=2)
    typer.echo(json.dumps(product, ensure_ascii=False, indent=2))


@app.command("intelligence-status")
def intelligence_status_command() -> None:
    """Show local product inventory; this command never performs external work."""
    settings, repository = bootstrap()
    status = IntelligenceService(settings, repository).status()
    typer.echo(json.dumps(status, ensure_ascii=False, indent=2))


@app.command("details-worker-once")
def details_worker_once_command() -> None:
    """Process at most one queued on-demand article, then exit."""
    settings, repository = bootstrap()
    result = DetailWorker(settings, repository).run_once()
    typer.echo(json.dumps(result.__dict__, ensure_ascii=False, indent=2))


@app.command("details-build")
def details_build_command(event_id: int) -> None:
    """Build one manual deep-research artifact without Telegram delivery."""
    settings, repository = bootstrap()
    try:
        result = build_detail_article(settings, repository, event_id)
    except (DetailResearchError, ValueError) as exc:
        typer.echo(f"Detailed research failed: {exc}", err=True)
        raise typer.Exit(code=2) from None
    typer.echo(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


@app.command("details-worker")
def details_worker_command(
    loop: Annotated[
        bool,
        typer.Option("--loop/--once", help="Keep polling or process at most one request."),
    ] = False,
    poll_interval_seconds: Annotated[
        float,
        typer.Option("--poll-interval-seconds", min=0.25, max=60),
    ] = 2.0,
) -> None:
    """Run the durable on-demand research worker outside the Telegram process."""
    settings, repository = bootstrap()
    worker = DetailWorker(settings, repository)
    if loop:
        worker.run_loop(poll_interval_seconds=poll_interval_seconds)
        return
    typer.echo(json.dumps(worker.run_once().__dict__, ensure_ascii=False, indent=2))


@app.command("details-status")
def details_status_command(
    limit: Annotated[int, typer.Option("--limit", min=1, max=100)] = 20,
) -> None:
    """Inspect the on-demand queue without network or model calls."""
    _, repository = bootstrap()
    typer.echo(
        json.dumps(
            {
                "counts": repository.detail_status_counts(),
                "recent": repository.recent_detail_requests(limit=limit),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("details-retry")
def details_retry_command(request_id: int) -> None:
    """Explicitly requeue one failed, evidence-poor, or budget-blocked request."""
    _, repository = bootstrap()
    try:
        request = repository.retry_detail_request(request_id)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from None
    typer.echo(json.dumps(request, ensure_ascii=False, indent=2))


@app.command("backfill")
def backfill() -> None:
    """Reserved for calibration backfill; safe no-op in phase 0."""
    phase_zero_noop("backfill")


@app.command("export-feedback")
def export_feedback(
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("exports"),
) -> None:
    """Export private feedback as JSON and Markdown for a manual review."""
    _, repository = bootstrap()
    rows = repository.export_feedback_rows()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "feedback.json"
    markdown_path = output_dir / "feedback.md"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Feedback export", "", f"Events: {len(rows)}", ""]
    for row in rows:
        lines.extend(
            [
                f"## Feedback {row['id']}",
                "",
                f"- Reaction: `{row['reaction'] or 'none'}`",
                f"- Action: `{row['action']}`",
                f"- Created: `{row['created_at']}`",
                f"- Comment: {row['comment'] or '—'}",
                "",
            ]
        )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    typer.echo(
        json.dumps(
            {
                "rows": len(rows),
                "json": str(json_path),
                "markdown": str(markdown_path),
            },
            ensure_ascii=False,
        )
    )


@app.command("add-feedback-note")
def add_feedback_note(
    delivery_id: int,
    comment: Annotated[str, typer.Option("--comment")],
    reaction: Annotated[str | None, typer.Option("--reaction")] = None,
) -> None:
    """Attach explicit feedback received outside the Telegram buttons."""
    if reaction not in {None, "like", "dislike", "superlike", "superdislike"}:
        raise typer.BadParameter("--reaction must be like, dislike, superlike, or superdislike")
    _, repository = bootstrap()
    try:
        feedback_id = repository.record_feedback_note(
            delivery_id,
            comment,
            reaction=reaction,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from None
    typer.echo(json.dumps({"feedback_id": feedback_id, "delivery_id": delivery_id}))


@app.command("sensors-collect")
def sensors_collect_command(
    source: Annotated[list[str] | None, typer.Option("--source")] = None,
    limit: Annotated[int, typer.Option("--limit", min=1, max=100)] = 10,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Run due sensors with cadence, error isolation and spending caps."""
    settings, repository = bootstrap()
    report = collect_due_sources(
        settings,
        repository,
        source_ids=source,
        limit_per_source=limit,
        dry_run=dry_run,
        force=force,
    )
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))


@app.command("sensors-status")
def sensors_status_command() -> None:
    _, repository = bootstrap()
    typer.echo(json.dumps(source_health_report(repository), ensure_ascii=False, indent=2))


@app.command("source-toggle")
def source_toggle_command(source_id: str, enabled: bool) -> None:
    settings, repository = bootstrap()
    if source_id not in {x.id for x in runtime_source_registry(settings).sources}:
        raise typer.BadParameter("Unknown source_id")
    repository.set_runtime_setting(f"source_enabled:{source_id}", str(enabled).lower())
    if not enabled:
        from app.collectors.browser_queue import BrowserTaskQueue

        BrowserTaskQueue(repository.database_path).cancel_pending(source_id)
    typer.echo(json.dumps({"source_id": source_id, "enabled": enabled}))


@app.command("backup-db")
def backup_db_command() -> None:
    from app.ops.snapshot import backup_database

    settings, _ = runtime()
    path = backup_database(settings.database_path, settings.data_dir / "backups")
    typer.echo(json.dumps({"backup": str(path) if path else None}))


@app.command("serve")
def serve_command() -> None:
    """Run Telegram, on-demand details and the configured schedule together."""
    from app.ops.service import serve

    settings, repository = bootstrap()
    asyncio.run(serve(settings, repository))


@app.command("browser-serve")
def browser_serve_command(
    extension_id: Annotated[str, typer.Option("--extension-id")],
    port: Annotated[int, typer.Option("--port", min=1024, max=65535)] = 8769,
) -> None:
    """Serve a local, explicitly paired read-only browser bridge."""
    from app.collectors.browser_bridge import BrowserBridgeServer
    from app.collectors.browser_queue import BrowserTaskQueue

    _, repository = bootstrap()
    queue = BrowserTaskQueue(repository.database_path)
    queue.initialize()
    server = BrowserBridgeServer(queue, extension_id=extension_id, port=port)
    typer.echo(f"Local bridge: {server.base_url}")
    typer.echo(f"Pairing token (local extension only): {server.pairing_token}")
    server.serve_forever()


@app.command("browser-request")
def browser_request_command(
    source_id: str = "x_for_you",
    limit: Annotated[int, typer.Option("--limit", min=1, max=80)] = 20,
) -> None:
    """Queue a read-only snapshot; never opens a tab or signs into an account."""
    from app.collectors.browser_queue import BrowserTaskQueue

    settings, repository = bootstrap()
    source = next((s for s in runtime_source_registry(settings).sources if s.id == source_id), None)
    if source is None or source.adapter_id != "browser_x":
        raise typer.BadParameter("Expected a configured browser_x source")
    enabled = repository.get_runtime_setting(f"source_enabled:{source_id}")
    if enabled == "false" or (enabled is None and not source.enabled):
        raise typer.BadParameter("Enable this source first with source-toggle")
    queue = BrowserTaskQueue(repository.database_path)
    queue.initialize()
    task = queue.enqueue(source_id, task_key=f"manual:{uuid.uuid4().hex}", max_items=limit)
    typer.echo(json.dumps(task, ensure_ascii=False, default=str))


@app.command("preferences-report")
def preferences_report_command(
    days: Annotated[int, typer.Option("--days", min=1, max=365)] = 30,
) -> None:
    """Explain recent feedback; proposes changes but never rewrites your preferences."""
    from app.radar.preference_patches import build_preference_report, render_preference_report

    _, repository = bootstrap()
    typer.echo(render_preference_report(build_preference_report(repository, days=days)))


@app.command("video-summary")
def video_summary_command(
    video: str,
    mode: Annotated[str, typer.Option("--mode")] = "short",
    cost_limit: Annotated[float, typer.Option("--cost-limit", min=0.001, max=0.25)] = 0.05,
    language: Annotated[str | None, typer.Option("--language")] = None,
    show_evidence: Annotated[bool, typer.Option("--show-evidence")] = False,
) -> None:
    """Fetch/cache captions only on this explicit request and build a grounded summary."""
    from app.video.summary import build_video_summary
    from app.video.transcripts import youtube_video_id

    if mode not in {"short", "long"}:
        raise typer.BadParameter("mode must be short or long")
    settings, repository = bootstrap()
    video_id = youtube_video_id(video)
    with repository.connection() as db:
        item = db.execute(
            "SELECT i.original_text,s.name FROM source_items i JOIN sources s ON s.id=i.source_id "
            "WHERE s.kind='youtube' AND i.external_id=? LIMIT 1",
            (video_id,),
        ).fetchone()
    metadata = {"title": item[0].splitlines()[0][:200], "channel": item[1]} if item else {}
    result = build_video_summary(
        settings,
        repository,
        video,
        mode=mode,
        cost_limit_usd=cost_limit,
        source_metadata=metadata,
        language_code=language,
    )
    if not show_evidence:
        result = {key: value for key, value in result.items() if key not in {"evidence", "draft"}}
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("videos-deliver")
def videos_deliver_command(
    limit: Annotated[int, typer.Option("--limit", min=1, max=3)] = 2,
) -> None:
    """Send fresh metadata-only video recommendations to the configured owner."""
    from app.telegram_bot.bot import build_bot, configured_owner_id
    from app.telegram_bot.sensory import send_video_recommendations

    settings, repository = bootstrap()
    owner = configured_owner_id(settings, repository)
    if owner is None:
        raise typer.BadParameter("Send /start to the bot first")

    async def deliver():
        bot = build_bot(settings)
        try:
            return await send_video_recommendations(
                bot, settings, repository, chat_id=owner, limit=limit
            )
        finally:
            await bot.session.close()

    typer.echo(json.dumps(asyncio.run(deliver()), ensure_ascii=False))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
