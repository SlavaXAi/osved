from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from datetime import time as clock_time
from typing import Any, Literal
from zoneinfo import ZoneInfo

from app.collectors.ingest import redact_text
from app.collectors.orchestrator import collect_due_sources
from app.config import DeliveryFormat, Settings, load_search_agent_registry
from app.db.card_predicates import finished_card_sql
from app.db.repository import Repository
from app.ops.budget import SearchBudgetTransport
from app.personalization.daily import load_context, record_selection
from app.pipeline.cards import generate_cards
from app.pipeline.dedupe import deduplicate_events
from app.pipeline.ranking import rank_event_candidates
from app.pipeline.selection import PROMOTIONAL_POST, prepare_candidates
from app.radar.search import run_search_radar
from app.research.facts import build_event_fact_sheet, select_research_documents
from app.research.web import ResearchEventService
from app.telegram_bot.bot import build_bot, configured_owner_id, send_cards_by_ids
from app.telegram_bot.sensory import send_video_recommendations

MOSCOW = ZoneInfo("Europe/Moscow")
SCHEDULE_ENABLED_SETTING = "schedule_enabled"
SCHEDULE_DRY_RUN_SETTING = "schedule_dry_run"
JobType = Literal["radar", "research", "delivery"]
PRODUCTION_MAX_AGE_DAYS = 5
RESEARCH_FAILURE_PREFIX = "scheduled_research_failure:"
RESEARCH_RETRY_HOURS = 24
RESEARCH_MAX_FAILURES = 2

# Do not use card creation, discovery, or latest mention time to resurrect an
# old event. Missing/malformed dates fail closed in SQLite's julianday().
FRESH_EVENT_SQL = """
    julianday(COALESCE(NULLIF(events.event_at, ''), source_items.published_at))
    BETWEEN julianday(?) AND julianday(?)
"""


def _fresh_window(now: datetime) -> tuple[str, str]:
    current = now.astimezone(UTC)
    return (
        (current - timedelta(days=PRODUCTION_MAX_AGE_DAYS)).isoformat(),
        (current + timedelta(minutes=10)).isoformat(),
    )


def is_web_discovery(event: dict[str, Any]) -> bool:
    # Older search items were stored as discovery_mode=feed; cluster provenance
    # retains the actual web_agent identity. Do not mistake X search for this.
    sources = [str(event.get("source_id") or ""), *(event.get("source_ids") or [])]
    return event.get("discovery_mode") == "web_search" or any(
        str(source).startswith("web_agent_") for source in sources
    )


def fresh_research_events(
    repository: Repository, *, now: datetime, limit: int = 50
) -> list[dict[str, Any]]:
    """Filter before LIMIT so a stale high-score backlog cannot starve news."""
    with repository.connection() as connection:
        rows = connection.execute(
            f"""SELECT events.*, source_items.source_url, source_items.outbound_urls_json,
                source_items.published_at AS source_published_at,
                source_items.discovery_mode, source_items.original_text, sources.id AS source_id,
                sources.kind AS source_kind, sources.admission_mode, sources.role AS source_role
            FROM events
            JOIN source_items ON source_items.id = events.primary_source_item_id
            JOIN sources ON sources.id = source_items.source_id
            WHERE events.status = 'candidate'
              AND NOT EXISTS(SELECT 1 FROM menu_preparation_events m
                WHERE m.event_id=events.id AND m.status IN('running','uncertain'))
              AND COALESCE(events.event_type, '') != 'system_test'
              AND {FRESH_EVENT_SQL}
              AND (sources.kind!='youtube' OR
                  CASE WHEN json_valid(source_items.media_metadata_json) THEN
                    json_extract(source_items.media_metadata_json,'$.is_short')=0 END
                  OR EXISTS (SELECT 1 FROM youtube_video_types type
                    WHERE type.video_id=source_items.external_id AND type.is_short=0))
              AND NOT EXISTS (SELECT 1 FROM cards
                  WHERE cards.event_id = events.id AND {finished_card_sql()})
              AND NOT EXISTS (SELECT 1 FROM event_fact_sheets facts
                  WHERE facts.event_id=events.id AND facts.validation_status='invalid'
                    AND julianday(facts.created_at)>julianday(?, '-24 hours'))
              AND NOT EXISTS (SELECT 1 FROM runtime_settings retry
                  WHERE retry.key = '{RESEARCH_FAILURE_PREFIX}' || events.id
                    AND CASE WHEN json_valid(retry.value) THEN
                        CASE WHEN json_type(retry.value, '$.attempts') = 'integer'
                          AND json_extract(retry.value, '$.attempts') = 1
                          AND json_type(retry.value, '$.retry_after') = 'text'
                          AND (json_extract(retry.value, '$.retry_after') LIKE '%Z'
                            OR (substr(json_extract(retry.value, '$.retry_after'), -6, 1)
                                IN ('+','-')
                              AND substr(json_extract(retry.value, '$.retry_after'), -3, 1) = ':'))
                          AND julianday(json_extract(retry.value, '$.retry_after')) IS NOT NULL
                        THEN julianday(json_extract(retry.value, '$.retry_after')) > julianday(?)
                        ELSE 1 END
                        ELSE 1 END)
            ORDER BY events.score DESC, events.latest_source_at DESC LIMIT ?""",
            (*_fresh_window(now), now.isoformat(), now.isoformat(), limit),
        ).fetchall()
    return repository.with_event_signal_metadata(
        [dict(row) for row in rows if not PROMOTIONAL_POST.search(row["original_text"] or "")]
    )


def record_research_failure(
    repository: Repository, event_id: int, error: str, *, now: datetime
) -> dict[str, Any]:
    """One delayed retry for an uncaught web/no-docs failure; never spin in a slot."""
    key = f"{RESEARCH_FAILURE_PREFIX}{event_id}"
    with repository.connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        previous = connection.execute(
            "SELECT value FROM runtime_settings WHERE key=?", (key,)
        ).fetchone()
        try:
            attempts = int(json.loads(previous[0]).get("attempts", 0)) if previous else 0
        except (ValueError, TypeError, AttributeError):
            attempts = 0
        attempts = min(RESEARCH_MAX_FAILURES, max(0, attempts) + 1)
        exhausted = attempts >= RESEARCH_MAX_FAILURES
        failure = {
            "attempts": attempts,
            "last_error": redact_text(error)[:500],
            "last_failed_at": now.isoformat(),
            "retry_after": None
            if exhausted
            else (now + timedelta(hours=RESEARCH_RETRY_HOURS)).isoformat(),
            "exhausted": exhausted,
        }
        connection.execute(
            """INSERT INTO runtime_settings(key,value,updated_at) VALUES(?,?,?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at""",
            (key, json.dumps(failure, ensure_ascii=False), now.isoformat()),
        )
        if exhausted:
            connection.execute(
                """UPDATE events SET status='research_insufficient',decision_reason=?,updated_at=?
                WHERE id=? AND status='candidate'""",
                (
                    f"Scheduled research stopped after {attempts} failures: "
                    + failure["last_error"],
                    now.isoformat(),
                    event_id,
                ),
            )
    return failure


def fresh_delivery_card_ids(
    repository: Repository,
    *,
    owner_id: int,
    now: datetime,
    limit: int = 2,
    settings: Settings | None = None,
) -> list[int]:
    with repository.connection() as connection:
        rows = connection.execute(
            f"""SELECT events.*, cards.id AS card_id, cards.created_at AS card_created_at,
                julianday(cards.created_at) AS card_created_jd,
                cards.source_urls_json,source_items.original_text,cards.rendered_html FROM cards
            JOIN events ON events.id = cards.event_id
            JOIN source_items ON source_items.id = events.primary_source_item_id
            JOIN sources ON sources.id = source_items.source_id
            WHERE {finished_card_sql()}
              AND events.status NOT IN ('merged', 'rejected')
              AND COALESCE(events.event_type, '') != 'system_test'
              AND CASE WHEN events.event_type='concept_brief'
                  THEN julianday(cards.created_at) BETWEEN julianday(?) AND julianday(?)
                  ELSE ({FRESH_EVENT_SQL}) END
              AND (sources.kind!='youtube' OR
                  CASE WHEN json_valid(source_items.media_metadata_json) THEN
                    json_extract(source_items.media_metadata_json,'$.is_short')=0 END
                  OR EXISTS (SELECT 1 FROM youtube_video_types type
                    WHERE type.video_id=source_items.external_id AND type.is_short=0))
              AND NOT EXISTS (SELECT 1 FROM cards newer
                  WHERE newer.event_id = cards.event_id AND {finished_card_sql("newer")}
                  AND newer.id > cards.id)
              AND NOT EXISTS (SELECT 1 FROM deliveries
                  JOIN cards delivered ON delivered.id = deliveries.card_id
                  WHERE delivered.event_id = events.id AND deliveries.chat_id = ?
                    AND (deliveries.delivery_status != 'failed'
                        OR EXISTS (SELECT 1 FROM feedback_events feedback
                            WHERE feedback.delivery_id = deliveries.id
                              AND feedback.reaction IS NOT NULL)))
            ORDER BY CASE WHEN events.event_type='concept_brief' THEN 0 ELSE 1 END,
              CASE WHEN events.event_type='concept_brief' THEN julianday(cards.created_at) END,
              events.score DESC, events.latest_source_at DESC, cards.id DESC LIMIT ?""",
            (*_fresh_window(now), *_fresh_window(now), owner_id, 150),
        ).fetchall()
    # Reconcile confirmed emergency/manual sends without inventing delivery IDs
    # or overwriting the live database with an old local copy.
    external_receipts = json.loads(
        repository.get_runtime_setting(f"news_external_receipts:{owner_id}") or "[]"
    )
    external_urls = {
        url.rstrip("/") for receipt in external_receipts for url in receipt["source_urls"]
    }
    unseen = [
        dict(row)
        for row in rows
        if not external_urls.intersection(
            url.rstrip("/") for url in json.loads(row["source_urls_json"] or "[]")
        )
    ]
    # Diversify the actual delivered batch, not only upstream candidates. A
    # high-scoring company must not monopolize a backlog built on different days.
    pool = [row for row in unseen if row["event_type"] != "concept_brief"]
    context = load_context(settings, repository, owner=owner_id, now=now)
    if context:
        with repository.connection() as connection:
            seen = {
                r[0]
                for r in connection.execute(
                    "SELECT e.fingerprint FROM deliveries d JOIN cards c ON c.id=d.card_id "
                    "JOIN events e ON e.id=c.event_id WHERE d.chat_id=? "
                    "AND d.delivery_status!='failed' AND COALESCE(e.fingerprint,'')!=''",
                    (owner_id,),
                )
            }
        pool = [r for r in pool if not r.get("fingerprint") or r["fingerprint"] not in seen]
    ranked = rank_event_candidates(
        pool,
        recent_events=repository.recent_valid_card_events(days=2),
        limit=max(0, limit),
        now=now,
        personalization=context,
    )
    record_selection(
        repository,
        context,
        owner=owner_id,
        stage="delivery_candidates",
        candidates=pool,
        selected=ranked,
    )
    # One concept can join the normal batch without competing against the
    # event/news score scale. Keep a daily cadence when a backlog accumulated.
    with repository.connection() as connection:
        day_start = now.astimezone(MOSCOW).replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        concept_count = connection.execute(
            "SELECT COUNT(*) FROM deliveries d JOIN cards c ON c.id=d.card_id "
            "JOIN events e ON e.id=c.event_id WHERE d.chat_id=? "
            "AND e.event_type='concept_brief' AND d.delivery_status!='failed' "
            "AND julianday(COALESCE(d.delivered_at,d.created_at))>=julianday(?) "
            "AND julianday(COALESCE(d.delivered_at,d.created_at))<julianday(?)",
            (owner_id, day_start.isoformat(), day_end.isoformat()),
        ).fetchone()[0]
    concepts = sorted(
        [row for row in unseen if row["event_type"] == "concept_brief"],
        key=lambda row: (row["card_created_jd"], row["card_id"]),
    )
    regular = [int(row["card_id"]) for row in ranked if row["event_type"] != "concept_brief"]
    if concepts and concept_count < 2 and limit > 0:
        return [int(concepts[0]["card_id"]), *regular][:limit]
    return regular[:limit]


JOB_TIMES: dict[JobType, tuple[clock_time, ...]] = {
    "radar": tuple(clock_time(hour, 0) for hour in (10, 12, 14, 16, 18, 20)),
    "research": (clock_time(11, 0), clock_time(15, 0), clock_time(19, 0)),
    "delivery": (clock_time(12, 30), clock_time(16, 30), clock_time(20, 0)),
}


@dataclass(frozen=True)
class DueSlot:
    job_type: JobType
    slot_at: datetime


JobExecutor = Callable[[JobType], dict[str, Any]]


def setting_bool(repository: Repository, key: str, default: bool) -> bool:
    raw = repository.get_runtime_setting(key)
    if raw is None:
        return default
    return raw.casefold() in {"1", "true", "yes", "on"}


def claim_runtime_window(repository: Repository, key: str, *, now: datetime) -> bool:
    with repository.connection() as connection:
        cursor = connection.execute(
            "INSERT OR IGNORE INTO runtime_settings(key,value,updated_at) VALUES(?,'running',?)",
            (key, now.isoformat()),
        )
    return cursor.rowcount == 1


def configure_schedule(repository: Repository, *, enabled: bool, dry_run: bool) -> None:
    repository.set_runtime_setting(SCHEDULE_ENABLED_SETTING, str(enabled).lower())
    repository.set_runtime_setting(SCHEDULE_DRY_RUN_SETTING, str(dry_run).lower())


def due_slots(
    now: datetime, *, grace_minutes: int = 45, settings: Settings | None = None
) -> list[DueSlot]:
    timezone = ZoneInfo(settings.schedule_timezone) if settings is not None else MOSCOW
    times_by_job = (
        {
            "radar": tuple(clock_time.fromisoformat(value) for value in settings.radar_times),
            "research": tuple(clock_time.fromisoformat(value) for value in settings.research_times),
            "delivery": tuple(clock_time.fromisoformat(value) for value in settings.delivery_times),
        }
        if settings is not None
        else JOB_TIMES
    )
    local_now = now.astimezone(timezone)
    due: list[DueSlot] = []
    for job_type, times in times_by_job.items():
        for planned_time in times:
            slot_at = datetime.combine(local_now.date(), planned_time, tzinfo=timezone)
            delay = local_now - slot_at
            if timedelta(0) <= delay <= timedelta(minutes=grace_minutes):
                due.append(DueSlot(job_type=job_type, slot_at=slot_at))
    return sorted(due, key=lambda slot: (slot.slot_at, slot.job_type))


class ScheduledJobExecutor:
    def __init__(
        self, settings: Settings, repository: Repository, *, now: datetime | None = None
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.now = now

    def current_time(self) -> datetime:
        return self.now or datetime.now(UTC)

    def __call__(self, job_type: JobType) -> dict[str, Any]:
        if job_type == "radar":
            return self.radar()
        if job_type == "research":
            return self.research()
        return self.delivery()

    def radar(
        self, *, provider: str = "auto", limit_per_source: int | None = None
    ) -> dict[str, Any]:
        limit_per_source = (
            self.settings.radar_collect_limit if limit_per_source is None else limit_per_source
        )
        collections = collect_due_sources(
            self.settings, self.repository, limit_per_source=limit_per_source, provider=provider
        )
        total_cost = collections["cost_usd"]
        total_errors = collections["errors"]
        search: dict[str, Any] = {"status": "disabled"}
        # Broad web discovery is bounded independently of individual source polls.
        if self.settings.scheduled_search_enabled and self.settings.openrouter_api_key:
            local_now = self.current_time().astimezone(ZoneInfo(self.settings.schedule_timezone))
            key = f"scheduled_search:{local_now.date()}:{local_now.hour // 4}"
            ceiling = self.settings.scheduled_search_max_cost_usd
            if ceiling <= 0 or (
                self.repository.current_month_cost() + ceiling > self.settings.budget_hard_stop_usd
            ):
                search = {"status": "budget_blocked"}
            elif self.repository.get_runtime_setting(key) is None:
                lenses = load_search_agent_registry(self.settings.search_agents_file).enabled_lenses
                count = min(len(lenses), self.settings.scheduled_search_max_agents)
                offset = (local_now.toordinal() * 3 + local_now.hour // 4) * count
                chosen = [lenses[(offset + i) % len(lenses)].id for i in range(count)]
                # Spend one existing search slot on a rotating current priority;
                # keep the other slots broad. No additional calls or topic quota.
                core = [
                    key
                    for key in self.settings.scheduled_priority_lenses
                    if any(lens.id == key for lens in lenses)
                ]
                if core and chosen:
                    priority = core[(local_now.toordinal() + local_now.hour // 4) % len(core)]
                    chosen = [priority, *[key for key in chosen if key != priority]][:count]
                    for lens in lenses[offset % len(lenses) :] + lenses[: offset % len(lenses)]:
                        if len(chosen) >= count:
                            break
                        if lens.id not in chosen:
                            chosen.append(lens.id)
                if chosen and claim_runtime_window(self.repository, key, now=local_now):
                    guard = SearchBudgetTransport(ceiling)
                    ledger_before = self.repository.current_month_cost()
                    try:
                        stats = run_search_radar(
                            self.settings,
                            self.repository,
                            lens_ids=chosen,
                            preview_limit=6,
                            transport=guard,
                        )
                        search = stats.as_dict()
                        total_cost += stats.cost_usd
                        total_errors += len(stats.errors)
                        self.repository.set_runtime_setting(
                            "latest_scheduled_batch", stats.batch_id or ""
                        )
                        self.repository.set_runtime_setting(key, "completed")
                    except Exception as exc:
                        search = {"status": "failed", "error": redact_text(str(exc))[:500]}
                        self.repository.set_runtime_setting(key, "failed")
                        total_errors += 1
                    finally:
                        budget = guard.snapshot()
                        search["budget"] = budget
                        # Parsing/API failures must not make paid attempts vanish
                        # from the local accounting. Unknown charges are explicitly
                        # conservative reservations, not reported invoice amounts.
                        if budget["run_id"]:
                            with self.repository.connection() as connection:
                                ledger_delta = float(
                                    connection.execute(
                                        """SELECT COALESCE(SUM(cost_usd),0) FROM usage_ledger
                                    WHERE CASE WHEN json_valid(metadata_json) THEN
                                        json_extract(metadata_json,'$.run_id') END=?""",
                                        (budget["run_id"],),
                                    ).fetchone()[0]
                                )
                        else:
                            ledger_delta = self.repository.current_month_cost() - ledger_before
                        unaccounted = max(0.0, budget["charged_or_reserved_usd"] - ledger_delta)
                        if unaccounted > 1e-8:
                            self.repository.record_usage(
                                "openrouter",
                                "scheduled_search_accounting_reserve",
                                cost_usd=unaccounted,
                                request_count=0,
                                metadata=budget,
                            )
                        total_cost = collections["cost_usd"] + max(
                            ledger_delta, budget["charged_or_reserved_usd"]
                        )
                else:
                    search = {"status": "already_attempted" if chosen else "no_enabled_lenses"}
            else:
                search = {"status": "already_attempted"}

        prepared = prepare_candidates(self.settings, self.repository)
        deduped = deduplicate_events(self.repository)
        return {
            "collections": collections,
            "search": search,
            "preparation": prepared.__dict__,
            "dedupe": deduped.__dict__,
            "cost_usd": total_cost,
            "errors": total_errors,
        }

    def research(self) -> dict[str, Any]:
        from app.radar.concepts import prepare_concept_brief

        concepts = prepare_concept_brief(self.settings, self.repository, now=self.current_time())
        news = self.research_news()
        return {
            **news,
            "concepts": concepts,
            "cards_created": news.get("cards_created", 0) + concepts["cards_created"],
            "cost_usd": news.get("cost_usd", 0) + concepts["cost_usd"],
            "errors": news.get("errors", 0) + concepts["errors"],
        }

    def research_news(self) -> dict[str, Any]:
        target = self.settings.scheduled_news_per_batch
        owner_id = configured_owner_id(self.settings, self.repository)
        ready = (
            len(
                fresh_delivery_card_ids(
                    self.repository,
                    settings=self.settings,
                    owner_id=owner_id,
                    now=self.current_time(),
                    limit=self.settings.scheduled_news_buffer_target,
                )
            )
            if owner_id is not None
            else 0
        )
        if ready >= self.settings.scheduled_news_buffer_target:
            return {
                "status": "backlog_ready",
                "events_seen": 0,
                "cards_created": 0,
                "ready_cards": ready,
                "cost_usd": 0.0,
                "errors": 0,
            }
        target = min(target, self.settings.scheduled_news_buffer_target - ready)
        candidates = fresh_research_events(self.repository, now=self.current_time(), limit=150)
        recent = self.repository.recent_valid_card_events(days=2)
        context = load_context(
            self.settings, self.repository, owner=owner_id, now=self.current_time()
        )
        pools = [
            rank_event_candidates(
                [event for event in candidates if is_web_discovery(event) == from_search],
                recent_events=recent,
                limit=self.settings.scheduled_research_max_attempts,
                now=self.current_time(),
                personalization=context,
            )
            for from_search in (True, False)
        ]
        # Search confidence and fixed-source scores are not calibrated to the
        # same scale. Give both discovery paths research opportunities; this is
        # not a publication quota and never rescues below-floor candidates.
        events = [
            event
            for i in range(max(map(len, pools)))
            for pool in pools
            for event in pool[i : i + 1]
        ][: self.settings.scheduled_research_max_attempts]
        record_selection(
            self.repository,
            context,
            owner=owner_id,
            stage="research_candidates",
            candidates=candidates,
            selected=events,
        )
        if not events:
            return {"events_seen": 0, "cards_created": 0, "cost_usd": 0.0, "errors": 0}
        outcomes: list[dict[str, Any]] = []
        started = time.monotonic()
        cost = 0.0
        created = 0
        for event in events:
            if (
                created >= target
                or time.monotonic() - started >= 360
                or cost >= self.settings.scheduled_research_max_cost_usd
                or self.repository.current_month_cost() >= self.settings.budget_hard_stop_usd
            ):
                break
            outcome = self.research_event(event)
            outcomes.append(outcome)
            cost += float(outcome.get("cost_usd") or 0)
            created += int(outcome.get("cards_created") or 0)
        # A rejected story no longer consumes the entire publication slot. The
        # same event is never retried within this loop; evidence guards stay on.
        return {
            **(outcomes[0] if len(outcomes) == 1 else {}),
            "status": (
                outcomes[0].get("status", "completed")
                if len(outcomes) == 1
                else "completed"
                if created
                else "no_publishable_cards"
            ),
            "events_seen": len(outcomes),
            "cards_created": created,
            "ready_cards": ready + created,
            "target_cards": target,
            "cost_usd": cost,
            "errors": sum(int(row.get("errors") or 0) for row in outcomes),
            "outcomes": outcomes,
        }

    def research_event(self, event: dict[str, Any]) -> dict[str, Any]:
        event_id = int(event["id"])
        saved_fact_sheet = self.repository.latest_valid_fact_sheet(event_id)
        cache_valid = False
        if saved_fact_sheet:
            try:
                expiry = datetime.fromisoformat(
                    saved_fact_sheet["expires_at"].replace("Z", "+00:00")
                )
                cache_valid = expiry.tzinfo is not None and expiry > self.current_time()
            except (ValueError, TypeError, KeyError):
                pass
        if cache_valid:
            web_result = {"status": "cached", "cost_usd": 0.0}
            fact_sheet = {"status": "valid", "cached": True, "cost_usd": 0.0}
        else:
            primary_url = str(event["source_url"]) if event["source_role"] == "primary" else None
            with self.repository.connection() as connection:
                ledger_start = connection.execute(
                    "SELECT COALESCE(MAX(id),0) FROM usage_ledger"
                ).fetchone()[0]
            try:
                prior_sheet = self.repository.latest_valid_fact_sheet(
                    event_id, require_current_policy=False
                )
                if prior_sheet and select_research_documents(
                    self.repository.event_research_documents(event_id),
                    event_headline=str(event["headline"]),
                ):
                    # Changed policy invalidates derived facts, not the existing
                    # full-text source documents. Re-extract locally held evidence.
                    web_result = {"status": "cached_documents_policy_refresh", "cost_usd": 0.0}
                else:
                    web_stats = ResearchEventService(self.settings, self.repository).research(
                        event_id,
                        query=str(event["headline"]),
                        primary_url=primary_url,
                        freshness="last_week",
                        max_pages=2,
                    )
                    web_result = web_stats.__dict__
                fact_sheet = build_event_fact_sheet(self.settings, self.repository, event_id)
            except Exception as exc:
                failure = record_research_failure(
                    self.repository,
                    event_id,
                    f"{type(exc).__name__}: {exc}",
                    now=self.current_time(),
                )
                with self.repository.connection() as connection:
                    cost = connection.execute(
                        """SELECT COALESCE(SUM(cost_usd),0) FROM usage_ledger
                        WHERE id>? AND CASE WHEN json_valid(metadata_json) THEN
                            json_extract(metadata_json,'$.event_id') END=?
                          AND (operation LIKE 'web.%' OR operation LIKE 'research.fact_sheet%')""",
                        (ledger_start, event_id),
                    ).fetchone()[0]
                return {
                    "events_seen": 1,
                    "event_id": event_id,
                    "cards_created": 0,
                    "status": (
                        "research_insufficient" if failure["exhausted"] else "retry_scheduled"
                    ),
                    "retry": failure,
                    "cost_usd": float(cost),
                    "errors": 1,
                }
        # Recovery clears only this event's transient failure history. Ordinary
        # invalid facts/generation retain their existing terminal status handling.
        with self.repository.connection() as connection:
            connection.execute(
                "DELETE FROM runtime_settings WHERE key=?",
                (f"{RESEARCH_FAILURE_PREFIX}{event_id}",),
            )
        generated = generate_cards(self.settings, self.repository, limit=1, event_id=event_id)
        return {
            "events_seen": 1,
            "event_id": event_id,
            "research": web_result,
            "fact_sheet": fact_sheet,
            "generation": generated.__dict__,
            "cards_created": generated.cards_created,
            "cost_usd": float(web_result["cost_usd"])
            + float(fact_sheet.get("cost_usd") or 0)
            + generated.cost_usd,
            "errors": generated.errors + (0 if fact_sheet.get("status") == "valid" else 1),
        }

    def delivery(self, *, limit: int | None = None) -> dict[str, Any]:
        limit = self.settings.scheduled_news_delivery_batch_size if limit is None else limit
        owner_id = configured_owner_id(self.settings, self.repository)
        if owner_id is None:
            return {"status": "owner_not_configured", "sent": 0, "cost_usd": 0.0, "errors": 0}
        from app.telegram_bot.news_queue import queue_enabled, send_next_news

        if queue_enabled(self.repository):
            result = asyncio.run(send_next_news(self.settings, self.repository, limit=limit))
            videos = self.daily_video_recommendation(owner_id=owner_id)
            return {
                **result,
                "news_sent": result.get("sent", 0),
                "video_sent": videos.get("sent", 0),
                "videos": videos,
                "sent": result.get("sent", 0) + videos.get("sent", 0),
            }
        card_ids = fresh_delivery_card_ids(
            self.repository,
            settings=self.settings,
            owner_id=owner_id,
            now=self.current_time(),
            limit=max(0, min(limit, 50)),
        )
        if card_ids:
            result = asyncio.run(
                send_cards_by_ids(
                    self.settings,
                    self.repository,
                    card_ids,
                    delivery_format=DeliveryFormat.REGULAR,
                )
            )
        else:
            result = {"status": "no_fresh_cards", "sent": 0, "errors": 0}
        videos = self.daily_video_recommendation(owner_id=owner_id)
        return {
            **result,
            "news_sent": result["sent"],
            "video_sent": videos.get("sent", 0),
            "sent": result["sent"] + videos.get("sent", 0),
            "videos": videos,
            "cost_usd": videos.get("cost_usd", 0.0),
            "errors": result["errors"] + videos.get("errors", 0),
        }

    def daily_video_recommendation(self, *, owner_id: int) -> dict[str, Any]:
        """Deliver unseen channel uploads; legacy method name, no daily quota.

        Per-video durable claims live in send_video_recommendations. A day's
        first delivery must not block uploads discovered later that same day.
        Subtitles and summaries remain strictly behind the user's buttons.
        """
        local_now = self.current_time().astimezone(MOSCOW)
        if local_now.time() < JOB_TIMES["delivery"][0]:
            return {"status": "before_delivery_window", "sent": 0}
        if not setting_bool(
            self.repository,
            "scheduled_video_recommendations_enabled",
            getattr(self.settings, "scheduled_video_recommendations_enabled", True),
        ):
            return {"status": "disabled", "sent": 0}
        with self.repository.connection() as connection:
            exists = connection.execute(
                """SELECT 1 FROM source_items i JOIN sources s ON s.id=i.source_id
                WHERE s.kind='youtube' AND COALESCE(
                    (SELECT CASE rs.value WHEN 'true' THEN 1 WHEN 'false' THEN 0 END
                     FROM runtime_settings rs WHERE rs.key='source_enabled:'||s.id),s.enabled)=1
                  AND NOT EXISTS (SELECT 1 FROM youtube_channels y
                    WHERE y.source_id=s.id AND y.removed_at IS NOT NULL)
                  AND julianday(i.published_at) BETWEEN julianday(?) AND julianday(?)
                  AND NOT EXISTS (SELECT 1 FROM video_recommendations v
                    WHERE v.source_item_id=i.id AND v.chat_id=?) LIMIT 1""",
                (*_fresh_window(self.current_time()), owner_id),
            ).fetchone()
            if not exists:
                return {"status": "no_fresh_videos", "sent": 0}

        async def send() -> dict[str, Any]:
            bot = build_bot(self.settings)
            try:
                return await send_video_recommendations(
                    bot, self.settings, self.repository, chat_id=owner_id, limit=None
                )
            finally:
                await bot.session.close()

        ledger_before = self.repository.current_month_cost()
        try:
            result = asyncio.run(send())
            return {
                **result,
                "cost_usd": max(0, self.repository.current_month_cost() - ledger_before),
            }
        except Exception as exc:
            return {
                "status": "failed",
                "sent": 0,
                "errors": 1,
                "error": redact_text(str(exc))[:500],
                "cost_usd": max(0, self.repository.current_month_cost() - ledger_before),
            }


def run_schedule_tick(
    settings: Settings,
    repository: Repository,
    *,
    now: datetime | None = None,
    force_dry_run: bool = False,
    executor: JobExecutor | None = None,
) -> dict[str, Any]:
    enabled = setting_bool(repository, SCHEDULE_ENABLED_SETTING, False)
    configured_dry_run = setting_bool(repository, SCHEDULE_DRY_RUN_SETTING, True)
    if not enabled and not force_dry_run:
        return {"status": "disabled", "slots": []}

    dry_run = force_dry_run or configured_dry_run
    mode = "dry_run" if dry_run else "live"
    current = now or datetime.now(MOSCOW)
    planned = due_slots(current, settings=settings)
    executed: list[dict[str, Any]] = []
    job_executor = executor or ScheduledJobExecutor(settings, repository, now=current)

    for slot in planned:
        slot_at = slot.slot_at.isoformat()
        slot_id = repository.claim_schedule_slot(slot.job_type, slot_at, mode)
        if slot_id is None:
            executed.append(
                {"job_type": slot.job_type, "slot_at": slot_at, "status": "already_claimed"}
            )
            continue
        if dry_run:
            metrics = {"external_requests": 0, "planned_job": slot.job_type}
            repository.update_schedule_slot(slot_id, status="skipped", metrics=metrics)
            executed.append({"job_type": slot.job_type, "slot_at": slot_at, "status": "dry_run"})
            continue

        run_id = repository.start_run(
            f"scheduled_{slot.job_type}", {"slot_at": slot_at, "mode": mode}
        )
        repository.update_schedule_slot(slot_id, status="running", run_id=run_id)
        try:
            metrics = job_executor(slot.job_type)
            errors = int(metrics.get("errors") or 0)
            cost = float(metrics.get("cost_usd") or 0)
            search = metrics.get("search") or {}
            failed = errors > 0 and (
                (slot.job_type == "research" and not metrics.get("cards_created"))
                or (
                    slot.job_type == "radar"
                    and search.get("lenses_attempted")
                    and not search.get("lenses_succeeded")
                )
            )
            status = "failed" if failed else "completed"
            repository.finish_run(
                run_id,
                status=status,
                processed_count=int(metrics.get("cards_created") or metrics.get("sent") or 0),
                error_count=errors,
                cost_usd=cost,
            )
            repository.update_schedule_slot(
                slot_id,
                status=status,
                run_id=run_id,
                metrics=metrics,
            )
            executed.append({"job_type": slot.job_type, "slot_at": slot_at, "status": status})
        except Exception as exc:
            error = redact_text(str(exc))[:500]
            repository.finish_run(run_id, status="failed", error_count=1, error=error)
            repository.update_schedule_slot(
                slot_id,
                status="failed",
                run_id=run_id,
                error=error,
            )
            executed.append(
                {
                    "job_type": slot.job_type,
                    "slot_at": slot_at,
                    "status": "failed",
                    "error": error,
                }
            )
    return {"status": mode, "now": current.astimezone(MOSCOW).isoformat(), "slots": executed}


def run_schedule_loop(
    settings: Settings, repository: Repository, interval_seconds: int = 60
) -> None:
    while True:
        run_schedule_tick(settings, repository)
        time.sleep(interval_seconds)
