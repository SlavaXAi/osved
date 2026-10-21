"""Explicit, bounded preparation over fresh signals already in the shared radar."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from datetime import UTC, datetime

from app.collectors.ingest import redact_text
from app.db.repository import Repository
from app.llm.openrouter import OpenRouterCardGenerator
from app.menu_batches.store import BatchStore, entity_labels, stamp
from app.ops.budget import SearchBudgetExceeded, SearchBudgetTransport
from app.ops.schedule import fresh_research_events
from app.pipeline.cards import generate_cards
from app.research.facts import OpenRouterFactSheetGenerator, build_event_fact_sheet
from app.research.web import MonidWebResearchClient, ResearchEventService

# Monid inspect, 2026-10-21. Admission holds are conservative, not invoices.
SEARCH_RESULT_RESERVE = 0.000253
PAGE_RESERVE = 0.002533


class PreparationRepository(Repository):
    def __init__(self, path, preparation_id):
        super().__init__(path)
        self.preparation_id = preparation_id

    def record_usage(self, provider, operation, *, metadata=None, **kwargs):
        return super().record_usage(
            provider,
            operation,
            metadata={**(metadata or {}), "menu_preparation_id": self.preparation_id},
            **kwargs,
        )


class PublicationTransport(SearchBudgetTransport):
    PROMPT_PRICE = 0.5
    COMPLETION_PRICE = 2.0
    MAX_OUTPUT_TOKENS = 6500
    MAX_PROMPT_BYTES = 180_000


class PreparationLimitExceeded(SearchBudgetExceeded):
    rejected_before_dispatch = True


def in_period(event, batch):
    try:
        value = datetime.fromisoformat(
            str(event.get("event_at") or event.get("source_published_at")).replace("Z", "+00:00")
        )
        return value.tzinfo is not None and datetime.fromisoformat(
            batch["period_start"]
        ) <= value <= datetime.fromisoformat(batch["period_end"])
    except (ValueError, TypeError):
        return False


class PreparationBudget:
    def __init__(self, settings, repository, limit, *, transport=None):
        self.settings, self.repository = settings, repository
        self.limit, self.web_reserved = limit, 0.0
        self.web_unknown = 0.0
        self.deadline = time.monotonic() + 360
        self.model = PublicationTransport(limit, **({"transport": transport} if transport else {}))

    def check(self, extra=0):
        charged = self.model.snapshot()["charged_or_reserved_usd"]
        if (
            time.monotonic() > self.deadline
            or self.model.snapshot()["unknown_cost_reserved_usd"] > 0
            or self.web_unknown > 0
            or charged + self.web_reserved + extra > self.limit + 1e-10
            or self.repository.current_month_cost() + extra >= self.settings.budget_hard_stop_usd
        ):
            raise PreparationLimitExceeded("Edition preparation budget exhausted before dispatch")

    def reserve_web(self, amount):
        self.check(amount)
        self.web_reserved += amount
        self.model.limit_usd = self.limit - self.web_reserved

    def transport(self, payload, key):
        try:
            _, estimate = self.model._prepare(payload)
            self.check(estimate)
            return self.model(payload, key)
        except SearchBudgetExceeded as exc:
            exc.rejected_before_dispatch = True
            raise

    def snapshot(self):
        model = self.model.snapshot()
        return {
            **model,
            "limit_usd": self.limit,
            "web_catalogue_reserved_usd": self.web_reserved,
            "web_unknown_reserved_usd": self.web_unknown,
            "charged_or_reserved_usd": model["charged_or_reserved_usd"] + self.web_reserved,
        }


class BoundedWeb(MonidWebResearchClient):
    def __init__(self, settings, budget):
        super().__init__(settings)
        self.budget = budget

    def search(self, query, *, freshness="last_week", num_results=10):
        self.budget.reserve_web(num_results * SEARCH_RESULT_RESERVE)
        try:
            result = super().search(query, freshness=freshness, num_results=num_results)
        except Exception:
            self.budget.web_unknown += num_results * SEARCH_RESULT_RESERVE
            raise
        return replace(result, cost_usd=len(result.results) * SEARCH_RESULT_RESERVE)

    def scrape(self, url):
        self.budget.reserve_web(PAGE_RESERVE)
        try:
            return replace(super().scrape(url), cost_usd=PAGE_RESERVE)
        except Exception:
            self.budget.web_unknown += PAGE_RESERVE
            raise


def prepare_news(settings, repository, batch, budget):
    candidates = fresh_research_events(repository, now=datetime.now(UTC), limit=150)
    allowed = {"news": {"rss", "web"}, "telegram": {"telegram"}, "twitter": {"x"}}.get(
        batch["kind"]
    )
    candidates = [
        e
        for e in candidates
        if e["source_kind"] != "youtube"
        and (not allowed or e["source_kind"] in allowed)
        and in_period(e, batch)
        and (
            batch["kind"] not in {"people", "institutions"}
            or entity_labels(settings, e, batch["kind"])
        )
    ]
    from app.personalization.daily import load_context
    from app.pipeline.ranking import rank_event_candidates

    context = load_context(settings, repository, owner=batch["owner_id"])
    if context:
        candidates = rank_event_candidates(candidates, limit=5, personalization=context)
    web = BoundedWeb(settings, budget)
    for event in candidates[:5]:
        budget.check()
        # Respect paused/removed sources before any paid request.
        with repository.connection() as db:
            source = db.execute(
                "SELECT enabled FROM sources WHERE id=?", (event["source_id"],)
            ).fetchone()
        override = repository.get_runtime_setting("source_enabled:" + event["source_id"])
        if override == "false" or (override is None and not source["enabled"]):
            continue
        event_id = event["id"]
        with repository.connection() as db:
            if db.execute(
                "SELECT 1 FROM menu_preparation_events WHERE event_id=? "
                "AND status IN('running','uncertain')",
                (event_id,),
            ).fetchone():
                continue
            db.execute(
                "INSERT INTO menu_preparation_events(preparation_id,event_id) VALUES(?,?)",
                (repository.preparation_id, event_id),
            )
        facts = repository.latest_valid_fact_sheet(event_id)
        if not facts:
            if not repository.event_research_documents(event_id):
                ResearchEventService(settings, repository, client=web).research(
                    event_id, freshness="last_week", max_pages=2
                )
            build_event_fact_sheet(
                settings,
                repository,
                event_id,
                generator=OpenRouterFactSheetGenerator(settings, transport=budget.transport),
            )
        budget.check()
        generate_cards(
            settings,
            repository,
            limit=1,
            event_id=event_id,
            generator=OpenRouterCardGenerator(settings, transport=budget.transport),
        )
        with repository.connection() as db:
            db.execute(
                "UPDATE menu_preparation_events SET status='complete' WHERE "
                "preparation_id=? AND event_id=?",
                (repository.preparation_id, event_id),
            )
        BatchStore(settings, repository).extend(batch["id"])
        if len(BatchStore(settings, repository).items(batch["id"])) >= 7:
            break


def prepare_videos(settings, repository, batch, budget):
    from app.telegram_bot.sensory import video_description
    from app.video.relevance import VideoRelevanceService

    service = VideoRelevanceService(
        settings,
        repository,
        batch["owner_id"],
        assessor=lambda *a, **kw: video_description(*a, **kw, transport=budget.transport),
    )
    with repository.connection() as db:
        rows = db.execute(
            "SELECT i.*,s.name source_name FROM source_items i JOIN sources s ON s.id=i.source_id "
            "JOIN youtube_video_types t ON t.video_id=i.external_id AND t.is_short=0 "
            "LEFT JOIN youtube_channels y ON y.source_id=s.id WHERE s.kind='youtube' "
            "AND s.admission_mode!='history' AND y.removed_at IS NULL "
            "AND COALESCE((SELECT CASE value WHEN 'true' THEN 1 WHEN 'false' THEN 0 END "
            "FROM runtime_settings WHERE key='source_enabled:'||s.id),s.enabled)=1 "
            "AND julianday(i.published_at) BETWEEN julianday(?) AND julianday(?) "
            "AND (y.not_before IS NULL OR julianday(i.published_at)>=julianday(y.not_before)) "
            "AND NOT EXISTS(SELECT 1 FROM video_recommendations v JOIN source_items seen "
            "ON seen.id=v.source_item_id WHERE v.chat_id=? AND seen.external_id=i.external_id) "
            "ORDER BY i.published_at DESC LIMIT 12",
            (batch["period_start"], batch["period_end"], batch["owner_id"]),
        ).fetchall()
    for row in rows:
        budget.check()
        service.assess(dict(row))
    BatchStore(settings, repository).extend(batch["id"])


def prepare(settings, repository, preparation):
    scoped = PreparationRepository(repository.database_path, preparation["id"])
    batch = BatchStore(settings, scoped).get(preparation["batch_id"])
    budget = PreparationBudget(settings, scoped, preparation["limit_usd"])
    status, error = "complete", None
    biography_cost = 0.0
    try:
        BatchStore(settings, scoped).extend(batch["id"])
        from app.menu_batches.store import KINDS

        if len(BatchStore(settings, scoped).items(batch["id"])) >= KINDS[batch["kind"]][2]:
            return
        if batch["kind"] != "youtube" and settings.openrouter_api_key is None:
            raise ValueError("Model access is not configured; no paid preparation dispatched")
        if batch["kind"] == "bio":
            from app.biography.engine import BiographyEngine
            from app.biography.storage import BiographyStore

            options = settings.model_copy(
                update={"biography_max_cost_usd": preparation["limit_usd"]}
            )

            # The biography engine keeps its own durable reservations and receipts.
            class ManualBiographyStore(BiographyStore):
                def start_cycle(self, *args, **kwargs):
                    cycle = super().start_cycle(*args, **kwargs)
                    with scoped.connection() as db:
                        db.execute(
                            "UPDATE menu_preparations SET biography_cycle_id=? WHERE id=?",
                            (cycle, preparation["id"]),
                        )
                    return cycle

            result = BiographyEngine(options, ManualBiographyStore(scoped)).prepare_one(kind="menu")
            with scoped.connection() as db:
                biography_cost = db.execute(
                    "SELECT COALESCE(SUM(cost_usd),0) FROM biography_usage WHERE cycle_id=?",
                    (result.get("cycle_id"),),
                ).fetchone()[0]
            if result.get("status") not in {"ready", "completed"}:
                error = str(result.get("status") or "not_ready")
                status = "budget_blocked" if error == "budget_blocked" else "failed"
        elif batch["kind"] == "youtube":
            prepare_videos(settings, scoped, batch, budget)
        else:
            prepare_news(settings, scoped, batch, budget)
    except SearchBudgetExceeded:
        status, error = "budget_blocked", "Preparation reached its reservation limit"
    except Exception as exc:
        status, error = "failed", redact_text(type(exc).__name__ + ": " + str(exc))[:350]
    finally:
        snapshot = budget.snapshot()
        if batch["kind"] == "bio":
            snapshot.update(
                mode="biography_durable_reservations", charged_or_reserved_usd=biography_cost
            )
        with scoped.connection() as db:
            accounted = db.execute(
                "SELECT COALESCE(SUM(cost_usd),0) FROM usage_ledger WHERE "
                "json_extract(metadata_json,'$.menu_preparation_id')=?",
                (preparation["id"],),
            ).fetchone()[0]
        accounted += biography_cost
        unaccounted = max(0, snapshot["charged_or_reserved_usd"] - accounted)
        if unaccounted > 1e-8:
            scoped.record_usage(
                "mixed",
                "menu.preparation.accounting_reserve",
                cost_usd=unaccounted,
                request_count=0,
                metadata=snapshot,
            )
        with scoped.connection() as db:
            uncertain = (
                snapshot.get("unknown_cost_reserved_usd", 0) > 0
                or snapshot.get("web_unknown_reserved_usd", 0) > 0
            )
            db.execute(
                "UPDATE menu_preparation_events SET status=? WHERE preparation_id=? "
                "AND status='running'",
                ("uncertain" if uncertain else "complete", preparation["id"]),
            )
            db.execute(
                "UPDATE menu_preparations SET status=?,cost_usd=?,budget_json=?,error=?,"
                "updated_at=? WHERE id=?",
                (
                    status,
                    max(accounted, snapshot["charged_or_reserved_usd"]),
                    json.dumps(snapshot),
                    error,
                    stamp(),
                    preparation["id"],
                ),
            )
        BatchStore(settings, scoped).extend(batch["id"])


def recover_preparations(repository):
    with repository.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        for prep in db.execute("SELECT * FROM menu_preparations WHERE status='running'").fetchall():
            cost = prep["limit_usd"]
            if prep["biography_cycle_id"]:
                cost = db.execute(
                    "SELECT COALESCE(SUM(cost_usd),0) FROM biography_usage WHERE cycle_id=?",
                    (prep["biography_cycle_id"],),
                ).fetchone()[0]
            else:
                known = db.execute(
                    "SELECT COALESCE(SUM(cost_usd),0) FROM usage_ledger WHERE "
                    "json_extract(metadata_json,'$.menu_preparation_id')=?",
                    (prep["id"],),
                ).fetchone()[0]
                if cost > known:
                    db.execute(
                        "INSERT INTO usage_ledger(provider,operation,cost_usd,request_count,"
                        "metadata_json) VALUES(?,?,?,0,?)",
                        (
                            "mixed",
                            "menu.preparation.interrupted_reserve",
                            cost - known,
                            json.dumps(
                                {
                                    "menu_preparation_id": prep["id"],
                                    "cost_status": "reservation_not_invoice",
                                }
                            ),
                        ),
                    )
            db.execute(
                "UPDATE menu_preparations SET status='uncertain',cost_usd=?,"
                "error='process restarted',updated_at=? WHERE id=?",
                (cost, stamp(), prep["id"]),
            )
            db.execute(
                "UPDATE menu_preparation_events SET status='uncertain' WHERE "
                "preparation_id=? AND status='running'",
                (prep["id"],),
            )
