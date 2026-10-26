"""Durable paid-call boundaries for the new expanded/article news reader."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import UTC, datetime
from urllib.error import HTTPError

from app.collectors.ingest import SECRET_KEYS, redact_text
from app.llm.accounting import (
    MODEL_PRICES_PER_MILLION,
    finite_money,
    nonnegative_int,
    resolve_usage_cost,
)
from app.llm.openrouter import OpenRouterRequestError, OpenRouterResponseError
from app.research.web import (
    SCRAPE_COST_USD,
    SEARCH_RESULT_COST_USD,
    ScrapeResponse,
    SearchResponse,
    WebResearchError,
)

MAX_RESPONSE_BYTES = 8_000_000
MAX_REQUEST_COST_USD = 0.02
DEFAULT_PRICE_CEILING = (0.5, 2.0)


class DetailProviderError(RuntimeError):
    def __init__(self, status, *, call_id=None, cost_usd=0.0, http_status=None, dispatched=False):
        self.status = status
        self.call_id = call_id
        self.cost_usd = cost_usd
        self.http_status = http_status
        self.dispatched = dispatched
        super().__init__(
            {
                "uncertain": "Paid request result is unconfirmed; automatic replay is blocked",
                "insufficient_credits": (
                    "Provider rejected this request with HTTP 402; explicit retry required"
                ),
                "budget_blocked": "Paid request would exceed its reserved budget",
                "failed": "Confirmed provider failure; automatic replay is blocked",
                "routing_unavailable": ("Request rejected before model dispatch; "
                                        "routing unavailable"),
            }.get(status, status)
        )


def _safe(value):
    """Keep full evidence/response text while removing credentials, never clipping it."""
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items() if not SECRET_KEYS.search(str(k))}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def _json(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _http_status(exc):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, HTTPError):
            return exc.code
        value = getattr(exc, "http_status", None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        exc = exc.__cause__
    return None


def _web_operation(provider, stage):
    return provider == "monid/context.dev" and stage in {"search", "scrape"}


_BUDGET_SCOPE_SQL = (
    "c.event_id=? AND c.baseline_fingerprint=? AND (c.depth=? OR "
    "(c.provider='monid/context.dev' AND c.stage IN ('search','scrape')))"
)


class DurableDetailProvider:
    def __init__(
        self,
        settings,
        repository,
        event_id,
        depth,
        baseline_fingerprint,
        *,
        request_budget_usd=MAX_REQUEST_COST_USD,
        retry_rejected=False,
        retry_authorization=None,
    ):
        if depth not in {"expanded", "article"} or not baseline_fingerprint:
            raise ValueError("Durable detail calls require a new depth and frozen baseline")
        budget = finite_money(request_budget_usd)
        configured = float(
            getattr(settings, f"news_reader_{depth}_max_cost_usd", MAX_REQUEST_COST_USD)
        )
        if budget is None or budget > configured:
            raise ValueError("Detail provider budget exceeds its configured depth cap")
        self.settings, self.repository, self.event_id = settings, repository, int(event_id)
        self.depth, self.baseline_fingerprint = depth, str(baseline_fingerprint)
        self.budget = budget
        self.retry_rejected = bool(retry_rejected)
        self.retry_authorization = str(retry_authorization) if retry_authorization else None
        self.attempted = set()
        self.last_call_id = None
        self.new_usage_ids = set()
        self.cached_call_ids = set()

    def _metadata(self, provider, stage, call_key, reservation, *, model=None):
        return {
            "event_id": self.event_id,
            "depth": self.depth,
            "baseline_fingerprint": self.baseline_fingerprint,
            "provider": provider,
            "stage": stage,
            "call_key": call_key,
            "model": model,
            "reservation_usd": reservation,
            "cost_estimated": True,
            "billing_basis": "reservation_before_dispatch",
        }

    @property
    def total_cost_usd(self):
        with self.repository.connection() as db:
            return float(
                db.execute(
                    "SELECT COALESCE(SUM(l.cost_usd),0) FROM details_provider_calls c "
                    "JOIN usage_ledger l ON l.id=c.usage_id WHERE " + _BUDGET_SCOPE_SQL,
                    (self.event_id, self.baseline_fingerprint, self.depth),
                ).fetchone()[0]
            )

    @property
    def incremental_cost_usd(self):
        """New reservations/charges made by this instance; cached traversal adds zero."""
        if not self.new_usage_ids:
            return 0.0
        placeholders = ",".join("?" for _ in self.new_usage_ids)
        with self.repository.connection() as db:
            return float(
                db.execute(
                    "SELECT COALESCE(SUM(cost_usd),0) FROM usage_ledger "
                    f"WHERE id IN ({placeholders})",
                    tuple(self.new_usage_ids),
                ).fetchone()[0]
            )

    def run(
        self,
        provider,
        stage,
        request,
        reservation_usd,
        call: Callable,
        *,
        encode=lambda value: value,
        decode=lambda value: value,
        cost=lambda value, reserve: (reserve, {"cost_estimated": True}),
        on_cache=lambda value: value,
    ):
        reserve = finite_money(reservation_usd)
        if reserve is None or reserve <= 0:
            raise DetailProviderError("budget_blocked")
        identity = {
            "event_id": self.event_id,
            "depth": None if _web_operation(provider, stage) else self.depth,
            "baseline": self.baseline_fingerprint,
            "provider": provider,
            "stage": stage,
            "request": request,
        }
        call_key = hashlib.sha256(_json(identity).encode()).hexdigest()
        model = request.get("model") if isinstance(request, dict) else None
        now = datetime.now(UTC).isoformat()
        with self.repository.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute(
                "SELECT * FROM details_provider_calls WHERE call_key=?", (call_key,)
            ).fetchone()
            if old:
                self.last_call_id = old["id"]
                if old["status"] == "completed":
                    try:
                        decoded = decode(json.loads(old["response_json"]))
                        self.cached_call_ids.add(old["id"])
                        return on_cache(decoded)
                    except (TypeError, ValueError, KeyError) as exc:
                        raise DetailProviderError("failed", call_id=old["id"]) from exc
                error = json.loads(old["error_json"] or "{}")
                if (
                    old["status"] == "rejected"
                    and error.get("http_status") == 404
                    and error.get("rejected_before_dispatch") is True
                ):
                    raise DetailProviderError(
                        "routing_unavailable", call_id=old["id"], cost_usd=0, http_status=404
                    )
                retryable = (
                    old["status"] == "rejected"
                    and error.get("http_status") == 402
                    and error.get("rejected_before_dispatch") is True
                )
                if not (
                    retryable
                    and self.retry_rejected
                    and self.retry_authorization
                    and old["retry_authorization"] != self.retry_authorization
                    and call_key not in self.attempted
                ):
                    accounted = db.execute(
                        "SELECT cost_usd FROM usage_ledger WHERE id=?", (old["usage_id"],)
                    ).fetchone()
                    raise DetailProviderError(
                        "insufficient_credits"
                        if retryable
                        else "failed"
                        if old["status"] == "failed"
                        else "uncertain",
                        call_id=old["id"],
                        cost_usd=0.0
                        if retryable
                        else (accounted[0] if accounted else old["reserved_usd"]),
                        http_status=error.get("http_status"),
                    )
            unresolved = (
                None
                if _web_operation(provider, stage)
                else db.execute(
                    "SELECT c.id,c.reserved_usd,l.cost_usd FROM details_provider_calls c "
                    "LEFT JOIN usage_ledger l ON l.id=c.usage_id "
                    "WHERE c.event_id=? AND c.depth=? AND c.baseline_fingerprint=? "
                    "AND NOT(c.provider='monid/context.dev' AND c.stage IN ('search','scrape')) "
                    "AND c.status IN ('preparing','uncertain') ORDER BY c.id LIMIT 1",
                    (self.event_id, self.depth, self.baseline_fingerprint),
                ).fetchone()
            )
            if unresolved:
                # An uncertain LLM result cannot be repaid by a model/prompt or
                # stage change. Web failures block their exact args, allowing
                # legitimate alternative pages to be read within the same budget.
                raise DetailProviderError(
                    "uncertain",
                    call_id=unresolved["id"],
                    cost_usd=(
                        unresolved["cost_usd"]
                        if unresolved["cost_usd"] is not None
                        else unresolved["reserved_usd"]
                    ),
                )
            spent = db.execute(
                "SELECT COALESCE(SUM(cost_usd),0) FROM usage_ledger "
                "WHERE strftime('%Y-%m',created_at)=strftime('%Y-%m','now')"
            ).fetchone()[0]
            in_flight = db.execute(
                "SELECT COALESCE(SUM(reserved_usd),0) FROM source_runs WHERE status='running'"
            ).fetchone()[0]
            request_spent = db.execute(
                "SELECT COALESCE(SUM(l.cost_usd),0) FROM details_provider_calls c "
                "JOIN usage_ledger l ON l.id=c.usage_id WHERE " + _BUDGET_SCOPE_SQL,
                (self.event_id, self.baseline_fingerprint, self.depth),
            ).fetchone()[0]
            if (
                request_spent + reserve > self.budget
                or spent + in_flight + reserve >= self.settings.budget_hard_stop_usd
            ):
                raise DetailProviderError("budget_blocked")
            metadata = self._metadata(provider, stage, call_key, reserve, model=model)
            usage_id = db.execute(
                "INSERT INTO usage_ledger(provider,operation,cost_usd,metadata_json) "
                "VALUES(?,?,?,?)",
                (provider, "details.guarded." + stage, reserve, _json(metadata)),
            ).lastrowid
            if old:
                call_id = old["id"]
                db.execute(
                    "UPDATE details_provider_calls SET status='preparing',usage_id=?,"
                    "reserved_usd=?,response_json=NULL,error_json=NULL,retry_authorization=?,"
                    "attempt_count=attempt_count+1,updated_at=? WHERE id=?",
                    (usage_id, reserve, self.retry_authorization, now, call_id),
                )
            else:
                call_id = db.execute(
                    "INSERT INTO details_provider_calls(event_id,depth,baseline_fingerprint,"
                    "call_key,provider,stage,status,usage_id,reserved_usd,created_at,updated_at) "
                    "VALUES(?,?,?,?,?,?,'preparing',?,?,?,?)",
                    (
                        self.event_id,
                        self.depth,
                        self.baseline_fingerprint,
                        call_key,
                        provider,
                        stage,
                        usage_id,
                        reserve,
                        now,
                        now,
                    ),
                ).lastrowid
        # The reservation is committed before dispatch. No provider runs under a DB lock.
        self.last_call_id = call_id
        self.new_usage_ids.add(usage_id)
        self.attempted.add(call_key)
        try:
            value = call()
        except Exception as exc:
            http_status = _http_status(exc)
            routing_rejected = (
                http_status == 404
                and isinstance(exc, OpenRouterRequestError)
                and getattr(exc, "rejected_before_dispatch", False) is True
            )
            rejected = http_status == 402 or routing_rejected
            charged = finite_money(getattr(exc, "cost_usd", None))
            confirmed = charged is not None and (
                bool(getattr(exc, "response_id", None)) or isinstance(exc, OpenRouterResponseError)
            )
            status = "rejected" if rejected else "failed" if confirmed else "uncertain"
            settled = 0.0 if rejected else charged if confirmed else max(reserve, charged or 0)
            error = {
                "exception_type": type(exc).__name__,
                "http_status": http_status,
                "rejected_before_dispatch": rejected,
            }
            self._finish(
                call_id,
                usage_id,
                status,
                None,
                error,
                settled,
                {
                    **metadata,
                    **_safe(getattr(exc, "billing_metadata", {}) or {}),
                    "cost_estimated": False
                    if rejected
                    else (
                        getattr(exc, "billing_metadata", {}).get("cost_estimated", False)
                        if confirmed
                        else True
                    ),
                    "response_id": getattr(exc, "response_id", None),
                    **error,
                },
            )
            raise DetailProviderError(
                "routing_unavailable"
                if routing_rejected
                else "insufficient_credits"
                if rejected
                else status,
                call_id=call_id,
                cost_usd=settled,
                http_status=http_status,
                dispatched=True,
            ) from exc
        try:
            response = _safe(encode(value))
            serialized = _json(response)
            if len(serialized.encode()) > MAX_RESPONSE_BYTES:
                raise ValueError("Provider response exceeds durable cache limit")
            settled, receipt = cost(value, reserve)
            settled = finite_money(settled)
            if settled is None:
                settled, receipt = reserve, {"cost_estimated": True}
            decoded = decode(response)
        except Exception as exc:
            self._finish(
                call_id,
                usage_id,
                "failed",
                None,
                {"exception_type": type(exc).__name__, "response_received": True},
                reserve,
                metadata,
            )
            raise DetailProviderError(
                "failed", call_id=call_id, cost_usd=reserve, dispatched=True
            ) from exc
        self._finish(
            call_id,
            usage_id,
            "completed",
            serialized,
            None,
            settled,
            {**metadata, **_safe(receipt)},
        )
        if self.total_cost_usd > self.budget:
            raise DetailProviderError(
                "budget_blocked", call_id=call_id, cost_usd=settled, dispatched=True
            )
        return decoded

    def _finish(self, call_id, usage_id, status, response, error, cost, metadata):
        with self.repository.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "UPDATE usage_ledger SET cost_usd=?,metadata_json=? WHERE id=?",
                (cost, _json(metadata), usage_id),
            )
            db.execute(
                "UPDATE details_provider_calls SET status=?,response_json=?,error_json=?,"
                "updated_at=? WHERE id=? AND usage_id=?",
                (
                    status,
                    response,
                    _json(error) if error else None,
                    datetime.now(UTC).isoformat(),
                    call_id,
                    usage_id,
                ),
            )

    def llm_transport(self, base_transport, stage):
        def transport(payload, key):
            # Clone the actual whole payload, including schemas and any repair message.
            request = json.loads(_json(payload))
            model = str(request.get("model") or "")
            provider = request.setdefault("provider", {})
            ceiling = provider.get("max_price") or {}
            prices = (finite_money(ceiling.get("prompt")), finite_money(ceiling.get("completion")))
            if any(price is None for price in prices):
                if model not in MODEL_PRICES_PER_MILLION:
                    raise DetailProviderError("budget_blocked")
                # An enforced routing ceiling, rather than a claim that old
                # model-price snapshots reflect today's provider tariff.
                prices = DEFAULT_PRICE_CEILING
                provider["max_price"] = {"prompt": prices[0], "completion": prices[1]}
            maximum = nonnegative_int(
                request.get("max_tokens") or request.get("max_completion_tokens")
            )
            if maximum <= 0:
                raise DetailProviderError("budget_blocked")
            reserve = (
                (len(_json(request).encode()) + 2048) * prices[0] + maximum * prices[1]
            ) / 1e6

            def settle(value, reservation):
                bill = resolve_usage_cost(
                    value.get("usage"),
                    model=str(value.get("model") or model),
                    price_ceiling_per_million=prices,
                    reservation_usd=reservation,
                )
                return bill.cost_usd if not bill.unavailable else reservation, bill.metadata()

            return self.run(
                "openrouter",
                stage,
                request,
                reserve,
                lambda: base_transport(request, key),
                cost=settle,
            )

        return transport

    def web_client(self, base_client):
        guard = self

        def web_call(*args, **kwargs):
            try:
                return guard.run(
                    *args, **kwargs, on_cache=lambda value: replace(value, cost_usd=0.0)
                )
            except DetailProviderError as exc:
                error = WebResearchError(str(exc))
                error.status = exc.status
                error.call_id = exc.call_id
                error.cost_usd = exc.cost_usd if exc.dispatched else 0.0
                error.retained_cost_usd = exc.cost_usd
                error.http_status = exc.http_status
                raise error from exc

        class GuardedWebClient:
            def search(self, query, *, freshness="last_month", num_results=10, limit=None):
                number = num_results if limit is None else limit
                if not isinstance(number, int) or isinstance(number, bool) or not 1 <= number <= 10:
                    raise ValueError("Search must remain bounded to 1–10 results")
                request = {"query": query, "freshness": freshness, "num_results": number}
                return web_call(
                    "monid/context.dev",
                    "search",
                    request,
                    number * SEARCH_RESULT_COST_USD,
                    lambda: base_client.search(query, freshness=freshness, num_results=number),
                    encode=asdict,
                    decode=lambda value: SearchResponse(**value),
                    cost=lambda value, _: (
                        value.cost_usd,
                        {"cost_estimated": True, "billing_basis": "configured_rate_estimate"},
                    ),
                )

            def scrape(self, url):
                return web_call(
                    "monid/context.dev",
                    "scrape",
                    {"url": url},
                    SCRAPE_COST_USD,
                    lambda: base_client.scrape(url),
                    encode=asdict,
                    decode=lambda value: ScrapeResponse(**value),
                    cost=lambda value, _: (
                        value.cost_usd,
                        {"cost_estimated": True, "billing_basis": "configured_rate_estimate"},
                    ),
                )

        return GuardedWebClient()

    def accounting_repository(self):
        guard = self

        class AccountingRepository:
            def __getattr__(self, name):
                return getattr(guard.repository, name)

            def record_usage(self, provider, operation, **kwargs):
                if not operation.startswith("details."):
                    return guard.repository.record_usage(provider, operation, **kwargs)
                if guard.last_call_id is None:
                    raise RuntimeError("Detail accounting annotation has no guarded provider call")
                # Keep downstream validation diagnostics without charging a second ledger row.
                with guard.repository.connection() as db:
                    row = db.execute(
                        "SELECT usage_id FROM details_provider_calls WHERE id=?",
                        (guard.last_call_id,),
                    ).fetchone()
                    stored = db.execute(
                        "SELECT metadata_json FROM usage_ledger WHERE id=?", (row[0],)
                    ).fetchone()
                    metadata = json.loads(stored[0])
                    metadata["validation_operation"] = operation
                    metadata["validation_metadata"] = _safe(kwargs.get("metadata") or {})
                    db.execute(
                        "UPDATE usage_ledger SET metadata_json=? WHERE id=?",
                        (_json(metadata), row[0]),
                    )

        return AccountingRepository()
