from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from app.config import Settings
from app.db.repository import Repository
from app.details.models import DetailArticleBuildResult
from app.details.provider_guard import DetailProviderError
from app.details.research import (
    DetailBudgetError,
    DetailEvidenceError,
    DetailGenerationError,
    build_detail_article,
    detail_max_cost,
)
from app.research.detail_policy import article_reference_fingerprint, detail_policy_versions


class DetailBuilder(Protocol):
    def __call__(
        self,
        settings: Settings,
        repository: Repository,
        event_id: int,
        *,
        generation_key: str | None = None,
        progress: Callable[[str], None] | None = None,
        depth: str = "legacy",
        reader_context: dict | None = None,
    ) -> DetailArticleBuildResult: ...


@dataclass(frozen=True)
class DetailWorkerResult:
    request_id: int | None
    status: str
    article_id: int | None = None
    cost_usd: float = 0.0
    error: str | None = None
    cache_hit: bool = False


class DetailWorker:
    """Single-job DB-backed worker for bounded on-demand research."""

    def __init__(
        self,
        settings: Settings,
        repository: Repository,
        *,
        builder: DetailBuilder = build_detail_article,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.builder = builder

    @property
    def lease_minutes(self) -> int:
        return int(getattr(self.settings, "details_worker_lease_minutes", 10))

    @property
    def max_cost_usd(self) -> float:
        return float(getattr(self.settings, "details_max_cost_usd", 0.02))

    @property
    def max_concurrency(self) -> int:
        return int(getattr(self.settings, "details_worker_concurrency", 1))

    def _progress(self, request_id: int, stage: str) -> None:
        updated = self.repository.update_detail_request_stage(
            request_id,
            stage,
            lease_minutes=self.lease_minutes,
        )
        if not updated:
            raise RuntimeError(f"Detail request {request_id} lost its worker lease")

    def run_once(self) -> DetailWorkerResult:
        request = self.repository.claim_next_detail_request(
            lease_minutes=self.lease_minutes,
            max_concurrency=self.max_concurrency,
        )
        if request is None:
            return DetailWorkerResult(request_id=None, status="idle")

        request_id = int(request["request_id"])
        generation_key = str(request["generation_key"])
        depth = str(request.get("depth") or "legacy")
        reader_context = dict(request.get("reader_context") or {})
        fingerprint = reader_context.get("context_fingerprint")
        cache_options = (
            {
                "depth": depth,
                "context_fingerprint": fingerprint,
                "reader_context": reader_context,
            }
            if depth != "legacy"
            else {}
        )
        cached = self.repository.cached_detailed_article(generation_key, **cache_options)
        if cached is None:
            cached = self.repository.cached_detailed_article_for_event(
                int(request["event_id"]), **cache_options
            )
        if cached is not None:
            attached = self.repository.complete_detail_request_from_cache(
                request_id,
                int(cached["id"]),
            )
            if not attached:
                return DetailWorkerResult(
                    request_id=request_id,
                    status="failed",
                    error="cached article could not be attached",
                )
            return DetailWorkerResult(
                request_id=request_id,
                status="ready",
                article_id=int(cached["id"]),
                cache_hit=True,
            )

        if self.repository.current_month_cost() >= self.settings.budget_hard_stop_usd:
            error = "monthly budget hard stop"
            self.repository.fail_detail_request(
                request_id,
                status="budget_blocked",
                error=error,
            )
            return DetailWorkerResult(
                request_id=request_id,
                status="budget_blocked",
                error=error,
            )

        try:
            result = self.builder(
                self.settings,
                self.repository,
                int(request["event_id"]),
                # Let the builder include the selected document hash in the final key.
                generation_key=None,
                progress=lambda stage: self._progress(request_id, stage),
                **({"depth": depth, "reader_context": reader_context} if depth != "legacy" else {}),
            )
        except DetailProviderError as exc:
            error = f"Detail provider {exc.status}; call_id={exc.call_id}"
            status = "budget_blocked" if exc.status == "budget_blocked" else "failed"
            self.repository.fail_detail_request(request_id, status=status, error=error)
            return DetailWorkerResult(
                request_id=request_id, status=status, cost_usd=float(exc.cost_usd), error=error
            )
        except DetailEvidenceError as exc:
            error = str(exc)
            self.repository.fail_detail_request(
                request_id,
                status="insufficient_evidence",
                error=error,
            )
            return DetailWorkerResult(
                request_id=request_id,
                status="insufficient_evidence",
                cost_usd=float(exc.cost_usd),
                error=error,
            )
        except DetailBudgetError as exc:
            error = str(exc)
            self.repository.fail_detail_request(
                request_id,
                status="budget_blocked",
                error=error,
            )
            return DetailWorkerResult(
                request_id=request_id,
                status="budget_blocked",
                cost_usd=float(exc.cost_usd),
                error=error,
            )
        except DetailGenerationError as exc:
            error = str(exc)
            self.repository.fail_detail_request(request_id, status="failed", error=error)
            return DetailWorkerResult(
                request_id=request_id,
                status="failed",
                cost_usd=float(exc.cost_usd),
                error=error,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self.repository.fail_detail_request(request_id, status="failed", error=error)
            return DetailWorkerResult(request_id=request_id, status="failed", error=error)

        if result.validation_status == "insufficient_evidence":
            error = "; ".join(result.validation_errors) or "insufficient evidence"
            self.repository.fail_detail_request(
                request_id,
                status="insufficient_evidence",
                error=error,
            )
            return DetailWorkerResult(
                request_id=request_id,
                status="insufficient_evidence",
                cost_usd=result.cost_usd,
                error=error,
            )
        if result.validation_status != "valid":
            error = "; ".join(result.validation_errors) or "invalid detail article"
            self.repository.fail_detail_request(request_id, status="failed", error=error)
            return DetailWorkerResult(
                request_id=request_id,
                status="failed",
                cost_usd=result.cost_usd,
                error=error,
            )
        expected_schema, expected_prompt = detail_policy_versions(depth)
        if (
            result.prompt_version != expected_prompt
            or result.schema_version != expected_schema
            or result.depth != depth
        ):
            error = "Detail builder returned an outdated evidence-policy artifact"
            self.repository.fail_detail_request(request_id, status="failed", error=error)
            return DetailWorkerResult(
                request_id=request_id, status="failed", cost_usd=result.cost_usd, error=error
            )
        if (
            result.event_id != int(request["event_id"])
            or (
                depth != "legacy"
                and result.reader_context.get("context_fingerprint") != fingerprint
            )
            or (
                depth == "article"
                and article_reference_fingerprint(result.reader_context)
                != article_reference_fingerprint(reader_context)
            )
        ):
            error = (
                "Detail builder returned an artifact for another event or frozen reader baseline"
            )
            self.repository.fail_detail_request(request_id, status="failed", error=error)
            return DetailWorkerResult(
                request_id=request_id, status="failed", cost_usd=result.cost_usd, error=error
            )
        max_cost = detail_max_cost(self.settings, depth)
        if result.cost_usd > max_cost:
            error = f"detail request cost ${result.cost_usd:.6f} exceeded ${max_cost:.6f} cap"
            self.repository.fail_detail_request(
                request_id,
                status="budget_blocked",
                error=error,
            )
            return DetailWorkerResult(
                request_id=request_id,
                status="budget_blocked",
                cost_usd=result.cost_usd,
                error=error,
            )

        provider_metadata = dict(result.provider_response_metadata)
        provider_metadata.setdefault("research_plan", result.research_plan)
        article_id = self.repository.store_detail_article_and_complete(
            request_id,
            generation_key=result.generation_key,
            schema_version=result.schema_version,
            prompt_version=result.prompt_version,
            model=result.model,
            provider_response_metadata=provider_metadata,
            fact_snapshot=result.fact_snapshot,
            document_ids=result.document_ids,
            structured_output=result.structured_output,
            rendered_html=result.rendered_html,
            source_urls=result.source_urls,
            validation_status=result.validation_status,
            validation_errors=result.validation_errors,
            generation_cost_usd=result.cost_usd,
            expires_at=result.expires_at,
            **(
                {
                    "depth": depth,
                    "reader_context": result.reader_context,
                    "context_fingerprint": fingerprint,
                }
                if depth != "legacy"
                else {}
            ),
        )
        return DetailWorkerResult(
            request_id=request_id,
            status="ready",
            article_id=article_id,
            cost_usd=result.cost_usd,
        )

    def run_loop(
        self,
        *,
        poll_interval_seconds: float = 2.0,
        stop_requested: Callable[[], bool] | None = None,
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        should_stop = stop_requested or (lambda: False)
        while not should_stop():
            result = self.run_once()
            if result.status == "idle":
                time.sleep(poll_interval_seconds)

    async def run_loop_async(
        self,
        *,
        poll_interval_seconds: float = 2.0,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        while stop_event is None or not stop_event.is_set():
            result = await asyncio.to_thread(self.run_once)
            if result.status == "idle":
                await asyncio.sleep(poll_interval_seconds)


def run_detail_worker_once(
    settings: Settings,
    repository: Repository,
    *,
    builder: DetailBuilder = build_detail_article,
) -> DetailWorkerResult:
    return DetailWorker(settings, repository, builder=builder).run_once()
