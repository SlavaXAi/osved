from __future__ import annotations

import copy
import json
import math
import threading
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from app.llm.openrouter import OpenRouterRequestError, Transport, http_transport


class BudgetState(StrEnum):
    OK = "ok"
    WARNING = "warning"
    HARD_STOP = "hard_stop"


@dataclass(frozen=True)
class BudgetDecision:
    state: BudgetState
    paid_operations_allowed: bool
    message: str


def evaluate_budget(month_cost: float, warning: float, hard_stop: float) -> BudgetDecision:
    if month_cost >= hard_stop:
        return BudgetDecision(
            state=BudgetState.HARD_STOP,
            paid_operations_allowed=False,
            message="Monthly hard stop reached; paid operations are disabled.",
        )
    if month_cost >= warning:
        return BudgetDecision(
            state=BudgetState.WARNING,
            paid_operations_allowed=True,
            message="Monthly warning threshold reached; reduce optional paid operations.",
        )
    return BudgetDecision(
        state=BudgetState.OK,
        paid_operations_allowed=True,
        message="Budget is within limits.",
    )


class SearchBudgetExceeded(OpenRouterRequestError):
    """A local search reservation was refused before making a paid request."""


class SearchBudgetTransport:
    """Shared, thread-safe reservation guard for one search-radar run.

    This is a conservative *estimate*, not an upstream invoice guarantee. Tool
    wrappers, repeated provider turns and tariffs can change upstream. Unknown
    charges keep their reservation, and an observed overrun stops later calls.
    The guard stores counters only, never prompts, credentials or responses.
    """

    PROMPT_PRICE = 0.25  # USD / million tokens; enforced via provider routing.
    COMPLETION_PRICE = 1.25
    SEARCH_PRICE = 0.007  # Exa auto, <=10 results, official docs 2026-09-07.
    MAX_PROMPT_BYTES = 120_000
    MAX_OUTPUT_TOKENS = 3000
    MAX_REQUESTS = 10

    def __init__(self, limit_usd: float, *, transport: Transport = http_transport) -> None:
        if not math.isfinite(limit_usd) or limit_usd < 0:
            raise ValueError("Search budget must be finite and nonnegative")
        self.limit_usd = limit_usd
        self.transport = transport
        self._lock = threading.Condition()
        self._reserved = 0.0
        self._charged = 0.0
        self._reported = 0.0
        self._unknown = 0.0
        self._requests = 0
        self._blocked = 0
        self._overrun = False
        self._run_id: str | None = None

    def bind_run(self, run_id: str) -> None:
        with self._lock:
            if self._run_id is not None and self._run_id != run_id:
                raise SearchBudgetExceeded("Search budget cannot be reused across runs")
            self._run_id = run_id

    def _prepare(self, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
        bounded = copy.deepcopy(payload)
        for message in bounded.get("messages") or []:
            # Only the known discovery context is compacted. Do not truncate
            # evidence, normalizer reports or other arbitrary JSON/LLM prompts.
            if message.get("role") != "user" or not isinstance(message.get("content"), str):
                continue
            try:
                context = json.loads(message["content"])
            except (ValueError, TypeError):
                continue
            if not isinstance(context, dict) or "SEARCH_WINDOW_START" not in context:
                continue
            if not {"LENS_ID", "INTEREST_PROFILE"} <= context.keys():
                continue
            # Signals already arrive round-robin by platform from compact_fixed_signals.
            signals = context.get("RECENT_FIXED_SIGNALS", [])
            context["RECENT_FIXED_SIGNALS"] = [
                {**row, "signal": str(row.get("signal", ""))[:300]}
                for row in signals[:12]
                if isinstance(row, dict)
            ]
            seen = context.get("SEEN_EVENTS", [])
            context["SEEN_EVENTS"] = [
                {**row, "headline": str(row.get("headline", ""))[:120]}
                for row in seen[:16]
                if isinstance(row, dict)
            ]
            feedback = []
            for row in context.get("FEEDBACK_EXAMPLES", [])[:4]:
                if not isinstance(row, dict):
                    continue
                comment = str(row.get("comment", ""))[:300]
                feedback.append(
                    {
                        **row,
                        "headline": str(row.get("headline", ""))[:100],
                        "summary": str(row.get("summary", ""))[: 300 - len(comment)],
                        "comment": comment,
                    }
                )
            context["FEEDBACK_EXAMPLES"] = feedback
            # Keep the complete seven-topic profile, goals and lens instructions.
            context["CONTEXT_MODE"] = "scheduled_compact_v1"
            message["content"] = json.dumps(context, ensure_ascii=False)
        if bounded.get("plugins") or ":online" in str(bounded.get("model", "")):
            raise SearchBudgetExceeded("Unbounded search plugins are not allowed")
        if bounded.get("stream") or bounded.get("n", 1) != 1:
            raise SearchBudgetExceeded("Only one non-streaming completion is budgeted")
        bounded.pop("stop_server_tools_when", None)
        bounded.pop("max_completion_tokens", None)
        output_tokens = min(int(bounded.get("max_tokens") or 2400), self.MAX_OUTPUT_TOKENS)
        if output_tokens < 1:
            raise SearchBudgetExceeded("Invalid completion limit")
        bounded["max_tokens"] = output_tokens
        provider = bounded.setdefault("provider", {})
        price = provider.setdefault("max_price", {})
        price["prompt"] = min(float(price.get("prompt", self.PROMPT_PRICE)), self.PROMPT_PRICE)
        price["completion"] = min(
            float(price.get("completion", self.COMPLETION_PRICE)), self.COMPLETION_PRICE
        )
        if any(not math.isfinite(value) or value < 0 for value in price.values()):
            raise SearchBudgetExceeded("Invalid provider price ceiling")
        provider["require_parameters"] = True
        searches = 0
        results = 0
        characters = 0
        for tool in bounded.get("tools") or []:
            if tool.get("type") != "openrouter:web_search" or searches:
                raise SearchBudgetExceeded("Only one bounded web-search tool is allowed")
            params = tool.setdefault("parameters", {})
            params.update(engine="exa", mode="auto")
            params["max_uses"] = 1
            params["max_results"] = min(max(int(params.get("max_results") or 3), 1), 3)
            params["max_total_results"] = min(max(int(params.get("max_total_results") or 3), 1), 3)
            params["max_characters"] = min(max(int(params.get("max_characters") or 1200), 1), 1200)
            searches = params["max_uses"]
            results = min(params["max_total_results"], searches * params["max_results"])
            characters = params["max_characters"]
        if searches:
            bounded["max_tool_calls"] = searches
        else:
            # A plain JSON normalization call has no server tools to limit.
            bounded.pop("max_tool_calls", None)
        encoded_bytes = len(json.dumps(bounded, ensure_ascii=False).encode("utf-8"))
        if encoded_bytes > self.MAX_PROMPT_BYTES:
            raise SearchBudgetExceeded("Search prompt exceeds bounded input size")
        # Reserve each possible model turn, rebilled history, tool result text
        # (up to 4 UTF-8 bytes/character) and output history. Byte counts are a
        # deliberately conservative token proxy, not a tokenizer/API guarantee.
        turns = searches + 1
        input_bound = turns * (encoded_bytes + 1500)
        input_bound += searches * (results * characters * 4 + output_tokens * 4)
        estimate = (
            input_bound * price["prompt"] / 1_000_000
            + turns * output_tokens * price["completion"] / 1_000_000
            + searches * self.SEARCH_PRICE
        ) * 1.15
        return bounded, estimate

    def __call__(self, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
        try:
            bounded, reservation = self._prepare(payload)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SearchBudgetExceeded("Invalid bounded search request") from exc
        with self._lock:
            deadline = time.monotonic() + 90
            # Parallel lenses may temporarily fill reservations even though
            # real costs are much lower. Let in-flight calls settle first; never
            # launch on credit and never wait indefinitely after a lost call.
            while (
                not self._overrun
                and self._reserved > 0
                and self._charged + reservation <= self.limit_usd
                and self._charged + self._reserved + reservation > self.limit_usd
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._lock.wait(timeout=remaining)
            if (
                self._overrun
                or self._requests >= self.MAX_REQUESTS
                or self._charged + self._reserved + reservation > self.limit_usd + 1e-12
            ):
                self._blocked += 1
                raise SearchBudgetExceeded("Scheduled search reservation budget exhausted")
            self._reserved += reservation
            self._requests += 1
        actual: float | None = None
        try:
            response = self.transport(bounded, api_key)
            usage = response.get("usage")
            value = usage.get("cost") if isinstance(usage, dict) else None
            if value is not None and not isinstance(value, bool):
                try:
                    parsed = float(value)
                    if math.isfinite(parsed) and parsed >= 0:
                        actual = parsed
                except (ValueError, TypeError):
                    pass
            return response
        except OpenRouterRequestError as exc:
            if getattr(exc, "rejected_before_dispatch", False):
                actual = 0.0
            raise
        finally:
            with self._lock:
                self._reserved = max(0.0, self._reserved - reservation)
                self._charged += actual if actual is not None else reservation
                self._reported += actual or 0.0
                if actual is None:
                    self._unknown += reservation
                self._overrun = self._overrun or self._charged > self.limit_usd + 1e-12
                self._lock.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "mode": "conservative_reservation_not_invoice_cap",
                "run_id": self._run_id,
                "limit_usd": self.limit_usd,
                "charged_or_reserved_usd": round(self._charged + self._reserved, 8),
                "reported_cost_usd": round(self._reported, 8),
                "unknown_cost_reserved_usd": round(self._unknown, 8),
                "inflight_reserved_usd": round(self._reserved, 8),
                "requests": self._requests,
                "blocked_requests": self._blocked,
                "reported_overrun": self._overrun,
            }
