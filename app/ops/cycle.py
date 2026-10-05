from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings, load_source_registry
from app.db.repository import Repository
from app.ops.budget import BudgetState, evaluate_budget


@dataclass(frozen=True)
class CycleResult:
    run_id: str
    source_count: int
    processed_count: int
    paid_operations_allowed: bool
    budget_state: BudgetState


def run_empty_cycle(settings: Settings, repository: Repository) -> CycleResult:
    """Phase-0 cycle: validate state without contacting external services."""
    run_id = repository.start_run("empty_cycle", {"phase": 0})
    try:
        registry = load_source_registry(settings.sources_file)
        source_count = repository.sync_sources(registry)
        month_cost = repository.current_month_cost()
        budget = evaluate_budget(
            month_cost,
            settings.budget_warning_usd,
            settings.budget_hard_stop_usd,
        )
        repository.finish_run(run_id, processed_count=0, cost_usd=0)
        return CycleResult(
            run_id=run_id,
            source_count=source_count,
            processed_count=0,
            paid_operations_allowed=budget.paid_operations_allowed,
            budget_state=budget.state,
        )
    except Exception as exc:
        repository.finish_run(run_id, status="failed", error_count=1, error=str(exc))
        raise
