from __future__ import annotations

import asyncio

import pytest

from app.web.budget import LocalBudget, LocalBudgetBlocked
from tests.test_web_api import FakeReports, client_for, fixture, login


def test_imported_cost_is_not_counted_and_local_spend_survives_restart(tmp_path):
    _, repo, _ = fixture(tmp_path)
    repo.record_usage("imported", "old-server-charge", cost_usd=6.4)
    budget = LocalBudget(repo, 2)
    budget.initialize(enabled=True)
    assert budget.snapshot()["spent_usd"] == 0
    budget.reserve("first", 0.5)
    repo.record_usage("mock", "local-test", cost_usd=0.13)
    assert budget.snapshot()["reserved_usd"] == 0.37
    budget.finish("first")
    assert budget.snapshot()["spent_usd"] == 0.13
    assert budget.snapshot()["remaining_usd"] == 1.87
    restarted = LocalBudget(repo, 2)
    restarted.initialize(enabled=True)
    assert restarted.snapshot() == budget.snapshot()


def test_uncertain_and_crashed_requests_keep_their_budget_holds(tmp_path):
    _, repo, _ = fixture(tmp_path)
    budget = LocalBudget(repo, 2)
    budget.initialize(enabled=True)
    budget.reserve("uncertain", 0.5)
    repo.record_usage("mock", "provider-reservation", cost_usd=0.1)
    budget.finish("uncertain", uncertain=True)
    assert budget.snapshot()["spent_usd"] == 0.1
    assert budget.snapshot()["reserved_usd"] == 0.4
    budget.reserve("crashed", 0.5)
    restarted = LocalBudget(repo, 2)
    restarted.initialize(enabled=True)
    assert restarted.snapshot()["reserved_usd"] == 0.9
    assert restarted.snapshot()["remaining_usd"] == 1
    restarted.reserve("later", 0.5)
    repo.record_usage("mock", "later-known-charge", cost_usd=0.3)
    restarted.finish("later")
    assert restarted.snapshot()["reserved_usd"] == 0.9
    assert restarted.snapshot()["spent_usd"] == 0.4
    assert restarted.snapshot()["remaining_usd"] == 0.7


def test_ceiling_never_increases_or_resets_on_restart(tmp_path):
    _, repo, _ = fixture(tmp_path)
    budget = LocalBudget(repo, 0.05)
    budget.initialize(enabled=True)
    repo.record_usage("mock", "test", cost_usd=0.04)
    new = LocalBudget(repo, 2)
    new.initialize(enabled=True)
    assert new.snapshot()["limit_usd"] == 0.05
    assert new.snapshot()["remaining_usd"] == 0.01
    with pytest.raises(LocalBudgetBlocked):
        new.reserve("no", 0.02)
    with pytest.raises(ValueError):
        LocalBudget(repo, 2.01)
    with pytest.raises(ValueError):
        LocalBudget(repo, float("nan"))


def test_api_rejects_worst_case_before_any_provider_call(tmp_path):
    settings, repo, delivery = fixture(tmp_path)
    calls = []
    reports = FakeReports()

    def builder(*args, **kwargs):
        calls.append("should-not-run")
        raise AssertionError("Unexpected paid call")

    async def exercise():
        client = await client_for(
            settings,
            repo,
            allow_paid=True,
            paid_budget_usd=0.01,
            reports=reports,
            article_builder=builder,
        )
        try:
            headers = await login(client)
            bootstrap = await (await client.get("/api/bootstrap")).json()
            assert bootstrap["paid_budget"]["limit_usd"] == 0.01
            assert bootstrap["paid_budget"]["spent_usd"] == 0
            response = await client.post(
                f"/api/items/news/{delivery['card_id']}/expand",
                headers=headers,
                json={"level": "article"},
            )
            assert response.status == 409
            assert (await response.json())["error"]["code"] == "local_budget_blocked"
            response = await client.post(
                "/api/items/video/1/expand", headers=headers, json={"level": "short"}
            )
            assert response.status == 409
            assert not calls and not reports.calls
            with repo.connection() as db:
                assert db.execute("SELECT COUNT(*) FROM web_jobs").fetchone()[0] == 0
                assert db.execute("SELECT COUNT(*) FROM usage_ledger").fetchone()[0] == 0
        finally:
            await client.close()

    asyncio.run(exercise())
