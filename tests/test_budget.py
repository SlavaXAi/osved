import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.ops.budget import (
    BudgetState,
    SearchBudgetExceeded,
    SearchBudgetTransport,
    evaluate_budget,
)


def test_budget_states() -> None:
    assert evaluate_budget(0, 20, 30).state == BudgetState.OK
    assert evaluate_budget(20, 20, 30).state == BudgetState.WARNING
    assert evaluate_budget(30, 20, 30).state == BudgetState.HARD_STOP
    assert not evaluate_budget(31, 20, 30).paid_operations_allowed


def request(*, search=False):
    value = {
        "model": "fixture/cheap",
        "messages": [{"role": "user", "content": "Find relevant recent technology news"}],
        "max_tokens": 1000,
    }
    if search:
        value["tools"] = [
            {
                "type": "openrouter:web_search",
                "parameters": {
                    "engine": "native",
                    "mode": "deep",
                    "max_results": 25,
                    "max_uses": 20,
                },
            }
        ]
    return value


def test_search_transport_caps_all_paid_dimensions_without_mutating_request():
    payload = request(search=True)
    payload["max_tokens"] = 12000
    payload["stop_server_tools_when"] = [{"unknown": "unbounded"}]
    original = copy.deepcopy(payload)
    seen = []
    guard = SearchBudgetTransport(
        1, transport=lambda p, k: seen.append(p) or {"usage": {"cost": 0.01}}
    )
    guard(payload, "secret-only-in-transport")
    bounded = seen[0]
    assert payload == original
    assert "stop_server_tools_when" not in bounded
    assert bounded["max_tokens"] == 3000
    assert bounded["max_tool_calls"] == 1
    assert bounded["provider"]["require_parameters"] is True
    assert bounded["provider"]["max_price"] == {"prompt": 0.25, "completion": 1.25}
    assert bounded["tools"][0]["parameters"] == {
        "engine": "exa",
        "mode": "auto",
        "max_uses": 1,
        "max_results": 3,
        "max_total_results": 3,
        "max_characters": 1200,
    }
    assert guard.snapshot()["reported_cost_usd"] == 0.01
    assert "secret" not in str(guard.snapshot())


def test_zero_budget_does_not_call_transport():
    calls = []
    guard = SearchBudgetTransport(0, transport=lambda p, k: calls.append(p))
    with pytest.raises(SearchBudgetExceeded):
        guard(request(), "unused")
    assert calls == []
    assert guard.snapshot()["blocked_requests"] == 1


def test_scheduled_context_compaction_preserves_profile_and_feedback_meaning():
    profile = {"topics": [f"topic-{i}" for i in range(7)], "goals": ["Useful awareness"]}
    context = {
        "SEARCH_WINDOW_START": "2026-09-01",
        "LENS_ID": "startups",
        "INTEREST_PROFILE": profile,
        "RECENT_FIXED_SIGNALS": [{"kind": "rss", "signal": "x" * 400}] * 40,
        "SEEN_EVENTS": [{"headline": "x" * 180}] * 120,
        "FEEDBACK_EXAMPLES": [
            {
                "headline": "h" * 180,
                "summary": "s" * 420,
                "comment": "c" * 360,
                "reaction": "superdislike",
                "weight": -2,
            }
        ]
        * 12,
    }
    payload = {
        **request(search=True),
        "messages": [
            {"role": "user", "content": json.dumps(context)},
        ],
    }
    bounded, estimate = SearchBudgetTransport(0.06)._prepare(payload)
    actual = json.loads(bounded["messages"][0]["content"])
    assert actual["INTEREST_PROFILE"] == profile
    assert len(actual["RECENT_FIXED_SIGNALS"]) == 12
    assert len(actual["SEEN_EVENTS"]) == 16
    assert len(actual["FEEDBACK_EXAMPLES"]) == 4
    assert actual["FEEDBACK_EXAMPLES"][0]["weight"] == -2
    assert actual["FEEDBACK_EXAMPLES"][0]["comment"] == "c" * 300
    assert estimate < 0.06


def test_transport_failure_keeps_unknown_charge_reservation():
    def failed(payload, key):
        raise TimeoutError("provider response was lost")

    guard = SearchBudgetTransport(0.0025, transport=failed)
    with pytest.raises(TimeoutError):
        guard(request(), "unused")
    assert guard.snapshot()["unknown_cost_reserved_usd"] > 0
    assert guard.snapshot()["inflight_reserved_usd"] == 0
    with pytest.raises(SearchBudgetExceeded):
        guard(request(), "unused")


@pytest.mark.parametrize("cost", [None, "oops", -1, float("nan"), True])
def test_unusable_usage_is_not_mistaken_for_a_free_call(cost):
    guard = SearchBudgetTransport(1, transport=lambda p, k: {"usage": {"cost": cost}})
    guard(request(), "unused")
    assert guard.snapshot()["unknown_cost_reserved_usd"] > 0


def test_reported_overrun_stops_all_further_calls():
    guard = SearchBudgetTransport(0.01, transport=lambda p, k: {"usage": {"cost": 0.02}})
    guard(request(), "unused")
    assert guard.snapshot()["reported_overrun"] is True
    with pytest.raises(SearchBudgetExceeded):
        guard(request(), "unused")
    assert guard.snapshot()["requests"] == 1


@pytest.mark.parametrize(
    "change",
    [
        {"tools": [{"type": "openrouter:subagent"}]},
        {"plugins": [{"id": "web"}]},
        {"stream": True},
        {"n": 2},
        {"messages": [{"content": "a" * 130_000}]},
        {"provider": {"max_price": {"prompt": -1}}},
    ],
)
def test_unbounded_payload_is_rejected_before_transport(change):
    calls = []
    guard = SearchBudgetTransport(1, transport=lambda p, k: calls.append(p))
    with pytest.raises(SearchBudgetExceeded):
        guard({**request(), **change}, "unused")
    assert calls == []


def test_parallel_requests_reserve_before_transport():
    started = threading.Event()
    release = threading.Event()

    def blocking(payload, key):
        started.set()
        assert release.wait(3)
        return {"usage": {"cost": 0.001}}

    guard = SearchBudgetTransport(0.0025, transport=blocking)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(guard, request(), "unused")
        assert started.wait(2)
        second = pool.submit(guard, request(), "unused")
        assert guard.snapshot()["requests"] == 1
        release.set()
        with pytest.raises(SearchBudgetExceeded):
            second.result()
        first.result()
    assert guard.snapshot()["requests"] == 1
    assert guard.snapshot()["inflight_reserved_usd"] == 0


def test_second_lens_waits_for_released_reserve_instead_of_being_dropped():
    started = threading.Event()
    release = threading.Event()
    payload = request(search=True)
    reservation = SearchBudgetTransport(1)._prepare(payload)[1]
    calls = []

    def bounded_transport(payload, key):
        calls.append(True)
        if len(calls) == 1:
            started.set()
            assert release.wait(3)
        return {"usage": {"cost": reservation / 10}}

    guard = SearchBudgetTransport(reservation * 1.5, transport=bounded_transport)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(guard, payload, "unused")
        assert started.wait(2)
        second = pool.submit(guard, payload, "unused")
        release.set()
        first.result()
        second.result()
    assert len(calls) == 2
    assert guard.snapshot()["requests"] == 2
    assert guard.snapshot()["blocked_requests"] == 0
