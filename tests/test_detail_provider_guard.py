import json
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError

import pytest

from app.config import Settings, SourceConfig, SourceRegistry
from app.db.repository import Repository
from app.details.provider_guard import DetailProviderError, DurableDetailProvider
from app.llm.openrouter import OpenRouterResponseError
from app.research.web import ScrapeResponse, SearchResponse, WebResearchError


@pytest.fixture
def env(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path, details_max_cost_usd=0.02)
    repository = Repository(settings.database_path)
    repository.migrate()
    repository.sync_sources(
        SourceRegistry(
            version=1,
            sources=[
                SourceConfig(
                    id="fixture",
                    name="Fixture",
                    kind="rss",
                    locator="https://example.org/feed",
                    role="radar",
                    region="global",
                    language="en",
                )
            ],
        )
    )
    # The parent owns the additive migration; this also exercises the agreed
    # schema while its implementation is being integrated in parallel.
    with repository.connection() as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS details_provider_calls("
            "id INTEGER PRIMARY KEY,event_id INTEGER REFERENCES events(id),depth TEXT,"
            "baseline_fingerprint TEXT,call_key TEXT UNIQUE,provider TEXT,stage TEXT,"
            "status TEXT,usage_id INTEGER REFERENCES usage_ledger(id),reserved_usd REAL,"
            "response_json TEXT,error_json TEXT,created_at TEXT,updated_at TEXT,"
            "attempt_count INTEGER NOT NULL DEFAULT 1,retry_authorization TEXT)"
        )
    event = repository.create_smoke_delivery(123, "https://example.org/story")["event_id"]
    return settings, repository, event


def guard(env, **kwargs):
    settings, repository, event = env
    return DurableDetailProvider(settings, repository, event, "expanded", "baseline-v1", **kwargs)


def payload(text="A bounded evidence package.", *, model="openai/gpt-4.1-mini"):
    return {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": 100,
    }


def response(cost=0.0004):
    return {
        "id": "confirmed-response",
        "model": "openai/gpt-4.1-mini",
        "choices": [{"message": {"content": '{"facts":[]}'}}],
        "usage": {"cost": cost, "is_byok": False},
    }


def test_reservation_commits_before_network_and_completed_response_reuses_cache(env):
    _, repo, _ = env
    calls = []

    def network(request, key):
        calls.append(request)
        with repo.connection() as db:
            db.execute("BEGIN IMMEDIATE")  # Would fail if dispatch held another write lock.
            call = db.execute("SELECT * FROM details_provider_calls").fetchone()
            usage = db.execute("SELECT * FROM usage_ledger").fetchone()
            assert call["status"] == "preparing" and usage["cost_usd"] > 0
        assert request["provider"]["max_price"] == {"prompt": 0.5, "completion": 2.0}
        return response()

    original = payload()
    first = guard(env).llm_transport(network, "fact_snapshot")(original, "offline-key")
    assert "provider" not in original
    second = guard(env).llm_transport(network, "fact_snapshot")(original, "changed-offline-key")
    assert first == second == response() and len(calls) == 1
    with repo.connection() as db:
        assert db.execute("SELECT count(*) FROM usage_ledger").fetchone()[0] == 1
        assert db.execute("SELECT cost_usd FROM usage_ledger").fetchone()[0] == 0.0004
        row = db.execute("SELECT * FROM details_provider_calls").fetchone()
        assert row["status"] == "completed"
        assert "offline-key" not in json.dumps(dict(row))


def test_uncertain_timeout_retains_reservation_and_never_replays_after_reconstruction(env):
    _, repo, _ = env
    calls = []

    def timeout(*args):
        calls.append(1)
        raise TimeoutError("Do not persist this private diagnostic")

    with pytest.raises(DetailProviderError) as first:
        guard(env).llm_transport(timeout, "fact_snapshot")(payload(), "offline")
    assert first.value.status == "uncertain" and first.value.cost_usd > 0
    with pytest.raises(DetailProviderError) as again:
        guard(env, retry_rejected=True, retry_authorization="explicit-retry").llm_transport(
            timeout, "fact_snapshot"
        )(payload(), "offline")
    assert again.value.status == "uncertain" and calls == [1]
    with repo.connection() as db:
        row = db.execute("SELECT * FROM details_provider_calls").fetchone()
        assert row["status"] == "uncertain"
        assert "private diagnostic" not in row["error_json"]
        assert db.execute("SELECT cost_usd FROM usage_ledger").fetchone()[0] == row["reserved_usd"]


@pytest.mark.parametrize("change", ["model", "payload", "nonce", "stage", "provider"])
def test_unknown_logical_job_blocks_all_fresh_dispatches_despite_changed_inputs(env, change):
    _, repo, _ = env

    def lost(*args):
        raise TimeoutError("Unknown dispatch result")

    with pytest.raises(DetailProviderError):
        guard(env).llm_transport(lost, "fact_snapshot")(payload(), "offline")
    with repo.connection() as db:
        original_usage = [dict(row) for row in db.execute("SELECT * FROM usage_ledger")]
        original_calls = [dict(row) for row in db.execute("SELECT * FROM details_provider_calls")]
    renewed = guard(env, retry_rejected=True, retry_authorization="new-user-retry")
    request = payload()
    stage = "fact_snapshot"
    if change == "model":
        request["model"] = "openai/gpt-5.6-luna"
    elif change == "payload":
        request["messages"][0]["content"] = "Changed document or prompt content"
    elif change == "nonce":
        request["messages"][0]["content"] += " Nonce should not be placed in a prompt."
    elif change == "stage":
        stage = "article"
    with pytest.raises(DetailProviderError) as blocked:
        if change == "provider":
            renewed.run(
                "another/provider",
                "search",
                {"query": "new probe"},
                0.001,
                lambda: pytest.fail("Do not dispatch another provider"),
            )
        else:
            renewed.llm_transport(lambda *args: pytest.fail("Must not dispatch"), stage)(
                request, "offline"
            )
    assert blocked.value.status == "uncertain"
    with repo.connection() as db:
        assert [dict(row) for row in db.execute("SELECT * FROM usage_ledger")] == original_usage
        assert [
            dict(row) for row in db.execute("SELECT * FROM details_provider_calls")
        ] == original_calls


def test_confirmed_cached_response_remains_free_to_read_when_another_stage_is_unknown(env):
    _, repo, _ = env
    completed = guard(env).llm_transport(lambda *args: response(), "fact_snapshot")
    completed(payload(), "offline")
    with pytest.raises(DetailProviderError):
        guard(env).run(
            "fixture",
            "search",
            {"query": "lost"},
            0.001,
            lambda: (_ for _ in ()).throw(TimeoutError("unknown")),
        )
    with repo.connection() as db:
        count = db.execute("SELECT count(*) FROM usage_ledger").fetchone()[0]
    assert (
        guard(env).llm_transport(lambda *args: pytest.fail("Only cache"), "fact_snapshot")(
            payload(), "offline"
        )
        == response()
    )
    with repo.connection() as db:
        assert db.execute("SELECT count(*) FROM usage_ledger").fetchone()[0] == count


def test_preparing_call_after_process_loss_is_not_automatically_dispatched(env):
    _, repo, _ = env
    original = guard(env).llm_transport(lambda *args: response(), "article")
    original(payload(), "offline")
    with repo.connection() as db:
        db.execute("UPDATE details_provider_calls SET status='preparing',response_json=NULL")
    with pytest.raises(DetailProviderError) as error:
        guard(env).llm_transport(lambda *args: pytest.fail("Do not dispatch"), "article")(
            payload(), "offline"
        )
    assert error.value.status == "uncertain"


def test_http402_needs_fresh_explicit_authorization_and_settles_each_rejection_to_zero(env):
    _, repo, _ = env
    calls = []

    def rejected(*args):
        calls.append(1)
        raise HTTPError("https://provider.example", 402, "Payment required", {}, None)

    request = payload()
    with pytest.raises(DetailProviderError) as error:
        guard(env).llm_transport(rejected, "article")(request, "offline")
    assert error.value.status == "insufficient_credits" and error.value.cost_usd == 0
    for retry in [guard(env), guard(env, retry_rejected=True)]:
        with pytest.raises(DetailProviderError):
            retry.llm_transport(rejected, "article")(request, "offline")
    explicit = guard(env, retry_rejected=True, retry_authorization="click-one")
    with pytest.raises(DetailProviderError):
        explicit.llm_transport(rejected, "article")(request, "offline")
    with pytest.raises(DetailProviderError):
        explicit.llm_transport(rejected, "article")(request, "offline")
    # A stale worker recreated after that rejection still carries click-one.
    with pytest.raises(DetailProviderError):
        guard(env, retry_rejected=True, retry_authorization="click-one").llm_transport(
            rejected, "article"
        )(request, "offline")
    assert calls == [1, 1]
    fresh = guard(env, retry_rejected=True, retry_authorization="click-two")
    assert (
        fresh.llm_transport(lambda *args: response(), "article")(request, "offline") == response()
    )
    with repo.connection() as db:
        assert db.execute("SELECT attempt_count FROM details_provider_calls").fetchone()[0] == 3
        costs = [r[0] for r in db.execute("SELECT cost_usd FROM usage_ledger ORDER BY id")]
    assert costs == [0, 0, 0.0004]


def test_http402_wrapped_as_transport_cause_remains_known_rejection(env):
    def masked(*args):
        try:
            raise HTTPError("https://provider.example", 402, "Payment required", {}, None)
        except HTTPError as error:
            raise RuntimeError("A sanitized transport failure") from error

    with pytest.raises(DetailProviderError) as error:
        guard(env).llm_transport(masked, "article")(payload(), "offline")
    assert error.value.status == "insufficient_credits" and error.value.http_status == 402


def test_confirmed_failed_response_keeps_accounted_receipt_without_automatic_replay(env):
    _, repo, _ = env
    calls = []

    def invalid(*args):
        calls.append(1)
        raise OpenRouterResponseError(
            "Returned payload is invalid",
            model="openai/gpt-4.1-mini",
            cost_usd=0.0008,
            prompt_tokens=100,
            completion_tokens=40,
            response_id="confirmed-invalid",
            billing_metadata={"cost_estimated": False, "billing_basis": "openrouter_reported"},
        )

    for _ in range(2):
        with pytest.raises(DetailProviderError) as error:
            guard(env).llm_transport(invalid, "article")(payload(), "offline")
        assert error.value.status == "failed" and error.value.cost_usd == 0.0008
    assert calls == [1]
    with repo.connection() as db:
        usage = db.execute("SELECT * FROM usage_ledger").fetchone()
        assert usage["cost_usd"] == 0.0008
        assert json.loads(usage["metadata_json"])["response_id"] == "confirmed-invalid"
        assert db.execute("SELECT status FROM details_provider_calls").fetchone()[0] == "failed"


@pytest.mark.parametrize("reason", ["request", "monthly", "unknown_price"])
def test_complete_payload_reservation_blocks_dispatch_before_spending(env, reason):
    settings, repo, _ = env
    request = payload("Heavy evidence " * 10000) if reason == "request" else payload()
    if reason == "monthly":
        repo.record_usage("fixture", "existing", cost_usd=settings.budget_hard_stop_usd - 0.00001)
    if reason == "unknown_price":
        request["model"] = "unknown/unpriced"
    with pytest.raises(DetailProviderError) as error:
        guard(env).llm_transport(lambda *args: pytest.fail("Must not dispatch"), "article")(
            request, "offline"
        )
    assert error.value.status == "budget_blocked"
    with repo.connection() as db:
        assert db.execute("SELECT count(*) FROM details_provider_calls").fetchone()[0] == 0


def test_job_budget_includes_earlier_calls_and_concurrent_pending_reservations(env):
    first = guard(env)
    for stage in ("search", "scrape"):
        first.run("fixture", stage, {"stage": stage}, 0.009, lambda: {"ok": True})
    assert first.total_cost_usd == pytest.approx(0.018)
    with pytest.raises(DetailProviderError) as error:
        guard(env).run("fixture", "article", {}, 0.009, lambda: pytest.fail("No dispatch"))
    assert error.value.status == "budget_blocked"


def test_new_depth_budget_is_independent_of_legacy_cap_and_preserves_explicit_price_ceiling(env):
    settings, repo, event = env
    assert settings.details_max_cost_usd == 0.02
    provider = DurableDetailProvider(
        settings, repo, event, "expanded", "larger-evidence", request_budget_usd=0.05
    )
    request = payload("Whole evidence package. " * 1500, model="openai/gpt-5.6-luna")
    request["provider"] = {"max_price": {"prompt": 0.5, "completion": 2.0}}
    request["max_tokens"] = 5000

    def transport(actual, key):
        assert actual["provider"]["max_price"] == {"prompt": 0.5, "completion": 2.0}
        return response()

    assert provider.llm_transport(transport, "fact_snapshot")(request, "offline") == response()
    with repo.connection() as db:
        reserved = db.execute("SELECT reserved_usd FROM details_provider_calls").fetchone()[0]
    assert 0.02 < reserved < 0.05
    with pytest.raises(ValueError, match="depth cap"):
        DurableDetailProvider(
            settings, repo, event, "expanded", "invalid-budget", request_budget_usd=0.08
        )


def test_web_responses_roundtrip_full_text_and_facade_does_not_charge_twice(env):
    _, repo, _ = env
    provider = guard(env)
    calls = []
    body = "Полный сохранённый текст страницы. " * 2000

    class Client:
        def search(self, query, **kwargs):
            calls.append("search")
            return SearchResponse(query, [{"url": "https://example.org/a"}], 0.00009)

        def scrape(self, url):
            calls.append("scrape")
            return ScrapeResponse(url, body, {"title": "Article"}, 0.0009)

    client = provider.web_client(Client())
    search = client.search("bounded query", limit=3)
    assert search.results[0]["url"] == "https://example.org/a"
    facade = provider.accounting_repository()
    facade.record_usage("monid/context.dev", "details.web.search", cost_usd=0.00009)
    document = client.scrape("https://example.org/a")
    assert document.markdown == body
    facade.record_usage(
        "monid/context.dev",
        "details.web.scrape.markdown",
        cost_usd=0.0009,
        metadata={"event_id": env[2]},
    )
    cached = guard(env).web_client(Client()).scrape("https://example.org/a")
    assert cached.markdown == document.markdown and cached.url == document.url
    assert cached.metadata == document.metadata and cached.cost_usd == 0
    assert calls == ["search", "scrape"]
    with repo.connection() as db:
        rows = db.execute("SELECT * FROM usage_ledger").fetchall()
    assert len(rows) == 2 and sum(row["cost_usd"] for row in rows) == pytest.approx(0.00099)
    for row in rows:
        metadata = json.loads(row["metadata_json"])
        assert metadata["cost_estimated"] is True
        assert metadata["billing_basis"] == "configured_rate_estimate"
    assert (
        json.loads(rows[1]["metadata_json"])["validation_operation"]
        == "details.web.scrape.markdown"
    )


def test_byok_cost_and_validation_metadata_preserve_true_accounting(env):
    _, repo, _ = env
    provider = guard(env)
    value = response()
    value["usage"] = {
        "cost": 0,
        "is_byok": True,
        "cost_details": {"upstream_inference_cost": 0.0007},
    }
    provider.llm_transport(lambda *args: value, "fact_snapshot")(payload(), "offline")
    provider.accounting_repository().record_usage(
        "openrouter",
        "details.fact_snapshot.invalid",
        cost_usd=0,
        metadata={"errors": ["Insufficient cited evidence"]},
    )
    with repo.connection() as db:
        row = db.execute("SELECT * FROM usage_ledger").fetchone()
        assert db.execute("SELECT count(*) FROM usage_ledger").fetchone()[0] == 1
    assert row["cost_usd"] == 0.0007
    metadata = json.loads(row["metadata_json"])
    assert metadata["billing_basis"] == "openrouter_plus_byok_upstream"
    assert metadata["cost_estimated"] is False
    assert metadata["validation_metadata"]["errors"] == ["Insufficient cited evidence"]


def test_same_pending_call_is_not_double_dispatched_by_another_worker(env):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def network(*args):
        calls.append(1)
        entered.set()
        assert release.wait(5)
        return response()

    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(guard(env).llm_transport(network, "article"), payload(), "offline")
        assert entered.wait(5)
        try:
            with pytest.raises(DetailProviderError) as duplicate:
                guard(env).llm_transport(network, "article")(payload(), "offline")
            assert duplicate.value.status == "uncertain"
        finally:
            release.set()
        assert first.result(timeout=5) == response()
    assert calls == [1]


def test_depth_and_baseline_are_part_of_response_identity(env):
    settings, repo, event = env
    calls = []

    def network(*args):
        calls.append(1)
        return response()

    for depth, baseline in [("expanded", "one"), ("article", "one"), ("expanded", "two")]:
        DurableDetailProvider(settings, repo, event, depth, baseline).llm_transport(
            network, "article"
        )(payload(), "offline")
    assert calls == [1, 1, 1]


def test_response_credentials_are_redacted_without_truncating_evidence(env):
    _, repo, _ = env
    value = response()
    value["authorization"] = "Bearer fake-token-value"
    value["choices"][0]["message"]["content"] = "A full result " * 2000 + " sk-or-v1-fake-key"
    result = guard(env).llm_transport(lambda *args: value, "article")(payload(), "offline")
    assert "authorization" not in result
    assert len(result["choices"][0]["message"]["content"]) > 16000
    with repo.connection() as db:
        stored = db.execute("SELECT response_json FROM details_provider_calls").fetchone()[0]
    assert "fake-key" not in stored and "fake-token-value" not in stored


@pytest.mark.parametrize("failure", ["timeout", "403"])
def test_failed_web_url_is_not_repaid_and_distinct_url_can_proceed(env, failure):
    _, repo, _ = env
    provider = guard(env)
    calls = []

    class Client:
        def scrape(self, url):
            calls.append(url)
            if url.endswith("/a"):
                if failure == "403":
                    raise HTTPError(url, 403, "Forbidden", {}, None)
                raise TimeoutError("Unknown receipt")
            return ScrapeResponse(url, "A successfully read full page.", {}, 0.0009)

    client = provider.web_client(Client())
    with pytest.raises(WebResearchError) as first:
        client.scrape("https://example.org/a")
    assert first.value.status == "uncertain" and first.value.cost_usd == 0.0009
    assert isinstance(first.value.__cause__, DetailProviderError)
    with pytest.raises(WebResearchError) as repeat:
        client.scrape("https://example.org/a")
    assert repeat.value.cost_usd == 0 and repeat.value.retained_cost_usd == 0.0009
    assert client.scrape("https://example.org/b").markdown == "A successfully read full page."
    assert calls == ["https://example.org/a", "https://example.org/b"]
    assert provider.total_cost_usd == pytest.approx(0.0018)
    assert provider.incremental_cost_usd == pytest.approx(0.0018)
    with repo.connection() as db:
        assert db.execute("SELECT count(*) FROM usage_ledger").fetchone()[0] == 2


def test_alternative_web_url_reservation_still_counts_failed_url_against_budget(env):
    _, repo, _ = env
    provider = guard(env, request_budget_usd=0.001)
    calls = []

    class Client:
        def scrape(self, url):
            calls.append(url)
            raise TimeoutError("Unconfirmed result")

    client = provider.web_client(Client())
    with pytest.raises(WebResearchError):
        client.scrape("https://example.org/a")
    with pytest.raises(WebResearchError) as blocked:
        client.scrape("https://example.org/b")
    assert blocked.value.status == "budget_blocked" and blocked.value.cost_usd == 0
    assert calls == ["https://example.org/a"]
    assert provider.total_cost_usd == pytest.approx(0.0009)
    with repo.connection() as db:
        assert db.execute("SELECT count(*) FROM usage_ledger").fetchone()[0] == 1


def test_article_reuses_expanded_web_receipts_with_zero_incremental_charge(env):
    settings, repo, event = env
    calls = []

    class Client:
        def search(self, query, **kwargs):
            calls.append(("search", query))
            return SearchResponse(query, [{"url": "https://example.org/a"}], 0.00009)

        def scrape(self, url):
            calls.append(("scrape", url))
            return ScrapeResponse(url, "Full shared evidence page.", {"title": "Same page"}, 0.0009)

    expanded = guard(env)
    first = expanded.web_client(Client())
    first.search("shared query", num_results=2)
    first.scrape("https://example.org/a")
    article = DurableDetailProvider(
        settings, repo, event, "article", "baseline-v1", request_budget_usd=0.08
    )
    second = article.web_client(Client())
    assert second.search("shared query", limit=2).cost_usd == 0
    page = second.scrape("https://example.org/a")
    assert page.markdown == "Full shared evidence page." and page.cost_usd == 0
    assert calls == [("search", "shared query"), ("scrape", "https://example.org/a")]
    assert article.incremental_cost_usd == 0
    assert article.total_cost_usd == expanded.total_cost_usd == pytest.approx(0.00099)
    with repo.connection() as db:
        assert db.execute("SELECT count(*) FROM details_provider_calls").fetchone()[0] == 2
        assert db.execute("SELECT count(*) FROM usage_ledger").fetchone()[0] == 2
        encoded = json.loads(
            db.execute(
                "SELECT response_json FROM details_provider_calls WHERE stage='scrape'"
            ).fetchone()[0]
        )
    assert encoded["cost_usd"] == 0.0009  # The original receipt is not re-stamped as free.


@pytest.mark.parametrize("operation", ["search", "scrape"])
def test_unknown_web_operation_stays_blocked_across_depths(env, operation):
    settings, repo, event = env
    calls = []

    class Client:
        def search(self, query, **kwargs):
            calls.append(query)
            raise TimeoutError("Unknown search result")

        def scrape(self, url):
            calls.append(url)
            raise TimeoutError("Unknown page result")

    first = guard(env)
    arguments = "https://example.org/a" if operation == "scrape" else "shared query"
    with pytest.raises(WebResearchError):
        getattr(first.web_client(Client()), operation)(arguments)
    article = DurableDetailProvider(
        settings, repo, event, "article", "baseline-v1", request_budget_usd=0.08
    )
    with pytest.raises(WebResearchError) as replay:
        getattr(article.web_client(Client()), operation)(arguments)
    assert replay.value.status == "uncertain" and replay.value.cost_usd == 0
    assert replay.value.retained_cost_usd == pytest.approx(0.0009)
    assert calls == [arguments] and article.incremental_cost_usd == 0
    assert article.total_cost_usd == pytest.approx(0.0009)
