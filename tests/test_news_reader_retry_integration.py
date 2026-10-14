from datetime import UTC, datetime, timedelta
from urllib.error import HTTPError

import pytest

from app.config import Settings, SourceConfig, SourceRegistry
from app.db.repository import Repository
from app.details import research
from app.details.provider_guard import DetailProviderError, DurableDetailProvider
from app.telegram_bot.news_panel import ensure_news_panel


def test_repository_retry_reaches_builder_guard_without_replaying_recovered_nonce(
    tmp_path, monkeypatch
):
    settings = Settings(_env_file=None, allowed_user_id=123, data_dir=tmp_path)
    repo = Repository(settings.database_path)
    repo.migrate()
    repo.sync_sources(
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
    delivery = repo.create_smoke_delivery(123, "https://example.org/story")
    repo.mark_delivery_sent(delivery["delivery_id"], 700)
    panel = ensure_news_panel(repo, delivery["delivery_id"])
    now = datetime(2026, 10, 13, 9, tzinfo=UTC)
    stamp = now.isoformat()
    request = repo.enqueue_detail_request(
        delivery["delivery_id"],
        123,
        depth="expanded",
        reader_context=panel["reader_context"],
        now=stamp,
    )
    guards = []

    def capture_guard(*args, **kwargs):
        guard = DurableDetailProvider(*args, **kwargs)
        guards.append(guard)
        return guard

    class StopBeforeRetrieval(Exception):
        pass

    class ProbeCollector:
        def collect(self, *args, **kwargs):
            raise StopBeforeRetrieval

    monkeypatch.setattr(research, "DurableDetailProvider", capture_guard)

    def builder_guard(current):
        # Exercise the production builder's options, stopping before retrieval.
        # The real durable transport below performs only an offline fake call.
        with pytest.raises(StopBeforeRetrieval):
            research.build_detail_article(
                settings,
                repo,
                current["event_id"],
                depth=current["depth"],
                reader_context=current["reader_context"],
                research_service=ProbeCollector(),
            )
        return guards[-1]

    payload = {
        "model": "openai/gpt-4.1-mini",
        "messages": [{"role": "user", "content": "Fixed evidence."}],
        "max_tokens": 100,
    }
    calls = []

    def rejected(*args):
        calls.append("402")
        raise HTTPError("https://provider.example", 402, "Payment required", {}, None)

    def assert_rejected(current):
        with pytest.raises(DetailProviderError) as error:
            builder_guard(current).llm_transport(rejected, "article")(payload, "offline")
        assert error.value.status == "insufficient_credits"

    assert repo.claim_next_detail_request(now=stamp)["id"] == request["id"]
    assert_rejected(request)
    assert repo.fail_detail_request(request["id"], error="HTTP 402", now=stamp)
    first_retry = repo.retry_detail_request(request["id"], now=stamp)
    first_nonce = first_retry["reader_context"]["retry_authorization"]
    assert first_retry["reader_context"]["retry_rejected"] is True
    assert repo.claim_next_detail_request(lease_minutes=1, now=stamp)["id"] == request["id"]
    assert_rejected(first_retry)
    assert calls == ["402", "402"]

    later = (now + timedelta(minutes=2)).isoformat()
    assert repo.recover_stale_detail_requests(now=later) == 1
    recovered = repo.claim_next_detail_request(now=later)
    assert recovered["reader_context"]["retry_authorization"] == first_nonce
    assert_rejected(recovered)
    assert calls == ["402", "402"]

    assert repo.fail_detail_request(request["id"], error="HTTP 402", now=later)
    second_retry = repo.retry_detail_request(request["id"], now=later)
    assert second_retry["reader_context"]["retry_authorization"] != first_nonce
    response = {
        "id": "confirmed-offline-response",
        "choices": [{"message": {"content": "Grounded article."}}],
        "usage": {"cost": 0.0004, "is_byok": False},
    }

    def succeeded(*args):
        calls.append("success")
        return response

    transport = builder_guard(second_retry).llm_transport(succeeded, "article")
    assert transport(payload, "offline") == response
    assert transport(payload, "offline") == response
    assert calls == ["402", "402", "success"]
    with repo.connection() as db:
        assert db.execute("SELECT attempt_count FROM details_provider_calls").fetchone()[0] == 3
        costs = [row[0] for row in db.execute("SELECT cost_usd FROM usage_ledger ORDER BY id")]
    assert costs == [0, 0, 0.0004]
