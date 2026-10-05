from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from app.config import load_source_registry
from app.db.repository import Repository
from app.research.detail_policy import DETAIL_PROMPT_VERSION

PROJECT_ROOT = Path(__file__).parents[1]


def sent_delivery(repository: Repository, *, chat_id: int = 123) -> dict[str, int | str]:
    repository.sync_sources(
        load_source_registry(PROJECT_ROOT / "tests/fixtures/config/sources.yaml")
    )
    delivery = repository.create_smoke_delivery(chat_id, "https://example.com/release")
    repository.mark_delivery_sent(int(delivery["delivery_id"]), 9001)
    return delivery


def valid_article_payload(*, expires_at: str = "2026-09-01T00:00:00+00:00") -> dict:
    return {
        "generation_key": "event:1:detail_v1",
        "schema_version": "detail_article_v1",
        "prompt_version": DETAIL_PROMPT_VERSION,
        "model": "fake/model",
        "provider_response_metadata": {"provider": "fake"},
        "fact_snapshot": {"facts": [{"text": "A verified fact."}]},
        "document_ids": [1, 2],
        "structured_output": {"headline": "A detailed article"},
        "rendered_html": "<b>A detailed article</b>",
        "source_urls": ["https://example.com/a", "https://independent.test/b"],
        "validation_status": "valid",
        "validation_errors": [],
        "generation_cost_usd": 0.005,
        "expires_at": expires_at,
    }


def test_enqueue_is_idempotent_and_claim_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / "details.sqlite3"
    repository = Repository(path)
    repository.migrate()
    delivery = sent_delivery(repository)

    first = repository.enqueue_detail_request(
        int(delivery["delivery_id"]),
        42,
        now="2026-08-25T08:00:00+00:00",
    )
    second = repository.enqueue_detail_request(
        int(delivery["delivery_id"]),
        42,
        now="2026-08-25T08:01:00+00:00",
    )

    assert first["created"] is True
    assert first["status"] == "queued"
    assert second["created"] is False
    assert second["request_id"] == first["request_id"]

    sync = repository.detail_requests_needing_keyboard_sync()
    assert sync[0]["display_status"] == "pending"
    assert repository.mark_detail_keyboard_synced(first["request_id"], "pending")
    assert repository.detail_requests_needing_keyboard_sync() == []

    claimed = repository.claim_next_detail_request(
        lease_minutes=10,
        now="2026-08-25T08:02:00+00:00",
    )
    assert claimed is not None
    assert claimed["request_id"] == first["request_id"]
    assert claimed["status"] == "running"
    assert claimed["stage"] == "planning"
    assert claimed["attempt_count"] == 1
    assert repository.detail_requests_needing_keyboard_sync() == []

    other_worker = Repository(path)
    assert (
        other_worker.claim_next_detail_request(
            lease_minutes=10,
            now="2026-08-25T08:03:00+00:00",
        )
        is None
    )


def test_parallel_enqueue_creates_one_request(tmp_path: Path) -> None:
    path = tmp_path / "parallel.sqlite3"
    repository = Repository(path)
    repository.migrate()
    delivery = sent_delivery(repository)
    delivery_id = int(delivery["delivery_id"])

    def enqueue() -> dict:
        return Repository(path).enqueue_detail_request(
            delivery_id,
            42,
            now="2026-08-25T08:00:00+00:00",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: enqueue(), range(2)))

    assert {result["request_id"] for result in results} == {1}
    assert sum(bool(result["created"]) for result in results) == 1
    assert repository.detail_status_counts()["queued"] == 1


def test_same_event_requests_are_single_flight_across_deliveries(tmp_path: Path) -> None:
    path = tmp_path / "single-flight.sqlite3"
    repository = Repository(path)
    repository.migrate()
    first_delivery = sent_delivery(repository)
    first = repository.enqueue_detail_request(
        int(first_delivery["delivery_id"]),
        42,
        now="2026-08-25T08:00:00+00:00",
    )
    with repository.connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO cards(
                event_id, card_version, prompt_version, structured_output_json,
                rendered_html, source_urls_json, primary_preview_url, validation_status
            ) VALUES (?, 2, 'test', '{}', '<b>Second</b>', '[]',
                      'https://example.com/release', 'valid')
            """,
            (first_delivery["event_id"],),
        )
        cursor = connection.execute(
            """
            INSERT INTO deliveries(
                card_id, chat_id, telegram_message_id, delivery_status, delivered_at
            ) VALUES (?, 123, 9002, 'sent', '2026-08-25T08:01:00+00:00')
            """,
            (cursor.lastrowid,),
        )
        second_delivery_id = int(cursor.lastrowid)
    second = repository.enqueue_detail_request(
        second_delivery_id,
        42,
        now="2026-08-25T08:01:00+00:00",
    )
    assert second["generation_key"] == first["generation_key"]

    claimed = repository.claim_next_detail_request(now="2026-08-25T08:02:00+00:00")

    assert claimed is not None
    assert Repository(path).claim_next_detail_request(now="2026-08-25T08:03:00+00:00") is None


def test_detail_claim_respects_global_worker_concurrency(tmp_path: Path) -> None:
    path = tmp_path / "global-concurrency.sqlite3"
    repository = Repository(path)
    repository.migrate()
    first_delivery = sent_delivery(repository)
    repository.enqueue_detail_request(
        int(first_delivery["delivery_id"]),
        42,
        now="2026-08-25T08:00:00+00:00",
    )
    second_delivery = repository.create_smoke_delivery(
        123,
        "https://example.com/another-release",
    )
    repository.mark_delivery_sent(int(second_delivery["delivery_id"]), 9002)
    repository.enqueue_detail_request(
        int(second_delivery["delivery_id"]),
        42,
        now="2026-08-25T08:01:00+00:00",
    )

    first = repository.claim_next_detail_request(now="2026-08-25T08:02:00+00:00")
    assert first is not None
    assert (
        Repository(path).claim_next_detail_request(
            max_concurrency=1,
            now="2026-08-25T08:03:00+00:00",
        )
        is None
    )
    second = Repository(path).claim_next_detail_request(
        max_concurrency=2,
        now="2026-08-25T08:03:00+00:00",
    )
    assert second is not None
    assert second["request_id"] != first["request_id"]


def test_stale_lease_is_recovered_without_duplicate_request(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "lease.sqlite3")
    repository.migrate()
    delivery = sent_delivery(repository)
    queued = repository.enqueue_detail_request(
        int(delivery["delivery_id"]),
        42,
        now="2026-08-25T08:00:00+00:00",
    )
    repository.claim_next_detail_request(
        lease_minutes=10,
        now="2026-08-25T08:01:00+00:00",
    )

    assert repository.recover_stale_detail_requests(now="2026-08-25T08:05:00+00:00") == 0
    assert repository.recover_stale_detail_requests(now="2026-08-25T08:12:00+00:00") == 1

    reclaimed = repository.claim_next_detail_request(now="2026-08-25T08:13:00+00:00")
    assert reclaimed is not None
    assert reclaimed["request_id"] == queued["request_id"]
    assert reclaimed["attempt_count"] == 2


def test_placeholder_claim_prevents_duplicate_send_and_recovers(tmp_path: Path) -> None:
    path = tmp_path / "placeholder.sqlite3"
    repository = Repository(path)
    repository.migrate()
    delivery = sent_delivery(repository)
    queued = repository.enqueue_detail_request(
        int(delivery["delivery_id"]),
        42,
        now="2026-08-25T08:00:00+00:00",
    )

    claimed = repository.claim_detail_placeholder_request(now="2026-08-25T08:01:00+00:00")
    assert claimed is not None
    assert claimed["request_id"] == queued["request_id"]
    assert claimed["parent_message_id"] == 9001
    assert (
        Repository(path).claim_detail_placeholder_request(now="2026-08-25T08:02:00+00:00") is None
    )

    reclaimed = Repository(path).claim_detail_placeholder_request(now="2026-08-25T08:04:00+00:00")
    assert reclaimed is not None
    assert reclaimed["request_id"] == queued["request_id"]
    assert repository.set_detail_placeholder_message_id(queued["request_id"], 9900)
    assert repository.claim_detail_placeholder_request(now="2026-08-25T08:05:00+00:00") is None


def test_article_completion_cache_and_delivery_retry(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "article.sqlite3")
    repository.migrate()
    delivery = sent_delivery(repository)
    request = repository.enqueue_detail_request(
        int(delivery["delivery_id"]),
        42,
        now="2026-08-25T08:00:00+00:00",
    )
    repository.claim_next_detail_request(now="2026-08-25T08:01:00+00:00")
    assert repository.update_detail_request_stage(
        request["request_id"],
        "writing",
        now="2026-08-25T08:02:00+00:00",
    )

    payload = valid_article_payload()
    payload["generation_key"] = request["generation_key"]
    article_id = repository.store_detail_article_and_complete(
        request["request_id"],
        now="2026-08-25T08:03:00+00:00",
        **payload,
    )
    ready = repository.get_detail_request(request["request_id"])
    assert ready is not None
    assert ready["status"] == "ready"
    assert ready["article_id"] == article_id
    assert repository.detail_requests_needing_keyboard_sync()[0]["display_status"] == "ready"
    assert not repository.mark_detail_keyboard_synced(request["request_id"], "pending")
    assert repository.mark_detail_keyboard_synced(request["request_id"], "ready")

    article = repository.cached_detailed_article(
        request["generation_key"],
        now="2026-08-25T08:04:00+00:00",
    )
    assert article is not None
    assert article["document_ids"] == [1, 2]
    assert article["fact_snapshot"]["facts"][0]["text"] == "A verified fact."
    assert repository.set_detail_placeholder_message_id(request["request_id"], 9900)

    delivery_claim = repository.claim_ready_detail_request_for_delivery(
        now="2026-08-25T08:04:30+00:00"
    )
    assert delivery_claim is not None
    assert delivery_claim["article_id"] == article_id
    assert (
        Repository(repository.database_path).claim_ready_detail_request_for_delivery(
            now="2026-08-25T08:05:00+00:00"
        )
        is None
    )
    assert repository.mark_detail_delivery_error(request["request_id"], "Telegram timeout")
    assert repository.get_detail_request(request["request_id"])["status"] == "ready"
    assert (
        repository.claim_ready_detail_request_for_delivery(now="2026-08-25T08:05:00+00:00")
        is not None
    )
    assert repository.mark_detail_request_sent(
        request["request_id"], now="2026-08-25T08:05:00+00:00"
    )
    assert repository.get_detail_request(request["request_id"])["status"] == "sent"
    assert not repository.mark_detail_delivery_error(
        request["request_id"], "late error from an expired delivery claimant"
    )
    assert repository.get_detail_request(request["request_id"])["status"] == "sent"

    with repository.connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO cards(
                event_id, card_version, prompt_version, structured_output_json,
                rendered_html, source_urls_json, primary_preview_url, validation_status
            ) VALUES (?, 2, 'test', '{}', '<b>Second</b>', '[]',
                      'https://example.com/release', 'valid')
            """,
            (delivery["event_id"],),
        )
        cursor = connection.execute(
            """
            INSERT INTO deliveries(
                card_id, chat_id, telegram_message_id, delivery_status, delivered_at
            ) VALUES (?, 123, 9002, 'sent', '2026-08-25T08:06:00+00:00')
            """,
            (cursor.lastrowid,),
        )
        second_delivery_id = int(cursor.lastrowid)
    cache_request = repository.enqueue_detail_request(
        second_delivery_id,
        42,
        max_daily_new=1,
        now="2026-08-25T08:07:00+00:00",
    )
    assert cache_request["cache_hit"] is True
    assert cache_request["status"] == "ready"
    assert cache_request["article_id"] == article_id
    assert (
        repository.claim_ready_detail_request_for_delivery(now="2026-08-25T08:08:00+00:00") is None
    )
    cached_placeholder = repository.claim_detail_placeholder_request(
        now="2026-08-25T08:08:00+00:00"
    )
    assert cached_placeholder is not None
    assert cached_placeholder["request_id"] == cache_request["request_id"]
    assert repository.set_detail_placeholder_message_id(cache_request["request_id"], 9901)
    assert (
        repository.claim_ready_detail_request_for_delivery(now="2026-08-25T08:09:00+00:00")
        is not None
    )


def test_daily_limit_does_not_count_repeat_or_cache_hit(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "limit.sqlite3")
    repository.migrate()
    first = sent_delivery(repository, chat_id=100)
    first_request = repository.enqueue_detail_request(
        int(first["delivery_id"]),
        42,
        max_daily_new=1,
        now="2026-08-25T08:00:00+00:00",
    )

    with repository.connection() as connection:
        card = connection.execute(
            "SELECT * FROM cards WHERE id = ?", (first["card_id"],)
        ).fetchone()
        cursor = connection.execute(
            """
            INSERT INTO cards(
                event_id, card_version, prompt_version, structured_output_json,
                rendered_html, source_urls_json, primary_preview_url, validation_status
            ) VALUES (?, 2, 'test', '{}', '<b>Second</b>', '[]',
                      'https://example.com/release', 'valid')
            """,
            (card["event_id"],),
        )
        second_card_id = int(cursor.lastrowid)
        cursor = connection.execute(
            """
            INSERT INTO deliveries(
                card_id, chat_id, telegram_message_id, delivery_status, delivered_at
            ) VALUES (?, 100, 9002, 'sent', '2026-08-25T08:01:00+00:00')
            """,
            (second_card_id,),
        )
        second_delivery_id = int(cursor.lastrowid)

    blocked = repository.enqueue_detail_request(
        second_delivery_id,
        42,
        generation_key="event:1:other",
        max_daily_new=1,
        now="2026-08-25T08:02:00+00:00",
    )
    assert blocked["status"] == "daily_limit"
    assert blocked["request_id"] is None
    assert (
        repository.enqueue_detail_request(
            int(first["delivery_id"]),
            42,
            max_daily_new=1,
            now="2026-08-25T08:03:00+00:00",
        )["request_id"]
        == first_request["request_id"]
    )


def test_failure_retry_and_detail_feedback_do_not_change_superreaction(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "feedback.sqlite3")
    repository.migrate()
    delivery = sent_delivery(repository)
    delivery_id = int(delivery["delivery_id"])
    repository.finalize_feedback(
        delivery_id,
        action="super_reaction_without_comment",
        reaction="superlike",
    )
    request = repository.enqueue_detail_request(
        delivery_id,
        42,
        now="2026-08-25T08:00:00+00:00",
    )
    repository.claim_next_detail_request(now="2026-08-25T08:01:00+00:00")
    assert repository.fail_detail_request(
        request["request_id"],
        status="insufficient_evidence",
        error="Only one full-text domain",
        now="2026-08-25T08:02:00+00:00",
    )
    retried = repository.retry_detail_request(
        request["request_id"], now="2026-08-25T08:03:00+00:00"
    )
    assert retried["status"] == "queued"
    repository.claim_next_detail_request(now="2026-08-25T08:04:00+00:00")
    payload = valid_article_payload()
    payload["generation_key"] = request["generation_key"]
    article_id = repository.store_detail_article_and_complete(
        request["request_id"],
        now="2026-08-25T08:05:00+00:00",
        **payload,
    )

    repository.record_detail_feedback(
        delivery_id,
        article_id,
        action="detail_useful",
        metadata={"latency_seconds": 65},
    )
    assert repository.current_reaction(delivery_id) == "superlike"
    with repository.connection() as connection:
        row = connection.execute(
            "SELECT action, metadata_json FROM feedback_events ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert row["action"] == "detail_useful"
    assert '"article_id"' in row["metadata_json"]


def test_terminal_placeholder_delivery_is_leased_and_idempotent(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "terminal-placeholder.sqlite3")
    repository.migrate()
    delivery = sent_delivery(repository)
    request = repository.enqueue_detail_request(
        int(delivery["delivery_id"]),
        42,
        now="2026-08-25T08:00:00+00:00",
    )
    assert repository.set_detail_placeholder_message_id(request["request_id"], 800)
    repository.claim_next_detail_request(now="2026-08-25T08:01:00+00:00")
    assert repository.fail_detail_request(
        request["request_id"],
        status="insufficient_evidence",
        error="Only one full-text domain",
        now="2026-08-25T08:02:00+00:00",
    )

    claimed = repository.claim_terminal_detail_placeholder(now="2026-08-25T08:03:00+00:00")
    assert claimed is not None
    assert claimed["placeholder_message_id"] == 800
    assert (
        Repository(repository.database_path).claim_terminal_detail_placeholder(
            now="2026-08-25T08:04:00+00:00"
        )
        is None
    )
    assert repository.mark_terminal_detail_placeholder_sent(
        request["request_id"],
        status="insufficient_evidence",
        claim_lease_until=claimed["placeholder_lease_until"],
    )
    assert repository.get_detail_request(request["request_id"])["stage"] == "complete"
    assert repository.claim_terminal_detail_placeholder(now="2026-08-25T08:06:00+00:00") is None


def test_only_sent_deliveries_can_enqueue(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "pending.sqlite3")
    repository.migrate()
    repository.sync_sources(
        load_source_registry(PROJECT_ROOT / "tests/fixtures/config/sources.yaml")
    )
    delivery = repository.create_smoke_delivery(123, "https://example.com/release")

    with pytest.raises(ValueError, match="has not been sent"):
        repository.enqueue_detail_request(int(delivery["delivery_id"]), 42)
