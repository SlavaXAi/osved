from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from app.collectors.browser_queue import BrowserTaskError, BrowserTaskQueue, sanitize_snapshot


@pytest.fixture
def queue(tmp_path):
    queue = BrowserTaskQueue(tmp_path / "browser.sqlite")
    queue.initialize()
    return queue


def snapshot(post_id="1234567890"):
    return {
        "feed_mode": "for_you",
        "posts": [
            {
                "id": post_id,
                "author": "karpathy",
                "text": "A new open-source agent.",
                "published_at": "2026-09-07T08:00:00Z",
                "url": "https://evil.example/overwrite",
                "links": [
                    "https://github.com/example/agent?utm_source=x#about",
                    "javascript:alert(1)",
                    "https://user:secret@example.com/",
                    "https://example.com/paper?token=secret",
                    "https://example.com/index.php?id=42&utm_source=x",
                ],
            }
        ],
    }


def test_sanitized_snapshot_never_keeps_credentials_or_arbitrary_post_url():
    data = snapshot()
    data["posts"][0]["text"] += " auth_token=abc123 ct0=private"
    result = sanitize_snapshot(data)
    encoded = json.dumps(result)
    assert "abc123" not in encoded
    assert "private" not in encoded
    assert "secret" not in encoded
    assert "evil.example" not in encoded
    assert "javascript" not in encoded
    assert result["posts"][0]["url"] == "https://x.com/karpathy/status/1234567890"
    assert result["posts"][0]["links"] == [
        "https://github.com/example/agent",
        "https://example.com/paper",
        "https://example.com/index.php?id=42",
    ]


@pytest.mark.parametrize("addition", [{"cookies": "secret"}, {"headers": {}}, {"execute": "js"}])
def test_unrecognized_result_fields_rejected(addition):
    with pytest.raises(BrowserTaskError, match="unexpected_result_fields"):
        sanitize_snapshot({**snapshot(), **addition})


def test_post_field_credentials_and_unbounded_payload_rejected():
    data = snapshot()
    data["posts"][0]["cookie"] = "secret"
    with pytest.raises(BrowserTaskError, match="unexpected_post_fields"):
        sanitize_snapshot(data)
    data = snapshot()
    data["posts"] *= 81
    with pytest.raises(BrowserTaskError, match="invalid_posts"):
        sanitize_snapshot(data)


def test_queue_claim_stage_restart_commit_replay(queue):
    task_id = queue.enqueue("x-for-you", task_key="morning")
    assert queue.enqueue("x-for-you", task_key="morning") == task_id
    claim = queue.claim()
    assert claim["id"] == task_id
    assert queue.claim() is None
    assert queue.stage(task_id, claim["claim_token"], snapshot())
    restarted = BrowserTaskQueue(queue.database_path)
    assert restarted.ready_results("x-for-you")[0]["result"]["posts"][0]["author"] == "karpathy"
    assert not restarted.stage(task_id, claim["claim_token"], snapshot())
    with pytest.raises(BrowserTaskError, match="result_conflict"):
        restarted.stage(task_id, claim["claim_token"], snapshot("9999999999"))
    restarted.mark_committed(task_id)
    restarted = BrowserTaskQueue(queue.database_path)
    assert restarted.ready_results()[0]["status"] == "committed"
    restarted.complete(task_id)
    restarted.complete(task_id)
    assert restarted.ready_results() == []
    assert not restarted.stage(task_id, claim["claim_token"], snapshot())


def test_claim_token_is_required(queue):
    task_id = queue.enqueue("x-for-you", task_key="morning")
    queue.claim()
    with pytest.raises(BrowserTaskError, match="invalid_claim"):
        queue.stage(task_id, "wrong", snapshot())
    with pytest.raises(BrowserTaskError, match="invalid_claim"):
        queue.fail(task_id, "wrong", "empty_feed")


def test_stale_lease_recovers_with_new_token_and_bounded_attempts(tmp_path):
    now = [1000.0]
    queue = BrowserTaskQueue(tmp_path / "browser.sqlite", clock=lambda: now[0])
    queue.initialize()
    task_id = queue.enqueue("x", task_key="a", max_attempts=2)
    first = queue.claim(lease_seconds=10)
    now[0] += 11
    second = queue.claim(lease_seconds=10)
    assert second["id"] == task_id
    assert second["claim_token"] != first["claim_token"]
    with pytest.raises(BrowserTaskError, match="invalid_claim"):
        queue.stage(task_id, first["claim_token"], snapshot())
    now[0] += 11
    assert queue.claim() is None
    assert queue.task_status(task_id)["status"] == "failed"
    assert queue.task_status(task_id)["attempt_count"] == 2


def test_expired_task_not_executed_and_staged_result_survives_ttl(tmp_path):
    now = [1000.0]
    queue = BrowserTaskQueue(tmp_path / "browser.sqlite", clock=lambda: now[0])
    queue.initialize()
    task_id = queue.enqueue("x", task_key="expired", ttl_seconds=10)
    now[0] += 11
    assert queue.claim() is None
    assert queue.task_status(task_id)["status"] == "expired"
    task_id = queue.enqueue("x", task_key="staged", ttl_seconds=10)
    claim = queue.claim()
    queue.stage(task_id, claim["claim_token"], snapshot())
    now[0] += 100000
    queue.claim()
    assert queue.ready_results()[0]["id"] == task_id


def test_claim_race_has_only_one_winner(queue):
    queue.enqueue("x", task_key="race")
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: queue.claim(), range(8)))
    assert len([result for result in results if result]) == 1


def test_claim_rechecks_disabled_source_even_if_collector_enqueued_after_toggle(queue):
    with queue.connection() as db:
        db.execute("CREATE TABLE sources(id TEXT PRIMARY KEY, enabled INTEGER)")
        db.execute("CREATE TABLE runtime_settings(key TEXT PRIMARY KEY,value TEXT)")
        db.execute("INSERT INTO sources VALUES('x',1)")
        db.execute("INSERT INTO runtime_settings VALUES('source_enabled:x','false')")
    task = queue.enqueue("x", task_key="racing-collector-after-source-off")
    assert queue.claim() is None
    assert queue.task_status(task)["error_code"] == "source_disabled"
    with queue.connection() as db:
        db.execute("UPDATE runtime_settings SET value='true'")
    renewed = queue.enqueue("x", task_key="explicit-reenable")
    assert queue.claim()["id"] == renewed


def test_reported_challenge_is_terminal_not_auto_retried(queue):
    task_id = queue.enqueue("x", task_key="challenge")
    claim = queue.claim()
    queue.fail(task_id, claim["claim_token"], "challenge")
    assert queue.claim() is None
    assert queue.latest_source_task("x")["error_code"] == "challenge"
    assert queue.task_status(task_id)["status"] == "failed"


def test_disable_invalidates_pending_claim_but_preserves_staged_observation(queue):
    task_id = queue.enqueue("x", task_key="cancel-me")
    claim = queue.claim()
    assert queue.cancel_pending("x") == 1
    assert queue.claim() is None
    with pytest.raises(BrowserTaskError, match="invalid_claim"):
        queue.stage(task_id, claim["claim_token"], snapshot())
    task_id = queue.enqueue("x", task_key="already-received")
    claim = queue.claim()
    queue.stage(task_id, claim["claim_token"], snapshot())
    assert queue.cancel_pending("x") == 0
    assert queue.ready_results()[0]["id"] == task_id


def test_schema_contains_no_cookie_or_browser_token_columns(queue):
    with sqlite3.connect(queue.database_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(browser_source_tasks)")}
    assert not columns & {"cookies", "auth_token", "ct0", "pairing_token", "authorization"}


def test_extension_uses_only_allowlisted_read_surface():
    directory = Path(__file__).parents[1] / "browser-extension"
    manifest = json.loads((directory / "manifest.json").read_text())
    assert set(manifest["permissions"]) == {"storage", "alarms"}
    assert set(manifest["host_permissions"]) == {"http://127.0.0.1:8769/*", "https://x.com/*"}
    content = (directory / "content.js").read_text()
    for forbidden in ["document.cookie", "fetch(", ".click(", ".scroll(", "scrollTo(", "eval("]:
        assert forbidden not in content
