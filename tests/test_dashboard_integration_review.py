"""Independent local-dashboard safety review. No network providers or Telegram calls."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import aiohttp
import pytest
from aiohttp.test_utils import TestClient, TestServer

from app.config import Settings, load_source_registry
from app.db.repository import Repository
from app.ops.snapshot import backup_database
from app.web.content import public_url, safe_html
from app.web.server import STATE, create_app

ROOT = Path(__file__).parents[1]
TOKEN = "review-private-key-not-a-real-user-secret"


class NoProviders:
    def cached_tldr(self, video):
        return None

    def cached_pdf(self, video):
        return None

    def build_tldr(self, *args, **kwargs):
        raise AssertionError("Unexpected provider call")

    def build(self, *args, **kwargs):
        raise AssertionError("Unexpected provider call")


def setup_repo(path):
    settings = Settings(
        data_dir=path,
        allowed_user_id=42,
        _env_file=None,
        contours_file=ROOT / "tests/fixtures/config/contours.yaml",
        openrouter_api_key=None,
        monid_api_key=None,
        telegram_bot_token=None,
    )
    repo = Repository(settings.database_path)
    repo.migrate()
    repo.sync_sources(load_source_registry(ROOT / "tests/fixtures/config/sources.yaml"))
    item = repo.create_smoke_delivery(42, "https://example.com/review")
    repo.mark_delivery_sent(item["delivery_id"], 9001)
    with repo.connection() as db:
        db.execute("UPDATE events SET status='card_ready' WHERE id=?", (item["event_id"],))
        db.execute(
            "UPDATE cards SET structured_output_json=?, rendered_html=? WHERE id=?",
            (
                json.dumps({"headline": "Test event", "lead": "A complete grounded lead."}),
                '<b>Test event</b>\n\nComplete context.<script>alert("xss")</script>',
                item["card_id"],
            ),
        )
    return settings, repo, item


async def logged_client(settings, repo, **kwargs):
    app = create_app(settings, repo, access_token=TOKEN, reports=NoProviders(), **kwargs)
    client = TestClient(TestServer(app), cookie_jar=aiohttp.CookieJar(unsafe=True))
    await client.start_server()
    response = await client.post("/api/session", json={"token": TOKEN})
    assert response.status == 200
    csrf = (await response.json())["csrf"]
    client.session.headers["X-CSRF-Token"] = csrf
    return client


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "data:text/html,<svg onload=alert(1)>",
        "file:///etc/passwd",
        "//example.com",
        "https://user:secret@example.com/",
    ],
)
def test_unsafe_urls_are_not_exposed_as_clickable_links(url):
    assert public_url(url) is None
    assert "href=" not in safe_html(f'<a href="{url}">label</a>')


def test_browser_html_strips_active_content_and_event_handlers():
    value = safe_html(
        '<p onclick="alert(1)">Text<img src=x onerror=alert(1)>'
        "<svg><script>BAD</script></svg>"
        '<a href="https://example.com" style="position:fixed">source</a></p>'
    )
    assert "<p>Text" in value
    unsafe = ("onclick", "onerror", "<img", "<svg", "BAD", "style=")
    assert all(marker not in value for marker in unsafe)
    assert 'rel="noopener noreferrer"' in value


def test_local_host_origin_csrf_and_token_boundaries(tmp_path):
    settings, repo, item = setup_repo(tmp_path)

    async def exercise():
        client = await logged_client(settings, repo)
        try:
            path = f"/api/items/news/{item['card_id']}/reaction"
            response = await client.get("/api/bootstrap", headers={"Host": "attacker.example"})
            assert response.status == 403
            response = await client.post(
                path, json={"reaction": "like"}, headers={"Origin": "https://attacker.example"}
            )
            assert response.status == 403
            response = await client.post(
                path, json={"reaction": "like"}, headers={"X-CSRF-Token": "wrong"}
            )
            assert response.status == 403
            client.session.cookie_jar.clear()
            response = await client.get("/api/feed")
            assert response.status == 401
        finally:
            await client.close()
        with repo.connection() as db:
            assert db.execute("SELECT count(*) FROM feedback_events").fetchone()[0] == 0

    asyncio.run(exercise())


def test_snapshot_reactions_do_not_touch_original_or_telegram_state(tmp_path):
    original_settings, original, item = setup_repo(tmp_path / "original")
    original.enqueue_detail_request(item["delivery_id"], 42)
    before = original.get_delivery_context(item["delivery_id"])
    backup = backup_database(original.database_path, tmp_path / "copy")
    assert backup is not None
    settings = original_settings.model_copy(update={"data_dir": backup.parent})
    local = Repository(backup)
    local.migrate()

    async def exercise():
        client = await logged_client(settings, local)
        try:
            path = f"/api/items/news/{item['card_id']}"
            response = await client.post(path + "/reaction", json={"reaction": "super_like"})
            assert response.status == 200
            assert (await response.json())["item"]["reaction"] == "super_like"
            response = await client.post(path + "/expand", json={"level": "context"})
            assert response.status == 200
            expanded = await response.json()
            assert expanded["status"] == "ready" and expanded["cached"]
            assert "Complete context" in expanded["content_html"]
            assert "script" not in expanded["content_html"]
            assert local.get_delivery_context(item["delivery_id"]) == before
            with local.connection() as db:
                assert db.execute("SELECT status FROM detail_requests").fetchone()[0] == "queued"
        finally:
            await client.close()

    asyncio.run(exercise())
    assert original.current_reaction(item["delivery_id"]) is None
    assert original.get_delivery_context(item["delivery_id"]) == before
    with sqlite3.connect(original.database_path.as_uri() + "?mode=ro", uri=True) as db:
        assert not db.execute("SELECT 1 FROM sqlite_master WHERE name LIKE 'web_%'").fetchall()
    assert local.current_reaction(item["delivery_id"]) == "superlike"


def test_disabled_paid_generation_cannot_queue_work(tmp_path):
    settings, repo, item = setup_repo(tmp_path)

    async def exercise():
        client = await logged_client(settings, repo)
        try:
            response = await client.post(
                f"/api/items/news/{item['card_id']}/expand", json={"level": "article"}
            )
            assert response.status == 409
            assert (await response.json())["error"]["code"] == "generation_disabled"
            response = await client.post(
                "/api/channels", json={"url": "https://www.youtube.com/@somebody"}
            )
            assert response.status == 409
            assert (await response.json())["error"]["code"] == "resolution_disabled"
            with repo.connection() as db:
                assert db.execute("SELECT count(*) FROM web_jobs").fetchone()[0] == 0
                assert db.execute("SELECT count(*) FROM usage_ledger").fetchone()[0] == 0
        finally:
            await client.close()

    asyncio.run(exercise())


def test_static_root_does_not_expose_dotfiles_parent_files_or_symlink_targets(tmp_path):
    settings, repo, _ = setup_repo(tmp_path / "data")
    static = tmp_path / "public"
    static.mkdir()
    (static / "index.html").write_text("Public shell")
    (static / ".env").write_text("PRIVATE")
    outside = tmp_path / "secret.txt"
    outside.write_text("PRIVATE")
    (static / "outside.txt").symlink_to(outside)

    async def exercise():
        client = await logged_client(settings, repo, static_dir=static)
        try:
            for path in ("/.env", "/outside.txt", "/%2e%2e/secret.txt", "/data/newsfeed.sqlite3"):
                response = await client.get(path)
                assert response.status == 404, path
                assert "PRIVATE" not in await response.text()
            assert (await client.get("/")).status == 200
        finally:
            await client.close()

    asyncio.run(exercise())


def test_failed_paid_job_is_not_automatically_retried_on_second_click(tmp_path):
    settings, repo, item = setup_repo(tmp_path)
    calls = []

    def fake_builder(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("private-provider-error-that-must-not-leak")

    async def exercise():
        client = await logged_client(settings, repo, allow_paid=True, article_builder=fake_builder)
        try:
            path = f"/api/items/news/{item['card_id']}/expand"
            first, second = await asyncio.gather(
                client.post(path, json={"level": "article"}),
                client.post(path, json={"level": "article"}),
            )
            a, b = await first.json(), await second.json()
            assert first.status == second.status == 202
            assert a["job_id"] == b["job_id"]
            state = client.server.app[STATE]
            if state.tasks:
                await asyncio.gather(*state.tasks)
            response = await client.get("/api/jobs/" + a["job_id"])
            result = await response.json()
            assert result["status"] == "failed"
            assert "private-provider" not in json.dumps(result)
            response = await client.post(path, json={"level": "article"})
            assert (await response.json())["job_id"] == a["job_id"]
            assert len(calls) == 1
        finally:
            await client.close()

    asyncio.run(exercise())
