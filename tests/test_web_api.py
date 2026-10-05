from __future__ import annotations

import asyncio
import json
from pathlib import Path

from aiohttp import CookieJar
from aiohttp.test_utils import TestClient, TestServer

from app.config import Settings, load_source_registry
from app.db.repository import Repository
from app.web.content import evidence_url, plain_text, safe_html
from app.web.folders import infer_reading_folder
from app.web.server import CONTENT_SECURITY_POLICY, create_app
from app.web.store import DashboardStore
from tests.test_details_worker import artifact

TOKEN = "test-private-dashboard-token-123456789"
ROOT = Path(__file__).parents[1]


class FakeReports:
    def __init__(self):
        self.calls = []
        self.short = None
        self.pdf = None

    def cached_tldr(self, video):
        return self.short

    def cached_pdf(self, video):
        return self.pdf

    def build_tldr(self, video, *, progress):
        self.calls.append("short")
        progress("writing")
        self.short = "Краткий содержательный пересказ без вызова API."
        return self.short


def fixture(tmp_path):
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        app_env="test",
        allowed_user_id=123,
        contours_file=ROOT / "tests/fixtures/config/contours.yaml",
        sources_file=ROOT / "tests/fixtures/config/sources.yaml",
    )
    repo = Repository(settings.database_path)
    repo.migrate()
    repo.sync_sources(load_source_registry(settings.sources_file))
    delivery = repo.create_smoke_delivery(123, "https://example.org/official")
    repo.mark_delivery_sent(delivery["delivery_id"], 99)
    with repo.connection() as db:
        db.execute(
            "UPDATE events SET status='card_ready',headline='Пекин меняет пошлины',"
            "score_features_json=? WHERE id=?",
            (json.dumps({"primary_contour": "world_economy"}), delivery["event_id"]),
        )
        db.execute(
            "UPDATE cards SET structured_output_json=?,rendered_html=? WHERE id=?",
            (
                json.dumps({"headline": "Пекин меняет пошлины", "lead": "Новые торговые правила."}),
                '<b>Пекин меняет пошлины</b>\n\nКонтекст и <a href="https://example.org/">источник</a>.',
                delivery["card_id"],
            ),
        )
        source_item = db.execute(
            "SELECT primary_source_item_id FROM events WHERE id=?", (delivery["event_id"],)
        ).fetchone()[0]
        db.execute(
            "INSERT INTO video_recommendations(source_item_id,chat_id,description_html,"
            "status,created_at) VALUES(?,123,'<b>Видео о стартапах</b>','sent',?)",
            (source_item, "2026-09-22T01:00:00+00:00"),
        )
    return settings, repo, delivery


async def client_for(settings, repo, **kwargs):
    client = TestClient(
        TestServer(create_app(settings, repo, access_token=TOKEN, **kwargs)),
        cookie_jar=CookieJar(unsafe=True),
    )
    await client.start_server()
    return client


async def login(client):
    response = await client.post("/api/session", json={"token": TOKEN})
    assert response.status == 200
    assert response.cookies["pars_dashboard_session"]["httponly"]
    assert response.cookies["pars_dashboard_session"]["samesite"] == "Strict"
    return {"X-CSRF-Token": (await response.json())["csrf"]}


def test_feed_context_reactions_saved_sources_and_region(tmp_path):
    settings, repo, delivery = fixture(tmp_path)
    reports = FakeReports()

    async def exercise():
        client = await client_for(settings, repo, reports=reports)
        try:
            assert (await client.get("/api/feed")).status == 401
            headers = await login(client)
            bootstrap = await (await client.get("/api/bootstrap")).json()
            assert bootstrap["status"]["news_count"] == 1
            assert bootstrap["status"]["mode"] == "local_snapshot"
            assert not bootstrap["features"]["paid_generation"]
            result = await (
                await client.get("/api/feed?folder=political_economy&region=beijing")
            ).json()
            assert result["region_filter"] == "text_match"
            assert result["items"][0]["title"] == "Пекин меняет пошлины"
            item = result["items"][0]
            assert item["sources"][0]["role"] == "evidence"
            route = f"/api/items/news/{delivery['card_id']}"
            context = await (
                await client.post(route + "/expand", headers=headers, json={"level": "context"})
            ).json()
            assert context["cached"] and "Контекст" in context["content_html"]
            assert (
                await client.post(route + "/expand", headers=headers, json={"level": "article"})
            ).status == 409
            for reaction in ("super_like", "dislike", "super_dislike", "like", None):
                response = await client.post(
                    route + "/reaction", headers=headers, json={"reaction": reaction}
                )
                assert response.status == 200
                assert (await response.json())["item"]["reaction"] == reaction
            assert repo.current_reaction(delivery["delivery_id"]) == ""
            saved = await client.post(route + "/saved", headers=headers, json={"saved": True})
            assert (await saved.json())["item"]["saved"]
            assert (await (await client.get("/api/feed?saved=1")).json())["total"] == 1
            assert (await client.get("/api/sources?folder=political_economy")).status == 200
            assert (await client.get("/api/channels")).status == 200
            assert reports.calls == []
            with repo.connection() as db:
                assert db.execute("SELECT COUNT(*) FROM detail_requests").fetchone()[0] == 0
                assert db.execute("SELECT COUNT(*) FROM usage_ledger").fetchone()[0] == 0
                row = db.execute(
                    "SELECT * FROM deliveries WHERE id=?", (delivery["delivery_id"],)
                ).fetchone()
                assert row["telegram_message_id"] == 99 and row["delivery_status"] == "sent"
        finally:
            await client.close()

    asyncio.run(exercise())


def test_security_rejects_cross_origin_host_csrf_and_invalid_input(tmp_path):
    settings, repo, delivery = fixture(tmp_path)

    async def exercise():
        client = await client_for(settings, repo, reports=FakeReports())
        try:
            assert (
                await client.post(
                    "/api/session", json={"token": TOKEN}, headers={"Origin": "https://evil.test"}
                )
            ).status == 403
            assert (await client.get("/api/bootstrap", headers={"Host": "evil.test"})).status == 403
            headers = await login(client)
            route = f"/api/items/news/{delivery['card_id']}/saved"
            assert (await client.post(route, json={"saved": True})).status == 403
            assert (await client.post(route, headers=headers, json={"saved": "true"})).status == 400
            assert (await client.get("/api/feed?cursor=@@@@")).status == 400
            assert (await client.get("/api/feed?folder=invalid")).status == 400
            assert (await client.get("/api/items/news/999999")).status == 404
            assert (
                await client.post(
                    "/api/channels", headers=headers, json={"url": "https://www.youtube.com/@YC"}
                )
            ).status == 409
            assert (
                await client.post(
                    "/api/channels", headers=headers, json={"url": "http://169.254.169.254/"}
                )
            ).status == 400
        finally:
            await client.close()

    asyncio.run(exercise())


def test_safe_markup_keeps_links_without_scripts_or_attributes():
    raw = (
        '<b onclick="alert(1)">Hello</b><script>secret()</script>'
        '<a href="javascript:alert(1)">bad</a><img src=x onerror=alert(1)>'
        '<a href="https://example.org/?x=1&y=2">source</a>'
    )
    result = safe_html(raw)
    assert "script" not in result and "onclick" not in result and "onerror" not in result
    assert "javascript" not in result and "secret" not in result
    assert "<b>Hello</b>" in result and 'rel="noopener noreferrer"' in result
    assert plain_text(result) == "Hello bad source"


def test_source_list_respects_runtime_pause_and_channel_removal(tmp_path):
    settings, repo, _ = fixture(tmp_path)
    store = DashboardStore(settings, repo, reports=FakeReports())
    store.initialize()
    with repo.connection() as db:
        db.execute(
            "UPDATE sources SET kind='youtube',locator='UCabcdefghijklmnopqrstuv' "
            "WHERE id=(SELECT id FROM sources ORDER BY id LIMIT 1)"
        )
        source = db.execute(
            "SELECT id,locator FROM sources WHERE kind='youtube' LIMIT 1"
        ).fetchone()
    repo.set_runtime_setting("source_enabled:" + source["id"], "false")
    channel = next(channel for channel in store.channels() if channel["id"] == source["id"])
    assert channel["enabled"] is False
    with repo.connection() as db:
        db.execute(
            "INSERT INTO youtube_channels(channel_id,source_id,title,added_by,"
            "added_at,not_before,removed_at) "
            "VALUES(?,?,'Channel',123,'2026-09-22','2026-09-22','2026-09-22')",
            (source["locator"], source["id"]),
        )
    assert not any(channel["id"] == source["id"] for channel in store.channels())


def test_display_only_folder_fallback_keeps_stored_assignment_and_ambiguity(tmp_path):
    settings, repo, _ = fixture(tmp_path)
    store = DashboardStore(settings, repo, reports=FakeReports())
    row = {"headline": "OpenAI выпустила новую модель", "rendered_html": "Нейросеть для работы."}
    assert store.folder_details(row) == ("ai_software", "inferred_rules")
    row["score_features_json"] = json.dumps({"primary_contour": "world_economy"})
    assert store.folder_details(row) == ("political_economy", "stored")
    assert (
        infer_reading_folder({"headline": "Тегеран: новые санкции и пошлины"})
        == "political_economy"
    )
    assert infer_reading_folder({"headline": "OpenAI и NASA"}) is None
    assert infer_reading_folder({"headline": "Что-то неожиданное"}) is None


def test_static_entry_and_prefixed_assets_share_safe_root(tmp_path):
    settings, repo, _ = fixture(tmp_path)
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html><title>OSVED</title>")
    (static / "dashboard.css").write_text("body { color: white }")

    async def exercise():
        client = await client_for(settings, repo, reports=FakeReports(), static_dir=static)
        try:
            response = await client.get("/")
            assert response.status == 200 and "OSVED" in await response.text()
            assert response.headers["Content-Security-Policy"] == CONTENT_SECURITY_POLICY
            assert "script-src 'self'" in CONTENT_SECURITY_POLICY
            assert "object-src 'none'" in CONTENT_SECURITY_POLICY
            assert "connect-src 'self' blob:" in CONTENT_SECURITY_POLICY
            response = await client.get("/static/dashboard.css")
            assert response.status == 200 and "color: white" in await response.text()
            assert response.headers["Content-Security-Policy"] == CONTENT_SECURITY_POLICY
            response = await client.get("/api/feed")
            assert response.status == 401
            assert response.headers["Content-Security-Policy"] == CONTENT_SECURITY_POLICY
            assert response.headers["Cache-Control"] == "no-store"
            response = await client.get("/", headers={"Host": "evil.test"})
            assert response.status == 403
            assert response.headers["Content-Security-Policy"] == CONTENT_SECURITY_POLICY
            assert (await client.get("/static/.env")).status == 404
        finally:
            await client.close()

    asyncio.run(exercise())


def test_evidence_presentation_excludes_media_without_changing_provenance(tmp_path):
    settings, repo, delivery = fixture(tmp_path)
    store = DashboardStore(settings, repo, reports=FakeReports())
    store.initialize()
    original = [
        "https://example.org/article",
        "https://example.org/cover.JPG?width=1200",
        "https://example.org/graphic%2Esvg",
        "https://example.org/movie.mp4#t=10",
        "https://example.org/report.pdf?download=1",
    ]
    with repo.connection() as db:
        db.execute(
            "UPDATE cards SET source_urls_json=? WHERE id=?",
            (json.dumps(original), delivery["card_id"]),
        )
    article = artifact(delivery["event_id"]).model_copy(
        update={
            "source_urls": [
                "https://example.org/article-image.webp",
                "https://example.org/analysis.pdf",
            ],
        }
    )
    store.store_article(article)
    result = store.item("news", delivery["card_id"])
    urls = [source["url"] for source in result["sources"]]
    assert "https://example.org/article" in urls
    assert "https://example.org/report.pdf?download=1" in urls
    assert "https://example.org/analysis.pdf" in urls
    assert not any(
        "cover" in url or "graphic" in url or "movie" in url or "article-image" in url
        for url in urls
    )
    with repo.connection() as db:
        saved = db.execute(
            "SELECT source_urls_json FROM cards WHERE id=?", (delivery["card_id"],)
        ).fetchone()[0]
    assert json.loads(saved) == original
    assert evidence_url("https://example.org/about.jpg-format")
    assert evidence_url("https://example.org/read?format=png")
    assert evidence_url("https://example.org/img.png") is None


def test_explicit_jobs_are_deduplicated_and_dont_claim_telegram_jobs(tmp_path):
    settings, repo, delivery = fixture(tmp_path)
    pending = repo.enqueue_detail_request(delivery["delivery_id"], 123)
    calls = []

    def builder(settings, repository, event_id, *, progress):
        calls.append(event_id)
        progress("writing")
        return artifact(event_id)

    async def exercise():
        client = await client_for(
            settings, repo, allow_paid=True, reports=FakeReports(), article_builder=builder
        )
        try:
            headers = await login(client)
            route = f"/api/items/news/{delivery['card_id']}/expand"
            responses = await asyncio.gather(
                *[client.post(route, headers=headers, json={"level": "article"}) for _ in range(4)]
            )
            results = [await response.json() for response in responses]
            jobs = {result["job_id"] for result in results if "job_id" in result}
            assert len(jobs) == 1
            job_id = jobs.pop()
            for _ in range(100):
                job = await (await client.get("/api/jobs/" + job_id)).json()
                if job["status"] not in {"queued", "running"}:
                    break
                await asyncio.sleep(0.01)
            assert job["status"] == "ready" and "Detailed article" in job["content_html"]
            assert calls == [delivery["event_id"]]
            assert (
                repo.detail_request_for_delivery(delivery["delivery_id"])["status"]
                == pending["status"]
            )
            response = await client.post(route, headers=headers, json={"level": "article"})
            assert (await response.json())["cached"]
        finally:
            await client.close()

    asyncio.run(exercise())


def test_pagination_returns_each_item_once_and_video_cached_short_is_free(tmp_path):
    settings, repo, delivery = fixture(tmp_path)
    reports = FakeReports()
    reports.short = "Сохранённый пересказ."

    async def exercise():
        client = await client_for(settings, repo, reports=reports)
        try:
            headers = await login(client)
            first = await (await client.get("/api/feed?limit=1")).json()
            assert first["has_more"]
            second = await (await client.get("/api/feed?limit=1&cursor=" + first["cursor"])).json()
            assert not second["has_more"]
            assert first["items"][0]["key"] != second["items"][0]["key"]
            video = next(
                item for item in first["items"] + second["items"] if item["kind"] == "video"
            )
            result = await (
                await client.post(
                    f"/api/items/video/{video['id']}/expand",
                    headers=headers,
                    json={"level": "short"},
                )
            ).json()
            assert result["cached"] and "Сохранённый" in result["content_html"]
            assert not reports.calls
        finally:
            await client.close()

    asyncio.run(exercise())
