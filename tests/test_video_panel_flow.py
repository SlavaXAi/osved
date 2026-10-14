import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.methods import EditMessageText

from app.config import Settings, SourceConfig, SourceRegistry
from app.db.repository import Repository
from app.telegram_bot.bot import build_router
from app.telegram_bot.sensory import VideoControl
from app.telegram_bot.video_panel import (
    VideoPanelControl,
    VideoResultControl,
    ensure_panel,
    get_panel,
    result_keyboard,
    show_cached_summary,
)
from app.telegram_bot.video_reports import VideoReportWorker, enqueue

OWNER = 123
VIDEO = "abcdefghijk"
SUMMARY = (
    "Автор объясняет, как агенты меняют реальные рабочие процессы и почему важны ограничения. "
    "Он сравнивает несколько подходов и приводит конкретные примеры применения. "
    "Польза зависит от качества входных данных и контроля действий системы. "
) * 5


class Service:
    def __init__(self, path):
        self.text = None
        self.calls = []
        self.path = path

    def cached_tldr(self, vid):
        return self.text

    def build_tldr(self, vid, progress=None):
        self.calls.append("tldr")
        self.text = SUMMARY
        return self.text

    def build(self, vid, progress=None):
        self.calls.append("pdf")
        self.path.write_bytes(b"fixture")
        return self.path


def handler(router, name):
    return next(h.callback for h in router.callback_query.handlers if h.callback.__name__ == name)


def query(bot, *, message_id=777, user_id=OWNER):
    message = SimpleNamespace(
        message_id=message_id,
        chat=SimpleNamespace(id=user_id, type="private"),
        edit_reply_markup=AsyncMock(),
        reply_markup=None,
    )
    return SimpleNamespace(
        from_user=SimpleNamespace(id=user_id), message=message, bot=bot, answer=AsyncMock()
    )


@pytest.fixture
def env(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, app_env="test", data_dir=tmp_path, allowed_user_id=OWNER)
    repo = Repository(settings.database_path)
    repo.migrate()
    repo.sync_sources(
        SourceRegistry(
            version=1,
            sources=[
                SourceConfig(
                    id="yt",
                    name="Channel",
                    kind="youtube",
                    locator="UC" + "a" * 22,
                    role="radar",
                    region="global",
                    language="en",
                )
            ],
        )
    )
    item = repo.insert_source_item_with_id(
        source_id="yt",
        external_id=VIDEO,
        source_url="https://www.youtube.com/watch?v=" + VIDEO,
        original_text="Название выпуска\nРабота агентов.",
        original_language="ru",
        published_at="2026-10-13T07:00:00+00:00",
        outbound_urls=[],
        media_metadata={"is_short": False},
        payload_hash="fixture",
        raw_payload={},
    )
    with repo.connection() as db:
        rec = db.execute(
            "INSERT INTO video_recommendations(source_item_id,chat_id,telegram_message_id,"
            "description_html,status,created_at) "
            "VALUES(?,123,777,'<b>Карточка</b>','sent',datetime('now'))",
            (item,),
        ).lastrowid
    bot = SimpleNamespace(
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=888)),
        edit_message_text=AsyncMock(),
        edit_message_reply_markup=AsyncMock(),
        send_document=AsyncMock(return_value=SimpleNamespace(message_id=999)),
        delete_message=AsyncMock(),
    )
    service = Service(tmp_path / "report.pdf")
    monkeypatch.setattr("app.telegram_bot.video_reports.VideoReportService", lambda *args: service)
    return settings, repo, bot, service, build_router(settings, repo), rec


def make_summary(env):
    settings, repo, bot, service, router, rec = env
    asyncio.run(
        handler(router, "video_handler")(
            query(bot), VideoControl(recommendation_id=rec, action="short")
        )
    )
    assert asyncio.run(VideoReportWorker(bot, settings, repo, service=service).tick())
    with repo.connection() as db:
        return dict(db.execute("SELECT * FROM video_pdf_jobs WHERE output_kind='tldr'").fetchone())


def test_summary_replaces_original_card_and_preserves_menu_and_source(env):
    settings, repo, bot, service, router, rec = env
    job = make_summary(env)
    bot.send_message.assert_not_awaited()
    assert job["text_message_id"] == 777 and job["status"] == "sent"
    final = bot.edit_message_text.call_args.kwargs
    assert final["message_id"] == 777 and "Смотреть видео" in final["text"]
    assert "Название выпуска" in final["text"] and final["text"].count("\n\n") >= 3
    labels = [b.text for line in final["reply_markup"].inline_keyboard for b in line]
    assert "👍 Полезно" in labels and "👎 Не полезно" in labels
    assert "Краткая сводка" in labels and "Подробный отчёт · PDF" in labels
    assert not any("Показывать" in label for label in labels)
    assert service.calls == ["tldr"]


def test_back_and_reopen_are_cached_and_stale_rating_is_rejected(env):
    settings, repo, bot, service, router, rec = env
    job = make_summary(env)
    panel = get_panel(repo, job["panel_id"])
    asyncio.run(
        handler(router, "video_panel_handler")(
            query(bot), VideoPanelControl(panel_id=panel["id"], action="card")
        )
    )
    stale = query(bot)
    asyncio.run(
        handler(router, "video_result_feedback_handler")(
            stale, VideoResultControl(job_id=job["id"], reaction="like")
        )
    )
    with repo.connection() as db:
        assert db.execute("SELECT count(*) FROM video_result_feedback").fetchone()[0] == 0
    asyncio.run(
        handler(router, "video_panel_handler")(
            query(bot), VideoPanelControl(panel_id=panel["id"], action="short")
        )
    )
    assert get_panel(repo, panel["id"])["view"] == "summary" and service.calls == ["tldr"]
    bot.send_message.assert_not_awaited()


def test_pdf_separate_with_feedback_and_original_panel_restored(env):
    settings, repo, bot, service, router, rec = env
    job = make_summary(env)
    asyncio.run(
        handler(router, "video_panel_handler")(
            query(bot), VideoPanelControl(panel_id=job["panel_id"], action="pdf")
        )
    )
    assert asyncio.run(VideoReportWorker(bot, settings, repo, service=service).tick())
    doc = bot.send_document.call_args.kwargs
    assert doc["reply_parameters"].message_id == 777
    assert [b.text for line in doc["reply_markup"].inline_keyboard for b in line] == [
        "👍 Полезно",
        "👎 Не полезно",
    ]
    bot.delete_message.assert_not_awaited()
    panel = get_panel(repo, job["panel_id"])
    assert panel["view"] == "summary"
    with repo.connection() as db:
        pdf = dict(db.execute("SELECT * FROM video_pdf_jobs WHERE output_kind='pdf'").fetchone())
    feedback = query(bot, message_id=999)
    asyncio.run(
        handler(router, "video_result_feedback_handler")(
            feedback, VideoResultControl(job_id=pdf["id"], reaction="like")
        )
    )
    with repo.connection() as db:
        assert db.execute("SELECT reaction FROM video_result_feedback").fetchone()[0] == "like"
    assert service.calls == ["tldr", "pdf"]


def test_forwarded_or_forged_result_cannot_train_owner_profile(env):
    settings, repo, bot, service, router, rec = env
    job = make_summary(env)
    for user, msg in [(456, 777), (OWNER, 998)]:
        asyncio.run(
            handler(router, "video_result_feedback_handler")(
                query(bot, user_id=user, message_id=msg),
                VideoResultControl(job_id=job["id"], reaction="dislike"),
            )
        )
    with repo.connection() as db:
        assert db.execute("SELECT count(*) FROM video_result_feedback").fetchone()[0] == 0


def test_old_standalone_alias_keeps_its_own_controls_after_reopen(env):
    settings, repo, bot, service, router, rec = env
    old = enqueue(settings, repo, service, VIDEO, OWNER, OWNER, 777, output_kind="tldr")
    with repo.connection() as db:
        db.execute(
            "UPDATE video_pdf_jobs SET status='sent',text_message_id=888 WHERE id=?", (old["id"],)
        )
    panel = ensure_panel(
        repo,
        chat_id=OWNER,
        message_id=777,
        video_id=VIDEO,
        original_html="Описание",
        recommendation_id=rec,
    )
    service.text = SUMMARY
    with repo.connection() as db:
        job = dict(db.execute("SELECT * FROM video_pdf_jobs WHERE id=?", (old["id"],)).fetchone())
    assert asyncio.run(show_cached_summary(bot, settings, repo, service, job, panel))
    q = query(bot, message_id=888)
    asyncio.run(
        handler(router, "video_result_feedback_handler")(
            q, VideoResultControl(job_id=job["id"], reaction="like")
        )
    )
    markup = q.message.edit_reply_markup.call_args.kwargs["reply_markup"]
    actions = [b.callback_data for line in markup.inline_keyboard for b in line]
    assert f"vreport:long:{job['id']}" in actions
    assert not any(a.startswith("vp:") for a in actions)
    assert service.calls == []


def test_older_summary_panel_keeps_its_full_menu_after_job_reopens_elsewhere(env):
    settings, repo, bot, service, router, rec = env
    job = make_summary(env)
    original_panel = get_panel(repo, job["panel_id"])
    newer_panel = ensure_panel(
        repo, chat_id=OWNER, message_id=888, video_id=VIDEO, original_html="Другое описание"
    )
    assert asyncio.run(show_cached_summary(bot, settings, repo, service, job, newer_panel))
    feedback = query(bot, message_id=777)
    asyncio.run(
        handler(router, "video_result_feedback_handler")(
            feedback, VideoResultControl(job_id=job["id"], reaction="like")
        )
    )
    markup = feedback.message.edit_reply_markup.call_args.kwargs["reply_markup"]
    buttons = [b for line in markup.inline_keyboard for b in line]
    assert {
        b.callback_data for b in buttons if b.callback_data and b.callback_data.startswith("vp:")
    } == {
        f"vp:{original_panel['id']}:short",
        f"vp:{original_panel['id']}:pdf",
        f"vp:{original_panel['id']}:card",
    }
    assert any(b.text == "Открыть видео" and b.url for b in buttons)
    assert any(b.text == "✓ 👍 Полезно" for b in buttons)
    assert service.calls == ["tldr"]
    with repo.connection() as db:
        assert db.execute("SELECT reaction FROM video_result_feedback").fetchone()[0] == "like"


def test_result_keyboard_rejects_origin_panel_not_displaying_that_job(env):
    _, repo, _, _, _, _ = env
    job = make_summary(env)
    with repo.connection() as db:
        db.execute("UPDATE video_panels SET view='card' WHERE id=?", (job["panel_id"],))
    with pytest.raises(PermissionError, match="not displaying"):
        result_keyboard(repo, job, message_id=777)


def refresh_module(env, monkeypatch):
    settings, _, bot, service, _, _ = env
    path = Path(__file__).parents[1] / "deploy" / "reader-flow-refresh-ui.py"
    spec = importlib.util.spec_from_file_location("reader_flow_refresh_ui", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "Settings", lambda: settings)
    monkeypatch.setattr(module, "build_bot", lambda _: bot)
    monkeypatch.setattr(module, "VideoReportService", lambda *args: service)
    monkeypatch.setattr(module.asyncio, "sleep", AsyncMock())
    bot.session = SimpleNamespace(close=AsyncMock())
    return module


@pytest.mark.parametrize("outcome", ["changed", "unchanged", "unavailable"])
def test_refresh_registers_only_confirmed_result_edit_receipts(env, monkeypatch, capsys, outcome):
    _, repo, bot, service, _, _ = env
    job = make_summary(env)
    with repo.connection() as db:
        db.execute("DELETE FROM video_result_messages")
    bot.edit_message_text.reset_mock()
    if outcome != "changed":
        bot.edit_message_text.side_effect = TelegramBadRequest(
            method=EditMessageText(text="fixture"),
            message="message is not modified" if outcome == "unchanged" else "message not found",
        )
    module = refresh_module(env, monkeypatch)
    asyncio.run(module.main())
    counts = json.loads(capsys.readouterr().out)
    assert counts["new_messages"] == counts["paid_calls"] == 0
    assert counts["summary_texts"] == (1 if outcome == "changed" else 0)
    if outcome != "changed":
        assert counts[outcome] == 1
    with repo.connection() as db:
        receipts = db.execute("SELECT * FROM video_result_messages").fetchall()
    assert len(receipts) == (0 if outcome == "unavailable" else 1)
    if receipts:
        assert receipts[0]["job_id"] == job["id"] and receipts[0]["message_id"] == 777
        assert receipts[0]["panel_id"] == job["panel_id"]
    assert service.calls == ["tldr"]
    bot.send_message.assert_not_awaited()
    bot.send_document.assert_not_awaited()


def test_refresh_skips_historic_job_receipt_when_panel_displays_a_different_job(env, monkeypatch):
    settings, repo, bot, service, _, _ = env
    old_job = make_summary(env)
    current = enqueue(settings, repo, service, VIDEO, OWNER, OWNER, 778, output_kind="tldr")
    with repo.connection() as db:
        db.execute("DELETE FROM video_result_messages")
        db.execute(
            "UPDATE video_pdf_jobs SET status='sent',text_message_id=777,panel_id=? WHERE id=?",
            (old_job["panel_id"], current["id"]),
        )
        db.execute(
            "UPDATE video_panels SET summary_job_id=? WHERE id=?",
            (current["id"], old_job["panel_id"]),
        )
    bot.edit_message_text.reset_mock()
    module = refresh_module(env, monkeypatch)
    asyncio.run(module.main())
    assert bot.edit_message_text.await_count == 1
    with repo.connection() as db:
        receipts = db.execute("SELECT job_id,message_id FROM video_result_messages").fetchall()
    assert [tuple(row) for row in receipts] == [(current["id"], 777)]
    assert get_panel(repo, old_job["panel_id"])["summary_job_id"] == current["id"]
    assert service.calls == ["tldr"]


def test_results_only_refresh_skips_recommendations_and_paces_requests(env, monkeypatch, capsys):
    _, _, bot, service, _, _ = env
    make_summary(env)
    bot.edit_message_text.reset_mock()
    bot.edit_message_reply_markup.reset_mock()
    module = refresh_module(env, monkeypatch)
    asyncio.run(module.main(results_only=True))
    counts = json.loads(capsys.readouterr().out)
    assert counts["results_only"] and counts["completed"]
    assert counts["retry_after"] == counts["recommendation_controls"] == 0
    assert counts["summary_texts"] == 1
    bot.edit_message_text.assert_awaited_once()
    bot.edit_message_reply_markup.assert_not_awaited()
    assert module.asyncio.sleep.await_count == 1
    assert all(call.args[0] >= 1.2 for call in module.asyncio.sleep.await_args_list)
    assert service.calls == ["tldr"]


def test_refresh_stops_on_long_rate_limit_and_reports_confirmed_partial_progress(
    env, monkeypatch, capsys
):
    settings, repo, bot, service, _, _ = env
    make_summary(env)
    created = []
    for message_id in (888, 999):
        job = enqueue(settings, repo, service, VIDEO, OWNER, OWNER, message_id, output_kind="tldr")
        created.append(job["id"])
        with repo.connection() as db:
            db.execute(
                "UPDATE video_pdf_jobs SET status='sent',text_message_id=? WHERE id=?",
                (message_id, job["id"]),
            )
    with repo.connection() as db:
        db.execute("DELETE FROM video_result_messages")
    bot.edit_message_text.reset_mock()
    bot.edit_message_text.side_effect = [
        SimpleNamespace(message_id=999),
        TelegramRetryAfter(
            method=EditMessageText(text="fixture"),
            message="Flood control exceeded",
            retry_after=269,
        ),
    ]
    module = refresh_module(env, monkeypatch)
    asyncio.run(module.main(results_only=True))
    output = capsys.readouterr()
    counts = json.loads(output.out)
    assert counts["completed"] is False and counts["retry_after"] == 269
    assert counts["summary_texts"] == 1 and counts["recommendation_controls"] == 0
    assert counts["new_messages"] == counts["paid_calls"] == 0
    assert output.err == ""
    message_ids = [call.kwargs["message_id"] for call in bot.edit_message_text.await_args_list]
    assert message_ids == [999, 888]
    with repo.connection() as db:
        receipts = db.execute("SELECT job_id,message_id FROM video_result_messages").fetchall()
    assert [tuple(row) for row in receipts] == [(created[-1], 999)]
    assert module.asyncio.sleep.await_count == 1
    bot.session.close.assert_awaited_once()
    bot.send_message.assert_not_awaited()
    bot.send_document.assert_not_awaited()
    assert service.calls == ["tldr"]
