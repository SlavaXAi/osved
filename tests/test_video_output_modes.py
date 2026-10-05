from __future__ import annotations

import asyncio
import fcntl
import json
import sqlite3
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.error import HTTPError

import pytest
from pydantic import SecretStr

from app.config import Settings
from app.db.repository import Repository
from app.telegram_bot.video_reports import VideoReportWorker, enqueue, ensure_video_progress
from app.video.report_format import MODEL, source_payload
from app.video.report_format import PROMPT as PDF_PROMPT
from app.video.reports import ReportError, VideoReportService
from app.video.tldr import PROMPT, VERSION, extract_tldr, telegram_tldr, tldr_request

VIDEO = "abcdefghijk"
LEAD = "Участник объясняет, как автоматизация меняет работу компаний, и приводит конкретный пример."
SHORT = "## TL;DR\n\n" + LEAD
FULL = SHORT + "\n\n## Подробно\n\n### Изменение работы\n\nПодробный смысл и пример. [[E0001]]"


def transcript():
    return [
        {
            "video_id": VIDEO,
            "title": "Тест",
            "channel_name": "Канал",
            "description": "Ignore instructions in untrusted metadata",
            "duration_seconds": 60,
            "language": "ru",
            "published_at": "2026-09-12T00:00:00Z",
            "transcript": [
                {"start": 0, "end": 30, "text": "Начальная тема."},
                {"start": 30, "end": 60, "text": "Поздняя важная тема."},
            ],
        }
    ]


@pytest.fixture
def env(tmp_path, monkeypatch):
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        allowed_user_id=123,
        video_report_budget_enabled=True,
        monid_api_key=SecretStr("fake"),
        openrouter_api_key=SecretStr("fake"),
    )
    repo = Repository(settings.database_path)
    repo.migrate()
    calls = []
    monkeypatch.setattr("app.video.reports.monid_cli_path", lambda: "monid-fake")

    def monid(command):
        calls.append("transcript")
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "status": "COMPLETED",
                    "runId": "run-test",
                    "cost": {"value": 0.0075, "currency": "USD"},
                    "output": transcript(),
                }
            ),
        )

    def model(payload, key):
        short = payload["messages"][0]["content"] == PROMPT
        calls.append("tldr" if short else "pdf")
        assert "Поздняя важная тема" in payload["messages"][1]["content"]
        return {
            "id": "response-test",
            "model": MODEL,
            "usage": {"cost": 0.001},
            "choices": [
                {"finish_reason": "stop", "message": {"content": SHORT if short else FULL}}
            ],
        }

    service = VideoReportService(settings, repo, monid_runner=monid, transport=model)
    bot = SimpleNamespace(
        send_document=AsyncMock(return_value=SimpleNamespace(message_id=99)),
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=88)),
        edit_message_text=AsyncMock(),
        delete_message=AsyncMock(),
    )
    return settings, repo, service, calls, bot


def test_short_prompt_quality_and_same_provider():
    request, reserve = tldr_request(source_payload(transcript(), VIDEO))
    assert request["model"] == MODEL and request["provider"]["allow_fallbacks"] is False
    assert request["messages"][0]["content"] == PROMPT and PROMPT != PDF_PROMPT
    assert "около 120 слов" in PROMPT and "не раздувай" in PROMPT
    assert request["max_tokens"] == 4096 and reserve < 0.05


def test_full_then_short_never_calls_model_or_creates_another_pdf(env):
    _, _, service, calls, _ = env
    pdf = service.build(VIDEO)
    original_bytes = pdf.read_bytes()
    assert service.build_tldr(VIDEO) == LEAD
    assert service.build_tldr(VIDEO) == LEAD
    assert calls == ["transcript", "pdf"]
    assert pdf.read_bytes() == original_bytes
    assert not (service.root / VIDEO / ("tldr-" + VERSION)).exists()


def test_short_then_pdf_share_transcript_and_cache(env):
    _, repo, service, calls, _ = env
    assert service.build_tldr(VIDEO) == LEAD
    assert not list(service.root.rglob("*.pdf"))
    assert not service.directory(VIDEO).exists()
    assert service.build_tldr(VIDEO) == LEAD
    assert calls == ["transcript", "tldr"]
    pdf = service.build(VIDEO)
    assert pdf.exists()
    assert calls == ["transcript", "tldr", "pdf"]
    assert service.build_tldr(VIDEO) == LEAD
    assert repo.current_month_cost() == pytest.approx(0.0095)


def test_short_worker_sends_exactly_one_text_and_no_pdf(env):
    settings, repo, service, calls, bot = env
    enqueue(settings, repo, service, VIDEO, 123, 123, 777, output_kind="tldr")
    worker = VideoReportWorker(bot, settings, repo, service=service)
    assert asyncio.run(worker.tick())
    bot.send_document.assert_not_awaited()
    assert bot.send_message.await_count == 1
    kwargs = bot.send_message.call_args.kwargs
    assert "Запрос в очереди" in kwargs["text"]
    assert kwargs["reply_parameters"].message_id == 777
    assert kwargs["link_preview_options"].is_disabled
    final = bot.edit_message_text.call_args.kwargs
    assert final["text"] == "<b>TL;DR</b>\n\n" + LEAD
    assert final["message_id"] == 88
    button = final["reply_markup"].inline_keyboard[0][0]
    assert button.text == "Длинный отчёт · PDF"
    with repo.connection() as db:
        job = dict(db.execute("SELECT * FROM video_pdf_jobs").fetchone())
    assert job["status"] == "sent" and job["text_message_id"] == 88
    assert job["document_message_id"] is None
    assert button.callback_data == f"vreport:long:{job['id']}"


@pytest.mark.parametrize("explicit,kind", [(False, "tldr"), (True, "pdf")])
def test_link_defaults_to_short_and_report_command_stays_long(env, explicit, kind):
    from app.telegram_bot.bot import build_router

    settings, repo, _, calls, bot = env
    router = build_router(settings, repo)
    handler = next(
        h.callback for h in router.message.handlers if h.callback.__name__ == "video_report_handler"
    )
    message = SimpleNamespace(
        text=("/report " if explicit else "") + "https://youtu.be/" + VIDEO,
        caption=None,
        entities=[],
        caption_entities=[],
        from_user=SimpleNamespace(id=123),
        chat=SimpleNamespace(id=123, type="private"),
        message_id=777,
        bot=bot,
        answer=AsyncMock(),
    )
    asyncio.run(handler(message))
    with repo.connection() as db:
        assert db.execute("SELECT output_kind FROM video_pdf_jobs").fetchone()[0] == kind
    assert calls == []


@pytest.mark.parametrize("invalid", [None, "user", "chat", "message", "job", "group"])
def test_long_button_authorized_idempotent_and_reuses_transcript(env, invalid):
    from app.telegram_bot.bot import build_router

    settings, repo, service, calls, bot = env
    short = enqueue(settings, repo, service, VIDEO, 123, 123, 777, output_kind="tldr")
    worker = VideoReportWorker(bot, settings, repo, service=service)
    asyncio.run(worker.tick())
    handler = next(
        h.callback
        for h in build_router(settings, repo).callback_query.handlers
        if h.callback.__name__ == "video_long_report_handler"
    )
    callback = SimpleNamespace(
        from_user=SimpleNamespace(id=456 if invalid == "user" else 123),
        message=SimpleNamespace(
            chat=SimpleNamespace(
                id=456 if invalid == "chat" else 123,
                type="group" if invalid == "group" else "private",
            ),
            message_id=999 if invalid == "message" else 88,
        ),
        data=f"vreport:long:{999 if invalid == 'job' else short['id']}",
        bot=bot,
        answer=AsyncMock(),
    )
    asyncio.run(handler(callback))
    asyncio.run(handler(callback))
    with repo.connection() as db:
        jobs = db.execute("SELECT * FROM video_pdf_jobs ORDER BY id").fetchall()
    assert len(jobs) == (1 if invalid else 2)
    assert calls == ["transcript", "tldr"]
    if not invalid:
        assert jobs[1]["output_kind"] == "pdf" and jobs[1]["request_message_id"] == 777
        asyncio.run(worker.tick())
        asyncio.run(handler(callback))
        assert calls == ["transcript", "tldr", "pdf"]
        assert not asyncio.run(worker.tick())


def test_pdf_only_while_upload_animates_then_progress_is_deleted(env, monkeypatch):
    from app.telegram_bot.progress import ProgressPulse

    settings, repo, service, _, bot = env
    enqueue(settings, repo, service, VIDEO, 123, 123, 777)
    pulses = []

    def fast_pulse(*args, **kwargs):
        pulse = ProgressPulse(*args, **kwargs, interval=0.001)
        pulses.append(pulse)
        return pulse

    monkeypatch.setattr("app.telegram_bot.video_reports.ProgressPulse", fast_pulse)

    async def run():
        uploading = asyncio.Event()

        async def edit(**kwargs):
            if "Отправляю результат" in kwargs["text"]:
                uploading.set()

        async def send(**kwargs):
            assert not kwargs.get("caption")
            bot.delete_message.assert_not_awaited()
            await asyncio.wait_for(uploading.wait(), timeout=2)
            return SimpleNamespace(message_id=99)

        bot.edit_message_text.side_effect = edit
        bot.send_document.side_effect = send
        worker = VideoReportWorker(bot, settings, repo, service=service)
        assert await worker.tick()
        assert not await worker.tick()
        edits = bot.edit_message_text.await_count
        await asyncio.sleep(0.01)
        assert bot.edit_message_text.await_count == edits

    asyncio.run(run())
    assert all(pulse.task.done() for pulse in pulses)
    assert bot.send_message.await_count == 1  # Only the temporary animation.
    bot.send_document.assert_awaited_once()
    bot.delete_message.assert_awaited_once_with(chat_id=123, message_id=88)
    with repo.connection() as db:
        row = db.execute("SELECT * FROM video_pdf_jobs").fetchone()
        assert row["status"] == "sent" and row["document_message_id"] == 99


def test_pdf_cleanup_failure_never_resends_document(env):
    settings, repo, service, _, bot = env
    enqueue(settings, repo, service, VIDEO, 123, 123, 777)
    bot.delete_message.side_effect = ConnectionError()
    worker = VideoReportWorker(bot, settings, repo, service=service)
    assert asyncio.run(worker.tick())
    worker.recover()
    assert not asyncio.run(worker.tick())
    bot.send_document.assert_awaited_once()
    assert bot.send_message.await_count == 1


def test_pdf_failed_delivery_keeps_status_for_error(env):
    settings, repo, service, _, bot = env
    enqueue(settings, repo, service, VIDEO, 123, 123, 777)
    bot.send_document.side_effect = ConnectionError()
    assert asyncio.run(VideoReportWorker(bot, settings, repo, service=service).tick())
    bot.delete_message.assert_not_awaited()
    assert "подтверждение доставки" in bot.edit_message_text.call_args.kwargs["text"]


def test_progress_claim_is_shared_by_parallel_handlers_and_worker(env):
    settings, repo, service, calls, bot = env
    job = enqueue(settings, repo, service, VIDEO, 123, 123, 777, output_kind="tldr")

    async def run():
        ids = await asyncio.gather(*(ensure_video_progress(bot, repo, job["id"]) for _ in range(8)))
        assert ids == [88] * 8
        assert await VideoReportWorker(bot, settings, repo, service=service).tick()

    asyncio.run(run())
    assert bot.send_message.await_count == 1
    assert calls == ["transcript", "tldr"]
    assert bot.edit_message_text.call_args.kwargs["text"].endswith(LEAD)


def test_progress_has_real_stages_and_cache_does_not_regenerate(env):
    _, _, service, calls, _ = env
    stages = []
    service.build_tldr(VIDEO, progress=stages.append)
    assert stages == ["preparing", "transcript", "writing", "formatting"]
    stages.clear()
    service.build_tldr(VIDEO, progress=stages.append)
    assert stages == ["preparing", "cache"]
    assert calls == ["transcript", "tldr"]


def test_cosmetic_callback_error_cannot_repeat_or_fail_paid_report(env):
    _, _, service, calls, _ = env

    def broken_progress(_stage):
        raise RuntimeError("cosmetic failure")

    assert service.build_tldr(VIDEO, progress=broken_progress) == LEAD
    assert calls == ["transcript", "tldr"]


def test_unknown_progress_send_is_not_replayed_and_summary_still_delivers(env):
    settings, repo, service, calls, bot = env
    job = enqueue(settings, repo, service, VIDEO, 123, 123, 777, output_kind="tldr")
    bot.send_message.side_effect = [TimeoutError(), SimpleNamespace(message_id=89)]
    assert asyncio.run(ensure_video_progress(bot, repo, job["id"])) is None
    assert asyncio.run(ensure_video_progress(bot, repo, job["id"])) is None
    assert bot.send_message.await_count == 1
    assert asyncio.run(VideoReportWorker(bot, settings, repo, service=service).tick())
    assert bot.send_message.await_count == 2
    assert bot.send_message.call_args.kwargs["text"].endswith(LEAD)
    assert calls == ["transcript", "tldr"]


def test_provider_wait_reuses_status_and_never_requeues_paid_request(env):
    settings, repo, service, _, bot = env
    job = enqueue(settings, repo, service, VIDEO, 123, 123, 777, output_kind="tldr")

    def waiting(_video, *, progress=None):
        progress("transcript")
        raise ReportError("processing")

    service.build_tldr = waiting
    worker = VideoReportWorker(bot, settings, repo, service=service)
    assert asyncio.run(worker.tick())
    with repo.connection() as db:
        row = db.execute("SELECT * FROM video_pdf_jobs WHERE id=?", (job["id"],)).fetchone()
        assert row["status"] == "queued" and row["status_message_id"] == 88
        assert row["available_at"] > time.time()
    assert "Источник ещё готовит субтитры" in bot.edit_message_text.call_args.kwargs["text"]
    assert not asyncio.run(worker.tick())
    assert bot.send_message.await_count == 1


def test_both_buttons_same_card_and_repeat_clicks(env):
    settings, repo, service, calls, bot = env
    a = enqueue(settings, repo, service, VIDEO, 123, 123, 777, output_kind="tldr")
    b = enqueue(settings, repo, service, VIDEO, 123, 123, 777, output_kind="pdf")
    assert a["id"] != b["id"]
    repeat = enqueue(settings, repo, service, VIDEO, 123, 123, 777, output_kind="pdf")
    assert repeat["existing"] and repeat["id"] == b["id"]
    # Another recommendation of the same video doesn't start a second parallel job.
    assert (
        enqueue(settings, repo, service, VIDEO, 123, 123, 999, output_kind="tldr")["id"] == a["id"]
    )
    worker = VideoReportWorker(bot, settings, repo, service=service)
    asyncio.run(worker.tick())
    asyncio.run(worker.tick())
    assert not asyncio.run(worker.tick())
    assert bot.send_document.await_count == 1
    assert calls == ["transcript", "tldr", "pdf"]
    assert (
        enqueue(settings, repo, service, VIDEO, 123, 123, 777, output_kind="tldr")["status"]
        == "sent"
    )


def test_short_cache_and_invalid_kind(env):
    settings, repo, service, _, _ = env
    service.build_tldr(VIDEO)
    assert (
        enqueue(settings, repo, service, VIDEO, 123, 123, 2, output_kind="tldr")["status"]
        == "queued"
    )
    with pytest.raises(ValueError):
        enqueue(settings, repo, service, VIDEO, 123, 123, 3, output_kind="unexpected")


@pytest.mark.parametrize("kind", ["pdf", "tldr"])
def test_no_daily_video_quota_after_many_completed_reports(env, kind, monkeypatch):
    settings, repo, service, calls, _ = env
    monkeypatch.setenv("VIDEO_REPORT_MAX_DAILY_NEW", "0")
    assert not hasattr(Settings(_env_file=None), "video_report_max_daily_new")
    now = time.time()
    with repo.connection() as db:
        db.executemany(
            "INSERT INTO video_pdf_jobs(video_id,user_id,chat_id,request_message_id,"
            "status,created_at,updated_at) VALUES(?,123,123,?,'sent',?,?)",
            [(f"v{i:010d}", 1000 + i, now, now) for i in range(125)],
        )
    result = enqueue(settings, repo, service, VIDEO, 123, 123, 777, output_kind=kind)
    assert result["status"] == "queued"
    assert not calls  # Accepting a request does not invoke paid providers.


def test_pending_queue_has_no_count_limit(env):
    settings, repo, service, _, _ = env
    for i in range(25):
        assert enqueue(settings, repo, service, f"v{i:010d}", 123, 123, i)["status"] == "queued"
    assert enqueue(settings, repo, service, VIDEO, 123, 123, 99)["status"] == "queued"


@pytest.mark.parametrize("kind", ["tldr", "pdf"])
def test_disabled_video_budget_preserves_cost_tracking(env, kind):
    settings, repo, service, calls, _ = env
    assert Settings(_env_file=None).video_report_budget_enabled is False
    settings.video_report_budget_enabled = False
    settings.budget_hard_stop_usd = 0
    settings.video_report_max_cost_usd = 0
    build = service.build_tldr if kind == "tldr" else service.build
    assert build(VIDEO)
    assert calls == ["transcript", kind]
    assert repo.current_month_cost() == pytest.approx(0.0085)
    assert build(VIDEO)
    assert calls == ["transcript", kind]


def test_short_delivery_uncertain_is_not_resent(env):
    settings, repo, service, _, bot = env
    bot.send_message.side_effect = ConnectionError()
    enqueue(settings, repo, service, VIDEO, 123, 123, 10, output_kind="tldr")
    worker = VideoReportWorker(bot, settings, repo, service=service)
    asyncio.run(worker.tick())
    worker.recover()
    assert not asyncio.run(worker.tick())
    with repo.connection() as db:
        assert db.execute("SELECT status FROM video_pdf_jobs").fetchone()[0] == "uncertain"
    bot.send_document.assert_not_awaited()


def test_short_unknown_model_response_is_not_retried(env):
    _, _, service, calls, _ = env

    def fail(*args):
        calls.append("timeout")
        raise TimeoutError()

    service.transport = fail
    for _ in range(2):
        with pytest.raises(ReportError, match="uncertain"):
            service.build_tldr(VIDEO)
    assert calls == ["transcript", "timeout"]


@pytest.mark.parametrize("kind", ["pdf", "tldr"])
def test_payment_rejection_waits_for_new_request_and_reuses_transcript(env, kind):
    settings, repo, service, calls, bot = env
    successful = service.transport

    def rejected(*args):
        calls.append("rejected")
        raise HTTPError("https://openrouter.ai/api/v1/chat/completions", 402, "secret", {}, None)

    service.transport = rejected
    first = enqueue(settings, repo, service, VIDEO, 123, 123, 10, output_kind=kind)
    worker = VideoReportWorker(bot, settings, repo, service=service)
    assert asyncio.run(worker.tick())
    assert not asyncio.run(worker.tick())  # No automatic retry or another paid call.
    with repo.connection() as db:
        row = db.execute("SELECT status,error FROM video_pdf_jobs WHERE id=?", (first["id"],))
        assert tuple(row.fetchone()) == ("failed", "insufficient_credits")
    assert "OpenRouter" in bot.edit_message_text.call_args.kwargs["text"]
    assert repo.current_month_cost() == pytest.approx(0.0075)
    errors = list((service.root / VIDEO).rglob("llm-errors/*.json"))
    assert len(errors) == 1 and "secret" not in errors[0].read_text()

    # A new user request after topping up can resume even after a process restart.
    restored = VideoReportService(
        settings, repo, monid_runner=service.monid_runner, transport=successful
    )
    second = enqueue(settings, repo, restored, VIDEO, 123, 123, 11, output_kind=kind)
    assert asyncio.run(VideoReportWorker(bot, settings, repo, service=restored).tick())
    with repo.connection() as db:
        assert (
            db.execute("SELECT status FROM video_pdf_jobs WHERE id=?", (second["id"],)).fetchone()[
                0
            ]
            == "sent"
        )
    assert calls == ["transcript", "rejected", kind]
    assert repo.current_month_cost() == pytest.approx(0.0085)
    assert errors[0].exists()  # The rejected attempt remains auditable.
    if kind == "pdf":
        bot.send_document.assert_awaited_once()


@pytest.mark.parametrize("kind", ["pdf", "tldr"])
@pytest.mark.parametrize("http_status", [429, 500])
def test_other_http_errors_remain_uncertain_and_reserved(env, kind, http_status):
    _, repo, service, calls, _ = env

    def failed(*args):
        calls.append("http-failure")
        raise HTTPError("https://openrouter.ai/api/v1/chat/completions", http_status, "", {}, None)

    service.transport = failed
    build = service.build if kind == "pdf" else service.build_tldr
    for _ in range(2):
        with pytest.raises(ReportError, match="uncertain"):
            build(VIDEO)
    assert calls == ["transcript", "http-failure"]
    assert repo.current_month_cost() > 0.0075
    error = json.loads(next((service.root / VIDEO).rglob("llm-errors/*.json")).read_text())
    assert error["http_status"] == http_status
    assert error["rejected_before_dispatch"] is False


@pytest.mark.parametrize("kind", ["pdf", "tldr"])
def test_old_payment_rejection_cannot_unlock_a_later_uncertain_attempt(env, kind):
    _, repo, service, calls, _ = env
    build = service.build if kind == "pdf" else service.build_tldr

    def failed(*args):
        calls.append("attempt")
        if calls.count("attempt") == 1:
            raise HTTPError("https://openrouter.ai/api/v1/chat/completions", 402, "", {}, None)
        raise TimeoutError()

    service.transport = failed
    with pytest.raises(ReportError, match="insufficient_credits"):
        build(VIDEO)
    for _ in range(2):
        with pytest.raises(ReportError, match="uncertain"):
            build(VIDEO)
    assert calls == ["transcript", "attempt", "attempt"]
    assert repo.current_month_cost() > 0.0075


@pytest.mark.parametrize("kind", ["pdf", "tldr"])
def test_http_200_error_body_is_not_treated_as_unbilled_rejection(env, kind):
    _, repo, service, calls, _ = env

    def failed(*args):
        calls.append("response-error")
        return {"error": {"code": 402, "message": "Failure after accepting the request"}}

    service.transport = failed
    build = service.build if kind == "pdf" else service.build_tldr
    for _ in range(2):
        with pytest.raises(ReportError, match="invalid_output"):
            build(VIDEO)
    assert calls == ["transcript", "response-error"]
    assert repo.current_month_cost() > 0.0075


def test_pdf_and_short_use_same_lock(env):
    _, _, service, calls, _ = env
    folder = service.root / VIDEO
    folder.mkdir(parents=True)
    with (folder / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for operation in (service.build, service.build_tldr):
            with pytest.raises(ReportError, match="processing"):
                operation(VIDEO)
    assert not calls


@pytest.mark.parametrize(
    "value",
    [
        "Без заголовка",
        "## TL;DR\n\n",
        "## TL;DR\n\nАбзац.\n\nВторой.",
        SHORT + "\n## Подробно\nТекст",
        SHORT + " [[E0001]]",
        SHORT + " https://example.org",
        "## TL;DR\n\n" + "а" * 3901,
    ],
)
def test_short_rejects_incomplete_or_oversized_format(value):
    with pytest.raises(ValueError):
        extract_tldr(value)


def test_tldr_escapes_html_and_full_extraction():
    assert extract_tldr(FULL, full_report=True) == LEAD
    assert telegram_tldr("Сравнение: 1 < 2 & 3.") == "<b>TL;DR</b>\n\nСравнение: 1 &lt; 2 &amp; 3."


def test_upgrade_preserves_old_jobs_and_idempotence(tmp_path):
    repo = Repository(tmp_path / "legacy.sqlite")
    # Build exactly the old schema, then insert sent and running legacy jobs.
    with sqlite3.connect(repo.database_path) as db:
        db.execute(
            "CREATE TABLE schema_migrations(version TEXT PRIMARY KEY,applied_at TEXT NOT NULL)"
        )
        for path in sorted(repo.migrations_path.glob("*.sql")):
            if path.stem >= "0017":
                break
            db.executescript(path.read_text())
            db.execute("INSERT INTO schema_migrations VALUES(?,datetime('now'))", (path.stem,))
        db.execute(
            "INSERT INTO video_pdf_jobs(id,video_id,user_id,chat_id,request_message_id,"
            "status,document_message_id,created_at,updated_at) "
            "VALUES(15,?,123,123,222,'sent',500,1,1)",
            (VIDEO,),
        )
        db.execute(
            "INSERT INTO video_pdf_jobs(id,video_id,user_id,chat_id,request_message_id,"
            "status,created_at,updated_at) VALUES(16,?,123,123,333,'running',1,1)",
            (VIDEO,),
        )
    assert repo.migrate() == [
        path.stem for path in sorted(repo.migrations_path.glob("*.sql")) if path.stem >= "0017"
    ]
    assert repo.migrate() == []
    with repo.connection() as db:
        rows = [dict(x) for x in db.execute("SELECT * FROM video_pdf_jobs ORDER BY id")]
        assert rows[0]["id"] == 15 and rows[0]["document_message_id"] == 500
        assert rows[1]["status"] == "running"
        assert all(r["output_kind"] == "pdf" for r in rows)
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
