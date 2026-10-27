import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram.dispatcher.event.bases import SkipHandler

from app.config import Settings
from app.db.repository import Repository
from app.details.provider_guard import DetailProviderError
from app.media.downloads import media_link
from app.personalization.daily import DailyContext, DailyPolicy
from app.reposts import media, provider, store
from app.reposts.history import ExportParser, export_samples, import_samples
from app.reposts.learning import CurationIndex, load_index
from app.reposts.service import RepostWorker
from app.telegram_bot.bot import build_router
from app.telegram_bot.news_panel import get_news_panel, show_base
from app.telegram_bot.x_reposts import RepostFilter


@pytest.fixture
def env(tmp_path):
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        allowed_user_id=123,
        openrouter_api_key="test",
        x_reposts_enabled=True,
    )
    repo = Repository(settings.database_path)
    repo.migrate()
    return settings, repo


def enqueue(env, ident="1234567890", message_id=25):
    return store.enqueue(
        env[1],
        owner=123,
        message_id=message_id,
        link=media_link("https://x.com/author/status/" + ident),
    )[0]


def receipt(ident="1234567890", text="An open source AI agent codes software in a repository."):
    return {
        "status": "COMPLETED",
        "output": {
            "id": ident,
            "text": text,
            "author": {"screen_name": "author"},
            "created_at": "Mon Oct 26 04:12:09 +0000 2026",
            "entities": {
                "media": [
                    {"type": "photo", "media_url_https": "https://pbs.twimg.com/media/photo.jpg"}
                ]
            },
        },
    }


def answer(paragraphs=None):
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(
                        {
                            "paragraphs": paragraphs
                            or [
                                "Открытый ИИ-агент пишет код в репозитории "
                                "и показывает результат работы."
                            ],
                            "thesis": "ИИ-агент самостоятельно меняет программный код",
                        }
                    )
                },
            }
        ],
        "usage": {"cost": 0.001},
    }


def test_enqueue_duplicate_and_untrusted_source(env):
    row = enqueue(env)
    repeated, created = store.enqueue(
        env[1],
        owner=123,
        message_id=100,
        link=media_link("https://twitter.com/another/status/1234567890?s=20"),
    )
    assert row["id"] == repeated["id"] and not created
    with env[1].connection() as db:
        assert db.execute("SELECT count(*) FROM x_curation_samples").fetchone()[0] == 1
        assert tuple(
            db.execute("SELECT enabled,candidate_enabled,role FROM sources").fetchone()
        ) == (0, 0, "own_history")
        assert db.execute("SELECT status FROM events").fetchone()[0] == "user_requested"
        assert not db.execute("PRAGMA foreign_key_check").fetchall()


def test_prepare_cached_full_long_response_no_repayment(env):
    row = enqueue(env)
    fetch = Mock(return_value=receipt())
    transport = Mock(return_value=answer(["Содержательный абзац с объяснением. " * 35] * 4))
    post, translated = provider.prepare(*env, row, fetcher=fetch, transport=transport)
    assert len("".join(translated["paragraphs"])) > 3900
    again = provider.prepare(*env, store.get(env[1], row["id"]), fetcher=fetch, transport=transport)
    assert again == (post, translated)
    assert fetch.call_count == transport.call_count == 1
    assert env[1].current_month_cost() == pytest.approx(0.002)


def test_paid_response_parse_repair_reuses_receipt(env):
    row = enqueue(env)
    transport = Mock(return_value=answer())
    fetch = Mock(return_value=receipt())
    original = provider.normalize
    with pytest.MonkeyPatch.context() as m:
        m.setattr(provider, "normalize", Mock(side_effect=ValueError("old schema")))
        with pytest.raises(ValueError):
            provider.prepare(*env, row, fetcher=fetch, transport=transport)
    assert provider.normalize is original
    provider.prepare(*env, row, fetcher=fetch, transport=transport)
    assert fetch.call_count == 1


def test_legacy_cli_raw_tweet_status_is_not_a_pending_run(env, monkeypatch):
    row = enqueue(env)
    raw = {**receipt()["output"], "status": "active"}
    path = env[0].data_dir / "x-reposts" / (row["tweet_id"] + ".json")
    path.parent.mkdir()
    path.write_text(json.dumps(raw))
    paid = Mock(side_effect=AssertionError("Do not refetch a saved post"))
    monkeypatch.setattr(provider.subprocess, "run", paid)
    envelope = provider.fetch(env[0], row)
    assert envelope["status"] == "COMPLETED"
    assert provider.normalize(envelope, row["tweet_id"])["text"] == raw["text"]
    assert not paid.called
    with pytest.raises(ValueError):
        provider.completed_receipt({"runId": "pending", "status": "RUNNING"}, row["tweet_id"])


def test_uncertain_paid_request_cannot_be_replayed(env):
    row = enqueue(env)
    transport = Mock(side_effect=TimeoutError())
    fetch = Mock(return_value=receipt())
    with pytest.raises(DetailProviderError):
        provider.prepare(*env, row, fetcher=fetch, transport=transport)
    with pytest.raises(DetailProviderError):
        provider.prepare(*env, store.get(env[1], row["id"]), fetcher=fetch, transport=transport)
    assert transport.call_count == fetch.call_count == 1


def test_zero_charge_failure_uses_one_cached_fallback(env):
    row = enqueue(env)
    failure = {
        "status": "COMPLETED",
        "runId": "failed-run",
        "cost": {"value": 0},
        "providerResponse": {"httpStatus": 400},
        "output": {"detail": "unavailable"},
    }
    first = Mock(side_effect=lambda *_: provider.completed_receipt(failure, row["tweet_id"]))
    fallback = Mock(
        return_value={
            "status": "COMPLETED",
            "runId": "fallback-run",
            "cost": {"value": 0.00022},
            "output": {
                "found": True,
                "data": {
                    "id": row["tweet_id"],
                    "text": "An AI agent writes code.",
                    "authorHandle": "writer",
                    "createdUtc": 1791808395,
                    "lang": "en",
                    "media": [],
                },
            },
        }
    )
    transport = Mock(return_value=answer())
    post, translated = provider.prepare(
        *env, row, fetcher=first, fallback_fetcher=fallback, transport=transport
    )
    assert post["author"] == "writer" and post["published_at"]
    assert not post["media"] and translated["paragraphs"]
    # Even a stale row reuses both completed calls; the failed primary is not paid again.
    provider.prepare(*env, row, fetcher=first, fallback_fetcher=fallback, transport=transport)
    assert first.call_count == fallback.call_count == transport.call_count == 1
    assert env[1].current_month_cost() == pytest.approx(0.00122)


@pytest.mark.parametrize("charged", [None, 0.001])
def test_uncertain_or_charged_fetch_never_starts_fallback(env, charged):
    row = enqueue(env)
    fallback = Mock(side_effect=AssertionError("Fallback must not spend again"))
    failure = {
        "status": "COMPLETED",
        "runId": "bad-run",
        "cost": {"value": charged},
        "output": {"error": "not a tweet"},
    }
    fetch = Mock(side_effect=lambda *_: provider.completed_receipt(failure, row["tweet_id"]))
    with pytest.raises(DetailProviderError):
        provider.prepare(*env, row, fetcher=fetch, fallback_fetcher=fallback)
    assert not fallback.called


def test_cli_stdout_receipt_preserves_confirmed_zero_cost(env, monkeypatch):
    row = enqueue(env)
    failure = {
        "status": "COMPLETED",
        "runId": "failed-run",
        "cost": {"value": 0},
        "providerResponse": {"httpStatus": 400},
        "output": {"detail": "unavailable"},
    }

    def cli(args, **kwargs):
        from pathlib import Path

        Path(args[args.index("-o") + 1]).write_text(json.dumps(failure["output"]))
        return SimpleNamespace(stdout=json.dumps(failure), returncode=0)

    monkeypatch.setattr(provider.subprocess, "run", cli)
    with pytest.raises(provider.ConfirmedPostFailure) as exc:
        provider.fetch(*env[:1], row)
    assert exc.value.cost_usd == 0
    saved = json.loads((env[0].data_dir / "x-reposts" / (row["tweet_id"] + ".json")).read_text())
    assert saved["runId"] == "failed-run" and saved["cost"]["value"] == 0


def test_wrong_post_and_media_hosts():
    with pytest.raises(ValueError):
        provider.normalize(receipt(), "9999")
    for url in [
        "https://pbs.twimg.com.evil.test/a",
        "http://pbs.twimg.com/a",
        "https://video.twimg.com@127.0.0.1/a",
        "https://127.0.0.1/a",
        "file:///tmp/x",
    ]:
        assert not provider.safe_media_url(url)
    data = receipt(text="")
    assert provider.normalize(data, "1234567890")["media"][0]["kind"] == "photo"


def test_video_variants_and_no_thumbnail_substitution():
    data = receipt()
    data["output"]["entities"]["media"] = [
        {
            "type": "video",
            "media_url_https": "https://pbs.twimg.com/thumb.jpg",
            "video_info": {
                "duration_millis": 12000,
                "variants": [
                    {
                        "content_type": "video/mp4",
                        "url": "https://video.twimg.com/low.mp4",
                        "bitrate": 200,
                    },
                    {
                        "content_type": "video/mp4",
                        "url": "https://video.twimg.com/high.mp4",
                        "bitrate": 800,
                    },
                    {"content_type": "video/mp4", "url": "https://localhost/x", "bitrate": 900},
                ],
            },
        }
    ]
    media_row = provider.normalize(data, "1234567890")["media"][0]
    assert media_row["kind"] == "video"
    assert len(media_row["variants"]) == 2
    assert media_row["variants"][0]["url"].endswith("high.mp4")


def test_html_export_joined_messages_and_errors_do_not_become_likes(env):
    parser = ExportParser()
    parser.feed(
        '<div class="message default" id="message1"><div class="date details" '
        'title="26.02.2026 17:00:05 UTC+03:00"></div><div class="from_name">Owner</div>'
        '<div class="text"><a href="https://x.com/author/status/1234567890">link</a></div></div>'
        '<div class="message default" id="message2"><div class="from_name">X Repost</div>'
        '<div class="text">Полезный текст про программирование и оркестрацию ИИ-агентов'
        "<br><br>Второй абзац.</div></div>"
        '<div class="message default joined" id="message3">'
        '<div class="text">❌ Message text is empty</div></div>'
    )
    samples, errors = export_samples(parser.messages, owner_name="Owner")
    assert len(samples) == 1 and errors == 1
    assert "\n\n" in samples[0]["text"]
    assert "empty" not in samples[0]["text"]
    assert import_samples(env[1], 123, samples) == 1
    assert import_samples(env[1], 123, samples) == 0
    with env[1].connection() as db:
        assert db.execute("SELECT vote FROM x_curation_samples").fetchone()[0] is None


def test_strong_rating_overrides_weak_link_and_does_not_count_duplicates(env):
    row = enqueue(env)
    store.save_post(env[1], row, provider.normalize(receipt(), row["tweet_id"]))
    now = datetime.now(UTC) + timedelta(seconds=1)
    text = "An open source AI agent codes software in a repository."
    weak = load_index(env[1], 123, now).adjustment(text)["adjustment"]
    store.vote(env[1], row, 1)
    positive = load_index(env[1], 123, now).adjustment(text)["adjustment"]
    store.vote(env[1], row, -1)
    index = load_index(env[1], 123, now)
    assert index.adjustment(text)["adjustment"] < 0 < weak < positive
    assert len(index.samples) == 1
    assert not load_index(env[1], 999, now).samples
    assert index.adjustment("An unrelated recipe for chocolate cake")["adjustment"] == 0


def test_daily_total_stays_bounded_with_curation():
    now = datetime.now(UTC)
    text = "Claude Code autonomous agent harness open source repository coding"
    rows = [
        {
            "tweet_id": str(i),
            "text": text,
            "author": str(i),
            "vote": 1,
            "selected_at": now.isoformat(),
            "voted_at": now.isoformat(),
            "provenance": "owner_link",
        }
        for i in range(20)
    ]
    context = DailyContext(
        DailyPolicy(priorities={"ai_coding": 2, "agent_systems": 2}),
        [],
        now,
        curations=CurationIndex(rows, now),
    )
    adjusted = context.adjustment({"headline": text})
    assert 0 < adjusted["curation"]["adjustment"] <= 0.45
    assert adjusted["adjustment"] <= context.policy.max_adjustment


def test_owner_feedback_routing_and_explicit_download(env):
    settings, repo = env
    handlers = build_router(settings, repo).message.handlers
    callback = next(h.callback for h in handlers if h.callback.__name__ == "x_repost_incoming")
    m = SimpleNamespace(
        text="https://x.com/a/status/1234567890",
        caption=None,
        entities=[],
        caption_entities=[],
        from_user=SimpleNamespace(id=999),
        chat=SimpleNamespace(id=999, type="private"),
        answer=AsyncMock(),
    )
    asyncio.run(callback(m))
    m.answer.assert_not_called()
    m.from_user.id = 123
    repo.feedback_session_for_user = lambda _: {"state": "awaiting_comment"}
    with pytest.raises(SkipHandler):
        asyncio.run(callback(m))
    m.text = "/download https://x.com/a/status/1234567890"
    assert not asyncio.run(RepostFilter()(m))


def test_worker_reuses_news_panel_and_keeps_full_media(env, monkeypatch):
    row = enqueue(env)
    with env[1].connection() as db:
        db.execute("UPDATE x_reposts SET panel_message_id=80 WHERE id=?", (row["id"],))
    row = store.get(env[1], row["id"])
    post = provider.normalize(receipt(), row["tweet_id"])
    trans = {"paragraphs": ["Полезный подробный абзац. " * 180], "thesis": "Агенты меняют код"}
    store.save_post(env[1], row, post)
    with env[1].connection() as db:
        db.execute(
            "UPDATE x_reposts SET translation_json=? WHERE id=?", (json.dumps(trans), row["id"])
        )
    row = store.get(env[1], row["id"])
    monkeypatch.setattr(
        media, "acquire", lambda *a: [{"kind": "photo", "file_id": "cached", "index": 0}]
    )
    bot = SimpleNamespace(
        edit_message_text=AsyncMock(),
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=81)),
        send_photo=AsyncMock(
            return_value=SimpleNamespace(
                message_id=82,
                photo=[SimpleNamespace(file_id="cached", width=600, height=400)],
                video=None,
            )
        ),
    )
    assert asyncio.run(RepostWorker(bot, *env).process(row))
    saved = store.get(env[1], row["id"])
    panel = get_news_panel(env[1], saved["delivery_id"])
    assert panel["reader_context"]["x_repost_id"] == row["id"]
    assert len(panel["baseline_html"]) > 3900
    args = bot.edit_message_text.call_args.kwargs
    assert args["message_id"] == 80 and "rich_message" not in args
    assert bot.send_photo.call_args.kwargs["photo"] == "cached"
    assert bot.send_message.call_count >= 1
    assert store.get(env[1], row["id"])["native_kind"] == "text"
    buttons = str(args["reply_markup"])
    assert "Подробнее" in buttons and "Задать вопрос" in buttons and "Разобрать тезис" in buttons
    assert asyncio.run(show_base(bot, env[1], panel))
    assert env[1].current_month_cost() == 0


def ready_repost(env):
    from app.reposts.service import bind_card
    from app.telegram_bot.news_panel import ensure_news_panel

    row = enqueue(env)
    row["panel_message_id"] = 80
    post = provider.normalize(receipt(), row["tweet_id"])
    translated = {"paragraphs": ["Автор показывает работу ИИ-агента."], "thesis": "Агент пишет код"}
    store.save_post(env[1], row, post)
    with env[1].connection() as db:
        db.execute(
            "UPDATE x_reposts SET panel_message_id=80,status='ready',translation_json=?,"
            "media_json=? WHERE id=?",
            (json.dumps(translated), '[{"kind":"photo","file_id":"cached"}]', row["id"]),
        )
    delivery = bind_card(env[1], row, post, translated)
    return store.get(env[1], row["id"]), ensure_news_panel(env[1], delivery)


def test_questions_navigation_restores_media_and_thesis_after_another_question(env):
    from test_news_reader_flow import article_result

    from app.db.news_questions import claim_input, open_session, queue_question
    from app.details.worker import DetailWorker
    from app.telegram_bot.news_panel import show_result

    row, panel = ready_repost(env)
    settings, repo = env
    bot = SimpleNamespace(edit_message_text=AsyncMock())
    first_article_id = None
    for message_id, question in [
        (-row["id"], "Разбери тезис про агентов"),
        (42, "Как это применять?"),
    ]:
        open_session(repo, panel)
        assert claim_input(repo, panel, message_id, "text", max_daily=10)
        request = queue_question(repo, panel, message_id, question)

        def builder(*args, request=request, message_id=message_id, **kwargs):
            result = article_result(request)
            result.generation_key += ":" + str(message_id)
            return result

        assert DetailWorker(settings, repo, builder=builder).run_once().status == "ready"
        request = repo.get_detail_request(request["request_id"])
        article = repo.get_detailed_article(request["article_id"])
        assert asyncio.run(show_result(bot, repo, panel, request, article))
        repo.mark_detail_request_sent(request["request_id"])
        assert not bot.edit_message_text.call_args.kwargs["rich_message"].media
        assert asyncio.run(show_base(bot, repo, panel))
        assert (
            bot.edit_message_text.call_args.kwargs["rich_message"].media[0].media.media == "cached"
        )
        if message_id < 0:
            first_article_id = article["id"]
    previous, restored = store.saved_thesis(repo, panel, -row["id"])
    assert previous and restored["article_id"] == first_article_id
    assert restored["reader_context"]["user_question"] == "Разбери тезис про агентов"
    assert restored["cache_hit"]


def test_repost_claim_is_readable_but_keeps_its_evidence_role(env):
    from test_news_questions import response

    from app.details.explainer import build_news_explanation

    row, panel = ready_repost(env)
    calls = []

    def transport(payload, key):
        context = json.loads(payload["messages"][1]["content"])
        sources = context["read_sources"]
        assert len(sources) == 1
        assert sources[0]["evidence_role"] == "author_claim_only_not_verified"
        assert "не истинности" not in context["news"]  # No UI boilerplate in the publication.
        calls.append(payload)
        return response(
            "finish",
            {
                "title": "Что показывает автор",
                "paragraphs": [
                    {
                        "text": "Автор описывает агента, который меняет код в репозитории. " * 5,
                        "source_ids": [sources[0]["id"]],
                        "evidence_quote": sources[0]["text"],
                    }
                ]
                * 3,
            },
        )

    result = build_news_explanation(
        *env,
        row["event_id"],
        depth="expanded",
        reader_context=panel["reader_context"],
        transport=transport,
    )
    assert result.validation_status == "valid" and len(calls) == 1
    assert result.provider_response_metadata["search_count"] == 0
    assert not env[1].intelligence_events(
        since=(datetime.now(UTC) - timedelta(days=1)).isoformat(),
        until=(datetime.now(UTC) + timedelta(days=1)).isoformat(),
    )  # A translated opinion must not enter the verified weekly/news corpus.


def test_retry_does_not_overwrite_an_open_analysis(env):
    row, panel = ready_repost(env)
    with env[1].connection() as db:
        db.execute(
            "UPDATE news_panels SET view='expanded' WHERE delivery_id=?", (panel["delivery_id"],)
        )
    with pytest.raises(ValueError, match="исходной"):
        store.retry(env[1], row)
    assert store.get(env[1], row["id"])["status"] == "ready"


def test_media_budget_stream_limit_and_redirect(tmp_path):
    import io

    class Response(io.BytesIO):
        headers = {}

    opener = SimpleNamespace(open=Mock(return_value=Response(b"too many bytes")))
    target = tmp_path / "video.mp4"
    with pytest.raises(ValueError, match="too_large"):
        media.download_file("https://video.twimg.com/a.mp4", target, cap=5, opener=opener)
    assert not target.exists() and not target.with_suffix(".mp4.part").exists()
    with pytest.raises(ValueError, match="outside X"):
        media.SafeRedirect().redirect_request(None, None, 302, "", {}, "http://127.0.0.1/x")


def test_future_vote_is_not_used_and_resubmitted_history_becomes_recent(env):
    row = enqueue(env)
    store.save_post(env[1], row, provider.normalize(receipt(), row["tweet_id"]))
    now = datetime.now(UTC)
    with env[1].connection() as db:
        db.execute(
            "UPDATE x_curation_samples SET vote=-1,voted_at=?",
            ((now + timedelta(days=1)).isoformat(),),
        )
    index = load_index(env[1], 123, now)
    assert index.samples[0]["vote"] is None
    assert (
        index.adjustment("An open source AI agent codes software in a repository.")["adjustment"]
        > 0
    )
    sample = dict(index.samples[0], tweet_id="9876543210", selected_at="2026-01-01T00:00:00+00:00")
    assert import_samples(env[1], 123, [sample]) == 1
    enqueue(env, ident="9876543210", message_id=77)
    with env[1].connection() as db:
        selected = db.execute(
            "SELECT selected_at FROM x_curation_samples WHERE tweet_id='9876543210'"
        ).fetchone()[0]
    assert datetime.fromisoformat(selected) >= now


def test_repeated_topic_selections_work_across_languages_without_one_author_dominating():
    now = datetime.now(UTC)
    rows = [
        {
            "tweet_id": str(n),
            "text": "Японские роботы помогают исследователям изучать движение.",
            "author": str(n % 2),
            "vote": None,
            "selected_at": now.isoformat(),
            "voted_at": None,
            "provenance": "owner_link",
        }
        for n in range(4)
    ]
    result = CurationIndex(rows, now).adjustment("Embodied intelligence and robotic manipulation")
    assert not result["examples"] and 0 < result["adjustment"] < 0.12
    assert result["topics"][0]["topic"] == "robotics"
    one_author = [{**r, "author": "one"} for r in rows]
    assert (
        CurationIndex(one_author, now).adjustment("Embodied intelligence and robotic manipulation")[
            "adjustment"
        ]
        == 0
    )


def test_export_keeps_original_when_old_bot_failed_only_to_translate():
    rows = [
        {
            "id": "m1",
            "sender": "Owner",
            "date": "26.02.2026 17:00:05 UTC+03:00",
            "text": "",
            "urls": ["https://x.com/author/status/1234567890"],
        },
        {
            "id": "m2",
            "sender": "X Repost",
            "date": "",
            "urls": [],
            "text": "⚠️ Не удалось перевести, вот оригинал:\n\n"
            "Robotics agents recover from error states.",
        },
    ]
    samples, _ = export_samples(rows, owner_name="Owner")
    assert samples[0]["text"] == "Robotics agents recover from error states."
