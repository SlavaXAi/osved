import asyncio
import hashlib
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr
from test_news_reader_flow import env as env  # noqa: F401

from app.db.news_questions import claim_input, open_session, queue_question, reply_session
from app.details.explainer import build_news_explanation, provider_fingerprint, public_url
from app.details.provider_guard import DetailProviderError, DurableDetailProvider
from app.details.transcription import transcribe_question
from app.details.worker import DetailWorker
from app.research.detail_policy import article_reference_fingerprint
from app.research.web import ScrapeResponse, SearchResponse
from app.telegram_bot.bot import send_ready_detail_articles
from app.telegram_bot.news_panel import ensure_news_panel, get_news_panel
from app.telegram_bot.news_reader import handle_news_action

QUOTE = "This model generates conversations and supports tool calling with a million token context."
BODY = QUOTE + "\n\n" + "The documentation describes compatible APIs and adjustable reasoning. " * 9
QUESTION = "Что умеет модель и как её применять?"


def setup(env):
    s, r, delivery, bot, router = env
    s.openrouter_api_key = SecretStr("offline")
    panel = ensure_news_panel(r, delivery["delivery_id"])
    doc_id = r.upsert_research_document(
        canonical_url="https://example.org/story",
        url="https://example.org/story",
        title="Model documentation",
        publisher="example.org",
        source_class="primary",
        content_text=BODY,
        retrieved_at=datetime.now(UTC).isoformat(),
        expires_at=(datetime.now(UTC) + timedelta(days=7)).isoformat(),
        metadata={"content_kind": "scraped_fulltext"},
    )
    r.attach_research_document(
        event_id=delivery["event_id"],
        research_document_id=doc_id,
        evidence_role="primary",
        relevance=1,
    )
    return s, r, panel, bot, router, doc_id


def finish(doc_id, bad_id=None, quote=QUOTE):
    text = (
        "Это модель для работы с разговорным контекстом и вызова инструментов. "
        "Согласно документации разработчика, её можно подключать к приложению через API. "
        "В таком сценарии приложение передаёт ей историю диалога и получает ответ или "
        "запрос на действие. Результат действия возвращается в контекст следующего шага. "
    )
    return {
        "title": "Что умеет модель",
        "paragraphs": [
            {"text": text, "source_ids": [bad_id or doc_id], "evidence_quote": quote},
            {
                "text": text
                + "Это полезно при создании помощника с доступом к функциям приложения.",
                "source_ids": [bad_id or doc_id],
                "evidence_quote": quote,
            },
        ],
    }


def response(name, args, number=1):
    return {
        "id": f"offline-{number}",
        "usage": {"cost": 0.001, "is_byok": False},
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call-{number}",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)},
                        }
                    ],
                },
            }
        ],
    }


def build(env, transport, depth="expanded", **kwargs):
    s, r, panel, _, _, _ = setup(env)
    return build_news_explanation(
        s,
        r,
        panel["reader_context"]["baseline_event_id"],
        depth=depth,
        reader_context={**panel["reader_context"], "reader_mode": "explainer", **kwargs},
        transport=transport,
    )


def test_one_primary_source_can_explain_release_without_new_domains_or_benchmarks(env):
    s, r, panel, bot, router, doc = setup(env)
    calls = []

    def transport(payload, key):
        calls.append(payload)
        return response("finish", finish(doc))

    result = build_news_explanation(
        s,
        r,
        panel["reader_context"]["baseline_event_id"],
        depth="expanded",
        reader_context={**panel["reader_context"], "reader_mode": "explainer"},
        transport=transport,
    )
    assert result.validation_status == "valid"
    assert result.document_ids == [doc] and result.provider_response_metadata["search_count"] == 0
    assert len(calls) == 1 and "\n\n" in result.rendered_html
    assert result.provider_response_metadata["source_quota"] is False
    # A stale worker or repeat traversal reuses the exact already-paid model response.
    result2 = build_news_explanation(
        s,
        r,
        panel["reader_context"]["baseline_event_id"],
        depth="expanded",
        reader_context={**panel["reader_context"], "reader_mode": "explainer"},
        transport=transport,
    )
    assert (
        len(calls) == 1 and result2.provider_response_metadata["incremental_attempt_cost_usd"] == 0
    )


def test_agent_repairs_unknown_source_and_invented_quote_without_displaying_them(env):
    s, r, panel, _, _, doc = setup(env)
    seen = []

    def transport(payload, key):
        seen.append(payload)
        args = (
            finish(doc, bad_id=999)
            if len(seen) == 1
            else finish(doc, quote="invented quote 1234567890")
            if len(seen) == 2
            else finish(doc)
        )
        return response("finish", args, len(seen))

    result = build_news_explanation(
        s,
        r,
        panel["reader_context"]["baseline_event_id"],
        depth="expanded",
        reader_context=panel["reader_context"],
        transport=transport,
    )
    assert len(seen) == 3 and result.document_ids == [doc]
    assert "invented quote" not in result.rendered_html


def test_model_can_search_read_and_answer_the_actual_question(env):
    s, r, panel, _, _, doc = setup(env)
    calls = []

    class Web:
        def search(self, q, **kwargs):
            assert "pricing" in q
            return SearchResponse(
                q, [{"url": "https://example.net/pricing", "title": "Pricing"}], 0.0005
            )

        def scrape(self, url):
            assert url == "https://example.net/pricing"
            return ScrapeResponse(url, BODY, {"title": "Pricing"}, 0.0009)

    def transport(payload, key):
        calls.append(payload)
        assert QUESTION in json.dumps(payload, ensure_ascii=False)
        if len(calls) == 1:
            return response("search", {"query": "model pricing"}, 1)
        if len(calls) == 2:
            return response("read", {"url": "https://example.net/pricing"}, 2)
        with r.connection() as db:
            new_id = db.execute(
                "SELECT id FROM research_documents WHERE url LIKE '%pricing'"
            ).fetchone()[0]
        return response("finish", finish(new_id), 3)

    result = build_news_explanation(
        s,
        r,
        panel["reader_context"]["baseline_event_id"],
        depth="article",
        reader_context={**panel["reader_context"], "user_question": QUESTION},
        transport=transport,
        web_client=Web(),
    )
    assert result.provider_response_metadata["search_count"] == 1
    assert result.provider_response_metadata["scrape_count"] == 1
    assert result.source_urls == ["https://example.net/pricing"]


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/a",
        "http://10.0.0.1/a",
        "https://localhost/a",
        "https://x.local/a",
        "https://user:pass@example.org/a",
    ],
)
def test_agent_cannot_fetch_private_or_credential_urls(url):
    assert public_url(url) is None


def claim(env, question=QUESTION, message_id=9010, voice=None):
    s, r, panel, bot, router, doc = setup(env)
    open_session(r, panel)
    session = reply_session(r, 123, 9001)
    assert claim_input(r, session, message_id, "voice" if voice else "text", max_daily=5)
    return queue_question(r, panel, message_id, question, voice=voice)


def test_questions_are_scoped_to_reply_owner_news_and_message(env):
    s, r, panel, _, _, _ = setup(env)
    open_session(r, panel)
    assert reply_session(r, 999, 9001) is None
    assert reply_session(r, 123, 999) is None
    session = reply_session(r, 123, 9001)
    assert claim_input(r, session, 9010, "text", max_daily=5)
    assert not claim_input(r, session, 9010, "text", max_daily=5)
    with pytest.raises(ValueError, match="дождись"):
        claim_input(r, session, 9011, "voice", max_daily=5)
    request = queue_question(r, panel, 9010, QUESTION)
    assert request["depth"] == "article" and request["reader_context"]["user_question"] == QUESTION
    assert reply_session(r, 123, 9001) is None
    assert request["reader_context"]["provider_fingerprint"] == provider_fingerprint(
        request["reader_context"]
    )


def test_repeated_questions_do_not_share_cache_or_uncertain_call_scope(env):
    first = claim(env)
    r = env[1]
    r.claim_next_detail_request()
    assert r.fail_detail_request(first["id"], error="fixture failed")
    panel = ensure_news_panel(r, first["delivery_id"])
    open_session(r, panel)
    session = reply_session(r, 123, 9001)
    assert claim_input(r, session, 9011, "text", max_daily=5)
    second = queue_question(r, panel, 9011, "Как устроены цены?")
    assert first["id"] == second["id"]
    assert article_reference_fingerprint(first["reader_context"]) != article_reference_fingerprint(
        second["reader_context"]
    )
    assert provider_fingerprint(first["reader_context"]) != provider_fingerprint(
        second["reader_context"]
    )
    with r.connection() as db:
        assert db.execute("SELECT count(*) FROM news_question_inputs").fetchone()[0] == 2


def test_voice_transcription_is_durable_russian_and_does_not_store_audio_in_ledger(env):
    digest = hashlib.sha256(b"fixture-audio").hexdigest()
    request = claim(
        env, question="", voice={"voice_input_sha256": digest, "voice_duration_seconds": 10}
    )
    s, r = env[:2]
    root = s.data_dir / "news-question-audio"
    root.mkdir()
    (root / (digest + ".ogg")).write_bytes(b"fixture-audio")
    reader = request["reader_context"]
    original = article_reference_fingerprint(reader)
    calls = []

    def transport(payload, key):
        calls.append(payload)
        assert payload["language"] == "ru" and payload["input_audio"]["format"] == "ogg"
        return {"text": QUESTION, "usage": {"cost": 0.0001, "is_byok": False}}

    def run():
        guard = DurableDetailProvider(
            s,
            r,
            request["event_id"],
            "article",
            provider_fingerprint(reader),
            request_budget_usd=0.08,
        )
        return transcribe_question(s, r, guard, reader, transport=transport)

    assert run() == QUESTION and run() == QUESTION and len(calls) == 1
    assert article_reference_fingerprint(reader) == original
    with r.connection() as db:
        ledger = " ".join(row[0] for row in db.execute("SELECT metadata_json FROM usage_ledger"))
        assert "fixture-audio" not in ledger and "input_audio" not in ledger
        assert db.execute("SELECT question FROM news_question_inputs").fetchone()[0] == QUESTION


def test_uncertain_voice_result_is_not_paid_again(env):
    digest = hashlib.sha256(b"audio").hexdigest()
    request = claim(
        env, question="", voice={"voice_input_sha256": digest, "voice_duration_seconds": 10}
    )
    s, r = env[:2]
    root = s.data_dir / "news-question-audio"
    root.mkdir()
    (root / (digest + ".ogg")).write_bytes(b"audio")
    reader = request["reader_context"]
    calls = []

    def transport(*a):
        calls.append(1)
        raise TimeoutError()

    for _ in range(2):
        guard = DurableDetailProvider(
            s,
            r,
            request["event_id"],
            "article",
            provider_fingerprint(reader),
            request_budget_usd=0.08,
        )
        with pytest.raises(DetailProviderError, match="unconfirmed"):
            transcribe_question(s, r, guard, reader, transport=transport)
    assert len(calls) == 1


def test_button_asks_for_input_without_starting_paid_research(env):
    s, r, panel, bot, router, doc = setup(env)
    bot.send_message.return_value = SimpleNamespace(message_id=9012)
    q = SimpleNamespace(
        bot=bot,
        answer=AsyncMock(),
        from_user=SimpleNamespace(id=123),
        message=SimpleNamespace(message_id=9001, chat=SimpleNamespace(id=123)),
    )
    asyncio.run(handle_news_action(q, r, s, panel["delivery_id"], "ask"))
    asyncio.run(handle_news_action(q, r, s, panel["delivery_id"], "ask"))
    assert bot.send_message.await_count == 1
    with r.connection() as db:
        assert db.execute("SELECT count(*) FROM detail_requests").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM usage_ledger").fetchone()[0] == 0
    assert reply_session(r, 123, 9012)["delivery_id"] == panel["delivery_id"]


def test_text_reply_runs_worker_and_replaces_original_card_with_navigation(env, monkeypatch):
    s, r, panel, bot, router, doc = setup(env)
    open_session(r, panel)
    handler = next(
        h.callback for h in router.message.handlers if h.callback.__name__ == "news_question_reply"
    )
    message = SimpleNamespace(
        message_id=9014,
        from_user=SimpleNamespace(id=123),
        chat=SimpleNamespace(id=123, type="private"),
        reply_to_message=SimpleNamespace(message_id=9001),
        text=QUESTION,
        voice=None,
        bot=bot,
        answer=AsyncMock(),
    )
    asyncio.run(handler(message))
    request = r.detail_request_for_delivery(panel["delivery_id"], depth="article")
    assert request["reader_context"]["user_question"] == QUESTION

    def builder(settings, repo, event_id, **kwargs):
        kwargs.pop("generation_key", None)
        return build_news_explanation(
            settings, repo, event_id, transport=lambda *a: response("finish", finish(doc)), **kwargs
        )

    assert DetailWorker(s, r, builder=builder).run_once().status == "ready"
    assert asyncio.run(send_ready_detail_articles(bot, r))["sent"] == 1
    assert bot.edit_message_text.call_args.kwargs["message_id"] == 9001
    assert get_news_panel(r, panel["delivery_id"])["view"] == "article"
    labels = [
        b.text
        for row in bot.edit_message_text.call_args.kwargs["reply_markup"].inline_keyboard
        for b in row
    ]
    assert "Запросить разбор" in labels and "Подробнее" in labels and "← Назад" in labels
    asyncio.run(handler(message))
    with r.connection() as db:
        assert db.execute("SELECT count(*) FROM news_question_inputs").fetchone()[0] == 1


def test_cancelled_or_expired_question_cannot_queue_after_download(env):
    _, r, panel, _, _, _ = setup(env)
    open_session(r, panel)
    session = reply_session(r, 123, 9001)
    assert claim_input(r, session, 9010, "voice", max_daily=5)
    with r.connection() as db:
        db.execute("DELETE FROM news_question_sessions")
    with pytest.raises(ValueError, match="отменён"):
        queue_question(r, panel, 9010, QUESTION)
    with r.connection() as db:
        assert db.execute("SELECT count(*) FROM detail_requests").fetchone()[0] == 0


def test_foreign_news_session_is_rejected_before_any_input_or_charge(env):
    _, r, panel, _, _, _ = setup(env)
    with pytest.raises(PermissionError):
        open_session(r, {**panel, "chat_id": 456})
    with pytest.raises(PermissionError):
        claim_input(
            r, {"chat_id": 456, "delivery_id": panel["delivery_id"]}, 9010, "voice", max_daily=5
        )
    with r.connection() as db:
        assert db.execute("SELECT count(*) FROM news_question_inputs").fetchone()[0] == 0


def test_voice_reply_transcribes_then_answers_inside_original_card(env, monkeypatch):
    from app.details import transcription

    s, r, panel, bot, router, doc = setup(env)
    open_session(r, panel)
    handler = next(
        h.callback for h in router.message.handlers if h.callback.__name__ == "news_question_reply"
    )

    async def download(voice, destination):
        destination.write(b"voice-fixture")

    bot.download = AsyncMock(side_effect=download)
    incoming = SimpleNamespace(
        message_id=9015,
        from_user=SimpleNamespace(id=123),
        chat=SimpleNamespace(id=123, type="private"),
        reply_to_message=SimpleNamespace(message_id=9001),
        text=None,
        voice=SimpleNamespace(duration=15, file_size=13),
        bot=bot,
        answer=AsyncMock(),
    )
    original = transcription.transcribe_question

    def transcript(settings, repo, guard, reader):
        return original(
            settings,
            repo,
            guard,
            reader,
            transport=lambda *a: {"text": QUESTION, "usage": {"cost": 0.0001, "is_byok": False}},
        )

    monkeypatch.setattr(transcription, "transcribe_question", transcript)
    asyncio.run(handler(incoming))
    request = r.detail_request_for_delivery(panel["delivery_id"], depth="article")
    assert request["reader_context"]["voice_input_sha256"]
    assert not request["reader_context"]["user_question"]
    assert get_news_panel(r, panel["delivery_id"])["view"] == "loading"
    seen = []

    def transport(payload, key):
        seen.append(payload)
        return response("finish", finish(doc))

    def builder(settings, repo, event_id, **kwargs):
        kwargs.pop("generation_key", None)
        return build_news_explanation(settings, repo, event_id, transport=transport, **kwargs)

    result = DetailWorker(s, r, builder=builder).run_once()
    assert result.status == "ready", result.error
    assert QUESTION in json.dumps(seen[0], ensure_ascii=False)
    assert asyncio.run(send_ready_detail_articles(bot, r))["sent"] == 1
    assert bot.edit_message_text.call_args.kwargs["message_id"] == 9001
    artifact = r.get_detailed_article(result.article_id)
    assert artifact["reader_context"]["user_question"] == QUESTION
    with r.connection() as db:
        assert db.execute("SELECT question FROM news_question_inputs").fetchone()[0] == QUESTION


def test_oversized_voice_does_not_download_or_queue(env):
    _, r, panel, bot, router, _ = setup(env)
    open_session(r, panel)
    handler = next(
        h.callback for h in router.message.handlers if h.callback.__name__ == "news_question_reply"
    )
    bot.download = AsyncMock()
    incoming = SimpleNamespace(
        message_id=9017,
        from_user=SimpleNamespace(id=123),
        chat=SimpleNamespace(id=123, type="private"),
        reply_to_message=SimpleNamespace(message_id=9001),
        text=None,
        voice=SimpleNamespace(duration=181, file_size=100),
        bot=bot,
        answer=AsyncMock(),
    )
    asyncio.run(handler(incoming))
    bot.download.assert_not_awaited()
    with r.connection() as db:
        assert db.execute("SELECT count(*) FROM news_question_inputs").fetchone()[0] == 0


def test_agent_payload_uses_parameters_supported_by_luna(env):
    s, r, panel, _, _, doc = setup(env)

    def transport(payload, key):
        assert "temperature" not in payload
        assert "parallel_tool_calls" not in payload
        return response("finish", finish(doc))

    result = build_news_explanation(
        s,
        r,
        panel["reader_context"]["baseline_event_id"],
        depth="expanded",
        reader_context=panel["reader_context"],
        transport=transport,
    )
    assert result.validation_status == "valid"


def test_confirmed_routing_rejection_is_free_and_does_not_block_fixed_parameters(env):
    from urllib.error import HTTPError

    from app.llm.openrouter import OpenRouterRequestError

    s, r, panel, _, _, _ = setup(env)
    calls = []
    scope = panel["reader_context"]["context_fingerprint"]
    guard = DurableDetailProvider(
        s,
        r,
        panel["reader_context"]["baseline_event_id"],
        "expanded",
        scope,
        request_budget_usd=0.05,
    )

    def rejected():
        calls.append(1)
        exc = OpenRouterRequestError("OpenRouter HTTP 404: No endpoints found")
        exc.rejected_before_dispatch = True
        raise exc from HTTPError(
            "https://openrouter.ai/api/v1/chat/completions", 404, "Not Found", {}, None
        )

    for _ in range(2):
        with pytest.raises(DetailProviderError) as e:
            guard.run("openrouter", "explanation", {"temperature": 0.2}, 0.01, rejected)
        assert e.value.status == "routing_unavailable" and e.value.cost_usd == 0
    assert len(calls) == 1
    assert guard.total_cost_usd == 0
    assert guard.run(
        "openrouter",
        "explanation",
        {"tools": ["finish"]},
        0.01,
        lambda: {"ok": True},
        cost=lambda *a: (0.001, {"cost_estimated": False}),
    ) == {"ok": True}


def test_unproven_http_404_keeps_reservation_and_blocks_replay(env):
    from urllib.error import HTTPError

    s, r, panel, _, _, _ = setup(env)
    guard = DurableDetailProvider(
        s,
        r,
        panel["reader_context"]["baseline_event_id"],
        "expanded",
        panel["reader_context"]["context_fingerprint"],
        request_budget_usd=0.05,
    )
    calls = []

    def fail():
        calls.append(1)
        raise HTTPError("https://provider.example", 404, "Unknown", {}, None)

    for _ in range(2):
        with pytest.raises(DetailProviderError) as e:
            guard.run("openrouter", "explanation", {"request": "unknown"}, 0.01, fail)
        assert e.value.status == "uncertain"
    assert len(calls) == 1 and guard.total_cost_usd == 0.01
