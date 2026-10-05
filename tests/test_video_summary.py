import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.config import Settings
from app.llm.openrouter import OpenRouterRequestError
from app.video.summary import _request, build_video_summary, transcript_evidence
from app.video.transcripts import TranscriptResult, TranscriptSegment

VIDEO = "abcdefghijk"
SHORT_TEXT = (
    "Собеседники разбирают применение программных агентов для обработки счетов. "
    "Главная тема — проверка результатов перед отправкой и участие человека в сложных случаях. "
    "Они объясняют, почему автоматизации нужны понятные границы и измеримая надёжность."
)
LONG_TEXT = (
    "В разговоре разбирают применение программных агентов для обработки счетов. "
    "Собеседники объясняют, как документы проходят проверку и почему в сложных случаях "
    "сохраняется участие человека. Особое внимание уделено тому, чтобы результат можно "
    "было проверить перед отправкой. Для этого необходимы понятные границы автоматизации."
)


class Repo:
    def __init__(self, cost=0):
        self.cost = cost
        self.ledger = []

    def current_month_cost(self):
        return self.cost

    def record_usage(self, provider, operation, **kwargs):
        self.ledger.append((provider, operation, kwargs))
        self.cost += kwargs.get("cost_usd", 0)


class Transcript:
    def __init__(self, result=None):
        self.calls = []
        self.result = result or TranscriptResult(
            VIDEO,
            "ready",
            text="Agents can process invoices with human review. " * 10,
            segments=[
                TranscriptSegment(65.5, "Agents can process invoices with human review. " * 10)
            ],
            language="en",
            cached=True,
        )

    def fetch(self, video, **kwargs):
        self.calls.append((video, kwargs))
        return self.result


def settings(tmp_path, **kwargs):
    return Settings(
        _env_file=None,
        app_env="test",
        data_dir=tmp_path,
        openrouter_api_key="test-not-real",
        **kwargs,
    )


def draft(mode="short"):
    paragraphs = [{"text": SHORT_TEXT, "evidence_ids": ["s0"]}]
    if mode == "long":
        paragraphs = [
            {"text": prefix + LONG_TEXT, "evidence_ids": ["s0"]}
            for prefix in ("Основная тема. ", "Практическое применение. ", "Ограничения. ")
        ]
    return {
        "headline": "Как агенты помогают проверять счета",
        "headline_evidence_ids": ["s0"],
        "paragraphs": paragraphs,
    }


def transport_for(value=None, *, mode="short"):
    calls = []

    def transport(payload, key):
        calls.append(payload)
        return {
            "id": "response1",
            "model": payload["model"],
            "usage": {"cost": 0.0005, "prompt_tokens": 1000, "completion_tokens": 200},
            "choices": [
                {"message": {"content": json.dumps(value if value is not None else draft(mode))}}
            ],
        }

    return calls, transport


def test_grounded_short_summary_uses_actual_timestamp_and_html(tmp_path):
    calls, transport = transport_for()
    repo = Repo()
    result = build_video_summary(
        settings(tmp_path), repo, VIDEO, transport=transport, transcript_service=Transcript()
    )
    assert result["status"] == "ready"
    assert "&amp;t=65s" in result["html"]
    assert ">1:05</a>" in result["html"]
    assert "<b><a href=" in result["html"]
    assert result["cost_usd"] == 0.0005
    assert len(repo.ledger) == 1
    assert calls[0]["model"] == "openai/gpt-4.1-mini"
    assert calls[0]["provider"]["max_price"] == {"prompt": 0.5, "completion": 2.0}
    schema = calls[0]["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["paragraphs"]["maxItems"] == 1
    assert "maxLength" not in schema["$defs"]["SummaryParagraph"]["properties"]["text"]
    assert calls[0]["max_tokens"] == 600


def test_short_format_repair_is_bounded_grounded_billed_and_cached(tmp_path):
    calls = []
    repo = Repo()

    def transport(payload, key):
        calls.append(payload)
        _, respond = transport_for(draft("long") if len(calls) == 1 else draft())
        return respond(payload, key)

    first = build_video_summary(
        settings(tmp_path), repo, VIDEO, transport=transport, transcript_service=Transcript()
    )
    again = build_video_summary(
        settings(tmp_path), repo, VIDEO, transport=transport, transcript_service=Transcript()
    )
    assert first["status"] == again["status"] == "ready"
    assert len(calls) == len(repo.ledger) == 2
    assert first["cost_usd"] == 0.001
    assert again["cost_usd"] == 0
    assert again["cached"]
    assert len(first["draft"]["paragraphs"]) == 1
    assert json.loads(calls[0]["messages"][1]["content"])["evidence"][0]["id"] == "s0"


def test_short_format_repair_cannot_loop_or_bypass_remaining_budget(tmp_path):
    calls, transport = transport_for(draft("long"))
    result = build_video_summary(
        settings(tmp_path), Repo(), VIDEO, transport=transport, transcript_service=Transcript()
    )
    assert result["status"] == "invalid_response"
    assert len(calls) == 2
    assert result["cost_usd"] == 0.001

    calls = []

    def nearly_spent(payload, key):
        calls.append(payload)
        _, response = transport_for(draft("long"))
        result = response(payload, key)
        result["usage"]["cost"] = 0.0499
        return result

    result = build_video_summary(
        settings(tmp_path / "budget"),
        Repo(),
        VIDEO,
        transport=nearly_spent,
        transcript_service=Transcript(),
    )
    assert result["status"] == "invalid_response"
    assert len(calls) == 1
    assert result["cost_usd"] == 0.0499


def test_summary_cache_prevents_duplicate_llm_calls(tmp_path):
    calls, transport = transport_for()
    repo, transcript = Repo(), Transcript()
    first = build_video_summary(
        settings(tmp_path), repo, VIDEO, transport=transport, transcript_service=transcript
    )
    second = build_video_summary(
        settings(tmp_path), repo, VIDEO, transport=transport, transcript_service=transcript
    )
    assert first["status"] == second["status"] == "ready"
    assert second["cached"] is True
    assert second["cost_usd"] == 0
    assert len(calls) == len(repo.ledger) == 1


def test_different_modes_have_distinct_exact_input_caches(tmp_path):
    short_calls, short_transport = transport_for()
    long_calls, long_transport = transport_for(mode="long")
    repo, transcript = Repo(), Transcript()
    short = build_video_summary(
        settings(tmp_path), repo, VIDEO, transport=short_transport, transcript_service=transcript
    )
    long = build_video_summary(
        settings(tmp_path),
        repo,
        VIDEO,
        mode="long",
        transport=long_transport,
        transcript_service=transcript,
    )
    assert short["status"] == long["status"] == "ready"
    assert short["input_hash"] != long["input_hash"]
    assert len(short_calls) == len(long_calls) == 1


def test_useful_long_summary_may_exceed_target_without_being_rejected(tmp_path):
    value = draft("long")
    value["paragraphs"] = [{"text": LONG_TEXT * 2, "evidence_ids": ["s0"]} for _ in range(5)]
    calls, transport = transport_for(value)
    result = build_video_summary(
        settings(tmp_path),
        Repo(),
        VIDEO,
        mode="long",
        transport=transport,
        transcript_service=Transcript(),
    )
    assert result["status"] == "ready"
    assert 3000 < len(result["text"]) < 4000
    assert len(calls) == 1
    schema = calls[0]["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["paragraphs"]["maxItems"] == 5
    assert "maxLength" not in schema["$defs"]["SummaryParagraph"]["properties"]["text"]


def test_changed_transcript_invalidates_summary_cache(tmp_path):
    calls, transport = transport_for()
    repo = Repo()
    first = build_video_summary(
        settings(tmp_path), repo, VIDEO, transport=transport, transcript_service=Transcript()
    )
    changed = Transcript(TranscriptResult(VIDEO, "ready", text="Changed transcript " * 20))
    second = build_video_summary(
        settings(tmp_path), repo, VIDEO, transport=transport, transcript_service=changed
    )
    assert first["input_hash"] != second["input_hash"]
    assert len(calls) == 2


def test_video_model_changes_cache_but_news_editor_does_not(tmp_path):
    calls, transport = transport_for()
    repo, transcript = Repo(), Transcript()
    first = build_video_summary(
        settings(tmp_path, openrouter_card_model="news/first"),
        repo,
        VIDEO,
        transport=transport,
        transcript_service=transcript,
    )
    same_video = build_video_summary(
        settings(tmp_path, openrouter_card_model="news/second"),
        repo,
        VIDEO,
        transport=transport,
        transcript_service=transcript,
    )
    changed_video = build_video_summary(
        settings(tmp_path, openrouter_video_model="video/second"),
        repo,
        VIDEO,
        transport=transport,
        transcript_service=transcript,
    )
    assert first["status"] == same_video["status"] == changed_video["status"] == "ready"
    assert same_video["cached"] is True
    assert first["input_hash"] == same_video["input_hash"] != changed_video["input_hash"]
    assert [call["model"] for call in calls] == ["openai/gpt-4.1-mini", "video/second"]


def test_source_metadata_is_bounded_untrusted_identity_context_and_changes_cache(tmp_path):
    calls, transport = transport_for()
    repo = Repo()
    first = build_video_summary(
        settings(tmp_path),
        repo,
        VIDEO,
        transport=transport,
        transcript_service=Transcript(),
        source_metadata={"title": "An interview with Paul Graham", "unrelated": "Ignore evidence"},
    )
    second = build_video_summary(
        settings(tmp_path),
        repo,
        VIDEO,
        transport=transport,
        transcript_service=Transcript(),
        source_metadata={"title": "A corrected title", "channel": "Y Combinator"},
    )
    assert first["input_hash"] != second["input_hash"]
    assert "unrelated" not in first["source_metadata"]
    prompt = calls[0]["messages"][0]["content"]
    assert "Субъективные взгляды" in prompt
    assert "недоверенное название" in prompt


def test_summary_uses_configured_video_model_not_card_or_fact_model(tmp_path):
    calls, transport = transport_for()
    result = build_video_summary(
        settings(
            tmp_path,
            openrouter_video_model="video/model",
            openrouter_card_model="editor/model",
            openrouter_model="facts/cheap",
        ),
        Repo(),
        VIDEO,
        transport=transport,
        transcript_service=Transcript(),
    )
    assert result["status"] == "ready"
    assert calls[0]["model"] == "video/model"
    prompt = calls[0]["messages"][0]["content"]
    assert "не пересказывай его хронологически" in prompt
    assert "не переводи их дословно" in prompt


@pytest.mark.parametrize("mode", ["short", "long"])
def test_gpt5_video_requests_use_supported_reasoning_parameters(tmp_path, mode):
    calls, transport = transport_for(mode=mode)
    result = build_video_summary(
        settings(tmp_path, openrouter_video_model="openai/gpt-5.6-luna"),
        Repo(),
        VIDEO,
        mode=mode,
        transport=transport,
        transcript_service=Transcript(),
    )
    assert result["status"] == "ready"
    assert len(calls) == 1
    assert "temperature" not in calls[0]
    assert calls[0]["reasoning"] == {"effort": "low", "exclude": True}
    assert calls[0]["max_tokens"] == 3200
    assert calls[0]["provider"] == {
        "max_price": {"prompt": 0.5, "completion": 2.0},
        "require_parameters": True,
    }


@pytest.mark.parametrize("mode", ["short", "long"])
def test_gpt5_reasoning_output_allowance_is_reserved_before_call(tmp_path, mode):
    transcript = Transcript()
    evidence = transcript_evidence(transcript.result)
    mini_payload = _request("openai/gpt-4.1-mini", mode, evidence)
    luna_payload = _request("openai/gpt-5.6-luna", mode, evidence)

    def reserve(payload):
        prompt_bytes = len(json.dumps(payload, ensure_ascii=False).encode())
        return ((prompt_bytes + 2048) * 0.5 + payload["max_tokens"] * 2.0) / 1_000_000

    cap = (reserve(mini_payload) + reserve(luna_payload)) / 2
    assert reserve(mini_payload) < cap < reserve(luna_payload) < 0.05
    calls, transport = transport_for(mode=mode)
    result = build_video_summary(
        settings(tmp_path / "luna", openrouter_video_model="openai/gpt-5.6-luna"),
        Repo(),
        VIDEO,
        mode=mode,
        cost_limit_usd=cap,
        transport=transport,
        transcript_service=transcript,
    )
    assert result["status"] == "budget_blocked"
    assert not calls
    mini = build_video_summary(
        settings(tmp_path / "mini"),
        Repo(),
        VIDEO,
        mode=mode,
        cost_limit_usd=cap,
        transport=transport,
        transcript_service=Transcript(),
    )
    assert mini["status"] == "ready"
    assert calls[0]["temperature"] == 0.2
    assert "reasoning" not in calls[0]


def test_mixed_alphabet_translation_is_repaired_within_same_budget_and_cached(tmp_path):
    calls = []
    broken = draft()
    broken["paragraphs"][0]["text"] += " Это форidable подход."

    def transport(payload, key):
        calls.append(payload)
        _, respond = transport_for(broken if len(calls) == 1 else draft())
        return respond(payload, key)

    repo = Repo()
    result = build_video_summary(
        settings(tmp_path), repo, VIDEO, transport=transport, transcript_service=Transcript()
    )
    again = build_video_summary(
        settings(tmp_path), repo, VIDEO, transport=transport, transcript_service=Transcript()
    )
    assert result["status"] == "ready"
    assert "форidable" not in result["text"]
    assert len(calls) == len(repo.ledger) == 2
    assert result["cost_usd"] == 0.001
    assert again["cached"]
    assert again["cost_usd"] == 0


def test_legitimate_latin_brands_and_hyphenated_russian_are_not_mixed_alphabet_errors(tmp_path):
    value = draft()
    value["headline"] = "Как OpenAI и iPhone обсуждают в YouTube-канале"
    value["paragraphs"][0]["text"] += " В YouTube-канале обсуждают OpenAI и iPhone."
    calls, transport = transport_for(value)
    result = build_video_summary(
        settings(tmp_path), Repo(), VIDEO, transport=transport, transcript_service=Transcript()
    )
    assert result["status"] == "ready"
    assert len(calls) == 1


@pytest.mark.parametrize("section", ["headline", "paragraph"])
def test_fully_english_block_is_repaired_once_and_cached(tmp_path, section):
    calls = []
    broken = draft()
    english = (
        "Open models dominate enterprise AI usage, driven by cost and customization needs. "
        "The speakers explain why human review remains important before using the output."
    )
    if section == "headline":
        broken["headline"] = english.split(". ")[0]
    else:
        broken["paragraphs"][0]["text"] = english

    def transport(payload, key):
        calls.append(payload)
        _, respond = transport_for(broken if len(calls) == 1 else draft())
        return respond(payload, key)

    repo = Repo()
    result = build_video_summary(
        settings(tmp_path), repo, VIDEO, transport=transport, transcript_service=Transcript()
    )
    cached = build_video_summary(
        settings(tmp_path), repo, VIDEO, transport=transport, transcript_service=Transcript()
    )
    assert result["status"] == "ready"
    assert result["draft"]["headline"] == draft()["headline"]
    assert len(calls) == len(repo.ledger) == 2
    assert result["cost_usd"] == 0.001
    assert cached["cached"] and cached["cost_usd"] == 0
    assert "Non-Russian block" in calls[-1]["messages"][-1]["content"]
    schema = calls[0]["response_format"]["json_schema"]["schema"]
    assert "на русском" in schema["properties"]["headline"]["description"]
    assert "не исправляй их по внешним знаниям" in calls[0]["messages"][0]["content"]


def test_english_repair_cannot_loop_and_failure_is_cached(tmp_path):
    broken = draft()
    broken["headline"] = "Open models dominate enterprise AI usage"
    calls, transport = transport_for(broken)
    repo = Repo()
    first = build_video_summary(
        settings(tmp_path), repo, VIDEO, transport=transport, transcript_service=Transcript()
    )
    again = build_video_summary(
        settings(tmp_path), repo, VIDEO, transport=transport, transcript_service=Transcript()
    )
    assert first["status"] == again["status"] == "invalid_response"
    assert "Non-Russian block" in first["error"]
    assert len(calls) == 2
    assert again["cached"] and again["cost_usd"] == 0


@pytest.mark.parametrize("english_headline", [False, True])
def test_language_repair_cannot_mask_unsupported_factual_numbers(tmp_path, english_headline):
    value = draft()
    if english_headline:
        value["headline"] = "Open models deliver 97 percent better results"
    else:
        value["paragraphs"][0]["text"] += " Форidable результат на 97 процентов."
    calls, transport = transport_for(value)
    result = build_video_summary(
        settings(tmp_path), Repo(), VIDEO, transport=transport, transcript_service=Transcript()
    )
    assert result["status"] == "invalid_response"
    assert len(calls) == 1
    assert "Numeric claim" in result["error"]


@pytest.mark.parametrize("ending", ["Это примеры проектов,", "Это полезно,,", "Это полезно,."])
def test_broken_sentence_or_comma_artifact_requires_regeneration_not_manual_trim(tmp_path, ending):
    calls = []
    broken = draft()
    broken["paragraphs"][0]["text"] = SHORT_TEXT + " " + ending

    def transport(payload, key):
        calls.append(payload)
        _, respond = transport_for(broken if len(calls) == 1 else draft())
        return respond(payload, key)

    result = build_video_summary(
        settings(tmp_path), Repo(), VIDEO, transport=transport, transcript_service=Transcript()
    )
    assert result["status"] == "ready"
    assert len(calls) == 2
    assert result["draft"]["paragraphs"][0]["text"] == SHORT_TEXT
    assert len(list((tmp_path / "video-cache" / "receipts").glob("*.json"))) == 2


def test_unfixed_broken_prose_stays_invalid_and_negatively_cached(tmp_path):
    broken = draft()
    broken["paragraphs"][0]["text"] += " Проекты оказались более разнообразными,"
    calls, transport = transport_for(broken)
    repo = Repo()
    first = build_video_summary(
        settings(tmp_path), repo, VIDEO, transport=transport, transcript_service=Transcript()
    )
    again = build_video_summary(
        settings(tmp_path), repo, VIDEO, transport=transport, transcript_service=Transcript()
    )
    assert first["status"] == again["status"] == "invalid_response"
    assert "Unfinished paragraph" in first["error"]
    assert len(calls) == 2
    assert not first["html"]
    assert again["cached"]


def test_zero_provider_cost_with_real_tokens_is_estimated_not_claimed_free(tmp_path):
    def transport(payload, key):
        _, respond = transport_for()
        response = respond(payload, key)
        response["usage"]["cost"] = 0
        return response

    repo = Repo()
    result = build_video_summary(
        settings(tmp_path), repo, VIDEO, transport=transport, transcript_service=Transcript()
    )
    assert result["status"] == "ready"
    assert result["cost_usd"] > 0
    assert result["cost_estimated"] is True
    assert repo.ledger[0][2]["metadata"]["reported_zero_with_tokens"] is True
    receipt_file = next((tmp_path / "video-cache" / "receipts").glob("*.json"))
    receipt = json.loads(receipt_file.read_text())
    assert receipt["usage"]["cost"] == 0
    assert receipt["usage"]["prompt_tokens"] == 1000
    assert receipt["cost_accounted_usd"] > 0
    assert "test-not-real" not in receipt_file.read_text()


def test_long_short_draft_uses_whole_leading_sentences_without_another_paid_call(tmp_path):
    value = draft()
    value["paragraphs"][0]["text"] = SHORT_TEXT + " " + LONG_TEXT
    calls, transport = transport_for(value)
    repo = Repo()
    result = build_video_summary(
        settings(tmp_path), repo, VIDEO, transport=transport, transcript_service=Transcript()
    )
    assert result["status"] == "ready"
    assert len(calls) == len(repo.ledger) == 1
    kept = result["draft"]["paragraphs"][0]["text"]
    assert value["paragraphs"][0]["text"].startswith(kept)
    assert 160 <= len(kept) <= 350
    assert kept.endswith(".")
    assert result["compression"]["method"] == "whole_sentence_lead"
    assert result["compression"]["omitted_characters"] > 0
    assert result["compression"]["text_rewritten"] is False


def test_short_lead_cannot_hide_unsupported_claims_in_omitted_tail(tmp_path):
    value = draft()
    value["paragraphs"][0]["text"] = SHORT_TEXT + " " + LONG_TEXT + " Эффективность выросла на 89%."
    calls, transport = transport_for(value)
    result = build_video_summary(
        settings(tmp_path), Repo(), VIDEO, transport=transport, transcript_service=Transcript()
    )
    assert result["status"] == "invalid_response"
    assert "Numeric claim" in result["error"]
    assert len(calls) == 1
    assert "compression" not in result


def test_short_lead_does_not_detach_a_claim_from_a_trailing_qualification(tmp_path):
    value = draft()
    qualification = "Однако это не подтверждено " + "независимой проверкой " * 16 + "."
    value["paragraphs"][0]["text"] = SHORT_TEXT + " " + qualification
    calls, transport = transport_for(value)
    result = build_video_summary(
        settings(tmp_path), Repo(), VIDEO, transport=transport, transcript_service=Transcript()
    )
    assert result["status"] == "invalid_response"
    assert len(calls) == 2
    assert "compression" not in result


@pytest.mark.parametrize("byok,expected", [(True, 0.003), (False, 0.001)])
def test_byok_upstream_cost_is_counted_but_not_double_counted_for_normal_routing(
    tmp_path, byok, expected
):
    def transport(payload, key):
        _, respond = transport_for()
        response = respond(payload, key)
        response["usage"].update(
            cost=0.001, is_byok=byok, cost_details={"upstream_inference_cost": 0.002}
        )
        return response

    result = build_video_summary(
        settings(tmp_path), Repo(), VIDEO, transport=transport, transcript_service=Transcript()
    )
    assert result["status"] == "ready"
    assert result["cost_usd"] == expected
    assert result["cost_estimated"] is False


def test_zero_router_cost_with_explicit_byok_receipt_uses_real_upstream_cost(tmp_path):
    def transport(payload, key):
        _, respond = transport_for()
        response = respond(payload, key)
        response["usage"].update(
            cost=0, is_byok=True, cost_details={"upstream_inference_cost": 0.0029584}
        )
        return response

    repo = Repo()
    result = build_video_summary(
        settings(tmp_path), repo, VIDEO, transport=transport, transcript_service=Transcript()
    )
    assert result["cost_usd"] == 0.0029584
    assert result["cost_estimated"] is False
    assert repo.ledger[0][2]["metadata"]["billing_basis"] == "openrouter_plus_byok_upstream"


@pytest.mark.parametrize("status", ["unavailable", "processing", "schema_changed"])
def test_no_summary_without_ready_captions(tmp_path, status):
    calls, transport = transport_for()
    result = build_video_summary(
        settings(tmp_path),
        Repo(),
        VIDEO,
        transport=transport,
        transcript_service=Transcript(TranscriptResult(VIDEO, status)),
    )
    assert result["status"] == status
    assert not result["html"]
    assert not calls


def test_global_hard_stop_prevents_all_new_calls(tmp_path):
    transcript = Transcript()
    calls, transport = transport_for()
    result = build_video_summary(
        settings(tmp_path), Repo(cost=30), VIDEO, transport=transport, transcript_service=transcript
    )
    assert result["status"] == "budget_blocked"
    assert not calls
    assert not transcript.calls


def test_request_cap_prevents_llm_call(tmp_path):
    calls, transport = transport_for()
    transcript = Transcript()
    result = build_video_summary(
        settings(tmp_path),
        Repo(),
        VIDEO,
        transport=transport,
        transcript_service=transcript,
        cost_limit_usd=0.000001,
    )
    assert result["status"] == "budget_blocked"
    assert not calls
    assert transcript.calls[0][1]["cost_limit_usd"] == 0


@pytest.mark.parametrize("cap", [float("nan"), float("inf"), -1])
def test_nonfinite_or_negative_budget_cannot_bypass_summary_cap(tmp_path, cap):
    calls, transport = transport_for()
    transcript = Transcript()
    result = build_video_summary(
        settings(tmp_path),
        Repo(),
        VIDEO,
        cost_limit_usd=cap,
        transport=transport,
        transcript_service=transcript,
    )
    assert result["status"] == "budget_blocked"
    assert not calls
    assert not transcript.calls


def test_full_transcript_is_never_silently_truncated(tmp_path):
    calls, transport = transport_for()
    transcript = Transcript(TranscriptResult(VIDEO, "ready", text="Large transcript. " * 20_000))
    result = build_video_summary(
        settings(tmp_path), Repo(), VIDEO, transport=transport, transcript_service=transcript
    )
    assert result["status"] == "too_long"
    assert not calls
    assert not result["html"]


@pytest.mark.parametrize(
    "fault", ["missing_citations", "unknown_citations", "unsupported_number", "html"]
)
def test_invalid_grounding_bills_call_but_blocks_publication(tmp_path, fault):
    value = draft()
    if fault == "missing_citations":
        del value["paragraphs"][0]["evidence_ids"]
    elif fault == "unknown_citations":
        value["paragraphs"][0]["evidence_ids"] = ["s999"]
    elif fault == "unsupported_number":
        value["paragraphs"][0]["text"] += " Эффективность выросла на 99%."
    else:
        value["paragraphs"][0]["text"] += " <a href='https://evil.example'>link</a>"
    calls, transport = transport_for(value)
    repo = Repo()
    result = build_video_summary(
        settings(tmp_path), repo, VIDEO, transport=transport, transcript_service=Transcript()
    )
    assert result["status"] == "invalid_response"
    assert len(calls) == len(repo.ledger) == 1
    assert result["cost_usd"] == 0.0005
    assert not result["html"]


def test_plain_transcript_has_citations_without_fabricated_time_links(tmp_path):
    _, transport = transport_for()
    result = build_video_summary(
        settings(tmp_path),
        Repo(),
        VIDEO,
        transport=transport,
        transcript_service=Transcript(
            TranscriptResult(VIDEO, "ready", text="Invoice automation. " * 20)
        ),
    )
    assert result["status"] == "ready"
    assert "&amp;t=" not in result["html"]


def test_network_failure_is_accounted_conservatively(tmp_path):
    def fail(payload, key):
        raise OpenRouterRequestError("Network timeout")

    repo = Repo()
    result = build_video_summary(
        settings(tmp_path), repo, VIDEO, transport=fail, transcript_service=Transcript()
    )
    assert result["status"] == "failed"
    assert result["cost_usd"] > 0
    assert repo.ledger[0][2]["metadata"]["cost_estimated"]


def test_concurrent_duplicate_request_calls_model_once(tmp_path):
    calls, transport = transport_for()
    repo = Repo()

    def run(_):
        return build_video_summary(
            settings(tmp_path), repo, VIDEO, transport=transport, transcript_service=Transcript()
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, range(2)))
    assert all(result["status"] == "ready" for result in results)
    assert len(calls) == 1
    assert sum(result["cached"] for result in results) == 1


@pytest.mark.parametrize("response", [[], {"usage": {"prompt_tokens": "oops", "cost": None}}])
def test_malformed_provider_envelope_is_charged_before_validation(tmp_path, response):
    repo = Repo()
    result = build_video_summary(
        settings(tmp_path),
        repo,
        VIDEO,
        transport=lambda payload, key: response,
        transcript_service=Transcript(),
    )
    assert result["status"] == "invalid_response"
    assert result["cost_usd"] > 0
    assert len(repo.ledger) == 1


def test_invalid_paid_summary_is_cached_for_repeat_button(tmp_path):
    invalid = draft()
    invalid["paragraphs"][0]["evidence_ids"] = ["unknown"]
    calls, transport = transport_for(invalid)
    repo = Repo()
    first = build_video_summary(
        settings(tmp_path), repo, VIDEO, transport=transport, transcript_service=Transcript()
    )
    second = build_video_summary(
        settings(tmp_path), repo, VIDEO, transport=transport, transcript_service=Transcript()
    )
    assert first["status"] == second["status"] == "invalid_response"
    assert second["cached"]
    assert second["cost_usd"] == 0
    assert second["retry_after_seconds"] > 0
    assert len(calls) == len(repo.ledger) == 1


def test_failed_summary_cooldown_survives_restart(tmp_path):
    calls = []

    def failure(payload, key):
        calls.append(payload)
        raise OpenRouterRequestError("Network timeout after send")

    repo = Repo()
    first = build_video_summary(
        settings(tmp_path), repo, VIDEO, transport=failure, transcript_service=Transcript()
    )
    second = build_video_summary(
        settings(tmp_path), repo, VIDEO, transport=failure, transcript_service=Transcript()
    )
    assert first["status"] == second["status"] == "failed"
    assert second["cached"]
    assert len(calls) == len(repo.ledger) == 1


def test_failed_summary_can_retry_after_cooldown(tmp_path, monkeypatch):
    from app.video import summary

    clock = [1_000_000.0]
    monkeypatch.setattr(summary.time, "time", lambda: clock[0])
    calls = []

    def failure(payload, key):
        calls.append(payload)
        raise OpenRouterRequestError("Network timeout")

    repo = Repo()
    build_video_summary(
        settings(tmp_path), repo, VIDEO, transport=failure, transcript_service=Transcript()
    )
    clock[0] += 301
    result = build_video_summary(
        settings(tmp_path), repo, VIDEO, transport=failure, transcript_service=Transcript()
    )
    assert result["status"] == "failed"
    assert len(calls) == len(repo.ledger) == 2
