from datetime import UTC, datetime, timedelta

import pytest

from app.collectors.platform_transport import MonidResponse, PlatformError
from app.video.transcripts import (
    CAPTIONS_ENDPOINT,
    CAPTIONS_RESULT_ENDPOINT,
    TranscriptService,
    youtube_video_id,
)

VIDEO = "abcdefghijk"
NOW = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)
LANGUAGES = {"captions": [{"language_code": "en"}, {"language_code": "zh-Hans"}]}
CONTENT = {
    "events": [
        {"tStartMs": 1200, "dDurationMs": 2000, "segs": [{"utf8": "Real"}, {"utf8": " words."}]}
    ]
}


def sequence(*responses):
    calls = []

    def fetch(endpoint, query):
        calls.append((endpoint, query))
        value = responses[len(calls) - 1]
        if isinstance(value, Exception):
            raise value
        return MonidResponse(value, 0 if endpoint == CAPTIONS_RESULT_ENDPOINT else 0.012)

    return calls, fetch


def test_transcript_fetches_only_on_explicit_request_and_cache_prevents_rebilling(tmp_path):
    calls, fetch = sequence(LANGUAGES, CONTENT)
    ledger = []
    service = TranscriptService(
        tmp_path, fetcher=fetch, clock=lambda: NOW, on_cost=lambda *args: ledger.append(args)
    )
    assert not calls
    result = service.fetch(VIDEO)
    assert result.status == "ready"
    assert result.text == "Real words."
    assert result.segments[0].start_seconds == 1.2
    assert result.cost_usd == 0.024
    assert sum(entry[1] for entry in ledger) == 0.024
    again = TranscriptService(tmp_path, fetcher=fetch, clock=lambda: NOW).fetch(VIDEO)
    assert again.cached
    assert again.cost_usd == 0
    assert len(calls) == 2


def test_captionless_result_is_billed_once_and_negatively_cached(tmp_path):
    calls, fetch = sequence({"captions": [], "message": "No transcript is available"})
    service = TranscriptService(tmp_path, fetcher=fetch, clock=lambda: NOW)
    result = service.fetch(VIDEO)
    assert result.status == "unavailable"
    assert result.cost_usd == 0.012
    assert service.fetch(VIDEO).cached
    assert len(calls) == 1


def test_budget_gate_precedes_paid_language_lookup(tmp_path):
    calls, fetch = sequence()
    result = TranscriptService(tmp_path, fetcher=fetch).fetch(VIDEO, cost_limit_usd=0.01)
    assert result.status == "disabled"
    assert not calls


@pytest.mark.parametrize("cap", [float("nan"), float("inf"), -1])
def test_caption_budget_rejects_nonfinite_or_negative_values(tmp_path, cap):
    calls, fetch = sequence()
    result = TranscriptService(tmp_path, fetcher=fetch).fetch(VIDEO, cost_limit_usd=cap)
    assert result.status == "disabled"
    assert not calls


def test_async_job_resumes_after_restart_without_restarting_paid_job(tmp_path):
    calls, fetch = sequence(
        LANGUAGES,
        {"status": "processing", "job_id": "test-job-123"},
        {"status": "completed", "content": CONTENT, "language_code": "en"},
    )
    service = TranscriptService(tmp_path, fetcher=fetch, clock=lambda: NOW)
    result = service.fetch(VIDEO)
    assert result.status == "processing"
    assert result.cost_usd == 0.024
    assert service.fetch(VIDEO).retry_after_seconds == 5
    assert len(calls) == 2
    restarted = TranscriptService(tmp_path, fetcher=fetch, clock=lambda: NOW + timedelta(seconds=6))
    ready = restarted.fetch(VIDEO, cost_limit_usd=0)
    assert ready.status == "ready"
    assert ready.cost_usd == 0
    assert calls[-1] == (CAPTIONS_RESULT_ENDPOINT, {"job_id": "test-job-123", "format": "json3"})
    assert sum(1 for endpoint, _ in calls if endpoint == CAPTIONS_ENDPOINT) == 2


def test_auto_generated_language_uses_actual_available_code(tmp_path):
    calls, fetch = sequence({"captions": [{"language_code": "a.en"}]}, CONTENT)
    result = TranscriptService(tmp_path, fetcher=fetch).fetch(VIDEO)
    assert result.language == "a.en"
    assert calls[1][1]["language_code"] == "a.en"


def test_unsupported_language_falls_back_to_real_language(tmp_path):
    calls, fetch = sequence({"captions": [{"language_code": "ja"}]}, CONTENT)
    result = TranscriptService(tmp_path, fetcher=fetch).fetch(VIDEO)
    assert result.language == "ja"


def test_regional_english_is_preferred_over_first_arabic_track(tmp_path):
    calls, fetch = sequence(
        {"captions": [{"language_code": "ar"}, {"language_code": "en-US"}]}, CONTENT
    )
    result = TranscriptService(tmp_path, fetcher=fetch).fetch(VIDEO)
    assert result.language == "en-US"
    assert calls[1][1]["language_code"] == "en-US"


def test_explicit_language_probe_is_one_call_and_preserves_previous_language(tmp_path):
    calls, fetch = sequence(
        {"captions": [{"language_code": "ar"}]},
        {"content": CONTENT, "language_code": "ar"},
        {"content": CONTENT, "language_code": "en"},
    )
    service = TranscriptService(tmp_path, fetcher=fetch)
    first = service.fetch(VIDEO)
    english = service.fetch(VIDEO, language_code="en", cost_limit_usd=0.012)
    assert first.language == "ar"
    assert english.status == "ready"
    assert english.language == "en"
    assert english.cost_usd == 0.012
    assert service.fetch(VIDEO).language == "en"
    assert service.fetch(VIDEO, language_code="ar", cost_limit_usd=0).cached
    assert len(calls) == 3


def test_failed_explicit_language_probe_never_erases_good_cache_or_rebills_immediately(tmp_path):
    calls, fetch = sequence(
        {"captions": [{"language_code": "ar"}]},
        {"content": CONTENT, "language_code": "ar"},
        {"content": CONTENT, "language_code": "ar"},
    )
    service = TranscriptService(tmp_path, fetcher=fetch)
    service.fetch(VIDEO)
    failed = service.fetch(VIDEO, language_code="en", cost_limit_usd=0.012)
    assert failed.status == "unavailable"
    assert failed.cost_usd == 0.012
    assert service.fetch(VIDEO).language == "ar"
    assert service.fetch(VIDEO, language_code="en", cost_limit_usd=0.012).cached
    assert len(calls) == 3


def test_explicit_language_obeys_one_call_budget_and_safe_cache_name(tmp_path):
    calls, fetch = sequence()
    service = TranscriptService(tmp_path, fetcher=fetch)
    assert service.fetch(VIDEO, language_code="en", cost_limit_usd=0).status == "disabled"
    with pytest.raises(ValueError):
        service.fetch(VIDEO, language_code="../../secret")
    assert not calls


@pytest.mark.parametrize("timestamp", [None, -2, float("nan"), float("inf")])
def test_partial_or_invalid_timing_preserves_all_text_without_fake_zero_links(tmp_path, timestamp):
    content = {
        "events": [
            {"tStartMs": timestamp, "segs": [{"utf8": "No valid timestamp."}]},
            {"tStartMs": 2000, "segs": [{"utf8": "A genuinely timed sentence."}]},
        ]
    }
    _, fetch = sequence(LANGUAGES, {"content": content})
    result = TranscriptService(tmp_path, fetcher=fetch).fetch(VIDEO)
    assert result.status == "ready"
    assert result.segments == []
    assert "No valid timestamp" in result.text
    assert "genuinely timed" in result.text


def test_plain_text_does_not_get_invented_timestamps(tmp_path):
    _, fetch = sequence(LANGUAGES, {"content": "Transcript without timestamps"})
    result = TranscriptService(tmp_path, fetcher=fetch).fetch(VIDEO)
    assert result.status == "ready"
    assert result.segments == []
    assert result.text == "Transcript without timestamps"


def test_srt_fallback_preserves_timestamp(tmp_path):
    _, fetch = sequence(
        LANGUAGES, {"content": "1\n00:02:03,400 --> 00:02:06,000\nA useful claim.\n"}
    )
    result = TranscriptService(tmp_path, fetcher=fetch).fetch(VIDEO)
    assert result.status == "ready"
    assert result.segments[0].start_seconds == 123.4


def test_failed_provider_bills_attempt_and_does_not_return_generated_content(tmp_path):
    calls, fetch = sequence(PlatformError("limited", status="rate_limited", cost_usd=0.012))
    ledger = []
    service = TranscriptService(
        tmp_path, fetcher=fetch, clock=lambda: NOW, on_cost=lambda *args: ledger.append(args)
    )
    result = service.fetch(VIDEO)
    assert result.status == "rate_limited"
    assert result.text == ""
    assert result.cost_usd == 0.012
    assert ledger[0][1] == 0.012
    assert service.fetch(VIDEO).cached
    assert len(calls) == 1


def test_invalid_result_is_schema_changed_not_no_captions(tmp_path):
    _, fetch = sequence({"wrong": "schema"})
    result = TranscriptService(tmp_path, fetcher=fetch).fetch(VIDEO)
    assert result.status == "schema_changed"
    assert result.cost_usd == 0.012


def test_oversized_transcript_is_rejected(tmp_path):
    _, fetch = sequence(LANGUAGES, {"content": "x" * 500_001})
    result = TranscriptService(tmp_path, fetcher=fetch).fetch(VIDEO)
    assert result.status == "schema_changed"
    assert not result.text


@pytest.mark.parametrize(
    "value",
    [
        VIDEO,
        f"https://youtu.be/{VIDEO}",
        f"https://www.youtube.com/watch?v={VIDEO}&t=100",
        f"https://www.youtube.com/shorts/{VIDEO}",
    ],
)
def test_video_id_normalization(value):
    assert youtube_video_id(value) == VIDEO


@pytest.mark.parametrize("value", ["../../secret", "https://evil.com/watch?v=abcdefghijk", "short"])
def test_invalid_ids_cannot_escape_cache_directory(tmp_path, value):
    calls, fetch = sequence()
    with pytest.raises(ValueError):
        TranscriptService(tmp_path, fetcher=fetch).fetch(value)
    assert not calls


def test_transient_poll_error_preserves_existing_paid_job(tmp_path):
    calls, fetch = sequence(
        LANGUAGES,
        {"status": "processing", "job_id": "test-job-123"},
        PlatformError("poll rate limited", status="rate_limited"),
        {"status": "completed", "content": CONTENT},
    )
    TranscriptService(tmp_path, fetcher=fetch, clock=lambda: NOW).fetch(VIDEO)
    limited = TranscriptService(
        tmp_path, fetcher=fetch, clock=lambda: NOW + timedelta(seconds=6)
    ).fetch(VIDEO)
    assert limited.status == "processing"
    assert limited.job_id == "test-job-123"
    assert limited.retry_after_seconds == 30
    cooling = TranscriptService(
        tmp_path, fetcher=fetch, clock=lambda: NOW + timedelta(seconds=15)
    ).fetch(VIDEO)
    assert cooling.status == "processing"
    assert len(calls) == 3
    final = TranscriptService(
        tmp_path, fetcher=fetch, clock=lambda: NOW + timedelta(seconds=37)
    ).fetch(VIDEO)
    assert final.status == "ready"
    assert final.cost_usd == 0
    assert sum(endpoint == CAPTIONS_ENDPOINT for endpoint, _ in calls) == 2
