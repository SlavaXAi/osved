from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import SecretStr

from app.config import Settings
from app.db.repository import Repository
from app.video.report_format import MODEL
from app.video.reports import ReportError, VideoReportService, save_json
from app.video.tldr import LEGACY_VERSIONS, VERSION

VIDEO = "abcdefghijk"
OLD_TEXT = "Автор  объясняет механизм и существенные условия вывода."
NEW_TEXT = "Автор объясняет новый подход."


def response(text: str = NEW_TEXT) -> dict:
    return {
        "id": "response-cache-test",
        "model": MODEL,
        "usage": {"cost": 0.001},
        "choices": [{"finish_reason": "stop", "message": {"content": "## TL;DR\n\n" + text}}],
    }


@pytest.fixture
def cache_env(tmp_path):
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        allowed_user_id=123,
        openrouter_api_key=SecretStr("fake"),
        monid_api_key=SecretStr("fake"),
    )
    repo = Repository(settings.database_path)
    repo.migrate()
    calls = []

    def transcript_request(command):
        calls.append("transcript")
        raise AssertionError("Test must not retrieve a transcript")

    def model(request, key):
        calls.append("model")
        return response()

    service = VideoReportService(settings, repo, monid_runner=transcript_request, transport=model)
    return repo, service, calls


def directory(service: VideoReportService, version: str) -> Path:
    return service.root / VIDEO / ("tldr-" + version)


def seed_result(folder: Path, text: str, *, digest: str | None = None) -> None:
    save_json(
        folder / "result.json",
        {
            "original": "## TL;DR\n\n" + text,
            "text_sha256": digest or hashlib.sha256(text.encode()).hexdigest(),
        },
    )


def seed_transcript(service: VideoReportService) -> None:
    save_json(
        service.root / VIDEO / "transcript.json",
        [
            {
                "video_id": VIDEO,
                "title": "Видео для теста кэша",
                "duration_seconds": 60,
                "transcript": [
                    {"start": 0, "end": 30, "text": "Начало и механизм."},
                    {"start": 30, "end": 60, "text": "Поздние условия и вывод."},
                ],
            }
        ],
    )


def seed_legacy_marker(repo: Repository, service: VideoReportService) -> tuple[Path, int]:
    folder = directory(service, LEGACY_VERSIONS[0])
    ledger_id = service._reserve("openrouter", 0.025, VIDEO, operation="video.tldr.openrouter")
    save_json(folder / "llm-started.json", {"ledger_id": ledger_id, "reserve_usd": 0.025})
    return folder, ledger_id


def usage_count(repo: Repository) -> int:
    with repo.connection() as db:
        return db.execute("SELECT COUNT(*) FROM usage_ledger").fetchone()[0]


@pytest.mark.parametrize("spacing", [OLD_TEXT, OLD_TEXT.replace("  ", "\n")])
def test_known_legacy_result_reuses_old_hash_and_never_pays_again(cache_env, spacing):
    repo, service, calls = cache_env
    legacy_text = " ".join(spacing.splitlines())
    seed_result(
        directory(service, LEGACY_VERSIONS[0]),
        spacing,
        digest=hashlib.sha256(legacy_text.encode()).hexdigest(),
    )
    # Cache still works if a user has no active model credentials or budget.
    service.settings.openrouter_api_key = None
    service.settings.monid_api_key = None
    service.settings.budget_hard_stop_usd = 0
    for _ in range(2):
        assert service.build_tldr(VIDEO) == " ".join(OLD_TEXT.split())
    assert calls == [] and usage_count(repo) == 0
    assert not directory(service, VERSION).exists()


@pytest.mark.parametrize(
    "old_text", ["Старый текст без точки", "Автор <b>объясняет</b> условия и **основной тезис**."]
)
def test_legacy_tolerance_does_not_automatically_repay_for_stricter_new_format(cache_env, old_text):
    repo, service, calls = cache_env
    seed_result(directory(service, LEGACY_VERSIONS[0]), old_text)
    assert service.build_tldr(VIDEO) == old_text
    assert calls == [] and usage_count(repo) == 0


def test_current_result_takes_priority_over_legacy_result(cache_env):
    repo, service, calls = cache_env
    seed_result(directory(service, LEGACY_VERSIONS[0]), OLD_TEXT)
    seed_result(directory(service, VERSION), NEW_TEXT)
    assert service.build_tldr(VIDEO) == NEW_TEXT
    assert calls == [] and usage_count(repo) == 0


def test_corrupt_current_digest_does_not_hide_usable_legacy_result(cache_env):
    repo, service, calls = cache_env
    seed_result(directory(service, VERSION), NEW_TEXT, digest="invalid")
    seed_result(directory(service, LEGACY_VERSIONS[0]), OLD_TEXT)
    assert service.build_tldr(VIDEO) == " ".join(OLD_TEXT.split())
    assert calls == [] and usage_count(repo) == 0


def test_completed_legacy_raw_response_recovers_after_crash_without_model_call(cache_env):
    repo, service, calls = cache_env
    folder, _ = seed_legacy_marker(repo, service)
    save_json(folder / "response.json", response(OLD_TEXT))
    original_ledger_count = usage_count(repo)
    assert not (folder / "result.json").exists()
    for _ in range(2):
        assert service.build_tldr(VIDEO) == " ".join(OLD_TEXT.split())
    assert calls == [] and usage_count(repo) == original_ledger_count
    assert not directory(service, VERSION).exists()


@pytest.mark.parametrize("invalid", ["wrong_model", "unfinished", "invalid_prose"])
def test_invalid_raw_legacy_response_does_not_unlock_new_paid_request(cache_env, invalid):
    repo, service, calls = cache_env
    folder, _ = seed_legacy_marker(repo, service)
    value = response(OLD_TEXT)
    if invalid == "wrong_model":
        value["model"] = "unknown/model"
    elif invalid == "unfinished":
        value["choices"][0]["finish_reason"] = "length"
    else:
        value["choices"][0]["message"]["content"] = "Unexpected output"
    save_json(folder / "response.json", value)
    for _ in range(2):
        with pytest.raises(ReportError, match="uncertain"):
            service.build_tldr(VIDEO)
    assert calls == [] and usage_count(repo) == 1


@pytest.mark.parametrize(
    "error",
    [
        None,
        {"http_status": 500, "rejected_before_dispatch": False},
        {"http_status": 402, "rejected_before_dispatch": False},
        {"http_status": 402, "rejected_before_dispatch": True, "ledger_id": "different"},
        {"http_status": 402, "rejected_before_dispatch": True},
    ],
)
def test_old_uncertain_or_mismatched_rejection_blocks_new_prompt_paid_attempt(cache_env, error):
    repo, service, calls = cache_env
    folder, ledger_id = seed_legacy_marker(repo, service)
    seed_transcript(service)
    if error is not None:
        save_json(folder / "llm-errors" / f"{ledger_id}.json", error)
    for _ in range(2):
        with pytest.raises(ReportError, match="uncertain"):
            service.build_tldr(VIDEO)
    assert calls == [] and usage_count(repo) == 1
    assert not (directory(service, VERSION) / "llm-started.json").exists()


def test_only_exact_recorded_402_allows_explicit_new_prompt_attempt(cache_env):
    repo, service, calls = cache_env
    folder, ledger_id = seed_legacy_marker(repo, service)
    save_json(
        folder / "llm-errors" / f"{ledger_id}.json",
        {"ledger_id": ledger_id, "http_status": 402, "rejected_before_dispatch": True},
    )
    seed_transcript(service)
    # Nothing runs until the next explicit build invocation.
    assert calls == [] and not (directory(service, VERSION) / "response.json").exists()
    assert service.build_tldr(VIDEO) == NEW_TEXT
    assert calls == ["model"] and usage_count(repo) == 2
    assert service.build_tldr(VIDEO) == NEW_TEXT
    assert calls == ["model"] and usage_count(repo) == 2
    assert (folder / "llm-errors" / f"{ledger_id}.json").exists()


def test_old_unrelated_402_file_cannot_unlock_newer_uncertain_marker(cache_env):
    repo, service, calls = cache_env
    folder, ledger_id = seed_legacy_marker(repo, service)
    save_json(
        folder / "llm-errors" / f"{ledger_id - 1}.json",
        {"ledger_id": ledger_id - 1, "http_status": 402, "rejected_before_dispatch": True},
    )
    with pytest.raises(ReportError, match="uncertain"):
        service.build_tldr(VIDEO)
    assert calls == [] and usage_count(repo) == 1
