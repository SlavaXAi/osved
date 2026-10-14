from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import SecretStr

from app.config import Settings, SourceConfig, SourceRegistry
from app.db.repository import Repository
from app.radar.concepts import (
    ConceptBrief,
    concept_interest_profile,
    prepare_concept_brief,
    recent_concept_sources,
    store_brief,
)

SOURCE_TEXT = (
    "Domain-native harness means an agent control loop built around domain tools and validations. "
    "Generic agent workflows fail on domain-specific tasks without explicit domain constraints. "
    "Human-in-the-loop checkpoint means a person confirms a consequential action before execution. "
    "This preserves human control over irreversible actions while automating preparation."
)


@pytest.fixture
def concept_env(tmp_path):
    settings = Settings(_env_file=None, app_env="test", data_dir=tmp_path, allowed_user_id=123)
    settings.openrouter_api_key = SecretStr("offline")
    repo = Repository(settings.database_path)
    repo.migrate()
    repo.sync_sources(
        SourceRegistry(
            version=1,
            sources=[
                SourceConfig(
                    id="x",
                    name="Author",
                    kind="x",
                    locator="author",
                    role="author_radar",
                    region="global",
                    language="en",
                )
            ],
        )
    )
    now = datetime(2026, 10, 13, 8, tzinfo=UTC)
    item = repo.insert_source_item_with_id(
        source_id="x",
        external_id="1234567890",
        source_url="https://x.com/author/status/1234567890",
        original_text=SOURCE_TEXT,
        original_language="en",
        published_at=(now - timedelta(hours=1)).isoformat(),
        outbound_urls=[],
        media_metadata={},
        payload_hash="concepts",
        raw_payload={},
    )
    return settings, repo, now, item


def concept(item, *, second=False):
    sentences = SOURCE_TEXT.split(". ")
    offset = 2 if second else 0
    return ConceptBrief.model_validate(
        {
            "term": "Human-in-the-loop checkpoint" if second else "Domain-native harness",
            "kind": "strategy",
            "definition": {
                "text": (
                    "Человек подтверждает существенное действие агента перед исполнением."
                    if second
                    else "Контур работы агента, опирающийся на инструменты предметной области."
                ),
                "source_item_id": item,
                "quote": sentences[offset],
            },
            "why_discussed": {
                "text": (
                    "Автор объясняет, как сохранить контроль над необратимыми действиями."
                    if second
                    else "Автор связывает подход со сбоями универсальных агентов "
                    "на сложных задачах."
                ),
                "source_item_id": item,
                "quote": sentences[offset + 1],
            },
            "practical_meaning": {
                "text": (
                    "Подготовку можно автоматизировать, сохраняя подтверждение человеком."
                    if second
                    else "Ограничения конкретной области становятся явной частью работы агента."
                ),
                "source_item_id": item,
                "quote": sentences[offset + 1],
            },
        }
    )


def response(briefs):
    return {
        "choices": [{"message": {"content": json.dumps({"briefs": briefs})}}],
        "usage": {"cost": 0.002, "prompt_tokens": 100, "completion_tokens": 500},
    }


def test_one_daily_call_creates_two_distinct_grounded_cards_and_splits_run_cost(concept_env):
    settings, repo, now, item = concept_env
    calls = []

    def transport(request, key):
        calls.append(request)
        assert request["max_tokens"] >= 3000
        assert (
            request["response_format"]["json_schema"]["schema"]["properties"]["briefs"]["maxItems"]
            == 2
        )
        return response([concept(item).model_dump(), concept(item, second=True).model_dump()])

    result = prepare_concept_brief(settings, repo, now=now, transport=transport)
    assert result == {"status": "ready", "cards_created": 2, "cost_usd": 0.002, "errors": 0}
    with repo.connection() as db:
        cards = db.execute("SELECT generation_cost_usd,rendered_html FROM cards").fetchall()
        assert len(cards) == 2
        assert sum(row[0] for row in cards) == pytest.approx(result["cost_usd"])
        assert all("единичный сигнал" in row[1] for row in cards)
        assert db.execute("SELECT COUNT(*) FROM usage_ledger").fetchone()[0] == 1
    assert (
        prepare_concept_brief(settings, repo, now=now, transport=transport)["status"]
        == "already_attempted"
    )
    assert len(calls) == 1


@pytest.mark.parametrize("bad", ["source_anchor", "structure"])
def test_invalid_draft_does_not_discard_valid_sibling_or_trigger_retry(concept_env, bad):
    settings, repo, now, item = concept_env
    invalid = concept(item, second=True).model_dump()
    if bad == "source_anchor":
        invalid["definition"]["quote"] = (
            "An invented phrase that is absent from the original source."
        )
    else:
        invalid["definition"]["text"] = "Too short"
    result = prepare_concept_brief(
        settings,
        repo,
        now=now,
        transport=lambda *args: response([invalid, concept(item).model_dump()]),
    )
    assert result == {"status": "ready", "cards_created": 1, "cost_usd": 0.002, "errors": 1}
    with repo.connection() as db:
        assert db.execute("SELECT generation_cost_usd FROM cards").fetchone()[0] == 0.002
        run = db.execute("SELECT status,error,response_json FROM concept_runs").fetchone()
        assert run[0] == "ready" and run[1]
        if bad == "source_anchor":
            assert "verbatim source anchor" in run[1]
        assert json.loads(run[2])["choices"]
    assert (
        prepare_concept_brief(
            settings, repo, now=now, transport=lambda *args: pytest.fail("Paid retry")
        )["status"]
        == "already_attempted"
    )


def test_repeated_term_does_not_fill_second_daily_card(concept_env):
    settings, repo, now, item = concept_env
    repeated = concept(item).model_dump()
    repeated["term"] = "domain native harness"
    result = prepare_concept_brief(
        settings,
        repo,
        now=now,
        transport=lambda *args: response([concept(item).model_dump(), repeated]),
    )
    assert result["cards_created"] == 1
    with repo.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM concept_briefs").fetchone()[0] == 1
        assert db.execute("SELECT generation_cost_usd FROM cards").fetchone()[0] == 0.002


def test_previously_explained_concept_is_skipped_but_new_sibling_is_kept(concept_env):
    settings, repo, now, item = concept_env
    store_brief(
        repo,
        concept(item),
        recent_concept_sources(repo, now),
        now=now - timedelta(days=1),
        model="offline",
        cost=0,
    )
    result = prepare_concept_brief(
        settings,
        repo,
        now=now,
        transport=lambda *args: response(
            [concept(item).model_dump(), concept(item, second=True).model_dump()]
        ),
    )
    assert result["cards_created"] == 1 and result["errors"] == 0
    with repo.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM concept_briefs").fetchone()[0] == 2


def test_batch_above_two_is_invalid_and_not_retried(concept_env):
    settings, repo, now, item = concept_env
    result = prepare_concept_brief(
        settings,
        repo,
        now=now,
        transport=lambda *args: response([concept(item).model_dump()] * 3),
    )
    assert result["status"] == "invalid" and result["cards_created"] == 0
    assert (
        prepare_concept_brief(
            settings, repo, now=now, transport=lambda *args: pytest.fail("Paid retry")
        )["status"]
        == "already_attempted"
    )


def test_empty_quality_batch_does_not_create_filler_or_retry(concept_env):
    settings, repo, now, _ = concept_env
    result = prepare_concept_brief(settings, repo, now=now, transport=lambda *args: response([]))
    assert result["status"] == "empty" and result["cards_created"] == 0
    assert (
        prepare_concept_brief(
            settings, repo, now=now, transport=lambda *args: pytest.fail("Paid retry")
        )["status"]
        == "already_attempted"
    )


def test_timeout_reserves_cost_and_same_corpus_is_not_repaid_on_next_day(concept_env):
    settings, repo, now, _ = concept_env

    def timeout(*args):
        raise TimeoutError("provider outcome unknown")

    result = prepare_concept_brief(settings, repo, now=now, transport=timeout)
    assert result["status"] == "uncertain" and 0 < result["cost_usd"] <= 0.02
    with repo.connection() as db:
        assert db.execute("SELECT cost_usd FROM usage_ledger").fetchone()[0] == result["cost_usd"]
    assert (
        prepare_concept_brief(
            settings,
            repo,
            now=now + timedelta(days=1),
            transport=lambda *args: pytest.fail("Uncertain corpus must not be repaid"),
        )["status"]
        == "already_attempted"
    )


def test_small_per_call_cap_blocks_larger_batch_before_paid_call(concept_env):
    settings, repo, now, _ = concept_env
    settings.concept_briefs_max_cost_usd = 0.001
    result = prepare_concept_brief(
        settings, repo, now=now, transport=lambda *args: pytest.fail("Request exceeds cap")
    )
    assert result["status"] == "request_too_large"
    with repo.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM concept_runs").fetchone()[0] == 0


def test_pre_upgrade_daily_claim_is_not_retried_for_second_card(concept_env):
    settings, repo, now, _ = concept_env
    with repo.connection() as db:
        db.execute(
            "INSERT INTO concept_runs(window_key,corpus_hash,status,created_at) VALUES(?,?,?,?)",
            ("2026-10-13", "legacy-corpus", "ready", now.isoformat()),
        )
    result = prepare_concept_brief(
        settings, repo, now=now, transport=lambda *args: pytest.fail("Upgrade must not repay today")
    )
    assert result["status"] == "already_attempted" and result["cards_created"] == 0
    with repo.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM concept_runs").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM usage_ledger").fetchone()[0] == 0


def test_prompt_requires_verbatim_original_language_quotes_separate_from_russian_explanation(
    concept_env,
):
    settings, repo, now, _ = concept_env
    checked = []

    def transport(request, key):
        prompt = request["messages"][0]["content"]
        for requirement in (
            "непрерывный дословный фрагмент",
            "исходный язык, регистр, пунктуацию, пробелы и переносы",
            "Не переводи даже отдельные слова внутри quote",
            "не меняй знак вопроса на точку",
            "не добавляй свои вводные слова",
            "Русское пояснение пиши отдельно, в поле text пояснения",
        ):
            assert requirement in prompt
        checked.append(True)
        return response([])

    result = prepare_concept_brief(settings, repo, now=now, transport=transport)
    assert result["status"] == "empty" and checked == [True]


def test_concept_choice_receives_current_personal_priorities_without_topic_bans(concept_env):
    settings, repo, now, item = concept_env
    profile = {
        "version": 7,
        "topics": ["agents", "startups", "geopolitics"],
        "goals": [
            "Главный интерес — AI coding, AI-агенты, harness и domain-native стартапы.",
            "Сохранять сильные события геополитики и остальные интересы.",
        ],
        "positive_patterns": ["Рабочие приёмы агентов и устройство предметных инструментов."],
        "avoid_patterns": ["Не добавлять слабые модные слова ради тематического перевеса."],
        "current_projects": ["Materium и прикладные AI-агенты."],
    }
    settings.interests_file = settings.data_dir / "interests.yaml"
    settings.interests_file.write_text(json.dumps(profile), encoding="utf-8")
    calls = []

    def transport(request, key):
        data = json.loads(request["messages"][1]["content"])
        assert data["interest_profile"] == profile
        prompt = request["messages"][0]["content"]
        assert "главный интерес должен заметно" in prompt
        assert "мягкий приоритет, а не обязательная квота" in prompt
        assert "не запрещай их ради тематического перевеса" in prompt
        calls.append(request)
        return response([concept(item).model_dump()])

    result = prepare_concept_brief(settings, repo, now=now, transport=transport)
    assert result["cards_created"] == 1 and len(calls) == 1
    with repo.connection() as db:
        metadata = json.loads(db.execute("SELECT metadata_json FROM usage_ledger").fetchone()[0])
        assert metadata["profile_version"] == 7


def test_large_profile_is_bounded_before_estimation_and_trimming(concept_env):
    settings, repo, now, _ = concept_env
    settings.interests_file = settings.data_dir / "interests.yaml"
    settings.interests_file.write_text(
        json.dumps(
            {
                "version": 8,
                "topics": ["agents_" + "x" * 60] * 33,
                "goals": ["Приоритет AI coding и harness. " + "Детали. " * 45] * 7,
                "positive_patterns": ["Содержательные рабочие приёмы. " * 11] * 13,
                "avoid_patterns": ["Никакого искусственного заполнения. " * 9] * 11,
                "current_projects": ["Прикладной AI. " * 21] * 7,
            }
        ),
        encoding="utf-8",
    )
    compact = concept_interest_profile(settings)
    assert len(compact["goals"]) == 6 and max(map(len, compact["goals"])) <= 320
    assert len(compact["positive_patterns"]) == 12
    assert len(compact["avoid_patterns"]) == 10
    assert len(compact["current_projects"]) == 6
    assert len(compact["topics"]) == 32
    calls = []

    def transport(request, key):
        estimate = (
            len(json.dumps(request, ensure_ascii=False).encode()) * 0.5 + request["max_tokens"] * 2
        ) / 1_000_000
        assert estimate <= settings.concept_briefs_max_cost_usd
        calls.append(request)
        return response([])

    result = prepare_concept_brief(settings, repo, now=now, transport=transport)
    assert result["status"] in {"empty", "request_too_large"}
    assert len(calls) == int(result["status"] == "empty")


def test_missing_personal_profile_uses_bounded_empty_preferences(concept_env):
    settings, _, _, _ = concept_env
    settings.interests_file = settings.data_dir / "missing.yaml"
    compact = concept_interest_profile(settings)
    assert compact == {
        "version": 1,
        "topics": [],
        "goals": [],
        "positive_patterns": [],
        "avoid_patterns": [],
        "current_projects": [],
    }


def add_corpus_posts(repo, now, *, count=8):
    registry = [
        SourceConfig(
            id="x",
            name="Author",
            kind="x",
            locator="author",
            role="author_radar",
            region="global",
            language="en",
        )
    ] + [
        SourceConfig(
            id=f"extra{i}",
            name=f"Author {i}",
            kind="x",
            locator=f"author{i}",
            role="author_radar",
            region="global",
            language="en",
        )
        for i in range(count)
    ]
    repo.sync_sources(SourceRegistry(version=1, sources=registry))
    for i in range(count):
        repo.insert_source_item_with_id(
            source_id=f"extra{i}",
            external_id=str(200000 + i),
            source_url=f"https://x.com/author{i}/status/{200000 + i}",
            original_text=f"A separate substantial concept {i}. "
            + "Detailed source evidence. " * 90,
            original_language="en",
            published_at=(now - timedelta(hours=2)).isoformat(),
            outbound_urls=[],
            media_metadata={},
            payload_hash=f"extra{i}",
            raw_payload={},
        )


@pytest.mark.parametrize("status", ["uncertain", "ready", "invalid", "empty"])
def test_profile_only_trim_changes_cannot_repay_same_observed_corpus(concept_env, status):
    settings, repo, now, item = concept_env
    add_corpus_posts(repo, now)
    settings.concept_briefs_max_cost_usd = 0.018
    settings.interests_file = settings.data_dir / "interests.yaml"
    settings.interests_file.write_text(json.dumps({"version": 7, "goals": ["AI coding"]}))
    calls = []

    def transport(request, key):
        calls.append(request)
        if status == "uncertain":
            raise TimeoutError("unknown provider outcome")
        if status == "empty":
            return response([])
        draft = concept(item).model_dump()
        if status == "invalid":
            draft["definition"]["quote"] = "A fabricated source quote that is not present."
        return response([draft])

    first = prepare_concept_brief(settings, repo, now=now, transport=transport)
    assert first["status"] == status
    original_count = len(json.loads(calls[0]["messages"][1]["content"])["sources"])
    settings.interests_file.write_text(
        json.dumps(
            {
                "version": 8,
                "goals": ["Главный интерес — AI coding и harness. " + "Детали. " * 30] * 6,
                "positive_patterns": ["Рабочие механизмы и практическая польза. " * 7] * 6,
            }
        )
    )
    # Confirm the larger profile would remove additional posts at the same cap.
    observed = recent_concept_sources(repo, now)
    request = json.loads(json.dumps(calls[0]))
    payload = json.loads(request["messages"][1]["content"])
    payload["interest_profile"] = concept_interest_profile(settings)
    payload["sources"] = list(observed)
    while payload["sources"]:
        request["messages"][1]["content"] = json.dumps(payload, ensure_ascii=False)
        estimate = (
            len(json.dumps(request, ensure_ascii=False).encode()) * 0.5 + request["max_tokens"] * 2
        ) / 1_000_000
        if estimate <= settings.concept_briefs_max_cost_usd:
            break
        payload["sources"].pop()
    assert 0 < len(payload["sources"]) < original_count
    result = prepare_concept_brief(
        settings,
        repo,
        now=now + timedelta(days=1),
        transport=lambda *args: pytest.fail("Profile-only trimming must not cause repayment"),
    )
    assert result["status"] == "already_attempted"
    with repo.connection() as db:
        stored_hash = db.execute("SELECT corpus_hash FROM concept_runs").fetchone()[0]
        assert (
            stored_hash == hashlib.sha256(json.dumps(observed, sort_keys=True).encode()).hexdigest()
        )
        assert db.execute("SELECT COUNT(*) FROM usage_ledger").fetchone()[0] == 1


@pytest.mark.parametrize("status", ["uncertain", "running", "ready", "invalid", "empty"])
def test_legacy_trimmed_corpus_is_not_repaid_after_profile_edit(concept_env, status):
    settings, repo, now, _ = concept_env
    add_corpus_posts(repo, now, count=3)
    snapshot = recent_concept_sources(repo, now)[:1]
    legacy_hash = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()
    with repo.connection() as db:
        db.execute(
            "INSERT INTO concept_runs(window_key,corpus_hash,status,created_at,"
            "source_snapshot_json) "
            "VALUES(?,?,?,?,?)",
            (
                "2026-10-12",
                legacy_hash,
                status,
                (now - timedelta(days=1)).isoformat(),
                json.dumps(snapshot),
            ),
        )
    assert (
        prepare_concept_brief(
            settings,
            repo,
            now=now,
            transport=lambda *args: pytest.fail("Legacy leading corpus must not be repaid"),
        )["status"]
        == "already_attempted"
    )


def test_new_post_before_legacy_snapshot_permits_a_genuinely_new_corpus(concept_env):
    settings, repo, now, _ = concept_env
    snapshot = recent_concept_sources(repo, now)
    with repo.connection() as db:
        db.execute(
            "INSERT INTO concept_runs(window_key,corpus_hash,status,created_at,"
            "source_snapshot_json) "
            "VALUES(?,?,'uncertain',?,?)",
            (
                "2026-10-12",
                hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest(),
                (now - timedelta(days=1)).isoformat(),
                json.dumps(snapshot),
            ),
        )
    repo.insert_source_item_with_id(
        source_id="x",
        external_id="3333333",
        source_url="https://x.com/author/status/3333333",
        original_text="A genuinely new explanation. "
        + "New evidence changes the source corpus. " * 5,
        original_language="en",
        published_at=(now - timedelta(minutes=10)).isoformat(),
        outbound_urls=[],
        media_metadata={},
        payload_hash="new",
        raw_payload={},
    )
    calls = []

    def transport(request, key):
        calls.append(request)
        return response([])

    assert prepare_concept_brief(settings, repo, now=now, transport=transport)["status"] == "empty"
    assert len(calls) == 1
