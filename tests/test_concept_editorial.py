from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime, timedelta

import pytest
import yaml

from app.config import Settings, SourceConfig, SourceRegistry, runtime_source_registry
from app.db.repository import Repository
from app.radar.concepts import (
    IndustryBrief,
    prepare_concept_brief,
    recent_concept_sources,
    render_brief,
    validate_brief,
)


@pytest.fixture
def editorial(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path, openrouter_api_key="offline")
    settings.interests_file = tmp_path / "missing.yaml"
    repo = Repository(settings.database_path)
    repo.migrate()
    repo.sync_sources(
        SourceRegistry(
            version=1,
            sources=[
                SourceConfig(
                    id="x",
                    name="Search",
                    kind="x",
                    locator="agents",
                    role="radar",
                    region="global",
                    language="en",
                    candidate_enabled=False,
                )
            ],
        )
    )
    return settings, repo, datetime(2026, 10, 28, 12, tzinfo=UTC)


def post(repo, now, identifier, author, text, *, age=1):
    return repo.insert_source_item_with_id(
        source_id="x",
        external_id=str(identifier),
        source_url=f"https://x.com/{author}/status/{identifier}",
        original_text=text,
        original_language="en",
        published_at=(now - timedelta(hours=age)).isoformat(),
        outbound_urls=[],
        media_metadata={},
        payload_hash=str(identifier),
        raw_payload={},
    )


def draft(item):
    return IndustryBrief.model_validate(
        {
            "term": "Проверка результата внутри агентного цикла",
            "kind": "engineering_practice",
            "paragraphs": [
                {
                    "text": "Команда запускает тесты внутри каждого цикла работы агента, "
                    "а не только в конце.",
                    "source_item_id": item,
                    "quote": "We built a coding agent with validation inside its loop.",
                },
                {
                    "text": "Вместо продолжения работы после ошибки агент получает "
                    "результат проверки и исправляет код.",
                    "source_item_id": item,
                    "quote": "Instead of ignoring a failure, it reads the test result "
                    "and fixes the code.",
                },
            ],
        }
    )


TEXT = (
    "We built a coding agent with validation inside its loop. "
    "Instead of ignoring a failure, it reads the test result and fixes the code."
)


def test_engineering_practice_needs_no_invented_literal_term_and_keeps_free_paragraphs(editorial):
    settings, repo, now = editorial
    item = post(repo, now, 1, "builder", TEXT)
    brief = draft(item)
    sources = recent_concept_sources(repo, now)
    calls = []

    def generate(request, _key):
        calls.append(request)
        schema = request["response_format"]["json_schema"]["schema"]
        assert "paragraphs" in schema["$defs"]["IndustryBrief"]["properties"]
        return {
            "choices": [{"message": {"content": json.dumps({"briefs": [brief.model_dump()]})}}],
            "usage": {"cost": 0.001},
        }

    assert validate_brief(brief, sources) == sources
    result = prepare_concept_brief(settings, repo, now=now, transport=generate)
    assert result["cards_created"] == 1
    with repo.connection() as db:
        card = dict(db.execute("SELECT * FROM cards").fetchone())
    assert card["rendered_html"] == render_brief(brief, sources)
    for boilerplate in [
        "Почему об этом говорят",
        "Что это меняет",
        "единичный сигнал",
        "распространённость",
    ]:
        assert boilerplate not in card["rendered_html"]
    assert card["rendered_html"].count("\n\n") == 3
    assert "https://x.com/builder/status/1" in card["rendered_html"]
    assert (
        prepare_concept_brief(settings, repo, now=now, transport=generate)["status"]
        == "already_attempted"
    )
    assert len(calls) == 1 and repo.current_month_cost() == pytest.approx(0.001)


@pytest.mark.parametrize("bad", ["quote", "id", "invented_term", "popularity"])
def test_free_structure_still_rejects_unsupported_content(editorial, bad):
    _, repo, now = editorial
    item = post(repo, now, 1, "builder", TEXT)
    candidate = draft(item).model_dump()
    if bad == "quote":
        candidate["paragraphs"][1]["quote"] = "This sentence does not appear in the source."
    elif bad == "id":
        candidate["paragraphs"][1]["source_item_id"] = item + 999
    elif bad == "invented_term":
        candidate["kind"] = "concept"
    else:
        candidate["paragraphs"][1]["text"] = (
            "Все обсуждают этот подход и повсеместно применяют его в своей работе."
        )
    with pytest.raises(ValueError):
        validate_brief(IndustryBrief.model_validate(candidate), recent_concept_sources(repo, now))


def test_older_substantive_posts_survive_busy_search_and_actual_authors_are_balanced(editorial):
    _, repo, now = editorial
    for i in range(10):
        post(repo, now, 100 + i, "busy", TEXT + f" Evaluation example {i}.", age=0.1 + i / 10)
    older = post(
        repo,
        now,
        200,
        "engineer",
        TEXT + " We tested the architecture with memory constraints.",
        age=8,
    )
    for i in range(3):
        post(repo, now, 300 + i, f"other{i}", TEXT + f" Different benchmark {i}.", age=6 + i)
    corpus = recent_concept_sources(repo, now)
    assert older in [row["id"] for row in corpus]
    counts = Counter(row["url"].split("/")[3] for row in corpus)
    assert len(counts) >= 4 and counts["busy"] <= 3
    assert sum(len(row["text"]) for row in corpus) <= 24_000
    repo.set_runtime_setting("source_enabled:x", "false")
    assert recent_concept_sources(repo, now) == []


def test_author_poll_override_preserves_legacy_tier_and_manual_pause(tmp_path):
    sources, authors = tmp_path / "sources.yaml", tmp_path / "authors.yaml"
    sources.write_text("version: 1\nsources: []\n")
    authors.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "authors": [
                    {
                        "handle": "builder",
                        "name": "Builder",
                        "initial_tier": "active",
                        "poll_interval_minutes": 1440,
                        "enabled": False,
                    },
                    {"handle": "analyst", "name": "Analyst", "initial_tier": "active"},
                ],
            }
        )
    )
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        sources_file=sources,
        x_authors_file=authors,
        sensory_sources_file=tmp_path / "none.yaml",
    )
    rows = {s.locator: s for s in runtime_source_registry(settings).sources}
    assert rows["builder"].poll_interval_minutes == 1440 and not rows["builder"].enabled
    assert rows["analyst"].poll_interval_minutes == 2880
