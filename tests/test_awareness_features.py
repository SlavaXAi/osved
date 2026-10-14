from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr

from app.config import Settings, SourceConfig, SourceRegistry
from app.db.repository import Repository
from app.ops.schedule import fresh_delivery_card_ids
from app.radar.concepts import (
    ConceptBrief,
    concept_key,
    prepare_concept_brief,
    recent_concept_sources,
    store_brief,
    validate_brief,
)
from app.telegram_bot.news_depth import brief_card_html
from app.telegram_bot.sensory import send_video_recommendations, video_keyboard
from app.video.preferences import channel_identity, preference, save_preference
from app.video.shorts import ShortsClassifier, classify_page, metadata_type, player_type

OWNER = 123
CHANNEL = "UC" + "a" * 22
VIDEO = "abcdefghijk"
TEXT = (
    "Domain-native harness means a control loop built around domain tools and validations. "
    "We discuss it because generic agent workflows fail on domain-specific tasks. "
    "Domain-native harness design makes domain constraints explicit to the agent."
)


@pytest.fixture
def env(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, app_env="test", data_dir=tmp_path, allowed_user_id=OWNER)
    repo = Repository(settings.database_path)
    repo.migrate()
    sources = [
        SourceConfig(
            id="yt",
            name="Channel",
            kind="youtube",
            locator=CHANNEL,
            role="radar",
            region="global",
            language="en",
        ),
        SourceConfig(
            id="x",
            name="Author",
            kind="x",
            locator="author",
            role="author_radar",
            region="global",
            language="en",
        ),
    ]
    repo.sync_sources(SourceRegistry(version=1, sources=sources))
    monkeypatch.setattr("app.telegram_bot.sensory.asyncio.sleep", AsyncMock())
    return settings, repo


def video(repo, video_id=VIDEO, *, source="yt", is_short=False):
    return repo.insert_source_item_with_id(
        source_id=source,
        external_id=video_id,
        source_url="https://www.youtube.com/watch?v=" + video_id,
        original_text="A substantive conversation about engineering reliable AI agents.",
        original_language="en",
        published_at=datetime.now(UTC).isoformat(),
        outbound_urls=[],
        media_metadata={"video_id": video_id, "channel_id": CHANNEL, "is_short": is_short},
        payload_hash=video_id,
        raw_payload={},
    )


def x_item(repo, *, external_id="1234567890", author="alice", text=TEXT, age=0):
    return repo.insert_source_item_with_id(
        source_id="x",
        external_id=external_id,
        source_url=f"https://x.com/{author}/status/{external_id}",
        original_text=text,
        original_language="en",
        published_at=(datetime.now(UTC) - timedelta(days=age)).isoformat(),
        outbound_urls=[],
        media_metadata={},
        payload_hash=external_id,
        raw_payload={},
    )


class Bot:
    def __init__(self):
        self.sent = []

    async def send_message(self, **kwargs):
        self.sent.append(kwargs)
        return SimpleNamespace(message_id=700 + len(self.sent))


@pytest.mark.parametrize("kind", [True, None])
def test_shorts_and_unknown_never_call_description_or_send(env, monkeypatch, kind):
    settings, repo = env
    video(repo, is_short=kind)
    monkeypatch.setattr(ShortsClassifier, "check_item", lambda *args: kind)
    description = AsyncMock(side_effect=AssertionError("No paid call"))
    monkeypatch.setattr("app.telegram_bot.sensory.video_description", description)
    bot = Bot()
    assert asyncio.run(send_video_recommendations(bot, settings, repo, chat_id=OWNER))["sent"] == 0
    with repo.connection() as db:
        assert db.execute("SELECT count(*) FROM video_recommendations").fetchone()[0] == 0
    assert not bot.sent and not description.called


def test_short_metadata_and_native_type_cache(env):
    _, repo = env
    item = video(repo, is_short=None)
    with repo.connection() as db:
        row = dict(db.execute("SELECT * FROM source_items WHERE id=?", (item,)).fetchone())
    calls = []
    classifier = ShortsClassifier(repo, probe=lambda vid: calls.append(vid) or True)
    assert classifier.check_item(row) is True
    assert (
        ShortsClassifier(Repository(repo.database_path), probe=lambda _: pytest.fail()).check_item(
            row
        )
        is True
    )
    assert calls == [VIDEO]
    with repo.connection() as db:
        assert (
            db.execute("SELECT processing_status FROM source_items").fetchone()[0]
            == "filtered_short"
        )


def test_unknown_is_cached_then_can_be_rechecked(env):
    _, repo = env
    item = video(repo, is_short=None)
    with repo.connection() as db:
        row = dict(db.execute("SELECT * FROM source_items WHERE id=?", (item,)).fetchone())
    classifier = ShortsClassifier(repo, probe=lambda _: None)
    assert classifier.check_item(row) is None
    assert ShortsClassifier(repo, probe=lambda _: pytest.fail()).check_item(row) is None
    with repo.connection() as db:
        db.execute(
            "UPDATE youtube_video_types SET retry_after=?",
            ((datetime.now(UTC) - timedelta(seconds=1)).isoformat(),),
        )
    assert ShortsClassifier(repo, probe=lambda _: False).check_item(row) is False


def test_short_duration_is_not_a_short_and_long_duration_is_definitive():
    assert metadata_type({"duration_seconds": 90}) is None
    assert metadata_type({"duration_seconds": 600}) is False
    assert player_type({"videoDetails": {"lengthSeconds": "181"}}) is False
    assert player_type({"videoDetails": {"lengthSeconds": "90", "isShorts": True}}) is True


@pytest.mark.parametrize(
    "route,expected",
    [
        (f"https://www.youtube.com/shorts/{VIDEO}", True),
        (f"https://www.youtube.com/watch?v={VIDEO}", False),
        ("https://www.youtube.com/watch?v=wrongxxxxxx", None),
        ("https://consent.youtube.com/", None),
    ],
)
def test_native_page_requires_exact_identity(route, expected):
    assert classify_page(f'<link rel="canonical" href="{route}">'.encode(), VIDEO) is expected


def test_less_selects_stronger_fit_and_more_resumes_cached_weaker_without_time(env, monkeypatch):
    settings, repo = env
    for vid in (VIDEO, "bcdefghijkl", "cdefghijklm"):
        video(repo, vid)
    scores = {VIDEO: 0.9, "bcdefghijkl": 0.65, "cdefghijklm": 0.5}

    def assess(item, profile):
        return {
            "description_html": item["external_id"],
            "relevance": scores[item["external_id"]],
            "matched_topics": ["agents"],
            "reason": "Personal content fit",
        }

    monkeypatch.setattr("app.video.relevance.local_assessment", assess)
    with repo.connection() as db:
        db.execute(
            "INSERT INTO youtube_channel_preferences VALUES(?,?,-1,?)",
            (OWNER, CHANNEL, datetime.now(UTC).isoformat()),
        )
    bot = Bot()
    first = asyncio.run(send_video_recommendations(bot, settings, repo, chat_id=OWNER))
    assert first["sent"] == 1
    monkeypatch.setattr(
        "app.video.relevance.local_assessment", lambda *args: pytest.fail("Use cache")
    )
    assert asyncio.run(send_video_recommendations(bot, settings, repo, chat_id=OWNER))["sent"] == 0
    save_preference(
        repo,
        chat_id=OWNER,
        channel_id=CHANNEL,
        value=1,
        recommendation_id=first["recommendation_ids"][0],
    )
    assert preference(Repository(repo.database_path), OWNER, CHANNEL) == 1
    assert asyncio.run(send_video_recommendations(bot, settings, repo, chat_id=OWNER))["sent"] == 2
    with repo.connection() as db:
        assert db.execute("SELECT count(*) FROM video_feedback_events").fetchone()[0] == 1
    assert not save_preference(
        repo,
        chat_id=OWNER,
        channel_id=CHANNEL,
        value=1,
        recommendation_id=first["recommendation_ids"][0],
    )


def test_preference_applies_to_channel_aliases_and_is_owner_specific(env):
    _, repo = env
    a = {"source_id": "a", "source_locator": CHANNEL}
    b = {"source_id": "b", "media_metadata_json": json.dumps({"channel_id": CHANNEL})}
    assert channel_identity(a) == channel_identity(b) == CHANNEL
    assert preference(repo, 456, CHANNEL) == 0


def test_more_ranks_before_newer_neutral_channel(env):
    settings, repo = env
    video(repo)
    other = SourceConfig(
        id="other",
        name="Other",
        kind="youtube",
        locator="UC" + "b" * 22,
        role="radar",
        region="global",
        language="en",
    )
    repo.sync_sources(SourceRegistry(version=1, sources=[other]))
    item = video(repo, "bcdefghijkl", source="other")
    with repo.connection() as db:
        db.execute(
            "UPDATE source_items SET media_metadata_json=? WHERE id=?",
            (json.dumps({"is_short": False, "channel_id": other.locator}), item),
        )
        db.execute(
            "INSERT INTO youtube_channel_preferences VALUES(?,?,1,?)",
            (OWNER, CHANNEL, datetime.now(UTC).isoformat()),
        )
    bot = Bot()
    result = asyncio.run(send_video_recommendations(bot, settings, repo, chat_id=OWNER, limit=1))
    with repo.connection() as db:
        row = db.execute(
            "SELECT i.source_id FROM video_recommendations v JOIN source_items i "
            "ON i.id=v.source_item_id WHERE v.id=?",
            (result["recommendation_ids"][0],),
        ).fetchone()
    assert row[0] == "yt"


def test_channel_buttons_are_absent_and_menu_controls_remain():
    for expanded in (True, False):
        markup = video_keyboard(1, expanded=expanded, channel_preference=-1)
        buttons = [b for row in markup.inline_keyboard for b in row]
        assert all("Показывать" not in b.text for b in buttons)
        actions = [b.callback_data.rsplit(":", 1)[-1] for b in buttons if b.callback_data]
        assert actions == (["short", "long", "dismiss"] if expanded else ["open", "dismiss"])
        assert all(len(b.callback_data.encode()) <= 64 for b in buttons if b.callback_data)


def brief(source_id):
    return ConceptBrief.model_validate(
        {
            "term": "Domain-native harness",
            "kind": "strategy",
            "definition": {
                "text": "Контур управления агентом, построенный вокруг инструментов "
                "и правил предметной области.",
                "source_item_id": source_id,
                "quote": TEXT.split(". ")[0],
            },
            "why_discussed": {
                "text": "Автор обсуждает этот подход из-за сбоев универсальных агентных "
                "процессов на специализированных задачах.",
                "source_item_id": source_id,
                "quote": TEXT.split(". ")[1],
            },
            "practical_meaning": {
                "text": "Такой подход явно передаёт агенту ограничения предметной области, "
                "по объяснению автора.",
                "source_item_id": source_id,
                "quote": TEXT.split(". ")[2],
            },
        }
    )


def test_concept_route_reads_posts_rejected_by_news_gate(env):
    _, repo = env
    item = x_item(repo)
    repo.mark_source_item_status(item, "rejected", "not_a_dated_event")
    assert [r["id"] for r in recent_concept_sources(repo, datetime.now(UTC))] == [item]
    x_item(repo, external_id="2", age=10)
    assert len(recent_concept_sources(repo, datetime.now(UTC))) == 1
    repo.set_runtime_setting("source_enabled:x", "false")
    assert recent_concept_sources(repo, datetime.now(UTC)) == []


def test_concept_card_is_full_in_feed_and_deduped(env):
    _, repo = env
    item = x_item(repo)
    now = datetime.now(UTC)
    sources = recent_concept_sources(repo, now)
    draft = brief(item)
    card_id = store_brief(repo, draft, sources, now=now, model="offline", cost=0)
    assert card_id is not None
    assert store_brief(repo, draft, sources, now=now, model="offline", cost=0) is None
    assert concept_key("domain native harness") == concept_key(draft.term)
    assert fresh_delivery_card_ids(repo, owner_id=OWNER, now=now, limit=5) == [card_id]
    card = repo.deliverable_cards_by_ids(OWNER, [card_id])[0]
    rendered = brief_card_html(card)
    assert "Почему об этом говорят" in rendered and "Что это меняет" in rendered
    assert "единичный сигнал" in rendered and sources[0]["url"] in rendered
    delivery = repo.ensure_delivery(card_id, OWNER)
    repo.mark_delivery_sent(delivery, 42)
    assert fresh_delivery_card_ids(repo, owner_id=OWNER, now=now, limit=5) == []


@pytest.mark.parametrize("bad", ["fake_quote", "fake_id", "absent_term", "popularity"])
def test_concept_rejects_invented_evidence_or_popularity(env, bad):
    _, repo = env
    item = x_item(repo)
    data = brief(item).model_dump()
    if bad == "fake_quote":
        data["definition"]["quote"] = "An invented quote that is not in the source."
    elif bad == "fake_id":
        data["definition"]["source_item_id"] += 1000
    elif bad == "absent_term":
        data["term"] = "Quantum economic harness"
    else:
        data["definition"]["text"] = (
            "Все обсуждают этот подход, он уже используется повсеместно "
            "в технологическом сообществе."
        )
    with pytest.raises(ValueError):
        validate_brief(
            ConceptBrief.model_validate(data), recent_concept_sources(repo, datetime.now(UTC))
        )


def test_concept_generation_is_daily_budgeted_and_unknown_not_retried(env):
    settings, repo = env
    settings.openrouter_api_key = SecretStr("offline")
    item = x_item(repo)
    calls = []

    def transport(request, key):
        calls.append(request)
        return {
            "choices": [
                {"message": {"content": json.dumps({"briefs": [brief(item).model_dump()]})}}
            ],
            "usage": {"cost": 0.001, "prompt_tokens": 100, "completion_tokens": 100},
        }

    result = prepare_concept_brief(settings, repo, transport=transport)
    assert result["cards_created"] == 1 and result["cost_usd"] == 0.001
    assert (
        prepare_concept_brief(settings, repo, transport=transport)["status"] == "already_attempted"
    )
    assert len(calls) == 1
    now = datetime.now(UTC) + timedelta(days=1)
    x_item(repo, external_id="2", text=TEXT + " A fresh additional discussion.")
    failure = prepare_concept_brief(
        settings, repo, now=now, transport=lambda *args: (_ for _ in ()).throw(TimeoutError())
    )
    assert failure["status"] == "uncertain" and failure["cost_usd"] > 0
    assert (
        prepare_concept_brief(settings, repo, now=now, transport=transport)["status"]
        == "already_attempted"
    )
    assert len(calls) == 1


def test_concept_hard_budget_prevents_call(env):
    settings, repo = env
    settings.openrouter_api_key = SecretStr("offline")
    x_item(repo)
    repo.record_usage("test", "test", cost_usd=settings.budget_hard_stop_usd)
    result = prepare_concept_brief(settings, repo, transport=lambda *args: pytest.fail("Paid call"))
    assert result["status"] == "budget_blocked"


def test_concept_does_not_merge_with_news_using_the_same_post(env):
    from app.pipeline.dedupe import deduplicate_events

    _, repo = env
    item = x_item(repo)
    now = datetime.now(UTC)
    sources = recent_concept_sources(repo, now)
    store_brief(repo, brief(item), sources, now=now, model="offline", cost=0)
    repo.attach_item_to_event(
        source_item_id=item,
        event_type="x_signal",
        headline="New agent workflow released",
        event_at=now.isoformat(),
        primary_url=sources[0]["url"],
        fingerprint="ordinary_news",
        score=7.0,
        score_features={},
        decision_reason="event",
    )
    assert deduplicate_events(repo).events_merged == 0


def test_short_type_checks_have_bounded_network_work(env):
    _, repo = env
    calls = []
    classifier = ShortsClassifier(repo, probe=lambda vid: calls.append(vid), max_probes=1)
    for vid in (VIDEO, "bcdefghijkl"):
        item = video(repo, vid, is_short=None)
        with repo.connection() as db:
            row = dict(db.execute("SELECT * FROM source_items WHERE id=?", (item,)).fetchone())
        assert classifier.check_item(row) is None
    assert len(calls) == 1


def test_concurrent_less_channel_sends_multiple_relevant_videos_once(env):
    settings, repo = env
    for vid in (VIDEO, "bcdefghijkl"):
        video(repo, vid)
    with repo.connection() as db:
        db.execute(
            "INSERT INTO youtube_channel_preferences VALUES(?,?,-1,?)",
            (OWNER, CHANNEL, datetime.now(UTC).isoformat()),
        )

    async def run():
        return await asyncio.gather(
            send_video_recommendations(Bot(), settings, repo, chat_id=OWNER),
            send_video_recommendations(Bot(), settings, repo, chat_id=OWNER),
        )

    assert sum(result["sent"] for result in asyncio.run(run())) == 2
    with repo.connection() as db:
        assert db.execute("SELECT count(*) FROM video_recommendations").fetchone()[0] == 2


def test_large_unicode_corpus_is_reduced_to_budget_before_model_call(env):
    settings, repo = env
    settings.openrouter_api_key = SecretStr("offline")
    sources = [
        SourceConfig(
            id=f"x{i}",
            name=f"Author {i}",
            kind="x",
            locator=f"author{i}",
            role="author_radar",
            region="global",
            language="ru",
        )
        for i in range(20)
    ]
    repo.sync_sources(SourceRegistry(version=1, sources=sources))
    with repo.connection() as db:
        for i in range(20):
            text = f"Автор {i} обсуждает стратегию. " + "Содержательное описание подхода " * 100
            db.execute(
                "INSERT INTO source_items(source_id,external_id,source_url,original_text,"
                "published_at,payload_hash) VALUES(?,?,?,?,?,?)",
                (
                    f"x{i}",
                    str(i),
                    f"https://x.com/author{i}/status/{10000 + i}",
                    text,
                    datetime.now(UTC).isoformat(),
                    str(i),
                ),
            )
    calls = []

    def transport(request, key):
        calls.append(request)
        estimate = (
            len(json.dumps(request, ensure_ascii=False).encode()) * 0.5 + request["max_tokens"] * 2
        ) / 1_000_000
        assert estimate <= settings.concept_briefs_max_cost_usd
        return {
            "choices": [{"message": {"content": '{"briefs":[]}'}}],
            "usage": {"cost": 0, "is_byok": False},
        }

    result = prepare_concept_brief(settings, repo, transport=transport)
    assert result["status"] == "empty" and len(calls) == 1
    with repo.connection() as db:
        assert db.execute("SELECT response_json FROM concept_runs").fetchone()[0]


@pytest.mark.parametrize("kind", [True, None, False])
def test_existing_youtube_news_candidates_and_cards_obey_short_filter(env, kind):
    from app.ops.schedule import fresh_research_events

    _, repo = env
    item = video(repo, is_short=kind)
    now = datetime.now(UTC)
    event, _ = repo.attach_item_to_event(
        source_item_id=item,
        event_type="youtube_signal",
        headline="Existing video candidate",
        event_at=now.isoformat(),
        primary_url="https://www.youtube.com/watch?v=" + VIDEO,
        fingerprint="existing_video",
        score=7.0,
        score_features={},
        decision_reason="existing",
    )
    candidates = fresh_research_events(repo, now=now)
    assert bool(candidates) == (kind is False)
    repo.create_card(
        event_id=event,
        prompt_version="test",
        model="offline",
        generation_cost_usd=0,
        structured_output={"headline": "Existing video"},
        rendered_html="Existing video",
        source_urls=["https://www.youtube.com/watch?v=" + VIDEO],
        primary_preview_url="https://www.youtube.com/watch?v=" + VIDEO,
    )
    assert bool(fresh_delivery_card_ids(repo, owner_id=OWNER, now=now, limit=5)) == (kind is False)
