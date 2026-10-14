import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr

from app.config import Settings, SourceConfig, SourceRegistry
from app.db.repository import Repository
from app.telegram_bot.sensory import send_video_recommendations
from app.video.relevance import VideoRelevanceService, personalization_profile, validated_assessment

OWNER = 123
CHANNEL = "UC" + "a" * 22


@pytest.fixture
def env(tmp_path, monkeypatch):
    interests = tmp_path / "interests.yaml"
    interests.write_text(
        "version: 1\ntopics: [ai, agents]\ngoals: [Понимать практическое применение агентов]\n"
    )
    settings = Settings(
        _env_file=None, data_dir=tmp_path, interests_file=interests, allowed_user_id=OWNER
    )
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
                    locator=CHANNEL,
                    role="radar",
                    region="global",
                    language="en",
                )
            ],
        )
    )
    monkeypatch.setattr("app.telegram_bot.sensory.asyncio.sleep", AsyncMock())
    return settings, repo


def item(
    repo, vid="abcdefghijk", text="A technical discussion about engineering reliable AI agents."
):
    identifier = repo.insert_source_item_with_id(
        source_id="yt",
        external_id=vid,
        source_url="https://www.youtube.com/watch?v=" + vid,
        original_text=text,
        original_language="en",
        published_at=datetime.now(UTC).isoformat(),
        outbound_urls=[],
        media_metadata={"video_id": vid, "channel_id": CHANNEL, "is_short": False},
        payload_hash=vid,
        raw_payload={},
    )
    with repo.connection() as db:
        return dict(
            db.execute(
                "SELECT i.*,s.name AS source_name,s.locator AS source_locator "
                "FROM source_items i JOIN sources s ON s.id=i.source_id WHERE i.id=?",
                (identifier,),
            ).fetchone()
        )


class Bot:
    def __init__(self):
        self.sent = []

    async def send_message(self, **kwargs):
        self.sent.append(kwargs)
        return SimpleNamespace(message_id=700 + len(self.sent))


def response():
    payload = {
        "title": "Агенты в практической работе",
        "description": "Практический выпуск об инженерии надёжных AI-агентов.",
        "relevance": 0.9,
        "matched_topics": ["ai", "agents"],
        "relevance_reason": "Соответствует цели применять агентов на практике.",
        "evidence_quotes": ["engineering reliable AI agents"],
    }
    return {
        "choices": [{"message": {"content": json.dumps(payload)}}],
        "usage": {"cost": 0.001, "is_byok": False},
    }


def test_one_call_combines_description_and_personal_match_and_is_cached(env, monkeypatch):
    settings, repo = env
    settings.openrouter_api_key = SecretStr("offline")
    video = item(repo)
    calls = []

    def model(request, key):
        calls.append(request)
        reader = json.loads(request["messages"][1]["content"])["reader"]
        assert reader["interests"]["goals"] == ["Понимать практическое применение агентов"]
        return response()

    monkeypatch.setattr("app.telegram_bot.sensory.http_transport", model)
    first = VideoRelevanceService(settings, repo, OWNER).assess(video)
    assert first["relevance"] == 0.9 and "<b>" in first["description_html"]
    assert VideoRelevanceService(settings, repo, OWNER).assess(video) == first
    assert len(calls) == 1
    with repo.connection() as db:
        assert db.execute("SELECT count(*) FROM usage_ledger").fetchone()[0] == 1
        assert db.execute("SELECT cost_usd FROM usage_ledger").fetchone()[0] == 0.001
        assert db.execute("SELECT response_json FROM video_relevance_assessments").fetchone()[0]


def test_semantic_assessment_requires_real_profile_topics_and_quotes(env):
    settings, repo = env
    video = item(repo)
    profile = personalization_profile(settings, repo, OWNER)
    payload = json.loads(response()["choices"][0]["message"]["content"])
    payload["matched_topics"] = ["fishing"]
    with pytest.raises(ValueError):
        validated_assessment(payload, item=video, profile=profile, description="x")
    payload["matched_topics"] = ["ai"]
    payload["evidence_quotes"] = ["An invented supporting quotation"]
    with pytest.raises(ValueError):
        validated_assessment(payload, item=video, profile=profile, description="x")


def test_unrelated_video_is_filtered_even_on_more_channel(env):
    settings, repo = env
    item(repo, text="A full tutorial about recipes for cakes and baking desserts.")
    with repo.connection() as db:
        db.execute(
            "INSERT INTO youtube_channel_preferences VALUES(?,?,1,?)",
            (OWNER, CHANNEL, datetime.now(UTC).isoformat()),
        )
    bot = Bot()
    assert asyncio.run(send_video_recommendations(bot, settings, repo, chat_id=OWNER))["sent"] == 0
    assert not bot.sent


def test_less_allows_multiple_strong_content_matches_without_timer(env):
    settings, repo = env
    settings.youtube_less_interval_hours = 168
    for vid in ["abcdefghijk", "bcdefghijkl", "cdefghijklm"]:
        item(repo, vid)
    with repo.connection() as db:
        db.execute(
            "INSERT INTO youtube_channel_preferences VALUES(?,?,-1,?)",
            (OWNER, CHANNEL, datetime.now(UTC).isoformat()),
        )
    assert (
        asyncio.run(send_video_recommendations(Bot(), settings, repo, chat_id=OWNER))["sent"] == 3
    )


def test_new_profile_changes_fit_without_reusing_other_users_feedback(env):
    settings, repo = env
    video = item(repo)
    assert VideoRelevanceService(settings, repo, OWNER).assess(video)["relevance"] >= 0.8
    settings.interests_file.write_text("version: 2\ntopics: [space]\n")
    assert VideoRelevanceService(settings, repo, OWNER).assess(video)["relevance"] < 0.45
    with repo.connection() as db:
        assert db.execute("SELECT count(*) FROM video_relevance_assessments").fetchone()[0] == 2


def test_uncertain_assessment_is_not_repaid_even_after_profile_changes(env, monkeypatch):
    settings, repo = env
    settings.openrouter_api_key = SecretStr("offline")
    video = item(repo)
    calls = []

    def model(*args):
        calls.append(1)
        raise TimeoutError("unknown result")

    monkeypatch.setattr("app.telegram_bot.sensory.http_transport", model)
    assert VideoRelevanceService(settings, repo, OWNER).assess(video) is None
    settings.interests_file.write_text("version: 2\ntopics: [ai]\n")
    assert VideoRelevanceService(settings, repo, OWNER).assess(video) is None
    assert len(calls) == 1 and repo.current_month_cost() > 0


def test_news_dislike_is_not_used_as_video_topic_dislike(env):
    settings, repo = env
    smoke = repo.create_smoke_delivery(OWNER, "https://example.com/news")
    repo.finalize_feedback(
        smoke["delivery_id"],
        action="reaction",
        reaction="dislike",
        comment="Плохая подача, тема хорошая",
    )
    assert personalization_profile(settings, repo, OWNER)["video_feedback"] == []
