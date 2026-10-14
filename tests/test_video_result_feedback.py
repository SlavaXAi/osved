from __future__ import annotations

import json
import time
from datetime import UTC, datetime

import pytest
from pydantic import SecretStr

from app.config import Settings, SourceConfig, SourceRegistry
from app.db.repository import Repository
from app.video.relevance import VideoRelevanceService, personalization_profile
from app.video.result_feedback import get_result_feedback, save_result_feedback

OWNER = 123
VIDEO = "abcdefghijk"


@pytest.fixture
def env(tmp_path):
    interests = tmp_path / "interests.yaml"
    interests.write_text("version: 1\ntopics: [ai, agents]\n")
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
                    name="Engineering Channel",
                    kind="youtube",
                    locator="UC" + "a" * 22,
                    role="radar",
                    region="global",
                    language="en",
                )
            ],
        )
    )
    return settings, repo


def job(repo, *, video=VIDEO, kind="tldr", owner=OWNER, chat=OWNER, status="sent"):
    stamp = time.time()
    with repo.connection() as db:
        # Each new request is a separate output; repeated callback saves are not.
        request_id = db.execute("SELECT count(*)+10 FROM video_pdf_jobs").fetchone()[0]
        return int(
            db.execute(
                "INSERT INTO video_pdf_jobs(video_id,user_id,chat_id,request_message_id,"
                "output_kind,status,text_message_id,document_message_id,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (video, owner, chat, request_id, kind, status, 80, 90, stamp, stamp),
            ).lastrowid
        )


def metadata(repo, *, video=VIDEO):
    identifier = repo.insert_source_item_with_id(
        source_id="yt",
        external_id=video,
        source_url="https://www.youtube.com/watch?v=" + video,
        original_text="Engineering reliable AI agents for industrial work.",
        original_language="en",
        published_at=datetime.now(UTC).isoformat(),
        outbound_urls=[],
        media_metadata={"video_id": video},
        payload_hash=video,
        raw_payload={},
    )
    with repo.connection() as db:
        return dict(
            db.execute(
                "SELECT i.*,s.name AS source_name FROM source_items i "
                "JOIN sources s ON s.id=i.source_id WHERE i.id=?",
                (identifier,),
            ).fetchone()
        )


def react(repo, identifier, reaction="like", **kwargs):
    return save_result_feedback(
        repo, job_id=identifier, chat_id=OWNER, user_id=OWNER, reaction=reaction, **kwargs
    )


def test_changes_are_idempotent_and_history_keeps_previous_reaction(env):
    _, repo = env
    identifier = job(repo)
    first = react(repo, identifier, message_id=80)
    assert first == {
        "job_id": identifier,
        "video_id": VIDEO,
        "output_kind": "tldr",
        "reaction": "like",
        "previous_reaction": None,
        "changed": True,
    }
    with repo.connection() as db:
        original = dict(db.execute("SELECT * FROM video_result_feedback").fetchone())
    assert react(repo, identifier)["changed"] is False
    with repo.connection() as db:
        assert dict(db.execute("SELECT * FROM video_result_feedback").fetchone()) == original
    assert react(repo, identifier, "dislike")["previous_reaction"] == "like"
    assert get_result_feedback(repo, job_id=identifier, chat_id=OWNER) == "dislike"
    assert get_result_feedback(repo, job_id=identifier, chat_id=999) is None
    with repo.connection() as db:
        events = db.execute(
            "SELECT reaction,previous_reaction FROM video_result_feedback_events ORDER BY id"
        ).fetchall()
        assert [tuple(row) for row in events] == [("like", None), ("dislike", "like")]
        assert db.execute("SELECT created_at FROM video_result_feedback").fetchone()[0] == (
            original["created_at"]
        )


@pytest.mark.parametrize("field", ["user_id", "chat_id", "job_id"])
def test_wrong_identity_cannot_rate_another_result(env, field):
    _, repo = env
    identifier = job(repo)
    arguments = {"job_id": identifier, "chat_id": OWNER, "user_id": OWNER, "reaction": "like"}
    arguments[field] = 999
    with pytest.raises(PermissionError):
        save_result_feedback(repo, **arguments)
    with repo.connection() as db:
        assert db.execute("SELECT count(*) FROM video_result_feedback").fetchone()[0] == 0


@pytest.mark.parametrize("status", ["queued", "running", "failed", "uncertain"])
def test_undelivered_jobs_cannot_receive_usefulness(env, status):
    _, repo = env
    with pytest.raises(ValueError, match="Only delivered"):
        react(repo, job(repo, status=status))


def test_callback_must_target_delivered_message_and_reaction_kind(env):
    _, repo = env
    tldr, pdf = job(repo), job(repo, kind="pdf")
    with pytest.raises(PermissionError, match="attached"):
        react(repo, tldr, message_id=90)
    with pytest.raises(PermissionError, match="attached"):
        react(repo, pdf, message_id=80)
    with pytest.raises(ValueError, match="like or dislike"):
        react(repo, pdf, "interest")
    with repo.connection() as db:
        db.execute("UPDATE video_pdf_jobs SET text_message_id=NULL WHERE id=?", (tldr,))
    with pytest.raises(ValueError, match="Only delivered"):
        react(repo, tldr)
    assert react(repo, pdf, message_id=90)["output_kind"] == "pdf"


def test_registered_delivery_alias_keeps_original_result_reaction_valid(env):
    _, repo = env
    identifier = job(repo)
    with repo.connection() as db:
        db.execute(
            "INSERT INTO video_result_messages"
            "(job_id,chat_id,message_id,output_kind,panel_id,created_at) VALUES(?,?,?,?,NULL,?)",
            (identifier, OWNER, 150, "tldr", datetime.now(UTC).isoformat()),
        )
    assert react(repo, identifier, message_id=150)["changed"]
    with pytest.raises(PermissionError):
        react(repo, identifier, message_id=151)


@pytest.mark.parametrize("view", ["card", "loading"])
def test_primary_summary_panel_requires_its_result_to_be_visible(env, view):
    _, repo = env
    identifier = job(repo)
    with repo.connection() as db:
        stamp = datetime.now(UTC).isoformat()
        db.execute(
            "INSERT INTO video_panels(chat_id,message_id,video_id,original_html,view,"
            "summary_job_id,created_at,updated_at) VALUES(?,?,?,'card',?,?,?,?)",
            (OWNER, 80, VIDEO, view, identifier, stamp, stamp),
        )
    with pytest.raises(PermissionError, match="Open this result"):
        react(repo, identifier, message_id=80)
    with repo.connection() as db:
        db.execute("UPDATE video_panels SET view='summary'")
    assert react(repo, identifier, message_id=80)["changed"]


def test_old_panel_alias_cannot_rate_hidden_result_after_another_panel_rebind(env):
    _, repo = env
    identifier = job(repo)
    other_result = job(repo)
    stamp = datetime.now(UTC).isoformat()
    with repo.connection() as db:
        old_panel = db.execute(
            "INSERT INTO video_panels(chat_id,message_id,video_id,original_html,view,"
            "summary_job_id,created_at,updated_at) VALUES(?,?,?,'card','card',?,?,?)",
            (OWNER, 150, VIDEO, identifier, stamp, stamp),
        ).lastrowid
        new_panel = db.execute(
            "INSERT INTO video_panels(chat_id,message_id,video_id,original_html,view,"
            "summary_job_id,created_at,updated_at) VALUES(?,?,?,'card','summary',?,?,?)",
            (OWNER, 80, VIDEO, identifier, stamp, stamp),
        ).lastrowid
        db.execute("UPDATE video_pdf_jobs SET panel_id=? WHERE id=?", (new_panel, identifier))
        db.execute(
            "INSERT INTO video_result_messages VALUES(?,?,?,?,?,?)",
            (identifier, OWNER, 150, "tldr", old_panel, stamp),
        )
    with pytest.raises(PermissionError, match="Open this result"):
        react(repo, identifier, message_id=150)
    with repo.connection() as db:
        db.execute(
            "UPDATE video_panels SET view='summary',summary_job_id=? WHERE id=?",
            (other_result, old_panel),
        )
    with pytest.raises(PermissionError, match="Open this result"):
        react(repo, identifier, message_id=150)
    assert react(repo, identifier, message_id=80)["changed"]


def test_latest_usefulness_per_video_and_format_is_primary_discovery_is_separate(env):
    settings, repo = env
    first = metadata(repo)
    second = metadata(repo, video="bcdefghijkl")
    with repo.connection() as db:
        for source_item in (first, second):
            recommendation = db.execute(
                "INSERT INTO video_recommendations(source_item_id,chat_id,description_html,"
                "status,created_at) VALUES(?,?,'x','sent',?)",
                (source_item["id"], OWNER, datetime.now(UTC).isoformat()),
            ).lastrowid
            db.execute(
                "INSERT INTO video_feedback_events(recommendation_id,action,created_at) "
                "VALUES(?,'interest',?)",
                (recommendation, datetime.now(UTC).isoformat()),
            )
    react(repo, job(repo), "like")
    react(repo, job(repo), "dislike")  # A newer TLDR supersedes the earlier TLDR example.
    react(repo, job(repo, kind="pdf"), "like")
    profile = personalization_profile(settings, repo, OWNER)
    assert len(profile["video_result_usefulness"]) == 1
    result = profile["video_result_usefulness"][0]
    assert result["video_id"] == VIDEO and result["channel"] == "Engineering Channel"
    assert {r["output_kind"]: r["reaction"] for r in result["results"]} == {
        "tldr": "dislike",
        "pdf": "like",
    }
    assert [r["video_id"] for r in profile["video_feedback"]] == ["bcdefghijkl"]
    assert profile["video_feedback"][0]["signal"].startswith("weaker_")
    policy = profile["video_feedback_policy"]
    assert policy["primary_signal"] == "video_result_usefulness"
    assert policy["negative_result_is_topic_or_channel_ban"] is False
    assert policy["open_or_request_means_satisfaction"] is False


def test_direct_link_uses_only_existing_verified_transcript_metadata(env):
    settings, repo = env
    identifier = job(repo)
    react(repo, identifier)
    folder = settings.data_dir / "video-reports" / VIDEO
    folder.mkdir(parents=True)
    path = folder / "transcript.json"
    path.write_text(
        json.dumps(
            [{"video_id": VIDEO, "title": "Industrial agents", "channel_name": "Direct channel"}]
        )
    )
    result = personalization_profile(settings, repo, OWNER)["video_result_usefulness"][0]
    assert result["channel"] == "Direct channel" and result["metadata"] == "Industrial agents"
    path.write_text(json.dumps([{"video_id": "bcdefghijkl", "title": "Wrong video"}]))
    result = personalization_profile(settings, repo, OWNER)["video_result_usefulness"][0]
    assert "metadata" not in result


def test_owner_feedback_is_not_read_for_other_chat_or_another_job_owner(env):
    settings, repo = env
    react(repo, job(repo))
    assert personalization_profile(settings, repo, 999)["video_result_usefulness"] == []
    foreign = job(repo, owner=999)
    with repo.connection() as db:
        stamp = datetime.now(UTC).isoformat()
        db.execute(
            "INSERT INTO video_result_feedback VALUES(?,?,?,?,?)",
            (foreign, OWNER, "like", stamp, stamp),
        )
    assert len(personalization_profile(settings, repo, OWNER)["video_result_usefulness"]) == 1


def test_onboarded_owner_feedback_works_without_environment_owner_id(env):
    settings, repo = env
    settings.allowed_user_id = None
    repo.set_runtime_setting("allowed_user_id", str(OWNER))
    react(repo, job(repo))
    assert len(personalization_profile(settings, repo, OWNER)["video_result_usefulness"]) == 1
    assert personalization_profile(settings, repo, 999)["video_result_usefulness"] == []


def test_new_result_feedback_does_not_repay_an_uncertain_assessment(env):
    settings, repo = env
    settings.openrouter_api_key = SecretStr("offline")
    video = metadata(repo)
    calls = []

    def uncertain(*args, **kwargs):
        calls.append(1)
        raise TimeoutError("unknown provider result")

    before = VideoRelevanceService(settings, repo, OWNER, assessor=uncertain)
    assert before.assess(video) is None
    react(repo, job(repo))
    after = VideoRelevanceService(settings, repo, OWNER, assessor=uncertain)
    assert before.profile_hash != after.profile_hash
    assert after.assess(video) is None
    assert calls == [1]
    with repo.connection() as db:
        assert db.execute("SELECT count(*) FROM video_relevance_assessments").fetchone()[0] == 1
