import json
from datetime import UTC, datetime, timedelta

import pytest
import yaml

from app.menu_batches.store import BatchStore
from app.ops.schedule import fresh_delivery_card_ids
from app.personalization.daily import (
    DailyContext,
    DailyPolicy,
    feedback_examples,
    feedback_scope,
    load_context,
    prompt_examples,
    record_selection,
    traits,
)
from app.pipeline.ranking import rank_event_candidates
from app.video.relevance import VideoRelevanceService
from tests.test_menu_batches import card
from tests.test_menu_batches import env as menu_env

NOW = datetime(2026, 10, 21, 13, tzinfo=UTC)
POLICY = DailyPolicy(
    priorities={
        "ai_coding": 1.4,
        "agent_systems": 1.5,
        "domain_workflows": 1.4,
        "open_source": 1,
        "world_affairs": 0.3,
    },
    genre_penalties={"corporate_roundup": 1.2, "access_or_ui": 0.8, "niche_security": 1.3},
)


@pytest.fixture
def daily_env(tmp_path):
    settings, repo, _ = menu_env.__wrapped__(tmp_path)
    settings.daily_personalization_enabled = True
    settings.daily_selection_file = tmp_path / "daily.yaml"
    settings.daily_selection_file.write_text(yaml.safe_dump(POLICY.model_dump()))
    return settings, repo


def event(n, headline, score=8, **extra):
    return {"id": n, "headline": headline, "score": score, **extra}


def context(examples=None, now=NOW):
    return DailyContext(POLICY, examples or [], now)


def vote(repo, *, n=1, reaction="like", comment=None, when=NOW, title="Claude Code workflow"):
    card_id, event_id = card(repo)
    with repo.connection() as db:
        db.execute("UPDATE events SET headline=? WHERE id=?", (title, event_id))
        db.execute("UPDATE cards SET rendered_html=? WHERE id=?", (title, card_id))
    delivery = repo.ensure_delivery(card_id, 123)
    repo.mark_delivery_sent(delivery, 1000 + n)
    repo.finalize_feedback(delivery, action="reaction", reaction=reaction, comment=comment)
    with repo.connection() as db:
        db.execute(
            "UPDATE feedback_events SET created_at=? WHERE delivery_id=?",
            (when.isoformat(), delivery),
        )
    return delivery, event_id


@pytest.mark.parametrize(
    ("comment", "scope"),
    [
        ("Сама новость хорошая, но непонятно в чем поинт, просто вода", "presentation_or_novelty"),
        ("Слишком много однотипных новостей от компании", "presentation_or_novelty"),
        ("https://www.youtube.com/@example", "source_request"),
        ("Пидары", "ambiguous"),
        ("Очень полезный продукт, такого побольше", "result_usefulness"),
    ],
)
def test_comments_have_separate_scopes(comment, scope):
    assert feedback_scope(comment) == scope


def test_profile_tags_and_page_footer_are_not_interest_evidence():
    row = event(
        1,
        "Новые тарифы Claude",
        topics_json='["agents","semiconductors"]',
        rendered_html="План стоит $20 в месяц.",
        original_text="AI coding harness GitHub",
    )
    assert not traits(row)


def test_substantive_workflow_beats_high_source_priority_access_changelog():
    workflow = event(1, "Agentic workflow: Claude Code harness проверяет тесты", 8)
    access = event(
        2,
        "Claude Code теперь доступен всем пользователям Pro и Max",
        14,
        score_features={"scoring_version": "v7a_1"},
    )
    assert rank_event_candidates([workflow, access], limit=1)[0]["id"] == 2
    ranked = rank_event_candidates([workflow, access], personalization=context(), limit=1)
    assert ranked[0]["id"] == 1
    assert workflow["score"] == 8 and "score_features" not in workflow


def test_domain_process_beats_pr_and_major_world_news_beats_weak_ai():
    business = event(1, "ИИ автоматизирует сверку счетов и бухгалтерию для клиентов", 8.2)
    pr = event(
        2,
        "Company Partners Showcase Regional Innovation Ecosystem",
        12.7,
        score_features={"scoring_version": "v7a_1"},
    )
    assert rank_event_candidates([pr, business], personalization=context(), limit=1)[0]["id"] == 1
    world = event(3, "Перемирие завершает войну, санкции отменены", 9)
    weak = event(4, "Claude Code исправляет опечатки в UI", 3.1)
    assert rank_event_candidates([weak, world], personalization=context(), limit=1)[0]["id"] == 3
    assert (
        rank_event_candidates(
            [event(5, "Claude Code agentic workflow", 2)], personalization=context()
        )
        == []
    )


def test_different_strong_ai_events_can_share_a_company_but_exact_duplicate_cannot():
    items = [
        event(
            i,
            f"Claude Code harness: новый рабочий процесс {i}",
            8.5,
            entities_json='{"organizations":["Lab"]}',
            fingerprint=f"story:{i}",
        )
        for i in range(1, 4)
    ]
    items.append(event(4, "Company roadmap", 6))
    items.append({**items[0], "id": 5})
    result = rank_event_candidates(items, personalization=context(), limit=5)
    assert len(result) == 4
    assert {r["id"] for r in result[:3]} == {1, 2, 3}


def test_all_saturation_axes_decay_and_feedback_is_bounded():
    candidate = event(1, "Claude Code harness", entities_json='{"organizations":["Lab"]}')
    recent = [{"entities_json": candidate["entities_json"], "delivered_at": NOW.isoformat()}] * 5
    old = [{**r, "delivered_at": (NOW - timedelta(days=8)).isoformat()} for r in recent]
    fresh_score = rank_event_candidates(
        [candidate], personalization=context(), recent_events=recent, now=NOW
    )[0]["effective_score"]
    old_score = rank_event_candidates(
        [candidate], personalization=context(), recent_events=old, now=NOW
    )[0]["effective_score"]
    assert old_score > fresh_score


def test_latest_vote_before_cutoff_is_used_without_future_leak_or_stale_comment(daily_env):
    _, repo = daily_env
    delivery, _ = vote(repo, reaction="superlike", comment="Полезно", when=NOW - timedelta(days=2))
    repo.finalize_feedback(delivery, action="reaction", reaction="dislike", comment=None)
    with repo.connection() as db:
        db.execute(
            "UPDATE feedback_events SET created_at=? "
            "WHERE id=(SELECT MAX(id) FROM feedback_events)",
            ((NOW + timedelta(days=1)).isoformat(),),
        )
    assert feedback_examples(repo, 123, now=NOW)[0]["reaction"] == "superlike"
    later = feedback_examples(repo, 123, now=NOW + timedelta(days=2))
    assert len(later) == 1 and later[0]["reaction"] == "dislike" and not later[0]["comment"]
    assert feedback_examples(repo, 999, now=NOW) == []


def test_presentation_complaints_and_source_links_do_not_create_topic_dislikes(daily_env):
    _, repo = daily_env
    for i in range(3):
        vote(repo, n=i, reaction="superdislike", comment="Неясный поинт, сама новость хорошая")
    vote(repo, n=4, reaction="superlike", comment="https://youtube.com/@a")
    examples = feedback_examples(repo, 123, now=NOW)
    assert all(e["topic_weight"] == 0 for e in examples)
    c = context(examples)
    assert not c.learned
    assert not any(e["scope"] == "source_request" for e in prompt_examples(c))


def test_another_result_keeps_its_presentation_lesson_without_duplicate_topic_votes(daily_env):
    _, repo = daily_env
    _, eid = vote(repo, reaction="dislike", comment="Неясный поинт, сама тема хорошая")
    second, _ = card(repo)
    with repo.connection() as db:
        db.execute("UPDATE cards SET event_id=?,card_version=2 WHERE id=?", (eid, second))
    delivery = repo.ensure_delivery(second, 123)
    repo.mark_delivery_sent(delivery, 2000)
    repo.finalize_feedback(delivery, action="reaction", reaction="like", comment=None)
    with repo.connection() as db:
        db.execute(
            "UPDATE feedback_events SET created_at=? WHERE delivery_id=?",
            ((NOW + timedelta(seconds=1)).isoformat(), delivery),
        )
    examples = feedback_examples(repo, 123, now=NOW + timedelta(seconds=2))
    assert len(examples) == 2
    assert sum(bool(e["topic_weight"]) for e in examples) == 1
    assert any(e["scope"] == "presentation_or_novelty" and e["topic_weight"] == 0 for e in examples)


def test_learning_requires_consistent_distinct_events_and_old_feedback_decays(daily_env):
    _, repo = daily_env
    delivery, _ = vote(repo)
    for _ in range(4):
        repo.finalize_feedback(delivery, action="reaction", reaction="like", comment=None)
    assert len(feedback_examples(repo, 123, now=NOW)) == 1
    assert not context(feedback_examples(repo, 123, now=NOW)).learned
    vote(repo, n=2)
    vote(repo, n=3)
    examples = feedback_examples(repo, 123, now=NOW)
    learned = context(examples).learned["ai_coding"]
    assert 0 < learned["weight"] <= 0.6
    assert len(set(learned["event_ids"])) == 3
    old = context(examples, now=NOW + timedelta(days=200)).learned["ai_coding"]
    assert old["weight"] < learned["weight"]
    vote(repo, n=4, reaction="dislike")
    vote(repo, n=5, reaction="dislike")
    assert "ai_coding" not in context(feedback_examples(repo, 123, now=NOW)).learned


def test_usage_is_a_small_signal_and_cannot_invent_a_new_preference():
    reads = [{"key": f"video:{i}", "traits": ["ai_coding"]} for i in range(4)]
    c = DailyContext(POLICY, [], NOW, reads)
    output = c.adjustment(event(1, "Claude Code"))
    assert output["engagement"] == 0.15
    neutral = DailyContext(DailyPolicy(), [], NOW, reads)
    assert neutral.adjustment(event(1, "Claude Code"))["adjustment"] == 0


def test_delivery_and_manual_menu_share_policy_without_model_calls_or_video_cache_churn(daily_env):
    settings, repo = daily_env
    before_video = VideoRelevanceService(
        settings.model_copy(update={"daily_personalization_enabled": False}), repo, 123
    ).profile_hash
    lower, _ = card(repo)
    higher, event_id = card(repo)
    with repo.connection() as db:
        db.execute("UPDATE events SET headline='Claude Code harness' WHERE id=?", (event_id,))
    now = datetime.now(UTC)
    ids = fresh_delivery_card_ids(repo, owner_id=123, now=now, settings=settings, limit=2)
    assert ids == [higher, lower]
    assert (
        BatchStore(settings, repo).candidates(
            123, "news", (now - timedelta(days=1)).isoformat(), now.isoformat()
        )[0]["entity_id"]
        == higher
    )
    assert VideoRelevanceService(settings, repo, 123).profile_hash == before_video
    assert repo.current_month_cost() == 0
    assert repo.get_runtime_setting("biography_enabled") is None
    assert (
        load_context(settings.model_copy(update={"daily_personalization_enabled": False}), repo)
        is None
    )


def test_trace_is_durable_idempotent_and_does_not_rewrite_scores(daily_env):
    _, repo = daily_env
    _, eid = card(repo)
    pool = [event(eid, "Claude Code harness", 7)]
    c = context()
    ranked = rank_event_candidates(pool, personalization=c)
    for _ in range(2):
        record_selection(repo, c, owner=123, stage="test", candidates=pool, selected=ranked)
    with repo.connection() as db:
        rows = db.execute("SELECT * FROM daily_selection_runs").fetchall()
        assert len(rows) == 1
        payload = json.loads(rows[0]["payload_json"])
        assert payload["rows"][0]["reason"]["version"] == "daily_editorial_v2"
        assert db.execute("SELECT score FROM events WHERE id=?", (eid,)).fetchone()[0] == 10
