from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime, timedelta

import pytest

from app.config import SourceConfig, SourceRegistry
from app.db.repository import Repository
from app.details.paging import telegram_html_visible_text, telegram_html_visible_units
from app.details.research import reader_target_units
from app.research.detail_policy import detail_policy_versions

NOW = datetime(2026, 10, 13, 9, tzinfo=UTC)
STAMP = NOW.isoformat()
EXPIRES = (NOW + timedelta(days=7)).isoformat()


def configure_source(repo):
    repo.sync_sources(
        SourceRegistry(
            version=1,
            sources=[
                SourceConfig(
                    id="fixture",
                    name="Fixture",
                    kind="rss",
                    locator="https://example.org/feed",
                    role="radar",
                    region="global",
                    language="en",
                )
            ],
        )
    )


def sent_delivery(repo, message_id=9001):
    delivery = repo.create_smoke_delivery(123, "https://example.org/story")
    repo.mark_delivery_sent(delivery["delivery_id"], message_id)
    return delivery


def frozen_context(delivery):
    context = {
        "baseline_card_id": delivery["card_id"],
        "baseline_html": delivery["rendered_html"],
        "baseline_visible_chars": 200,
        "baseline_document_ids": [1, 2],
        "baseline_source_urls": ["https://example.org/a", "https://independent.test/b"],
        "retry_rejected": False,
    }
    context["context_fingerprint"] = hashlib.sha256(
        json.dumps(context, sort_keys=True).encode()
    ).hexdigest()
    return context


@pytest.fixture
def env(tmp_path):
    repo = Repository(tmp_path / "reader.sqlite3")
    repo.migrate()
    configure_source(repo)
    delivery = sent_delivery(repo)
    return repo, delivery, frozen_context(delivery)


def enqueue(env, depth="expanded", *, context=None, now=STAMP):
    repo, delivery, original = env
    return repo.enqueue_detail_request(
        delivery["delivery_id"],
        123,
        depth=depth,
        reader_context=original if context is None else context,
        now=now,
    )


def article_payload(request, *, depth=None, fingerprint=None, key=None, expires=EXPIRES):
    depth = depth or request["depth"]
    schema, prompt = detail_policy_versions(depth)
    context = dict(request["reader_context"])
    fingerprint = fingerprint or context["context_fingerprint"]
    return {
        "depth": depth,
        "reader_context": context,
        "context_fingerprint": fingerprint,
        "generation_key": key or f"reader:{request['event_id']}:{depth}:{fingerprint}",
        "schema_version": schema,
        "prompt_version": prompt,
        "model": "offline/model",
        "provider_response_metadata": {"offline": True},
        "fact_snapshot": {"facts": [{"text": "A verified fixture fact."}]},
        "document_ids": [1, 2, 3, 4, 5],
        "structured_output": {"headline": "Grounded fixture reader"},
        "rendered_html": "<b>Grounded fixture reader</b>\n\nComplete fixture prose.",
        "source_urls": ["https://example.org/a", "https://independent.test/b"],
        "validation_status": "valid",
        "validation_errors": [],
        "generation_cost_usd": 0.01,
        "expires_at": expires,
        "now": STAMP,
    }


def make_ready(env, depth="expanded"):
    repo, _, _ = env
    request = enqueue(env, depth)
    claimed = repo.claim_next_detail_request(now=STAMP)
    assert claimed["id"] == request["id"]
    article_id = repo.store_detail_article_and_complete(request["id"], **article_payload(request))
    return repo.get_detail_request(request["id"]), article_id


@pytest.mark.parametrize("status", ["ready", "sent"])
def test_article_freezes_actual_full_expanded_text_even_if_archived(env, status):
    repo, _, original = env
    more, artifact_id = make_ready(env)
    rendered = '<b>Полный разбор 😀</b>\n\n<a href="https://example.org/a">Содержание</a>. ' * 20
    with repo.connection() as db:
        db.execute(
            "UPDATE detailed_articles SET rendered_html=?,expires_at=? WHERE id=?",
            (rendered, STAMP, artifact_id),
        )
        db.execute("UPDATE detail_requests SET status=? WHERE id=?", (status, more["id"]))
    article = enqueue(env, "article")
    ctx = article["reader_context"]
    assert ctx["context_fingerprint"] == original["context_fingerprint"]
    assert ctx["expanded_reference_article_id"] == artifact_id
    assert ctx["expanded_reference_text"] == telegram_html_visible_text(rendered)
    assert ctx["expanded_reference_units"] == telegram_html_visible_units(rendered)
    assert reader_target_units("article", ctx) == 2 * telegram_html_visible_units(rendered)
    with repo.connection() as db:
        db.execute(
            "UPDATE detailed_articles SET rendered_html='Changed later' WHERE id=?", (artifact_id,)
        )
    assert enqueue(env, "article")["reader_context"] == ctx


@pytest.mark.parametrize("damage", ["queued", "failed", "fingerprint", "event", "invalid"])
def test_article_does_not_use_unfinished_foreign_or_invalid_more(env, damage):
    repo, _, _ = env
    more, artifact_id = make_ready(env)
    with repo.connection() as db:
        if damage in {"queued", "failed"}:
            db.execute("UPDATE detail_requests SET status=? WHERE id=?", (damage, more["id"]))
        elif damage == "fingerprint":
            db.execute(
                "UPDATE detailed_articles SET context_fingerprint=? WHERE id=?",
                ("b" * 64, artifact_id),
            )
        elif damage == "event":
            other = repo.create_smoke_delivery(456, "https://example.org/other")
            db.execute(
                "UPDATE detailed_articles SET event_id=? WHERE id=?",
                (other["event_id"], artifact_id),
            )
        else:
            db.execute(
                "UPDATE detailed_articles SET validation_status='invalid' WHERE id=?",
                (artifact_id,),
            )
    assert "expanded_reference_text" not in enqueue(env, "article")["reader_context"]


def test_article_first_reference_stays_frozen_after_more_finishes(env):
    repo, _, _ = env
    more = enqueue(env)
    first = enqueue(env, "article")
    assert "expanded_reference_text" not in first["reader_context"]
    assert repo.claim_next_detail_request(now=STAMP)["id"] == more["id"]
    repo.store_detail_article_and_complete(more["id"], **article_payload(more))
    assert enqueue(env, "article")["reader_context"] == first["reader_context"]


def test_article_after_more_cannot_reuse_article_first_cache_or_attach_it(env):
    repo, delivery, original = env
    first, first_artifact = make_ready(env, "article")
    with repo.connection() as db:
        other_id = db.execute(
            "INSERT INTO deliveries(card_id,chat_id,delivery_status,telegram_message_id) "
            "VALUES(?,456,'sent',9002)",
            (delivery["card_id"],),
        ).lastrowid
    more = repo.enqueue_detail_request(
        other_id, 456, depth="expanded", reader_context=original, now=STAMP
    )
    assert repo.claim_next_detail_request(now=STAMP)["id"] == more["id"]
    repo.store_detail_article_and_complete(more["id"], **article_payload(more))
    article = repo.enqueue_detail_request(
        other_id, 456, depth="article", reader_context=original, now=STAMP
    )
    assert article["status"] == "queued" and not article["cache_hit"]
    assert article["reader_context"]["context_fingerprint"] == original["context_fingerprint"]
    assert (
        article["reader_context"]["article_reference_fingerprint"]
        != first["reader_context"]["article_reference_fingerprint"]
    )
    assert (
        repo.cached_detailed_article_for_event(
            delivery["event_id"],
            depth="article",
            context_fingerprint=original["context_fingerprint"],
            reader_context=article["reader_context"],
            now=STAMP,
        )
        is None
    )
    assert repo.claim_next_detail_request(now=STAMP)["id"] == article["id"]
    assert not repo.complete_detail_request_from_cache(article["id"], first_artifact, now=STAMP)
    with pytest.raises(ValueError, match="foreign reading reference"):
        repo.store_detail_article_and_complete(
            article["id"], **article_payload(first, key="foreign-reading-ref")
        )


def test_policy_delivery_preserves_expanded_v2_and_requires_article_v3(env, monkeypatch):
    monkeypatch.setattr("app.db.repository.utc_now", lambda: STAMP)
    repo, _, _ = env
    _, expanded_id = make_ready(env)
    _, article_id = make_ready(env, "article")
    assert {row["article_id"] for row in repo.ready_detail_requests()} == {expanded_id, article_id}
    with repo.connection() as db:
        db.execute(
            "UPDATE detailed_articles SET prompt_version=? WHERE id=?",
            (detail_policy_versions("expanded")[1], article_id),
        )
    assert {row["article_id"] for row in repo.ready_detail_requests()} == {expanded_id}


def add_panel(env, request, *, view="loading", active=None):
    repo, delivery, context = env
    with repo.connection() as db:
        db.execute(
            "INSERT INTO news_panels(delivery_id,chat_id,message_id,baseline_html,"
            "reader_context_json,view,active_request_id) VALUES(?,123,9001,?,?,?,?)",
            (
                delivery["delivery_id"],
                delivery["rendered_html"],
                json.dumps(context),
                view,
                request["id"] if active is None else active,
            ),
        )


def test_migration_preserves_legacy_ids_artifacts_feedback_and_autoincrement(tmp_path):
    repo = Repository(tmp_path / "legacy.sqlite3")
    current_migrations = repo.migrations_path
    legacy_migrations = tmp_path / "legacy-migrations"
    legacy_migrations.mkdir()
    for path in current_migrations.glob("*.sql"):
        if path.stem < "0023_news_reader":
            shutil.copy2(path, legacy_migrations / path.name)
    repo.migrations_path = legacy_migrations
    repo.migrate()
    configure_source(repo)
    delivery = sent_delivery(repo)
    legacy_schema, legacy_prompt = detail_policy_versions("legacy")
    with repo.connection() as db:
        db.execute(
            "INSERT INTO detailed_articles(id,event_id,generation_key,schema_version,"
            "prompt_version,model,fact_snapshot_json,structured_output_json,rendered_html,"
            "validation_status,expires_at) VALUES(5,?,?,?,?,?,'{}','{}','Legacy article',"
            "'valid',?)",
            (
                delivery["event_id"],
                "legacy-fixed-key",
                legacy_schema,
                legacy_prompt,
                "offline/model",
                EXPIRES,
            ),
        )
        db.execute(
            "INSERT INTO detail_requests(id,delivery_id,card_id,event_id,chat_id,user_id,"
            "generation_key,status,stage,attempt_count,article_id,placeholder_message_id,"
            "requested_at,completed_at,updated_at) "
            "VALUES(11,?,?,?,123,123,'legacy-fixed-key','sent','complete',2,5,974,?,?,?)",
            (
                delivery["delivery_id"],
                delivery["card_id"],
                delivery["event_id"],
                STAMP,
                STAMP,
                STAMP,
            ),
        )
        db.execute(
            "INSERT INTO feedback_events(id,delivery_id,event_id,card_id,action,metadata_json) "
            "VALUES(77,?,?,?,'detail_useful','{\"article_id\":5}')",
            (delivery["delivery_id"], delivery["event_id"], delivery["card_id"]),
        )
        old_request = dict(db.execute("SELECT * FROM detail_requests").fetchone())
        old_article = dict(db.execute("SELECT * FROM detailed_articles").fetchone())
        old_feedback = dict(db.execute("SELECT * FROM feedback_events").fetchone())
    repo.migrations_path = current_migrations
    assert "0023_news_reader" in repo.migrate()
    with repo.connection() as db:
        request = dict(db.execute("SELECT * FROM detail_requests WHERE id=11").fetchone())
        article = dict(db.execute("SELECT * FROM detailed_articles WHERE id=5").fetchone())
        feedback = dict(db.execute("SELECT * FROM feedback_events WHERE id=77").fetchone())
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
    assert {key: request[key] for key in old_request} == old_request
    assert {key: article[key] for key in old_article} == old_article
    assert feedback == old_feedback
    assert request["depth"] == article["depth"] == "legacy"
    assert request["reader_context_json"] == article["reader_context_json"] == "{}"
    new = repo.enqueue_detail_request(
        delivery["delivery_id"],
        123,
        depth="expanded",
        reader_context=frozen_context(delivery),
        now=STAMP,
    )
    assert new["id"] > 11 and repo.get_detail_request(11)["article_id"] == 5


def test_each_depth_has_its_own_idempotent_request_even_when_new_tiers_exist_first(env):
    repo, delivery, _ = env
    expanded = enqueue(env)
    article = enqueue(env, "article")
    legacy = repo.enqueue_detail_request(delivery["delivery_id"], 123, now=STAMP)
    assert len({expanded["id"], article["id"], legacy["id"]}) == 3
    assert legacy["depth"] == "legacy"
    assert enqueue(env)["id"] == expanded["id"] and not enqueue(env)["created"]
    assert enqueue(env, "article")["id"] == article["id"]
    assert repo.detail_request_for_delivery(delivery["delivery_id"])["id"] == legacy["id"]
    assert (
        repo.detail_request_for_delivery(delivery["delivery_id"], depth="expanded")["id"]
        == (expanded["id"])
    )


def test_enqueue_rejects_wrong_frozen_baseline_and_owner(env):
    repo, delivery, context = env
    enqueue(env)
    changed = {**context, "context_fingerprint": "f" * 64}
    with pytest.raises(ValueError, match="baseline changed"):
        enqueue(env, context=changed)
    with pytest.raises(PermissionError, match="Foreign"):
        enqueue(env, "article", context={**context, "baseline_card_id": delivery["card_id"] + 1})
    with pytest.raises(PermissionError, match="Foreign"):
        repo.enqueue_detail_request(
            delivery["delivery_id"], 456, depth="article", reader_context=context, now=STAMP
        )


def test_cache_is_partitioned_by_depth_fingerprint_policy_expiry_and_validation(env):
    repo, delivery, context = env
    request, article_id = make_ready(env)
    kwargs = {
        "depth": "expanded",
        "context_fingerprint": context["context_fingerprint"],
        "now": STAMP,
    }
    assert (
        repo.cached_detailed_article_for_event(delivery["event_id"], **kwargs)["id"] == article_id
    )
    assert repo.cached_detailed_article(request["generation_key"], **kwargs)["id"] == article_id
    assert (
        repo.cached_detailed_article_for_event(
            delivery["event_id"],
            depth="article",
            now=STAMP,
            context_fingerprint=context["context_fingerprint"],
        )
        is None
    )
    assert (
        repo.cached_detailed_article_for_event(
            delivery["event_id"], **{**kwargs, "context_fingerprint": "a" * 64}
        )
        is None
    )
    assert repo.cached_detailed_article_for_event(delivery["event_id"], now=STAMP) is None
    with repo.connection() as db:
        db.execute(
            "UPDATE detailed_articles SET prompt_version='old-policy' WHERE id=?", (article_id,)
        )
    assert repo.cached_detailed_article_for_event(delivery["event_id"], **kwargs) is None
    with repo.connection() as db:
        db.execute(
            "UPDATE detailed_articles SET prompt_version=?,expires_at=? WHERE id=?",
            (detail_policy_versions("expanded")[1], STAMP, article_id),
        )
    assert repo.cached_detailed_article_for_event(delivery["event_id"], **kwargs) is None
    with repo.connection() as db:
        db.execute(
            "UPDATE detailed_articles SET expires_at=?,validation_status='invalid' WHERE id=?",
            (EXPIRES, article_id),
        )
    assert repo.cached_detailed_article_for_event(delivery["event_id"], **kwargs) is None


@pytest.mark.parametrize("change", ["expiry", "policy"])
@pytest.mark.parametrize("status", ["ready", "sent"])
def test_existing_paid_request_reopens_archive_without_queue_or_policy_restamp(env, change, status):
    repo, _, _ = env
    request, article = make_ready(env)
    if status == "sent":
        repo.mark_detail_request_sent(request["id"], now=STAMP)
    with repo.connection() as db:
        if change == "expiry":
            db.execute("UPDATE detailed_articles SET expires_at=? WHERE id=?", (STAMP, article))
        else:
            db.execute(
                "UPDATE detailed_articles SET prompt_version='old-policy' WHERE id=?", (article,)
            )
        original = dict(
            db.execute("SELECT * FROM detailed_articles WHERE id=?", (article,)).fetchone()
        )
    reopened = enqueue(env)
    assert reopened["id"] == request["id"] and not reopened["created"]
    assert reopened["status"] == status and reopened["article_id"] == article
    with repo.connection() as db:
        assert (
            dict(db.execute("SELECT * FROM detailed_articles WHERE id=?", (article,)).fetchone())
            == original
        )
        assert db.execute("SELECT count(*) FROM detail_requests").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM usage_ledger").fetchone()[0] == 0


@pytest.mark.parametrize("damage", ["depth", "fingerprint", "event", "invalid"])
def test_existing_archival_pointer_cannot_reopen_foreign_or_invalid_article(env, damage):
    repo, _, _ = env
    request, article = make_ready(env)
    with repo.connection() as db:
        if damage == "depth":
            db.execute("UPDATE detailed_articles SET depth='article' WHERE id=?", (article,))
        elif damage == "fingerprint":
            db.execute(
                "UPDATE detailed_articles SET context_fingerprint=? WHERE id=?", ("f" * 64, article)
            )
        elif damage == "invalid":
            db.execute(
                "UPDATE detailed_articles SET validation_status='invalid' WHERE id=?", (article,)
            )
    if damage == "event":
        other = sent_delivery(repo, message_id=9002)
        with repo.connection() as db:
            db.execute(
                "UPDATE detailed_articles SET event_id=? WHERE id=?", (other["event_id"], article)
            )
    try:
        reopened = enqueue(env)
    except (ValueError, PermissionError):
        return
    assert reopened["status"] not in {"ready", "sent"}
    assert repo.get_detail_request(request["id"])["status"] != "queued"


def test_event_cache_lookup_for_new_depth_requires_a_frozen_baseline(env):
    repo, delivery, _ = env
    make_ready(env)
    assert (
        repo.cached_detailed_article_for_event(delivery["event_id"], depth="expanded", now=STAMP)
        is None
    )


@pytest.mark.parametrize("scope", ["queued", "depth", "fingerprint", "policy"])
def test_store_requires_running_request_and_matching_reader_scope(env, scope):
    repo, _, _ = env
    request = enqueue(env)
    data = article_payload(request)
    if scope != "queued":
        assert repo.claim_next_detail_request(now=STAMP)["id"] == request["id"]
    if scope == "depth":
        data.update(article_payload(request, depth="article"))
    elif scope == "fingerprint":
        data["context_fingerprint"] = "f" * 64
    elif scope == "policy":
        data["prompt_version"] = "old-policy"
    with pytest.raises(ValueError):
        repo.store_detail_article_and_complete(request["id"], **data)
    with repo.connection() as db:
        assert db.execute("SELECT count(*) FROM detailed_articles").fetchone()[0] == 0


def test_store_rejects_collision_with_outdated_artifact_even_if_incoming_policy_is_current(env):
    repo, _, _ = env
    request, article = make_ready(env)
    repo.mark_detail_request_sent(request["id"], now=STAMP)
    with repo.connection() as db:
        db.execute(
            "UPDATE detailed_articles SET prompt_version='old-policy' WHERE id=?", (article,)
        )
        db.execute(
            "UPDATE detail_requests SET status='running',article_id=NULL WHERE id=?",
            (request["id"],),
        )
    with pytest.raises(ValueError):
        repo.store_detail_article_and_complete(request["id"], **article_payload(request))


def test_cache_attach_requires_running_state_and_same_depth_fingerprint(env):
    repo, _, _ = env
    expanded, article_id = make_ready(env)
    long_request = enqueue(env, "article")
    assert repo.claim_next_detail_request(now=STAMP)["id"] == long_request["id"]
    assert not repo.complete_detail_request_from_cache(long_request["id"], article_id, now=STAMP)
    with repo.connection() as db:
        db.execute(
            "UPDATE detail_requests SET status='running',article_id=NULL WHERE id=?",
            (expanded["id"],),
        )
    assert repo.complete_detail_request_from_cache(expanded["id"], article_id, now=STAMP)
    assert repo.get_detail_request(expanded["id"])["cache_hit"]
    assert not repo.complete_detail_request_from_cache(expanded["id"], article_id, now=STAMP)


@pytest.mark.parametrize("damage", ["fingerprint", "expiry", "policy", "invalid"])
def test_cache_attach_rejects_unusable_article_without_changing_running_request(env, damage):
    repo, _, _ = env
    request, article = make_ready(env)
    with repo.connection() as db:
        db.execute(
            "UPDATE detail_requests SET status='running',article_id=NULL WHERE id=?",
            (request["id"],),
        )
        if damage == "fingerprint":
            db.execute(
                "UPDATE detailed_articles SET context_fingerprint=? WHERE id=?", ("f" * 64, article)
            )
        elif damage == "expiry":
            db.execute("UPDATE detailed_articles SET expires_at=? WHERE id=?", (STAMP, article))
        elif damage == "policy":
            db.execute(
                "UPDATE detailed_articles SET prompt_version='old-policy' WHERE id=?", (article,)
            )
        else:
            db.execute(
                "UPDATE detailed_articles SET validation_status='invalid' WHERE id=?", (article,)
            )
    assert not repo.complete_detail_request_from_cache(request["id"], article, now=STAMP)
    unchanged = repo.get_detail_request(request["id"])
    assert unchanged["status"] == "running" and unchanged["article_id"] is None


def test_active_ready_delivery_claim_checks_artifact_baseline_fingerprint(env):
    repo, _, _ = env
    request, article = make_ready(env)
    add_panel(env, request)
    with repo.connection() as db:
        db.execute(
            "UPDATE detailed_articles SET context_fingerprint=? WHERE id=?", ("f" * 64, article)
        )
    assert repo.claim_ready_detail_request_for_delivery(now=STAMP) is None


@pytest.mark.parametrize(
    "view,active",
    [
        (None, None),
        ("base", True),
        ("expanded", True),
        ("article", True),
        ("error", True),
        ("loading", False),
    ],
)
def test_hidden_ready_artifact_is_not_claimed_for_delivery(env, view, active):
    repo, _, _ = env
    request, _ = make_ready(env)
    if view is not None:
        add_panel(env, request, view=view)
        if active is False:
            with repo.connection() as db:
                db.execute("UPDATE news_panels SET active_request_id=NULL")
    assert repo.claim_ready_detail_request_for_delivery(now=STAMP) is None
    assert repo.get_detail_request(request["id"])["status"] == "ready"


def test_visible_active_loading_panel_claims_result_in_original_message(env):
    repo, _, _ = env
    request, article = make_ready(env)
    add_panel(env, request)
    result = repo.claim_ready_detail_request_for_delivery(now=STAMP)
    assert result["id"] == request["id"] and result["article_id"] == article
    assert result["placeholder_message_id"] == result["parent_message_id"] == 9001
    assert repo.claim_ready_detail_request_for_delivery(now=STAMP) is None


def test_explicit_retry_refreshes_authorization_and_lease_recovery_preserves_it(env):
    repo, _, context = env
    request = enqueue(env)
    repo.claim_next_detail_request(now=STAMP)
    assert repo.fail_detail_request(request["id"], error="Provider HTTP 402", now=STAMP)
    retried = repo.retry_detail_request(request["id"], now=STAMP)
    first = retried["reader_context"]["retry_authorization"]
    assert retried["reader_context"]["retry_rejected"] is True
    assert (
        first and retried["reader_context"]["context_fingerprint"] == context["context_fingerprint"]
    )
    assert repo.claim_next_detail_request(lease_minutes=1, now=STAMP)["id"] == request["id"]
    later = (NOW + timedelta(minutes=2)).isoformat()
    assert repo.recover_stale_detail_requests(now=later) == 1
    recovered = repo.get_detail_request(request["id"])
    assert recovered["reader_context"]["retry_rejected"] is True
    assert recovered["reader_context"]["retry_authorization"] == first
    repo.claim_next_detail_request(now=later)
    repo.fail_detail_request(request["id"], error="Provider HTTP 402", now=later)
    second = repo.retry_detail_request(request["id"], now=later)["reader_context"][
        "retry_authorization"
    ]
    assert second != first
