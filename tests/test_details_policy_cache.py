from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.details.models import DetailFact
from app.details.research import detail_generation_key
from app.details.worker import DetailWorker
from app.research.detail_policy import DETAIL_PROMPT_VERSION
from app.research.facts import EvidenceQuote, fact_safety_errors
from app.research.policy import FACT_EXTRACTION_PROMPT_VERSION, FACT_VALIDATION_POLICY_VERSION
from app.telegram_bot.bot import details_acknowledgement
from tests.test_details_repository import valid_article_payload
from tests.test_details_research import fact_documents, fact_sheet
from tests.test_details_worker import artifact, setup_worker


def stored_article(tmp_path: Path):
    settings, repository, delivery = setup_worker(tmp_path)
    result = DetailWorker(
        settings, repository, builder=lambda _s, _r, event_id, **_kw: artifact(event_id)
    ).run_once()
    assert result.status == "ready"
    return settings, repository, delivery, result.article_id


def retire_policy(repository, article_id: int) -> None:
    # Synthetic legacy artifact: emulate an upgrade after an older writer ran.
    with repository.connection() as connection:
        connection.execute(
            "UPDATE detailed_articles SET prompt_version='detail_article_ru_v2' WHERE id=?",
            (article_id,),
        )


def second_delivery(repository, delivery) -> int:
    with repository.connection() as connection:
        return int(
            connection.execute(
                """INSERT INTO deliveries(card_id,chat_id,telegram_message_id,delivery_status)
            VALUES (?,456,9002,'sent')""",
                (delivery["card_id"],),
            ).lastrowid
        )


def test_detail_identity_depends_on_both_fact_policies() -> None:
    assert FACT_EXTRACTION_PROMPT_VERSION in DETAIL_PROMPT_VERSION
    assert FACT_VALIDATION_POLICY_VERSION in DETAIL_PROMPT_VERSION
    assert detail_generation_key(1, []) != detail_generation_key(
        1, [], prompt_version="detail_article_ru_v2"
    )


def test_legacy_cache_is_rejected_by_event_and_exact_key(tmp_path: Path) -> None:
    _, repository, delivery, article_id = stored_article(tmp_path)
    article = repository.get_detailed_article(article_id)
    assert repository.cached_detailed_article(article["generation_key"])
    assert repository.cached_detailed_article_for_event(delivery["event_id"])
    retire_policy(repository, article_id)

    assert repository.cached_detailed_article(article["generation_key"]) is None
    assert repository.cached_detailed_article_for_event(delivery["event_id"]) is None
    assert repository.get_detailed_article(article_id)["rendered_html"] == article["rendered_html"]


@pytest.mark.parametrize("explicit_key", [False, True])
def test_enqueue_never_attaches_stale_policy_cache(tmp_path: Path, explicit_key: bool) -> None:
    _, repository, delivery, article_id = stored_article(tmp_path)
    article = repository.get_detailed_article(article_id)
    retire_policy(repository, article_id)
    request = repository.enqueue_detail_request(
        second_delivery(repository, delivery),
        42,
        generation_key=article["generation_key"] if explicit_key else None,
    )
    assert request["status"] == "queued"
    assert request["article_id"] is None
    assert request["cache_hit"] is False
    claimed = repository.claim_next_detail_request()
    assert claimed["request_id"] == request["request_id"]
    assert not repository.complete_detail_request_from_cache(request["request_id"], article_id)


@pytest.mark.parametrize("status", ["ready", "sent"])
def test_repeated_stale_button_requires_explicit_retry_and_keeps_feedback(
    tmp_path: Path,
    status: str,
) -> None:
    settings, repository, delivery, article_id = stored_article(tmp_path)
    delivery_id = int(delivery["delivery_id"])
    request = repository.detail_request_for_delivery(delivery_id)
    repository.set_detail_placeholder_message_id(request["request_id"], 9900)
    repository.finalize_feedback(
        delivery_id, action="super_reaction_without_comment", reaction="superlike"
    )
    repository.record_detail_feedback(delivery_id, article_id, action="detail_useful")
    if status == "sent":
        assert repository.mark_detail_request_sent(request["request_id"])
    with repository.connection() as connection:
        before_feedback = [
            tuple(row) for row in connection.execute("SELECT * FROM feedback_events")
        ]
    retire_policy(repository, article_id)
    old_article = repository.get_detailed_article(article_id)

    repeated = repository.enqueue_detail_request(delivery_id, 42)
    assert repeated["status"] == "failed"
    assert repeated["article_id"] == article_id
    assert repeated["placeholder_message_id"] == 9900
    assert repeated["created"] is False
    assert repeated["cache_hit"] is False
    assert details_acknowledgement(repeated)[1] is True
    assert repository.current_reaction(delivery_id) == "superlike"

    def no_automatic_paid_work(*_args, **_kwargs):
        pytest.fail("Policy invalidation must not start paid research without explicit retry")

    assert (
        DetailWorker(settings, repository, builder=no_automatic_paid_work).run_once().status
        == "idle"
    )
    assert repository.get_detailed_article(article_id) == old_article
    with repository.connection() as connection:
        assert [
            tuple(row) for row in connection.execute("SELECT * FROM feedback_events")
        ] == before_feedback
    assert repository.retry_detail_request(request["request_id"])["status"] == "queued"


@pytest.mark.parametrize("expired", [False, True])
def test_pending_delivery_rejects_old_policy_or_expired_article(
    tmp_path: Path,
    expired: bool,
) -> None:
    settings, repository, delivery, article_id = stored_article(tmp_path)
    request = repository.detail_request_for_delivery(delivery["delivery_id"])
    repository.set_detail_placeholder_message_id(request["request_id"], 9900)
    if expired:
        with repository.connection() as connection:
            connection.execute(
                "UPDATE detailed_articles SET expires_at='2000-01-01' WHERE id=?", (article_id,)
            )
    else:
        retire_policy(repository, article_id)

    assert repository.ready_detail_requests() == []
    assert repository.claim_ready_detail_request_for_delivery() is None
    assert repository.get_detail_request(request["request_id"])["status"] == "failed"
    assert DetailWorker(settings, repository).run_once().status == "idle"
    assert repository.get_detailed_article(article_id) is not None


def test_worker_cannot_fallback_to_stale_event_cache(tmp_path: Path) -> None:
    settings, repository, delivery, article_id = stored_article(tmp_path)
    retire_policy(repository, article_id)
    repository.enqueue_detail_request(second_delivery(repository, delivery), 42)
    calls = []

    def builder(_settings, _repository, event_id, **_kwargs):
        calls.append(event_id)
        return artifact(event_id).model_copy(
            update={
                "generation_key": detail_generation_key(event_id, []),
            }
        )

    result = DetailWorker(settings, repository, builder=builder).run_once()
    assert result.status == "ready"
    assert result.cache_hit is False
    assert result.article_id != article_id
    assert calls == [delivery["event_id"]]


def test_current_cache_still_avoids_another_build(tmp_path: Path) -> None:
    _, repository, delivery, article_id = stored_article(tmp_path)
    request = repository.enqueue_detail_request(second_delivery(repository, delivery), 42)
    assert request["status"] == "ready"
    assert request["cache_hit"] is True
    assert request["article_id"] == article_id
    repeated = repository.enqueue_detail_request(request["delivery_id"], 42)
    assert repeated["status"] == "ready"


def test_worker_rejects_outdated_builder_result(tmp_path: Path) -> None:
    settings, repository, _delivery = setup_worker(tmp_path)
    result = DetailWorker(
        settings,
        repository,
        builder=lambda _s, _r, event_id, **_kw: artifact(event_id).model_copy(
            update={"prompt_version": "old"}
        ),
    ).run_once()
    assert result.status == "failed"
    assert "outdated" in result.error
    assert repository.status_counts()["valid_detailed_articles"] == 0


@pytest.mark.parametrize("field", ["prompt_version", "schema_version"])
def test_repository_will_not_store_a_new_stale_policy_artifact(tmp_path: Path, field: str) -> None:
    _, repository, _delivery = setup_worker(tmp_path)
    request = repository.claim_next_detail_request()
    payload = valid_article_payload(expires_at="2099-01-01T00:00:00+00:00")
    payload["generation_key"] = request["generation_key"]
    payload[field] = "old-policy"
    with pytest.raises(ValueError, match="current evidence policy"):
        repository.store_detail_article_and_complete(request["request_id"], **payload)
    assert repository.status_counts()["valid_detailed_articles"] == 0
    assert repository.get_detail_request(request["request_id"])["status"] == "running"


def test_six_literal_quotes_fit_details_but_false_quote_still_fails() -> None:
    fact = fact_sheet().facts[0]
    quotes = [fact.evidence_quotes[0]] * 6
    detailed = DetailFact(fact_id=1, **{**fact.model_dump(), "evidence_quotes": quotes})
    assert len(detailed.evidence_quotes) == 6
    documents = {int(document["id"]): document for document in fact_documents()}
    valid = fact.model_copy(update={"evidence_quotes": quotes})
    assert not any(
        "not a literal excerpt" in error for error in fact_safety_errors(valid, documents)
    )
    invalid = fact.model_copy(
        update={
            "evidence_quotes": [
                *quotes[:5],
                EvidenceQuote(document_id=1, quote="This sentence never appeared in the source"),
            ]
        }
    )
    assert any("not a literal excerpt" in error for error in fact_safety_errors(invalid, documents))
    with pytest.raises(ValidationError):
        DetailFact(fact_id=1, **{**fact.model_dump(), "evidence_quotes": quotes + quotes[:1]})
