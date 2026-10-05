from __future__ import annotations

from pathlib import Path

from app.config import load_source_registry
from app.db.repository import Repository
from app.radar.feedback import (
    FEEDBACK_EXAMPLES_VERSION,
    MAX_FEEDBACK_EXAMPLES,
    load_feedback_examples,
)

PROJECT_ROOT = Path(__file__).parents[1]


def discovery_delivery(
    repository: Repository,
    *,
    index: int,
    evaluation_stage: str = "discovery",
) -> int:
    source_url = f"https://example.com/event-{index}"
    item_id = repository.insert_source_item_with_id(
        source_id="web_agent_frontier_global",
        external_id=f"feedback-event-{index}",
        source_url=source_url,
        original_text=(f"Concrete technology event number {index}. " * 4),
        original_language="en",
        published_at="2026-08-23T10:00:00+00:00",
        outbound_urls=[],
        media_metadata=None,
        payload_hash=f"feedback-hash-{index}",
        raw_payload={},
    )
    assert item_id is not None
    event_id, _ = repository.attach_item_to_event(
        source_item_id=item_id,
        event_type="web_signal",
        headline=f"Событие номер {index} изменило рынок технологий",
        event_at="2026-08-23T10:00:00+00:00",
        primary_url=source_url,
        fingerprint=f"feedback-fingerprint-{index}",
        score=8,
        score_features={"base_score": 8},
        decision_reason="test",
        topics=["ai", "manufacturing"],
    )
    card_id = repository.create_candidate_preview_card(
        event_id=event_id,
        prompt_version="search_candidate_preview_v1",
        model="test-model",
        generation_cost_usd=0,
        structured_output={
            "evaluation_stage": evaluation_stage,
            "candidate": {
                "headline": f"Событие номер {index} изменило рынок технологий",
                "summary": (
                    f"Компания запустила измеримое производство номер {index} "
                    "и раскрыла мощность предприятия."
                ),
                "topics": ["ai", "manufacturing"],
            },
        },
        rendered_html=f"<b>Событие {index}</b>",
        source_urls=[source_url],
        primary_preview_url=source_url,
    )
    return repository.create_candidate_batch(
        chat_id=123,
        card_ids=[card_id],
        batch_id=f"feedback-batch-{index}",
    )[0]


def repository(tmp_path: Path) -> Repository:
    result = Repository(tmp_path / "feedback-examples.sqlite3")
    result.migrate()
    result.sync_sources(load_source_registry(PROJECT_ROOT / "tests/fixtures/config/sources.yaml"))
    return result


def test_latest_reaction_wins_without_inheriting_stale_comment(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    delivery_id = discovery_delivery(repo, index=1)
    repo.finalize_feedback(
        delivery_id,
        action="superreaction_comment",
        reaction="superlike",
        comment="Тема отличная, но нужно больше сравнений.",
    )
    repo.finalize_feedback(
        delivery_id,
        action="reaction",
        reaction="dislike",
        comment=None,
    )

    examples = load_feedback_examples(repo)

    assert len(examples) == 1
    assert examples[0]["reaction"] == "dislike"
    assert examples[0]["weight"] == -1
    assert examples[0]["comment"] == ""
    assert "сравнений" not in str(examples[0])


def test_examples_are_discovery_only_weighted_and_bounded(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    for index in range(MAX_FEEDBACK_EXAMPLES + 2):
        delivery_id = discovery_delivery(repo, index=index)
        reaction = "superlike" if index % 2 else "superdislike"
        repo.finalize_feedback(
            delivery_id,
            action="superreaction_silent",
            reaction=reaction,
        )
    ignored_delivery = discovery_delivery(repo, index=99, evaluation_stage="publication")
    repo.finalize_feedback(
        ignored_delivery,
        action="reaction",
        reaction="like",
    )

    examples = load_feedback_examples(repo, limit=99)

    assert len(examples) == MAX_FEEDBACK_EXAMPLES
    assert {item["weight"] for item in examples} == {-2, 2}
    assert all("номер 99" not in item["headline"] for item in examples)
    expected_fields = {"headline", "summary", "topics", "reaction", "comment", "weight"}
    assert all(set(item) == expected_fields for item in examples)
    assert FEEDBACK_EXAMPLES_VERSION == "search_feedback_examples_v1"
