from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.radar.preference_patches import build_preference_report, render_preference_report
from tests.test_radar_feedback import discovery_delivery, repository


def test_latest_reaction_has_no_stale_comment_and_report_is_read_only(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    delivery = discovery_delivery(repo, index=1)
    old_id = repo.finalize_feedback(
        delivery,
        action="superreaction_comment",
        reaction="superlike",
        comment="Тема супер, нужно усилить поиск этой области",
    )
    latest_id = repo.finalize_feedback(delivery, action="reaction", reaction="dislike")
    with repo.connection() as connection:
        before = [tuple(row) for row in connection.execute("SELECT * FROM feedback_events")]
    report = build_preference_report(repo)
    assert report["read_only"]
    assert report["automatic_changes"] == 0
    assert report["examples"][0]["feedback_id"] == latest_id
    assert report["examples"][0]["comment"] == ""
    assert report["proposals"] == []
    assert old_id not in [example["feedback_id"] for example in report["examples"]]
    with repo.connection() as connection:
        after = [tuple(row) for row in connection.execute("SELECT * FROM feedback_events")]
    assert after == before
    assert "Изменения не применены" in render_preference_report(report)


def test_bad_presentation_does_not_make_a_topic_negative(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    for index in range(3):
        delivery = discovery_delivery(repo, index=index)
        repo.finalize_feedback(
            delivery,
            action="superreaction_comment",
            reaction="superdislike",
            comment="Тема интересная, но структура ужасная и слишком много цифр в заголовке",
        )
    report = build_preference_report(repo)
    assert len(report["proposals"]) == 3
    assert all(patch["scope"] == "card_quality" for patch in report["proposals"])
    assert all(patch["requires_review"] for patch in report["proposals"])
    assert all(patch["automatic_effect"] == "none" for patch in report["proposals"])


def test_repeated_consistent_events_create_evidence_linked_topic_proposals(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    ids = []
    for index in range(3):
        delivery = discovery_delivery(repo, index=index)
        ids.append(repo.finalize_feedback(delivery, action="reaction", reaction="like"))
    report = build_preference_report(repo)
    assert {patch["target"] for patch in report["proposals"]} == {"ai", "manufacturing"}
    for patch in report["proposals"]:
        assert patch["direction"] == "increase"
        assert patch["feedback_ids"] == ids
        assert len(patch["event_ids"]) == 3
        assert patch["status"] == "proposed"
    assert build_preference_report(repo)["proposals"] == report["proposals"]


def test_superreaction_without_comment_is_a_specific_example_not_a_topic_ban(
    tmp_path: Path,
) -> None:
    repo = repository(tmp_path)
    delivery = discovery_delivery(repo, index=1)
    repo.finalize_feedback(delivery, action="superreaction_silent", reaction="superdislike")
    report = build_preference_report(repo)
    assert len(report["proposals"]) == 1
    assert report["proposals"][0]["scope"] == "example"
    assert report["proposals"][0]["direction"] == "negative_example"
    assert report["proposals"][0]["confidence"] == 0.4
    assert build_preference_report(repo, chat_id=456)["examples"] == []
    assert build_preference_report(repo, limit=0)["examples"] == []


def test_detail_feedback_is_separate_and_dates_are_bounded(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    delivery = discovery_delivery(repo, index=1)
    repo.finalize_feedback(delivery, action="reaction", reaction="superlike")
    context = repo.get_delivery_context(delivery)
    assert context
    with repo.connection() as connection:
        for action in ("detail_useful", "detail_not_worth_it"):
            connection.execute(
                "INSERT INTO feedback_events "
                "(delivery_id, event_id, card_id, action, metadata_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (delivery, context["event_id"], context["card_id"], action, '{"article_id": 7}'),
            )
    report = build_preference_report(repo)
    assert report["reaction_counts"] == {"superlike": 1}
    assert len(report["details_feedback"]) == 1
    assert report["details_feedback"][0]["action"] == "detail_not_worth_it"
    later = build_preference_report(repo, now=datetime.now(UTC) + timedelta(days=20), days=7)
    assert later["examples"] == []
    assert later["details_feedback"] == []
