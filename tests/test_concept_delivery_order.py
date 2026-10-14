from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.config import Settings, SourceConfig, SourceRegistry
from app.db.repository import Repository
from app.ops.schedule import fresh_delivery_card_ids

OWNER = 123
NOW = datetime.fromisoformat("2026-10-13T16:30:00+03:00")


@pytest.fixture
def delivery_repo(tmp_path):
    settings = Settings(_env_file=None, app_env="test", data_dir=tmp_path, allowed_user_id=OWNER)
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
    return repo


def card(repo, *, concept=True, source_at=None, created_at=None, score=6.5, event_at=None):
    source_at = source_at or NOW - timedelta(hours=2)
    created_at = created_at or NOW - timedelta(hours=1)
    with repo.connection() as db:
        count = db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    source_url = f"https://x.com/author/status/{100000 + count}"
    item = repo.insert_source_item_with_id(
        source_id="x",
        external_id=str(100000 + count),
        source_url=source_url,
        original_text="A concrete explanation of a useful strategy with documentary evidence.",
        original_language="en",
        published_at=source_at.isoformat(),
        outbound_urls=[],
        media_metadata={},
        payload_hash=f"delivery-order-{count}",
        raw_payload={},
    )
    with repo.connection() as db:
        event = db.execute(
            "INSERT INTO events(status,event_type,headline,event_at,first_seen_at,"
            "latest_source_at,primary_source_item_id,score) VALUES('card_ready',?,?,?,?,?,?,?)",
            (
                "concept_brief" if concept else "release",
                f"Distinct useful explanation {count}",
                (event_at or source_at).isoformat(),
                created_at.isoformat(),
                source_at.isoformat(),
                item,
                score,
            ),
        ).lastrowid
    card_id = repo.create_card(
        event_id=event,
        prompt_version="fixture",
        model="offline",
        generation_cost_usd=0,
        structured_output={"card_kind": "concept_brief" if concept else "news"},
        rendered_html="<b>A useful explanation</b>",
        source_urls=[source_url],
        primary_preview_url=source_url,
        validation_status="valid",
    )
    with repo.connection() as db:
        db.execute("UPDATE cards SET created_at=? WHERE id=?", (created_at.isoformat(), card_id))
    return card_id


def delivery(repo, card_id, *, at=NOW, status="sent", owner=OWNER):
    delivery_id = repo.ensure_delivery(card_id, owner)
    with repo.connection() as db:
        db.execute(
            "UPDATE deliveries SET delivery_status=?,created_at=?,delivered_at=? WHERE id=?",
            (status, at.isoformat(), at.isoformat() if status == "sent" else None, delivery_id),
        )
    return delivery_id


def test_oldest_prepared_concept_is_first_even_when_newer_has_higher_score(delivery_repo):
    repo = delivery_repo
    newer = card(repo, created_at=NOW - timedelta(hours=1), score=100)
    oldest = card(repo, created_at=NOW - timedelta(days=2), score=3)
    assert fresh_delivery_card_ids(repo, owner_id=OWNER, now=NOW, limit=2) == [oldest]
    delivery(repo, oldest)
    assert fresh_delivery_card_ids(repo, owner_id=OWNER, now=NOW, limit=2) == [newer]


def test_queue_order_compares_instants_across_timezone_offsets(delivery_repo):
    repo = delivery_repo
    older = card(repo, created_at=datetime.fromisoformat("2026-10-13T12:00:00+03:00"))
    card(repo, created_at=datetime.fromisoformat("2026-10-13T10:00:00+00:00"))
    assert fresh_delivery_card_ids(repo, owner_id=OWNER, now=NOW, limit=1) == [older]


def test_one_concept_per_batch_leaves_space_for_regular_news(delivery_repo):
    repo = delivery_repo
    first = card(repo, created_at=NOW - timedelta(hours=3))
    card(repo, created_at=NOW - timedelta(hours=2))
    news = {card(repo, concept=False, score=10), card(repo, concept=False, score=9)}
    selected = fresh_delivery_card_ids(repo, owner_id=OWNER, now=NOW, limit=3)
    assert selected[0] == first and set(selected[1:]) == news


@pytest.mark.parametrize("status", ["sent", "pending", "uncertain"])
def test_two_concepts_per_moscow_day_and_ambiguous_deliveries_consume_capacity(
    delivery_repo, status
):
    repo = delivery_repo
    first = card(repo, created_at=NOW - timedelta(hours=3))
    second = card(repo, created_at=NOW - timedelta(hours=2))
    third = card(repo, created_at=NOW - timedelta(hours=1))
    assert fresh_delivery_card_ids(repo, owner_id=OWNER, now=NOW, limit=1) == [first]
    delivery(repo, first, at=NOW - timedelta(minutes=30), status=status)
    assert fresh_delivery_card_ids(repo, owner_id=OWNER, now=NOW, limit=1) == [second]
    delivery(repo, second, at=NOW - timedelta(minutes=15), status=status)
    assert fresh_delivery_card_ids(repo, owner_id=OWNER, now=NOW, limit=1) == []
    with repo.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM deliveries WHERE card_id=?", (third,)).fetchone()[
            0
        ] == 0


def test_moscow_midnight_resets_capacity_even_with_two_deliveries_within_24_hours(delivery_repo):
    repo = delivery_repo
    for hour in (19, 20):
        sent = card(repo, created_at=NOW - timedelta(days=1))
        delivery(repo, sent, at=datetime.fromisoformat(f"2026-10-12T{hour}:45:00+00:00"))
    now = datetime.fromisoformat("2026-10-12T21:30:00+00:00")  # 13 October, 00:30 Moscow.
    queued = card(
        repo, created_at=now - timedelta(hours=1), source_at=now - timedelta(hours=2)
    )
    assert fresh_delivery_card_ids(repo, owner_id=OWNER, now=now, limit=1) == [queued]


def test_utc_previous_date_deliveries_on_current_moscow_date_count_toward_capacity(delivery_repo):
    repo = delivery_repo
    for minute in (5, 10):
        sent = card(repo, created_at=NOW - timedelta(days=1))
        delivery(repo, sent, at=datetime.fromisoformat(f"2026-10-12T21:{minute:02}:00+00:00"))
    now = datetime.fromisoformat("2026-10-12T21:30:00+00:00")
    card(repo, created_at=now - timedelta(hours=1), source_at=now - timedelta(hours=2))
    assert fresh_delivery_card_ids(repo, owner_id=OWNER, now=now, limit=1) == []


def test_recent_concept_is_not_stranded_by_old_source_but_old_news_stays_stale(delivery_repo):
    repo = delivery_repo
    old_source = NOW - timedelta(days=8)
    concept = card(repo, source_at=old_source)
    card(repo, concept=False, source_at=old_source, score=100)
    card(repo, concept=False, event_at=old_source, score=100)
    assert fresh_delivery_card_ids(repo, owner_id=OWNER, now=NOW, limit=3) == [concept]


@pytest.mark.parametrize("created_age", [timedelta(days=5, seconds=1), timedelta(minutes=-11)])
def test_stale_or_future_concept_creation_date_does_not_reenter_queue(
    delivery_repo, created_age
):
    repo = delivery_repo
    card(repo, created_at=NOW - created_age)
    assert fresh_delivery_card_ids(repo, owner_id=OWNER, now=NOW, limit=2) == []


def test_concept_five_day_creation_boundary_is_inclusive(delivery_repo):
    repo = delivery_repo
    boundary = card(repo, created_at=NOW - timedelta(days=5))
    assert fresh_delivery_card_ids(repo, owner_id=OWNER, now=NOW, limit=1) == [boundary]


def test_regular_batch_is_filled_completely_after_concept_capacity_is_used(delivery_repo):
    repo = delivery_repo
    for _ in range(2):
        delivered = card(repo)
        delivery(repo, delivered)
    queued = card(repo)
    news = {card(repo, concept=False, score=10), card(repo, concept=False, score=9)}
    selected = fresh_delivery_card_ids(repo, owner_id=OWNER, now=NOW, limit=2)
    assert set(selected) == news and queued not in selected


def test_large_high_score_news_backlog_does_not_hide_a_waiting_concept(delivery_repo):
    repo = delivery_repo
    queued = card(repo, score=3)
    for _ in range(151):
        card(repo, concept=False, score=10)
    selected = fresh_delivery_card_ids(repo, owner_id=OWNER, now=NOW, limit=2)
    assert len(selected) == 2 and selected[0] == queued


def test_sent_today_counts_today_when_delivery_was_reserved_yesterday(delivery_repo):
    repo = delivery_repo
    for _ in range(2):
        sent = card(repo, created_at=NOW - timedelta(days=1))
        delivery_id = delivery(repo, sent, at=NOW - timedelta(hours=2))
        with repo.connection() as db:
            db.execute(
                "UPDATE deliveries SET created_at=? WHERE id=?",
                ((NOW - timedelta(days=1)).isoformat(), delivery_id),
            )
    card(repo)
    assert fresh_delivery_card_ids(repo, owner_id=OWNER, now=NOW, limit=1) == []


def test_other_owner_deliveries_do_not_consume_capacity(delivery_repo):
    repo = delivery_repo
    available_to_owner = []
    for _ in range(2):
        delivered = card(repo, created_at=NOW - timedelta(days=2))
        available_to_owner.append(delivered)
        delivery(repo, delivered, owner=456)
    card(repo)
    selected = fresh_delivery_card_ids(repo, owner_id=OWNER, now=NOW, limit=1)
    assert selected == [available_to_owner[0]]


def test_failed_delivery_does_not_consume_daily_capacity(delivery_repo):
    repo = delivery_repo
    first = card(repo, created_at=NOW - timedelta(hours=3))
    failed = card(repo, created_at=NOW - timedelta(hours=2))
    delivery(repo, first)
    delivery(repo, failed, status="failed")
    assert fresh_delivery_card_ids(repo, owner_id=OWNER, now=NOW, limit=1) == [failed]


@pytest.mark.parametrize("limit", [0, -1])
def test_nonpositive_batch_capacity_returns_nothing(delivery_repo, limit):
    repo = delivery_repo
    card(repo)
    card(repo, concept=False)
    assert fresh_delivery_card_ids(repo, owner_id=OWNER, now=NOW, limit=limit) == []
