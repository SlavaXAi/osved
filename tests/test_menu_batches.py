import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.config import Settings, SourceConfig, SourceRegistry
from app.db.repository import Repository
from app.menu_batches.preparation import PreparationBudget, recover_preparations
from app.menu_batches.store import KINDS, BatchStore
from app.ops.budget import SearchBudgetExceeded
from app.ops.schedule import fresh_delivery_card_ids
from app.telegram_bot.bot import build_router
from app.telegram_bot.menu_batches import BatchAction, BatchWorker, batch_keyboard, batch_text
from app.video.relevance import VideoRelevanceService

OWNER = 123


@pytest.fixture
def env(tmp_path):
    profile = tmp_path / "interests.yaml"
    profile.write_text("version: 1\ntopics: [ai, agents]\n")
    people = tmp_path / "people.yaml"
    people.write_text(
        "version: 1\npersonas:\n  - id: ada_example\n    name: Ada Example\n"
        "    role: Engineer\n    source_strategy:\n      x_handles: [AdaExample]\n"
    )
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        interests_file=profile,
        weekly_personas_file=people,
        allowed_user_id=OWNER,
        menu_batches_enabled=True,
    )
    repo = Repository(settings.database_path)
    repo.migrate()
    repo.sync_sources(
        SourceRegistry(
            version=1,
            sources=[
                SourceConfig(
                    id=kind,
                    kind=kind,
                    name=kind,
                    locator="https://example.org/" + kind,
                    role="primary",
                    region="global",
                    language="en",
                )
                for kind in ("rss", "web", "telegram", "x", "youtube")
            ],
        )
    )
    return settings, repo, BatchStore(settings, repo)


def card(
    repo, kind="rss", *, age=0.1, entities=None, fingerprint=None, stage="final", valid="valid"
):
    now = datetime.now(UTC)
    with repo.connection() as db:
        n = db.execute("SELECT count(*) FROM source_items").fetchone()[0]
    published = (now - timedelta(days=age)).isoformat()
    item = repo.insert_source_item_with_id(
        source_id=kind,
        external_id=f"item-{n}",
        source_url=f"https://example.org/item/{n}",
        original_text="A concrete event about AI agents.",
        original_language="en",
        published_at=published,
        outbound_urls=[],
        media_metadata={},
        payload_hash=f"hash-{n}",
        raw_payload={},
    )
    with repo.connection() as db:
        event = db.execute(
            "INSERT INTO events(status,headline,event_type,entities_json,event_at,"
            "first_seen_at,latest_source_at,primary_source_item_id,score,fingerprint) "
            "VALUES('card_ready',?,'release',?,?,?,?,?,10,?)",
            (
                f"Event {n}",
                json.dumps(entities or {}),
                published,
                published,
                published,
                item,
                fingerprint,
            ),
        ).lastrowid
    identifier = repo.create_card(
        event_id=event,
        prompt_version="fixture",
        model="fixture",
        generation_cost_usd=0,
        structured_output={"evaluation_stage": stage},
        rendered_html=f"<b>Event {n}</b>\n\nVerified details. "
        f'<a href="https://example.org/item/{n}">Source</a>',
        source_urls=[f"https://example.org/item/{n}"],
        primary_preview_url=f"https://example.org/item/{n}",
        validation_status=valid,
    )
    return identifier, event


def attach_header(repo, batch, message_id=99):
    with repo.connection() as db:
        db.execute(
            "UPDATE menu_batches SET status_message_id=?,header_status='sent' WHERE id=?",
            (message_id, batch),
        )


def handler(router, name, message=True):
    return next(
        h.callback
        for h in (router.message if message else router.callback_query).handlers
        if h.callback.__name__ == name
    )


def fake_message(text="/news", *, owner=OWNER, chat=OWNER, number=1, bot=None):
    bot = bot or SimpleNamespace(
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=701)),
        edit_message_text=AsyncMock(),
    )
    return SimpleNamespace(
        text=text,
        message_id=number,
        from_user=SimpleNamespace(id=owner),
        chat=SimpleNamespace(id=chat, type="private"),
        bot=bot,
        answer=AsyncMock(return_value=SimpleNamespace(message_id=99)),
    )


@pytest.mark.parametrize(
    "kind,source", [("news", "rss"), ("telegram", "telegram"), ("twitter", "x")]
)
def test_sections_are_separate_and_never_call_providers(env, kind, source):
    settings, repo, store = env
    ids = {k: card(repo, k)[0] for k in ("rss", "web", "telegram", "x")}
    batch, _ = store.request(OWNER, kind, "cmd:" + kind)
    selected = {i["entity_id"] for i in store.items(batch)}
    assert selected == ({ids["rss"], ids["web"]} if kind == "news" else {ids[source]})
    assert repo.current_month_cost() == 0
    assert len(store.items(batch)) <= KINDS[kind][2]


def test_people_and_institutions_are_current_events_not_biographies(env):
    _, repo, store = env
    person, _ = card(repo, "telegram", entities={"persons": ["Ada Example"]})
    organization, _ = card(repo, "rss", entities={"organizations": ["Research Institute"]})
    card(repo, "web")
    people, _ = store.request(OWNER, "people", "people")
    institutions, _ = store.request(OWNER, "institutions", "institutions")
    assert [i["entity_id"] for i in store.items(people)] == [person]
    assert [i["entity_id"] for i in store.items(institutions)] == [organization]


def test_registry_name_matches_known_person_without_external_lookup(env):
    _, repo, store = env
    identifier, event = card(repo)
    with repo.connection() as db:
        db.execute(
            "UPDATE events SET headline='Ada Example explains agent engineering' WHERE id=?",
            (event,),
        )
    batch, _ = store.request(OWNER, "people", "people")
    assert store.items(batch)[0]["entity_id"] == identifier


def test_exposure_is_shared_with_daily_feed_and_other_sections(env):
    _, repo, store = env
    old, _ = card(repo, entities={"persons": ["Ada Example"]})
    d = repo.ensure_delivery(old, OWNER)
    repo.mark_delivery_sent(d, 101)
    fresh, _ = card(repo, entities={"persons": ["Ada Example"]})
    news, _ = store.request(OWNER, "news", "news")
    assert [i["entity_id"] for i in store.items(news)] == [fresh]
    people, _ = store.request(OWNER, "people", "people")
    assert store.items(people) == []
    assert fresh_delivery_card_ids(repo, owner_id=OWNER, now=datetime.now(UTC)) == []


def test_only_latest_finished_version_of_a_story_is_sent(env):
    _, repo, store = env
    old, event = card(repo, "rss", fingerprint="same-story")
    latest = repo.create_card(
        event_id=event,
        prompt_version="updated",
        model="fixture",
        generation_cost_usd=0,
        structured_output={"evaluation_stage": "final"},
        rendered_html="<b>Updated grounded card</b>",
        source_urls=["https://example.org"],
        primary_preview_url="https://example.org",
        validation_status="valid",
    )
    batch, _ = store.request(OWNER, "news", "a")
    assert [i["entity_id"] for i in store.items(batch)] == [latest]
    assert old != latest


def test_stale_future_invalid_and_preview_cards_never_fill_an_edition(env):
    _, repo, store = env
    for options in ({"age": 5}, {"age": -1}, {"valid": "invalid"}, {"stage": "candidate_preview"}):
        card(repo, **options)
    fresh, _ = card(repo)
    batch, _ = store.request(OWNER, "news", "news")
    assert [i["entity_id"] for i in store.items(batch)] == [fresh]


def test_disabled_sources_are_not_used(env):
    _, repo, store = env
    card(repo, "telegram")
    repo.set_runtime_setting("source_enabled:telegram", "false")
    batch, _ = store.request(OWNER, "telegram", "tg")
    assert store.items(batch) == []


def test_parallel_commands_reserve_one_batch_and_one_set_of_items(env):
    _, repo, store = env
    card(repo)
    with ThreadPoolExecutor(max_workers=6) as pool:
        values = list(pool.map(lambda n: store.request(OWNER, "news", f"cmd-{n}"), range(6)))
    assert len({v[0] for v in values}) == 1
    assert sum(v[1] for v in values) == 1
    assert len(store.items(values[0][0])) == 1
    assert (
        BatchStore(store.settings, Repository(repo.database_path)).request(OWNER, "news", "cmd-0")[
            0
        ]
        == values[0][0]
    )


def test_timeout_and_restart_never_resend_an_uncertain_item(env):
    settings, repo, store = env
    card(repo)
    batch, _ = store.request(OWNER, "news", "news")
    attach_header(repo, batch)
    bot = SimpleNamespace(
        send_message=AsyncMock(side_effect=TimeoutError()), edit_message_text=AsyncMock()
    )
    worker = BatchWorker(bot, settings, repo)
    assert asyncio.run(worker.tick())
    assert store.items(batch)[0]["status"] == "uncertain"
    assert not asyncio.run(worker.tick())
    assert bot.send_message.await_count == 1
    with repo.connection() as db:
        assert db.execute("SELECT delivery_status FROM deliveries").fetchone()[0] == "uncertain"
    assert (
        BatchStore(settings, Repository(repo.database_path)).request(
            OWNER, "news", "new", force=True
        )[0]
        != batch
    )
    assert fresh_delivery_card_ids(repo, owner_id=OWNER, now=datetime.now(UTC)) == []


def test_expired_send_lease_becomes_uncertain_without_network(env):
    _, repo, store = env
    card(repo)
    batch, _ = store.request(OWNER, "news", "news")
    attach_header(repo, batch)
    claimed = store.claim_item()
    with repo.connection() as db:
        db.execute(
            "UPDATE menu_batch_items SET lease_until=? WHERE id=?", (time.time() - 1, claimed["id"])
        )
    assert store.claim_item() is None
    assert store.items(batch)[0]["status"] == "uncertain"


def test_batch_does_not_wait_for_feedback_or_change_daily_frontier(env):
    settings, repo, store = env
    card(repo)
    card(repo)
    repo.set_runtime_setting("news_queue_current:123", "777")
    batch, _ = store.request(OWNER, "news", "news")
    attach_header(repo, batch)
    bot = SimpleNamespace(
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=100)),
        edit_message_text=AsyncMock(),
    )
    worker = BatchWorker(bot, settings, repo)
    asyncio.run(worker.tick())
    asyncio.run(worker.tick())
    assert bot.send_message.await_count == 2
    assert store.get(batch)["status"] == "sent"
    assert repo.get_runtime_setting("news_queue_current:123") == "777"
    assert repo.current_month_cost() == 0


def video(settings, repo, vid, *, short=False, score=0.9):
    identifier = repo.insert_source_item_with_id(
        source_id="youtube",
        external_id=vid,
        source_url="https://www.youtube.com/watch?v=" + vid,
        original_text="AI agents engineering.",
        original_language="en",
        published_at=datetime.now(UTC).isoformat(),
        outbound_urls=[],
        media_metadata={},
        payload_hash=vid,
        raw_payload={},
    )
    profile = VideoRelevanceService(settings, repo, OWNER).profile_hash
    now = datetime.now(UTC).isoformat()
    with repo.connection() as db:
        if short is not None:
            db.execute(
                "INSERT INTO youtube_video_types(video_id,is_short,basis,checked_at) "
                "VALUES(?,?,?,?)",
                (vid, int(short), "fixture", time.time()),
            )
        db.execute(
            "INSERT INTO video_relevance_assessments(source_item_id,chat_id,profile_hash,"
            "status,assessment_json,created_at,updated_at) VALUES(?,?,?,'ready',?,?,?)",
            (
                identifier,
                OWNER,
                profile,
                json.dumps(
                    {
                        "relevance": score,
                        "description_html": "<b>AI agents</b>\n\nA technical discussion.",
                    }
                ),
                now,
                now,
            ),
        )
    return identifier


def test_youtube_uses_only_cached_relevant_confirmed_non_shorts(env):
    settings, repo, store = env
    good = video(settings, repo, "abcdefghijk")
    video(settings, repo, "bcdefghijkl", short=True)
    video(settings, repo, "cdefghijklm", short=None)
    video(settings, repo, "defghijklmn", score=0.1)
    batch, _ = store.request(OWNER, "youtube", "youtube")
    assert [i["entity_id"] for i in store.items(batch)] == [good]
    assert repo.current_month_cost() == 0


def test_new_menu_is_owner_only_and_does_not_shadow_bio_management(env):
    settings, repo, _ = env
    router = build_router(settings, repo)
    callback = handler(router, "batch_command_handler")
    private = fake_message()
    asyncio.run(callback(private))
    assert private.answer.await_count == 1
    for message in (fake_message(owner=999), fake_message(chat=-123)):
        asyncio.run(callback(message))
        assert message.answer.await_count == 0
    assert handler(router, "biography_status")
    assert handler(router, "people_handler")
    assert handler(router, "news_handler")
    assert handler(router, "videos_handler")


def test_double_command_does_not_duplicate_header_or_items(env):
    settings, repo, _ = env
    callback = handler(build_router(settings, repo), "batch_command_handler")
    msg = fake_message()
    asyncio.run(callback(msg))
    asyncio.run(callback(msg))
    assert msg.answer.await_count == 1


def test_preparation_requires_two_explicit_clicks_and_is_idempotent(env):
    settings, repo, store = env
    batch, _ = store.request(OWNER, "news", "news")
    attach_header(repo, batch)
    callback = handler(build_router(settings, repo), "batch_action_handler", False)
    query = SimpleNamespace(
        id="q",
        from_user=SimpleNamespace(id=OWNER),
        message=fake_message(number=99),
        bot=SimpleNamespace(edit_message_text=AsyncMock()),
        answer=AsyncMock(),
    )
    data = BatchAction(action="offer", value=str(batch))
    asyncio.run(callback(query, data))
    with repo.connection() as db:
        assert db.execute("SELECT status FROM menu_preparations").fetchone()[0] == "offered"
    asyncio.run(callback(query, BatchAction(action="prepare", value=str(batch))))
    asyncio.run(callback(query, BatchAction(action="prepare", value=str(batch))))
    with repo.connection() as db:
        assert db.execute("SELECT count(*) FROM menu_preparations").fetchone()[0] == 1
        assert db.execute("SELECT status FROM menu_preparations").fetchone()[0] == "queued"
    assert repo.current_month_cost() == 0


def test_forged_preparation_callback_is_refused(env):
    settings, repo, store = env
    batch, _ = store.request(OWNER, "news", "news")
    attach_header(repo, batch)
    callback = handler(build_router(settings, repo), "batch_action_handler", False)
    query = SimpleNamespace(
        from_user=SimpleNamespace(id=OWNER), message=fake_message(number=100), answer=AsyncMock()
    )
    asyncio.run(callback(query, BatchAction(action="offer", value=str(batch))))
    with repo.connection() as db:
        assert db.execute("SELECT count(*) FROM menu_preparations").fetchone()[0] == 0


def test_budget_refuses_next_call_and_unknown_outcome_is_never_repaid(env):
    settings, repo, _ = env
    calls = []

    def transport(payload, key):
        calls.append(payload)
        raise TimeoutError("unconfirmed paid call")

    budget = PreparationBudget(settings, repo, 0.05, transport=transport)
    payload = {
        "model": "fixture",
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 100,
    }
    with pytest.raises(TimeoutError):
        budget.transport(payload, "test")
    with pytest.raises(SearchBudgetExceeded):
        budget.transport(payload, "test")
    assert len(calls) == 1
    budget = PreparationBudget(settings, repo, 0.0001, transport=transport)
    with pytest.raises(SearchBudgetExceeded):
        budget.transport(payload, "test")
    assert len(calls) == 1


def test_interrupted_preparation_is_reconciled_once_not_replayed(env):
    _, repo, store = env
    batch, _ = store.request(OWNER, "news", "news")
    now = datetime.now(UTC).isoformat()
    with repo.connection() as db:
        db.execute(
            "INSERT INTO menu_preparations(batch_id,owner_id,status,limit_usd,created_at,"
            "updated_at) VALUES(?,?,'running',.05,?,?)",
            (batch, OWNER, now, now),
        )
    recover_preparations(repo)
    recover_preparations(repo)
    assert repo.current_month_cost() == pytest.approx(0.05)
    with repo.connection() as db:
        assert db.execute("SELECT status FROM menu_preparations").fetchone()[0] == "uncertain"


def test_manual_biography_uses_reserve_without_enabling_daily_schedule(env):
    from app.biography.storage import BiographyStore

    settings, repo, store = env
    bio = BiographyStore(repo)
    bio.initialize()
    bio.set_enabled(False)
    with repo.connection() as db:
        person_id = db.execute("SELECT id FROM biography_people WHERE living=0 LIMIT 1").fetchone()[
            0
        ]
    cycle = bio.start_cycle(datetime.now(UTC).date().isoformat())
    identifier = bio.save_issue(
        person_id,
        cycle,
        {
            "draft": {
                "headline": "Learning and applying ideas in practical work",
                "headline_fact_ids": [1],
                "paragraphs": [
                    {
                        "text": "A documented episode of learning and practical work. " * 4,
                        "fact_ids": [1],
                    }
                    for _ in range(3)
                ],
                "image_caption": "An illustration of the historical context.",
                "source_ids": [1, 2],
            },
            "image": {
                "url": "https://upload.wikimedia.org/wikipedia/commons/a/a0/example.jpg",
                "page_url": "https://commons.wikimedia.org/wiki/File:Example.jpg",
                "title": "Example",
                "description": "Historical context",
                "creator": "Example",
                "license": "CC BY-SA 4.0",
            },
            "sources": [
                {"url": "https://example.org/one", "title": "First source"},
                {"url": "https://example.org/two", "title": "Second source"},
            ],
        },
        {},
    )
    bio.finish_cycle(cycle, "ready")
    batch, _ = store.request(OWNER, "bio", "bio")
    attach_header(repo, batch)
    bot = SimpleNamespace(
        send_rich_message=AsyncMock(return_value=SimpleNamespace(message_id=222)),
        edit_message_text=AsyncMock(),
    )
    assert asyncio.run(BatchWorker(bot, settings, repo).tick())
    assert store.items(batch)[0]["entity_id"] == identifier
    assert store.items(batch)[0]["status"] == "sent"
    assert not bio.enabled(True)
    assert bio.status()["ready"] == 0
    assert repo.current_month_cost() == 0


def test_empty_output_is_honest_and_has_an_explicit_preparation_button(env):
    _, _, store = env
    batch, _ = store.request(OWNER, "twitter", "x")
    text = batch_text(store, store.get(batch))
    assert "Новых готовых" in text
    assert any(
        "Подготовить новые" in b.text
        for row in batch_keyboard(store, store.get(batch)).inline_keyboard
        for b in row
    )


def test_paid_preparation_creates_a_durable_item_with_scoped_receipts(env, monkeypatch):
    from app.menu_batches.preparation import prepare

    settings, repo, store = env
    from pydantic import SecretStr

    settings.openrouter_api_key = SecretStr("offline-fixture")
    old, event = card(repo)
    with repo.connection() as db:
        db.execute("DELETE FROM cards WHERE id=?", (old,))
        db.execute("UPDATE events SET status='candidate' WHERE id=?", (event,))
    batch, _ = store.request(OWNER, "news", "news")
    attach_header(repo, batch)
    now = datetime.now(UTC).isoformat()
    with repo.connection() as db:
        prep_id = db.execute(
            "INSERT INTO menu_preparations(batch_id,owner_id,status,limit_usd,"
            "created_at,updated_at) VALUES(?,?,'running',.05,?,?)",
            (batch, OWNER, now, now),
        ).lastrowid

    class Research:
        def __init__(self, settings, repository, client=None):
            self.repository = repository

        def research(self, *args, **kwargs):
            self.repository.record_usage("fixture", "web.fixture", cost_usd=0.001)

    def facts(*args, **kwargs):
        return {"status": "valid"}

    def generate(settings, repository, *, event_id, **kwargs):
        repository.record_usage("fixture", "card.fixture", cost_usd=0.002)
        repository.create_card(
            event_id=event_id,
            prompt_version="fixture",
            model="fixture",
            generation_cost_usd=0.002,
            structured_output={"evaluation_stage": "final"},
            rendered_html="<b>Prepared card</b>",
            source_urls=["https://example.org"],
            primary_preview_url="https://example.org",
            validation_status="valid",
        )

    monkeypatch.setattr("app.menu_batches.preparation.ResearchEventService", Research)
    monkeypatch.setattr("app.menu_batches.preparation.build_event_fact_sheet", facts)
    monkeypatch.setattr("app.menu_batches.preparation.generate_cards", generate)
    prep = {"id": prep_id, "batch_id": batch, "limit_usd": 0.05}
    prepare(settings, repo, prep)
    assert len(store.items(batch)) == 1
    assert store.get(batch)["status"] == "queued"
    with repo.connection() as db:
        saved = db.execute("SELECT * FROM menu_preparations WHERE id=?", (prep_id,)).fetchone()
        assert saved["status"] == "complete"
        assert saved["cost_usd"] == pytest.approx(0.003)
        assert db.execute("SELECT status FROM menu_preparation_events").fetchone()[0] == "complete"
        assert (
            db.execute(
                "SELECT count(*) FROM usage_ledger WHERE "
                "json_extract(metadata_json,'$.menu_preparation_id')=?",
                (prep_id,),
            ).fetchone()[0]
            == 2
        )


def test_rebuilding_a_completed_batch_does_not_leave_a_phantom_active_queue(env):
    settings, repo, store = env
    card(repo)
    batch, _ = store.request(OWNER, "news", "news")
    attach_header(repo, batch)
    item = store.claim_item()
    store.finish(item, "sent", message_id=77)
    store.extend(batch)
    assert store.get(batch)["status"] == "sent"


def test_removed_source_is_checked_again_before_sending(env):
    settings, repo, store = env
    card(repo)
    batch, _ = store.request(OWNER, "news", "news")
    attach_header(repo, batch)
    repo.set_runtime_setting("source_enabled:rss", "false")
    bot = SimpleNamespace(send_message=AsyncMock(), edit_message_text=AsyncMock())
    asyncio.run(BatchWorker(bot, settings, repo).tick())
    assert bot.send_message.await_count == 0
    assert store.items(batch)[0]["status"] == "skipped"


def test_automatic_research_waits_while_a_manual_paid_preparation_is_running(env):
    _, repo, store = env
    batch, _ = store.request(OWNER, "news", "news")
    now = datetime.now(UTC).isoformat()
    with repo.connection() as db:
        db.execute(
            "INSERT INTO menu_preparations(batch_id,owner_id,status,limit_usd,created_at,"
            "updated_at) VALUES(?,?,'running',.05,?,?)",
            (batch, OWNER, now, now),
        )
    assert repo.claim_schedule_slot("research", now, "live") is None
    with repo.connection() as db:
        db.execute("UPDATE menu_preparations SET status='complete'")
    assert repo.claim_schedule_slot("research", now, "live") is not None


def test_deleted_header_can_be_reopened_without_rebuilding_materials(env):
    settings, repo, store = env
    callback = handler(build_router(settings, repo), "batch_command_handler")
    msg = fake_message()
    asyncio.run(callback(msg))
    msg.bot.edit_message_text.side_effect = RuntimeError("message to edit not found")
    asyncio.run(callback(msg))
    assert msg.answer.await_count == 2
    with repo.connection() as db:
        assert db.execute("SELECT count(*) FROM menu_batches").fetchone()[0] == 1


def test_stale_paid_confirmation_does_not_dispatch_preparation(env):
    settings, repo, store = env
    batch, _ = store.request(OWNER, "news", "news", now=datetime.now(UTC) - timedelta(hours=1))
    attach_header(repo, batch)
    callback = handler(build_router(settings, repo), "batch_action_handler", False)
    query = SimpleNamespace(
        from_user=SimpleNamespace(id=OWNER),
        message=fake_message(number=99),
        answer=AsyncMock(),
        bot=SimpleNamespace(edit_message_text=AsyncMock()),
    )
    asyncio.run(callback(query, BatchAction(action="offer", value=str(batch))))
    asyncio.run(callback(query, BatchAction(action="prepare", value=str(batch))))
    with repo.connection() as db:
        assert db.execute("SELECT status FROM menu_preparations").fetchone()[0] == "offered"


def test_period_comparison_uses_instants_and_falls_back_to_source_time():
    from app.menu_batches.preparation import in_period

    batch = {"period_start": "2026-10-20T09:00:00+00:00", "period_end": "2026-10-21T09:00:00+00:00"}
    assert in_period({"event_at": "2026-10-21T11:59:00+03:00"}, batch)
    assert in_period({"source_published_at": "2026-10-20T10:00:00Z"}, batch)
    assert not in_period({"event_at": "2026-10-21T13:00:00+03:00"}, batch)
    assert not in_period({"event_at": "invalid"}, batch)
