import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import SecretStr
from test_menu_batches import OWNER, fake_message, handler
from test_menu_batches import env as env

from app.menu_batches.people import prepare_people
from app.menu_batches.preparation import PreparationBudget, PreparationRepository
from app.research.web import ScrapeResponse, SearchResponse
from app.telegram_bot.bot import build_router
from app.telegram_bot.menu_batches import BatchAction, BatchWorker
from app.telegram_bot.news_depth import brief_card_html

TEXT = (
    "A journalist reports that Ada Example announced a new agent memory architecture. "
    "The team separates durable memory from the working context and checks retrieval with tests. "
    "This is a report of a public decision, rather than a direct quotation from the engineer."
)


class Web:
    def __init__(self, *, fallback=False, broken=False):
        self.searches = []
        self.reads = []
        self.fallback = fallback
        self.broken = broken

    def search(self, query, *, freshness, num_results):
        self.searches.append((query, freshness, num_results))
        assert num_results == 10
        results = (
            []
            if self.fallback and freshness == "last_month"
            else [dict(url="https://example.org/interview", title="Ada Example interview")]
        )
        if self.broken:
            results.insert(0, dict(url="https://example.org/unreadable"))
        return SearchResponse(query, results, 0.002)

    def scrape(self, url):
        self.reads.append(url)
        if url.endswith("unreadable"):
            raise TimeoutError("This page is unavailable")
        return ScrapeResponse(
            url,
            TEXT,
            {"title": "Ada Example interview", "publishedTime": "2026-07-29T10:00:00+00:00"},
            0.002,
        )


def setup(env, generate):
    settings, repo, store = env
    settings.openrouter_api_key = SecretStr("offline")
    batch, _ = store.request_person(OWNER, "ada_example", "first-person-request")
    with repo.connection() as db:
        prep = db.execute("SELECT * FROM menu_preparations WHERE batch_id=?", (batch,)).fetchone()
    scoped = PreparationRepository(repo.database_path, prep["id"])
    budget = PreparationBudget(settings, scoped, prep["limit_usd"], transport=generate)
    return settings, scoped, store.get(batch), budget


def response(request, _key, *, bad_sibling=False):
    data = json.loads(request["messages"][1]["content"])
    document = data["documents"][0]
    material = dict(
        headline="Новый подход к памяти агента",
        kind="action",
        paragraphs=[
            dict(
                text="Журналисты сообщают, что Ада представила архитектуру памяти агента. "
                "Она отделяет постоянную память от текущего контекста.",
                document_id=document["id"],
                quote=TEXT.split(". ")[0],
            )
        ],
    )
    materials = [material]
    if bad_sibling:
        materials.append(
            dict(
                headline="Неподтверждённая карточка",
                kind="statement",
                paragraphs=[
                    dict(
                        text="Этот тезис не имеет подтверждения в прочитанном источнике.",
                        document_id=99999,
                        quote="This quotation is invented and must be rejected.",
                    )
                ],
            )
        )
    return {
        "choices": [{"message": {"content": json.dumps({"materials": materials})}}],
        "usage": {"cost": 0.001, "prompt_tokens": 100, "completion_tokens": 100},
    }


def test_people_command_opens_person_buttons_without_preparing_or_paying(env):
    settings, repo, _ = env
    call = handler(build_router(settings, repo), "batch_command_handler")
    message = fake_message("/people")
    asyncio.run(call(message))
    markup = message.answer.call_args.kwargs["reply_markup"]
    assert any(
        b.callback_data == "mb:person:ada_example" for row in markup.inline_keyboard for b in row
    )
    with repo.connection() as db:
        assert db.execute("SELECT count(*) FROM menu_preparations").fetchone()[0] == 0
    assert repo.current_month_cost() == 0


def test_person_click_queues_fresh_search_without_extra_confirmation_and_is_idempotent(env):
    settings, repo, store = env
    call = handler(build_router(settings, repo), "batch_action_handler", False)
    query = SimpleNamespace(
        id="first",
        from_user=SimpleNamespace(id=OWNER),
        message=fake_message("/people"),
        answer=AsyncMock(),
    )
    asyncio.run(call(query, BatchAction(action="person", value="ada_example")))
    query.id = "second"
    asyncio.run(call(query, BatchAction(action="person", value="ada_example")))
    with repo.connection() as db:
        rows = db.execute("SELECT * FROM menu_preparations").fetchall()
        assert len(rows) == 1 and rows[0]["status"] == "queued"
        assert rows[0]["limit_usd"] == 0.08
        assert store.get(rows[0]["batch_id"])["persona_id"] == "ada_example"
    assert repo.current_month_cost() == 0


def test_old_empty_people_preparation_button_opens_the_person_picker(env):
    settings, repo, store = env
    batch, _ = store.request(OWNER, "people", "legacy-empty")
    with repo.connection() as db:
        db.execute("UPDATE menu_batches SET status_message_id=99 WHERE id=?", (batch,))
    call = handler(build_router(settings, repo), "batch_action_handler", False)
    query = SimpleNamespace(
        from_user=SimpleNamespace(id=OWNER), message=fake_message(number=99), answer=AsyncMock()
    )
    asyncio.run(call(query, BatchAction(action="offer", value=str(batch))))
    markup = query.message.answer.call_args.kwargs["reply_markup"]
    assert any(
        b.callback_data == "mb:person:ada_example" for row in markup.inline_keyboard for b in row
    )
    with repo.connection() as db:
        assert not db.execute("SELECT 1 FROM menu_preparations").fetchone()


def test_one_read_source_and_reported_action_can_publish_without_existing_radar_events(env):
    settings, repo, batch, budget = setup(env, response)
    web = Web()
    prepare_people(settings, repo, batch, budget, client=web)
    assert len(web.searches) == 1 and len(web.reads) == 1
    with repo.connection() as db:
        card = dict(db.execute("SELECT * FROM cards").fetchone())
        answer = db.execute("SELECT * FROM people_search_answers").fetchone()
    assert "29.07.2026" in card["rendered_html"]  # Original date, never today's date.
    assert "https://example.org/interview" in card["rendered_html"]
    visible = brief_card_html(card)
    assert "29.07.2026" in visible and "https://example.org/interview" in visible
    assert "Она отделяет постоянную память от текущего контекста." in visible
    assert len(env[2].items(batch["id"])) == 1 and answer["response_json"]
    assert budget.model.snapshot()["requests"] == 1
    # These cards already belong to the outbox, so no automatic double-send.
    from app.ops.schedule import fresh_delivery_card_ids

    assert fresh_delivery_card_ids(repo, owner_id=OWNER, now=datetime.now(UTC)) == []


def test_bad_sibling_and_one_failed_page_do_not_block_good_material(env):
    def generate(request, key):
        return response(request, key, bad_sibling=True)

    settings, repo, batch, budget = setup(env, generate)
    web = Web(broken=True)
    prepare_people(settings, repo, batch, budget, client=web)
    assert len(env[2].items(batch["id"])) == 1
    assert budget.web_unknown > 0 and budget.snapshot()["charged_or_reserved_usd"] < 0.08
    with repo.connection() as db:
        answer = db.execute("SELECT * FROM people_search_answers").fetchone()
    assert answer["status"] == "ready" and "source anchor" in answer["error"]


def test_sparse_person_gets_wider_search_without_faking_recency(env):
    settings, repo, batch, budget = setup(env, response)
    web = Web(fallback=True)
    prepare_people(settings, repo, batch, budget, client=web)
    assert [r[1] for r in web.searches] == ["last_month", "last_year"]
    assert len(env[2].items(batch["id"])) == 1


def test_saved_response_and_evidence_replay_without_any_new_paid_step(env):
    settings, repo, batch, budget = setup(env, response)
    prepare_people(settings, repo, batch, budget, client=Web())
    cost = repo.current_month_cost()
    denied = Mock(side_effect=AssertionError("No new paid calls"))
    no_web = SimpleNamespace(search=denied, scrape=denied)
    next_budget = PreparationBudget(settings, repo, 0.08, transport=denied)
    prepare_people(settings, repo, batch, next_budget, client=no_web)
    assert not denied.called and repo.current_month_cost() == cost
    assert len(env[2].items(batch["id"])) == 1


def test_unknown_model_call_is_not_replayed(env):
    denied = Mock(side_effect=TimeoutError())
    settings, repo, batch, budget = setup(env, denied)
    with pytest.raises(TimeoutError):
        prepare_people(settings, repo, batch, budget, client=Web())
    no_web = SimpleNamespace(search=Mock(side_effect=AssertionError()), scrape=Mock())
    with pytest.raises(ValueError, match="not confirmed"):
        prepare_people(
            settings,
            repo,
            batch,
            PreparationBudget(settings, repo, 0.08, transport=denied),
            client=no_web,
        )
    assert denied.call_count == 1 and not no_web.search.called


def test_different_people_do_not_share_cached_jobs_and_monthly_cap_is_enforced(env):
    settings, repo, store = env
    import yaml

    people = yaml.safe_load(settings.weekly_personas_file.read_text())
    people["personas"].append(
        dict(id="other", name="Other Person", role="Researcher", source_strategy={"x_handles": []})
    )
    settings.weekly_personas_file.write_text(yaml.safe_dump(people))
    first, _ = store.request_person(OWNER, "ada_example", "a")
    second, _ = store.request_person(OWNER, "other", "b")
    assert first != second
    repo.record_usage("fixture", "existing", cost_usd=settings.budget_hard_stop_usd - 0.01)
    assert store.request_person(OWNER, "other", "cached")[0] == second
    with repo.connection() as db:
        db.execute("UPDATE menu_preparations SET status='complete'")
    with pytest.raises(ValueError, match="budget"):
        store.request_person(OWNER, "other", "late", now=datetime.now(UTC) + timedelta(days=1))


def test_new_search_does_not_wait_for_unrelated_scheduled_slot(env, monkeypatch):
    settings, repo, store = env
    store.request_person(OWNER, "ada_example", "person-priority")
    # A slot's existence used to suppress every manual preparation.
    with repo.connection() as db:
        db.execute(
            "INSERT INTO schedule_slots(job_type,slot_at,mode,status) "
            "VALUES('radar',?,'live','planned')",
            (datetime.now(UTC).isoformat(),),
        )
    called = []
    monkeypatch.setattr(
        "app.menu_batches.preparation.prepare", lambda *args: called.append(args[-1])
    )
    bot = SimpleNamespace(edit_message_text=AsyncMock())
    assert asyncio.run(BatchWorker(bot, settings, repo).tick())
    assert called[0]["persona_id"] == "ada_example"
