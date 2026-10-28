import json

import pytest
import yaml
from test_menu_batches import OWNER, card
from test_menu_batches import env as env

from app.menu_batches.store import entity_labels


def aliases(settings):
    path = settings.weekly_personas_file
    data = yaml.safe_load(path.read_text())
    data["personas"][0]["aliases"] = ["Ада Пример", "Ады Пример", ""]
    path.write_text(yaml.safe_dump(data, allow_unicode=True))


def test_russian_aliases_and_declined_names_are_matched_as_whole_words(env):
    settings, _, _ = env
    aliases(settings)
    for text in ["Ада Пример представила агента", "Новое выступление Ады Пример"]:
        assert entity_labels(settings, {"headline": text}, "people") == ["Ada Example"]
    assert not entity_labels(settings, {"headline": "Вышел продукт Ада Примерная"}, "people")
    assert not entity_labels(settings, {"headline": "Новая модель агента"}, "people")


@pytest.mark.parametrize(
    "entities",
    [
        {"persons": [], "people": [{"name": "New Person"}]},
        {"persons": [{"canonical_name": "New Person"}]},
        {"people": "New Person"},
    ],
)
def test_explicit_person_entities_are_not_lost_when_another_field_is_empty(env, entities):
    assert entity_labels(env[0], {"entities_json": json.dumps(entities)}, "people") == [
        "New Person"
    ]


def test_tracked_x_author_is_a_person_without_self_mention(env):
    settings = env[0]
    event = {
        "source_url": "https://x.com/AdaExample/status/1234567890",
        "headline": "I released an agent",
    }
    assert entity_labels(settings, event, "people") == ["Ada Example"]
    for url in [
        "https://x.com.evil.test/AdaExample/status/123",
        "https://evil.test/AdaExample/status/123",
        "https://AdaExample@x.com/other/status/123",
    ]:
        assert not entity_labels(settings, {**event, "source_url": url}, "people")


def test_invalid_entity_json_does_not_hide_a_valid_person_alias(env):
    settings = env[0]
    aliases(settings)
    assert entity_labels(
        settings, {"entities_json": "{broken", "headline": "Заявление Ады Пример"}, "people"
    ) == ["Ada Example"]


def test_ready_people_edition_really_selects_alias_match_and_keeps_pause_guard(env):
    settings, repo, store = env
    aliases(settings)
    identifier, event = card(repo, entities={"organizations": ["Example AI"]})
    with repo.connection() as db:
        db.execute(
            "UPDATE events SET headline=? WHERE id=?", ("Ады Пример: запуск нового агента", event)
        )
    batch, _ = store.request(OWNER, "people", "people-alias-proof")
    assert [item["entity_id"] for item in store.items(batch)] == [identifier]
    repo.set_runtime_setting("source_enabled:rss", "false")
    assert (
        store.candidates(
            OWNER, "people", store.get(batch)["period_start"], store.get(batch)["period_end"]
        )
        == []
    )
