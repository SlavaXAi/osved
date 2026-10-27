import json
from unittest.mock import Mock

import pytest
from test_x_reposts import enqueue, receipt
from test_x_reposts import env as env

from app.reposts import provider, store


def response(paragraphs, *, thesis="Тезис", finish="stop"):
    return {
        "choices": [
            {
                "finish_reason": finish,
                "message": {"content": json.dumps({"paragraphs": paragraphs, "thesis": thesis})},
            }
        ],
        "usage": {"cost": 0.001},
    }


def test_complete_translation_survives_trailing_schema_noise():
    paragraphs = [
        "Первый абзац сохраняет все 12 847 файлов и путь /private/cache.",
        "Второй абзац: «Прямая цитата». " * 100,
        "Последняя мысль автора, без пересказа и сокращений.",
    ]
    result = provider.parse_translation(
        response(["\n\n".join(paragraphs), "thesis ור=", "", None, " \n "]), "Original post"
    )
    assert result["paragraphs"] == [p.strip() for p in paragraphs]
    assert result["thesis"] == "Тезис"
    assert set(result["structure_repairs"]) == {
        "paragraph_breaks",
        "empty_paragraph",
        "trailing_schema_fragment",
    }


@pytest.mark.parametrize(
    "paragraphs,expected",
    [
        ("Первый абзац.\n\nВторой абзац.", ["Первый абзац.", "Второй абзац."]),
        ([{"text": "Прямой перевод."}, "", None], ["Прямой перевод."]),
        (["Первый.\n \n \nВторой."], ["Первый.", "Второй."]),
        (["שלום", "ИИ", "{}", "0"], ["שלום", "ИИ", "{}", "0"]),
        (["Первый.", "thesis: ИИ"], ["Первый.", "thesis: ИИ"]),
        (["Первый.", "thesis = AI"], ["Первый.", "thesis = AI"]),
        (["Первый.", '"thesis": ""'], ["Первый."]),
    ],
)
def test_mechanical_shapes_and_meaningful_short_text(paragraphs, expected):
    assert provider.parse_translation(response(paragraphs), "Original")["paragraphs"] == expected


def test_literal_schema_example_in_original_is_preserved():
    fragment = '"thesis": ""'
    result = provider.parse_translation(
        response(["Пример JSON:", fragment]), "Example JSON: " + fragment
    )
    assert result["paragraphs"] == ["Пример JSON:", fragment]


@pytest.mark.parametrize("paragraphs", [[None, ""], ["thesis="], [False], [{"other": "text"}]])
def test_unusable_answers_never_turn_into_a_fabricated_publication(paragraphs):
    with pytest.raises(ValueError):
        provider.parse_translation(response(paragraphs), "Original")


def test_truncated_translation_still_requires_review():
    with pytest.raises(ValueError, match="оборвался"):
        provider.parse_translation(response(["Незаконченный перевод"], finish="length"), "Original")


def test_more_than_sixty_real_paragraphs_are_not_silently_cut():
    paragraphs = [f"Полный абзац {i}." for i in range(65)]
    assert provider.parse_translation(response(paragraphs), "Original")["paragraphs"] == paragraphs


def test_paid_bad_structure_is_repaired_from_cache_without_key_or_network(env, monkeypatch):
    settings, repo = env
    row = enqueue(env)
    paragraphs = ["Весь первый абзац.", "Весь второй абзац."]
    fetch = Mock(return_value=receipt())
    transport = Mock(return_value=response(["\n\n".join(paragraphs), "thesis ור=", ""]))
    with monkeypatch.context() as m:
        m.setattr(provider, "parse_translation", Mock(side_effect=ValueError("old strict check")))
        with pytest.raises(ValueError, match="old strict check"):
            provider.prepare(settings, repo, row, fetcher=fetch, transport=transport)
    assert fetch.call_count == transport.call_count == 1
    before = repo.current_month_cost()
    with repo.connection() as db:
        saved = [
            tuple(r) for r in db.execute("SELECT id,response_json FROM details_provider_calls")
        ]
    denied = Mock(side_effect=AssertionError("A format repair must never spend again"))
    changed = settings.model_copy(
        update={"openrouter_search_model": "different-model", "openrouter_api_key": None}
    )
    _, translated = provider.prepare(
        changed,
        repo,
        store.get(repo, row["id"]),
        fetcher=denied,
        fallback_fetcher=denied,
        transport=denied,
    )
    assert translated["paragraphs"] == paragraphs
    assert not denied.called and repo.current_month_cost() == before
    with repo.connection() as db:
        assert [
            tuple(r) for r in db.execute("SELECT id,response_json FROM details_provider_calls")
        ] == saved
    assert (
        provider.prepare(changed, repo, store.get(repo, row["id"]), transport=denied)[1]
        == translated
    )
    assert not denied.called


def test_corrupt_paid_cache_is_not_treated_as_permission_to_spend(env):
    settings, repo = env
    row = enqueue(env)
    provider.prepare(
        settings,
        repo,
        row,
        fetcher=Mock(return_value=receipt()),
        transport=Mock(return_value=response(["Полный перевод."])),
    )
    with repo.connection() as db:
        db.execute("UPDATE x_reposts SET translation_json=NULL WHERE id=?", (row["id"],))
        db.execute(
            "UPDATE details_provider_calls SET response_json='null' WHERE stage='x_translation'"
        )
    denied = Mock(side_effect=AssertionError("Must not spend on a replacement answer"))
    before = repo.current_month_cost()
    with pytest.raises(ValueError, match="повторная оплата отключена"):
        provider.prepare(
            settings.model_copy(update={"openrouter_search_model": "changed-model"}),
            repo,
            store.get(repo, row["id"]),
            transport=denied,
            fetcher=denied,
        )
    assert not denied.called and repo.current_month_cost() == before
