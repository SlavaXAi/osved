import pytest

from app.llm.openrouter import HOOK_ACTION_PATTERNS, semantic_hook_errors


@pytest.mark.parametrize(
    "prefix",
    [
        "В представленных материалах",
        "В представленном описании",
        "В представленном подтверждении",
        "По представлению автора",
    ],
)
def test_presented_source_context_is_not_a_product_release(prefix):
    basis = (
        "Тариф Catch составляет 99 долларов в месяц; независимая проверка условий не подтверждена."
    )
    text = prefix + " тариф Catch составляет 99 долларов в месяц; "
    text += "независимая проверка условий не подтверждена."
    assert not HOOK_ACTION_PATTERNS["release"].search(text)
    assert semantic_hook_errors(text, basis, supporting_text=basis) == []


def test_company_representatives_are_not_a_release_verb():
    text = "Представители компании поясняют условия тарифа Catch."
    assert not HOOK_ACTION_PATTERNS["release"].search(text)


@pytest.mark.parametrize(
    "text",
    [
        "Компания представила сервис Catch.",
        "Компания представит сервис Catch.",
        "Компания представляет сервис Catch.",
        "Компании представляют сервис Catch.",
        "Сервис Catch представлен компанией.",
        "Модель Catch представлена компанией.",
        "Приложение Catch представлено компанией.",
        "Продукты Catch представлены компанией.",
        "Компания выпустила сервис Catch.",
        "Компания опубликовала сервис Catch.",
    ],
)
def test_actual_release_verbs_and_short_passives_still_require_source_support(text):
    basis = "Сервис Catch обрабатывает электронную почту и планирует встречи."
    assert HOOK_ACTION_PATTERNS["release"].search(text)
    errors = semantic_hook_errors(text, basis, supporting_text=basis)
    assert any("action absent" in error and "release" in error for error in errors)
    assert semantic_hook_errors(text, text, supporting_text=text) == []


@pytest.mark.parametrize(
    "text,action",
    [
        ("Компания приобрела сервис Catch.", "acquire"),
        ("Компания закрыла сервис Catch.", "close"),
        ("Компания удалила данные Catch.", "delete"),
    ],
)
def test_other_action_guards_are_unchanged(text, action):
    basis = "Сервис Catch обрабатывает электронную почту и планирует встречи."
    errors = semantic_hook_errors(text, basis, supporting_text=basis)
    assert any("action absent" in error and action in error for error in errors)


def test_body_source_publication_is_not_checked_as_a_headline_release():
    basis = "Catch получила финансирование и доступна руководителям компаний."
    paragraph = "Сообщение о финансировании и доступности Catch опубликовано отраслевым изданием."
    strict = semantic_hook_errors(paragraph, basis, supporting_text=basis)
    assert any("action absent" in error and "release" in error for error in strict)
    assert (
        semantic_hook_errors(
            paragraph,
            basis,
            supporting_text=basis,
            check_hook_actions=False,
        )
        == []
    )


def test_body_negative_audit_statement_is_not_checked_as_a_headline_release():
    basis = "У Catch нет публичного аудита безопасности; компания описывает свои меры защиты."
    paragraph = "Публичного аудита безопасности Catch не представлено."
    assert (
        semantic_hook_errors(
            paragraph,
            basis,
            supporting_text=basis,
            check_hook_actions=False,
        )
        == []
    )


def test_disabling_body_action_heuristic_preserves_other_semantic_checks():
    cases = [
        (
            "Google описывает ассистента для планирования встреч.",
            "OpenAI описывает ассистента для планирования встреч.",
            "organization",
        ),
        (
            "Catch описывает ассистента для планирования встреч.",
            "Catch описывает космическую робототехнику для планирования встреч.",
            "subject",
        ),
        (
            "Astra reaches the Critical level of cybersecurity capability under the Framework.",
            "Astra достигла критического уровня кибербезопасности согласно Framework.",
            "security strength",
        ),
        (
            "Catch описывает ассистента для планирования встреч.",
            "Река течёт среди соснового леса.",
            "conservative paraphrase",
        ),
    ]
    for basis, paragraph, expected in cases:
        errors = semantic_hook_errors(
            paragraph,
            basis,
            supporting_text=basis,
            check_hook_actions=False,
        )
        assert any(expected in error for error in errors), errors


def test_development_intent_is_not_a_completed_build_in_a_headline():
    basis = "Catch получила финансирование для создания ИИ-ассистента руководителя."
    headline = "Catch создала ИИ-ассистента руководителя."
    errors = semantic_hook_errors(headline, basis, supporting_text=basis)
    assert any("action absent" in error and "build" in error for error in errors)
