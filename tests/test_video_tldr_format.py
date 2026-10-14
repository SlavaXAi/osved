from __future__ import annotations

import re
from html import unescape

import pytest

from app.video.tldr import (
    LEGACY_VERSIONS,
    PROMPT,
    TELEGRAM_TEXT_LIMIT,
    VERSION,
    extract_tldr,
    format_legacy_paragraphs,
    telegram_tldr,
)

VIDEO = "abcdefghijk"


def test_prompt_requests_meaningful_prose_and_keeps_author_attribution():
    assert "2000–2400" in PROMPT and "3–4 связных абзаца" in PROMPT
    assert "не раздувай" in PROMPT and "1–2 абзаца" in PROMPT
    assert "Сохраняй, кто высказывает мнение" in PROMPT
    assert "завершённым предложением" in PROMPT
    assert "подзаголовки, списки" in PROMPT
    assert VERSION not in LEGACY_VERSIONS
    assert "c1b14664e329d27f" in LEGACY_VERSIONS


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_extract_preserves_paragraphs_and_normalizes_wrapped_lines(count):
    paragraphs = [f"Автор объясняет\n механизм {index}." for index in range(count)]
    value = "## TL;DR\n\n" + "\n\n".join(paragraphs)
    assert extract_tldr(value, require_finished_sentences=True) == "\n\n".join(
        p.replace("\n ", " ") for p in paragraphs
    )


def test_legacy_extraction_does_not_reject_an_old_unpunctuated_result():
    assert extract_tldr("## TL;DR\n\nСтарый сохранённый текст") == "Старый сохранённый текст"
    with pytest.raises(ValueError, match="invalid_output"):
        extract_tldr("## TL;DR\n\nНовая мысль без окончания", require_finished_sentences=True)


@pytest.mark.parametrize(
    "body",
    [
        "\n\n".join("Абзац." for _ in range(5)),
        "Тезис.\n\n- Первый пункт.",
        "Тезис.\n## Заголовок\nСледующая мысль.",
        "Автор предлагает **совет**.",
        "Автор <b>предлагает</b> совет.",
        "Тезис. https://example.org",
        "Доказательство [[E0001]].",
        "😃" * 1951,
    ],
)
def test_rejects_nonprose_unsafe_or_oversized_result(body):
    with pytest.raises(ValueError, match="invalid_output"):
        extract_tldr("## TL;DR\n\n" + body, require_finished_sentences=True)


def test_extract_full_report_still_reuses_its_lead():
    full = "## TL;DR\n\nАвтор объясняет причину.\n\n## Подробно\n\nПолный разбор. [[E0001]]"
    assert extract_tldr(full, full_report=True) == "Автор объясняет причину."


def test_self_contained_telegram_result_escapes_metadata_and_prose():
    result = telegram_tldr(
        "Автор сравнивает: 1 < 2 & 3.\n\nПри этих условиях результат меняется.",
        title='Объяснение <механизма> & "условий"',
        video_id=VIDEO,
    )
    assert result.startswith("<b>Объяснение &lt;механизма&gt; &amp; &quot;условий&quot;</b>\n\n")
    assert "1 &lt; 2 &amp; 3." in result
    assert result.endswith(
        f'\n\n<a href="https://www.youtube.com/watch?v={VIDEO}">Смотреть видео</a>'
    )
    assert "\n\nПри этих условиях" in result


@pytest.mark.parametrize("video", ["javascript:alert(1)", 'abcdefghij"', "https://example.org"])
def test_source_link_cannot_be_replaced_with_untrusted_url(video):
    with pytest.raises(ValueError):
        telegram_tldr("Полная мысль.", title="Видео", video_id=video)


def test_source_url_is_canonicalized():
    result = telegram_tldr("Полная мысль.", title="Видео", video_id=f"https://youtu.be/{VIDEO}")
    assert f'href="https://www.youtube.com/watch?v={VIDEO}"' in result


def test_renderer_limit_includes_title_source_label_and_utf16_units():
    title = "Т" * 400
    overhead = len((title + "\n\n\n\nСмотреть видео").encode("utf-16-le")) // 2
    body = "а" * (TELEGRAM_TEXT_LIMIT - overhead - 1) + "."
    result = telegram_tldr(body, title=title, video_id=VIDEO)
    displayed = unescape(re.sub(r"<[^>]*>", "", result))
    assert len(displayed.encode("utf-16-le")) // 2 == TELEGRAM_TEXT_LIMIT
    with pytest.raises(ValueError, match="invalid_output"):
        telegram_tldr(body + "а", title=title, video_id=VIDEO)
    with pytest.raises(ValueError, match="invalid_output"):
        telegram_tldr("😀" * 1900 + ".", title=title, video_id=VIDEO)


def test_renderer_legacy_single_argument_stays_compatible():
    assert telegram_tldr("Сравнение: 1 < 2 & 3.") == (
        "<b>TL;DR</b>\n\nСравнение: 1 &lt; 2 &amp; 3."
    )


def test_renderer_keeps_previously_accepted_cached_markup_as_escaped_text():
    old = "Автор <b>объясняет</b> условия и **главный тезис**."
    assert extract_tldr("## TL;DR\n\n" + old) == old
    assert "&lt;b&gt;объясняет&lt;/b&gt;" in telegram_tldr(old, video_id=VIDEO)
    with pytest.raises(ValueError, match="invalid_output"):
        extract_tldr("## TL;DR\n\n" + old, require_finished_sentences=True)


def test_renderer_title_fallback_and_validation():
    assert telegram_tldr("Полная мысль.", title=" \n ").startswith("<b>TL;DR</b>")
    with pytest.raises(ValueError, match="invalid_metadata"):
        telegram_tldr("Полная мысль.", title="Т" * 501)
    with pytest.raises(ValueError, match="invalid_metadata"):
        telegram_tldr("Полная мысль.", title=123)
    with pytest.raises(ValueError, match="invalid_metadata"):
        telegram_tldr("Полная мысль.", title="😀" * 251)


def test_new_generated_limit_fits_source_title_and_footer_while_cache_stays_usable():
    longest_new = "а" * 3499 + "."
    assert (
        extract_tldr("## TL;DR\n\n" + longest_new, require_finished_sentences=True) == longest_new
    )
    rendered = telegram_tldr(longest_new, title="Т" * 500, video_id=VIDEO)
    displayed = unescape(re.sub(r"<[^>]*>", "", rendered))
    assert len(displayed.encode("utf-16-le")) // 2 < TELEGRAM_TEXT_LIMIT
    old_long = "а" * 3899 + "."
    assert extract_tldr("## TL;DR\n\n" + old_long) == old_long
    with pytest.raises(ValueError, match="invalid_output"):
        extract_tldr("## TL;DR\n\n" + longest_new + ".", require_finished_sentences=True)


def test_long_legacy_synopsis_gets_paragraphs_without_new_content():
    sentences = [
        f"Автор объясняет механизм {index}, его экономические последствия и конкретные условия "
        "применения этого подхода на практике."
        for index in range(9)
    ]
    original = " ".join(sentences)
    formatted = format_legacy_paragraphs(original)
    paragraphs = formatted.split("\n\n")
    assert len(paragraphs) == 3
    assert " ".join(formatted.split()) == original
    assert all(p.endswith(".") for p in paragraphs)
    rendered = telegram_tldr(original, title="Видео", video_id=VIDEO)
    assert formatted in rendered


def test_four_long_legacy_sentences_get_two_balanced_paragraphs():
    sentences = [
        "Автор описывает перестройку производства: сначала компания разбирает повторяемые "
        "операции, определяет ограничения оборудования и на этой основе решает, где "
        "автоматизация может изменить весь процесс, а не только отдельную задачу.",
        "Далее он связывает результат с организацией работы: инженерные команды должны "
        "получать данные об ошибках и задержках, иначе более быстрое оборудование оставляет "
        "старые узкие места и не даёт ожидаемого изменения выпуска.",
        "Экономический эффект, по его объяснению, зависит от загрузки предприятия, цены "
        "перенастройки и доступности специалистов, поэтому одинаковое решение может "
        "оказаться полезным на одном заводе и слишком дорогим на другом.",
        "В завершение автор предлагает оценивать переход по нескольким полным циклам "
        "производства: сравнивать качество, расходы и время выполнения заказа, сохраняя "
        "исходные условия сравнения вместо выводов по одному удачному запуску.",
    ]
    original = " ".join(sentences)
    assert len(original) >= 700
    formatted = format_legacy_paragraphs(original)
    assert formatted.split("\n\n") == [" ".join(sentences[:2]), " ".join(sentences[2:])]
    assert " ".join(formatted.split()) == original
    assert telegram_tldr(original, title="Видео", video_id=VIDEO).count("\n\n") == 3


def test_two_long_legacy_sentences_get_two_complete_paragraphs_without_word_changes():
    sentences = [
        "По объяснению автора, внедрение начинается с описания существующего процесса, "
        "его ограничений, повторяемых операций и источников ошибок, после чего команда "
        "сравнивает возможные решения по стоимости, срокам и требованиям к обслуживанию, "
        "поскольку высокая скорость отдельного оборудования сама по себе не показывает "
        "пользу для всего предприятия и может лишь перенести задержки на следующую стадию.",
        "В 2025 г. Россия рассматривается в примере автора как один из рынков, где эффект "
        "такого перехода нужно измерять на нескольких полных циклах с одинаковыми "
        "условиями, учитывая обучение работников, изменения качества и перенастройку "
        "производства, а окончательный вывод он связывает с устойчивым сокращением "
        "расходов и времени заказа вместо отдельной демонстрации новой машины.",
    ]
    original = " ".join(sentences)
    assert len(original) >= 700
    formatted = format_legacy_paragraphs(original)
    assert formatted.split("\n\n") == sentences
    assert "г.\n\nРоссия" not in formatted
    assert " ".join(formatted.split()) == original


def test_long_legacy_paragraph_does_not_leave_a_tiny_second_paragraph():
    original = "Автор объясняет " + "механизм " * 85 + ". Короткое условие."
    assert len(original) >= 700
    assert format_legacy_paragraphs(original) == original


def test_legacy_split_respects_short_material_existing_paragraphs_and_abbreviations():
    short = "Автор объясняет единственную мысль. Это короткий выпуск."
    assert format_legacy_paragraphs(short) == short
    existing = short + "\n\nУсловия автора сохранены."
    assert format_legacy_paragraphs(existing) == existing
    sentence = "В 2025 г. Россия изменила подход, по объяснению автора, при этих условиях."
    original = " ".join([sentence * 2 for _ in range(6)])
    formatted = format_legacy_paragraphs(original)
    assert "г.\n\nРоссия" not in formatted
    assert " ".join(formatted.split()) == original


def test_long_legacy_text_without_enough_sentence_boundaries_is_not_cut():
    original = "Автор объясняет " + "механизм " * 120 + "."
    assert format_legacy_paragraphs(original) == original
