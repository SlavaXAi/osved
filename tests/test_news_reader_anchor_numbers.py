from app.details.models import (
    DeepArticleDraft,
    DeepArticleParagraph,
    DeepArticleSection,
    DeepSourceAnchor,
    DetailFact,
    DetailFactSnapshot,
)
from app.details.research import repair_deep_article_metadata, validate_deep_article
from app.pipeline.numbers import extract_numeric_values
from app.research.facts import EvidenceQuote
from tests.test_details_research import document, fact_sheet, valid_article


def composite_context(*, supported=True):
    rows = fact_sheet().facts
    date = " Правила вступят в силу 12 ноября."
    date_quote = rows[1].text + date if supported else rows[1].text
    facts = [
        DetailFact(
            fact_id=1,
            text=rows[0].text,
            document_ids=[2],
            evidence_quotes=[EvidenceQuote(document_id=2, quote=rows[0].text)],
            kind=rows[0].kind,
        ),
        DetailFact(
            fact_id=2,
            text=rows[1].text + date,
            document_ids=[1, 2],
            evidence_quotes=[
                EvidenceQuote(document_id=1, quote=date_quote),
                EvidenceQuote(document_id=2, quote=rows[1].text),
            ],
            kind=rows[1].kind,
        ),
        *[
            DetailFact(
                fact_id=index,
                text=row.text,
                document_ids=[1],
                evidence_quotes=[EvidenceQuote(document_id=1, quote=row.text)],
                kind=row.kind,
            )
            for index, row in enumerate(rows[2:], start=3)
        ],
    ]
    docs = [
        document(
            1,
            "https://wire.example/atlas",
            "independent",
            "\n".join([date_quote, rows[2].text, rows[3].text]) * 10,
        ),
        document(
            2,
            "https://openai.com/atlas",
            "primary",
            "\n".join([rows[0].text, rows[1].text]) * 10,
        ),
    ]
    snapshot = DetailFactSnapshot(
        event_id=7,
        event_type="software_release",
        facts=facts,
        coverage=fact_sheet().coverage,
        document_ids=[1, 2],
        source_domains=["wire.example", "openai.com"],
    )
    draft = valid_article()
    draft = draft.model_copy(
        update={
            "lead": "Правила вступят в силу 12 ноября. " + draft.lead,
            "source_anchors": [
                draft.source_anchors[0].model_copy(
                    update={
                        "phrase": "Правила вступят в силу 12 ноября",
                        "document_id": 2,
                        "role": "primary",
                    }
                ),
                draft.source_anchors[1].model_copy(update={"document_id": 1}),
            ],
        }
    )
    return draft, snapshot, docs


def numeric_anchor_errors(draft, snapshot, docs):
    return [
        error
        for error in validate_deep_article(draft, snapshot=snapshot, documents=docs)
        if "phrase numbers are not supported" in error
    ]


def test_composite_fact_date_anchor_is_routed_to_its_literal_quote_document():
    draft, snapshot, docs = composite_context()
    assert numeric_anchor_errors(draft, snapshot, docs)

    repaired = repair_deep_article_metadata(draft, snapshot=snapshot, documents=docs)

    date_anchors = [
        anchor
        for anchor in repaired.source_anchors
        if "12" in extract_numeric_values(anchor.phrase)
    ]
    assert date_anchors
    assert all(anchor.fact_id == 2 for anchor in date_anchors)
    assert all(anchor.document_id == 1 for anchor in date_anchors)
    assert all(anchor.role == "independent" for anchor in date_anchors)
    assert repaired.lead == draft.lead
    assert repaired.sections == draft.sections
    assert numeric_anchor_errors(repaired, snapshot, docs) == []


def test_metadata_repair_does_not_fake_missing_number_in_any_source_quote():
    draft, snapshot, docs = composite_context(supported=False)

    repaired = repair_deep_article_metadata(draft, snapshot=snapshot, documents=docs)

    assert repaired.lead == draft.lead
    assert numeric_anchor_errors(repaired, snapshot, docs)


def test_document_body_number_does_not_replace_a_fact_evidence_quote():
    draft, snapshot, docs = composite_context(supported=False)
    docs[1]["content_text"] += "\nAn unrelated item mentions 12 November."

    assert numeric_anchor_errors(draft, snapshot, docs)


def test_saved_usage_policy_artifact_repairs_both_date_fact_and_document():
    # Captured artifact 6: date anchor was fact 4/doc 9685, while its paragraph
    # cited [3, 4] and only fact 3's quote on doc 9509 supported November 12.
    date_text = (
        "Обновление вступает в силу 12 ноября 2026 года. Основным механизмом применения "
        "Anthropic называет способность Claude завершать такие взаимодействия: функция "
        "уже позволяет моделям заканчивать редкие разговоры с настойчиво оскорбляющими "
        "пользователями на Claude.ai и Claude Code."
    )
    abuse_text = (
        "Anthropic добавила в Usage Policy запрет на длительное и неоправданное "
        "оскорбительное или жестокое обращение с моделями."
    )
    conversation_text = (
        "Связанная с правилом функция позволяет Claude завершать редкие разговоры с "
        "настойчиво оскорбляющими пользователями на Claude.ai и Claude Code; после "
        "завершения пользователь не может отправлять новые сообщения в этом разговоре, "
        "но может сразу начать новый чат, а другие разговоры в аккаунте не затрагиваются."
    )
    date_quote = "The updated policy takes effect on November 12."
    conversation_quote = (
        "When Claude chooses to end a conversation, the user will no longer be able "
        "to send new messages in that conversation."
    )
    alignment_quote = (
        "This addition aligns with a step we've already taken, allowing Claude models "
        "to end rare conversations with persistently abusive users on Claude.ai and Claude Code."
    )
    snapshot = DetailFactSnapshot(
        event_id=1419,
        event_type="company_event",
        facts=[
            DetailFact(
                fact_id=1,
                text=abuse_text,
                document_ids=[9509],
                evidence_quotes=[
                    EvidenceQuote(
                        document_id=9509,
                        quote=(
                            "We’ve added a prohibition on sustained and needless abusive "
                            "or cruel behavior toward our models."
                        ),
                    )
                ],
                kind="capability",
            ),
            DetailFact(
                fact_id=3,
                text="Обновлённая политика вступает в силу 12 ноября 2026 года.",
                document_ids=[9509],
                evidence_quotes=[EvidenceQuote(document_id=9509, quote=date_quote)],
                kind="availability",
            ),
            DetailFact(
                fact_id=4,
                text=conversation_text,
                document_ids=[9685, 9509],
                evidence_quotes=[
                    EvidenceQuote(document_id=9685, quote=conversation_quote),
                    EvidenceQuote(document_id=9509, quote=alignment_quote),
                ],
                kind="capability",
            ),
        ],
        coverage=fact_sheet().coverage,
        document_ids=[9509, 9685],
        source_domains=["anthropic.com"],
    )
    docs = [
        document(
            9509,
            "https://www.anthropic.com/news/2026-usage-policy-update",
            "primary",
            date_quote + "\n" + alignment_quote,
        ),
        document(
            9685,
            "https://www.anthropic.com/research/end-subset-conversations",
            "primary",
            conversation_quote,
        ),
    ]
    phrase = "Обновление вступает в силу 12 ноября"
    draft = DeepArticleDraft(
        headline="Anthropic добавила правило об обращении с моделями",
        headline_fact_ids=[1],
        lead=abuse_text,
        lead_fact_ids=[1],
        sections=[
            DeepArticleSection(
                heading="Действие обновления",
                paragraphs=[DeepArticleParagraph(text=date_text, fact_ids=[3, 4])],
            ),
            DeepArticleSection(
                heading="Завершение разговора",
                paragraphs=[DeepArticleParagraph(text=conversation_text, fact_ids=[4])],
            ),
        ],
        source_anchors=[
            DeepSourceAnchor(phrase=phrase, fact_id=4, document_id=9685, role="primary"),
            DeepSourceAnchor(
                phrase="Anthropic добавила в Usage Policy запрет",
                fact_id=1,
                document_id=9509,
                role="primary",
            ),
        ],
        used_document_ids=[9509, 9685],
    )
    old = next(anchor for anchor in draft.source_anchors if anchor.phrase == phrase)
    assert (old.fact_id, old.document_id) == (4, 9685)
    assert numeric_anchor_errors(draft, snapshot, docs)

    repaired = repair_deep_article_metadata(draft, snapshot=snapshot, documents=docs)

    anchor = next(anchor for anchor in repaired.source_anchors if anchor.phrase == phrase)
    assert (anchor.fact_id, anchor.document_id, anchor.role) == (3, 9509, "primary")
    assert repaired.lead == draft.lead
    assert [p.text for s in repaired.sections for p in s.paragraphs] == [
        p.text for s in draft.sections for p in s.paragraphs
    ]
    assert numeric_anchor_errors(repaired, snapshot, docs) == []
