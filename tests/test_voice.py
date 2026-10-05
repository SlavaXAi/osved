from pathlib import Path

from app.config import Settings, load_source_registry
from app.db.repository import Repository
from app.pipeline.voice import (
    AuthorDraft,
    AuthorDraftGeneration,
    VoiceParagraph,
    author_draft_errors,
    generate_author_draft,
    render_author_draft_html,
)
from app.research.dossier import build_event_dossier
from app.research.policy import fact_policy_metadata

PROJECT_ROOT = Path(__file__).parents[1]


def setup_voice_card(tmp_path: Path) -> tuple[Settings, Repository, dict]:
    settings = Settings(
        app_env="test",
        data_dir=tmp_path,
        sources_file=PROJECT_ROOT / "tests/fixtures/config/sources.yaml",
        interests_file=PROJECT_ROOT / "tests/fixtures/config/interests.yaml",
        style_profile_file=PROJECT_ROOT / "tests/fixtures/config/style-profile.yaml",
        telegram_session_path=tmp_path / "telegram-reader.session",
    )
    repository = Repository(settings.database_path)
    repository.migrate()
    repository.sync_sources(load_source_registry(settings.sources_file))
    smoke = repository.create_smoke_delivery(123, "https://example.com/release")
    document_id = repository.upsert_research_document(
        canonical_url="https://example.com/release",
        url="https://example.com/release",
        source_class="primary",
        retrieved_at="2026-08-14T00:00:00+00:00",
        expires_at="2099-01-01T00:00:00+00:00",
        content_text="Компания выпустила агентскую систему для длительных задач.",
    )
    repository.attach_research_document(
        event_id=smoke["event_id"],
        research_document_id=document_id,
        evidence_role="primary",
    )
    repository.create_fact_sheet(
        event_id=smoke["event_id"],
        schema_version="neutral_fact_sheet_v2",
        facts=[
            {
                "text": "Компания выпустила агентскую систему для длительных задач.",
                "kind": "release",
                "document_ids": [document_id],
            },
            {
                "text": "Система доступна через программный интерфейс.",
                "kind": "availability",
                "document_ids": [document_id],
            },
            {
                "text": "Один запуск может продолжаться до 8 часов.",
                "kind": "capability",
                "document_ids": [document_id],
            },
        ],
        coverage={
            "event_type": "other",
            "event_identity": True,
            "conflicts": [],
            **fact_policy_metadata(),
        },
        validation_status="valid",
        expires_at="2099-01-01T00:00:00+00:00",
    )
    build_event_dossier(repository, smoke["event_id"])
    return settings, repository, smoke


def valid_author_draft() -> AuthorDraft:
    return AuthorDraft(
        headline="Агентская система берёт задачи продолжительностью до 8 часов",
        headline_evidence_ids=[2],
        paragraphs=[
            VoiceParagraph(
                kind="fact",
                text="Компания выпустила агентскую систему для длительных задач.",
                evidence_ids=[0],
                thesis_id=None,
            ),
            VoiceParagraph(
                kind="fact",
                text=(
                    "Доступ организован через программный интерфейс, один запуск длится до 8 часов."
                ),
                evidence_ids=[1, 2],
                thesis_id=None,
            ),
            VoiceParagraph(
                kind="interpretation",
                text=(
                    "Агенты постепенно превращаются из окна с ответами в отдельные "
                    "производственные единицы."
                ),
                evidence_ids=[],
                thesis_id="agents_as_production",
            ),
        ],
    )


def test_author_draft_is_separate_grounded_output(tmp_path: Path) -> None:
    settings, repository, smoke = setup_voice_card(tmp_path)

    class FakeGenerator:
        def generate(self, **kwargs):
            return AuthorDraftGeneration(
                draft=valid_author_draft(),
                model=kwargs["model"],
                cost_usd=0.002,
                prompt_tokens=200,
                completion_tokens=100,
                response_id="voice-1",
            )

    result = generate_author_draft(
        settings,
        repository,
        smoke["card_id"],
        generator=FakeGenerator(),
    )

    assert result["status"] == "valid"
    stored = repository.latest_author_draft(smoke["card_id"])
    assert stored is not None
    assert stored["style_profile_version"] == "test_voice_v1"
    assert "<a href=" in stored["rendered_html"]
    assert repository.status_counts()["valid_author_drafts"] == 1


def test_voice_validation_blocks_first_person_and_unsupported_numbers() -> None:
    draft = valid_author_draft().model_copy(
        update={
            "paragraphs": [
                VoiceParagraph(
                    kind="fact",
                    text="Я уже использовал систему и считаю, что она экономит 90% времени.",
                    evidence_ids=[0],
                    thesis_id=None,
                ),
                *valid_author_draft().paragraphs[1:],
            ]
        }
    )
    errors = author_draft_errors(
        draft,
        evidence_snippets=[
            {"id": 0, "text": "Компания выпустила систему."},
            {"id": 1, "text": "Доступ идёт через API."},
            {"id": 2, "text": "Один запуск длится до 8 часов."},
        ],
        allowed_thesis_ids={"agents_as_production"},
    )

    assert any("first-person" in error for error in errors)
    assert any("unsupported numeric value 90%" in error for error in errors)


def test_author_renderer_escapes_model_text() -> None:
    draft = valid_author_draft().model_copy(
        update={"headline": "Новая <агентская> система работает до 8 часов"}
    )
    rendered = render_author_draft_html(draft, "https://example.com/?a=1&b=2")

    assert "&lt;агентская&gt;" in rendered
    assert "a=1&amp;b=2" in rendered
    assert "<i>Агенты постепенно превращаются" in rendered
