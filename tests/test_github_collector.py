from pathlib import Path

from app.collectors.github import GitHubReleasesCollector, release_text, release_urls
from app.config import Settings, SourceKind, load_source_registry
from app.db.repository import Repository

PROJECT_ROOT = Path(__file__).parents[1]


def sample_release(release_id: int = 101) -> dict:
    return {
        "id": release_id,
        "html_url": "https://github.com/QwenLM/qwen-code/releases/tag/v1.2.0",
        "tag_name": "v1.2.0",
        "name": "Qwen Code 1.2",
        "body": (
            "This release adds a long-running agent mode and a new context window. "
            "See https://qwen.ai/docs?utm_source=github for the migration guide."
        ),
        "draft": False,
        "prerelease": False,
        "published_at": "2026-08-13T09:30:00Z",
        "author": {"login": "qwen-bot"},
    }


def test_release_normalization_preserves_primary_release_url() -> None:
    source = next(
        source
        for source in load_source_registry(
            PROJECT_ROOT / "tests/fixtures/config/sources.yaml"
        ).sources
        if source.id == "github_qwen_code"
    )
    release = sample_release()
    source_url = release["html_url"]

    assert release_text(source, release).startswith(
        "Qwen Code Releases: Qwen Code 1.2 (v1.2.0) — Release"
    )
    assert release_urls(release, source_url) == ["https://qwen.ai/docs"]


def test_github_collection_skips_drafts_and_is_idempotent(tmp_path: Path) -> None:
    settings = Settings(
        app_env="test",
        data_dir=tmp_path,
        sources_file=PROJECT_ROOT / "tests/fixtures/config/sources.yaml",
        interests_file=PROJECT_ROOT / "tests/fixtures/config/interests.yaml",
        telegram_session_path=tmp_path / "telegram-reader.session",
    )
    repository = Repository(settings.database_path)
    repository.migrate()
    registry = load_source_registry(settings.sources_file)
    repository.sync_sources(registry)
    draft = sample_release(102) | {"draft": True}
    collector = GitHubReleasesCollector(
        settings,
        repository,
        fetcher=lambda source, limit: [sample_release(), draft],
    )

    first = collector.collect(limit_per_source=5)
    second = collector.collect(limit_per_source=5)
    github_sources = [source for source in registry.sources if source.kind == SourceKind.GITHUB]

    assert first.sources_attempted == len(github_sources) == 2
    assert first.items_seen == 2
    assert first.items_inserted == 2
    assert second.items_inserted == 0
    assert first.cost_usd == 0
    assert repository.current_month_cost() == 0
