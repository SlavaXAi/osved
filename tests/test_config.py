from pathlib import Path

from app.config import (
    X_TIER_POLL_INTERVAL_MINUTES,
    DeliveryFormat,
    Settings,
    SourceKind,
    SourceRole,
    XAuthorTier,
    load_contour_registry,
    load_frontier_ideas_registry,
    load_source_registry,
    load_weekly_persona_registry,
    load_x_author_registry,
    runtime_source_registry,
)

PROJECT_ROOT = Path(__file__).parents[1]


def test_telegram_registry_contains_approved_allowlist() -> None:
    registry = load_source_registry(PROJECT_ROOT / "tests/fixtures/config/sources.yaml")
    locators = {source.locator for source in registry.sources if source.kind == SourceKind.TELEGRAM}
    assert locators == {
        "test_history",
        "ict_moscow_ai",
        "blockchainRF",
        "overbafer1",
        "cryptoEssay",
        "prompt_design",
        "ai_newz",
    }


def test_own_channel_is_excluded_from_candidates() -> None:
    registry = load_source_registry(PROJECT_ROOT / "tests/fixtures/config/sources.yaml")
    own = next(source for source in registry.sources if source.role == SourceRole.OWN_HISTORY)
    assert own.locator == "test_history"
    assert own.enabled
    assert not own.candidate_enabled


def test_delivery_format_defaults_to_regular_and_accepts_article() -> None:
    assert Settings(_env_file=None).delivery_format == DeliveryFormat.REGULAR
    assert (
        Settings(_env_file=None, delivery_format="article").delivery_format
        == DeliveryFormat.ARTICLE
    )


def test_x_author_registry_contains_all_pdf_handles() -> None:
    registry = load_x_author_registry(PROJECT_ROOT / "tests/fixtures/config/x-authors.yaml")

    assert len(registry.authors) == 50
    assert len({author.handle.casefold() for author in registry.authors}) == 50
    assert {author.handle for author in registry.authors} >= {
        "karpathy",
        "ilyasut",
        "JensenHuang",
        "sama",
        "rowancheung",
        "ttunguz",
        "ylecun",
        "kaifulee",
    }
    assert "tunguz" not in {author.handle for author in registry.authors}
    assert "yannlecun" not in {author.handle for author in registry.authors}
    assert len(registry.enabled_authors) == 42
    assert {author.handle for author in registry.authors if not author.enabled} == {
        "KirkDBorne",
        "HMsheikh4",
        "PaulYacoubian",
        "antgrasso",
        "cynthiabreazeal",
        "rossdawson",
        "tessalau",
        "v_vashishta",
    }
    assert {author.handle for author in registry.authors if author.weekly_persona} == {
        "JensenHuang",
        "sama",
        "demishassabis",
    }


def test_runtime_registry_adds_enabled_x_authors_with_tier_polling() -> None:
    settings = Settings(
        _env_file=None,
        sources_file=PROJECT_ROOT / "tests/fixtures/config/sources.yaml",
        x_authors_file=PROJECT_ROOT / "tests/fixtures/config/x-authors.yaml",
    )
    fixed = load_source_registry(settings.sources_file)
    authors = load_x_author_registry(settings.x_authors_file)
    runtime = runtime_source_registry(settings)
    x_sources = [
        source
        for source in runtime.sources
        if source.kind == SourceKind.X and source.strategy == "author_timeline"
    ]

    assert {source.id for source in runtime.sources} >= {source.id for source in fixed.sources}
    assert len(runtime.sources) >= len(fixed.sources) + len(authors.authors)
    assert len(x_sources) == 50
    assert all(source.role == SourceRole.AUTHOR_RADAR for source in x_sources)
    assert all(not source.candidate_enabled for source in x_sources)
    assert all(source.id == f"x_{source.locator.casefold()}" for source in x_sources)

    source_by_handle = {source.locator: source for source in x_sources}
    author_by_handle = {author.handle: author for author in authors.authors}
    for handle, source in source_by_handle.items():
        tier = author_by_handle[handle].initial_tier
        assert source.poll_interval_minutes == X_TIER_POLL_INTERVAL_MINUTES[tier]
        assert source.enabled == author_by_handle[handle].enabled

    assert source_by_handle["rowancheung"].poll_interval_minutes == 720
    assert source_by_handle["sama"].poll_interval_minutes == 2880
    assert source_by_handle["ilyasut"].poll_interval_minutes == 10080
    assert source_by_handle["ttunguz"].poll_interval_minutes == 10080
    assert source_by_handle["ylecun"].poll_interval_minutes == 10080
    assert source_by_handle["lexfridman"].poll_interval_minutes == 10080
    assert not source_by_handle["KirkDBorne"].enabled
    assert X_TIER_POLL_INTERVAL_MINUTES[XAuthorTier.SOSVEDE] == 7 * 24 * 60


def test_weekly_persona_registry_has_compact_required_core() -> None:
    registry = load_weekly_persona_registry(
        PROJECT_ROOT / "tests/fixtures/config/weekly-personas.yaml"
    )
    personas = {persona.id: persona for persona in registry.personas}

    assert len(personas) == 11
    assert "liberman_brothers" in personas
    assert personas.keys() >= {
        "elon_musk",
        "sam_altman",
        "jensen_huang",
        "peter_thiel",
        "vladimir_putin",
    }
    assert all(persona.skip_if_no_material_update for persona in personas.values())
    assert personas["vladimir_putin"].source_strategy.official_transcript_sources
    assert personas["elon_musk"].source_strategy.x_handles == ["elonmusk"]


def test_contour_registry_and_details_pilot_defaults() -> None:
    registry = load_contour_registry(PROJECT_ROOT / "tests/fixtures/config/contours.yaml")
    frontier = load_frontier_ideas_registry(
        PROJECT_ROOT / "tests/fixtures/config/frontier-ideas.yaml"
    )
    settings = Settings(_env_file=None)

    assert len(registry.enabled_contours) == 7
    assert frontier.lens.id == "frontier_ideas"
    assert settings.frontier_ideas_file == (
        Path("tests/fixtures/config/frontier-ideas.yaml").resolve()
    )
    assert settings.details_enabled
    assert settings.details_max_daily_new == 5
    assert settings.details_max_cost_usd == 0.02
