from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import time
from enum import StrEnum
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.contours.models import ContourRegistry
from app.frontier_ideas.config import FrontierIdeasRegistry


class SourceKind(StrEnum):
    TELEGRAM = "telegram"
    X = "x"
    RSS = "rss"
    GITHUB = "github"
    WEB = "web"
    REDDIT = "reddit"
    YOUTUBE = "youtube"
    V2EX = "v2ex"
    WEIBO = "weibo"
    ZHIHU = "zhihu"


class SourceRole(StrEnum):
    OWN_HISTORY = "own_history"
    RADAR = "radar"
    AUTHOR_RADAR = "author_radar"
    PRIMARY = "primary"
    LOCAL_MEDIA = "local_media"
    INTERPRETER = "interpreter"


class DeliveryFormat(StrEnum):
    REGULAR = "regular"
    ARTICLE = "article"


class XAuthorTier(StrEnum):
    STREAM = "stream"
    ACTIVE = "active"
    SOSVEDE = "sparse"


class SourceConfig(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    name: str
    kind: SourceKind
    locator: str
    role: SourceRole
    enabled: bool = True
    candidate_enabled: bool = True
    priority: Literal["p0", "p1", "p2"] = "p1"
    region: str
    language: str
    topics: list[str] = Field(default_factory=list)
    include_keywords: list[str] = Field(default_factory=list)
    exclude_keywords: list[str] = Field(default_factory=list)
    poll_interval_minutes: int = Field(default=60, ge=5, le=10080)
    calibration_lookback_days: int = Field(default=21, ge=0, le=90)
    production_max_age_days: int = Field(default=5, ge=1, le=30)
    daily_candidate_cap: int = Field(default=20, ge=0, le=500)
    adapter_id: str = "auto"
    strategy: str = "feed"
    admission_mode: (
        Literal["trigger_only", "event_candidate", "primary_evidence", "history"] | None
    ) = None
    auth_profile_alias: str | None = None
    daily_call_cap: int = Field(default=12, ge=0, le=288)
    monthly_cost_cap_usd: float = Field(default=1.0, ge=0, le=30)
    options: dict[str, str | int | float | bool | list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def resolve_admission(self) -> SourceConfig:
        if self.admission_mode is None:
            self.admission_mode = (
                "history"
                if self.role == SourceRole.OWN_HISTORY
                else "trigger_only"
                if not self.candidate_enabled
                else "primary_evidence"
                if self.role == SourceRole.PRIMARY
                else "event_candidate"
            )
        if self.admission_mode in {"trigger_only", "history"}:
            self.candidate_enabled = False
        return self

    @model_validator(mode="after")
    def own_history_is_not_a_candidate(self) -> SourceConfig:
        if self.role == SourceRole.OWN_HISTORY and self.candidate_enabled:
            raise ValueError("own_history sources must have candidate_enabled=false")
        return self


class SourceRegistry(BaseModel):
    version: int = Field(ge=1)
    sources: list[SourceConfig]

    @model_validator(mode="after")
    def source_ids_are_unique(self) -> SourceRegistry:
        ids = [source.id for source in self.sources]
        if len(ids) != len(set(ids)):
            raise ValueError("source ids must be unique")
        return self

    @property
    def enabled_sources(self) -> list[SourceConfig]:
        return [source for source in self.sources if source.enabled]


class InterestProfile(BaseModel):
    version: int = Field(ge=1)
    vertical: str = "technology"
    production_max_age_days: int = Field(default=5, ge=1, le=30)
    calibration_lookback_days: int = Field(default=21, ge=0, le=90)
    topics: list[str] = Field(default_factory=list)
    regions_of_interest: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    positive_patterns: list[str] = Field(default_factory=list)
    avoid_patterns: list[str] = Field(default_factory=list)
    current_projects: list[str] = Field(default_factory=list)
    exploration_fraction: float = Field(default=0.15, ge=0, le=0.5)


class SearchAgentLens(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    title: str
    enabled: bool = True
    prompt_version: str
    instructions: str = Field(min_length=40)
    topics: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)
    max_events: int = Field(default=4, ge=1, le=8)


class SearchAgentRegistry(BaseModel):
    version: int = Field(ge=1)
    common_prompt_version: str
    lenses: list[SearchAgentLens]

    @model_validator(mode="after")
    def lens_ids_are_unique(self) -> SearchAgentRegistry:
        ids = [lens.id for lens in self.lenses]
        if len(ids) != len(set(ids)):
            raise ValueError("search-agent lens ids must be unique")
        return self

    @property
    def enabled_lenses(self) -> list[SearchAgentLens]:
        return [lens for lens in self.lenses if lens.enabled]


class XAuthorConfig(BaseModel):
    handle: str = Field(pattern=r"^[A-Za-z0-9_]{1,15}$")
    name: str
    topics: list[str] = Field(default_factory=list)
    initial_tier: XAuthorTier
    enabled: bool = True
    weekly_persona: bool = False


class XAuthorRegistry(BaseModel):
    version: int = Field(ge=1)
    authors: list[XAuthorConfig]

    @model_validator(mode="after")
    def handles_are_unique(self) -> XAuthorRegistry:
        handles = [author.handle.casefold() for author in self.authors]
        if len(handles) != len(set(handles)):
            raise ValueError("X author handles must be unique (case-insensitive)")
        return self

    @property
    def enabled_authors(self) -> list[XAuthorConfig]:
        return [author for author in self.authors if author.enabled]


class WeeklyPersonaSourceStrategy(BaseModel):
    x_handles: list[str] = Field(default_factory=list)
    include_interviews: bool = True
    official_transcript_sources: list[str] = Field(default_factory=list)


class WeeklyPersonaConfig(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    name: str
    role: str
    topics: list[str] = Field(default_factory=list)
    source_strategy: WeeklyPersonaSourceStrategy
    skip_if_no_material_update: bool = True


class WeeklyPersonaRegistry(BaseModel):
    version: int = Field(ge=1)
    personas: list[WeeklyPersonaConfig]

    @model_validator(mode="after")
    def persona_ids_are_unique(self) -> WeeklyPersonaRegistry:
        ids = [persona.id for persona in self.personas]
        if len(ids) != len(set(ids)):
            raise ValueError("weekly-persona ids must be unique")
        return self


X_TIER_POLL_INTERVAL_MINUTES: dict[XAuthorTier, int] = {
    # TikHub returns roughly twenty timeline rows per call. Start conservatively:
    # fast news accounts twice a day, regular analysts every two days, and rare
    # high-signal people weekly. Runtime observations can move an author later.
    XAuthorTier.STREAM: 720,
    XAuthorTier.ACTIVE: 2880,
    XAuthorTier.SOSVEDE: 10080,
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_ignore_empty=True,
        extra="ignore",
    )

    app_env: Literal["development", "test", "production"] = "development"
    data_dir: Path = Path("data")
    sources_file: Path = Path("config/sources.yaml")
    interests_file: Path = Path("config/interests.yaml")
    search_agents_file: Path = Path("config/search-agents.yaml")
    contours_file: Path = Path("config/contours.yaml")
    frontier_ideas_file: Path = Path("config/frontier-ideas.yaml")
    x_authors_file: Path = Path("config/x-authors.yaml")
    weekly_personas_file: Path = Path("config/weekly-personas.yaml")
    style_profile_file: Path = Path("config/style-profile.yaml")
    sensory_sources_file: Path = Path("config/sensory-sources.yaml")
    log_level: str = "INFO"

    telegram_bot_token: SecretStr | None = None
    telegram_api_base_url: str | None = None
    media_downloads_enabled: bool = True
    menu_batches_enabled: bool = False
    daily_personalization_enabled: bool = False
    daily_selection_file: Path = Path("config/daily-selection.yaml")
    menu_preparation_max_cost_usd: float = Field(default=0.05, gt=0, le=0.25)
    media_max_file_bytes: int = Field(default=1_900_000_000, ge=1_000_000, le=1_900_000_000)
    media_max_storage_bytes: int = Field(default=4_500_000_000, ge=100_000_000)
    media_min_free_bytes: int = Field(default=1_000_000_000, ge=100_000_000)
    media_timeout_seconds: int = Field(default=1800, ge=30, le=7200)
    allowed_user_id: int | None = None
    telegram_api_id: int | None = None
    telegram_api_hash: SecretStr | None = None
    telegram_session_path: Path = Path("data/telegram-reader.session")
    delivery_format: DeliveryFormat = DeliveryFormat.REGULAR
    news_delivery_mode: Literal["scheduled", "on_rating", "hybrid"] = "hybrid"

    openrouter_api_key: SecretStr | None = None
    openrouter_model: str | None = "google/gemini-2.5-flash-lite"
    openrouter_card_model: str | None = "openai/gpt-5.6-luna"
    openrouter_video_model: str | None = "openai/gpt-4.1-mini"
    openrouter_fallback_model: str | None = "openai/gpt-4.1-mini"
    openrouter_search_model: str = "openai/gpt-5.6-luna"
    monid_api_key: SecretStr | None = None
    youtube_rss_fallback_enabled: bool = False

    # Legacy setting is accepted for compatibility; selection uses relevance, not time.
    youtube_less_interval_hours: int = Field(default=0, ge=0, le=168)
    concept_briefs_enabled: bool = True
    concept_briefs_max_cost_usd: float = Field(default=0.02, ge=0, le=0.05)

    # Explicit owner requests: no count cap; monetary caps are optional.
    openrouter_biography_model: str = "qwen/qwen3.5-flash-02-23"
    biography_enabled: bool = False
    biography_reserve_size: int = Field(default=3, ge=1, le=3)
    biography_max_cost_usd: float = Field(default=0.25, gt=0, le=0.25)
    biography_max_daily_cycles: int = Field(default=3, ge=1, le=3)
    video_reports_enabled: bool = True
    video_report_budget_enabled: bool = False
    video_report_max_cost_usd: float = Field(default=0.50, ge=0, le=1)

    # On-demand deep research is opt-in at the interaction level. It is enabled
    # for the local pilot, but every request remains bounded by these hard caps.
    details_enabled: bool = True
    details_max_daily_new: int = Field(default=5, ge=0, le=100)
    details_max_searches: int = Field(default=2, ge=0, le=5)
    details_max_scrapes: int = Field(default=4, ge=0, le=10)
    details_max_cost_usd: float = Field(default=0.02, ge=0, le=1)
    details_cache_days: int = Field(default=7, ge=1, le=90)
    details_worker_concurrency: int = Field(default=1, ge=1, le=4)
    details_worker_lease_minutes: int = Field(default=10, ge=1, le=60)

    news_reader_max_searches: int = Field(default=2, ge=0, le=5)
    news_reader_expanded_max_scrapes: int = Field(default=6, ge=0, le=10)
    news_reader_article_max_scrapes: int = Field(default=8, ge=0, le=12)
    news_reader_expanded_max_cost_usd: float = Field(default=0.05, ge=0, le=1)
    news_reader_article_max_cost_usd: float = Field(default=0.08, ge=0, le=1)

    budget_target_usd: float = Field(default=8.0, ge=0)
    budget_warning_usd: float = Field(default=9.0, ge=0)
    budget_hard_stop_usd: float = Field(default=10.0, ge=0)
    sensory_monthly_cap_usd: float = Field(default=5.0, ge=0, le=30)
    sensory_daily_cap_usd: float = Field(default=0.30, ge=0, le=5)
    scheduled_search_enabled: bool = True
    scheduled_priority_lenses: list[str] = Field(default_factory=list)
    radar_collect_limit: int = Field(default=10, ge=1, le=100)
    schedule_timezone: str = "Europe/Moscow"
    radar_times: list[str] = Field(
        default_factory=lambda: ["10:00", "12:00", "14:00", "16:00", "18:00", "20:00"]
    )
    research_times: list[str] = Field(default_factory=lambda: ["11:00", "15:00", "19:00"])
    delivery_times: list[str] = Field(default_factory=lambda: ["12:30", "16:30", "20:00"])
    scheduled_search_max_agents: int = Field(default=2, ge=1, le=8)
    scheduled_search_max_cost_usd: float = Field(default=0.06, ge=0, le=1)
    # Three daily deliveries: a target, never a requirement to publish filler.
    scheduled_news_per_batch: int = Field(default=6, ge=1, le=10)
    scheduled_news_delivery_batch_size: int = Field(default=5, ge=1, le=10)
    news_extra_batch_size: int = Field(default=3, ge=1, le=10)
    scheduled_news_buffer_target: int = Field(default=18, ge=1, le=100)
    scheduled_research_max_attempts: int = Field(default=12, ge=1, le=20)
    scheduled_research_max_cost_usd: float = Field(default=0.15, ge=0, le=1)

    @model_validator(mode="after")
    def budgets_are_ordered(self) -> Settings:
        if not (self.budget_target_usd <= self.budget_warning_usd <= self.budget_hard_stop_usd):
            raise ValueError("budget limits must satisfy target <= warning <= hard stop")
        return self

    @model_validator(mode="after")
    def schedule_is_valid(self) -> Settings:
        try:
            ZoneInfo(self.schedule_timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("schedule_timezone must be a valid IANA timezone") from exc
        for times in (self.radar_times, self.research_times, self.delivery_times):
            if len(times) != len(set(times)):
                raise ValueError("schedule times must be unique")
            for value in times:
                if len(value) != 5 or value[2] != ":":
                    raise ValueError("schedule times must use HH:MM")
                time.fromisoformat(value)
        return self

    @property
    def database_path(self) -> Path:
        return self.data_dir / "newsfeed.sqlite3"

    def ensure_runtime_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.telegram_session_path.parent.mkdir(parents=True, exist_ok=True)

    def credential_status(self) -> dict[str, bool]:
        return {
            "telegram_bot": self.telegram_bot_token is not None,
            "telegram_reader": self.telegram_api_id is not None
            and self.telegram_api_hash is not None,
            "openrouter": self.openrouter_api_key is not None,
            "monid": self.monid_api_key is not None,
            "allowed_user": self.allowed_user_id is not None,
        }


def load_source_registry(path: Path) -> SourceRegistry:
    with path.open("r", encoding="utf-8") as source_file:
        payload = yaml.safe_load(source_file)
    return SourceRegistry.model_validate(payload)


def load_interest_profile(path: Path) -> InterestProfile:
    with path.open("r", encoding="utf-8") as interest_file:
        payload = yaml.safe_load(interest_file)
    return InterestProfile.model_validate(payload)


def load_search_agent_registry(path: Path) -> SearchAgentRegistry:
    with path.open("r", encoding="utf-8") as search_file:
        payload = yaml.safe_load(search_file)
    return SearchAgentRegistry.model_validate(payload)


def load_contour_registry(path: Path) -> ContourRegistry:
    with path.open("r", encoding="utf-8") as contour_file:
        payload = yaml.safe_load(contour_file)
    return ContourRegistry.model_validate(payload)


def load_frontier_ideas_registry(path: Path) -> FrontierIdeasRegistry:
    with path.open("r", encoding="utf-8") as frontier_file:
        payload = yaml.safe_load(frontier_file)
    return FrontierIdeasRegistry.model_validate(payload)


def load_x_author_registry(path: Path) -> XAuthorRegistry:
    with path.open("r", encoding="utf-8") as author_file:
        payload = yaml.safe_load(author_file)
    return XAuthorRegistry.model_validate(payload)


def load_weekly_persona_registry(path: Path) -> WeeklyPersonaRegistry:
    with path.open("r", encoding="utf-8") as persona_file:
        payload = yaml.safe_load(persona_file)
    return WeeklyPersonaRegistry.model_validate(payload)


def runtime_source_registry(settings: Settings) -> SourceRegistry:
    """Return configured and owner-added sources without mutating configuration."""
    fixed_registry = load_source_registry(settings.sources_file)
    sensory_sources = (
        load_source_registry(settings.sensory_sources_file).sources
        if settings.sensory_sources_file.exists()
        and (
            "sensory_sources_file" in settings.model_fields_set
            or settings.sources_file.resolve().parent
            == settings.sensory_sources_file.resolve().parent
        )
        else []
    )
    author_registry = load_x_author_registry(settings.x_authors_file)
    x_sources = [
        SourceConfig(
            id=f"x_{author.handle.casefold()}",
            name=f"X — {author.name}",
            kind=SourceKind.X,
            locator=author.handle,
            role=SourceRole.AUTHOR_RADAR,
            enabled=author.enabled,
            # X accounts seed search and validation. Their posts are not trusted
            # as publishable events until corroborated by the search radar.
            candidate_enabled=False,
            priority="p1",
            region="global",
            language="en",
            topics=author.topics,
            poll_interval_minutes=X_TIER_POLL_INTERVAL_MINUTES[author.initial_tier],
            calibration_lookback_days=21,
            production_max_age_days=(8 if author.initial_tier == XAuthorTier.SOSVEDE else 5),
            daily_candidate_cap={
                XAuthorTier.STREAM: 30,
                XAuthorTier.ACTIVE: 12,
                XAuthorTier.SOSVEDE: 5,
            }[author.initial_tier],
            strategy="author_timeline",
            adapter_id="monid_x_author",
        )
        for author in author_registry.authors
    ]
    sources = [*fixed_registry.sources, *x_sources, *sensory_sources]
    # Owner-added channels live in SQLite, not a writable YAML file. Read only:
    # config loading must not create a database or implicitly run migrations.
    if settings.database_path.exists():
        with closing(
            sqlite3.connect(settings.database_path.resolve().as_uri() + "?mode=ro", uri=True)
        ) as db:
            db.row_factory = sqlite3.Row
            if db.execute(
                "SELECT 1 FROM sqlite_master WHERE name='x_profile_subscriptions'"
            ).fetchone():
                for row in db.execute("SELECT * FROM x_profile_subscriptions"):
                    seed = next((s for s in sources if s.id == row["source_id"]), None)
                    if seed:
                        continue  # Configured authors keep their tier and pause.
                    sources.append(
                        SourceConfig(
                            id=row["source_id"],
                            name="X — @" + row["handle"],
                            kind=SourceKind.X,
                            locator=row["handle"],
                            role=SourceRole.AUTHOR_RADAR,
                            candidate_enabled=False,
                            priority="p1",
                            region="global",
                            language="und",
                            topics=[],
                            poll_interval_minutes=720,
                            daily_call_cap=2,
                            daily_candidate_cap=20,
                            adapter_id="monid_x_author",
                            strategy="author_timeline",
                            options={"not_before": row["not_before"]},
                        )
                    )
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='youtube_channels'").fetchone():
                for row in db.execute("SELECT * FROM youtube_channels"):
                    sources = [s for s in sources if s.id != row["source_id"]]
                    sources.append(
                        SourceConfig(
                            id=row["source_id"],
                            name="YouTube — " + row["title"],
                            kind=SourceKind.YOUTUBE,
                            locator=row["channel_id"],
                            role=SourceRole.RADAR,
                            enabled=row["removed_at"] is None,
                            region="global",
                            language="und",
                            topics=[],
                            adapter_id="youtube_rss",
                            strategy="channel_uploads",
                            admission_mode="trigger_only",
                            poll_interval_minutes=240,
                            daily_call_cap=6,
                            daily_candidate_cap=15,
                            options={
                                "not_before": row["not_before"],
                                "removed": row["removed_at"] is not None,
                            },
                        )
                    )
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='telegram_channels'").fetchone():
                for row in db.execute("SELECT * FROM telegram_channels"):
                    seed = next((s for s in sources if s.id == row["source_id"]), None)
                    source = seed or SourceConfig(
                        id=row["source_id"],
                        name=row["title"],
                        kind=SourceKind.TELEGRAM,
                        locator=row["username"],
                        role=SourceRole.RADAR,
                        region="global",
                        language="und",
                        poll_interval_minutes=120,
                        daily_call_cap=6,
                        adapter_id="auto",
                        strategy="feed",
                    )
                    sources = [s for s in sources if s.id != row["source_id"]]
                    sources.append(
                        source.model_copy(
                            update={
                                "name": row["title"],
                                "locator": row["username"],
                                "enabled": row["removed_at"] is None,
                                "options": {
                                    **source.options,
                                    "not_before": row["not_before"],
                                    "removed": row["removed_at"] is not None,
                                },
                            }
                        )
                    )
    return SourceRegistry(
        version=fixed_registry.version,
        sources=sources,
    )
