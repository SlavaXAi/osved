from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest
import yaml

from app.collectors.base import (
    CollectedItem,
    CollectionBatch,
    SourceCapabilities,
    SourceHealth,
)
from app.collectors.ingest import ingest_item, sanitize_payload
from app.collectors.orchestrator import collect_due_sources
from app.collectors.state import claim_source, finish_source
from app.config import Settings, SourceConfig, SourceRegistry
from app.db.repository import Repository

NOW = datetime(2026, 9, 7, 7, tzinfo=UTC)


def setup(tmp_path):
    source = SourceConfig(
        id="sample",
        name="Sample",
        kind="reddit",
        locator="LocalLLaMA",
        role="radar",
        region="global",
        language="en",
        adapter_id="fake",
        strategy="new",
        admission_mode="trigger_only",
        poll_interval_minutes=60,
        daily_call_cap=3,
    )
    registry = SourceRegistry(version=1, sources=[source])
    source_file = tmp_path / "sources.yaml"
    source_file.write_text(yaml.safe_dump(registry.model_dump(mode="json")))
    authors = tmp_path / "authors.yaml"
    authors.write_text("version: 1\nauthors: []\n")
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        sources_file=source_file,
        x_authors_file=authors,
        sensory_sources_file=tmp_path / "missing.yaml",
    )
    repository = Repository(settings.database_path)
    repository.migrate()
    repository.sync_sources(registry)
    return settings, repository, source


class Adapter:
    def __init__(self, batch=None):
        self.calls = 0
        self.batch = batch or CollectionBatch(
            items=[
                CollectedItem(
                    "p1",
                    "https://www.reddit.com/r/LocalLLaMA/comments/p1/test/",
                    "A new model released with independently reported performance "
                    "and reproducible code.",
                    NOW.isoformat(),
                )
            ]
        )

    def capabilities(self):
        return SourceCapabilities(estimated_cost_usd=0.0015)

    def healthcheck(self, source):
        return SourceHealth()

    def collect(self, request):
        self.calls += 1
        return self.batch


def test_sensor_idempotency_cadence_and_trigger_boundary(tmp_path):
    settings, repo, _ = setup(tmp_path)
    adapter = Adapter()
    first = collect_due_sources(settings, repo, adapters={"fake": adapter}, now=NOW)
    second = collect_due_sources(settings, repo, adapters={"fake": adapter}, now=NOW)
    third = collect_due_sources(
        settings, repo, adapters={"fake": adapter}, now=NOW + timedelta(hours=1)
    )
    assert first["items_inserted"] == 1
    assert second["sources"][0]["status"] == "not_due"
    assert third["items_inserted"] == 0
    assert adapter.calls == 2
    assert len(repo.recent_fixed_signals("2026-09-01")) == 1
    assert repo.unprocessed_source_items()[0]["candidate_enabled"] == 0
    with repo.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
        assert db.execute("SELECT observation_count FROM source_items").fetchone()[0] == 2


def test_global_sensor_budget_reservation_prevents_parallel_overspend(tmp_path):
    settings, repo, source = setup(tmp_path)
    settings.sensory_daily_cap_usd = 0.002
    one, status = claim_source(repo, settings, source, "fake", 0.0015, now=NOW)
    assert one and status == "running"
    assert claim_source(repo, settings, source, "fake", 0.0015, now=NOW)[1] == "running"
    finish_source(repo, one, status="ok", cost_usd=0.0015, now=NOW)
    assert (
        claim_source(repo, settings, source, "fake", 0.0015, now=NOW, force=True)[1]
        == "budget_blocked"
    )


def test_source_error_isolated_and_cooldown_persisted(tmp_path):
    settings, repo, _ = setup(tmp_path)
    adapter = Adapter(CollectionBatch(status="challenge", error="Login challenge"))
    result = collect_due_sources(settings, repo, adapters={"fake": adapter}, now=NOW)
    assert result["errors"] == 1
    retry = collect_due_sources(
        settings, repo, adapters={"fake": adapter}, now=NOW + timedelta(hours=2)
    )
    assert retry["sources"][0]["status"] == "cooldown"
    assert adapter.calls == 1


def test_offline_replay_does_not_consume_network_cadence_or_call_cap(tmp_path):
    settings, repo, source = setup(tmp_path)
    source.daily_call_cap = 1
    with repo.connection() as db:
        db.execute(
            """INSERT INTO source_runs(
                id,source_id,adapter_id,strategy,status,started_at,lease_until)
            VALUES('replay','sample','fake','snapshot_replay','replayed',?,?)""",
            (NOW.isoformat(), NOW.isoformat()),
        )
    run_id, status = claim_source(repo, settings, source, "fake", 0, now=NOW)
    assert run_id and status == "running"
    finish_source(repo, run_id, status="ok", now=NOW)
    assert claim_source(repo, settings, source, "fake", 0, now=NOW)[1] == "not_due"
    assert (
        claim_source(repo, settings, source, "fake", 0, now=NOW, force=True)[1] == "daily_call_cap"
    )


def test_source_toggle_blocks_calls_and_dry_run_has_no_requests(tmp_path):
    settings, repo, _ = setup(tmp_path)
    adapter = Adapter()
    preview = collect_due_sources(settings, repo, adapters={"fake": adapter}, dry_run=True, now=NOW)
    assert preview["sources"][0]["status"] == "planned"
    assert adapter.calls == 0
    repo.set_runtime_setting("source_enabled:sample", "false")
    assert collect_due_sources(settings, repo, adapters={"fake": adapter}, now=NOW)["sources"] == []


def test_input_sanitization_preserves_content_not_credentials(tmp_path):
    _, repo, source = setup(tmp_path)
    raw = {"cookie": "sensitive", "post": {"Authorization": "Bearer sensitive", "text": "content"}}
    assert sanitize_payload(raw) == {"post": {"text": "content"}}
    item = CollectedItem(
        "safe",
        "https://example.com/post?utm_source=x&token=secret",
        "actual content",
        NOW.isoformat(),
        raw_payload=raw,
    )
    assert ingest_item(repo, source, item, now=NOW)
    with repo.connection() as db:
        row = db.execute("SELECT * FROM source_items").fetchone()
        assert "sensitive" not in str(dict(row))
        assert row["source_url"] == "https://example.com/post"
    with pytest.raises(ValueError):
        ingest_item(
            repo,
            source,
            CollectedItem(
                "future",
                "https://example.com/future",
                "body",
                (NOW + timedelta(days=2)).isoformat(),
            ),
            now=NOW,
        )


def test_migrations_are_atomic_and_concurrent(tmp_path):
    path = tmp_path / "concurrent.sqlite3"
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(lambda _: Repository(path).migrate(), range(3)))
    assert sum(bool(x) for x in results) == 1
    assert Repository(path).migrate() == []


def test_failed_migration_rolls_back_schema_and_version(tmp_path):
    import sqlite3

    repo = Repository(tmp_path / "broken.sqlite3")
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "0001_broken.sql").write_text("CREATE TABLE example(id INT);\nINVALID SQL;\n")
    repo.migrations_path = migrations
    with pytest.raises(sqlite3.OperationalError):
        repo.migrate()
    with repo.connection() as db:
        assert db.execute("SELECT 1 FROM sqlite_master WHERE name='example'").fetchone() is None
