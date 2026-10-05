from __future__ import annotations

import json
import sqlite3
import stat
from pathlib import Path

import pytest

from app.ops.dashboard_dev import development_settings, session_token, snapshot


def source_database(tmp_path: Path) -> Path:
    source = tmp_path / "source.sqlite3"
    with sqlite3.connect(source) as db:
        db.execute("CREATE TABLE cards(id INTEGER PRIMARY KEY)")
        db.execute("INSERT INTO cards VALUES (1)")
    return source


def test_snapshot_is_online_backup_and_source_is_unchanged(tmp_path):
    source = source_database(tmp_path)
    before = source.read_bytes()
    directory = tmp_path / "dev"
    result = snapshot(directory, source=source)
    assert source.read_bytes() == before
    assert stat.S_IMODE(result.stat().st_mode) == 0o600
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    with sqlite3.connect(result) as db:
        assert db.execute("SELECT count(*) FROM cards").fetchone()[0] == 1
        db.execute("DELETE FROM cards")
    assert source.read_bytes() == before
    marker = json.loads((directory / "snapshot.json").read_text())
    assert marker["isolated"] and not marker["production_sync"]
    assert marker["snapshot_at"]
    with pytest.raises(FileExistsError):
        snapshot(directory, source=source)


def test_dashboard_refuses_non_snapshot_and_strips_provider_and_bot_keys(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="non-snapshot"):
        development_settings(tmp_path)
    source = source_database(tmp_path)
    directory = tmp_path / "dev"
    snapshot(directory, source=source)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "private-bot-token")
    monkeypatch.setenv("MONID_API_KEY", "private-monid-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "private-router-key")
    settings = development_settings(directory)
    assert settings.data_dir == directory.resolve()
    assert settings.telegram_bot_token is None
    assert settings.monid_api_key is None
    assert settings.openrouter_api_key is None


def test_snapshot_requires_one_source_and_private_stable_token(tmp_path):
    with pytest.raises(ValueError):
        snapshot(tmp_path)
    with pytest.raises(ValueError):
        snapshot(tmp_path, source=Path("x"), ssh="pars")
    value = session_token(tmp_path)
    assert len(value) >= 32
    assert session_token(tmp_path) == value
    assert stat.S_IMODE((tmp_path / "dashboard.token").stat().st_mode) == 0o600
