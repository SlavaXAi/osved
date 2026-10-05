from __future__ import annotations

import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path


def backup_database(path: Path, output_dir: Path) -> Path | None:
    if not path.exists():
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"newsfeed-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%f')}.sqlite3"
    with (sqlite3.connect(f"file:{path}?mode=ro", uri=True) as source,
          sqlite3.connect(target) as destination):
        source.backup(destination)
        if destination.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("Database backup failed integrity check")
    os.chmod(target, 0o600)
    return target
