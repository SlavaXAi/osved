"""Local service health: SQLite readable and the supervised scheduler is ticking."""

from __future__ import annotations

import sqlite3
import sys
from datetime import UTC, datetime

from app.config import Settings


def check() -> bool:
    settings = Settings()
    try:
        with sqlite3.connect(f"file:{settings.database_path}?mode=ro", uri=True, timeout=5) as db:
            row = db.execute(
                "SELECT value FROM runtime_settings WHERE key='service_last_tick'"
            ).fetchone()
        if row is None:
            return False
        last = datetime.fromisoformat(row[0])
        if last.tzinfo is None:
            return False
        # A bounded radar/research call can legitimately take several minutes.
        return 0 <= (datetime.now(UTC) - last).total_seconds() < 900
    except (sqlite3.Error, ValueError, TypeError, OSError):
        return False


if __name__ == "__main__":
    healthy = check()
    print("healthy" if healthy else "unhealthy")
    sys.exit(0 if healthy else 1)
