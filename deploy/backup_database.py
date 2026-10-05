"""Low-memory, online SQLite backup. No Telegram/LLM imports or network calls.

Run on the host with Python's standard library. Backups stay on this VPS;
they protect against accidental database damage, not loss of the whole server.
No automatic deletion: retention can be configured after measuring growth.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path


def backup(source: Path, output_dir: Path) -> Path:
    source = source.resolve(strict=True)
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    temporary = output_dir / f"osved-{stamp}.partial"
    target = output_dir / f"osved-{stamp}.sqlite3"
    # Create with private permissions from the first byte, not just after completion.
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        with (
            sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as origin,
            sqlite3.connect(temporary) as destination,
        ):
            origin.backup(destination, pages=256, sleep=0.05)
            if destination.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("Backup integrity check failed")
            if destination.execute("PRAGMA foreign_key_check").fetchone() is not None:
                raise RuntimeError("Backup foreign key check failed")
        temporary.rename(target)
    except Exception:
        # Failed partials are retained for diagnosis; never reported as a backup.
        raise
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    target = backup(args.source, args.output_dir)
    print(json.dumps({"backup": str(target), "bytes": target.stat().st_size, "integrity": "ok"}))


if __name__ == "__main__":
    main()
