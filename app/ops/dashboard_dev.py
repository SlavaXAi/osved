"""Isolated, loopback-only OSVED dashboard; never starts production workers."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shlex
import sqlite3
import subprocess
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from app.config import Settings
from app.db.repository import Repository

DEFAULT_DIRECTORY = Path("data/dashboard-dev")
REMOTE_DB = os.environ.get("OSVED_REMOTE_DB", "/opt/osved/data/newsfeed.sqlite3")
REMOTE_REPORTS = os.environ.get("OSVED_REMOTE_REPORTS", "/opt/osved/data/video-reports")


def snapshot(directory: Path, *, source: Path | None = None, ssh: str | None = None) -> Path:
    """Online backup of a read-only source. Refuse to replace local development state."""
    if bool(source) == bool(ssh):
        raise ValueError("Choose exactly one local source or SSH host")
    target = directory / "newsfeed.sqlite3"
    if target.exists():
        raise FileExistsError("Snapshot already exists; choose another --data-dir to refresh")
    if source and not source.is_file():
        raise FileNotFoundError(source)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(directory, 0o700)
    # Create privately before sqlite opens it. An incomplete copy is never served.
    staging = directory / "snapshot-incomplete.sqlite3"
    with staging.open("xb"):
        os.chmod(staging, 0o600)
    if source:
        with (
            closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)) as origin,
            closing(sqlite3.connect(staging)) as destination,
        ):
            origin.backup(destination)
    else:
        if not ssh or ssh.startswith("-"):
            raise ValueError("Invalid SSH host")
        # No remote temporary files or commands that mutate the source database.
        code = (
            "import sqlite3,sys; "
            f"s=sqlite3.connect('file:{REMOTE_DB}?mode=ro',uri=True); "
            "d=sqlite3.connect(':memory:'); s.backup(d); "
            "sys.stdout.buffer.write(d.serialize()); d.close(); s.close()"
        )
        with staging.open("wb") as output:
            subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", ssh,
                 "python3 -c " + shlex.quote(code)],
                stdout=output, check=True, timeout=180,
            )
    with closing(sqlite3.connect(staging)) as check:
        check.execute("PRAGMA journal_mode=DELETE")
        if check.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("Snapshot integrity check failed")
        count = check.execute("SELECT count(*) FROM cards").fetchone()[0]
    staging.rename(target)
    now = datetime.now(UTC).isoformat()
    metadata = {
        "created_at": now,
        "snapshot_at": now,
        "source": "server_readonly_backup" if ssh else "local_readonly_backup",
        "card_count": count,
        "isolated": True,
        "production_sync": False,
    }
    meta = directory / "snapshot.json"
    with meta.open("x", encoding="utf-8") as output:
        os.chmod(meta, 0o600)
        json.dump(metadata, output, ensure_ascii=False, indent=2)
    return target


def copy_report_cache(directory: Path, ssh: str) -> None:
    if not ssh or ssh.startswith("-"):
        raise ValueError("Invalid SSH host")
    if not (directory / "newsfeed.sqlite3").is_file():
        raise ValueError("Create the isolated snapshot first")
    if (directory / "video-reports").exists():
        raise FileExistsError("Report cache already exists; it will not be overwritten")
    subprocess.run(
        ["rsync", "-az", "--partial", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=10",
         f"{ssh}:{REMOTE_REPORTS}/", str(directory / "video-reports") + "/"],
        check=True, timeout=300,
    )


def development_settings(directory: Path, *, allow_paid: bool = False) -> Settings:
    directory = directory.resolve()
    marker = directory / "snapshot.json"
    if not marker.is_file() or not (directory / "newsfeed.sqlite3").is_file():
        raise ValueError("Refusing to serve a non-snapshot database; run snapshot first")
    metadata = json.loads(marker.read_text())
    if metadata.get("isolated") is not True:
        raise ValueError("Snapshot must be marked isolated")
    settings = Settings(_env_file=".env" if allow_paid else None)
    # Even a developer shell with production environment variables cannot enable bots.
    updates = {
        "app_env": "development", "data_dir": directory,
        "telegram_bot_token": None, "telegram_api_id": None, "telegram_api_hash": None,
        "telegram_session_path": directory / "unused-telegram.session",
    }
    if not allow_paid:
        updates.update({"openrouter_api_key": None, "monid_api_key": None})
    return settings.model_copy(update=updates)


def session_token(directory: Path) -> str:
    path = directory / "dashboard.token"
    if path.exists():
        os.chmod(path, 0o600)
        return path.read_text().strip()
    token = secrets.token_urlsafe(32)
    with path.open("x", encoding="utf-8") as output:
        os.chmod(path, 0o600)
        output.write(token)
    return token


def serve(
    directory: Path, *, port: int = 4188, allow_paid: bool = False, paid_budget: float = 2.0,
) -> None:
    from aiohttp import web

    from app.web import create_app

    settings = development_settings(directory, allow_paid=allow_paid)
    repository = Repository(settings.database_path)
    repository.migrate()
    token = session_token(directory)
    application = create_app(
        settings, repository, access_token=token, allow_paid=allow_paid,
        static_dir=Path(__file__).resolve().parents[1] / "web" / "static",
        paid_budget_usd=paid_budget,
    )
    print(f"OSVED local dashboard: http://127.0.0.1:{port}/")
    print(f"Local login key: {directory / 'dashboard.token'} (private file)")
    print("Telegram polling, collectors and production writes are not started.")
    if allow_paid:
        print("Explicit local paid requests enabled. Snapshot ledger is not live VPS accounting.")
    web.run_app(application, host="127.0.0.1", port=port, access_log=None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DIRECTORY)
    commands = parser.add_subparsers(dest="command", required=True)
    pull = commands.add_parser("snapshot")
    group = pull.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", type=Path)
    group.add_argument("--ssh")
    pull.add_argument("--reports", action="store_true")
    run = commands.add_parser("serve")
    run.add_argument("--port", type=int, default=4188)
    run.add_argument("--allow-paid", action="store_true")
    run.add_argument("--paid-budget", type=float, default=2.0)
    args = parser.parse_args()
    if args.command == "snapshot":
        target = snapshot(args.data_dir, source=args.source, ssh=args.ssh)
        if args.reports and args.ssh:
            copy_report_cache(args.data_dir, args.ssh)
        print(f"Private isolated snapshot ready: {target}")
    else:
        serve(
            args.data_dir, port=args.port,
            allow_paid=args.allow_paid, paid_budget=args.paid_budget,
        )


if __name__ == "__main__":
    main()
