"""Operator preparation/status commands; delivery stays with the supervised worker."""

from __future__ import annotations

import argparse
import json

from app.biography.engine import BiographyEngine
from app.biography.storage import BiographyStore
from app.config import Settings
from app.db.repository import Repository


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["status", "prepare", "pause", "resume"])
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--person-id", type=int)
    parser.add_argument("--resume-cycle", type=int)
    args = parser.parse_args()
    settings = Settings()
    repository = Repository(settings.database_path)
    repository.migrate()
    store = BiographyStore(repository)
    store.initialize()
    if args.action in {"pause", "resume"}:
        store.set_enabled(args.action == "resume")
        print(json.dumps({"enabled": store.enabled()}, ensure_ascii=False))
        return
    result = (
        store.status()
        if args.action == "status"
        else BiographyEngine(settings, store).prepare_one(
            daily_limit=6 if args.bootstrap else None,
            kind="bootstrap" if args.bootstrap else "prepare",
            forced_person_id=args.person_id,
            resume_cycle_id=args.resume_cycle,
        )
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
