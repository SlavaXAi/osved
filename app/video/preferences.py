"""Explicit channel preference, shared by every upload and persisted across restarts."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from urllib.parse import parse_qs, urlsplit

from app.db.repository import Repository


def channel_identity(item: dict) -> str:
    try:
        metadata = json.loads(item.get("media_metadata_json") or "{}")
    except (ValueError, TypeError):
        metadata = {}
    metadata = metadata if isinstance(metadata, dict) else {}
    for value in (item.get("channel_id"), metadata.get("channel_id"), item.get("source_locator")):
        value = str(value or "")
        if re.fullmatch(r"UC[A-Za-z0-9_-]{22}", value):
            return value
        if "://" in value:
            parsed = urlsplit(value)
            channel = (parse_qs(parsed.query).get("channel_id") or [""])[0]
            if parsed.path.startswith("/channel/"):
                channel = parsed.path.split("/")[2]
            if re.fullmatch(r"UC[A-Za-z0-9_-]{22}", channel):
                return channel
    return "source:" + str(item["source_id"])


def preference(repository: Repository, chat_id: int, channel_id: str) -> int:
    with repository.connection() as db:
        row = db.execute(
            "SELECT preference FROM youtube_channel_preferences WHERE chat_id=? AND channel_id=?",
            (chat_id, channel_id),
        ).fetchone()
    return int(row[0]) if row else 0


def save_preference(repository, *, chat_id, channel_id, value, recommendation_id):
    if value not in {-1, 1}:
        raise ValueError("Expected more or less")
    stamp = datetime.now(UTC).isoformat()
    with repository.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        old = db.execute(
            "SELECT preference FROM youtube_channel_preferences WHERE chat_id=? AND channel_id=?",
            (chat_id, channel_id),
        ).fetchone()
        if old and old[0] == value:
            return False
        db.execute(
            "INSERT INTO youtube_channel_preferences VALUES(?,?,?,?) "
            "ON CONFLICT(chat_id,channel_id) DO UPDATE SET "
            "preference=excluded.preference,updated_at=excluded.updated_at",
            (chat_id, channel_id, value, stamp),
        )
        db.execute(
            "INSERT INTO video_feedback_events(recommendation_id,action,created_at) VALUES(?,?,?)",
            (recommendation_id, "channel_more" if value > 0 else "channel_less", stamp),
        )
    return True
