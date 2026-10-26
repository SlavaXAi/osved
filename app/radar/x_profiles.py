"""Explicit owner subscriptions to X profiles. Adding never calls a paid API."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from urllib.parse import urlsplit

from app.config import SourceKind, runtime_source_registry

HOSTS = {"x.com", "www.x.com", "twitter.com", "www.twitter.com", "mobile.twitter.com"}
RESERVED = {
    "i",
    "home",
    "explore",
    "search",
    "settings",
    "login",
    "logout",
    "signup",
    "messages",
    "notifications",
    "compose",
    "intent",
    "share",
    "tos",
    "privacy",
    "jobs",
    "communities",
    "hashtag",
    "about",
    "help",
    "download",
    "account",
}
HANDLE = re.compile(r"[A-Za-z0-9_]{1,15}")


def profile_handle(value: str) -> str:
    value = value.strip()
    if value.startswith("@"):
        handle = value[1:]
    else:
        if re.match(r"(?:www\.|mobile\.)?(?:x|twitter)\.com/", value, re.I):
            value = "https://" + value
        try:
            p = urlsplit(value)
            if (
                p.scheme not in {"http", "https"}
                or (p.hostname or "").lower() not in HOSTS
                or p.username
                or p.password
                or p.port
            ):
                raise ValueError
            parts = p.path.strip("/").split("/")
            if len(parts) != 1 and not (
                len(parts) == 2 and parts[1] in {"with_replies", "media", "highlights", "articles"}
            ):
                raise ValueError
            handle = parts[0]
        except ValueError as exc:
            raise ValueError("Пришли ссылку на профиль: https://x.com/имя") from exc
    if not HANDLE.fullmatch(handle) or handle.casefold() in RESERVED:
        raise ValueError("Нужна ссылка на аккаунт X, например https://x.com/Polymarket")
    return handle.casefold()


def add_profile(settings, repository, value: str, user_id: int) -> dict:
    handle = profile_handle(value)
    registry = runtime_source_registry(settings)
    seed = next(
        (
            s
            for s in registry.sources
            if s.kind == SourceKind.X and s.locator.lstrip("@").casefold() == handle
        ),
        None,
    )
    stamp = datetime.now(UTC).isoformat()
    with repository.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        old = db.execute(
            "SELECT source_id FROM x_profile_subscriptions WHERE handle=?", (handle,)
        ).fetchone()
        existing = bool(old or seed)
        source_id = old["source_id"] if old else seed.id if seed else "x_" + handle
        # Repeated links preserve the pause, original follow boundary and polling
        # state. Configured authors retain their existing settings and identity.
        db.execute(
            "INSERT OR IGNORE INTO x_profile_subscriptions "
            "(handle,source_id,added_by,added_at,not_before) VALUES(?,?,?,?,?)",
            (handle, source_id, user_id, stamp, stamp),
        )
    registry = runtime_source_registry(settings)
    repository.sync_sources(registry)
    source = next(s for s in registry.sources if s.id == source_id)
    override = repository.get_runtime_setting("source_enabled:" + source_id)
    return dict(
        handle=handle,
        source_id=source_id,
        existing=existing,
        enabled=override == "true" if override is not None else source.enabled,
    )
