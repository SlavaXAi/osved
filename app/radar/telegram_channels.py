"""Public Telegram channels explicitly selected by the owner."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from urllib.parse import unquote, urlsplit

from app.config import Settings, SourceKind, SourceRole, runtime_source_registry
from app.db.repository import Repository

USERNAME = re.compile(r"[a-zA-Z][a-zA-Z0-9_]{3,31}")
RESERVED = {"joinchat", "addstickers", "addemoji", "share", "proxy", "socks", "login", "boost"}


class TelegramChannelError(ValueError):
    pass


def channel_username(value: str) -> str:
    value = value.strip()
    if value.startswith("@"):
        name = value[1:]
    else:
        if value.startswith(("t.me/", "telegram.me/", "www.t.me/")):
            value = "https://" + value
        try:
            parsed = urlsplit(value)
            if (
                parsed.scheme not in {"http", "https"}
                or parsed.hostname not in {"t.me", "www.t.me", "telegram.me", "www.telegram.me"}
                or parsed.username
                or parsed.password
                or parsed.port
            ):
                raise ValueError
            parts = unquote(parsed.path).strip("/").split("/")
            if parts[0] == "s":
                parts = parts[1:]
            if not parts or len(parts) > 2 or (len(parts) == 2 and not parts[1].isdigit()):
                raise ValueError
            name = parts[0]
        except ValueError as exc:
            raise TelegramChannelError("Пришли публичную ссылку t.me/имя или @имя канала.") from exc
    if not USERNAME.fullmatch(name) or name.casefold() in RESERVED:
        raise TelegramChannelError("Нужен публичный канал с @именем. Закрытые ссылки не подойдут.")
    return name.casefold()


def channel_identity(chat) -> tuple[int, str, str]:
    if chat.type != "channel" or not chat.username:
        raise TelegramChannelError("Нужен публичный Telegram-канал, у которого есть @имя.")
    if not isinstance(chat.id, int) or chat.id >= 0:
        raise TelegramChannelError("Не удалось подтвердить канал.")
    return chat.id, channel_username("@" + chat.username), (chat.title or chat.username)[:180]


def add_channel(settings: Settings, repository: Repository, chat, user_id: int) -> dict:
    """Persist a Bot API-verified identity. Repeated confirmations are idempotent."""
    chat_id, username, title = channel_identity(chat)
    registry = runtime_source_registry(settings)
    seed = next(
        (
            s
            for s in registry.sources
            if s.kind == SourceKind.TELEGRAM and s.locator.lstrip("@").casefold() == username
        ),
        None,
    )
    if seed and seed.role == SourceRole.OWN_HISTORY:
        raise TelegramChannelError("Этот канал подключён как личный архив. Его роль сохранена.")
    stamp = datetime.now(UTC)
    with repository.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        old = db.execute(
            "SELECT * FROM telegram_channels WHERE chat_id=? OR username=?",
            (chat_id, username),
        ).fetchone()
        if old and old["chat_id"] not in {None, chat_id}:
            raise TelegramChannelError(
                "У этого @имени изменился владелец. Сначала удали старый канал."
            )
        if old and old["removed_at"] is None:
            db.execute(
                "UPDATE telegram_channels SET username=?,title=? WHERE source_id=?",
                (username, title, old["source_id"]),
            )
            result = dict(old, username=username, title=title, existing=True)
        elif seed and not old and seed.enabled:
            # A fixed source is already followed. Do not silently resume a paused source.
            result = {"source_id": seed.id, "username": username, "title": title, "existing": True}
        else:
            source_id = old["source_id"] if old else seed.id if seed else f"tg_user_{abs(chat_id)}"
            db.execute(
                "INSERT INTO telegram_channels"
                "(source_id,chat_id,username,title,added_by,added_at,not_before) "
                "VALUES(?,?,?,?,?,?,?) "
                "ON CONFLICT(source_id) DO UPDATE SET chat_id=excluded.chat_id,"
                "username=excluded.username,title=excluded.title,removed_at=NULL,"
                "not_before=excluded.not_before",
                (
                    source_id,
                    chat_id,
                    username,
                    title,
                    user_id,
                    stamp.isoformat(),
                    (stamp - timedelta(days=1)).isoformat(),
                ),
            )
            db.execute(
                "INSERT INTO runtime_settings(key,value,updated_at) VALUES(?,'true',?) "
                "ON CONFLICT(key) DO UPDATE SET value='true',updated_at=excluded.updated_at",
                ("source_enabled:" + source_id, stamp.isoformat()),
            )
            result = {
                "source_id": source_id,
                "username": username,
                "title": title,
                "existing": False,
            }
    repository.sync_sources(runtime_source_registry(settings))
    override = repository.get_runtime_setting("source_enabled:" + result["source_id"])
    result["enabled"] = override != "false"
    return result


def channel_rows(settings: Settings, repository: Repository) -> list[dict]:
    registry = runtime_source_registry(settings)
    repository.sync_sources(registry)
    result = []
    for source in registry.sources:
        if (
            source.kind != SourceKind.TELEGRAM
            or source.role == SourceRole.OWN_HISTORY
            or source.options.get("removed")
        ):
            continue
        override = repository.get_runtime_setting("source_enabled:" + source.id)
        result.append(
            {
                "source_id": source.id,
                "username": source.locator.lstrip("@"),
                "title": source.name.removeprefix("Telegram — "),
                "enabled": override == "true" if override is not None else source.enabled,
            }
        )
    return sorted(result, key=lambda r: (r["title"].casefold(), r["source_id"]))


def change_channel(settings, repository, source_id, action, user_id):
    row = next((r for r in channel_rows(settings, repository) if r["source_id"] == source_id), None)
    if row is None or action not in {"pause", "resume", "remove"}:
        raise TelegramChannelError("Канал уже удалён или кнопка устарела.")
    stamp = datetime.now(UTC).isoformat()
    with repository.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        if action == "remove":
            db.execute(
                "INSERT INTO telegram_channels"
                "(source_id,username,title,added_by,added_at,not_before,removed_at) "
                "VALUES(?,?,?,?,?,?,?) "
                "ON CONFLICT(source_id) DO UPDATE SET removed_at=excluded.removed_at",
                (source_id, row["username"], row["title"], user_id, stamp, stamp, stamp),
            )
            db.execute("UPDATE sources SET enabled=0 WHERE id=?", (source_id,))
        db.execute(
            "INSERT INTO runtime_settings(key,value,updated_at) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at",
            ("source_enabled:" + source_id, "true" if action == "resume" else "false", stamp),
        )
