"""Public channel resolution and durable follows. No cookies, playback or model calls."""

from __future__ import annotations

import fcntl
import hashlib
import json
import re
from datetime import UTC, datetime
from urllib.parse import quote, unquote, urlsplit
from xml.etree import ElementTree

from app.collectors.platform_transport import fetch_public, monid_fetch, unwrap_data
from app.collectors.rss import UNSAFE_XML_RE
from app.config import Settings, runtime_source_registry
from app.db.repository import Repository

ENDPOINT = "/api/v1/youtube/web_v2/get_channel_id"
PRICE = 0.0015
CHANNEL_ID = re.compile(r"UC[A-Za-z0-9_-]{22}")


class ChannelError(ValueError):
    pass


def channel_url(value: str) -> str:
    """Accept channel URLs only; discard tracking queries before external calls."""
    if len(value) > 2048:
        raise ChannelError("Слишком длинная ссылка.")
    if value.startswith(("youtube.com/", "www.youtube.com/", "m.youtube.com/")):
        value = "https://" + value
    p = urlsplit(value)
    if (
        p.scheme not in {"http", "https"}
        or p.hostname not in {"youtube.com", "www.youtube.com", "m.youtube.com"}
        or p.username
        or p.password
        or p.port
    ):
        raise ChannelError("Нужна публичная ссылка на канал YouTube.")
    parts = unquote(p.path).strip("/").split("/")
    if parts and parts[-1] in {"videos", "shorts", "streams", "featured", "podcasts", "about"}:
        parts.pop()
    valid = False
    if len(parts) == 1 and re.fullmatch(r"@[\w.\-·]{1,100}", parts[0]):
        parts[0] = parts[0].casefold()
        valid = True
    elif len(parts) == 2:
        valid = (parts[0] == "channel" and bool(CHANNEL_ID.fullmatch(parts[1]))) or (
            parts[0] in {"c", "user"} and bool(re.fullmatch(r"[\w.\-]{1,100}", parts[1]))
        )
    if not valid:
        raise ChannelError("Нужна ссылка на канал: youtube.com/@имя или /channel/UC…")
    return "https://www.youtube.com/" + "/".join(quote(s, safe="@-._") for s in parts)


def feed_title(payload: bytes, expected_id: str) -> str:
    if UNSAFE_XML_RE.search(payload):
        raise ChannelError("Не удалось проверить ленту канала.")
    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError as exc:
        raise ChannelError("YouTube не вернул ленту канала. Попробуй позже.") from exc
    ns = {"a": "http://www.w3.org/2005/Atom", "yt": "http://www.youtube.com/xml/schemas/2015"}
    if root.tag != "{http://www.w3.org/2005/Atom}feed":
        raise ChannelError("Не удалось подтвердить канал по публичной ленте.")
    actual = root.findtext("yt:channelId", "", ns)
    # YouTube's Atom channelId is sometimes the UC suffix rather than the full ID.
    if actual not in {expected_id, expected_id[2:]}:
        raise ChannelError("YouTube вернул ленту другого канала; добавление отменено.")
    title = " ".join(root.findtext("a:title", "", ns).split())[:180]
    if not title:
        raise ChannelError("В ленте нет названия канала.")
    return title


class ChannelService:
    def __init__(
        self, settings: Settings, repository: Repository, *, fetcher=fetch_public, paid=None
    ):
        self.settings, self.repository, self.fetcher = settings, repository, fetcher
        self.paid = paid or (
            lambda url: monid_fetch(
                ENDPOINT, {"channel_url": url}, settings.data_dir / "channel-resolutions"
            )
        )

    def resolve(self, url: str) -> str:
        direct = url.rsplit("/", 1)[-1] if "/channel/" in url else ""
        if CHANNEL_ID.fullmatch(direct):
            return direct
        with self.repository.connection() as db:
            old = db.execute("SELECT * FROM youtube_channel_urls WHERE url=?", (url,)).fetchone()
            if old:
                if old["status"] == "ready":
                    return old["channel_id"]
                raise ChannelError(
                    "Предыдущий запрос этой ссылки не завершился. "
                    "Повторно не оплачиваю: нужна проверка, либо ссылка /channel/UC…"
                )
            if self.settings.monid_api_key is None:
                raise ChannelError("Для @имени нужен Monid. Можно прислать ссылку /channel/UC…")
            db.execute("BEGIN IMMEDIATE")
            count = db.execute(
                "SELECT COUNT(*) FROM youtube_channel_urls WHERE date(attempted_at)=date('now')"
            ).fetchone()[0]
            cost = db.execute(
                "SELECT COALESCE(SUM(cost_usd),0) FROM usage_ledger "
                "WHERE strftime('%Y-%m',created_at)=strftime('%Y-%m','now')"
            ).fetchone()[0]
            if count >= 30 or cost + PRICE >= self.settings.budget_hard_stop_usd:
                raise ChannelError("Достигнут лимит расходов или 30 распознаваний каналов за день.")
            usage = db.execute(
                "INSERT INTO usage_ledger(provider,operation,request_count,cost_usd,metadata_json) "
                "VALUES('monid','youtube.channel.resolve',1,?,?)",
                (PRICE, json.dumps({"url": url, "reserved": True})),
            ).lastrowid
            db.execute(
                "INSERT INTO youtube_channel_urls(url,status,attempted_at,usage_id) "
                "VALUES(?,'resolving',?,?)",
                (url, datetime.now(UTC).isoformat(), usage),
            )
        try:
            result = self.paid(url)
            payload = unwrap_data(result.payload)
            channel = payload.get("channel_id", "") if isinstance(payload, dict) else ""
            if not CHANNEL_ID.fullmatch(channel):
                raise ChannelError("Провайдер не определил ID канала.")
            with self.repository.connection() as db:
                db.execute(
                    "UPDATE youtube_channel_urls SET status='ready',channel_id=?,run_id=? "
                    "WHERE url=?",
                    (channel, result.run_id, url),
                )
                db.execute(
                    "UPDATE usage_ledger SET cost_usd=?,metadata_json=? WHERE id=?",
                    (result.cost_usd, json.dumps({"url": url, "run_id": result.run_id}), usage),
                )
            return channel
        except Exception as exc:
            with self.repository.connection() as db:
                db.execute("UPDATE youtube_channel_urls SET status='failed' WHERE url=?", (url,))
            raise ChannelError(
                "Не удалось распознать канал. Запрос сохранён, "
                "платного автоповтора не будет. Можно прислать /channel/UC…"
            ) from exc

    def add(self, value: str, user_id: int) -> dict:
        url = channel_url(value)
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        # Cross-process lock also protects two near-simultaneous Telegram messages.
        with (self.settings.data_dir / "youtube-channels.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            channel = self.resolve(url)
            registry = runtime_source_registry(self.settings)
            self.repository.sync_sources(registry)
            with self.repository.connection() as db:
                old = db.execute(
                    "SELECT * FROM youtube_channels WHERE channel_id=?", (channel,)
                ).fetchone()
                if old and old["removed_at"] is None:
                    return dict(old, existing=True)
                count = db.execute(
                    "SELECT COUNT(*) FROM youtube_channels WHERE removed_at IS NULL"
                ).fetchone()[0]
                if count >= 100:
                    raise ChannelError(
                        "В радаре уже 100 добавленных каналов. Сначала удали ненужные."
                    )
            # Validate even direct IDs. No private session or video playback involved.
            try:
                title = feed_title(
                    self.fetcher(f"https://www.youtube.com/feeds/videos.xml?channel_id={channel}"),
                    channel,
                )
            except ChannelError:
                raise
            except Exception as exc:
                raise ChannelError(
                    "YouTube пока не отдал публичную ленту. Канал не добавлен; "
                    "попробуй ссылку позже, повторного распознавания не потребуется."
                ) from exc
            seed = next(
                (s for s in registry.sources if s.kind.value == "youtube" and s.locator == channel),
                None,
            )
            source_id = (
                seed.id
                if seed
                else "youtube_user_" + hashlib.sha256(channel.encode()).hexdigest()[:20]
            )
            stamp = datetime.now(UTC).isoformat()
            with self.repository.connection() as db:
                db.execute(
                    "INSERT INTO youtube_channels"
                    "(channel_id,source_id,title,added_by,added_at,not_before) "
                    "VALUES(?,?,?,?,?,?) ON CONFLICT(channel_id) DO UPDATE SET "
                    "removed_at=NULL,title=excluded.title,not_before=excluded.not_before",
                    (channel, source_id, title, user_id, stamp, stamp),
                )
                db.execute(
                    "INSERT INTO runtime_settings(key,value,updated_at) VALUES(?,'true',?) "
                    "ON CONFLICT(key) DO UPDATE SET value='true',updated_at=excluded.updated_at",
                    ("source_enabled:" + source_id, stamp),
                )
            self.repository.sync_sources(runtime_source_registry(self.settings))
            return {
                "source_id": source_id,
                "channel_id": channel,
                "title": title,
                "existing": False,
            }


def channel_rows(settings: Settings, repository: Repository) -> list[dict]:
    repository.sync_sources(runtime_source_registry(settings))
    with repository.connection() as db:
        return [
            dict(r)
            for r in db.execute("""SELECT s.id AS source_id,s.locator AS channel_id,
            s.name AS title,COALESCE((SELECT CASE value WHEN 'true' THEN 1 ELSE 0 END
            FROM runtime_settings WHERE key='source_enabled:'||s.id),s.enabled) AS enabled
            FROM sources s LEFT JOIN youtube_channels y ON y.source_id=s.id
            WHERE s.kind='youtube' AND y.removed_at IS NULL ORDER BY s.name,s.id""")
        ]


def change_channel(
    settings: Settings, repository: Repository, source_id: str, action: str, user_id: int
):
    rows = channel_rows(settings, repository)
    row = next((r for r in rows if r["source_id"] == source_id), None)
    if not row or action not in {"pause", "resume", "remove"}:
        raise ChannelError("Этот канал уже удалён или кнопка устарела.")
    stamp = datetime.now(UTC).isoformat()
    with repository.connection() as db:
        if action == "remove":
            db.execute(
                "INSERT INTO youtube_channels"
                "(channel_id,source_id,title,added_by,added_at,not_before,removed_at) "
                "VALUES(?,?,?,?,?,?,?) ON CONFLICT(channel_id) DO UPDATE SET "
                "removed_at=excluded.removed_at",
                (
                    row["channel_id"],
                    source_id,
                    row["title"].removeprefix("YouTube — "),
                    user_id,
                    stamp,
                    stamp,
                    stamp,
                ),
            )
            db.execute("UPDATE sources SET enabled=0 WHERE id=?", (source_id,))
        db.execute(
            "INSERT INTO runtime_settings(key,value,updated_at) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at",
            ("source_enabled:" + source_id, "true" if action == "resume" else "false", stamp),
        )
    # Source items, already-delivered cards and reactions deliberately remain intact.
