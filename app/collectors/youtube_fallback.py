"""Opt-in YouTube metadata fallback; bounded calls, exact dates and durable billing."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from app.collectors.base import CollectedItem, CollectionBatch, CollectionRequest
from app.collectors.platform_transport import MonidResponse, PlatformError, monid_fetch, unwrap_data
from app.collectors.rss import extract_urls
from app.config import Settings
from app.db.repository import Repository
from app.video.shorts import player_type

LIST_ENDPOINT = "/api/v1/youtube/web_v2/get_channel_videos"
DETAIL_ENDPOINT = "/api/v1/youtube/web_v2/get_video_info"
CALL_PRICE = 0.0015
MAX_NEW_DETAILS = 5
CACHE_HOURS = 4


def _listing(payload: Any, channel_id: str) -> dict:
    body = unwrap_data(payload)
    channel = body.get("channel") if isinstance(body, dict) else None
    videos = body.get("videos") if isinstance(body, dict) else None
    if (
        not isinstance(channel, dict)
        or channel.get("id") != channel_id
        or not isinstance(channel.get("name"), str)
        or not channel["name"].strip()
        or not isinstance(videos, list)
    ):
        raise PlatformError(
            "YouTube fallback channel identity/list mismatch", status="schema_changed"
        )
    return {
        "channel_id": channel_id,
        "title": " ".join(channel["name"].split())[:180],
        "videos": [
            {key: row.get(key) for key in ("video_id", "published_time", "is_live", "is_short")}
            for row in videos[:100]
            if isinstance(row, dict)
        ],
    }


def _detail(payload: Any, channel_id: str, video_id: str) -> dict:
    body = unwrap_data(payload)
    player = body.get("playerResponse", {}) if isinstance(body, dict) else {}
    if not isinstance(player, dict):
        raise PlatformError("YouTube fallback player metadata missing", status="schema_changed")
    details = player.get("videoDetails", {})
    micro = player.get("microformat", {}).get("playerMicroformatRenderer", {})
    if (
        not isinstance(details, dict)
        or not isinstance(micro, dict)
        or details.get("videoId") != video_id
        or (details.get("channelId") or micro.get("externalChannelId")) != channel_id
    ):
        raise PlatformError("YouTube fallback video identity mismatch", status="schema_changed")
    # Relative labels and date-only values cannot establish follow-time freshness.
    published = micro.get("publishDate") or micro.get("uploadDate") or ""
    if not isinstance(published, str) or "T" not in published:
        raise PlatformError(
            "YouTube fallback has no exact publication time", status="schema_changed"
        )
    stamp = datetime.fromisoformat(published.replace("Z", "+00:00"))
    if stamp.tzinfo is None or not isinstance(details.get("title"), str) or not details["title"]:
        raise PlatformError("YouTube fallback metadata is incomplete", status="schema_changed")
    return {
        "video_id": video_id,
        "channel_id": channel_id,
        "title": details["title"][:1000],
        "description": str(details.get("shortDescription") or "")[:12000],
        "author": str(details.get("author") or micro.get("ownerChannelName") or "")[:180],
        "published_at": stamp.astimezone(UTC).isoformat(),
        "is_short": player_type(player),
        "duration_seconds": details.get("lengthSeconds"),
    }


class YouTubeFallback:
    def __init__(self, settings: Settings, repository: Repository, *, paid=None):
        self.settings = settings
        self.repository = repository
        self.directory = settings.data_dir / "youtube-fallback"
        self.paid = paid or (
            lambda endpoint, query: monid_fetch(endpoint, query, self.directory / "runs")
        )

    def _reserve(self, channel_id: str, source=None) -> int:
        if not self.settings.youtube_rss_fallback_enabled or self.settings.monid_api_key is None:
            raise PlatformError("YouTube Monid fallback is not enabled", status="disabled")
        stamp = datetime.now(UTC).isoformat()
        month, today = stamp[:7], stamp[:10]
        with self.repository.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            total = db.execute(
                "SELECT COALESCE(SUM(cost_usd),0) FROM usage_ledger WHERE substr(created_at,1,7)=?",
                (month,),
            ).fetchone()[0]
            reserved = db.execute(
                "SELECT COALESCE(SUM(reserved_usd),0) FROM source_runs WHERE status='running'"
            ).fetchone()[0]
            # Fallback costs are already in the ledger: exclude their mirrored source_runs.
            sensory = db.execute(
                "SELECT COALESCE(SUM(cost_usd+reserved_usd),0), "
                "COALESCE(SUM(CASE WHEN substr(started_at,1,10)=? "
                "THEN cost_usd+reserved_usd ELSE 0 END),0) FROM source_runs "
                "WHERE substr(started_at,1,7)=? AND adapter_id!='youtube_rss'",
                (today, month),
            ).fetchone()
            fallback = db.execute(
                "SELECT COALESCE(SUM(cost_usd),0), "
                "COALESCE(SUM(CASE WHEN substr(created_at,1,10)=? THEN cost_usd ELSE 0 END),0), "
                "COALESCE(SUM(CASE WHEN json_extract(metadata_json,'$.channel_id')=? "
                "THEN cost_usd ELSE 0 END),0) FROM usage_ledger "
                "WHERE substr(created_at,1,7)=? AND operation='youtube.fallback'",
                (today, channel_id, month),
            ).fetchone()
            if (
                total + reserved + CALL_PRICE >= self.settings.budget_hard_stop_usd
                or sensory[0] + fallback[0] + CALL_PRICE > self.settings.sensory_monthly_cap_usd
                or sensory[1] + fallback[1] + CALL_PRICE > self.settings.sensory_daily_cap_usd
                or fallback[2] + CALL_PRICE > (source.monthly_cost_cap_usd if source else 1.0)
            ):
                raise PlatformError("YouTube fallback budget limit reached", status="disabled")
            return db.execute(
                "INSERT INTO usage_ledger(provider,operation,request_count,cost_usd,metadata_json) "
                "VALUES('monid','youtube.fallback',1,?,?)",
                (CALL_PRICE, json.dumps({"channel_id": channel_id, "reserved": True})),
            ).lastrowid

    def _write(self, path: Path, data: dict) -> None:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)

    def _call(self, endpoint: str, query: dict, channel_id: str, normalize, *, source=None):
        self.directory.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(json.dumps([endpoint, query], sort_keys=True).encode()).hexdigest()
        path = self.directory / (key + ".json")
        with (self.directory / (key + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            now = datetime.now(UTC)
            if path.exists():
                cache = json.loads(path.read_text(encoding="utf-8"))
                fresh = now - datetime.fromisoformat(cache["fetched_at"]) < timedelta(
                    hours=CACHE_HOURS
                )
                if cache["status"] in {"running", "uncertain"}:
                    raise PlatformError(
                        "An unfinished YouTube fallback request needs inspection; no paid replay",
                        status="partial",
                    )
                if cache["status"] == "ready" and (fresh or endpoint == DETAIL_ENDPOINT):
                    return cache["data"], 0.0, cache["fetched_at"]
                if cache["status"] == "failed" and fresh:
                    raise PlatformError(cache["error"], status=cache["error_status"])
            usage = self._reserve(channel_id, source)
            record = {"fetched_at": now.isoformat(), "status": "running", "usage_id": usage}
            self._write(path, record)
            cost = CALL_PRICE
            run_id = None
            received = False
            try:
                result: MonidResponse = self.paid(endpoint, query)
                cost, run_id = result.cost_usd, result.run_id
                received = True
                data = normalize(result.payload)
                self._write(path, dict(record, status="ready", data=data, run_id=run_id))
            except Exception as exc:
                if isinstance(exc, PlatformError):
                    status = exc.status
                    # Schema checks happen after a successfully billed provider response.
                    if not received:
                        cost = exc.cost_usd
                else:
                    status = "schema_changed" if received else "partial"
                self._write(
                    path,
                    dict(
                        record,
                        status="uncertain" if status == "partial" else "failed",
                        error=str(exc),
                        error_status=status,
                        run_id=run_id,
                    ),
                )
                raise PlatformError(str(exc), status=status, cost_usd=cost) from exc
            finally:
                with self.repository.connection() as db:
                    db.execute(
                        "UPDATE usage_ledger SET cost_usd=?,metadata_json=? WHERE id=?",
                        (
                            cost,
                            json.dumps(
                                {
                                    "channel_id": channel_id,
                                    "endpoint": endpoint,
                                    "run_id": run_id,
                                    "reserved": False,
                                }
                            ),
                            usage,
                        ),
                    )
            return data, cost, now.isoformat()

    def channel(self, channel_id: str, *, source=None):
        if not re.fullmatch(r"UC[A-Za-z0-9_-]{22}", channel_id):
            raise PlatformError("Invalid fallback channel ID")
        return self._call(
            LIST_ENDPOINT,
            {"channel_id": channel_id, "language_code": "en-US", "need_format": True},
            channel_id,
            lambda payload: _listing(payload, channel_id),
            source=source,
        )

    def collect(self, request: CollectionRequest, channel_id: str) -> CollectionBatch:
        cost = 0.0
        items, errors = [], []
        failure_status = "schema_changed"
        metadata = {"usage_recorded": True, "transport": "monid_fallback"}
        try:
            listing, charge, fetched_at = self.channel(channel_id, source=request.source)
            cost += charge
            cutoff = request.now - timedelta(days=request.source.production_max_age_days)
            not_before = request.source.options.get("not_before")
            if not_before:
                cutoff = max(cutoff, datetime.fromisoformat(str(not_before)))
            paid_details = 0
            seen = set()
            for row in listing["videos"]:
                video_id = row.get("video_id")
                if not isinstance(video_id, str) or not re.fullmatch(
                    r"[A-Za-z0-9_-]{11}", video_id
                ):
                    errors.append("Invalid video identity")
                    continue
                if video_id in seen or row.get("is_live") or row.get("is_short") is True:
                    continue
                seen.add(video_id)
                # Relative dates avoid old-video calls, never become publication dates.
                relative = re.fullmatch(
                    r"(?:Streamed |Premiered )?(\d+) "
                    r"(second|minute|hour|day|week|month|year)s? ago",
                    str(row.get("published_time") or ""),
                )
                if relative:
                    seconds = {
                        "second": 1,
                        "minute": 60,
                        "hour": 3600,
                        "day": 86400,
                        "week": 604800,
                        "month": 2419200,
                        "year": 31536000,
                    }
                    latest = datetime.fromisoformat(fetched_at) - timedelta(
                        seconds=int(relative[1]) * seconds[relative[2]]
                    )
                    if latest < cutoff:
                        continue
                if paid_details >= MAX_NEW_DETAILS:
                    errors.append("New-video metadata limit reached; remaining videos deferred")
                    break
                try:
                    detail, charge, _ = self._call(
                        DETAIL_ENDPOINT,
                        {"video_id": video_id, "language_code": "en-US", "need_format": False},
                        channel_id,
                        lambda payload, v=video_id: _detail(payload, channel_id, v),
                        source=request.source,
                    )
                except PlatformError as exc:
                    cost += exc.cost_usd
                    errors.append(str(exc))
                    if exc.status in {"disabled", "partial"}:
                        failure_status = exc.status
                        break
                    paid_details += 1
                    continue
                cost += charge
                paid_details += int(charge > 0)
                published = datetime.fromisoformat(detail["published_at"])
                if not cutoff <= published <= request.now + timedelta(minutes=10):
                    continue
                if detail.get("is_short") is True:
                    continue
                url = "https://www.youtube.com/watch?v=" + video_id
                items.append(
                    CollectedItem(
                        external_id=video_id,
                        canonical_url=url,
                        title=detail["title"],
                        text=(detail["title"] + "\n\n" + detail["description"]).strip()[:12000],
                        published_at=detail["published_at"],
                        author=detail["author"],
                        language=request.source.language,
                        content_type="video",
                        discovery_mode="channel_uploads",
                        outbound_urls=extract_urls(detail["description"], source_url=url),
                        media_metadata={
                            "video_id": video_id,
                            "channel_id": channel_id,
                            "content_basis": "title_and_description",
                            "transcript_fetched": False,
                            "is_short": detail.get("is_short"),
                            "duration_seconds": detail.get("duration_seconds"),
                        },
                        raw_payload={**detail, "content_basis": "metadata_only"},
                    )
                )
            items.sort(key=lambda item: item.published_at, reverse=True)
            status = (
                "partial"
                if errors and items
                else failure_status
                if errors
                else ("ok" if items else "empty")
            )
            return CollectionBatch(
                items=items[: request.limit],
                status=status,
                cost_usd=cost,
                error="; ".join(errors)[:1000] or None,
                metadata=metadata,
            )
        except (PlatformError, ValueError, TypeError) as exc:
            if isinstance(exc, PlatformError):
                cost += exc.cost_usd
            return CollectionBatch(
                status=getattr(exc, "status", "schema_changed"),
                error=str(exc),
                cost_usd=cost,
                metadata=metadata,
            )
