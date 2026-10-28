"""On-demand YouTube fallback with durable receipts and verified file qualities."""

from __future__ import annotations

import asyncio
import fcntl
import json
import math
import re
import time
import uuid
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from app.db.repository import Repository
from app.media.downloads import MediaError

API = "https://api.monid.ai"
ENDPOINT = "/youtube/video"
RESERVE = 0.01
TTL = 24 * 3600


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False))
    temporary.chmod(0o600)
    temporary.replace(path)


def api(settings, method, path, body=None):
    if not settings.monid_api_key:
        raise MediaError("provider_unavailable")
    req = Request(
        API + path,
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={
            "Authorization": "Bearer " + settings.monid_api_key.get_secret_value(),
            "Content-Type": "application/json",
            "X-Monid-Client": "osved-media",
        },
    )
    with build_opener().open(req, timeout=30) as response:
        raw = response.read(8_000_001)
    if len(raw) > 8_000_000:
        raise MediaError("invalid_response")
    return json.loads(raw)


def receipt_cost(receipt):
    if receipt.get("status") == "BLOCKED":
        return 0.0, False
    value = (receipt.get("cost") or {}).get("value")
    if value is None:
        report = (receipt.get("billing") or {}).get("reportedCost") or {}
        if report.get("currency") == "USD" and report.get("unit") == "MICRO_DOLLAR":
            value = report.get("value", RESERVE * 1e6) / 1e6
    if isinstance(value, (float, int)) and math.isfinite(value) and value >= 0:
        return float(value), False
    return RESERVE, True


def settle(repo, state):
    cost, estimated = receipt_cost(state["receipt"])
    with repo.connection() as db:
        db.execute(
            "UPDATE usage_ledger SET cost_usd=?,metadata_json=? WHERE id=?",
            (
                cost,
                json.dumps(
                    {
                        "reservation_id": state["reservation_id"],
                        "video_id": state["video_id"],
                        "run_id": state["receipt"].get("runId"),
                        "cost_estimated": estimated,
                    }
                ),
                state["usage_id"],
            ),
        )


def reserve(repo, state, settings):
    with repo.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        total = db.execute(
            "SELECT COALESCE(SUM(cost_usd),0) FROM usage_ledger "
            "WHERE strftime('%Y-%m',created_at)=strftime('%Y-%m','now')"
        ).fetchone()[0]
        if total + RESERVE >= settings.budget_hard_stop_usd:
            raise MediaError("media_budget_blocked")
        return db.execute(
            "INSERT INTO usage_ledger(provider,operation,request_count,cost_usd,metadata_json) "
            "VALUES(?,?,1,?,?)",
            (
                "monid/mrscraper",
                "media.youtube.resolve",
                RESERVE,
                json.dumps(
                    {
                        "reservation_id": state["reservation_id"],
                        "video_id": state["video_id"],
                        "cost_estimated": True,
                    }
                ),
            ),
        ).lastrowid


def resolve(settings, video_id, *, cancelled=lambda: False, transport=api):
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise MediaError("invalid_response")
    folder = settings.data_dir / "youtube-media-receipts" / video_id
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = folder / "state.json"
    repo = Repository(settings.database_path)
    with (folder / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(path.read_text()) if path.exists() else None
        if state and state["status"] == "starting":
            raise MediaError("media_paid_uncertain")
        if state and state["status"] == "complete":
            settle(repo, state)
            if time.time() - state["created_at"] < TTL:
                return state["receipt"]
            save(folder / (state["reservation_id"] + ".json"), state)
            state = None
        if not state or state["status"] == "budget_blocked":
            if state:
                save(folder / (state["reservation_id"] + ".json"), state)
            if cancelled():
                raise MediaError("cancelled")
            catalogue = transport(
                settings, "POST", "/v1/inspect", {"provider": "mrscraper", "endpoint": ENDPOINT}
            )
            pricing = catalogue.get("price") or {}
            amount = pricing.get("amount") or {}
            price = amount.get("value")
            if (
                pricing.get("type") != "PER_CALL"
                or amount.get("currency") != "USD"
                or not isinstance(price, (float, int))
                or not 0 <= price <= RESERVE
            ):
                raise MediaError("media_budget_blocked")
            state = {
                "video_id": video_id,
                "reservation_id": uuid.uuid4().hex,
                "created_at": time.time(),
                "status": "starting",
            }
            save(path, state)
            try:
                state["usage_id"] = reserve(repo, state, settings)
            except MediaError:
                state["status"] = "budget_blocked"
                save(path, state)
                raise
            save(path, state)  # Written before dispatch: unknown POSTs cannot be replayed.
            try:
                receipt = transport(
                    settings,
                    "POST",
                    "/v1/run",
                    {
                        "provider": "mrscraper",
                        "endpoint": ENDPOINT,
                        "input": {"body": {"url": "https://www.youtube.com/watch?v=" + video_id}},
                    },
                )
            except HTTPError as exc:
                if exc.code in {400, 401, 402, 404, 429}:
                    state.update(
                        status="budget_blocked", receipt={"status": "FAILED", "cost": {"value": 0}}
                    )
                    save(path, state)
                    settle(repo, state)
                    raise MediaError("media_budget_blocked") from exc
                raise
            run_id = receipt.get("runId")
            if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", run_id):
                raise MediaError("media_paid_uncertain")
            state.update(status="running", run_id=run_id, receipt=receipt)
            save(path, state)
        deadline = time.monotonic() + min(settings.media_timeout_seconds, 360)
        while True:
            receipt = state["receipt"]
            status = receipt.get("status")
            if status in {"COMPLETED", "FAILED", "BLOCKED", "STOPPED", "TIMED_OUT"}:
                state["status"] = "budget_blocked" if status == "BLOCKED" else "complete"
                save(path, state)  # Retain the paid answer before parsing its structure.
                settle(repo, state)
                return receipt
            if cancelled():
                raise MediaError("cancelled")
            if time.monotonic() >= deadline:
                raise MediaError("media_provider_pending")
            time.sleep(2)
            state["receipt"] = transport(settings, "GET", "/v1/runs/" + quote(state["run_id"]))
            save(path, state)


def safe_url(url):
    try:
        p = urlsplit(url)
        return (
            p.scheme == "https"
            and p.hostname == "mrscraper-files.s3.us-west-2.amazonaws.com"
            and not p.username
            and not p.password
            and p.port in (None, 443)
        )
    except (ValueError, TypeError):
        return False


class SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not safe_url(newurl):
            raise MediaError("invalid_response")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_source(service, url, target, cancelled):
    if not safe_url(url):
        raise MediaError("invalid_response")
    cap = min(service.settings.media_max_file_bytes, 500_000_000)
    partial = target.with_suffix(".part")
    started = time.monotonic()
    try:
        with build_opener(SafeRedirect()).open(url, timeout=45) as response:
            if int(response.headers.get("Content-Length") or 0) > cap:
                raise MediaError("too_large")
            total = 0
            with partial.open("wb") as output:
                while chunk := response.read(256 * 1024):
                    if cancelled():
                        raise MediaError("cancelled")
                    if time.monotonic() - started > service.settings.media_timeout_seconds:
                        raise MediaError("timeout")
                    total += len(chunk)
                    if total > cap:
                        raise MediaError("too_large")
                    service.check_space()
                    output.write(chunk)
            if not total:
                raise MediaError("invalid_response")
        with partial.open("rb") as sample:
            if sample.read(12)[4:8] != b"ftyp":
                raise MediaError("invalid_response")
        partial.chmod(0o600)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


async def source(service, req, *, cancelled=lambda: False):
    video_id = req["external_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise MediaError("invalid_response")
    folder = service.root / "youtube" / video_id
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    target, marker = folder / "source.mp4", folder / "source.json"
    if target.is_file() and marker.is_file():
        saved = json.loads(marker.read_text())
        if saved["size"] == target.stat().st_size and time.time() - saved["created_at"] < TTL:
            return {**saved, "path": str(target)}
    receipt = await asyncio.to_thread(resolve, service.settings, video_id, cancelled=cancelled)
    if receipt.get("status") == "BLOCKED":
        raise MediaError("monid_control_blocked")
    raw = receipt.get("output") or {}
    if (
        receipt.get("status") != "COMPLETED"
        or raw.get("ok") is not True
        or raw.get("video_id") != video_id
        or not safe_url(raw.get("download_url"))
    ):
        raise MediaError("provider_unavailable")
    await asyncio.to_thread(download_source, service, raw["download_url"], target, cancelled)
    probe = json.loads(
        await service.command(
            [
                "ffprobe",
                "-v",
                "error",
                "-protocol_whitelist",
                "file,pipe",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(target),
            ],
            timeout=30,
            cancelled=cancelled,
        )
    )
    streams = probe.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
    width, height = int(video.get("width") or 0), int(video.get("height") or 0)
    if not 0 < width <= 16384 or not 0 < height <= 16384 or not audio:
        raise MediaError("invalid_response")
    saved = {
        "title": str(raw.get("title") or "Видео")[:300],
        "width": width,
        "height": height,
        "duration": int(float(probe["format"]["duration"])),
        "vcodec": video.get("codec_name"),
        "acodec": audio.get("codec_name"),
        "size": target.stat().st_size,
        "created_at": time.time(),
    }
    save(marker, saved)
    return {**saved, "path": str(target)}


def manifest(info):
    quality = min(info["width"], info["height"])
    choices = {
        str(q): {
            "label": str(q) + "p",
            "selector": "monid:" + str(q),
            "size": round(info["size"] * (q / quality) ** 1.8),
        }
        for q in [144, 240, 360, 480, 720, 1080, 1440, 2160]
        if q <= quality
    }
    if str(quality) not in choices:
        choices[str(quality)] = {
            "label": str(quality) + "p",
            "selector": "monid:" + str(quality),
            "size": info["size"],
        }
    choices["mp3"] = {
        "label": "MP3 · звук",
        "selector": "monid:mp3",
        "size": info["duration"] * 16000,
    }
    return {
        "backend": "monid_youtube",
        "title": info["title"],
        "items": [
            {
                "title": info["title"],
                "duration": info["duration"],
                "choices": choices,
                "playlist_index": 1,
                "multiple": False,
            }
        ],
    }


async def download(service, req, job, *, cancelled=lambda: False):
    info = await source(service, req, cancelled=cancelled)
    current = manifest(info)
    if req.get("id"):
        with Repository(service.settings.database_path).connection() as db:
            db.execute(
                "UPDATE media_requests SET manifest_json=?,updated_at=? WHERE id=?",
                (json.dumps(current, ensure_ascii=False), time.time(), req["id"]),
            )
    choice = job["choice"]
    if choice not in current["items"][0]["choices"]:
        raise MediaError("quality_unavailable")
    directory = service.directory(job["id"])
    directory.mkdir(exist_ok=True)
    target = directory / ("audio.mp3" if choice == "mp3" else "video.mp4")
    args = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-v",
        "error",
        "-threads",
        "1",
        "-protocol_whitelist",
        "file,pipe",
        "-i",
        info["path"],
    ]
    width, height = info["width"], info["height"]
    if choice == "mp3":
        args += ["-vn", "-c:a", "libmp3lame", "-b:a", "128k"]
    else:
        q = int(choice)
        resize = q < min(width, height)
        if resize:
            args += ["-vf", f"scale=-2:{q}" if width >= height else f"scale={q}:-2"]
        args += (
            ["-c:v", "copy"]
            if not resize and info["vcodec"] == "h264"
            else [
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-threads",
                "1",
                "-pix_fmt",
                "yuv420p",
            ]
        )
        args += ["-c:a", "copy" if info["acodec"] == "aac" else "aac", "-movflags", "+faststart"]
        if resize:
            width, height = (
                (round(width * q / height / 2) * 2, q)
                if width >= height
                else (q, round(height * q / width / 2) * 2)
            )
    await service.command(
        args + [str(target)],
        timeout=service.settings.media_timeout_seconds,
        cancelled=cancelled,
        directory=directory,
    )
    if not target.is_file() or not 0 < target.stat().st_size <= service.max_bytes:
        raise MediaError("too_large")
    final = json.loads(
        await service.command(
            [
                "ffprobe",
                "-v",
                "error",
                "-protocol_whitelist",
                "file,pipe",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(target),
            ],
            timeout=30,
            cancelled=cancelled,
        )
    )
    video = next((s for s in final.get("streams", []) if s.get("codec_type") == "video"), {})
    result = {
        "path": str(target),
        "kind": "audio" if choice == "mp3" else "video",
        "size": target.stat().st_size,
        "title": info["title"],
        "duration": int(float(final["format"]["duration"])),
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
    }
    if choice != "mp3":
        from app.reposts.media import video_thumbnail

        thumbnail = await asyncio.to_thread(video_thumbnail, target)
        if thumbnail:
            result["thumbnail_path"] = str(thumbnail)
    save(directory / "ready.json", result)
    return result
