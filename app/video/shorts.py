"""Free native Shorts checks; an unknown type never authorizes automatic delivery."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from app.db.repository import Repository

VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}")
MAX_PAGE = 4_000_000


def metadata_type(metadata: dict) -> bool | None:
    for key in ("is_short", "isShort", "isShorts"):
        value = metadata.get(key)
        if isinstance(value, bool):
            return value
    # Long duration is definitive; short duration alone is not.
    try:
        if float(metadata.get("duration_seconds") or metadata.get("lengthSeconds") or 0) > 180:
            return False
    except (ValueError, TypeError):
        pass
    return None


def player_type(player: dict) -> bool | None:
    details = player.get("videoDetails") or {}
    micro = (player.get("microformat") or {}).get("playerMicroformatRenderer") or {}
    for metadata in (details, micro):
        answer = metadata_type(metadata)
        if answer is not None:
            return answer
    return None


def _route(url: str, video_id: str) -> bool | None:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in {"www.youtube.com", "youtube.com"}:
        return None
    if parsed.path.rstrip("/") == "/shorts/" + video_id:
        return True
    if parsed.path == "/watch" and parse_qs(parsed.query).get("v") == [video_id]:
        return False
    return None


class _Canonical(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "link" and values.get("rel") == "canonical":
            self.urls.append(values.get("href", ""))
        if tag == "meta" and values.get("property") == "og:url":
            self.urls.append(values.get("content", ""))


def classify_page(payload: bytes, video_id: str) -> bool | None:
    text = payload.decode("utf-8", errors="replace")
    canonical = _Canonical()
    canonical.feed(text)
    answers = {_route(url, video_id) for url in canonical.urls} - {None}
    return answers.pop() if len(answers) == 1 else None


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def probe_video_type(video_id: str) -> bool | None:
    """Read only the native Shorts route. No cookies, playback or paid API."""
    if not VIDEO_ID.fullmatch(video_id):
        return None
    url = "https://www.youtube.com/shorts/" + video_id
    request = Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "text/html"})
    try:
        with build_opener(_NoRedirect()).open(request, timeout=10) as response:
            payload = response.read(MAX_PAGE + 1)
            return classify_page(payload, video_id) if len(payload) <= MAX_PAGE else None
    except HTTPError as exc:
        if exc.code in {301, 302, 303, 307, 308}:
            # A redirect is proof of long-form only when it names this exact video.
            route = _route(urljoin(url, exc.headers.get("Location", "")), video_id)
            return False if route is False else None
    except (URLError, OSError, TimeoutError, ValueError):
        pass
    return None


class ShortsClassifier:
    def __init__(self, repository: Repository, *, probe=probe_video_type, max_probes=20):
        self.repository, self.probe = repository, probe
        self.remaining_probes = max_probes

    def cached(self, video_id: str) -> bool | None:
        with self.repository.connection() as db:
            row = db.execute(
                "SELECT is_short FROM youtube_video_types WHERE video_id=?", (video_id,)
            ).fetchone()
        return bool(row[0]) if row and row[0] is not None else None

    def check_item(self, item: dict) -> bool | None:
        try:
            metadata = json.loads(item.get("media_metadata_json") or "{}")
        except (ValueError, TypeError):
            metadata = {}
        metadata = metadata if isinstance(metadata, dict) else {}
        parsed = urlsplit(str(item.get("source_url") or ""))
        video_id = metadata.get("video_id") or (parse_qs(parsed.query).get("v") or [""])[0]
        if not video_id and parsed.path.startswith("/shorts/"):
            video_id = parsed.path.split("/")[2]
        if not isinstance(video_id, str) or not VIDEO_ID.fullmatch(video_id):
            return None
        known = metadata_type(metadata)
        if _route(str(item.get("source_url") or ""), video_id) is True:
            known = True
        now = datetime.now(UTC)
        with self.repository.connection() as db:
            row = db.execute(
                "SELECT * FROM youtube_video_types WHERE video_id=?", (video_id,)
            ).fetchone()
        if known is None and row:
            if row["is_short"] is not None:
                known = bool(row["is_short"])
            elif row["retry_after"] and datetime.fromisoformat(row["retry_after"]) > now:
                return None
        if known is None:
            if self.remaining_probes <= 0:
                return None
            self.remaining_probes -= 1
            known = self.probe(video_id)
        with self.repository.connection() as db:
            db.execute(
                "INSERT INTO youtube_video_types VALUES(?,?,?,?,?) "
                "ON CONFLICT(video_id) DO UPDATE SET is_short=excluded.is_short,"
                "basis=excluded.basis,checked_at=excluded.checked_at,"
                "retry_after=excluded.retry_after",
                (
                    video_id,
                    known,
                    "metadata_or_native_route",
                    now.isoformat(),
                    (now + timedelta(hours=6)).isoformat() if known is None else None,
                ),
            )
            if known is not None:
                metadata.update(video_id=video_id, is_short=known)
                db.execute(
                    "UPDATE source_items SET media_metadata_json=?,"
                    "processing_status=CASE WHEN ? THEN 'filtered_short' "
                    "WHEN processing_status='video_pending_type' THEN 'new' "
                    "ELSE processing_status END WHERE id=?",
                    (json.dumps(metadata), known, item["id"]),
                )
        return known
