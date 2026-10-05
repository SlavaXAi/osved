"""Read projection over the bot database, plus isolated dashboard state.

No collector, publication, Telegram queue or source synchronization is triggered
by reading the dashboard. Classification is a view and never rewrites history.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any
from urllib.parse import urlsplit

from app.config import Settings, load_contour_registry
from app.contours.classifier import classify_event
from app.contours.portfolio import event_primary_contour
from app.db.card_predicates import finished_card_sql
from app.db.repository import Repository, utc_now
from app.telegram_bot.news_depth import brief_card_html
from app.video.reports import VideoReportService
from app.video.tldr import telegram_tldr
from app.web.content import evidence_url, plain_text, public_url, safe_html
from app.web.folders import infer_reading_folder

FOLDERS = {
    "all": "Вся лента",
    "ai_software": "AI и software",
    "startups": "Стартапы",
    "political_economy": "Политэкономика",
    "space": "Космос",
    "people_institutions": "Люди и институты",
    "human_civilization": "Философия",
    "unclassified": "Другие события",
}
REACTIONS = {
    "super_dislike": "superdislike",
    "dislike": "dislike",
    "like": "like",
    "super_like": "superlike",
    None: "",
}
REACTIONS_OUT = {value: key for key, value in REACTIONS.items()}


def decoded(value: object, fallback: Any) -> Any:
    try:
        result = json.loads(value) if isinstance(value, str) else value
        return result if isinstance(result, type(fallback)) else fallback
    except (TypeError, ValueError):
        return fallback


class DashboardStore:
    def __init__(self, settings: Settings, repository: Repository, *, reports=None):
        self.settings, self.repository = settings, repository
        self.reports = reports or VideoReportService(settings, repository)
        try:
            self.contours = load_contour_registry(settings.contours_file)
        except (OSError, ValueError):
            self.contours = None

    def initialize(self) -> None:
        # Web-only tables are intentionally not part of the bot's migration stream.
        with self.repository.connection() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS web_saved (
                kind TEXT NOT NULL, item_id INTEGER NOT NULL, saved INTEGER NOT NULL,
                updated_at TEXT NOT NULL, PRIMARY KEY(kind,item_id))""")
            db.execute("""CREATE TABLE IF NOT EXISTS web_video_reactions (
                item_id INTEGER PRIMARY KEY, reaction TEXT, updated_at TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS web_jobs (
                id TEXT PRIMARY KEY, job_key TEXT UNIQUE NOT NULL, kind TEXT NOT NULL,
                item_id INTEGER NOT NULL, level TEXT NOT NULL, status TEXT NOT NULL,
                stage TEXT NOT NULL, result_json TEXT, error TEXT, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL)""")

    def folder(self, row: dict) -> str:
        return self.folder_details(row)[0]

    def folder_details(self, row: dict) -> tuple[str, str]:
        contour = event_primary_contour(row)
        basis = "stored" if contour else "inferred_rules"
        if not contour and self.contours:
            # Enrich historical thin metadata using the complete existing publication,
            # not a new provider call or a rewrite of the original event.
            classification = {
                **row,
                "original_text": plain_text(
                    row.get("rendered_html")
                    or row.get("description_html")
                    or row.get("original_text")
                ),
            }
            contour = classify_event(classification, self.contours).primary_contour
        if not contour:
            contour = infer_reading_folder(row)
        if contour in {"geopolitics", "world_economy"}:
            return "political_economy", basis
        return (contour, basis) if contour in FOLDERS else ("unclassified", "unclassified")

    def news_rows(self, item_id: int | None = None) -> list[dict]:
        constraint = " AND c.id=?" if item_id is not None else ""
        with self.repository.connection() as db:
            rows = db.execute(
                f"""SELECT c.*, e.headline AS event_headline,
                e.event_at, e.latest_source_at, e.topics_json, e.regions_json,
                e.entities_json, e.score_features_json, si.original_text,
                si.source_url AS trigger_url, s.id AS source_id, s.name AS source_name,
                COALESCE(w.saved,0) AS saved,
                (SELECT f.reaction FROM feedback_events f
                 WHERE f.card_id=c.id AND f.reaction IS NOT NULL ORDER BY f.id DESC LIMIT 1)
                 AS reaction
                FROM cards c JOIN events e ON e.id=c.event_id
                LEFT JOIN source_items si ON si.id=e.primary_source_item_id
                LEFT JOIN sources s ON s.id=si.source_id
                LEFT JOIN web_saved w ON w.kind='news' AND w.item_id=c.id
                WHERE {finished_card_sql("c")}
                AND e.status NOT IN ('test','duplicate','rejected')
                AND c.id=(SELECT MAX(c2.id) FROM cards c2
                          WHERE c2.event_id=c.event_id AND {finished_card_sql("c2")})
                {constraint}
                ORDER BY c.created_at DESC,c.id DESC""",
                (item_id,) if item_id is not None else (),
            ).fetchall()
        return [dict(row) for row in rows]

    def video_rows(self, item_id: int | None = None) -> list[dict]:
        constraint = " AND v.id=?" if item_id is not None else ""
        with self.repository.connection() as db:
            rows = db.execute(
                f"""SELECT v.*, i.source_url, i.original_text,
                i.published_at, i.raw_payload_json, s.id AS source_id, s.name AS source_name,
                s.topics_json, COALESCE(w.saved,0) AS saved,
                r.reaction AS web_reaction, r.updated_at AS web_reaction_at,
                (SELECT f.action FROM video_feedback_events f
                 WHERE f.recommendation_id=v.id AND f.action IN ('interest','dismiss')
                 ORDER BY f.id DESC LIMIT 1) AS bot_reaction
                FROM video_recommendations v JOIN source_items i ON i.id=v.source_item_id
                JOIN sources s ON s.id=i.source_id
                LEFT JOIN web_saved w ON w.kind='video' AND w.item_id=v.id
                LEFT JOIN web_video_reactions r ON r.item_id=v.id
                WHERE v.description_html<>'' AND v.status NOT IN ('failed','generating')
                {constraint} ORDER BY i.published_at DESC,v.id DESC""",
                (item_id,) if item_id is not None else (),
            ).fetchall()
        return [dict(row) for row in rows]

    def source_refs(self, row: dict, kind: str) -> list[dict]:
        result, seen = [], set()
        urls = decoded(row.get("source_urls_json"), []) if kind == "news" else []
        primary = row.get("primary_preview_url") if kind == "news" else row.get("source_url")
        for value in [primary] + urls:
            url = evidence_url(value)
            if url and url not in seen:
                result.append(
                    {
                        "url": url,
                        "title": urlsplit(url).hostname,
                        "role": "evidence" if kind == "news" else "original",
                    }
                )
                seen.add(url)
        trigger = evidence_url(row.get("trigger_url"))
        if trigger and trigger not in seen:
            result.append(
                {"url": trigger, "title": row.get("source_name") or "Радар", "role": "trigger"}
            )
        return result

    def serialize(self, row: dict, kind: str, *, detail: bool = False) -> dict:
        draft = decoded(row.get("structured_output_json"), {})
        title = plain_text(draft.get("headline") or row.get("event_headline") or "")
        if kind == "news":
            brief = safe_html(brief_card_html(row))
            body = safe_html(row["rendered_html"])
            summary = plain_text(draft.get("lead")) or plain_text(brief)
            reaction = REACTIONS_OUT.get(row.get("reaction"))
            classification = {**row, "headline": title}
        else:
            raw = decoded(row.get("raw_payload_json"), {})
            title = (
                plain_text(raw.get("title")) or str(row.get("original_text") or "").split("\n")[0]
            )
            body = brief = safe_html(row["description_html"])
            summary = plain_text(body)
            reaction = (
                row["web_reaction"]
                if row.get("web_reaction_at")
                else {
                    "interest": "like",
                    "dismiss": "dislike",
                }.get(row.get("bot_reaction"))
            )
            classification = {**row, "headline": title, "summary": summary}
        folder, folder_basis = self.folder_details(classification)
        result = {
            "kind": kind,
            "id": row["id"],
            "key": f"{kind}:{row['id']}",
            "title": title or "Публикация",
            "summary": summary,
            "body_html": brief,
            "published_at": row.get("event_at") or row.get("published_at") or row["created_at"],
            "created_at": row["created_at"],
            "folder": folder,
            "folder_title": FOLDERS[folder],
            "folder_basis": folder_basis,
            "source_name": row.get("source_name") or "Исследование OSVED",
            "sources": self.source_refs(row, kind),
            "url": public_url(row.get("primary_preview_url") or row.get("source_url")),
            "reaction": reaction,
            "saved": bool(row["saved"]),
            "regions": decoded(row.get("regions_json"), []),
            "levels": {
                "context": kind == "news",
                "article": kind == "news",
                "short": kind == "video",
                "long": kind == "video",
            },
        }
        if detail:
            result["content"] = {
                "brief": brief,
                "context": body if kind == "news" else None,
                "article": None,
                "short": None,
            }
            result["pdf_url"] = None
            if kind == "news":
                # Historical valid articles remain readable with their original date.
                with self.repository.connection() as db:
                    article = db.execute(
                        "SELECT rendered_html,source_urls_json,created_at "
                        "FROM detailed_articles WHERE event_id=? AND validation_status='valid' "
                        "ORDER BY created_at DESC,id DESC LIMIT 1",
                        (row["event_id"],),
                    ).fetchone()
                if article:
                    result["content"]["article"] = safe_html(article["rendered_html"])
                    result["article_created_at"] = article["created_at"]
                    known = {source["url"] for source in result["sources"]}
                    for url in decoded(article["source_urls_json"], []):
                        if evidence_url(url) and url not in known:
                            result["sources"].append(
                                {"url": url, "title": urlsplit(url).hostname, "role": "evidence"}
                            )
                            known.add(url)
            else:
                try:
                    short = self.reports.cached_tldr(row["source_url"])
                    if short:
                        result["content"]["short"] = safe_html(telegram_tldr(short))
                    if self.reports.cached_pdf(row["source_url"]):
                        result["pdf_url"] = f"/api/items/video/{row['id']}/pdf"
                except ValueError:
                    pass
        return result

    def item(self, kind: str, item_id: int, *, detail: bool = True) -> dict | None:
        rows = self.news_rows(item_id) if kind == "news" else self.video_rows(item_id)
        return self.serialize(rows[0], kind, detail=detail) if rows else None

    def raw_item(self, kind: str, item_id: int) -> dict | None:
        rows = self.news_rows(item_id) if kind == "news" else self.video_rows(item_id)
        return rows[0] if rows else None

    def all_items(self) -> list[dict]:
        items = [self.serialize(row, "news") for row in self.news_rows()]
        items += [self.serialize(row, "video") for row in self.video_rows()]
        return sorted(items, key=lambda item: (item["published_at"], item["key"]), reverse=True)

    def overview(self) -> dict:
        items = self.all_items()
        counts = Counter(item["folder"] for item in items)
        return {
            "folders": [
                {"id": key, "title": title, "count": len(items) if key == "all" else counts[key]}
                for key, title in FOLDERS.items()
            ],
            "status": {
                "mode": "local_snapshot",
                "latest_at": items[0]["published_at"] if items else None,
                "news_count": sum(item["kind"] == "news" for item in items),
                "video_count": sum(item["kind"] == "video" for item in items),
                "monthly_cost_usd": self.repository.current_month_cost(),
            },
        }

    def set_saved(self, kind: str, item_id: int, saved: bool) -> None:
        with self.repository.connection() as db:
            db.execute(
                "INSERT INTO web_saved(kind,item_id,saved,updated_at) VALUES(?,?,?,?) "
                "ON CONFLICT(kind,item_id) DO UPDATE SET saved=excluded.saved, "
                "updated_at=excluded.updated_at",
                (kind, item_id, saved, utc_now()),
            )

    def react(self, kind: str, row: dict, reaction: str | None) -> None:
        stamp = utc_now()
        with self.repository.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if kind == "video":
                db.execute(
                    "INSERT INTO web_video_reactions(item_id,reaction,updated_at) "
                    "VALUES(?,?,?) ON CONFLICT(item_id) DO UPDATE SET "
                    "reaction=excluded.reaction,updated_at=excluded.updated_at",
                    (row["id"], reaction, stamp),
                )
                db.execute(
                    "INSERT INTO video_feedback_events(recommendation_id,action,created_at) "
                    "VALUES(?,?,?)",
                    (
                        row["id"],
                        {
                            "like": "interest",
                            "super_like": "interest",
                            "dislike": "dismiss",
                            "super_dislike": "dismiss",
                        }.get(reaction, "web_clear"),
                        stamp,
                    ),
                )
                return
            # Append to the bot's reaction history without changing keyboard, sessions,
            # sending Telegram messages, or advancing its recommendation queue.
            delivery = db.execute(
                "SELECT id FROM deliveries WHERE card_id=? ORDER BY id DESC LIMIT 1", (row["id"],)
            ).fetchone()
            if delivery is None:
                delivery_id = db.execute(
                    "INSERT INTO deliveries(card_id,chat_id,delivery_status) "
                    "VALUES(?,?,'web_only')",
                    (row["id"], self.settings.allowed_user_id or 0),
                ).lastrowid
            else:
                delivery_id = delivery["id"]
            db.execute(
                "INSERT INTO feedback_events(delivery_id,event_id,card_id,action,reaction,"
                "previous_reaction,metadata_json,created_at) VALUES(?,?,?,?,?,?,?,?)",
                (
                    delivery_id,
                    row["event_id"],
                    row["id"],
                    "web_reaction",
                    REACTIONS[reaction],
                    row.get("reaction"),
                    '{"origin":"dashboard"}',
                    stamp,
                ),
            )

    def sources(self, folder: str = "all") -> list[dict]:
        associated: set[str] = set()
        if folder != "all":
            for kind, rows in (("news", self.news_rows()), ("video", self.video_rows())):
                for row in rows:
                    if self.serialize(row, kind)["folder"] == folder and row.get("source_id"):
                        associated.add(row["source_id"])
        with self.repository.connection() as db:
            rows = [
                dict(row)
                for row in db.execute(
                    "SELECT s.*,r.value AS enabled_override,y.removed_at AS channel_removed_at "
                    "FROM sources s LEFT JOIN runtime_settings r ON r.key='source_enabled:'||s.id "
                    "LEFT JOIN youtube_channels y ON y.source_id=s.id ORDER BY s.name,s.id"
                )
            ]
        result = []
        for row in rows:
            if row["role"] == "own_history" or row["channel_removed_at"]:
                continue
            enabled = (
                row["enabled_override"] == "true"
                if row["enabled_override"] is not None
                else bool(row["enabled"])
            )
            if (
                folder != "all"
                and row["id"] not in associated
                and self.folder(
                    {
                        "headline": row["name"],
                        "topics_json": row["topics_json"],
                    }
                )
                != folder
            ):
                continue
            locator = row["locator"]
            url = public_url(locator)
            if not url and row["kind"] == "telegram":
                url = "https://t.me/" + locator.lstrip("@")
            if not url and row["kind"] == "youtube":
                url = "https://www.youtube.com/channel/" + locator
            if not url and row["kind"] == "x":
                url = "https://x.com/" + locator.lstrip("@")
            result.append(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "kind": row["kind"],
                    "url": public_url(url),
                    "enabled": enabled,
                    "topics": decoded(row["topics_json"], []),
                    "last_success_at": row["last_success_at"],
                    "status": "paused"
                    if not enabled
                    else "error"
                    if row["last_error"]
                    else "ready",
                    "role": row["role"],
                }
            )
        return result

    def channels(self) -> list[dict]:
        return [
            {"id": row["id"], "name": row["name"], "url": row["url"], "enabled": row["enabled"]}
            for row in self.sources()
            if row["kind"] == "youtube"
        ]

    def store_article(self, result) -> None:
        with self.repository.connection() as db:
            db.execute(
                """INSERT OR IGNORE INTO detailed_articles(event_id,generation_key,
                schema_version,prompt_version,model,provider_response_metadata_json,
                fact_snapshot_json,document_ids_json,structured_output_json,rendered_html,
                source_urls_json,validation_status,validation_errors_json,generation_cost_usd,expires_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    result.event_id,
                    result.generation_key,
                    result.schema_version,
                    result.prompt_version,
                    result.model,
                    json.dumps(result.provider_response_metadata),
                    json.dumps(result.fact_snapshot),
                    json.dumps(result.document_ids),
                    json.dumps(result.structured_output),
                    result.rendered_html,
                    json.dumps(result.source_urls),
                    result.validation_status,
                    json.dumps(result.validation_errors),
                    result.cost_usd,
                    result.expires_at,
                ),
            )
