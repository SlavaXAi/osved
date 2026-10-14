"""Explicit usefulness of delivered video results, separate from discovery clicks."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from app.db.repository import Repository
from app.video.transcripts import youtube_video_id


def get_result_feedback(repository: Repository, *, job_id: int, chat_id: int) -> str | None:
    with repository.connection() as db:
        row = db.execute(
            "SELECT reaction FROM video_result_feedback WHERE job_id=? AND chat_id=?",
            (job_id, chat_id),
        ).fetchone()
    return row["reaction"] if row else None


def save_result_feedback(
    repository: Repository,
    *,
    job_id: int,
    chat_id: int,
    user_id: int,
    reaction: str,
    message_id: int | None = None,
) -> dict:
    """Change one result's reaction atomically; repeating it creates no extra vote."""
    if reaction not in {"like", "dislike"}:
        raise ValueError("Expected like or dislike")
    stamp = datetime.now(UTC).isoformat()
    with repository.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        job = db.execute("SELECT * FROM video_pdf_jobs WHERE id=?", (job_id,)).fetchone()
        if job is None or job["chat_id"] != chat_id or job["user_id"] != user_id:
            raise PermissionError("Result does not belong to this owner and chat")
        delivered_id = (
            job["text_message_id"]
            if job["output_kind"] == "tldr"
            else job["document_message_id"]
        )
        delivered_alias = None
        if message_id is not None and db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='video_result_messages'"
        ).fetchone():
            delivered_alias = db.execute(
                "SELECT 1 FROM video_result_messages WHERE job_id=? AND chat_id=? "
                "AND message_id=? AND output_kind=?",
                (job_id, chat_id, message_id, job["output_kind"]),
            ).fetchone()
        if job["status"] != "sent" or (delivered_id is None and not delivered_alias):
            raise ValueError("Only delivered results can receive usefulness feedback")
        if message_id is not None and delivered_id != message_id and not delivered_alias:
            raise PermissionError("Reaction is not attached to the delivered result")
        if (
            message_id is not None
            and job["output_kind"] == "tldr"
            and db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='video_panels'"
            ).fetchone()
        ):
            # A job can be reopened in another card. Inspect the actual origin
            # message, not only the job's newest panel binding or old receipt.
            panel = db.execute(
                "SELECT * FROM video_panels WHERE chat_id=? AND message_id=?",
                (chat_id, message_id),
            ).fetchone()
            if panel and (
                panel["video_id"] != job["video_id"]
                or panel["view"] != "summary"
                or panel["summary_job_id"] != job_id
            ):
                raise PermissionError("Open this result before rating its panel")
        old = db.execute(
            "SELECT reaction FROM video_result_feedback WHERE job_id=? AND chat_id=?",
            (job_id, chat_id),
        ).fetchone()
        previous = old["reaction"] if old else None
        changed = previous != reaction
        if changed:
            db.execute(
                "INSERT INTO video_result_feedback(job_id,chat_id,reaction,created_at,updated_at) "
                "VALUES(?,?,?,?,?) ON CONFLICT(job_id,chat_id) DO UPDATE SET "
                "reaction=excluded.reaction,updated_at=excluded.updated_at",
                (job_id, chat_id, reaction, stamp, stamp),
            )
            db.execute(
                "INSERT INTO video_result_feedback_events"
                "(job_id,chat_id,reaction,previous_reaction,created_at) VALUES(?,?,?,?,?)",
                (job_id, chat_id, reaction, previous, stamp),
            )
    return {
        "job_id": job_id,
        "video_id": job["video_id"],
        "output_kind": job["output_kind"],
        "reaction": reaction,
        "previous_reaction": previous,
        "changed": changed,
    }


def _cached_metadata(settings, video_id: str) -> dict:
    """Use existing transcript metadata for direct-link jobs; never fetch a transcript."""
    try:
        if youtube_video_id(video_id) != video_id:
            return {}
        path = settings.data_dir / "video-reports" / video_id / "transcript.json"
        if not path.is_file() or path.stat().st_size > 8_000_000:
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = value.get("output")
        if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
            return {}
        metadata = value[0]
        if metadata.get("video_id") != video_id:
            return {}
        return {
            "channel": str(metadata.get("channel_name") or "")[:120],
            "metadata": "\n".join(
                str(metadata.get(key) or "") for key in ("title", "description")
            ).strip()[:700],
        }
    except (OSError, ValueError, TypeError):
        return {}


def result_usefulness_examples(settings, repository, chat_id, *, limit=8) -> tuple[list, set]:
    """One video example, latest vote per format, including mixed format feedback."""
    from app.telegram_bot.bot import configured_owner_id

    owner_id = configured_owner_id(settings, repository)
    with repository.connection() as db:
        rows = db.execute(
            """SELECT f.reaction,f.updated_at,j.video_id,j.output_kind,j.id AS job_id
            FROM video_result_feedback f JOIN video_pdf_jobs j ON j.id=f.job_id
            WHERE f.chat_id=? AND j.chat_id=? AND j.user_id=? AND j.status='sent'
            ORDER BY f.updated_at DESC,j.id DESC""",
            (chat_id, chat_id, owner_id),
        ).fetchall()
        rated_ids = {row["video_id"] for row in rows}
        selected_ids = list(dict.fromkeys(row["video_id"] for row in rows))[:limit]
        metadata_by_id = {}
        if selected_ids:
            placeholders = ",".join("?" for _ in selected_ids)
            metadata_rows = db.execute(
                f"SELECT i.external_id,i.original_text,s.name FROM source_items i "
                f"JOIN sources s ON s.id=i.source_id WHERE i.external_id IN ({placeholders}) "
                "AND s.kind='youtube' ORDER BY i.id DESC",
                selected_ids,
            ).fetchall()
            for metadata in metadata_rows:
                metadata_by_id.setdefault(
                    metadata["external_id"],
                    {
                        "channel": metadata["name"],
                        "metadata": str(metadata["original_text"] or "")[:700],
                    },
                )
    examples = {}
    for row in rows:
        video = row["video_id"]
        if video not in selected_ids:
            continue
        if video not in examples:
            metadata = metadata_by_id.get(video)
            if metadata is None:
                metadata = _cached_metadata(settings, video)
            examples[video] = {
                "video_id": video,
                **metadata,
                "results": [],
            }
        # Repeated requests of one video/format remain a single preference example.
        if not any(r["output_kind"] == row["output_kind"] for r in examples[video]["results"]):
            examples[video]["results"].append(
                {"output_kind": row["output_kind"], "reaction": row["reaction"]}
            )
    return list(examples.values()), rated_ids
