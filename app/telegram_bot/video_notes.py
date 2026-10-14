"""Convert owner-uploaded videos locally and reply with a native Telegram video note."""

from __future__ import annotations

import asyncio
import json
import logging
import math
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory

from aiogram.filters import Command
from aiogram.types import FSInputFile, ReplyParameters

MAX_INPUT_BYTES = 20 * 1024 * 1024
MAX_SECONDS = 60
DIAMETER = 384
logger = logging.getLogger(__name__)


class VideoNoteError(ValueError):
    pass


def uploaded_video(message):
    if message.video:
        return message.video
    document = message.document
    if document and (
        (document.mime_type or "").startswith("video/")
        or Path(document.file_name or "").suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}
    ):
        return document
    return None


async def _run(*args: str, timeout: int) -> bytes:
    process = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout)
    except BaseException:
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
            await process.wait()
        raise
    if process.returncode:
        raise VideoNoteError("Не удалось прочитать видео. Попробуй прислать его в формате MP4.")
    return stdout


async def convert_video(source: Path, target: Path) -> int:
    metadata = json.loads(await _run(
        "ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe",
        "-show_entries", "format=duration:stream=codec_type,width,height,duration",
        "-of", "json", str(source), timeout=20,
    ))
    streams = metadata.get("streams", [])
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if video is None:
        raise VideoNoteError("В файле не найдено видео.")
    try:
        duration = float(metadata.get("format", {}).get("duration") or video.get("duration"))
    except (TypeError, ValueError) as exc:
        raise VideoNoteError("Не удалось определить длительность видео.") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise VideoNoteError("Не удалось определить длительность видео.")
    if duration > MAX_SECONDS:
        raise VideoNoteError("Кружочек может быть до 60 секунд. Пришли видео покороче.")
    await _run(
        "ffmpeg", "-nostdin", "-y", "-v", "error", "-protocol_whitelist", "file,pipe",
        "-threads", "1", "-i", str(source), "-map", "0:v:0", "-map", "0:a:0?",
        "-t", str(MAX_SECONDS), "-vf",
        f"crop=min(iw\\,ih):min(iw\\,ih),scale={DIAMETER}:{DIAMETER},setsar=1,fps=30",
        "-filter_threads", "1", "-c:v", "libx264", "-threads", "1",
        "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", "-map_metadata", "-1",
        str(target), timeout=120,
    )
    if not target.exists() or not 0 < target.stat().st_size < 50 * 1024 * 1024:
        raise VideoNoteError("Не удалось подготовить кружочек. Попробуй другое видео.")
    return min(MAX_SECONDS, math.ceil(duration))


def register_video_note_handlers(router, settings, repository):
    from app.telegram_bot.bot import is_allowed

    lock = asyncio.Lock()

    @router.message(Command("circle"))
    async def video_note_handler(message):
        user_id = message.from_user.id if message.from_user else None
        if message.chat.type != "private" or not is_allowed(settings, repository, user_id):
            return
        original = message if uploaded_video(message) else message.reply_to_message
        media = uploaded_video(original) if original else None
        if media is None:
            await message.answer(
                "Пришли видео с подписью /circle или ответь командой /circle на видео. "
                "До 60 секунд и 20 МБ."
            )
            return
        if (media.file_size or 0) > MAX_INPUT_BYTES:
            await message.answer("Для кружочка пришли видео до 20 МБ и до 60 секунд.")
            return
        if getattr(media, "duration", 0) > MAX_SECONDS:
            await message.answer("Кружочек может быть до 60 секунд. Пришли видео покороче.")
            return
        if lock.locked():
            await message.answer("Уже делаю кружочек. Пришли следующее видео после него.")
            return
        async with lock:
            progress = await message.answer("Делаю кружочек…")
            try:
                with TemporaryDirectory(prefix="osved-video-note-") as directory:
                    source = Path(directory) / "input"
                    target = Path(directory) / "circle.mp4"
                    await asyncio.wait_for(
                        message.bot.download(media, destination=source), timeout=60
                    )
                    if source.stat().st_size > MAX_INPUT_BYTES:
                        raise VideoNoteError("Для кружочка пришли видео до 20 МБ.")
                    duration = await convert_video(source, target)
                    await message.answer_video_note(
                        FSInputFile(target), duration=duration, length=DIAMETER,
                        reply_parameters=ReplyParameters(
                            message_id=original.message_id, allow_sending_without_reply=True
                        ), request_timeout=90,
                    )
                with suppress(Exception):
                    await progress.delete()
            except VideoNoteError as exc:
                await progress.edit_text(str(exc))
            except TimeoutError:
                await progress.edit_text("Обработка заняла слишком долго. Попробуй видео покороче.")
            except Exception as exc:
                logger.warning("Video note failed: %s", type(exc).__name__)
                await progress.edit_text("Не получилось отправить кружочек. Попробуй ещё раз.")
