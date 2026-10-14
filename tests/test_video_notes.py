from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.types import Chat, Message

from app.config import Settings
from app.db.repository import Repository
from app.telegram_bot import video_notes
from app.telegram_bot.bot import build_router


def environment(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path, allowed_user_id=123)
    repo = Repository(settings.database_path)
    repo.migrate()
    router = build_router(settings, repo)
    handler = next(h.callback for h in router.message.handlers
                   if h.callback.__name__ == "video_note_handler")
    assert router.message.handlers[0].callback == handler
    return handler


def message(user=123, kind="private", size=100, duration=2):
    return SimpleNamespace(
        from_user=SimpleNamespace(id=user), chat=SimpleNamespace(type=kind), message_id=42,
        reply_to_message=None,
        video=SimpleNamespace(file_id="video", file_size=size, duration=duration), document=None,
        answer=AsyncMock(return_value=SimpleNamespace(delete=AsyncMock(), edit_text=AsyncMock())),
        answer_video_note=AsyncMock(), bot=SimpleNamespace(download=AsyncMock()),
    )


@pytest.mark.parametrize("user,kind,size,duration", [
    (456, "private", 100, 2), (123, "group", 100, 2),
    (123, "private", video_notes.MAX_INPUT_BYTES + 1, 2), (123, "private", 100, 61),
])
def test_reject_without_download(tmp_path, user, kind, size, duration):
    handler = environment(tmp_path)
    incoming = message(user, kind, size, duration)
    asyncio.run(handler(incoming))
    incoming.bot.download.assert_not_awaited()
    incoming.answer_video_note.assert_not_awaited()


def test_upload_reply_and_cleanup(tmp_path, monkeypatch):
    handler = environment(tmp_path)
    incoming = message()
    paths = []

    async def download(media, destination):
        destination.write_bytes(b"input")
        paths.append(destination)

    async def convert(source, target):
        assert source.read_bytes() == b"input"
        target.write_bytes(b"mp4")
        paths.append(target)
        return 2

    incoming.bot.download.side_effect = download
    monkeypatch.setattr(video_notes, "convert_video", convert)
    asyncio.run(handler(incoming))
    incoming.answer_video_note.assert_awaited_once()
    call = incoming.answer_video_note.call_args
    assert call.kwargs["length"] == 384
    assert call.kwargs["reply_parameters"].message_id == 42
    assert all(not path.exists() for path in paths)
    incoming.answer.return_value.delete.assert_awaited_once()


def test_error_cleanup(tmp_path, monkeypatch):
    handler = environment(tmp_path)
    incoming = message()
    paths = []

    async def download(media, destination):
        destination.write_bytes(b"input")
        paths.append(destination)

    incoming.bot.download.side_effect = download
    monkeypatch.setattr(video_notes, "convert_video", AsyncMock(
        side_effect=video_notes.VideoNoteError("Не удалось прочитать видео.")
    ))
    asyncio.run(handler(incoming))
    incoming.answer_video_note.assert_not_awaited()
    assert all(not path.exists() for path in paths)
    incoming.answer.return_value.edit_text.assert_awaited_once()


def test_video_document_and_existing_note():
    incoming = message()
    incoming.video = None
    incoming.document = SimpleNamespace(mime_type="application/octet-stream", file_name="clip.MOV")
    assert video_notes.uploaded_video(incoming) is incoming.document
    incoming.document.file_name = "notes.pdf"
    assert video_notes.uploaded_video(incoming) is None


def test_reply_command_uses_original_video(tmp_path, monkeypatch):
    handler = environment(tmp_path)
    incoming = message()
    incoming.reply_to_message = message()
    incoming.reply_to_message.message_id = 12
    incoming.video = None

    async def download(media, destination):
        assert media is incoming.reply_to_message.video
        destination.write_bytes(b"input")

    incoming.bot.download.side_effect = download
    monkeypatch.setattr(video_notes, "convert_video", AsyncMock(return_value=2))
    asyncio.run(handler(incoming))
    assert incoming.answer_video_note.call_args.kwargs["reply_parameters"].message_id == 12


def test_command_without_video_shows_usage(tmp_path):
    incoming = message()
    incoming.video = None
    asyncio.run(environment(tmp_path)(incoming))
    incoming.bot.download.assert_not_awaited()
    assert "/circle" in incoming.answer.call_args.args[0]


@pytest.mark.parametrize("text,caption,accepted", [
    (None, None, False), (None, "Обычное видео", False),
    (None, "/circle", True), ("/circle", None, True),
])
def test_registered_filter_requires_command(tmp_path, text, caption, accepted):
    settings = Settings(_env_file=None, data_dir=tmp_path, allowed_user_id=123)
    repo = Repository(settings.database_path)
    repo.migrate()
    handler = build_router(settings, repo).message.handlers[0]
    incoming = Message(message_id=1, date=datetime.now(UTC),
                       chat=Chat(id=123, type="private"), text=text, caption=caption)
    result = asyncio.run(handler.filters[0].callback(incoming, bot=AsyncMock()))
    assert bool(result) is accepted


def test_probe_rejects_long_file_before_encoding(tmp_path, monkeypatch):
    run = AsyncMock(return_value=b'{"format":{"duration":"61"},'
                    b'"streams":[{"codec_type":"video"}]}')
    monkeypatch.setattr(video_notes, "_run", run)
    with pytest.raises(video_notes.VideoNoteError, match="60 секунд"):
        asyncio.run(video_notes.convert_video(tmp_path / "input", tmp_path / "output.mp4"))
    assert run.await_count == 1
