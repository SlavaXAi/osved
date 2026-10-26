"""Text and voice replies bound to a particular delivered news card."""

from __future__ import annotations

import hashlib
import io
from contextlib import suppress

from aiogram.types import ForceReply, ReplyParameters

from app.db.news_questions import claim_input, open_session, queue_question, reply_session
from app.details.transcription import MAX_AUDIO_BYTES, MAX_VOICE_SECONDS
from app.telegram_bot.news_panel import (
    ensure_news_panel,
    refresh_markup,
    set_loading,
    show_result,
)


async def ask_question(query, repository, panel):
    previous = open_session(repository, panel)
    await query.answer("Ответь на новость текстом или голосом")
    if previous and previous.get("prompt_message_id"):
        return
    prompt = await query.bot.send_message(
        chat_id=panel["chat_id"],
        text=(
            "Что разобрать в этой новости? Напиши вопрос или запиши голосовое "
            "в ответ на новость или на это сообщение.\n\n"
            "Например: «Что умеет эта модель и где её можно применить?»"
        ),
        reply_parameters=ReplyParameters(message_id=panel["message_id"]),
        reply_markup=ForceReply(selective=True, input_field_placeholder="Вопрос к новости"),
    )
    with repository.connection() as db:
        db.execute(
            "UPDATE news_question_sessions SET prompt_message_id=? "
            "WHERE chat_id=? AND delivery_id=?",
            (prompt.message_id, panel["chat_id"], panel["delivery_id"]),
        )
    await refresh_markup(query.bot, repository, panel["delivery_id"])


class LimitedAudio(io.BytesIO):
    def write(self, data):
        if self.tell() + len(data) > MAX_AUDIO_BYTES:
            raise ValueError("Голосовое слишком большое. Пришли запись короче.")
        return super().write(data)


def register_news_question_handlers(router, settings, repository):
    from app.telegram_bot.bot import is_allowed

    def session_for(message):
        owner = message.from_user.id if message.from_user else None
        if (
            owner is None
            or not is_allowed(settings, repository, owner)
            or message.chat.type != "private"
            or message.chat.id != owner
            or not message.reply_to_message
            or (message.text and message.text.startswith("/"))
        ):
            return None
        return reply_session(repository, owner, message.reply_to_message.message_id)

    async def is_question(message):
        return session_for(message) is not None

    @router.message(is_question)
    async def news_question_reply(message):
        session = session_for(message)
        if not session:
            return
        from app.reposts.store import for_delivery

        repost = for_delivery(repository, session["delivery_id"])
        if repost and repost["status"] != "ready":
            await message.answer("Сначала дождись готовности публикации.")
            return
        if not settings.details_enabled:
            await message.answer("Разборы сейчас выключены.")
            return
        text = (message.text or "").strip()
        voice = message.voice
        if not voice and not 2 <= len(text) <= 6000:
            await message.answer("Напиши вопрос текстом или запиши голосовое сообщение.")
            return
        if voice and (
            not 0 < voice.duration <= MAX_VOICE_SECONDS or (voice.file_size or 0) > MAX_AUDIO_BYTES
        ):
            await message.answer("Для вопроса пришли голосовое до 3 минут и 8 МБ.")
            return
        try:
            if not claim_input(
                repository,
                session,
                message.message_id,
                "voice" if voice else "text",
                max_daily=settings.details_max_daily_new,
            ):
                return
        except ValueError as exc:
            await message.answer(str(exc))
            return
        try:
            voice_context = None
            if voice:
                buffer = LimitedAudio()
                await message.bot.download(voice, destination=buffer)
                audio = buffer.getvalue()
                if not audio:
                    raise ValueError("Не удалось загрузить голосовое. Пришли вопрос текстом.")
                digest = hashlib.sha256(audio).hexdigest()
                root = settings.data_dir / "news-question-audio"
                root.mkdir(parents=True, exist_ok=True)
                path = root / (digest + ".ogg")
                if not path.exists():
                    with path.open("xb") as handle:
                        handle.write(audio)
                    path.chmod(0o600)
                voice_context = {
                    "voice_input_sha256": digest,
                    "voice_duration_seconds": voice.duration,
                }
            panel = ensure_news_panel(repository, session["delivery_id"])
            request = queue_question(
                repository, panel, message.message_id, text, voice=voice_context
            )
            if request["status"] == "ready":
                artifact = repository.get_detailed_article(request["article_id"])
                if await show_result(message.bot, repository, panel, request, artifact):
                    repository.mark_detail_request_sent(request["request_id"])
            else:
                await set_loading(message.bot, repository, panel, request)
            if session.get("prompt_message_id"):
                with suppress(Exception):
                    await message.bot.delete_message(
                        chat_id=message.chat.id, message_id=session["prompt_message_id"]
                    )
        except Exception as exc:
            with repository.connection() as db:
                db.execute(
                    "UPDATE news_question_inputs SET status='failed',error=? "
                    "WHERE chat_id=? AND message_id=? AND status='receiving'",
                    (type(exc).__name__, message.chat.id, message.message_id),
                )
            await message.answer(
                str(exc)
                if isinstance(exc, ValueError)
                else "Не удалось принять вопрос. Попробуй написать текстом."
            )
