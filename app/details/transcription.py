"""One durable Russian voice transcription; audio is never placed in the call ledger."""

from __future__ import annotations

import base64
import hashlib
import json
from urllib.request import Request, urlopen

from app.llm.accounting import resolve_usage_cost

MODEL = "openai/whisper-large-v3"
MAX_AUDIO_BYTES = 8_000_000
MAX_VOICE_SECONDS = 180


def audio_transport(payload, key):
    request = Request(
        "https://openrouter.ai/api/v1/audio/transcriptions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
            "X-OpenRouter-Title": "OSVED news questions",
        },
        method="POST",
    )
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read(1_000_000))


def transcribe_question(settings, repository, guard, reader, *, transport=audio_transport):
    digest = reader["voice_input_sha256"]
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid saved audio identity")
    duration = reader["voice_duration_seconds"]
    if not isinstance(duration, (int, float)) or not 0 < duration <= MAX_VOICE_SECONDS:
        raise ValueError("Invalid voice duration")
    path = settings.data_dir / "news-question-audio" / (digest + ".ogg")
    if not path.is_file() or path.stat().st_size > MAX_AUDIO_BYTES:
        raise ValueError("Saved voice message is missing or too large")
    audio = path.read_bytes()
    if not audio or hashlib.sha256(audio).hexdigest() != digest:
        raise ValueError("Saved voice message does not match its identity")
    payload = {
        "model": MODEL,
        "input_audio": {"data": base64.b64encode(audio).decode("ascii"), "format": "ogg"},
        "language": "ru",
    }

    def cost(response, reservation):
        bill = resolve_usage_cost(response.get("usage"), model=MODEL, reservation_usd=reservation)
        return (reservation if bill.unavailable else bill.cost_usd), bill.metadata()

    response = guard.run(
        "openrouter",
        "transcription",
        {"model": MODEL, "sha256": digest, "language": "ru", "duration": duration},
        0.01,
        lambda: transport(payload, settings.openrouter_api_key.get_secret_value()),
        cost=cost,
    )
    text = str(response.get("text") or "").strip()
    if not 2 <= len(text) <= 6000:
        raise ValueError("Speech recognition returned no usable question")
    reader["user_question"] = text
    with repository.connection() as db:
        db.execute(
            "UPDATE news_question_inputs SET question=? WHERE delivery_id=? AND message_id=?",
            (text, reader.get("baseline_delivery_id", -1), reader.get("input_message_id", -1)),
        )
    return text
