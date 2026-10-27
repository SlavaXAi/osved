import asyncio
import re
import shutil
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.types import FSInputFile
from test_x_reposts import env as env  # noqa: F401
from test_x_reposts import ready_repost

from app.details.paging import paginate_native_html, telegram_html_visible_text, utf16_units
from app.reposts import media, native, store
from app.telegram_bot.news_panel import get_news_panel, show_base


def video(tmp_path):
    path = tmp_path / "source.mp4"
    path.write_bytes(b"original mp4 placeholder")
    thumbnail = tmp_path / "source.thumb.jpg"
    thumbnail.write_bytes(b"jpeg frame placeholder")
    return {
        "kind": "video",
        "path": str(path),
        "file_id": "legacy-square",
        "index": 0,
        "width": 1920,
        "height": 1080,
        "duration": 89,
        "size": path.stat().st_size,
        "thumbnail_path": str(thumbnail),
    }


def bot():
    return SimpleNamespace(
        edit_message_media=AsyncMock(
            return_value=SimpleNamespace(
                video=SimpleNamespace(file_id="native-wide", width=1920, height=1080),
                rich_message=None,
            )
        ),
        edit_message_text=AsyncMock(),
        edit_message_caption=AsyncMock(),
        send_video=AsyncMock(
            return_value=SimpleNamespace(
                message_id=91, video=SimpleNamespace(file_id="native-wide", width=1920, height=1080)
            )
        ),
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=92)),
    )


def test_short_post_is_native_video_and_reuploads_legacy_square_id(env, tmp_path):
    row, panel = ready_repost(env)
    api = bot()
    files = [video(tmp_path)]
    saved = asyncio.run(native.publish(api, env[1], row, panel, files))
    sent = api.edit_message_media.call_args.kwargs
    assert sent["message_id"] == row["panel_message_id"]
    assert isinstance(sent["media"].media, FSInputFile)  # The corrupt legacy ID cannot be reused.
    assert (sent["media"].width, sent["media"].height) == (1920, 1080)
    assert isinstance(sent["media"].thumbnail, FSInputFile)
    assert sent["media"].caption == panel["baseline_html"]
    assert saved[0]["file_id"] == "native-wide" and saved[0]["native_file_id"]
    assert not api.edit_message_text.called and not api.send_video.called
    assert store.get(env[1], row["id"])["native_kind"] == "video"
    assert asyncio.run(show_base(api, env[1], panel))
    back = api.edit_message_caption.call_args.kwargs
    assert back["caption"] == panel["baseline_html"] and "rich_message" not in back


def test_long_post_keeps_full_translation_separate_and_does_not_repeat_media(env, tmp_path):
    row, panel = ready_repost(env)
    api = bot()
    files = [video(tmp_path)]
    body = "«Полный перевод с сохранением цитаты». " * 50
    panel = native.refresh_baseline(env[1], panel, body)
    asyncio.run(native.publish(api, env[1], row, panel, files))
    assert api.edit_message_text.call_args.kwargs["text"] == body.rstrip()
    assert "caption" not in api.send_video.call_args.kwargs
    assert isinstance(api.send_video.call_args.kwargs["thumbnail"], FSInputFile)
    assert not api.edit_message_media.called
    asyncio.run(native.publish(api, env[1], row, panel, files))
    assert api.send_video.call_count == 1
    assert store.get(env[1], row["id"])["native_kind"] == "text"
    assert env[1].current_month_cost() == 0


def test_unknown_companion_send_is_not_replayed_and_retry_is_blocked(env, tmp_path):
    row, panel = ready_repost(env)
    api = bot()
    panel = native.refresh_baseline(env[1], panel, "Длинный полный перевод. " * 100)
    api.send_video.side_effect = TimeoutError()
    for _ in range(2):
        with pytest.raises(native.UnconfirmedPublication):
            asyncio.run(native.publish(api, env[1], row, panel, [video(tmp_path)]))
    assert api.send_video.call_count == 1
    with pytest.raises(ValueError, match="не подтверждена"):
        store.retry(env[1], row)


def test_long_unicode_paragraph_is_delivered_without_dropped_words():
    original = "Смысл & факты <без сокращений> 👩‍💻. " * 500
    import html

    pages = paginate_native_html(html.escape(original), page_limit=4096)
    assert len(pages) > 1 and all(utf16_units(telegram_html_visible_text(p)) <= 4096 for p in pages)

    def normalize(s):
        return re.sub(r"\s+", "", s)

    assert normalize("".join(telegram_html_visible_text(p) for p in pages)) == normalize(original)


def test_native_video_details_use_caption_pages_then_restore_original(env):
    from test_news_reader_flow import article_result

    from app.db.news_questions import claim_input, open_session, queue_question
    from app.details.worker import DetailWorker
    from app.telegram_bot.news_panel import reader_pages, show_result

    row, panel = ready_repost(env)
    api = bot()
    repo = env[1]
    with repo.connection() as db:
        db.execute("UPDATE x_reposts SET native_kind='video' WHERE id=?", (row["id"],))
    open_session(repo, panel)
    assert claim_input(repo, panel, 42, "text", max_daily=10)
    req = queue_question(repo, panel, 42, "Объясни подробнее")

    def builder(*args, **kwargs):
        result = article_result(req)
        result.rendered_html = "<b>Пояснение</b>\n\n" + (
            "Подробный связный абзац с сохранением фактов. " * 40
        )
        return result

    assert DetailWorker(env[0], repo, builder=builder).run_once().status == "ready"
    req = repo.get_detail_request(req["request_id"])
    article = repo.get_detailed_article(req["article_id"])
    pages = reader_pages(repo, panel, article)
    assert len(pages) > 1 and all(utf16_units(telegram_html_visible_text(p)) <= 1024 for p in pages)
    assert asyncio.run(show_result(api, repo, panel, req, article))
    assert api.edit_message_caption.called and not api.edit_message_text.called
    assert asyncio.run(show_base(api, repo, get_news_panel(repo, panel["delivery_id"])))
    assert api.edit_message_caption.call_args.kwargs["caption"] == panel["baseline_html"]
    corrected = native.refresh_baseline(
        repo, panel, "Исправленный близкий перевод исходного поста."
    )
    assert (
        corrected["reader_context"]["context_fingerprint"]
        == req["reader_context"]["context_fingerprint"]
    )
    assert asyncio.run(show_result(api, repo, corrected, req, article))
    assert asyncio.run(show_base(api, repo, corrected))
    assert api.edit_message_caption.call_args.kwargs["caption"] == corrected["baseline_html"]


@pytest.mark.skipif(
    not shutil.which("ffmpeg"), reason="real ffmpeg is checked in the release container"
)
@pytest.mark.parametrize(
    "size,sar,expected",
    [("320x180", "1", (320, 180)), ("180x320", "1", (180, 320)), ("320x180", "2", (640, 180))],
)
def test_real_video_geometry_preserves_display_ratio(tmp_path, size, sar, expected):
    import subprocess

    path = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=blue:s={size}:d=1",
            "-vf",
            "setsar=" + sar,
            "-c:v",
            "libx264",
            "-threads",
            "1",
            str(path),
        ],
        check=True,
    )
    normalized, geometry = media.prepare_video(path, cap=49_000_000)
    assert (geometry["width"], geometry["height"]) == expected
    assert media.video_geometry(normalized)["sar"] == "1:1"
    before = normalized.read_bytes()
    thumbnail = media.video_thumbnail(normalized)
    assert thumbnail and 0 < thumbnail.stat().st_size < 200_000
    preview = media.video_geometry(thumbnail)
    assert max(preview["width"], preview["height"]) <= 320
    assert abs(preview["width"] / preview["height"] - expected[0] / expected[1]) < 0.03
    assert normalized.read_bytes() == before
    if sar == "1":
        assert normalized == path
