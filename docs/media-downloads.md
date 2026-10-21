# Download media to Telegram

Send `/download https://…` with one public YouTube or X/Twitter URL. The bot shows
available formats and sends the selected video or MP3 as a Telegram file.
Ordinary X links show this menu too. Manually submitted YouTube summary cards also
offer **Скачать видео / MP3**. No scheduled feed downloads media automatically.

The owner can forward the resulting message through Telegram. The caption keeps
the original URL. A direct Shorts URL is supported; this does not enable Shorts
in recommendations. Posts with multiple videos expose each item and an option
to request all of them.

The implementation uses yt-dlp and FFmpeg, without an LLM or a paid scraping API.
The tested YouTube extraction client is `android_vr`. Source access can still fail
for private, removed, region-restricted, or rate-limited videos. Do not upload
browser cookies to the server as an implicit fallback.

## Configuration

| Variable | Default | Purpose |
|---|---:|---|
| `MEDIA_DOWNLOADS_ENABLED` | true | Enable manual media downloads |
| `MEDIA_MAX_FILE_BYTES` | 1900000000 | Additional per-file ceiling |
| `MEDIA_MAX_STORAGE_BYTES` | 4500000000 | Temporary media storage ceiling |
| `MEDIA_MIN_FREE_BYTES` | 1000000000 | Keep free disk headroom |
| `MEDIA_TIMEOUT_SECONDS` | 1800 | Download/conversion timeout |
| `TELEGRAM_API_BASE_URL` | empty | Optional private local Bot API endpoint |

With the normal Telegram Bot API, the actual file ceiling is 49 MB. A local
Bot API server permits larger uploads; the bot caps them at 1.9 GB. Use the
official server in `--local` mode, provide your own `TELEGRAM_API_ID` and
`TELEGRAM_API_HASH`, and keep its endpoint private. Mount its file directory at
the same absolute path in the bot container, because `getFile` can return local
paths. Complete Telegram's documented `logOut` migration before switching a
running bot. Test messages, downloads, existing summaries, and PDFs afterwards.

Do not silently downscale a chosen video to fit the upload limit. If the file
is too large, select a smaller available quality.

## State and recovery

Migration `0027_media_downloads` stores manual requests, jobs, and Telegram
`file_id` values. The media queue has one worker and at most five active queued
downloads. Duplicate clicks reuse the existing job. Repeated successful requests
reuse the file in Telegram for the same bot. Cancellation stops the child process.

Completed files are removed locally after delivery. A crash during transfer to
Telegram becomes `uncertain`; it is not silently sent again. A crash while
preparing a file resumes from the durable job and can reuse a completed local
file. Keep the database and media directory outside disposable containers.

The current release deliberately does not depend on Monid. The API catalog is
not proof that returned YouTube URLs can be downloaded from the server's IP.

Official references: [Telegram files](https://core.telegram.org/bots/api#sending-files),
[local Bot API](https://github.com/tdlib/telegram-bot-api),
[yt-dlp](https://github.com/yt-dlp/yt-dlp).
