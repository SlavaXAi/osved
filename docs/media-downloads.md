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

### Prepared local Bot API deployment

`compose.telegram-api.yaml` is an opt-in overlay for `compose.server.yaml`.
It is **not enabled by the normal deployment**. It pins the tested linux/amd64
Bot API 10.3 image by digest, publishes no host port, and puts the API and bot
on a separate Docker bridge. The API only receives its two application
credentials. Both processes use uid/gid 10001 so the bot can read local
`getFile` paths through the read-only mount at the same absolute path.
Confirm that uid/gid against the actual runtime before enabling it.

To prepare storage on the server (no transport change):

```sh
install -d -m 0700 -o 10001 -g 10001 data/telegram-bot-api data/telegram-bot-api/tmp
```

Activation requires a controlled migration, not just adding an environment
variable:

1. Obtain your own API ID/hash at my.telegram.org and put them in the private
   `.env`. Do not use sample or another application's published credentials.
   Validate the merged Compose configuration with `config --quiet`.
2. Back up the Compose file and SQLite database. Wait for active downloads,
   reports and deliveries to finish. Record the bot update offset and current
   image; stop only `newsfeed`.
3. Start only the `telegram-bot-api` service with both Compose files. Check its
   local listener and shared-directory permissions. Do not authorize the live
   bot there yet. A healthy listener alone does not verify Telegram access.
4. Call `logOut` on the cloud Bot API once and verify success. Set the private
   local endpoint using the overlay and start `newsfeed` with both Compose files.
   Future deployments must keep using both files while local transport is active.
5. Verify polling, existing card edits, voice/file downloads, PDFs, cached
   `file_id` reuse, and a real video larger than 50 MB. Check disk headroom and
   cleanup after the upload. Only then consider large-file delivery released.

Rollback also needs a transport migration: stop `newsfeed`, log out on the
local server, and allow for Telegram's cloud-login delay (at least ten minutes
after the successful `logOut`). Preserve the live database and update offset, then
restore cloud transport. Do not run polling on both endpoints concurrently.

As of 2026-10-21 the image/version is verified and storage prepared on the
existing server, but activation is blocked by app-registration throttling at
my.telegram.org. The production upload limit remains 49 MB. Large files must
remain one file: automatic splitting and compression are not enabled.

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
