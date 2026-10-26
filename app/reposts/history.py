"""Import a specific Telegram HTML chat export, never its scripts or media files."""

from __future__ import annotations

import argparse
import json
import re
from contextlib import suppress
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

from app.media.downloads import media_link


class ExportParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.messages = []
        self.stack = []
        self.current = None
        self.previous_sender = ""

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = set(attrs.get("class", "").split())
        if tag == "div" and {"message", "default"} <= classes:
            self.current = {
                "id": attrs.get("id", ""),
                "sender": "",
                "date": "",
                "text": "",
                "urls": [],
                "media": [],
            }
        field = None
        if self.current is not None:
            if "from_name" in classes:
                field = "sender"
            elif tag == "div" and classes == {"text"}:
                field = "text"
            if {"date", "details"} <= classes:
                self.current["date"] = attrs.get("title", "")
            if tag == "a" and attrs.get("href"):
                url = attrs["href"]
                if url.startswith(("https://x.com/", "https://twitter.com/")):
                    self.current["urls"].append(url)
            if "media_video" in classes or "video_file" in classes:
                self.current["media"].append("video")
            if "media_photo" in classes or "photo" in classes:
                self.current["media"].append("photo")
            if tag == "br" and any(f == "text" for _, f, _ in self.stack):
                self.current["text"] += "\n"
        if tag not in {"br", "img", "meta", "link", "hr", "input"}:
            self.stack.append((tag, field, tag == "div" and {"message", "default"} <= classes))

    def handle_endtag(self, tag):
        if not self.stack or not any(t == tag for t, _, _ in self.stack):
            return
        while self.stack:
            old_tag, _, message_root = self.stack.pop()
            if message_root and self.current is not None:
                row = self.current
                row["sender"] = row["sender"].strip() or self.previous_sender
                self.previous_sender = row["sender"]
                row["text"] = row["text"].strip()
                self.messages.append(row)
                self.current = None
            if old_tag == tag:
                break

    def handle_data(self, data):
        if self.current is None:
            return
        for _, field, _ in reversed(self.stack):
            if field:
                self.current[field] += data
                break


def read_export(path):
    path = Path(path)
    files = sorted(path.parent.glob("messages*.html"), key=lambda p: (len(p.name), p.name))
    parser = ExportParser()
    for file in files:
        parser.feed(file.read_text(encoding="utf-8"))
    return list({m["id"]: m for m in parser.messages}.values())


def export_samples(messages, *, owner_name, bot_name="X Repost"):
    """Only owner links are selections. Adjacent bot text is contextual, not a like."""
    result = {}
    pending = None
    errors = 0
    for message in messages:
        if message["sender"] == owner_name:
            pending = None
            links = []
            for url in message["urls"]:
                try:
                    link = media_link(url)
                except ValueError:
                    continue
                if link[0] == "x" and link[1] not in [v[1] for v in links]:
                    links.append(link)
            if len(links) != 1:
                continue  # Never assign a translation to an ambiguous multi-link input.
            _, ident, url = links[0]
            try:
                date = datetime.strptime(message["date"], "%d.%m.%Y %H:%M:%S UTC%z")
            except ValueError:
                continue
            pending = result.setdefault(
                ident,
                {
                    "tweet_id": ident,
                    "source_url": url,
                    "author": "",
                    "text": "",
                    "selected_at": date.astimezone(UTC).isoformat(),
                    "provenance": "telegram_export:owner_link:adjacent_bot_text",
                    "message_ids": [],
                },
            )
            pending["message_ids"].append(message["id"])
            for original in message["urls"]:
                match = re.search(r"(?:x|twitter)\.com/([\w]+)/status/", original)
                if match and match[1] != "i":
                    pending["author"] = match[1].lower()
        elif message["sender"] == bot_name and pending is not None:
            body = message["text"].strip()
            # A failed translation can still contain the useful original post.
            if body.startswith("⚠️ Не удалось перевести, вот оригинал:"):
                errors += 1
                body = body.partition("\n\n")[2].strip()
            elif re.match(
                r"^(?:❌|Ошибка\s*:|Error\s*:|Не удалось\s|Не могу\s|👋|Команды:)", body, re.I
            ):
                errors += 1
                continue
            # A response with an explicit different post link cannot annotate this input.
            ids = set()
            for url in message["urls"]:
                with suppress(ValueError):
                    ids.add(media_link(url)[1])
            if ids and pending["tweet_id"] not in ids:
                continue
            body = re.sub(r"https?://\S+", "", body).strip()
            body = re.sub(r"(?:Источник|Оригинал)\s*:?\s*$", "", body).strip()
            if len(body) > 25 and body not in pending["text"]:
                pending["text"] = (pending["text"] + "\n\n" + body).strip()[:20000]
    return list(result.values()), errors


def import_samples(repository, owner, samples):
    now = datetime.now(UTC).isoformat()
    with repository.connection() as db:
        before = db.total_changes
        for sample in samples:
            db.execute(
                "INSERT OR IGNORE INTO x_curation_samples(chat_id,tweet_id,source_url,author,"
                "text,provenance,selected_at,imported_at) VALUES(?,?,?,?,?,?,?,?)",
                (
                    owner,
                    sample["tweet_id"],
                    sample["source_url"],
                    sample["author"],
                    sample["text"],
                    sample["provenance"],
                    sample["selected_at"],
                    now,
                ),
            )
        return db.total_changes - before


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path)
    p.add_argument("--owner-name", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    messages = read_export(args.path)
    samples, errors = export_samples(messages, owner_name=args.owner_name)
    payload = {"message_count": len(messages), "error_messages": errors, "samples": samples}
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    args.output.chmod(0o600)
    print(
        json.dumps(
            {
                "messages": len(messages),
                "unique_links": len(samples),
                "with_text": sum(bool(s["text"]) for s in samples),
                "errors": errors,
            }
        )
    )


if __name__ == "__main__":
    main()
