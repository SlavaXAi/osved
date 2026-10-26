"""Explainable local similarity learning. No model fine-tuning or provider calls."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from datetime import UTC, datetime
from functools import cached_property

STOP = set(
    [
        "этот",
        "этого",
        "этой",
        "этом",
        "которые",
        "который",
        "которая",
        "новых",
        "новый",
        "новая",
        "новые",
        "сейчас",
        "теперь",
        "будет",
        "будут",
        "может",
        "могут",
        "очень",
        "чтобы",
        "когда",
        "только",
        "также",
        "более",
        "после",
        "через",
        "между",
        "потому",
        "всего",
        "первый",
        "можно",
        "своих",
        "своей",
        "себя",
        "если",
        "даже",
        "есть",
        "было",
        "были",
        "менее",
        "уже",
        "они",
        "она",
        "оно",
        "это",
        "как",
        "что",
        "для",
        "про",
        "при",
        "под",
        "над",
        "без",
        "или",
        "его",
        "нам",
        "вам",
        "нас",
        "вас",
        "the",
        "this",
        "that",
        "with",
        "from",
        "into",
        "about",
        "have",
        "has",
        "been",
        "will",
        "would",
        "could",
        "their",
        "your",
        "they",
        "them",
        "these",
        "those",
        "more",
        "than",
        "just",
        "some",
        "what",
        "when",
        "where",
        "which",
        "how",
        "and",
        "are",
        "but",
        "not",
        "all",
        "you",
        "now",
        "use",
        "new",
        "one",
        "our",
        "out",
        "its",
        "can",
        "first",
        "people",
        "world",
        "today",
        "сегодня",
        "люди",
        "человек",
        "весь",
        "всем",
        "мира",
        "мир",
        "оригинал",
        "источник",
        "автор",
        "пост",
        "поста",
        "twitter",
        "video",
        "https",
        "youtube",
        "tiktok",
        "com",
        "www",
        "x",
        "formerly",
        "telegram",
    ]
)


def tokens(text):
    text = re.sub(r"https?://\S+|@[A-Za-z0-9_]+|<[^>]+>", " ", text.lower())
    words = re.findall(r"[a-zа-яё][a-zа-яё0-9_+-]{2,}", text)
    return set(
        w[:7] if re.match(r"[а-яё]", w) and len(w) > 7 else w for w in words if w not in STOP
    )


class CurationIndex:
    def __init__(self, samples, now):
        self.now = now
        self.samples = [r for r in samples if len(tokens(r["text"])) >= 3]
        self.terms = [tokens(r["text"]) for r in self.samples]
        counts = Counter(t for terms in self.terms for t in terms)
        self.idf = {t: 1 + math.log((1 + len(self.samples)) / (1 + n)) for t, n in counts.items()}
        self.norms = [math.sqrt(sum(self.idf[t] ** 2 for t in terms)) for terms in self.terms]

    @cached_property
    def topic_support(self):
        from app.personalization.daily import GENRES, traits

        support = {}
        for row in self.samples:
            if row.get("vote") == -1:
                continue
            selected = datetime.fromisoformat(row["selected_at"])
            selected = selected.replace(tzinfo=UTC) if selected.tzinfo is None else selected
            age = max(0, (self.now - selected).total_seconds() / 86400)
            # Topic affinity uses selections only. Result ratings stay local to similar posts.
            weight = 0.2 * 2 ** (-age / 180)
            for topic in traits({"headline": row["text"]}) - GENRES:
                support.setdefault(topic, []).append((row, weight))
        return support

    def topic_affinity(self, text):
        from app.personalization.daily import GENRES, traits

        evidence = []
        for topic in traits({"headline": text}) - GENRES:
            rows = self.topic_support.get(topic, [])
            authors = {r["author"] for r, _ in rows if r["author"]}
            if len(rows) < 3 or len(authors) < 2:
                continue
            by_author = {}
            for row, weight in rows:
                key = row["author"] or row["tweet_id"]
                by_author[key] = min(0.4, by_author.get(key, 0) + weight)
            mass = sum(by_author.values())
            evidence.append(
                {
                    "topic": topic,
                    "selections": len(rows),
                    "authors": len(authors),
                    "adjustment": round(0.12 * mass / (mass + 2), 4),
                }
            )
        # Correlated labels cannot multiply one preference into a dominant boost.
        return max((e["adjustment"] for e in evidence), default=0), evidence

    @cached_property
    def digest(self):
        return hashlib.sha256(json.dumps(self.samples, sort_keys=True).encode()).hexdigest()[:20]

    def adjustment(self, text):
        topic_score, topics = self.topic_affinity(text)
        terms = tokens(text)
        norm = math.sqrt(
            sum(self.idf.get(t, 1 + math.log(1 + len(self.samples))) ** 2 for t in terms)
        )
        matches = []
        if not norm:
            return {"adjustment": topic_score, "examples": [], "topics": topics}
        for row, known, known_norm in zip(self.samples, self.terms, self.norms, strict=True):
            overlap = terms & known
            if len(overlap) < 3:
                continue
            similarity = sum(self.idf[t] ** 2 for t in overlap) / (norm * known_norm)
            if similarity < 0.12:
                continue
            date = datetime.fromisoformat(row.get("voted_at") or row["selected_at"])
            date = date.replace(tzinfo=UTC) if date.tzinfo is None else date
            age = max(0, (self.now - date).total_seconds() / 86400)
            vote = row.get("vote")
            weight = float(vote) if vote else 0.2
            decay = 2 ** (-age / (90 if vote else 180))
            matches.append(
                {
                    "tweet_id": row["tweet_id"],
                    "author": row["author"],
                    "similarity": round(similarity, 4),
                    "signal": weight,
                    "weighted": weight * decay * similarity,
                    "provenance": row["provenance"],
                }
            )
        matches.sort(key=lambda r: abs(r["weighted"]), reverse=True)
        chosen, authors = [], Counter()
        for row in matches:
            # One prolific author must not dominate the learned neighborhood.
            key = row["author"] or row["tweet_id"]
            if authors[key] >= 2:
                continue
            authors[key] += 1
            chosen.append(row)
            if len(chosen) == 8:
                break
        weighted = sum(r["weighted"] for r in chosen)
        mass = sum(abs(r["weighted"]) for r in chosen)
        score = 0.6 * weighted / (mass + 0.8)
        # A dislike of a similar result must not be cancelled by a broad topic affinity.
        total = score + topic_score if score >= 0 else score
        return {
            "adjustment": round(max(-0.45, min(0.45, total)), 4),
            "examples": chosen,
            "topics": topics,
        }


def load_index(repository, owner, now):
    with repository.connection() as db:
        exists = db.execute(
            "SELECT 1 FROM sqlite_master WHERE name='x_curation_samples'"
        ).fetchone()
        if not exists:
            return CurationIndex([], now)
        rows = db.execute(
            "SELECT * FROM x_curation_samples WHERE chat_id=? "
            "AND julianday(selected_at)<=julianday(?) ORDER BY selected_at DESC LIMIT 1500",
            (owner, now.isoformat()),
        ).fetchall()
    samples = []
    for row in rows:
        sample = dict(row)
        if sample["voted_at"] and datetime.fromisoformat(sample["voted_at"]) > now:
            sample["vote"] = None
            sample["voted_at"] = None
        samples.append(sample)
    return CurationIndex(samples, now)
