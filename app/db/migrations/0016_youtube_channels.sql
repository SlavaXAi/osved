CREATE TABLE youtube_channels (
    channel_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    added_by INTEGER NOT NULL,
    added_at TEXT NOT NULL,
    not_before TEXT NOT NULL,
    removed_at TEXT
);
CREATE TABLE youtube_channel_urls (
    url TEXT PRIMARY KEY,
    channel_id TEXT,
    status TEXT NOT NULL CHECK(status IN ('resolving','ready','failed')),
    attempted_at TEXT NOT NULL,
    usage_id INTEGER,
    run_id TEXT
);
