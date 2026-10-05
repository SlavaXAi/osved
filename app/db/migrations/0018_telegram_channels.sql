CREATE TABLE telegram_channels (
    source_id TEXT PRIMARY KEY,
    chat_id INTEGER UNIQUE,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    title TEXT NOT NULL,
    added_by INTEGER NOT NULL,
    added_at TEXT NOT NULL,
    not_before TEXT NOT NULL,
    removed_at TEXT
);
