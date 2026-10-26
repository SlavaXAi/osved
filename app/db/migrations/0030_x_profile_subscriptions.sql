CREATE TABLE x_profile_subscriptions (
    handle TEXT PRIMARY KEY COLLATE NOCASE,
    source_id TEXT NOT NULL UNIQUE,
    added_by INTEGER NOT NULL,
    added_at TEXT NOT NULL,
    not_before TEXT NOT NULL
);
