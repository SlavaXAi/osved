CREATE TABLE youtube_video_types (
    video_id TEXT PRIMARY KEY,
    is_short INTEGER CHECK (is_short IN (0,1)),
    basis TEXT NOT NULL,
    checked_at TEXT NOT NULL,
    retry_after TEXT
);

CREATE TABLE youtube_channel_preferences (
    chat_id INTEGER NOT NULL,
    channel_id TEXT NOT NULL,
    preference INTEGER NOT NULL CHECK (preference IN (-1,0,1)),
    updated_at TEXT NOT NULL,
    PRIMARY KEY(chat_id,channel_id)
);

CREATE TABLE concept_runs (
    window_key TEXT PRIMARY KEY,
    corpus_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    usage_id INTEGER REFERENCES usage_ledger(id),
    created_at TEXT NOT NULL,
    finished_at TEXT,
    error TEXT,
    source_snapshot_json TEXT NOT NULL DEFAULT '[]',
    response_json TEXT
);

CREATE TABLE concept_briefs (
    concept_key TEXT PRIMARY KEY,
    term TEXT NOT NULL,
    card_id INTEGER NOT NULL UNIQUE REFERENCES cards(id),
    source_item_ids_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
