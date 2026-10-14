CREATE TABLE video_relevance_assessments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_item_id INTEGER NOT NULL REFERENCES source_items(id),
    chat_id INTEGER NOT NULL,
    profile_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    assessment_json TEXT,
    response_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(source_item_id,chat_id,profile_hash)
);
