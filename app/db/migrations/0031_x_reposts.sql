CREATE TABLE x_reposts (
    id INTEGER PRIMARY KEY,
    chat_id INTEGER NOT NULL,
    tweet_id TEXT NOT NULL,
    source_url TEXT NOT NULL,
    request_message_id INTEGER NOT NULL,
    panel_message_id INTEGER,
    event_id INTEGER NOT NULL REFERENCES events(id),
    delivery_id INTEGER REFERENCES deliveries(id),
    status TEXT NOT NULL DEFAULT 'queued',
    post_json TEXT,
    translation_json TEXT,
    media_json TEXT NOT NULL DEFAULT '[]',
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(chat_id,tweet_id)
);
CREATE TABLE x_curation_samples (
    chat_id INTEGER NOT NULL,
    tweet_id TEXT NOT NULL,
    source_url TEXT NOT NULL,
    author TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL DEFAULT '',
    provenance TEXT NOT NULL,
    selected_at TEXT NOT NULL,
    imported_at TEXT NOT NULL,
    vote INTEGER CHECK(vote IN (-1,1)),
    voted_at TEXT,
    PRIMARY KEY(chat_id,tweet_id)
);
CREATE INDEX x_reposts_status ON x_reposts(status,id);
