CREATE TABLE IF NOT EXISTS media_requests (
    id INTEGER PRIMARY KEY,
    chat_id INTEGER NOT NULL,
    request_message_id INTEGER NOT NULL,
    platform TEXT NOT NULL,
    external_id TEXT NOT NULL,
    source_url TEXT NOT NULL,
    panel_message_id INTEGER,
    video_panel_id INTEGER,
    status TEXT NOT NULL DEFAULT 'new',
    manifest_json TEXT,
    error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(chat_id, request_message_id)
);
CREATE TABLE IF NOT EXISTS media_download_jobs (
    id INTEGER PRIMARY KEY,
    request_id INTEGER NOT NULL REFERENCES media_requests(id),
    item_index INTEGER NOT NULL,
    choice TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    error TEXT,
    result_json TEXT,
    sent_message_id INTEGER,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS media_active_job ON media_download_jobs
    (request_id,item_index,choice) WHERE status IN ('queued','running','sending');
CREATE TABLE IF NOT EXISTS media_file_cache (
    cache_key TEXT NOT NULL,
    bot_id INTEGER NOT NULL,
    file_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY(cache_key,bot_id)
);
