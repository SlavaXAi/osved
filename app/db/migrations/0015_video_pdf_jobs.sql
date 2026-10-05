CREATE TABLE IF NOT EXISTS video_pdf_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL,
    user_id INTEGER NOT NULL,
    chat_id INTEGER NOT NULL,
    request_message_id INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    error TEXT,
    status_message_id INTEGER,
    document_message_id INTEGER,
    available_at REAL NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(chat_id, request_message_id)
);
CREATE INDEX IF NOT EXISTS video_pdf_jobs_pending ON video_pdf_jobs(status, available_at);
