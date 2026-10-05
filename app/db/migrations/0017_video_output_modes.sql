-- Retain existing jobs/IDs and permit both outputs from the same recommendation.
ALTER TABLE video_pdf_jobs RENAME TO video_pdf_jobs_legacy;
CREATE TABLE video_pdf_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL,
    user_id INTEGER NOT NULL,
    chat_id INTEGER NOT NULL,
    request_message_id INTEGER NOT NULL,
    output_kind TEXT NOT NULL DEFAULT 'pdf' CHECK(output_kind IN ('pdf','tldr')),
    status TEXT NOT NULL DEFAULT 'queued',
    error TEXT,
    status_message_id INTEGER,
    document_message_id INTEGER,
    text_message_id INTEGER,
    available_at REAL NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(chat_id, request_message_id, output_kind)
);
INSERT INTO video_pdf_jobs(id,video_id,user_id,chat_id,request_message_id,status,error,
    status_message_id,document_message_id,available_at,created_at,updated_at)
SELECT id,video_id,user_id,chat_id,request_message_id,status,error,
    status_message_id,document_message_id,available_at,created_at,updated_at
FROM video_pdf_jobs_legacy;
DROP TABLE video_pdf_jobs_legacy;
CREATE INDEX video_pdf_jobs_pending ON video_pdf_jobs(status, available_at);
