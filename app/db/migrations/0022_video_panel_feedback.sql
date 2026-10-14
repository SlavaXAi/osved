CREATE TABLE video_panels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    video_id TEXT NOT NULL,
    recommendation_id INTEGER REFERENCES video_recommendations(id),
    original_html TEXT NOT NULL,
    view TEXT NOT NULL DEFAULT 'card',
    summary_job_id INTEGER REFERENCES video_pdf_jobs(id),
    pdf_job_id INTEGER REFERENCES video_pdf_jobs(id),
    active_job_id INTEGER REFERENCES video_pdf_jobs(id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(chat_id,message_id)
);
ALTER TABLE video_pdf_jobs ADD COLUMN panel_id INTEGER REFERENCES video_panels(id);

CREATE TABLE video_result_feedback (
    job_id INTEGER NOT NULL REFERENCES video_pdf_jobs(id),
    chat_id INTEGER NOT NULL,
    reaction TEXT NOT NULL CHECK(reaction IN ('like','dislike')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(job_id,chat_id)
);
CREATE TABLE video_result_feedback_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL REFERENCES video_pdf_jobs(id),
    chat_id INTEGER NOT NULL,
    reaction TEXT NOT NULL CHECK(reaction IN ('like','dislike')),
    previous_reaction TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE video_result_messages (
    job_id INTEGER NOT NULL REFERENCES video_pdf_jobs(id),
    chat_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    output_kind TEXT NOT NULL CHECK(output_kind IN ('tldr','pdf')),
    panel_id INTEGER REFERENCES video_panels(id),
    created_at TEXT NOT NULL,
    PRIMARY KEY(job_id,chat_id,message_id)
);
