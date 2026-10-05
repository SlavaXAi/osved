CREATE TABLE video_recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_item_id INTEGER NOT NULL REFERENCES source_items(id),
    chat_id INTEGER NOT NULL,
    telegram_message_id INTEGER,
    description_html TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    UNIQUE(source_item_id,chat_id)
);
CREATE TABLE video_feedback_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recommendation_id INTEGER NOT NULL REFERENCES video_recommendations(id),
    action TEXT NOT NULL,
    created_at TEXT NOT NULL
);
