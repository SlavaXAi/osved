CREATE TABLE IF NOT EXISTS feedback_sessions (
    delivery_id INTEGER PRIMARY KEY REFERENCES deliveries(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL,
    reaction TEXT NOT NULL,
    state TEXT NOT NULL,
    prompt_message_id INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS feedback_sessions_user_idx
ON feedback_sessions(user_id, updated_at DESC);
