CREATE TABLE IF NOT EXISTS card_media (
    card_id INTEGER PRIMARY KEY REFERENCES cards(id) ON DELETE CASCADE,
    policy_version TEXT NOT NULL,
    mode TEXT NOT NULL CHECK (mode IN ('telegram_preview', 'text_only')),
    preview_url TEXT,
    image_url TEXT,
    source_document_id INTEGER REFERENCES research_documents(id) ON DELETE SET NULL,
    checks_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS card_media_mode_idx
ON card_media(mode, created_at DESC);
