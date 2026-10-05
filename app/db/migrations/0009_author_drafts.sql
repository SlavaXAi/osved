CREATE TABLE IF NOT EXISTS author_drafts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    card_id INTEGER NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
    draft_version INTEGER NOT NULL,
    style_profile_version TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    model TEXT NOT NULL,
    generation_cost_usd REAL NOT NULL DEFAULT 0 CHECK (generation_cost_usd >= 0),
    structured_output_json TEXT NOT NULL,
    rendered_html TEXT NOT NULL,
    validation_status TEXT NOT NULL,
    validation_errors_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(card_id, draft_version)
);

CREATE INDEX IF NOT EXISTS author_drafts_card_idx
ON author_drafts(card_id, draft_version DESC);
